"""Scoring a program on a split, with the token accounting the chart runs on.

Three things here are load-bearing beyond "call the model and count the hits".

**Tokens are measured, not assumed.** Every eval reads the LM's own call history
and reports mean input/output tokens per call from the provider's own counters.
(Not `dspy.track_usage()` -- see the comment at `history_start` for why that one
silently reports zero here.)
The cost axis is then `measured tokens x published rate`, which is the only way
to make a reasoning model and a direct model comparable -- they differ by ~7x in
output tokens on identical work, and no headline $/1M rate can show that.

**A billed call is not a classification.** DSPy's `ChatAdapter` re-sends an
item through `JSONAdapter` when its own field format fails to parse, so one
classification can arrive as two billed calls. `cost_per_1k_calls` is what the
bill says per call, `cost_per_1k_items` is what one classification actually
costs, and `n_calls` is what makes the two reconcilable. A retry is invisible in
accuracy -- the metric scores whatever the retry returned -- and visible only in
the bill, which is how an optimizer can "improve" a prompt into costing double.

**Failures are counted, not silently zeroed.** A model that returns unparseable
output and a model that returns a confident wrong answer both score 0, but they
are different products with different fixes. `parse_failures` and `errors` keep
them apart, because "your cheap model is 8% less accurate" and "your cheap model
returns garbage 8% of the time" should not look identical on a chart.

Concurrency is 12 by default: measured on this machine, ollama throughput peaks
at 2.8 items/s around 12 threads and *degrades* past 16 as requests queue.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import dspy
import numpy as np

from .metric import score_prediction
from .models import HOSTED, LOCAL, ModelSpec
from .stats import Interval, accuracy_ci

DEFAULT_THREADS = 12

# Models that reject sampling parameters outright rather than ignoring them.
# Anthropic deprecated temperature/top_p/top_k on recent Claude models, and the
# OpenAI reasoning tiers accept only the default temperature. Sending one is a
# hard error, so these are matched by substring and skipped.
_NO_SAMPLING_PARAMS = ("claude-opus-4-", "claude-sonnet-4-6", "claude-opus-5",
                       "claude-sonnet-5", "claude-fable-5", "claude-haiku-4-5",
                       "gpt-5")

# OpenAI reasoning tiers, which DSPy additionally gates on a large max_tokens.
_REASONING_TIER = ("gpt-5",)


def history_cap() -> int:
    """The number of calls DSPy keeps in `lm.history` before evicting.

    Both the evaluator and the optimizer compute usage by slicing this list,
    which treats it as a complete ledger. It is not: `BaseLM.update_history`
    does `self.history.pop(0)` once the list exceeds the cap, so on a long
    enough run the earliest calls are gone and every token total, `n_calls`
    and therefore `cost_per_1k_items` comes out SILENTLY LOW.

    That is not a rounding error, it can invert the chart. A one-call model at
    $1.00/1k and a two-call model truly at $1.10/1k report $1.00 and $0.9167
    once eviction bites, so the more expensive model looks cheaper -- exactly
    the ranking this product sells.

    The lever is `dspy.settings.max_history_size` (default 10,000), NOT the
    module constant `dspy.clients.base_lm.MAX_HISTORY_SIZE`, which bounds only
    the separate GLOBAL_HISTORY list.
    """
    try:
        import dspy
        cap = getattr(dspy.settings, "max_history_size", None)
        if isinstance(cap, int) and cap > 0:
            return cap
    except Exception:
        pass
    return 10_000


def assert_ledger_intact(lm, start: int, what: str) -> None:
    """Refuse to report usage that eviction may have truncated.

    Raising beats warning here: a warning scrolls past and the wrong number
    still lands in results.json and on the chart. Raising costs the run and
    keeps the benchmark honest, and the fix is one setting.

    `start` is the offset the caller is about to slice from. It used to be
    accepted and ignored, which left the one failure it can prove unchecked:
    once eviction has popped from the front, a recorded offset points past the
    end and `lm.history[start:]` comes back EMPTY, i.e. zero tokens and zero
    calls rather than a loud error.
    """
    cap = history_cap()
    n = len(lm.history)
    if start > n:
        raise RuntimeError(
            f"{what}: lm.history has shrunk from {start:,} entries to {n:,}, so DSPy "
            f"evicted calls from the front and the slice this measurement reads "
            f"(lm.history[{start:,}:]) is empty -- it would report zero tokens and "
            f"zero calls instead of failing.")
    if n >= cap:
        raise RuntimeError(
            f"{what}: lm.history holds {n:,} entries at the {cap:,} cap, so DSPy has "
            f"evicted earlier calls and token totals would be silently LOW (this can "
            f"invert the cost ranking). Raise it before the run, e.g. "
            f"dspy.settings.configure(max_history_size={max(cap * 4, 100_000)}).")
        # note: no partial-credit path -- a truncated ledger is not a number.


def requested_max_tokens(spec: ModelSpec, max_tokens: int) -> int:
    """The output budget a call to `spec` will REALLY carry.

    DSPy refuses OpenAI reasoning models unless max_tokens is >= 16000 (or
    None) and temperature is 1.0 (or None), because a small cap can truncate
    the model mid-reasoning and return empty content. max_tokens is a ceiling,
    not a spend commitment -- the bill still follows the tokens actually
    emitted, which we measure.

    Factored out because the capability screen has to gate on this number, not
    on the configured one. The bump used to live inside `build_lm` *below* the
    `spec.fits()` call, so a gpt-5-family deployment whose `max_output` sat
    under 16,000 was screened at 1,024, passed, and then received a request it
    could not honour -- the exact "refuse before spending, not after
    truncating" failure the gate exists to prevent. `experiment._budget_for`
    calls this for the same reason.
    """
    if any(tag in spec.call_id for tag in _REASONING_TIER):
        return max(max_tokens, 16000)
    return max_tokens


def build_lm(spec: ModelSpec, max_tokens: int = 1024, cache: bool = False):
    """Instantiate the right LM for a spec.

    Local models go through OllamaLM for the `think` switch that litellm cannot
    reach; hosted models go through dspy.LM as usual.
    """
    if spec.runtime == LOCAL:
        from .ollama_lm import OllamaLM
        return OllamaLM(spec.call_id, think=spec.think, max_tokens=max_tokens, cache=cache)

    if spec.runtime != HOSTED:
        raise ValueError(f"{spec.key}: runtime {spec.runtime!r} has no LM")

    # Resolve the budget BEFORE the gate, so the gate screens the number the
    # request will actually carry.
    max_tokens = requested_max_tokens(spec, max_tokens)

    # Refuse before spending, not after truncating. A deployment capped below
    # the requested budget does not error -- it returns a short, truncated
    # answer that fails to parse, so the row reads as a model too weak to
    # follow a format rather than a budget the endpoint could never honour.
    # A limit of 0 means "not recorded" and cannot disqualify anything.
    fine, why = spec.fits(max_tokens)
    if not fine:
        raise ValueError(f"{spec.key}: {why}. Lower max_tokens or drop this model.")

    kwargs = dict(model=spec.call_id, max_tokens=max_tokens, cache=cache)
    if spec.api_base:
        kwargs["api_base"] = spec.api_base
    if spec.env_key:
        kwargs["api_key"] = os.environ.get(spec.env_key, "")
    if spec.extra_body:
        kwargs["extra_body"] = spec.extra_body

    # Anthropic prompt caching is OPT-IN: it only happens if the request carries
    # an explicit cache_control marker, whereas OpenAI caches a long shared
    # prefix automatically. Without this, a many-class benchmark silently
    # compares Claude's UNCACHED cost against everyone else's CACHED cost --
    # measured on banking77, Claude got 0% cached while the OpenAI models got
    # ~60%, inflating Opus from ~$2.70 to $18.62 per 1k. That is not a real
    # price difference, it is a missing flag, and charting it would be an
    # indefensible comparison.
    if spec.call_id.startswith("anthropic/"):
        kwargs["cache_control_injection_points"] = [{"location": "message", "role": "system"}]

    if not any(tag in spec.call_id for tag in _NO_SAMPLING_PARAMS):
        kwargs["temperature"] = 0.0

    # Some deployments accept a system message, return 200, and discard it.
    # ChatAdapter puts the instructions, the label enum and the format contract
    # there, so the model would see a bare sentence and the row would read as a
    # weak model rather than a broken transport.
    if spec.drops_system_role:
        from .system_role_lm import SystemRoleFoldingLM
        return SystemRoleFoldingLM(**kwargs)
    return dspy.LM(**kwargs)


@dataclass
class EvalResult:
    model_key: str
    label: str
    split: str
    n: int
    correct: np.ndarray = field(repr=False)
    predictions: list[str] = field(repr=False)
    gold: list[str] = field(repr=False)
    accuracy: Interval
    mean_in_tokens: float
    mean_out_tokens: float
    total_in_tokens: int
    total_out_tokens: int
    wall_seconds: float
    parse_failures: int
    errors: int
    cost_per_1k_calls: float
    instruction: str = ""
    # Cached input tokens, reported by the provider. Defaulted so older result
    # files (written before caching was tracked) still deserialise.
    mean_cached_tokens: float = 0.0
    total_cached_tokens: int = 0
    mean_cache_write_tokens: float = 0.0
    total_cache_write_tokens: int = 0
    # What wall_seconds MEANS for this row. "batch" = duration of the whole
    # concurrent evaluation (so ~num_threads smaller than per-request latency);
    # "fit" = model training time, which is not a latency at all. Mixing the
    # two under one label is a comparison no reader can make correctly.
    timing_basis: str = "batch"
    # Billed LM calls behind this eval -- NOT the number of classifications.
    # 0 for the non-LM rows in encoders.py, which make no call at all.
    n_calls: int = 0
    # Cost per 1,000 *classifications*, which is what the chart axis promises.
    # None means "derive it from cost_per_1k_calls", so the conversion lives in
    # exactly one place and encoders.py needs no per-item arithmetic of its own.
    cost_per_1k_items: float | None = None

    # Per-item confidence in the prediction, 0-1, or None when the row cannot
    # produce one. Added because every downstream routing question needs it and
    # nothing could ask: the free rows discarded their probability distributions,
    # so the cascade pages had to be built on the TF-IDF row even though the
    # encoder beats it on 6 of 7 datasets, and two independent analyses had to
    # monkey-patch sklearn and torch to recover what was already computed.
    confidence: list[float] | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.cost_per_1k_items is None:
            # n_calls == 0 means nothing was billed per call, so there is no
            # conversion to do and the per-call figure IS the per-item figure.
            # NaN (unverified rate card) has to survive either branch rather
            # than collapsing to a plottable zero.
            self.cost_per_1k_items = (self.cost_per_1k_calls * self.n_calls / self.n
                                      if self.n_calls and self.n else self.cost_per_1k_calls)

    @property
    def retry_rate(self) -> float:
        """Extra billed calls per classification; 0.0 when every item took one.

        This is the only place a retry shows up. `ChatAdapter` scores the
        retry's answer, so accuracy looks normal, and `cost_per_1k_calls` is
        per call, so the price looks normal too -- the doubled spend exists
        only in the product of the two.
        """
        if not self.n or not self.n_calls:
            return 0.0
        return max(0.0, (self.n_calls - self.n) / self.n)

    @property
    def per_class_accuracy(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for lab in sorted(set(self.gold)):
            idx = [i for i, g in enumerate(self.gold) if g == lab]
            out[lab] = float(self.correct[idx].mean()) if idx else float("nan")
        return out

    def summary(self) -> str:
        cache = f"({self.mean_cached_tokens:.0f} cached)" if self.mean_cached_tokens else ""
        # The two costs are the same number unless a retry pulled them apart,
        # so the per-item figure appears only when it disagrees -- and then it
        # is unmissable, which is the whole reason n_calls is tracked. retries=
        # sits with unparseable=/errors= because it is the same kind of fact: a
        # thing that went wrong without changing the score.
        cost = f"${self.cost_per_1k_calls:.4f}/1k"
        retries = ""
        if self.retry_rate:
            cost = (f"${self.cost_per_1k_calls:.4f}/1k-call "
                    f"${self.cost_per_1k_items:.4f}/1k-item")
            retries = f"  retries={self.retry_rate:.1%}"
        return (f"{self.label:<32} acc={self.accuracy}  "
                f"tok={self.mean_in_tokens:.0f}in{cache}/{self.mean_out_tokens:.0f}out  "
                f"{cost}  "
                f"{self.wall_seconds:.0f}s  unparseable={self.parse_failures}  "
                f"errors={self.errors}{retries}")

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key, "label": self.label, "split": self.split,
            "n": self.n, "accuracy": self.accuracy.to_dict(),
            "per_class_accuracy": self.per_class_accuracy,
            "mean_in_tokens": self.mean_in_tokens, "mean_out_tokens": self.mean_out_tokens,
            "total_in_tokens": self.total_in_tokens, "total_out_tokens": self.total_out_tokens,
            "mean_cached_tokens": self.mean_cached_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "mean_cache_write_tokens": self.mean_cache_write_tokens,
            "total_cache_write_tokens": self.total_cache_write_tokens,
            "timing_basis": self.timing_basis,
            "wall_seconds": self.wall_seconds, "parse_failures": self.parse_failures,
            "errors": self.errors, "cost_per_1k_calls": self.cost_per_1k_calls,
            "n_calls": self.n_calls, "cost_per_1k_items": self.cost_per_1k_items,
            "predictions": self.predictions, "gold": self.gold,
            "correct": self.correct.tolist(), "instruction": self.instruction,
        }


# ── reading rows back off disk ───────────────────────────────────────────────
# Six scripts re-chart or re-rank a finished run from its results.json without
# re-running the models, and each needs the per-classification cost. They must
# all agree: a copy of this arithmetic that drifts puts a per-call number back
# under a per-classification axis, which is the bug this whole pair of fields
# exists to prevent. So it lives here, next to the fields it mirrors.

def row_n_calls(row: dict) -> int:
    """Billed LM calls behind a stored result row.

    Rows written before `n_calls` existed reconstruct it exactly rather than
    approximately: `mean_in_tokens` IS `total_in_tokens / n_calls`, so the
    division inverts it and the rounding only undoes float error. Zero input
    tokens means no LM was involved at all -- the majority, TF-IDF and encoder
    rows -- and those made no billed calls.
    """
    if row.get("n_calls") is not None:
        return row["n_calls"]
    mean_in = row.get("mean_in_tokens") or 0
    return round(row["total_in_tokens"] / mean_in) if mean_in else 0


def row_cost_per_1k_items(row: dict) -> float:
    """Cost per 1,000 classifications for a stored result row."""
    if row.get("cost_per_1k_items") is not None:
        return row["cost_per_1k_items"]
    n_calls = row_n_calls(row)
    if not n_calls:
        return row["cost_per_1k_calls"]
    return row["cost_per_1k_calls"] * n_calls / row["n"]


def run_eval(spec: ModelSpec, program, examples, labels: tuple[str, ...],
             split: str = "test", num_threads: int = DEFAULT_THREADS,
             max_tokens: int = 1024, progress: bool = True) -> EvalResult:
    """Score `program` on `examples`, measuring accuracy, tokens, and wall time."""
    lm = build_lm(spec, max_tokens=max_tokens)
    n = len(examples)
    predictions: list[str] = [""] * n
    correct = np.zeros(n, dtype=bool)
    parse_failures = 0
    errors = 0

    def one(i_ex):
        i, ex = i_ex
        # The context must be re-entered *inside* the worker: dspy.context is
        # thread-local, so a context opened on the main thread is invisible here
        # and every call would raise "No LM is loaded". Silent, and it looks
        # exactly like a model scoring 0%.
        try:
            with dspy.context(lm=lm):
                return i, program(sentence=ex.sentence), None
        except Exception as exc:  # a dead call is a zero, not a crashed run
            return i, None, exc

    # Token accounting reads the LM's own history rather than dspy.track_usage():
    # the usage tracker is stored in thread-local settings, so calls made on
    # worker threads never reach a tracker opened on the main thread and it
    # silently reports zero. history is appended per call from any thread.
    history_start = len(lm.history)

    t0 = time.time()
    with dspy.context(lm=lm):
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for i, pred, exc in pool.map(one, enumerate(examples)):
                if exc is not None:
                    errors += 1
                    predictions[i] = f"<error: {type(exc).__name__}>"
                    continue
                score, predicted, gold_label = score_prediction(examples[i], pred, labels)
                correct[i] = score == 1.0
                predictions[i] = predicted or "<unparseable>"
                if not predicted:
                    parse_failures += 1
    wall = time.time() - t0

    assert_ledger_intact(lm, history_start, f"run_eval({spec.key})")
    new_calls = lm.history[history_start:]
    total_in = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in new_calls)
    total_out = sum((c.get("usage") or {}).get("completion_tokens", 0) for c in new_calls)

    # Cached input is billed at a fraction of the normal rate and providers
    # engage it automatically once a shared prefix is long enough. On a
    # many-class prompt the label list is the overwhelming majority of every
    # request and is identical every time, so ignoring this overstates cost
    # several-fold. Providers report it under prompt_tokens_details.
    def _cached(call):
        det = (call.get("usage") or {}).get("prompt_tokens_details")
        if det is None:
            return 0
        if isinstance(det, dict):
            return det.get("cached_tokens") or 0
        return getattr(det, "cached_tokens", 0) or 0
    total_cached = sum(_cached(c) for c in new_calls)

    # Cache WRITES were previously discarded, so the 1.25x Anthropic premium on
    # creating a cache entry was billed as ordinary input and every cached run
    # understated its own cost. litellm surfaces the quantity three ways
    # depending on provider and version, and dspy stores dict(usage) verbatim,
    # so check all three rather than assume one.
    def _cache_writes(call):
        u = call.get("usage") or {}
        direct = u.get("cache_creation_input_tokens")
        if direct:
            return direct
        det = u.get("prompt_tokens_details")
        if isinstance(det, dict):
            return det.get("cache_creation_tokens") or 0
        return getattr(det, "cache_creation_tokens", 0) or 0 if det is not None else 0
    total_cache_write = sum(_cache_writes(c) for c in new_calls)
    # Divide by calls, not items: an adapter retry or a ChainOfThought step can
    # make several LM calls per item, and the bill counts calls.
    n_calls = len(new_calls)
    # The floor of 1 guards the division and nothing else. Reporting it as the
    # call count would have a run whose every call raised claim it made one.
    per_call = max(n_calls, 1)
    mean_in, mean_out = total_in / per_call, total_out / per_call
    mean_cached = total_cached / per_call
    mean_cache_write = total_cache_write / per_call

    instruction = ""
    for _, predictor in program.named_predictors():
        instruction = predictor.signature.instructions
        break

    result = EvalResult(
        model_key=spec.key, label=spec.label, split=split, n=n,
        correct=correct, predictions=predictions,
        gold=[ex.label for ex in examples],
        accuracy=accuracy_ci(correct),
        mean_in_tokens=mean_in, mean_out_tokens=mean_out,
        total_in_tokens=total_in, total_out_tokens=total_out,
        wall_seconds=wall, parse_failures=parse_failures, errors=errors,
        mean_cached_tokens=mean_cached, total_cached_tokens=total_cached,
        mean_cache_write_tokens=mean_cache_write,
        total_cache_write_tokens=total_cache_write,
        # A row that billed NOTHING for items it was asked to classify has an
        # UNMEASURED price, not a zero one. Without this, a model whose every
        # call raised (a 404 on an undeployed name is the common case -- the
        # registry has rows that 404 until their deployment exists) reported
        # mean_in = mean_out = 0, priced at $0.0000, and landed at the chart's
        # axis floor labelled "free": the most flattering position on the plot,
        # awarded to the row that answered nothing. NaN is the file's existing
        # signal for "the caller has to decide", and the chart already refuses
        # to draw it.
        cost_per_1k_calls=(spec.cost_per_1k_calls(mean_in, mean_out, mean_cached,
                                                  mean_cache_write)
                           if n_calls or not n else float("nan")),
        n_calls=n_calls,
        instruction=instruction,
    )
    if progress:
        print("  " + result.summary(), flush=True)
    # Said out loud regardless of `progress`, because the caller that silences
    # per-row output (gepa_one.py) is the one that most needs to hear it: GEPA
    # scored a prompt that fights the adapter as a win, since the retry is
    # invisible in accuracy and visible only in the bill. 5% is the noise floor
    # -- a stray transient retry lands under it, a prompt asking for JSON under
    # ChatAdapter lands near 100%.
    if n_calls > n * 1.05:
        print(f"  ! {spec.key}: {n_calls} billed calls for {n} items "
              f"(retry rate {result.retry_rate:.1%}) -- ChatAdapter is falling "
              f"back to JSONAdapter and billing twice for one classification. "
              f"Cost is ${result.cost_per_1k_items:.5f}/1k classifications, "
              f"not ${result.cost_per_1k_calls:.5f}.", flush=True)
    return result
