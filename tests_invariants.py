"""Invariants that must hold or the numbers are meaningless.

Not an exhaustive unit-test suite -- these are the specific properties whose
violation would silently produce a wrong-but-plausible chart. Run before trusting
a result.

    python tests_invariants.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from downshift import stats as S
from downshift.data import load_financial_phrasebank
from downshift.metric import normalize, score_prediction

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


print("Splits:")
task = load_financial_phrasebank(n_train=200, n_val=120, n_test=250, seed=0)
splits = {n: {e.sentence for e in getattr(task, n)} for n in ("train", "val", "test", "pool")}
for a in splits:
    for b in splits:
        if a < b:
            check(f"{a} and {b} are disjoint", not (splits[a] & splits[b]),
                  f"{len(splits[a] & splits[b])} shared")
check("test set is the requested size", len(task.test) == 250, f"{len(task.test)}")
check("no duplicate sentences survive", sum(len(v) for v in splits.values()) == len(set().union(*splits.values())))
bal = task.class_balance("test")
check("test set is stratified (all 3 classes present)", all(v > 0.05 for v in bal.values()),
      str({k: f"{v:.1%}" for k, v in bal.items()}))
_, maj = task.majority_baseline
check("majority baseline is reported", 0.5 < maj < 0.7, f"{maj:.1%}")

print("\nSplit determinism:")
t2 = load_financial_phrasebank(n_train=200, n_val=120, n_test=250, seed=0)
check("same seed gives identical test set",
      [e.sentence for e in task.test] == [e.sentence for e in t2.test])
t3 = load_financial_phrasebank(n_train=200, n_val=120, n_test=250, seed=1)
check("different seed gives a different test set",
      [e.sentence for e in task.test] != [e.sentence for e in t3.test])

print("\nLabel normalisation (formatting must not count as being wrong):")
for raw, want in [("**Positive**.", "positive"), ("Label: negative", "negative"),
                  (" NEUTRAL \n", "neutral"), ("the sentiment is positive", "positive")]:
    check(f"{raw!r} -> {want!r}", normalize(raw) == want, normalize(raw))


class _Ex:
    def __init__(self, label): self.label = label; self.sentence = "x"


check("decorated correct answer scores 1.0",
      score_prediction(_Ex("positive"), _Ex("**Positive**"), ("negative", "neutral", "positive"))[0] == 1.0)
check("unparseable answer scores 0.0 and yields no label",
      score_prediction(_Ex("positive"), _Ex("banana"), ("negative", "neutral", "positive"))[:2] == (0.0, ""))

# A label set is not guaranteed to be lowercase. banking77 has exactly one
# capitalised label out of 77, and because normalize() lowercases the answer
# while the membership test compared against the raw label strings, a model
# that answered that intent PERFECTLY scored 0 and was booked as a parse
# failure. It cost every LLM row a flat 1.6pp -- and only the LLM rows, since
# the classical baselines predict by index and never reach this matcher, so it
# biased exactly the comparison the benchmark exists to make.
_MIXED = ("Refund_not_showing_up", "card_payment_fee_charged", "atm_support")
_score, _pred, _gold = score_prediction(_Ex("Refund_not_showing_up"),
                                        _Ex("Refund_not_showing_up"), _MIXED)
check("a correct answer in a differently-cased label set scores 1.0",
      _score == 1.0, f"score={_score} predicted={_pred!r} gold={_gold!r}")
check("the returned labels stay canonical, not lowercased",
      _pred in _MIXED and _gold in _MIXED, f"predicted={_pred!r} gold={_gold!r}")
check("a decorated mixed-case answer still resolves",
      score_prediction(_Ex("Refund_not_showing_up"),
                       _Ex("**refund_not_showing_up**."), _MIXED)[0] == 1.0)
check("a genuinely invented label still scores 0 with no label",
      score_prediction(_Ex("atm_support"), _Ex("get_virtual_card"), _MIXED)[:2] == (0.0, ""))

print("\nStatistics:")
ci = S.wilson_ci(85, 100)
check("Wilson interval brackets the point estimate", ci.lo < ci.point < ci.hi, str(ci))
check("Wilson interval stays inside [0,1] at the boundary",
      0 <= S.wilson_ci(100, 100).lo and S.wilson_ci(100, 100).hi <= 1.0, str(S.wilson_ci(100, 100)))
check("Wilson interval narrows as n grows",
      S.wilson_ci(850, 1000).half_width < S.wilson_ci(85, 100).half_width)

rng = np.random.default_rng(0)
a = rng.random(250) < 0.90
check("identical vectors give p = 1.0", S.mcnemar_test(a, a).p_value == 1.0)
b = a.copy(); b[rng.choice(250, 30, replace=False)] = False
mc = S.mcnemar_test(a, b)
check("a strictly-better model is detected", mc.p_value < 0.01 and mc.diff > 0,
      f"p={mc.p_value:.5f} diff={mc.diff:+.3f}")
check("mid-p is less conservative than exact conditional",
      S.mcnemar_test(a, b, method="mid-p").p_value <= S.mcnemar_test(a, b, method="exact").p_value)

ni_same = S.non_inferiority_test(a, a, margin=0.03)
check("a model is non-inferior to itself", ni_same.passes, ni_same.verdict)
ni_worse = S.non_inferiority_test(b, a, margin=0.03)
check("a clearly worse model fails non-inferiority", not ni_worse.passes, ni_worse.verdict)
# Two genuinely different models on a tiny set. Comparing a vector to itself
# would pass trivially and correctly -- every bootstrap resample of (x - x) is
# exactly zero -- so it tests nothing about sample size.
tiny_a = a[:20].copy()
tiny_b = tiny_a.copy(); tiny_b[0] = ~tiny_b[0]; tiny_b[1] = ~tiny_b[1]
tiny = S.non_inferiority_test(tiny_b, tiny_a, margin=0.03)
check("a tiny test set returns 'inconclusive' rather than a false pass",
      not tiny.passes and "inconclusive" in tiny.verdict, tiny.verdict)
check("a model is non-inferior to an identical copy of itself",
      S.non_inferiority_test(a, a.copy(), margin=0.03).passes)

# Holm works on RANK order, so a significant p-value is rejected wherever it
# sits in the input list -- both 0.001 entries clear their thresholds here.
check("Holm rejects by rank, not by input position",
      S.holm_bonferroni([0.001, 0.9, 0.001]) == [True, False, True],
      str(S.holm_bonferroni([0.001, 0.9, 0.001])))
check("Holm is stricter than uncorrected alpha",
      S.holm_bonferroni([0.04, 0.04, 0.04]) == [False, False, False],
      str(S.holm_bonferroni([0.04, 0.04, 0.04])))
check("Holm on an empty list is empty", S.holm_bonferroni([]) == [])

print("\nCost model:")
from downshift.models import BY_KEY, UNKNOWN
opus = BY_KEY["claude-opus-4-8"]
direct, reasoning = opus.cost_per_1k_calls(264, 5), opus.cost_per_1k_calls(264, 300)
check("reasoning tokens raise measured cost", reasoning > direct * 3,
      f"${direct:.4f} -> ${reasoning:.4f} per 1k")
check("zero usage costs zero", opus.cost_per_1k_calls(0, 0) == 0.0)
check("every registered hosted model has a known price",
      all(BY_KEY[k].price_known for k in BY_KEY if BY_KEY[k].runtime == "hosted"),
      str([k for k in BY_KEY if BY_KEY[k].runtime == "hosted" and not BY_KEY[k].price_known]))
# This guard used to read `... if not BY_KEY["claude-opus-4-8"].price_known
# else True`. Opus HAS a known price, so the whole assertion collapsed to
# True and tested nothing -- it would not have caught the chart drawing an
# unverified price at the axis floor, i.e. as FREE, which is the most
# flattering spot on the plot. Construct a genuinely unpriced spec instead.
from downshift.chart import ChartRow, plot_cost_vs_accuracy
from downshift.models import UNKNOWN, ModelSpec, HOSTED, OPEN
_unpriced = ModelSpec("probe-unpriced", HOSTED, OPEN, "Unpriced probe",
                      price_in=UNKNOWN, price_out=UNKNOWN)
check("an unverified price is not a known price", not _unpriced.price_known)
_rows = [ChartRow(label="ok", family="llm", cost_per_1k=0.2, accuracy=0.8, ci_lo=0.7, ci_hi=0.9),
         ChartRow(label="unpriced", family="llm", cost_per_1k=float("nan"),
                  accuracy=0.8, ci_lo=0.7, ci_hi=0.9)]
try:
    plot_cost_vs_accuracy(_rows, "/dev/null")
    _refused = False
except ValueError:
    _refused = True
check("an unknown price is refused by the chart, not drawn as free", _refused)

# The cost caption is DERIVED from the rows now. It was hand-copied into five
# scripts and every copy claimed "Locally-run rows are $0 marginal" while the
# same charts plotted six local rows at $0.003-$0.061.
from downshift.chart import cost_footnote
_mixed = [ChartRow(label="enc", family="encoder", cost_per_1k=0.0, accuracy=0.9,
                   ci_lo=0.8, ci_hi=0.95, cost_basis="free"),
          ChartRow(label="local-llm", family="llm", cost_per_1k=0.061, accuracy=0.5,
                   ci_lo=0.4, ci_hi=0.6, cost_basis="proxy")]
_cap = cost_footnote(_mixed, "2026-09-09")
# Assert the PROPERTY, not the prose. Pinning exact wording is the same
# brittleness that produced the original bug: the caption was hand-copied into
# five scripts and drifted from the data. What must hold is that a caption
# describing a mix of conventions names both, and never claims the priced row
# is free.
check("caption names the proxy convention when a proxy row is present", "PROXY" in _cap)
check("caption names the $0 convention when a free row is present", "$0" in _cap)
check("caption does not claim every local row is free",
      "Locally-run rows are $0 marginal" not in _cap)
_free_only = cost_footnote([_mixed[0]], "2026-09-09")
check("caption omits the proxy clause when nothing is a proxy", "PROXY" not in _free_only)

# A dataset is meant to be addable as DATA, not code (data.py:47). So every
# registered task's own labels must survive scoring against themselves, and no
# two may collide -- otherwise a new task silently scores 0 or credits one
# label's answer to another. This is the guard that was missing when
# normalize() was mangling `answerable` -> `able`.
# TaskSpec carries no static label set -- labels come from the dataset's
# ClassLabel feature at load time -- so read them from each task's saved run.
# That is offline and exact. Limitation, stated rather than hidden: this covers
# tasks that have been RUN. A brand-new task is still first checked by the
# label-shape guards below, which use the strings that actually broke.
from downshift.data import TASKS
from downshift.metric import normalize, score_prediction as _sp

class _Ans:
    def __init__(self, label): self.label = label

_task_labels = {}
for _key in TASKS:
    _f = Path(f"runs/{_key}/results.json")
    if not _f.exists():
        continue
    _rows = json.loads(_f.read_text())["results"]
    _task_labels[_key] = tuple(sorted({g for r in _rows.values() for g in r.get("gold", [])}))
check("every registered task has labels available to check",
      set(_task_labels) == set(TASKS),
      f"unchecked: {sorted(set(TASKS) - set(_task_labels))}")

for _key, _labels in _task_labels.items():
    if not _labels:
        continue
    _self_fail = [l for l in _labels if _sp(_Ans(l), _Ans(l), tuple(_labels))[0] != 1.0]
    check(f"[{_key}] every label scores 1.0 against itself", not _self_fail, str(_self_fail))
    _norm = {}
    for l in _labels:
        _norm.setdefault(normalize(l), []).append(l)
    _collide = {k: v for k, v in _norm.items() if len(v) > 1}
    check(f"[{_key}] no two labels collide under normalize()", not _collide, str(_collide))

# Shape guards, independent of any dataset. These are the exact label strings
# that scored 0 while being exactly right, because normalize() ran before any
# match was attempted: a prefix strip with no word boundary ate "answer" out of
# "answerable", and punctuation-to-space turned "U.S." into "u s". LABEL_0 is
# the one that matters in practice -- HuggingFace auto-converted datasets name
# classes that way.
for _lab, _set in [("answerable", ("answerable", "unanswerable")),
                   ("sentiment_positive", ("sentiment_positive", "sentiment_negative")),
                   ("U.S.", ("U.S.", "U.K.")),
                   ("LABEL_0", ("LABEL_0", "LABEL_1")),
                   ("it is fine", ("it is fine", "it is broken"))]:
    check(f"a label named {_lab!r} scores itself 1.0",
          _sp(_Ans(_lab), _Ans(_lab), _set)[0] == 1.0)
# ...and two labels that only normalize() would confuse must stay distinct.
check("punctuation-distinct labels are not conflated",
      _sp(_Ans("Dr"), _Ans("Dr."), ("Dr.", "Dr"))[0] == 0.0)
check("decoration is still stripped for genuinely decorated answers",
      _sp(_Ans("positive"), _Ans("**Positive**."), ("positive", "negative"))[0] == 1.0)

print("\nA billed call is not a classification:")
# The chart axis says "per 1,000 classifications". Cost was reported per billed
# CALL, and DSPy re-sends an item through a second adapter when the first fails
# to parse -- so one row was billed 500 calls for 251 items and charted at half
# its real cost. At one call per item the two numbers coincide, which is exactly
# why it survived every other row and needs pinning here rather than by eye.
import glob
import json
import math

from downshift.evaluate import EvalResult
from downshift.stats import accuracy_ci


def _row(n, n_calls, per_call):
    return EvalResult(
        model_key="k", label="L", split="test", n=n,
        correct=[True] * n, predictions=["x"] * n, gold=["x"] * n,
        accuracy=accuracy_ci([True] * n),
        mean_in_tokens=1.0, mean_out_tokens=1.0,
        total_in_tokens=n_calls, total_out_tokens=n_calls,
        wall_seconds=0.0, parse_failures=0, errors=0,
        cost_per_1k_calls=per_call, n_calls=n_calls)


one, two = _row(250, 250, 0.1), _row(250, 500, 0.1)
check("one call per item: per-item cost equals per-call cost",
      one.cost_per_1k_items == one.cost_per_1k_calls, f"${one.cost_per_1k_items:.5f}")
check("two calls per item: per-item cost is exactly double",
      two.cost_per_1k_items == 2 * two.cost_per_1k_calls,
      f"${two.cost_per_1k_calls:.5f}/call -> ${two.cost_per_1k_items:.5f}/item")
check("the per-call cost is unchanged by retries (this is why it hid the bug)",
      one.cost_per_1k_calls == two.cost_per_1k_calls)
check("a retry is visible in summary(), the only place it shows",
      "retries" in two.summary() and "retries" not in one.summary())
# Local rows make no API calls at all; they must not inherit a call count, or a
# tooltip claims 251 billed calls for a model that never touched a network.
free = _row(250, 0, 0.0)
check("a row with no billed calls reports per-item == per-call",
      free.n_calls == 0 and free.cost_per_1k_items == free.cost_per_1k_calls)
# An unpriced model must stay off the cost axis rather than reappear at $0.00 --
# the same guard as the per-call case above, now on the derived number.
check("unknown price stays NaN per item, not 0.0",
      math.isnan(_row(250, 500, float("nan")).cost_per_1k_items))
check("EvalResult still constructs without the new fields",
      EvalResult(model_key="k", label="L", split="test", n=1, correct=[True],
                 predictions=["x"], gold=["x"], accuracy=accuracy_ci([True]),
                 mean_in_tokens=0.0, mean_out_tokens=0.0, total_in_tokens=0,
                 total_out_tokens=0, wall_seconds=0.0, parse_failures=0,
                 errors=0, cost_per_1k_calls=0.0).n_calls == 0)

# And the same property on every row already written to disk: a stale writer or
# a hand-edited file would silently revert the chart to per-call cost.
print("\nStored results reconcile with the cost model:")
for path in sorted(glob.glob("runs/*/results.json")):
    task_name = path.split("/")[1]
    rows = json.loads(open(path).read())["results"]
    bad_calls, bad_cost, understated = [], [], []
    for key, r in rows.items():
        mean_in, n = r["mean_in_tokens"], r["n"]
        want = round(r["total_in_tokens"] / mean_in) if mean_in > 0 else 0
        if r.get("n_calls") != want:
            bad_calls.append(key)
        got, per_call = r.get("cost_per_1k_items"), r["cost_per_1k_calls"]
        expect = per_call * r["n_calls"] / n if r["n_calls"] else per_call
        if not (math.isnan(expect) and got is not None and math.isnan(got)):
            if got is None or abs(got - expect) > 1e-12:
                bad_cost.append(key)
        # Fewer billed calls than items would mean the history window was
        # measured wrong -- the same bug with the sign flipped, understating
        # the bill instead of the cost per item.
        if mean_in > 0 and r["n_calls"] < n:
            understated.append(f"{key}({r['n_calls']}<{n})")
    check(f"{task_name}: every row's call count matches its own token totals",
          not bad_calls, str(bad_calls))
    check(f"{task_name}: every row's per-item cost is per-call x calls/item",
          not bad_cost, str(bad_cost))
    check(f"{task_name}: no LM row bills fewer calls than it has items",
          not understated, str(understated))

print("\nOutput format is the adapter's contract, not the optimizer's:")
# GEPA appended `Return a JSON object with exactly: {"label": ...}` to an
# instruction, ChatAdapter could not parse the result, DSPy retried every item
# through JSONAdapter, and the bill doubled while accuracy stayed flat. The pin
# has to survive re-application, or re-pinning an already-pinned prompt grows
# the prompt on every optimization round.
from downshift.program import (FORMAT_CONTRACT, instruction_changed,
                               optimizable_body, with_format_contract)

seed = "Classify the sentiment of a financial news sentence."
once = with_format_contract(seed)
check("pinning adds the contract", FORMAT_CONTRACT in once)
check("pinning twice is the same as pinning once", with_format_contract(once) == once,
      f"{len(once)} vs {len(with_format_contract(once))} chars")
check("the pin is not itself a prompt change", not instruction_changed(seed, once))
check("a real rewrite is still detected",
      instruction_changed(seed, with_format_contract(seed + "\nPrefer neutral.")))
# The pin must be purely ADDITIVE. It runs after GEPA has already scored the
# candidate, so any deletion means the shipped prompt is not the prompt that
# earned the score. An earlier version deleted lines it read as rival format
# directives: one candidate lost 61 of 430 tokens and scored 0.250 on the same
# val items GEPA had scored 0.767 on -- a 51-point silent regression reported as
# a win. Deleting was never needed; appending alone was measured at 1.000 billed
# calls per item with the rogue line left in.
rogue = seed + '\nOutput format (strict):\n{"label": "positive"|"neutral"}'
pinned_rogue = with_format_contract(rogue)
check("pinning deletes nothing from the candidate",
      optimizable_body(pinned_rogue) == rogue.strip())
check("the pin is the last word, so it overrides what it did not delete",
      pinned_rogue.rstrip().endswith(FORMAT_CONTRACT.rstrip()))
check("a multi-line format directive survives pinning intact",
      "Output format (strict):" in pinned_rogue and '{"label"' in pinned_rogue)

# ── deployment limits vs the vendor's own card ──────────────────────────────
#
# ModelSpec carries the DEPLOYMENT's limits and model_cards carries the upstream
# model's, and they disagree constantly -- Azure caps Mistral Large 3 at 4,096
# output tokens where Mistral publishes no cap at all and a 256k context where
# Azure gives 128k. That is not an error to be reconciled away; the deployment's
# number is the one that binds at call time and the vendor's is the one that
# tells you what the weights can do. What IS an error is an undocumented gap,
# because then a reader cannot tell a real restriction from a stale transcription.
from downshift.model_cards import CARDS
from downshift.models import ALL_MODELS

undocumented = []
for spec in ALL_MODELS:
    card = CARDS.get(spec.key)
    if card is None or card.caveat:
        continue
    if ((card.context and spec.context_window and card.context != spec.context_window)
            or (card.max_output and spec.max_output and card.max_output != spec.max_output)):
        undocumented.append(spec.key)
check("every deployment/vendor limit gap is explained in a caveat",
      not undocumented, f"undocumented: {undocumented}")

# A limit of 0 must mean "not recorded" everywhere, never "no output allowed" --
# a model that silently could not be asked for any tokens would be excluded from
# every sweep with no reason given.
zero_but_used = [s.key for s in ALL_MODELS
                 if s.runtime == "hosted" and s.max_output == 0 and not s.fits(1024)[0]]
check("an unrecorded limit never disqualifies a model", not zero_but_used,
      f"blocked on unknown limits: {zero_but_used}")

print("\nCapability screen (the gate must stay wired, and stay honest):")
import inspect
from downshift.data import get_task, load_task
from downshift.experiment import ExperimentConfig, screen_candidates, _budget_for, run_experiment
from downshift.models import BY_KEY, suitable_for

# 1. The gate is REACHABLE. It was fully built and never called once: fits()
# had one caller that passed prompt_tokens=0 (so the context branch could not
# fire) and suitable_for had none at all. Dead code that looks live is worse
# than missing code, because the docstring claims a protection nobody has.
src = inspect.getsource(run_experiment)
check("run_experiment actually calls the capability screen",
      "screen_candidates(cfg, task)" in src)
check("the sweep iterates the screened list, not the raw config",
      'for key in screen["runnable"]' in src,
      "otherwise a rejected model is still called")

fpb = load_task(get_task("financial_phrasebank"), 200, 120, 250, 0)
d = fpb.demands()
check("task demands are measured, not zero",
      d.max_output_tokens > 0 and d.p95_prompt_tokens > 0, str(d))
check("p95 prompt length is bracketed by mean and max",
      d.mean_prompt_tokens <= d.p95_prompt_tokens <= d.max_prompt_tokens)

# 2. It must actually refuse an impossible model. If this stops firing, the
# whole stage has silently become a no-op again.
big = ExperimentConfig(task="financial_phrasebank", optimize=[],
                       candidates=["gpt-5-nano", "mistral-large-3"], max_tokens=8000)
s_big = screen_candidates(big, fpb)
check("a model capped below the requested budget is refused",
      "mistral-large-3" in dict(s_big["rejected"]),
      f"rejected: {list(dict(s_big['rejected']))}")
check("and a model that fits is kept", "gpt-5-nano" in s_big["runnable"])

# 3. It must NOT refuse on an unknown limit -- that would silently drop every
# Fireworks row, whose caps the Azure catalogue does not carry.
unk = ExperimentConfig(task="financial_phrasebank", optimize=[],
                       candidates=["glm-5.3-flash"], max_tokens=8000)
s_unk = screen_candidates(unk, fpb)
check("an unrecorded limit passes, flagged rather than rejected",
      "glm-5.3-flash" in s_unk["runnable"]
      and "glm-5.3-flash" in dict(s_unk["unverified"]))

# 4. The budget the screen checks must be the budget the run requests. A screen
# against a different number reads as "checked" while checking nothing.
#
# `want` goes through requested_max_tokens for the same reason the screen does:
# build_lm raises the budget to 16,000 for the OpenAI reasoning tier, so a
# `want` computed from the config alone asserted that the screen checks 1,024
# for gpt-5-nano while the request carries 16,000 -- the check passed and the
# mismatch it exists to catch was the one it was measuring against.
from downshift.evaluate import requested_max_tokens as _req_mt

# The expectation comes from `_profile_for`, not from a second copy of the rule
# it replaced. `key.endswith("-reasoning")` was the rule _profile_for was
# rewritten for getting wrong, so asserting against it made the test disagree
# with production on exactly the rows the rewrite was about -- and the loop only
# covered two keys, so it could not see the disagreement.
from downshift.experiment import _profile_for as _prof_for
for key in ("gpt-5-nano", "qwen3-1.7b-reasoning", "qwen3-1.7b-direct",
            "grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning"):
    spec = BY_KEY.get(key)
    if spec is None:
        continue
    cfg_b = ExperimentConfig()
    configured = (cfg_b.reasoning_max_tokens
                  if _prof_for(key) == "reasoning" or spec.think or "reasoning" in key
                  else cfg_b.max_tokens)
    want = _req_mt(spec, configured)
    check(f"screen budget matches run budget for {key}",
          _budget_for(cfg_b, key, spec) == want, f"{_budget_for(cfg_b, key, spec)}")

# A vendor deployment that reasons natively must keep the reasoning OUTPUT
# budget even though it runs under the DIRECT wrapper. Conflating the two
# dropped the Grok pair from 2,500 output tokens to 1,024, which a reasoning
# deployment spends inside its own reasoning before emitting a label.
for key in ("grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning"):
    spec = BY_KEY.get(key)
    if spec is None:
        continue
    cfg_b = ExperimentConfig()
    check(f"{key} runs the DIRECT wrapper", _prof_for(key) == "direct", _prof_for(key))
    check(f"{key} still gets the reasoning output budget",
          _budget_for(cfg_b, key, spec) == cfg_b.reasoning_max_tokens,
          f"{_budget_for(cfg_b, key, spec)} vs {cfg_b.reasoning_max_tokens}")

# The pin detector has to be ABLE to notice a deletion. Asked through
# `optimizable_body` on both sides it was False by construction on every run, so
# it could not have caught the 51-point regression it exists to catch.
#
# Note what this does and does not assert. Today's `with_format_contract` only
# appends -- it strips the contract and re-adds it, nothing else -- so no input
# makes it delete, and an earlier version of this test wrongly expected a rogue
# trailer to be removed. So the predicate is exercised directly: it must be
# False when the selected text survives and True when it does not.
from downshift.program import with_format_contract as _wfc

def _pin_altered(selected, optimized):
    return selected.strip() not in optimized

_clean = "Classify the sentiment."
check("pin detector is quiet when the pin only appends",
      not _pin_altered(_clean, _wfc(_clean)))
check("pin detector fires when the shipped text lost the selected body",
      _pin_altered(_clean, "Answer with one word.\n\nSomething else entirely."))
check("and today's pin genuinely only appends, for every shape tried",
      all(t.strip() in _wfc(t) for t in
          (_clean, "Classify.\n\nReturn a JSON object with one key.", "  spaced  ")))

# Auto-selecting the reference must not land on a free row. `results` holds the
# classical and encoder rows too, and on banking77 the encoder is the most
# accurate row on the board, which would make "non-inferior to a free TF-IDF"
# the published question.
import json as _json
from pathlib import Path as _Path
_f = _Path("runs/banking77/results.json")
if _f.exists():
    _res = _json.loads(_f.read_text())["results"]
    _prompted = {k: v for k, v in _res.items() if v["n_calls"] > 0}
    _auto = max(_prompted, key=lambda k: _prompted[k]["accuracy"]["point"])
    check("auto-selected reference is a prompted row, not a free one",
          _res[_auto]["n_calls"] > 0, f"{_auto}")
    _overall = max(_res, key=lambda k: _res[k]["accuracy"]["point"])
    check("and the most accurate row on that task IS a free one, so it mattered",
          _res[_overall]["n_calls"] == 0, f"{_overall}")

# 5. The reflection model is screened too. It used to be built unguarded at the
# top of run_gepa, so an undersized one raised AFTER every paid eval was billed.
bad_ref = ExperimentConfig(task="financial_phrasebank", candidates=["gpt-5-nano"],
                           optimize=["gpt-5-nano"], reflection_model="mistral-large-3")
check("an undersized reflection model is caught before spending",
      not screen_candidates(bad_ref, fpb)["reflection_ok"])
check("reflection_max_tokens is on the config, not a hidden default",
      hasattr(ExperimentConfig(), "reflection_max_tokens"))

# 6. Split sizes must be exactly what was asked for. Per-class rounding used to
# overshoot with no trim: 29 classes -> 261 items, 151 -> 302, and banking77's
# test split was 251. The caller pays per item.
from downshift.data import _stratified_take
for n_classes in (3, 29, 77, 151):
    by = {f"l{i}": list(range(200)) for i in range(n_classes)}
    before = sum(len(v) for v in by.values())
    got = _stratified_take(np.random.default_rng(0), by, 250, tuple(by))
    check(f"stratified take returns exactly 250 at {n_classes} classes",
          len(got) == 250, f"got {len(got)}")
    check(f"and conserves every item at {n_classes} classes",
          len(got) + sum(len(v) for v in by.values()) == before)

# 7. An unrecorded cache minimum must not read as a verified one. It used to
# return (True, "") -- indistinguishable from a checked pass -- while 8 rows
# claim a 0.10-0.20x cached rate with no minimum on file.
from downshift.models import HOSTED_MODELS
silent = [s.key for s in HOSTED_MODELS
          if s.price_cached >= 0 and not s.cache_min_tokens
          and s.cache_applies(271) == (True, "")]
check("an unrecorded cache minimum is flagged, not assumed verified",
      not silent, f"silently assumed: {silent}")

# 8. The six defects the second review pass left in place, each with the exact
# input that reproduced it. They are grouped here because every one of them
# moves a published number rather than raising, which is the class of bug this
# file exists to catch.
print("\nRegressions (each of these shipped a wrong-but-plausible number):")

# A refusal or a negation names the label it is rejecting, and a bare substring
# test credited it. Both of these were scored 1.0.
check("a refusal is not credited as the label inside it",
      _sp(_Ans("no"), _Ans("unknown"), ("yes", "no"))[0] == 0.0,
      str(_sp(_Ans("no"), _Ans("unknown"), ("yes", "no"))))
check("a negated label is not credited",
      _sp(_Ans("positive"), _Ans("The sentiment is not positive"),
          ("positive", "negative", "neutral"))[0] == 0.0)
# ...while the whole-word rule must still recover a label from a longer answer,
# including the nested banking77 pairs.
check("a decorated label that begins with a stripped prefix still resolves",
      _sp(_Ans("answerable"), _Ans("**answerable**"),
          ("answerable", "unanswerable"))[0] == 1.0)
check("a nested label is recovered from a verbose answer",
      _sp(_Ans("virtual_card_not_working"),
          _Ans("The intent is virtual_card_not_working"),
          ("card_not_working", "virtual_card_not_working"))[0] == 1.0)
check("the shorter half of a nested pair is not credited for the longer",
      _sp(_Ans("card_not_working"),
          _Ans("The intent is virtual_card_not_working"),
          ("card_not_working", "virtual_card_not_working"))[0] == 0.0)
check("two labels named in one answer are still refused",
      _sp(_Ans("positive"), _Ans("could be positive or negative"),
          ("positive", "negative", "neutral"))[0] == 0.0)

# The non-inferiority decision and the p-value printed beside it must be one
# test on one sample. They were drawn on `seed` and `seed + 1`, and experiment.py
# ANDs them together, so the two could disagree at the boundary.
def _paired(n, a_only, b_only, both):
    ca = np.array([1] * a_only + [0] * b_only + [1] * both + [0] * (n - a_only - b_only - both))
    cb = np.array([0] * a_only + [1] * b_only + [1] * both + [0] * (n - a_only - b_only - both))
    return ca, cb

_disagree = []
for _n in (50, 100, 200, 250):
    for _a in range(0, 21, 2):
        for _b in range(0, 21, 2):
            if _a + _b + 30 > _n:
                continue
            _ni = S.non_inferiority_test(*_paired(_n, _a, _b, 30), n_boot=4000)
            if _ni.passes != (_ni.p_value < 0.05):
                _disagree.append((_n, _a, _b, _ni.passes, round(_ni.p_value, 4)))
check("the non-inferiority verdict and its p-value agree on every config",
      not _disagree, f"{len(_disagree)} disagreements, e.g. {_disagree[:2]}")

# An undersized pool must name the shortfall. It used to under-fill the last
# split silently, or divide by zero once the pool was empty.
from downshift.data import _stratified_take as _take

def _pool(total, n_classes=3):
    per = total // n_classes
    return {f"c{i}": [type("E", (), {"label": f"c{i}", "sentence": f"c{i}-{j}"})()
                      for j in range(per)] for i in range(n_classes)}

for _total, _want in ((600, "refused"), (450, "refused"), (900, "served")):
    _by, _rng, _sizes, _how = _pool(_total), np.random.default_rng(0), [], "served"
    try:
        for _n in (250, 200, 200):
            _sizes.append(len(_take(_rng, _by, _n, tuple(_by))))
    except ValueError:
        _how = "refused"
    except ZeroDivisionError:
        _how = "ZeroDivisionError"
    check(f"a {_total}-row pool asked for 650 items is {_want}", _how == _want,
          f"{_how}, sizes {_sizes}")

# The `-reasoning` suffix is our prompt profile, not part of a vendor deployment
# name. Both xAI rows matched it (the non-reasoning one too) and were wrapped in
# ChainOfThought, so the pair that isolates the weights switch had two switches
# flipped.
from downshift.experiment import _profile_for as _prof
check("a vendor deployment named '-reasoning' keeps the DIRECT profile",
      _prof("grok-4-1-fast-reasoning") == "direct", _prof("grok-4-1-fast-reasoning"))
check("and so does its '-non-reasoning' sibling",
      _prof("grok-4-1-fast-non-reasoning") == "direct",
      _prof("grok-4-1-fast-non-reasoning"))
check("a real profile pair (same call_id) still reads as REASONING",
      _prof("qwen3-1.7b-reasoning") == "reasoning", _prof("qwen3-1.7b-reasoning"))

# The screen must check the LONGEST prompt. Every test item is scored, so an
# item over the context window returns a provider error and is booked wrong --
# on a model the screen had just reported as fitting every deployment.
import inspect as _inspect
from downshift.experiment import screen_candidates as _screen
_screen_src = _inspect.getsource(_screen)
check("the capability screen sizes context against the longest prompt",
      "max_prompt_tokens" in _screen_src and "p95_prompt_tokens" not in _screen_src)

# A log x-axis must not be widened by its own tick list: set_xticks fixes the
# ticks AND expands the view to contain them, and the locator's list runs a
# decade past the data.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
from downshift.chart import ChartRow as _Row, plot_cost_vs_accuracy as _plot
import tempfile as _tempfile

_rows_x = [_Row("free local", "local", 0.0, 0.40, 0.34, 0.46),
           _Row("cheap host", "openai", 0.05, 0.62, 0.56, 0.68),
           _Row("frontier", "anthropic", 20.0, 0.81, 0.76, 0.86)]
_seen = {}
_orig_savefig = _plt.Figure.savefig


def _spy(self, *a, **k):
    _seen["xlim"] = self.axes[0].get_xlim()
    return _orig_savefig(self, *a, **k)


_plt.Figure.savefig = _spy
try:
    _plot(_rows_x, str(Path(_tempfile.mkdtemp()) / "x.png"),
          majority_baseline=0.33, accuracy_bar=0.60)
finally:
    _plt.Figure.savefig = _orig_savefig
check("a free row's floor tick does not widen the axis past the priciest row",
      _seen["xlim"][1] < 50.0, f"right edge {_seen['xlim'][1]:.1f} for a $20 max")
check("no figure is left open after charting", not _plt.get_fignums(),
      str(_plt.get_fignums()))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("All invariants hold.")
