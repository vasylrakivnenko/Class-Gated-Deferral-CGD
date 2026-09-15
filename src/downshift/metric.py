"""Scoring, and the textual feedback GEPA reflects on.

GEPA differs from a plain search optimizer in one respect that decides how well
it works: it does not merely see that a candidate scored 0.62, it *reads* why
each item failed and writes a new instruction in response. The feedback string
is therefore not logging -- it is the optimizer's only view of the task, and a
metric that returns "score: 0" teaches it nothing.

So `classification_feedback` says which label was produced, which was wanted,
and what distinction the confusion turns on. On Financial PhraseBank the
dominant error is neutral-vs-directional: models read any mention of money as
sentiment, when the annotation guideline only counts an explicit change in the
company's prospects. Naming that in the feedback is what lets GEPA write it into
the instruction, and it is the single biggest lever on the final number.

One caveat this file exists to enforce: `feedback_hint` encodes human knowledge
about the dataset. That is legitimate -- a real user knows their own labelling
guideline -- but it must be measured on the *test* split, never the val split
GEPA selects against, or we are just reporting how well we hinted.

The other half of this file is `PromptCostPenalty`, which is what makes the
optimizer able to see money at all. GEPA optimizes the metric it is given, so an
accuracy-only metric prices prompt length at zero and GEPA duly spends it:
measured on the eight optimized rows on disk it replaced a 101-character
instruction with a 1,500-2,600-character rubric and multiplied the per-
classification bill by 2.1-3.6x. Sometimes that is a good trade (Qwen3 0.6B,
28.8% -> 52.4%, mid-p McNemar < 1e-4 against its own stock prompt) and sometimes
it buys a statistically unproven +1.6pp for a certain +262% bill (GPT-5.4-nano,
mid-p 0.4421). Nothing the accuracy-only metric can see distinguishes those two.
The penalty is opt-in and defaults to off, so every number already on disk stays
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import dspy

# Distinctions worth naming when a specific confusion shows up. Anything not
# listed falls back to the generic message.
_CONFUSION_HINTS = {
    ("neutral", "positive"): (
        "The sentence reports an actual improvement in the company's position "
        "(growth, profit up, a won contract), not a neutral fact."
    ),
    ("neutral", "negative"): (
        "The sentence reports an actual deterioration (decline, loss, layoffs, "
        "a lost contract), not a neutral fact."
    ),
    ("positive", "neutral"): (
        "Reporting a number, a transaction, a date or a plan is NOT positive "
        "sentiment. Neutral is correct unless the sentence states that the "
        "company's position improved."
    ),
    ("negative", "neutral"): (
        "Describing a cost, a debt, a restructuring or a risk factor is NOT "
        "negative sentiment on its own. Neutral is correct unless the sentence "
        "states that the company's position worsened."
    ),
    ("positive", "negative"): (
        "Direction reversed: the sentence describes a decline, not a gain. "
        "Check which way the numbers move and who benefits."
    ),
    ("negative", "positive"): (
        "Direction reversed: the sentence describes a gain, not a decline. "
        "Check which way the numbers move and who benefits."
    ),
}


def _undecorated(value) -> str:
    """Lowercase and drop decoration, WITHOUT eating any leading word.

    `normalize` does two separable jobs, and only the first is safe to apply
    before a match is attempted: removing the punctuation a small model wraps
    its answer in, and stripping a leading "label"/"answer"/"sentiment". The
    second destroys any label that legitimately begins with one of those words,
    so `_resolve` tries this half on its own first -- `**answerable**` resolves
    here, where `normalize` hands `_canonical` the string "able" and it scores 0.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    for junk in ("*", "`", '"', "'", ".", ":"):
        text = text.replace(junk, " ")
    return " ".join(text.split())


def normalize(value) -> str:
    """Collapse a model's answer to a bare lowercase label.

    Small models decorate: '**Positive**.', 'Label: positive', 'positive\\n'.
    Scoring those as wrong would inflate the accuracy gap between big and small
    models with what is really a formatting difference, which is the opposite of
    what an honest benchmark should do.
    """
    text = _undecorated(value)
    for prefix in ("label", "sentiment", "answer", "the sentiment is", "it is"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


# Words that invert the label they precede. A refusal or a denial names the
# label it is rejecting, so a matcher that ignores them reads "not positive" as
# a vote for `positive`.
_NEGATORS = (
    "not", "no", "never", "non", "neither", "nor", "without",
    "isn't", "isnt", "wasn't", "wasnt", "aren't", "arent",
    "cannot", "can't", "cant", "don't", "dont", "doesn't", "doesnt",
    "rather than", "instead of", "other than", "unlike", "besides",
)


def _mentions(answer: str, label: str) -> bool:
    """Is `label` named in `answer` as a whole word, and not negated?

    The plain `label in answer` test this replaces scored two different kinds
    of wrong answer as correct, in the direction that flatters the models:

        labels ('yes','no'),  answer "unknown"               -> booked as 'no'
        labels (positive...), answer "the sentiment is not positive"
                                                             -> booked 'positive'

    Both inflate accuracy, and only the LLM rows reach this matcher (the
    classical and encoder rows predict by class index), so both bias exactly
    the comparison the benchmark exists to make.

    Whole-word means the character on each side is not a word character, with
    `_` counted as part of the word as `\\b` does. That is also what keeps
    nested label sets straight: `card_not_working` no longer matches inside
    `virtual_card_not_working`, so a verbose answer naming the long label
    resolves to it instead of being refused as ambiguous.
    """
    needle = label.lower()
    if not needle:
        return False
    at = answer.find(needle)
    while at >= 0:
        before = answer[at - 1] if at else ""
        end = at + len(needle)
        after = answer[end] if end < len(answer) else ""
        boundary = not ((before.isalnum() or before == "_")
                        or (after.isalnum() or after == "_"))
        if boundary and not _negated_before(answer, at):
            return True
        at = answer.find(needle, at + 1)
    return False


def _negated_before(answer: str, at: int) -> bool:
    """Does a negation word immediately precede the label at `at`?"""
    words = answer[:at].split()
    if not words:
        return False
    return words[-1] in _NEGATORS or " ".join(words[-2:]) in _NEGATORS


def _canonical(answer: str, labels: tuple[str, ...]) -> str:
    """Map a normalized answer back to the label as `labels` spells it, or "".

    Case-insensitive on BOTH sides, which is the entire point. `normalize`
    lowercases the model's answer, so testing it against the raw label strings
    silently rejects any label that is not itself lowercase. banking77 has
    exactly one such label out of 77 -- `Refund_not_showing_up` -- so a model
    that answered that intent PERFECTLY matched nothing, scored 0, and was
    booked as a parse failure. Measured on the run on disk: 4 of 251 test items
    carry that gold label and every LLM row lost exactly those 4, a flat -1.6pp.
    The classical and encoder rows predict by class index and never reach this
    matcher, so the bug biased only the LLM arm of the comparison the benchmark
    exists to make.

    Returns the CANONICAL spelling rather than the lowercased one because
    callers compare these strings back against `labels`:
    `EvalResult.per_class_accuracy` keys its report by them and `run_eval`
    stores them next to raw gold strings. Lowercasing everything would fix the
    match and break those.

    A label set containing two labels differing only in case is not resolvable
    by a case-insensitive matcher at all; here the first would win.
    """
    for label in labels:
        if answer == label.lower():
            return label
    return ""


def _resolve(raw, labels: tuple[str, ...]) -> str:
    """Match `raw` to a label, trying the UNTOUCHED string first.

    Order of operations is the whole fix. `normalize` exists to strip
    decoration a small model adds ('**Positive**.', 'Label: positive'), and it
    does that by deleting `* ` " ' . :` and by stripping a leading "label",
    "sentiment", "answer", "the sentiment is" or "it is". Applied BEFORE any
    match attempt, it mangles labels that legitimately contain those
    characters or begin with those words:

        answerable          -> able        (prefix "answer" eaten)
        sentiment_positive  -> _positive   (prefix "sentiment" eaten)
        U.S.                -> u s         (dots turned into spaces)
        LABEL_0             -> _0          (prefix "label" eaten)

    Each of those is an exact, correctly-parsed answer that scored 0. The last
    is not hypothetical: HuggingFace auto-converted datasets routinely name
    classes LABEL_0/LABEL_1, and data.py advertises that adding a dataset is a
    data change rather than a code change.

    The same collapsing also runs the other way and scores a WRONG answer 1.0,
    because two distinct labels can normalize to one string -- 'Dr.'/'Dr',
    'a.b'/'a b', 'positive'/'sentiment positive'. Matching the raw answer first
    keeps distinct labels distinct.

    So: exact match, then a UNIQUE case-insensitive match, and only then the
    normalizing path for genuinely decorated answers. Verified a no-op on both
    published runs -- every label there satisfies normalize(l) == l.lower(),
    so anything that matched before still matches by the same route.
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if text in labels:                       # exact, no normalization at all
        return text
    lowered = text.lower()
    # Unique case-insensitive match only. A label set holding two labels that
    # differ solely in case is not resolvable this way, and crediting the
    # first would score one label's answer as the other's; refusing is correct.
    ci = [l for l in labels if l.lower() == lowered]
    if len(ci) == 1:
        return ci[0]
    # Decoration stripped but no prefix eaten. This layer is what makes the
    # `answerable -> able` case in the docstring above actually hold once the
    # answer is decorated: '**answerable**' matches neither the exact nor the
    # case-insensitive test, and prefix-stripping would leave "able".
    bare = _canonical(_undecorated(text), labels)
    if bare:
        return bare
    return _canonical(normalize(text), labels)


def score_prediction(gold, pred, labels: tuple[str, ...]) -> tuple[float, str, str]:
    """Return (score, predicted_label, gold_label). Unparseable output scores 0.

    An unparseable answer is a genuine product failure, not a technicality: a
    classifier whose output you cannot read is unusable no matter how clever it
    was. It is scored 0 and reported separately, because a model that fails this
    way needs a different fix (formatting, adapter, max_tokens) than one that is
    merely inaccurate.

    Matching is case-insensitive (see `_canonical`), so "answered correctly in
    the wrong case" is no longer conflated with "returned garbage" -- but an
    answer that names no label at all still scores 0 and still counts as a parse
    failure, which is the distinction this function exists to draw.
    """
    raw_gold = getattr(gold, "label", None)
    # An unrecognised gold label falls back to its normalized form rather than
    # to "", so a dataset whose labels do not match `labels` still reports what
    # it actually asked for instead of silently reporting nothing. Gold goes
    # through the same resolver as the prediction -- it used to be normalized
    # unconditionally, so a label like `answerable` became `able` on BOTH sides
    # and the GEPA feedback string quoted a gold answer the dataset never held.
    gold_label = _resolve(raw_gold, labels) or normalize(raw_gold)

    answer = _undecorated(getattr(pred, "label", None))
    predicted = _resolve(getattr(pred, "label", None), labels)
    if not predicted:
        # Recover a label mentioned inside a longer answer before giving up.
        # `.lower()` on the label here too: this line carried the same defect,
        # so a capitalised label could never be recovered from a longer answer.
        hits = [l for l in labels if _mentions(answer, l)]
        # Label sets nest, and the plain `len(hits) == 1` test refused every
        # nested case as unparseable. banking77 has three such pairs --
        # card_not_working / virtual_card_not_working, and exchange_rate inside
        # both card_payment_wrong_exchange_rate and
        # wrong_exchange_rate_for_cash_withdrawal -- covering 16 of 251 test
        # items, so a verbose-but-exactly-right answer on any of them scored 0
        # and was booked as a parse failure. Same shape as the
        # `Refund_not_showing_up` casing bug documented in `_canonical`, and
        # biased in the same direction: only the LLM arm reaches this matcher.
        #
        # When every hit is a substring of the longest one, the answer named a
        # single label and the shorter hits are fragments of its own text, so
        # the longest is the answer. Two genuinely different labels mentioned in
        # one reply is still ambiguous and still refused.
        if len(hits) > 1:
            longest = max(hits, key=len)
            if all(h.lower() in longest.lower() for h in hits):
                hits = [longest]
        predicted = hits[0] if len(hits) == 1 else ""

    # `predicted and` is load-bearing: without it, a dataset with a missing gold
    # label would score an unparseable answer 1.0 for agreeing there is no
    # answer. An empty prediction is a failure, never a match.
    return (1.0 if predicted and predicted == gold_label else 0.0), predicted, gold_label


# ── prompt cost, in the one unit the optimizer can act on ────────────────────

# A single fixed yardstick, deliberately NOT the student model's own tokenizer.
# The penalty below is a *ratio* of two prompt lengths, and tokenizers agree on
# such a ratio to within a few percent, so one shared encoding keeps the same
# instruction worth the same penalty on every model -- which is what makes two
# models' runs comparable. Checked against the eight optimized rows on disk:
# this model's predicted per-call input-token multiplier vs the multiplier those
# runs actually measured from the providers' own counters agrees to within 5% on
# every row (2.14 vs 2.13, 2.50 vs 2.55, 2.55 vs 2.67, 2.66 vs 2.70, 2.71 vs
# 2.68, 3.09 vs 3.21, 3.10 vs 3.05), across four different tokenizers.
_REFERENCE_ENCODING = "o200k_base"


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding(_REFERENCE_ENCODING)


def count_tokens(text: str) -> int:
    """Tokens in `text` under the fixed reference encoding."""
    if not text:
        return 0
    try:
        return len(_encoder().encode(text))
    except Exception:
        # tiktoken arrives transitively (litellm hard-depends on it). If it ever
        # does not, degrade to the standard ~4-chars-per-token rule rather than
        # silently switching off a penalty the user asked for. Ratios survive
        # this substitution far better than absolute counts do.
        return max(1, round(len(text) / 4))


@dataclass(frozen=True)
class PromptCostPenalty:
    """What one accuracy point is worth, stated as a prompt-length exchange rate.

    The score handed to GEPA becomes

        accuracy  -  margin x (prompt_input_tokens(candidate) / prompt_input_tokens(seed) - 1)

    so `margin` has exactly one meaning: **the accuracy (as a 0-1 fraction) that
    a doubling of per-classification prompt cost has to buy in order to break
    even.** At `margin=0.03` a candidate that doubles the prompt bill must win
    3.0 accuracy points; one that triples it must win 6.0. `margin=0.0` -- the
    default everywhere -- makes the penalty identically zero and reproduces the
    accuracy-only objective byte for byte.

    Why this unit and not a dollars-per-accuracy-point lambda: the product
    already asks the user for `non_inferiority_margin` (default 0.03), i.e. "how
    much accuracy I will trade away before I care". Setting `margin` to that
    same number says something a buyer can actually check: *a prompt that
    doubles my bill has to do better than merely non-inferior -- it has to win
    by my whole margin.* A lambda in $/point is not a number anyone can defend,
    and a knob nobody can reason about never gets turned on.

    Why the denominator is the whole rendered prompt and not the instruction
    alone: the instruction is ~20 of ~220 input tokens per call here, so
    instruction length alone inflated 20x on the runs on disk while the bill
    inflated 2.1-3.6x. Charging the 20x would forbid every rewrite; charging the
    measured prompt ratio charges what the invoice actually does.

    Scope, stated plainly, because it bounds what the knob can promise:
      * INPUT tokens only. Output length is not a deterministic function of the
        instruction, and reading per-example usage inside the metric is the
        thread-local trap that has already zeroed a run in this codebase. So the
        penalty models the half of the bill the instruction provably controls
        and therefore UNDERSTATES total inflation on models whose output also
        grows.
      * No prompt caching. A provider that caches a long shared prefix (measured
        here: gpt-5-nano's total cost rose only 1.08x while its input tokens rose
        2.70x) is charged by this penalty as if it did not, so the knob is
        conservative -- it discourages length that provider is nearly giving
        away.
      * Search-time model, not a measurement. `evaluate.run_eval` measures the
        real bill from provider counters; this only has to rank candidates.
    """

    margin: float
    overhead_tokens: int
    baseline_instruction_tokens: int
    unavailable: str = ""     # why the penalty is inert despite being requested

    @property
    def baseline_prompt_tokens(self) -> int:
        return self.overhead_tokens + self.baseline_instruction_tokens

    @property
    def active(self) -> bool:
        return self.margin > 0 and not self.unavailable and self.baseline_prompt_tokens > 0

    def prompt_tokens(self, instruction: str) -> int:
        """Input tokens one classification sends under `instruction`."""
        return self.overhead_tokens + count_tokens(instruction)

    def ratio(self, instruction: str) -> float:
        """Per-call input-cost multiplier vs the seed instruction (seed -> 1.0)."""
        if not self.baseline_prompt_tokens:
            return 1.0
        return self.prompt_tokens(instruction) / self.baseline_prompt_tokens

    def points(self, instruction: str) -> float:
        """Score to SUBTRACT for this instruction's prompt cost.

        Positive means "deduct this much", so displays must negate it.

        Signed, with no floor: a candidate SHORTER than the seed earns a small
        bonus. That keeps the objective a single linear functional with no kink
        to reason about, and the bonus is bounded by construction -- deleting the
        instruction entirely still leaves the adapter's own scaffolding, worth
        at most `margin x instruction_tokens / prompt_tokens` (~0.3pp here).
        """
        if not self.active:
            return 0.0
        return self.margin * (self.ratio(instruction) - 1.0)

    def describe(self) -> str:
        """One-line name for the objective GEPA is actually maximizing."""
        if self.margin <= 0:
            return "val accuracy"
        if self.unavailable:
            return f"val accuracy (cost penalty requested but inert: {self.unavailable})"
        return (f"val accuracy - {self.margin:.3f} x (prompt input tokens / "
                f"{self.baseline_prompt_tokens} - 1)")

    def note(self, instruction: str) -> str:
        """The cost fact to put in front of the reflection model.

        A reflective optimizer that is *told* "this is 2.9x the baseline prompt
        for -5.6 points" can shorten the rubric; one that only sees a lower
        number cannot tell length from wrongness. This is the difference between
        a penalty GEPA can respond to and a penalty that just makes it flail.
        """
        if not self.active:
            return ""
        tokens = self.prompt_tokens(instruction)
        return (
            f"Prompt cost: this instruction is {count_tokens(instruction)} tokens, so every "
            f"classification sends {tokens} input tokens against the baseline's "
            f"{self.baseline_prompt_tokens} -- {self.ratio(instruction):.2f}x the prompt bill, "
            f"scored at {-self.points(instruction) * 100:+.1f} accuracy points "
            f"(rate: {self.margin * 100:.1f} points per doubling). Length is not free: keep only "
            f"rules that fix errors visible above, and cut anything that restates the task, "
            f"repeats another rule, or describes the output format."
        )


PROMPT_SAMPLE_SIZE = 32


def prompt_cost_penalty(program, examples, margin: float,
                        sample_size: int = PROMPT_SAMPLE_SIZE) -> PromptCostPenalty:
    """Pin a `PromptCostPenalty` to the prompt `program` actually renders.

    The overhead is measured, not guessed: ChatAdapter formats real examples and
    the instruction's own tokens are subtracted, leaving the field headers, the
    type-pinned label list, the trailing format reminder and the input text --
    218 of 238 tokens on Financial PhraseBank. Guessing it is not an option: the
    overhead is what turns a 20x instruction ratio into the ~2.7x bill ratio the
    invoice shows. (238 against 259 measured from the provider's counters is the
    tokenizer difference, not a modelling error, and it cancels in the ratio.)

    Averaged over the first `sample_size` examples rather than taken from one,
    because the quantity the bill is proportional to is the MEAN prompt length,
    and a single example is a 1-sample estimate of it that swings the penalty by
    ~15% depending on whether that example happens to be a long sentence.
    Deterministic: the split is already seeded, so the same run gets the same
    denominator.

    A `margin <= 0` returns an inert penalty without touching the adapter, so
    the default path is unchanged. If rendering fails the penalty is returned
    inert *with a reason*, which `describe()` then carries into the run record --
    a requested knob must never go quietly missing.
    """
    if margin <= 0:
        return PromptCostPenalty(margin=0.0, overhead_tokens=0,
                                 baseline_instruction_tokens=0)
    try:
        from dspy.adapters.chat_adapter import ChatAdapter

        signature = next(iter(program.named_predictors()))[1].signature
        adapter = ChatAdapter()
        sample = list(examples or [])[:max(1, sample_size)] or [None]
        rendered = []
        for example in sample:
            inputs = {name: getattr(example, name, "") or ""
                      for name in signature.input_fields}
            rendered.append(count_tokens("\n".join(
                f"{m.get('role', '')}: {m.get('content', '')}"
                for m in adapter.format(signature, [], inputs))))
        instruction_tokens = count_tokens(signature.instructions)
        overhead = round(sum(rendered) / len(rendered)) - instruction_tokens
        if overhead <= 0:
            raise ValueError(f"non-positive prompt overhead ({overhead} tokens)")
        return PromptCostPenalty(margin=float(margin), overhead_tokens=overhead,
                                 baseline_instruction_tokens=instruction_tokens)
    except Exception as exc:
        return PromptCostPenalty(margin=float(margin), overhead_tokens=0,
                                 baseline_instruction_tokens=0,
                                 unavailable=f"{type(exc).__name__}: {exc}")


def instruction_behind(pred, trace=None, pred_trace=None) -> str | None:
    """The instruction text that produced `pred`, or None if it cannot be known.

    Read off the prediction itself. `dspy.Predict` builds every result with
    `Prediction.from_completions(..., signature=signature)`, so the signature
    that rendered *this* call rides along on the object the metric is handed --
    deterministic, thread-safe, and no per-call attribution required.

    It has to come from `pred` rather than the trace because GEPA's two scoring
    paths disagree: the full-valset evaluation that produces
    `val_aggregate_scores` -- the numbers GEPA selects and reports its best
    candidate on -- runs through `dspy.Evaluate`, which calls the metric as
    `metric(example, prediction)` with no trace at all, while only the
    reflective minibatch goes through `bootstrap_trace_data` and passes one. A
    trace-only lookup would penalise the accept gate and leave the Pareto scores
    unpenalised, which is worse than not penalising anything. The trace is kept
    as a fallback for the reflection path, where `pred` can be a FailedPrediction.

    Explicitly NOT read from `lm.history[-1]`, `dspy.context` or
    `dspy.track_usage`: all three are thread-local, GEPA runs `num_threads`
    workers of its own, and that trap has already silently zeroed a whole run
    here.
    """
    instructions = getattr(getattr(getattr(pred, "_completions", None),
                                   "signature", None), "instructions", None)
    if isinstance(instructions, str):
        return instructions
    for candidate_trace in (pred_trace, trace):
        for step in reversed(list(candidate_trace or [])):
            signature = getattr(step[0], "signature", None) if len(step) else None
            instructions = getattr(signature, "instructions", None)
            if isinstance(instructions, str):
                return instructions
    return None


def classification_feedback(labels: tuple[str, ...], hint: bool = True,
                            cost: PromptCostPenalty | None = None):
    """Build a GEPA-compatible metric closure.

    DSPy validates arity at construction time -- the callable must bind exactly
    (gold, pred, trace, pred_name, pred_trace) -- so the signature below is
    fixed, not stylistic.

    `cost` is optional and inert by default. When it is active the score becomes
    `accuracy - cost.points(instruction)` in *every* path GEPA scores through,
    and `cost.note(...)` is appended to the feedback so the reflection model is
    told why. When it is inert (the default) neither the score nor a single byte
    of feedback changes.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        score, predicted, gold_label = score_prediction(gold, pred, labels)
        sentence = getattr(gold, "sentence", "")

        if score == 1.0:
            feedback = f"Correct: '{gold_label}'."
        elif not predicted:
            raw = getattr(pred, "label", None)
            feedback = (
                f"Unusable output. Expected exactly one of {list(labels)}, got {raw!r}. "
                f"The correct answer was '{gold_label}'. Answer with the single label word "
                f"and nothing else -- no explanation, no punctuation, no formatting."
            )
        else:
            detail = _CONFUSION_HINTS.get((predicted, gold_label), "") if hint else ""
            feedback = (
                f"Wrong: answered '{predicted}', correct answer is '{gold_label}'. "
                f"Sentence: \"{sentence}\". {detail}".strip()
            )

        # The penalty is one constant per candidate, so subtracting it from every
        # item leaves the aggregate at exactly `mean(accuracy) - penalty` and
        # leaves each item's Pareto comparison shifted by the same amount --
        # a shorter candidate that merely ties on an item now dominates it.
        # An instruction we cannot read is charged nothing: never invent a
        # penalty out of a measurement that failed.
        if cost is not None and cost.active:
            instruction = instruction_behind(pred, trace, pred_trace)
            if instruction is not None:
                score -= cost.points(instruction)
                feedback = f"{feedback}\n{cost.note(instruction)}"

        # GEPA's compile path passes pred_name and wants per-predictor feedback;
        # plain evaluation calls the same function with none and wants a float.
        if pred_name is None:
            return score
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


def exact_match(labels: tuple[str, ...]):
    """Plain float metric for evaluation, where feedback is unwanted overhead."""

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return score_prediction(gold, pred, labels)[0]

    return metric
