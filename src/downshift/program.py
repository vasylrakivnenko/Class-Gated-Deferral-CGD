"""The DSPy program under optimization, and the two cost profiles it runs in.

The unit of comparison in this product is not a model, it is a (model, program,
prompt) triple. Saying "Qwen3-1.7B scores 72%" is meaningless without saying
which of these it ran as -- the same weights swing double-digit accuracy and
~10x output cost depending on the wrapper.

We hold the program fixed at two profiles and let GEPA optimize the instruction
inside each:

  DIRECT    dspy.Predict, reasoning off. Emits the label and stops -- roughly
            35 output tokens. This is the configuration a cost-cut actually
            ships on.
  REASONING dspy.ChainOfThought, reasoning on. Measured locally at ~260 output
            tokens for the same call: 7x the billed output for one word of
            answer. Included precisely so the chart can show what it costs.

Keeping both on the chart is the honest move. A "cheap model" that thinks for
300 tokens before answering is frequently *more* expensive per task than a
pricier model that answers directly, and a cost chart that plots only $/token
hides that completely.
"""

from __future__ import annotations

from typing import Literal

import dspy

# The output *format* is the adapter's business, not the instruction's: DSPy
# derives it from the signature's fields and restates it in every request. An
# instruction that asks for a different shape cannot change how the reply is
# parsed -- it only makes the reply unparseable, and ChatAdapter then silently
# re-sends the item through JSONAdapter, so one classification is billed twice.
# GEPA cannot see that cost: its metric reads the final answer, which the retry
# still produces. Measured on banking77, a reflection-written trailer
# (`Return a JSON object with exactly: {"label": ...}`) cost 500 billed calls
# for 251 items and scored as a win. So this contract is pinned to whatever the
# optimizer writes (optimize.py re-pins after every compile) instead of being
# left for the optimizer to respect.
#
# It is deliberately NOT pinned to the stock instructions below. Those are ours,
# contain no format directive, and measured at exactly 1.00 billed calls per
# item with and without it -- so pinning there would add ~66 input tokens to
# every call for no effect, and would make every row measured before this
# existed non-comparable with every row measured after.
#
# The text names ChatAdapter's field markers because that is what the program
# actually runs under -- nothing in this codebase configures an adapter, and
# `dspy.settings.adapter is None` means ChatAdapter. Wording is envelope-scoped
# ("do not wrap"), not content-scoped, so a generative task whose answer is
# itself a JSON document is still expressible.
FORMAT_CONTRACT = (
    "Response format is fixed by this prompt's field structure, not by the "
    "task: put each output field under its own `[[ ## field ## ]]` header as a "
    "plain value, then emit `[[ ## completed ## ]]`. Never wrap it in JSON, "
    "markdown or a code fence; never add or rename a field."
)

# The contract is APPENDED and nothing is deleted. An earlier version also
# stripped lines that looked like rival format directives, and that was a worse
# bug than the one it fixed: stripping runs *after* GEPA has scored a candidate,
# so the shipped prompt was no longer the prompt that earned the score. One run
# lost 61 of 430 tokens that way and scored 0.250 on the same val items GEPA had
# scored 0.767 on -- a 51-point silent regression, reported as a win.
#
# Deleting was never needed. Measured on banking77 gpt-5.4-nano: leaving the
# rogue `Return a JSON object ...` line in place and merely appending this
# contract after it still gave exactly 1.000 billed calls per item, identical to
# deleting it. It is the last word the model obeys. So the pin is purely
# additive, which makes it incapable of changing what was scored, and the only
# residue is a few tokens of text the contract overrides.
def optimizable_body(instruction: str) -> str:
    """The part of an instruction the optimizer owns.

    Change-detection must compare *this*, not the raw instruction: the contract
    is re-appended after optimization, and a fixed trailer carried by both sides
    is not a rewrite.
    """
    return instruction.replace(FORMAT_CONTRACT, "").strip()


def with_format_contract(instruction: str) -> str:
    """Pin the format contract to the end of `instruction`.

    Idempotent by construction -- an instruction that already ends in the
    contract comes back byte-identical -- which is what lets `run_gepa`
    re-pin unconditionally without inventing a diff.
    """
    body = optimizable_body(instruction)
    return f"{body}\n\n{FORMAT_CONTRACT}" if body else FORMAT_CONTRACT


def instruction_changed(baseline: str, optimized: str) -> bool:
    """Did the optimizer actually rewrite the instruction?"""
    return optimizable_body(baseline) != optimizable_body(optimized)


def record_prompt_changed(opt: dict | None) -> bool | None:
    """Same question, asked of a serialized optimization record.

    `None` means there is nothing to ask -- no record, or a record whose run
    errored, so the instruction on file is not the product of an optimization.
    Every script that re-charts a finished run needs this, and the scores
    cannot answer it: re-scoring one unchanged prompt drifts on cache warmth
    and a handful of flipped items, and reading that drift as an improvement is
    exactly the credit-for-noise this exists to prevent.
    """
    if not opt or opt.get("error"):
        return None
    if opt.get("instruction_changed") is not None:  # written by newer runs
        return bool(opt["instruction_changed"])
    return instruction_changed(opt.get("baseline_instruction", ""),
                               opt.get("optimized_instruction", ""))


def build_generative_signature(task_description: str, output_desc: str,
                               input_desc: str = "The input to process."):
    """Signature for a task whose answer is not one of a fixed label set.

    The output field is a plain `str`, not `Literal[...]`. That is the whole
    point of a generative task: there is no enumerable answer space to
    constrain to, which is also precisely why a bag-of-words classifier cannot
    compete on one -- it has no way to emit an answer it has never seen as a
    class.
    """
    return dspy.Signature(
        {
            "sentence": (str, dspy.InputField(desc=input_desc)),
            "label": (str, dspy.OutputField(desc=output_desc)),
        },
        task_description,
    )


def build_signature(labels: tuple[str, ...], task_description: str | None = None):
    """Construct a classification signature with the label set pinned by type.

    Built through DSPy's programmatic `Signature(fields, instructions)` form
    rather than a `class Classify(dspy.Signature)` body. That is not stylistic:
    this module uses `from __future__ import annotations`, which turns every
    class-body annotation into a string, so `label: Literal[labels]` reaches
    pydantic as the unresolvable ForwardRef `'Literal[labels]'`. It survives the
    ChatAdapter path and then explodes the moment DSPy falls back to the
    JSONAdapter -- i.e. intermittently, mid-run. Passing the real type object
    sidesteps the whole problem and keeps label sets data-driven.

    `Literal[...]` earns its place: DSPy renders the allowed values into the
    prompt and parses against them, so a small model is much less likely to
    answer "Positive." or "the sentiment is positive". A naive string-equality
    metric scores those as wrong, converting a formatting quirk into a fake
    accuracy gap -- which would penalise precisely the small models we are
    trying to give a fair hearing.
    """
    # NOT wrapped in with_format_contract(): a stock instruction is written by
    # us, carries no format directive, and measured at exactly 1.00 billed calls
    # per item with and without the contract -- so pinning here would add ~66
    # input tokens to every call for no effect, and make rows measured before
    # the contract existed non-comparable with rows measured after. The pin is
    # applied where the risk actually is: to whatever GEPA writes (optimize.py).
    instructions = task_description or (
        "Classify the sentiment of a financial news sentence from the point of "
        "view of an investor reading it."
    )
    return dspy.Signature(
        {
            "sentence": (str, dspy.InputField(desc="A sentence from financial news.")),
            "label": (
                Literal[tuple(labels)],  # type: ignore[valid-type]
                dspy.OutputField(desc=f"The sentiment. Exactly one of: {', '.join(labels)}."),
            ),
        },
        instructions,
    )


class DirectClassifier(dspy.Module):
    """Single Predict call. The cheap path -- no reasoning tokens."""

    def __init__(self, labels: tuple[str, ...], task_description: str | None = None):
        super().__init__()
        self.classify = dspy.Predict(build_signature(labels, task_description))

    def forward(self, sentence: str):
        return self.classify(sentence=sentence)


class ReasoningClassifier(dspy.Module):
    """ChainOfThought. Usually more accurate, always more expensive."""

    def __init__(self, labels: tuple[str, ...], task_description: str | None = None):
        super().__init__()
        self.classify = dspy.ChainOfThought(build_signature(labels, task_description))

    def forward(self, sentence: str):
        return self.classify(sentence=sentence)


PROFILES = {
    "direct": DirectClassifier,
    "reasoning": ReasoningClassifier,
}


def build_program(profile: str, labels: tuple[str, ...], task_description: str | None = None,
                  spec=None):
    """Build the module for a task.

    `spec` (a TaskSpec) selects the generative path when present and marked
    generative; without it the classification path is used, which keeps every
    existing caller working unchanged.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    if spec is not None and getattr(spec, "is_generative", False):
        sig = build_generative_signature(
            task_description or spec.description,
            spec.output_desc or "The answer.",
        )
        mod = dspy.ChainOfThought(sig) if profile == "reasoning" else dspy.Predict(sig)
        return _Wrapped(mod)
    return PROFILES[profile](labels, task_description)


class _Wrapped(dspy.Module):
    """Adapts a bare Predict/ChainOfThought to the (sentence=...) call shape
    the evaluator uses for every task."""

    def __init__(self, inner):
        super().__init__()
        self.classify = inner

    def forward(self, sentence: str):
        return self.classify(sentence=sentence)
