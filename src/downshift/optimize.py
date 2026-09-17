"""Running GEPA, and keeping its results honest.

GEPA evolves the *instruction text* of each predictor: it runs the program,
reads the textual feedback our metric returns on failures, and asks a strong
"reflection" model to rewrite the prompt. Nothing is fine-tuned and no weights
move -- the artifact is a better prompt for the same cheap model.

Three facts about this shape the code:

**GEPA sees the val set.** It scores every candidate on `valset` and keeps a
Pareto frontier. Val accuracy is therefore a selection statistic, not a measure-
ment, and it is optimistically biased by construction -- reporting it as "the
result" is the single easiest way to publish a number that will not replicate.
`OptimizationRun` records val and test separately and the gap between them
*is* the overfitting readout.

**The reflection model is the expensive half.** It is called with long traces
and writes long instructions, so it can cost more than the entire evaluation of
the cheap student model. `reflection_tokens` is tracked separately because the
product's pricing (pass-through plus margin) depends on it, and because a
"cheap model" story that quietly spends frontier tokens to get there deserves
to be seen.

**The metric cannot see the bill.** GEPA scores a candidate on accuracy alone,
so a prompt that fights the adapter -- and therefore gets every item silently
re-sent through JSONAdapter at double the price -- looks like a free win. The
output format is not the instruction's to specify, so it is pinned by
`program.FORMAT_CONTRACT` and re-pinned on whatever GEPA returns.

That pin closes one specific leak. It does not make the optimizer cost-aware,
and an accuracy-only metric prices prompt length at zero, so GEPA spends it:
2.1-3.6x the per-classification bill on the eight optimized rows on disk.
`cost_penalty_margin` is the opt-in that puts length in the objective (see
`metric.PromptCostPenalty` for the unit and why it is that unit). It defaults to
0.0, which reproduces the accuracy-only objective exactly.

Because the aggregate score's *meaning* changes when that knob is on, this
module records the two things separately and never lets them be confused:
`val_score_before/after` are always **val accuracy**, penalty removed, so a
penalized run's numbers stay comparable with every run already on disk;
`val_objective_before/after` are what GEPA actually maximized, and `objective`
names it in words. The "after" of each pair is indexed by `best_idx` -- the
candidate `compile()` actually returned -- rather than by `max(scores)`, which
answers a different question the moment the ranking is not by accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import dspy

from .evaluate import DEFAULT_THREADS, build_lm
from .metric import classification_feedback, count_tokens, prompt_cost_penalty
from .models import ModelSpec
from .program import (build_program, instruction_changed, optimizable_body,
                      with_format_contract)


@dataclass
class OptimizationRun:
    model_key: str
    profile: str
    budget_metric_calls: int
    baseline_instruction: str
    optimized_instruction: str
    val_score_before: float
    val_score_after: float
    wall_seconds: float
    reflection_calls: int
    reflection_in_tokens: int
    reflection_out_tokens: int
    student_calls: int
    student_in_tokens: int
    student_out_tokens: int
    error: str = ""

    # ── what the optimizer was actually asked to maximize ─────────────────
    # `val_score_before/after` above are val ACCURACY in every case, so they
    # mean the same thing whether the cost penalty was on or off and stay
    # comparable with the runs already on disk. These four say what changed.
    cost_penalty_margin: float = 0.0   # accuracy forfeited per doubling of prompt cost
    objective: str = "val accuracy"    # in words, for whoever reads the JSON
    val_objective_before: float = float("nan")
    val_objective_after: float = float("nan")

    # The deterministic input-token model the penalty ranked candidates by,
    # recorded so the penalty can be recomputed from the file. Search-time
    # figures: they exclude the format contract, which is pinned after compile
    # and so was carried by no candidate during the search.
    baseline_prompt_tokens: int = 0
    selected_prompt_tokens: int = 0
    baseline_instruction_tokens: int = 0
    selected_instruction_tokens: int = 0

    # True when re-pinning the format contract rewrote the selected candidate,
    # i.e. `val_score_after` was measured on text that is NOT what shipped.
    # This is now structurally impossible -- the pin only appends (see
    # program.py) -- and the field is kept as the assertion that says so. It
    # was not always impossible: the pin used to delete lines it read as rival
    # format directives, one candidate lost 61 of its 430 tokens that way, and
    # the shipped prompt scored 0.250 on the same val items GEPA had scored
    # 0.767 on. A 51-point regression, reported as a win.
    pin_altered_instruction: bool = False

    # WHICH prompt the val scores describe. GEPA searches and selects among
    # UNPINNED candidates; the format contract is appended afterwards and
    # nothing is re-scored. So on any run that rewrote the instruction,
    # `val_score_after` is a property of `selected_instruction`, NOT of the
    # `optimized_instruction` that actually ships -- the shipped prompt carries
    # 64 extra tokens the selection never saw. The test accuracy in results.json
    # IS measured on the shipped prompt, so the headline is sound; it is this
    # selection statistic that is mislabelled if read as "the score of the
    # prompt we ship". Named rather than silently reinterpreted, because the
    # honest fix (scoring candidates under the contract) is a change to the
    # search itself.
    val_scored_on: str = "shipped_prompt"

    @property
    def val_gain(self) -> float:
        return self.val_score_after - self.val_score_before

    @property
    def val_objective_gain(self) -> float:
        """Gain on the objective GEPA ranked by. Equals `val_gain` when off."""
        return self.val_objective_after - self.val_objective_before

    @property
    def prompt_cost_ratio(self) -> float:
        """Per-call input-cost multiplier of the selected prompt vs the seed.

        1.0 when unmeasured, so a reader never mistakes "not recorded" for
        "free". This is a model of the bill, not the bill: `EvalResult`'s
        `cost_per_1k_items` is the measured number.
        """
        if not self.baseline_prompt_tokens:
            return 1.0
        return self.selected_prompt_tokens / self.baseline_prompt_tokens

    @property
    def val_cost_penalty(self) -> float:
        """Accuracy points DEDUCTED from the selected candidate for its length.

        Positive means charged (a longer prompt than the seed), negative means
        credited, 0.0 when the penalty was off. Displays must negate it.
        """
        after, objective = self.val_score_after, self.val_objective_after
        if after != after or objective != objective:   # NaN: nothing recorded
            return 0.0
        return after - objective

    @property
    def instruction_changed(self) -> bool:
        """Did GEPA actually rewrite the instruction, ignoring the pinned format
        contract that both sides carry?

        Recorded on the run rather than left to callers, which were each doing
        `optimized.strip() != baseline.strip()`. That test now has a fixed
        trailer on both sides, and "GEPA rewrote the prompt" vs "no candidate
        beat the baseline" is exactly the distinction the chart has already been
        caught getting wrong once, so it belongs in one place.
        """
        return instruction_changed(self.baseline_instruction, self.optimized_instruction)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["val_gain"] = self.val_gain
        d["val_objective_gain"] = self.val_objective_gain
        d["prompt_cost_ratio"] = self.prompt_cost_ratio
        d["val_cost_penalty"] = self.val_cost_penalty
        d["instruction_changed"] = self.instruction_changed
        return d


def _usage_delta(lm, start: int) -> tuple[int, int, int]:
    # The optimizer makes far more calls than one eval -- hundreds of rollouts
    # per compile -- so this is the site most likely to hit the eviction cap.
    from .evaluate import assert_ledger_intact
    assert_ledger_intact(lm, start, "GEPA usage accounting")
    calls = lm.history[start:]
    return (len(calls),
            sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in calls),
            sum((c.get("usage") or {}).get("completion_tokens", 0) for c in calls))


def _candidate_instruction(details, index: int) -> str:
    """The instruction of GEPA's `index`-th candidate, or "" if unavailable.

    `DspyGEPAResult.candidates` holds built programs, so this reads the same
    text the metric saw while scoring that candidate. Used to invert the cost
    penalty out of a recorded aggregate score.
    """
    try:
        candidate = list(getattr(details, "candidates", None) or [])[index]
        return next(iter(candidate.named_predictors()))[1].signature.instructions
    except Exception:
        return ""


def run_gepa(spec: ModelSpec, task, profile: str, reflection_spec: ModelSpec,
             max_metric_calls: int = 240, num_threads: int = DEFAULT_THREADS,
             max_tokens: int = 1024, reflection_max_tokens: int = 8000,
             hint: bool = True, seed: int = 0,
             cost_penalty_margin: float = 0.0) -> tuple[object, OptimizationRun]:
    """Optimize `profile`'s instruction for `spec` and report what it cost.

    Returns (optimized_program, run_record). On failure the original program is
    returned with `error` set -- one model failing to optimize should cost us a
    row on the chart, not the whole run.

    `cost_penalty_margin` is the accuracy (0-1 fraction) a *doubling* of
    per-classification prompt cost must buy to break even; 0.0, the default,
    leaves the objective accuracy-only and every byte of behaviour unchanged.
    Setting it to the user's `non_inferiority_margin` states "a prompt that
    doubles my bill must beat the baseline by my whole margin, not merely tie
    it". See `metric.PromptCostPenalty` for the unit's exact scope.
    """
    student_lm = build_lm(spec, max_tokens=max_tokens)
    # The reflection model must be free to write a long instruction, and if it is
    # a reasoning model its <think> block is drawn from the same budget.
    reflection_lm = build_lm(reflection_spec, max_tokens=reflection_max_tokens)

    # Seed the search from the TASK's own instruction, not the module default.
    # Without this, GEPA on a non-sentiment task starts from "classify the
    # sentiment of a financial news sentence" and burns its budget climbing out
    # of a prompt that describes the wrong problem entirely.
    task_spec = getattr(task, "spec", None)
    description = getattr(task_spec, "description", None)
    # Pass the spec so a generative task reaches build_generative_signature
    # instead of silently being compiled as a Literal[...] classification over
    # every answer in the dataset. (load_task refuses generative tasks today,
    # so this is belt-and-braces until the loader is wired.)
    program = build_program(profile, task.labels, description, task_spec)

    # Measured against the prompt this program actually renders on the split
    # GEPA scores -- the adapter scaffolding plus the input sentence is 218 of
    # 238 input tokens here, and it is precisely that denominator that turns a
    # 20x instruction ratio into the ~2.7x bill ratio the invoice shows.
    cost = prompt_cost_penalty(program, task.val or task.train or task.test,
                               cost_penalty_margin)
    metric = classification_feedback(task.labels, hint=hint, cost=cost)

    s_start, r_start = len(student_lm.history), len(reflection_lm.history)
    t0 = time.time()

    # GEPA spawns its own worker threads, and dspy.context is thread-local, so
    # the student LM has to be set globally here rather than scoped -- the same
    # trap that silently zeroed the evaluator. Global means it has to be put
    # back: without the restore, whichever student was optimized last stayed
    # the process-wide default LM after run_gepa returned, on the exception
    # path too, so any later call that did not open its own dspy.context ran
    # against the wrong model and was billed to the wrong row.
    previous_lm = getattr(dspy.settings, "lm", None)
    dspy.configure(lm=student_lm)

    baseline_instruction = next(iter(program.named_predictors()))[1].signature.instructions
    error = ""
    optimized = program

    try:
        gepa = dspy.GEPA(
            metric=metric,
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            num_threads=num_threads,
            track_stats=True,
            seed=seed,
        )
        optimized = gepa.compile(program, trainset=task.train, valset=task.val)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # Nothing below this point calls the LM -- the val bookkeeping reads
        # GEPA's own records -- so restoring here is safe and unconditional.
        dspy.configure(lm=previous_lm)

    wall = time.time() - t0

    # Re-pin the output contract on whatever came back. GEPA is free to rewrite
    # the instruction but not the output format: the reflection model reads a
    # trace, decides the format is part of the problem, and writes its own --
    # on banking77 it appended `Return a JSON object with exactly: {"label":
    # ...}`, which ChatAdapter cannot parse, so every item took the JSONAdapter
    # retry and was billed twice. The metric never sees the retry, so that
    # candidate scores as a free win and the optimizer will keep finding it.
    # `with_format_contract` is idempotent, so a candidate that left the
    # contract alone is returned byte-identical and no fake diff appears.
    #
    # Read the selected instruction BEFORE pinning: that is the text GEPA
    # scored, so it is the text whose token count explains the objective. The
    # contract is a fixed trailer no candidate carried during the search, so
    # including it here would report a ratio the optimizer never saw.
    selected_instruction = next(iter(optimized.named_predictors()))[1].signature.instructions

    # Pin only what the optimizer actually rewrote. A run where no candidate
    # beat the baseline ships the seed instruction, which is ours, carries no
    # format directive, and was measured at exactly 1.00 billed calls per item
    # without the contract -- so pinning it would buy nothing and cost the
    # contract's 64 input tokens on every call, which is 1.19x the stock prompt
    # for a row whose whole finding is "GEPA found no improvement". The risk
    # this guards against is created by the rewrite, so it is priced onto the
    # rewrite.
    rewrote = instruction_changed(baseline_instruction, selected_instruction)
    if rewrote:
        for _, predictor in optimized.named_predictors():
            pinned = with_format_contract(predictor.signature.instructions)
            if pinned != predictor.signature.instructions:
                predictor.signature = predictor.signature.with_instructions(pinned)

    optimized_instruction = next(iter(optimized.named_predictors()))[1].signature.instructions

    # Val scores come from GEPA's own bookkeeping when available. They are
    # selection statistics on a set GEPA optimized against -- never the headline.
    #
    # Two numbers, kept apart on purpose. `val_objective_*` is the aggregate
    # GEPA ranked by; `val_score_*` is val accuracy with the cost penalty taken
    # back out, so it keeps one meaning across penalized and unpenalized runs.
    # The inversion is exact: the penalty is one constant per candidate, so the
    # aggregate is `mean(accuracy) - penalty` and adding the penalty back
    # recovers the accuracy. (An example that *raised* rather than answered is
    # scored by GEPA's own `failure_score=0.0` and so never had the penalty
    # applied; each such item leaves the recovered accuracy high by
    # penalty/n_val. Errors are counted in the eval rows and are rare, and this
    # is a selection statistic either way -- the headline is the test split.)
    #
    # Indexed by `best_idx`, which is the candidate `compile()` returned, not
    # `max(scores)`. They coincide while the ranking is by accuracy, and they
    # answer different questions the moment it is not.
    val_before = val_after = float("nan")
    obj_before = obj_after = float("nan")
    details = getattr(optimized, "detailed_results", None)
    if details is not None:
        scores = [float(s) for s in (getattr(details, "val_aggregate_scores", None) or [])]
        if scores:
            best = getattr(details, "best_idx", None)
            if not isinstance(best, int) or not 0 <= best < len(scores):
                best = max(range(len(scores)), key=scores.__getitem__)
            obj_before, obj_after = scores[0], scores[best]
            val_before = obj_before + cost.points(_candidate_instruction(details, 0))
            val_after = obj_after + cost.points(_candidate_instruction(details, best))

    s_calls, s_in, s_out = _usage_delta(student_lm, s_start)
    r_calls, r_in, r_out = _usage_delta(reflection_lm, r_start)

    return optimized, OptimizationRun(
        model_key=spec.key, profile=profile, budget_metric_calls=max_metric_calls,
        baseline_instruction=baseline_instruction,
        optimized_instruction=optimized_instruction,
        val_score_before=val_before, val_score_after=val_after,
        wall_seconds=wall,
        reflection_calls=r_calls, reflection_in_tokens=r_in, reflection_out_tokens=r_out,
        student_calls=s_calls, student_in_tokens=s_in, student_out_tokens=s_out,
        error=error,
        cost_penalty_margin=float(cost_penalty_margin),
        objective=cost.describe(),
        val_objective_before=obj_before, val_objective_after=obj_after,
        baseline_prompt_tokens=cost.baseline_prompt_tokens,
        selected_prompt_tokens=cost.prompt_tokens(selected_instruction) if cost.active else 0,
        baseline_instruction_tokens=count_tokens(baseline_instruction),
        selected_instruction_tokens=count_tokens(selected_instruction),
        # The pin always APPENDS the contract; it "altered" the candidate only
        # if it also dropped lines, which is what makes the scored text differ
        # from the shipped text. Asked of the optimizable BODY on both sides:
        # the old test was `optimizable_body(selected) != selected.strip()`,
        # which fires whenever the candidate merely CONTAINS the contract even
        # though `with_format_contract` is idempotent and changed nothing -- a
        # false positive on the exact assertion this field exists to make, and
        # experiment.py prints a "the format pin rewrote this candidate after
        # GEPA scored it, trust the test row not the val number" warning off it.
        # Asked WITHOUT going through `optimizable_body`, which is what made the
        # previous spelling vacuous: `optimized` is `with_format_contract(selected)`
        # and that is `optimizable_body(selected) + contract`, so comparing the two
        # stripped bodies is comparing a value with itself -- False by construction
        # on every run, for a field whose whole job is to notice the day it is not.
        # Substring instead: if the selected text still appears verbatim in what
        # ships, the pin only appended. If it does not, the pin removed something,
        # which is the 51-point regression above.
        pin_altered_instruction=(
            selected_instruction.strip() not in optimized_instruction),
        # "unpinned_candidate" whenever the pin actually changed the text that
        # ships, i.e. exactly when GEPA rewrote the instruction. Note this is
        # true even though `pin_altered_instruction` is False: that flag asks
        # the narrower question "did the pin DELETE anything", which it no
        # longer can. Appending is still a change to the shipped prompt.
        val_scored_on=("unpinned_candidate"
                       if optimized_instruction != selected_instruction
                       else "shipped_prompt"),
    )
