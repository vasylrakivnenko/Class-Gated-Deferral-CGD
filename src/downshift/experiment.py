"""End-to-end run: load, optimize, evaluate on held-out test, compare, chart.

The order of operations here *is* the methodology, so it is worth stating:

1. Split once, up front. Test is carved out first and no optimizer ever sees it.
2. Score every candidate's *stock* prompt on test. This is the "before".
3. Optimize selected candidates with GEPA on train+val only. By default the
   objective is accuracy alone, which is how every result on disk was measured;
   `cost_penalty_margin` adds prompt cost to it (see `metric.PromptCostPenalty`).
   Whichever objective ran, the recorded `val_score_*` are val ACCURACY, so a
   penalized run's numbers stay comparable with an unpenalized one's.
4. Re-score on the same test set. This is the "after".
5. Compare against the reference model with a paired test (McNemar) and a
   non-inferiority test against the user's margin -- not independent intervals,
   which are the wrong tool for two models scored on identical items.
6. Report what could not be distinguished, as prominently as what could.

Step 6 is the one most benchmark posts skip. With a 250-item test set the 95%
interval is about +/-4.5 points, so a good half of the comparisons anyone wants
to make are simply not resolvable at this sample size. Saying so is the product.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import encoders, stats
from .chart import ChartRow, cost_footnote, plot_cost_vs_accuracy
from .data import get_task, load_task, split_fingerprint
from .evaluate import EvalResult, run_eval
from .models import BY_KEY, LOCAL, OPEN, ModelSpec, missing_keys
from .optimize import OptimizationRun, run_gepa
from .program import build_program


@dataclass
class ExperimentConfig:
    candidates: list[str] = field(default_factory=lambda: [
        "qwen3-0.6b-direct", "qwen3-1.7b-direct", "qwen3-1.7b-reasoning", "qwen3-4b-direct",
    ])
    optimize: list[str] = field(default_factory=lambda: [
        "qwen3-0.6b-direct", "qwen3-1.7b-direct",
    ])
    reference_model: str = ""          # the "expensive" model we compare against
    reflection_model: str = "gpt-oss-20b-direct"
    run_encoders: bool = True
    finetune_encoder: bool = True
    accuracy_bar: float | None = None  # None -> reference accuracy minus margin
    non_inferiority_margin: float = 0.03
    # Accuracy (0-1 fraction) that a DOUBLING of per-classification prompt cost
    # must buy for GEPA to consider a longer prompt worth it. 0.0 -- the default
    # -- leaves the optimizer accuracy-only, exactly as every run on disk was
    # measured. Setting it to `non_inferiority_margin` says "a prompt that
    # doubles my bill has to beat the baseline by my whole margin, not merely
    # tie it", which is a claim the same user already made about accuracy.
    # Units and scope: metric.PromptCostPenalty.
    cost_penalty_margin: float = 0.0
    max_metric_calls: int = 400
    n_train: int = 200
    n_val: int = 200
    n_test: int = 250
    max_tokens: int = 1024
    reasoning_max_tokens: int = 2500
    # GEPA asks the reflection model for a whole rewritten instruction, so its
    # budget is an order of magnitude above a label's. Named here rather than
    # left as a run_gepa default so the capability screen checks the number the
    # run will actually request -- a screen against a different budget is worse
    # than none, because it reads as having checked.
    reflection_max_tokens: int = 8000
    num_threads: int = 12
    seed: int = 0
    task: str = "financial_phrasebank"   # key into data.TASKS
    out_dir: str = "runs"


def _profile_for(key: str) -> str:
    return "reasoning" if key.endswith("-reasoning") else "direct"


def _budget_for(cfg: "ExperimentConfig", key: str, spec: ModelSpec) -> int:
    """The `max_tokens` this candidate will actually be asked for.

    Must mirror the two call sites that build the LM, because the gate is only
    honest if it screens on the number that will really be requested. A budget
    is not a demand -- the bill follows the tokens emitted -- but the endpoint
    rejects or truncates against the budget, so that is what has to fit.
    """
    profile = _profile_for(key)
    return cfg.reasoning_max_tokens if profile == "reasoning" or spec.think else cfg.max_tokens


def screen_candidates(cfg: "ExperimentConfig", task) -> dict:
    """Refuse impossible candidates before a single paid call is made.

    This is the DEDUCTIVE stage, and its value is bounded by arithmetic rather
    than by judgement: every exclusion here is a count that cannot be argued
    with, and nothing here predicts whether a model is any good.

    Honesty about what it currently buys: on both tasks in the library it
    rejects NOTHING. They need 12 and 21 output tokens against a registry whose
    smallest cap is 4,096, and ~2k of context against a 128,000-token floor.
    That is a fact about the task library having no variance yet, not a win, and
    the printed summary says so rather than implying a filter did work. It
    starts to bite on the first long-input or strict-output task, and it already
    separates the five Fireworks rows -- whose limits nobody has recorded -- from
    the rows that were actually checked.

    Three buckets, never two. `unverified` passes with the uncertainty named,
    because "verified to fit" and "nobody checked" must not read the same in a
    published claim.
    """
    demands = task.demands()
    runnable, rejected, unverified, unavailable = [], [], [], []
    for key in cfg.candidates:
        spec = BY_KEY.get(key)
        if spec is None:
            rejected.append((key, f"{key} is not in the registry"))
            continue
        if not spec.available:
            unavailable.append((key, f"{spec.label}: no API key or endpoint configured"))
            continue
        fine, why = spec.fits(_budget_for(cfg, key, spec), demands.p95_prompt_tokens)
        if not fine:
            rejected.append((key, why))
        else:
            runnable.append(key)
            if why:
                unverified.append((key, why))

    # The reflection model is screened too, and on a bigger output budget: GEPA
    # asks it for a whole rewritten instruction, not a label. It used to be
    # built unguarded at the top of `run_gepa`, so an undersized one raised
    # AFTER every paid baseline eval had already been billed. Its PROMPT length
    # is deliberately not screened -- GEPA composes that internally from
    # examples and feedback and we never see it -- so this checks the output cap
    # only and `fits` names the context window as unchecked.
    reflection_ok, reflection_note = True, ""
    reflection = BY_KEY.get(cfg.reflection_model)
    if cfg.optimize:
        if reflection is None:
            reflection_ok, reflection_note = False, (
                f"reflection_model {cfg.reflection_model!r} is not in the registry")
        elif not reflection.available:
            reflection_ok, reflection_note = False, (
                f"{reflection.label}: no API key or endpoint configured")
        else:
            reflection_ok, reflection_note = reflection.fits(cfg.reflection_max_tokens)

    return {"demands": demands, "runnable": runnable, "rejected": rejected,
            "unverified": unverified, "unavailable": unavailable,
            "reflection_ok": reflection_ok, "reflection_note": reflection_note}


def print_screen(screen: dict) -> None:
    """Say what the screen did, including when the answer is 'nothing'."""
    d = screen["demands"]
    print(f"\nCapability screen (before spending): {d}")
    n_ok, n_no = len(screen["runnable"]), len(screen["rejected"])
    if n_no:
        print(f"  {n_no} candidate(s) CANNOT serve this task and will not be called:")
        for key, why in screen["rejected"]:
            print(f"    - {key}: {why}")
    else:
        print(f"  0 of {n_ok + n_no} candidates excluded by capability -- this task fits "
              f"every deployment on file, so the screen bought nothing here.")
    for key, why in screen["unverified"]:
        print(f"  ? {key}: passes UNVERIFIED -- {why}")
    for key, why in screen["unavailable"]:
        print(f"  - {key}: {why}")
    if not screen["reflection_ok"]:
        print(f"  ! reflection model unusable: {screen['reflection_note']}")
    elif screen["reflection_note"]:
        print(f"  ? reflection model passes UNVERIFIED -- {screen['reflection_note']}")


# Keyed by result key, not by ModelSpec: the encoder rows are produced by
# encoders.py and never appear in the model registry, so a registry lookup
# returns None and silently paints them as prompted LLMs on the chart.
_CLASSICAL = {"majority", "tfidf-logreg", "frozen-embed"}
_ENCODER = {"finetuned-encoder", "modernbert"}

# Rows that always get a chart label regardless of how crowded the plot is:
# the honesty-floor baselines, the fine-tuned-encoder upsell, and the
# frontier reference. Every prompted LLM's stock/optimized pair is labeled
# separately by _label_text() via its GEPA link, so it does not need to be
# listed here too.
ALWAYS_LABEL_KEYS = _CLASSICAL | _ENCODER


def _prompt_changed(record: OptimizationRun) -> bool:
    """Whether GEPA actually rewrote the instruction, by comparing the text.

    The scored numbers cannot answer this. A same-prompt re-score moves both
    accuracy and cost on its own -- cache warmth shifts the price, and at
    temperature 0 a handful of items still flip -- so a chart that reads
    "the numbers moved" as "GEPA improved it" credits the optimizer with
    measurement noise. banking77 deepseek-v4-flash did exactly that: a
    byte-identical 111-char instruction charted as a 1.6pp gain.

    Delegates to the record, which compares the *optimizable body* -- the fixed
    format contract is appended to both sides, so a raw string compare happens
    to agree today but would report a change the moment the two sides carry
    different trailers.
    """
    return record.instruction_changed


def _family_for(key: str) -> str:
    if key in _CLASSICAL:
        return "Classical baseline"
    if key in _ENCODER:
        return "Trained encoder"
    return "Prompted LLM"


def validate_config(cfg) -> None:
    """Reject a configuration whose acceptance decisions cannot be computed.

    Every PASS/FAIL on the chart is a PAIRED test against `reference_model`, so
    if that model is not itself evaluated there is nothing to pair against.
    run_demo.py named claude-opus-4-8 as its reference and never listed it in
    `candidates`; run_experiment then did `results.get(ref_key)` -> None and
    skipped the entire McNemar / non-inferiority block. No error, no warning,
    no comparisons -- the documented quick start silently produced a chart with
    no acceptance decisions in it at all. Failing here costs a second; failing
    silently costs the whole point of the run.
    """
    if cfg.reference_model not in cfg.candidates:
        raise ValueError(
            f"reference_model {cfg.reference_model!r} is not in candidates, so it will "
            f"never be evaluated and every paired comparison would be skipped. Add it to "
            f"candidates, or pick a reference from: {sorted(cfg.candidates)}")
    unknown = [k for k in (*cfg.candidates, *cfg.optimize) if k not in BY_KEY]
    if unknown:
        raise ValueError(f"unknown model keys in config: {unknown}")


def cost_basis_for(key: str, cost_per_1k: float) -> str:
    """Which of the chart's three cost conventions this row's price is.

    Split out here because five separate scripts build ChartRows and each one
    used to hard-code a caption claiming every local row was free. The rule:
    a row costing nothing is "free"; a row that RUNS locally but carries a
    nonzero number is a "proxy" estimate off somebody else's rate card, not a
    bill; everything else is a measured hosted bill. See chart.cost_footnote.
    """
    from .models import BY_KEY, LOCAL
    if cost_per_1k is None or cost_per_1k <= 1e-9:
        return "free"
    spec = BY_KEY.get(key.replace("+gepa", ""))
    if spec is not None and spec.runtime == LOCAL:
        return "proxy"
    return "measured"


def run_experiment(cfg: ExperimentConfig) -> dict:
    validate_config(cfg)
    # Per-task directory: datasets are ADDED, not swapped, and a result only
    # means anything beside the task it was measured on. Writing them all to
    # one results.json would silently overwrite the previous dataset's run.
    out_dir = Path(cfg.out_dir) / cfg.task
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(f"Loading task (train={cfg.n_train} val={cfg.n_val} test={cfg.n_test}, seed={cfg.seed})")
    task_spec = get_task(cfg.task)
    task = load_task(task_spec, cfg.n_train, cfg.n_val, cfg.n_test, cfg.seed)
    maj_label, maj_acc = task.majority_baseline
    print(f"  {task}")
    print(f"  majority class '{maj_label}' = {maj_acc:.1%}  |  {task.n_duplicates_dropped} duplicates dropped")
    print(f"  {stats.resolution_warning(cfg.n_test)}")

    locked = missing_keys()
    if locked:
        print("  NOTE: hosted models skipped for want of a key: "
              + "; ".join(f"{k} ({len(v)})" for k, v in locked.items()))

    # Deductive screen BEFORE any spending. Runs ahead of the encoder block so
    # that a run which cannot call a single LLM says so in the first seconds
    # rather than after a GPU fine-tune.
    screen = screen_candidates(cfg, task)
    print_screen(screen)
    if not screen["runnable"]:
        raise SystemExit(
            "No candidate can serve this task. Nothing was spent.\n  "
            + "\n  ".join(f"{k}: {w}" for k, w in
                          screen["rejected"] + screen["unavailable"]))

    results: dict[str, EvalResult] = {}
    optimizations: dict[str, OptimizationRun] = {}

    # ── non-generative baselines ──────────────────────────────────────────
    if cfg.run_encoders:
        print("\nBaselines (no LLM):")
        encoder_train = task.train + task.pool
        results["majority"] = encoders.run_majority(encoder_train, task.test)
        print("  " + results["majority"].summary())
        results["tfidf-logreg"] = encoders.run_tfidf(encoder_train, task.test, cfg.seed)
        print("  " + results["tfidf-logreg"].summary())
        try:
            results["frozen-embed"] = encoders.run_frozen_embeddings(encoder_train, task.test, seed=cfg.seed)
            print("  " + results["frozen-embed"].summary())
        except Exception as exc:
            print(f"  frozen-embed skipped: {type(exc).__name__}: {str(exc)[:120]}")
        if cfg.finetune_encoder:
            try:
                results["finetuned-encoder"] = encoders.run_finetuned_encoder(
                    encoder_train, task.test, seed=cfg.seed)
                print("  " + results["finetuned-encoder"].summary())
            except Exception as exc:
                print(f"  finetuned-encoder skipped: {type(exc).__name__}: {str(exc)[:120]}")

    # ── prompted LLMs, stock prompt ───────────────────────────────────────
    print("\nPrompted LLMs (stock prompt, on held-out test):")
    # `screen["runnable"]` rather than `cfg.candidates`: the screen already
    # dropped what cannot work and said why, so a second silent skip here would
    # hide the reason. It preserves cfg order.
    for key in screen["runnable"]:
        spec = BY_KEY[key]
        profile = _profile_for(key)
        mt = _budget_for(cfg, key, spec)
        try:
            results[key] = run_eval(spec, build_program(profile, task.labels, task_spec.description, task_spec), task.test,
                                    task.labels, num_threads=cfg.num_threads, max_tokens=mt)
        except Exception as exc:
            print(f"  {key} FAILED: {type(exc).__name__}: {str(exc)[:140]}")

    # ── GEPA ──────────────────────────────────────────────────────────────
    reflection = BY_KEY.get(cfg.reflection_model)
    if cfg.optimize and reflection is not None and reflection.available:
        print(f"\nGEPA optimization (reflection model: {reflection.label}, "
              f"budget {cfg.max_metric_calls} metric calls):")
        if cfg.cost_penalty_margin > 0:
            print(f"  objective: val accuracy MINUS {cfg.cost_penalty_margin:.1%} "
                  f"per doubling of per-classification prompt cost")
        for key in cfg.optimize:
            spec = BY_KEY.get(key)
            if spec is None or not spec.available:
                continue
            profile = _profile_for(key)
            mt = cfg.reasoning_max_tokens if profile == "reasoning" or spec.think else cfg.max_tokens
            print(f"  optimizing {spec.label} ...", flush=True)
            optimized, record = run_gepa(spec, task, profile, reflection,
                                         max_metric_calls=cfg.max_metric_calls,
                                         num_threads=cfg.num_threads, max_tokens=mt,
                                         reflection_max_tokens=cfg.reflection_max_tokens,
                                         seed=cfg.seed,
                                         cost_penalty_margin=cfg.cost_penalty_margin)
            optimizations[key] = record
            if record.error:
                print(f"    GEPA failed: {record.error[:160]}")
                continue
            changed = _prompt_changed(record)
            # "val acc" is spelled out because it is NOT what GEPA ranked by
            # once the cost penalty is on; the penalized objective is printed
            # beside it rather than in place of it, so the two are never read
            # as the same number.
            print(f"    {record.wall_seconds:.0f}s | val acc {record.val_score_before:.3f} -> "
                  f"{record.val_score_after:.3f} | reflection {record.reflection_calls} calls / "
                  f"{record.reflection_out_tokens} out tok | instruction "
                  f"{'REWRITTEN' if changed else 'unchanged (no candidate beat baseline)'}")
            if record.pin_altered_instruction:
                print(f"    ! the format pin rewrote this candidate after GEPA scored it, so "
                      f"val acc {record.val_score_after:.3f} describes the scored text, not the "
                      f"shipped one -- trust the test row below, not the val number")
            if record.cost_penalty_margin > 0:
                print(f"      objective (ranked by) {record.val_objective_before:.3f} -> "
                      f"{record.val_objective_after:.3f} | prompt "
                      f"{record.baseline_instruction_tokens} -> "
                      f"{record.selected_instruction_tokens} instruction tokens, "
                      f"{record.prompt_cost_ratio:.2f}x input cost, scored "
                      f"{-record.val_cost_penalty * 100:+.1f} accuracy points")
            results[key + "+gepa"] = run_eval(
                spec, optimized, task.test, task.labels, num_threads=cfg.num_threads,
                max_tokens=mt, progress=True)
            results[key + "+gepa"].label = spec.label + " + GEPA"
    elif cfg.optimize:
        print(f"\nGEPA skipped: reflection model {cfg.reflection_model!r} unavailable.")

    # ── statistics ────────────────────────────────────────────────────────
    ref_key = cfg.reference_model or max(
        results, key=lambda k: results[k].accuracy.point)
    ref = results.get(ref_key)
    comparisons: dict[str, dict] = {}
    if ref is not None:
        margin = cfg.non_inferiority_margin
        bar = cfg.accuracy_bar if cfg.accuracy_bar is not None else ref.accuracy.point - margin
        print(f"\nPaired comparisons against '{ref.label}' "
              f"({ref.accuracy.point:.1%}), non-inferiority margin {margin:.0%}:")
        raw_p: dict[str, float] = {}
        for key, res in results.items():
            if key == ref_key or res.n != ref.n:
                continue
            mc = stats.mcnemar_test(res.correct, ref.correct)
            ni = stats.non_inferiority_test(res.correct, ref.correct, margin=margin)
            comparisons[key] = {"mcnemar": mc.to_dict(), "non_inferiority": ni.to_dict()}
            raw_p[key] = mc.p_value
        rejects = stats.holm_bonferroni(list(raw_p.values()))
        for (key, p), rej in zip(raw_p.items(), rejects):
            comparisons[key]["holm_significant"] = bool(rej)

        # ── multiplicity on the decision that is actually published ──
        #
        # The Holm correction above is computed on the TWO-SIDED McNemar p, and
        # nothing reads it: the PASS/FAIL on the chart and the `passes` field in
        # results.json both come from the per-row non-inferiority CI, at an
        # uncorrected one-sided 5% each. So the family-wise error rate of the
        # decision the product sells is uncontrolled, while the correction that
        # exists guards a statistic no decision uses. (It also could not serve
        # as the gate: two-sided significance also fires for a model that is
        # significantly BETTER, which would veto good candidates.)
        #
        # Correcting it is a methodology choice with published consequences, so
        # this records BOTH and changes no verdict on its own. On the saved
        # Financial PhraseBank run the full family is m=31 and three rows lose
        # their PASS under Holm -- gpt-5-nano, deepseek-v4-flash and
        # cohere-command-a-plus, two of which are cheap rows, i.e. exactly the
        # "cheapest model that clears your bar" answer. Restricting the family
        # to the rows that currently pass (m=11) flips none, because ~20 of the
        # 31 hypotheses are hopeless a priori (majority, qwen3-0.6b, tfidf) and
        # inflate the correction. Which family is right is the maintainer's
        # call; `passes` stays as-is until that call is made.
        ni_p = [comparisons[k]["non_inferiority"]["p_value"] for k in raw_p]
        ni_rejects = stats.holm_bonferroni(ni_p)
        for key, rej in zip(raw_p, ni_rejects):
            comparisons[key]["non_inferiority_holm_significant"] = bool(rej)
            comparisons[key]["passes_holm_corrected"] = bool(
                comparisons[key]["non_inferiority"]["passes"] and rej)
        n_would_flip = sum(1 for k in raw_p
                           if comparisons[k]["non_inferiority"]["passes"]
                           and not comparisons[k]["passes_holm_corrected"])
        if n_would_flip:
            print(f"  NOTE: {n_would_flip} of "
                  f"{sum(1 for k in raw_p if comparisons[k]['non_inferiority']['passes'])} "
                  f"PASS verdict(s) would not survive Holm correction over this family of "
                  f"{len(ni_p)} non-inferiority tests. `passes` is UNCORRECTED; see "
                  f"passes_holm_corrected on each comparison.")
            ni = comparisons[key]["non_inferiority"]
            verdict = ni["verdict"]
            mark = "PASS" if ni["passes"] else "----"
            print(f"  [{mark}] {results[key].label:<34} {results[key].accuracy.point:6.1%}  "
                  f"diff {ni['diff']:+.1%}  p={p:.3f}"
                  f"{' (holm sig)' if rej else ''}  {verdict}")
    else:
        bar = cfg.accuracy_bar

    # ── chart ─────────────────────────────────────────────────────────────
    rows: list[ChartRow] = []
    for key, res in results.items():
        base_key = key.replace("+gepa", "")
        family = _family_for(base_key)
        linked = None
        if key.endswith("+gepa") and base_key in results:
            b = results[base_key]
            linked = (b.cost_per_1k_items, b.accuracy.point)
        opt_record = optimizations.get(base_key) if key.endswith("+gepa") else None
        # Both ends of the arrow are per-classification costs, which is what
        # the axis says. cost_per_1k_calls is per billed LM call and undercounts
        # any row whose items took an adapter retry -- by 1.99x at worst here.
        rows.append(ChartRow(
            label=res.label, family=family,
            cost_basis=cost_basis_for(key, res.cost_per_1k_items),
            cost_per_1k=res.cost_per_1k_items, accuracy=res.accuracy.point,
            ci_lo=res.accuracy.lo, ci_hi=res.accuracy.hi,
            annotate="", linked_from=linked,
            always_label=(key in ALWAYS_LABEL_KEYS or key == ref_key),
            prompt_changed=(None if opt_record is None or opt_record.error
                            else _prompt_changed(opt_record))))

    chart_path = str(out_dir / "cost_vs_accuracy.png")
    plot_cost_vs_accuracy(
        rows, chart_path, majority_baseline=maj_acc, accuracy_bar=bar,
        title="Cheapest model that clears the bar",
        subtitle=(f"{task_spec.label} - {len(task.test)} held-out test items, "
                  f"never seen by the optimizer. Bars are 95% Wilson intervals."),
        footnote=cost_footnote(rows, "2026-09-09"))
    print(f"\nChart: {chart_path}")

    payload = {
        "config": asdict(cfg),
        "task": {"name": task.name, "key": cfg.task, "label": task_spec.label, "labels": list(task.labels),
                 "majority_label": maj_label, "majority_accuracy": maj_acc,
                 "n_train": len(task.train), "n_val": len(task.val),
                 "n_test": len(task.test), "n_pool": len(task.pool),
                 # Content hash of the ordered test split. Every paired
                 # comparison here is positional, and same-seed/same-size is
                 # NOT sufficient to prove two runs scored the same items.
                 "split_fingerprint": split_fingerprint(task.test),
                 "duplicates_dropped": task.n_duplicates_dropped,
                 "resolution_note": stats.resolution_warning(cfg.n_test)},
        # What the task demanded and who was refused before any spending. Kept
        # on the run, not just printed, so a published chart can answer "was
        # this model considered?" -- absent from the chart has three different
        # causes (impossible, unavailable, never listed) and they must not
        # collapse into one silence.
        "capability_screen": {
            "demands": screen["demands"].as_dict(),
            "runnable": list(screen["runnable"]),
            "rejected": {k: w for k, w in screen["rejected"]},
            "unverified": {k: w for k, w in screen["unverified"]},
            "unavailable": {k: w for k, w in screen["unavailable"]},
            "reflection_ok": screen["reflection_ok"],
            "reflection_note": screen["reflection_note"]},
        "results": {k: v.to_dict() for k, v in results.items()},
        "optimizations": {k: v.to_dict() for k, v in optimizations.items()},
        "comparisons": comparisons,
        "reference_model": ref_key,
        "accuracy_bar": bar,
        "wall_seconds": time.time() - started,
        "chart": chart_path,
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Results: {results_path}  ({payload['wall_seconds']:.0f}s total)")
    return payload
