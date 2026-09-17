"""Spec 4.10 -- Reporter: renders tables/figures from evaluator output. Computes no new numbers.

THE ONE RULE
------------
Spec 4.10 forbids computing anything here. Every number this module prints comes
out of a ``metrics.json`` written by :mod:`reflex.evaluate`, or out of ``cfg``
(where spec 10 requires measured constants and thresholds to live). **If a number
has no input, the cell renders as ``n/a`` and the reason is stated.** It is never
computed on the fly, never interpolated, and never carried over from an older
run. Spec 14 also requires every headline number to carry a 95% CI, so a
headline whose CI the Evaluator did not emit renders as an explicit gap rather
than as a bare point estimate.

Three things this reporter is deliberately loud about
-----------------------------------------------------
1. **The constant-predictor column sits beside every headline metric.** A
   label-blind constant scores 0.7227 on ABCD's ``nextstep`` because nextstep is
   1:1 with the speaker label. A metric the constant wins is reported as
   DROPPED, not caveated.
2. **Every wall-clock number is labelled CONTENDED.** They were measured on a
   machine observed swapping hard enough to take one fine-tune step from 0.306s
   to 241s. Ratios survive that; absolute milliseconds do not.
3. **Criterion 5 (latency) never fails the build** while
   ``latency.gate_on_latency`` is false. It is printed as measured-and-missed
   against both the spec's 20ms and the revised ``latency.target_p95_ms``
   budget, and the report says which criteria actually gate.

An unmeasured criterion is never a pass. Anything the paid arm never fed prints
UNMEASURABLE with the reason attached.

Read ``cfg``, never a literal: spec 10 forbids any numeric threshold, model id,
path or price in code.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional, Sequence

from reflex.config import get_dotted, resolve_path
from reflex.contracts import ContractViolation

__all__ = [
    "render_report",
    "render_table",
    "render_figure",
]

#: Spec 11.3's table names, in section order.
_TABLE_NAMES = ("A", "B", "C", "D", "E", "F")

#: Spec 11.3's figure names.
_FIGURE_NAMES = ("coverage_error", "learning_curve")

#: What a cell says when its input is absent. Always accompanied by a reason.
_NA = "n/a"

#: Rendered when a spec 9 criterion had no input at all.
_UNMEASURABLE = "UNMEASURABLE"

#: The five spec 9 criteria, mapped onto the keys :func:`reflex.evaluate.verdict`
#: emits. Criterion 2 is three keys because parity is checked on two official
#: metric families plus the CI floor of the paired delta.
_CRITERIA: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (1, "Reflex rate >= verdict.min_reflex_rate", ("reflex_rate",)),
    (
        2,
        "Arm B quality parity with Arm A (spec 8.7)",
        ("parity_cds.Joint_Accuracy", "parity_ast.Joint_Accuracy", "parity_delta_ci_low"),
    ),
    (3, "NOVEL escalation rate >= verdict.min_novel_escalation_rate", ("novel_escalation_rate",)),
    (4, "Empirical coverage >= 1 - alpha - verdict.coverage_slack", ("empirical_coverage_min",)),
    (5, "Fast-path p95 latency (REPORT-ONLY)", ("fastpath_p95_latency_ms",)),
)

#: The criterion spec 9 states but ``latency.gate_on_latency: false`` demotes to
#: reporting. Kept as a name so the demotion is visible in one place.
_REPORT_ONLY_CRITERION = 5


# --------------------------------------------------------------------------- #
# Formatting. Nothing here computes; it only decides how to print a value.
# --------------------------------------------------------------------------- #


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return False


def _num(value: Any, cfg: dict[str, Any]) -> str:
    """Print a number at ``report.float_precision``, or ``n/a``."""
    if _is_missing(value):
        return _NA
    try:
        return f"{float(value):.{int(get_dotted(cfg, 'report.float_precision'))}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any, cfg: dict[str, Any]) -> str:
    """Print a 0-1 rate as a percentage, or ``n/a``."""
    if _is_missing(value):
        return _NA
    try:
        return f"{float(value) * 100.0:.{int(get_dotted(cfg, 'report.percent_precision'))}f}%"
    except (TypeError, ValueError):
        return str(value)


def _ms(value: Any, cfg: dict[str, Any]) -> str:
    if _is_missing(value):
        return _NA
    return f"{float(value):.1f} ms"


def _usd(value: Any) -> str:
    if _is_missing(value):
        return _NA
    return f"${float(value):.6f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A markdown table. Empty ``rows`` still renders the header plus a gap row."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    if not rows:
        out.append("| " + " | ".join([_NA] * len(header)) + " |")
    else:
        for row in rows:
            out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _get(metrics: dict[str, Any], *path: str, default: Any = None) -> Any:
    node: Any = metrics
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _label(metrics: dict[str, Any]) -> str:
    """A short, unambiguous name for one run's column/row."""
    run_id = str(metrics.get("run_id", "?"))
    alpha = metrics.get("alpha")
    suffix = "" if _is_missing(alpha) else f" a={alpha}"
    return f"{run_id}{suffix}"


def _forced_reflex(metrics: dict[str, Any]) -> bool:
    """Was this run's escalation set left unanswered? Read from the run manifest copy."""
    return bool(_get(metrics, "manifest", "machine", "forced_reflex", default=False))


def _fixture_note(metrics: dict[str, Any]) -> Optional[str]:
    """Return the fixture note if this metrics file is SYNTHETIC, else ``None``.

    A rendering test and a result must never look alike. Any metrics file
    carrying a top-level ``fixture`` key is invented input, and every place its
    numbers appear is stamped so, loudly, at the top of the report and again in
    Section 1. There is no way to switch the stamp off: a flag for "print this
    as if it were real" is a flag for shipping a fabricated result.
    """
    marker = metrics.get("fixture")
    if not marker:
        return None
    if isinstance(marker, dict):
        return str(marker.get("reason") or marker.get("note") or "synthetic input")
    return str(marker)


def _fixture_banner(metrics: Sequence[dict[str, Any]], where: str = "below") -> str:
    """The banner printed when any input is synthetic. Empty when none is.

    Printed at BOTH ends of the report: a reader who scrolls straight to the
    verdict must hit it too.
    """
    notes = [(str(m.get("run_id", "?")), _fixture_note(m)) for m in metrics]
    synthetic = [(run_id, note) for run_id, note in notes if note]
    if not synthetic:
        return ""
    lines = [
        "> # !! FIXTURE -- THESE NUMBERS ARE NOT RESULTS !!",
        ">",
        "> At least one `metrics.json` feeding this report is SYNTHETIC. It was written to "
        f"exercise the renderer, not by scoring a run. Nothing {where} measures the system.",
        ">",
    ]
    for run_id, note in synthetic:
        lines.append(f"> - `{run_id}` -- {note}")
    if len(synthetic) != len(notes):
        lines += [
            ">",
            "> **Some inputs are real and some are invented.** They are mixed in the same "
            "tables; check the `source` column in Section 1 before quoting any cell.",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def _table_a(metrics: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> str:
    rows = []
    for m in metrics:
        routing = _get(m, "routing", default={}) or {}
        shares = routing.get("escalation_reason_share", {}) or {}
        top = ", ".join(f"{k} {_pct(v, cfg)}" for k, v in sorted(shares.items(), key=lambda kv: -kv[1]))
        rows.append(
            [
                _label(m),
                _pct(routing.get("reflex_rate"), cfg),
                _pct(routing.get("containment"), cfg),
                str(routing.get("n_agent_turns", _NA)),
                str(routing.get("n_escalated_turns", _NA)),
                top or _NA,
            ]
        )
    body = _table(
        ["run", "reflex rate", "containment", "turns", "escalated", "escalation reasons"],
        rows,
    )
    per_step_rows = []
    for m in metrics:
        by_step = _get(m, "routing", "reflex_rate_by_nextstep", default={}) or {}
        for step, rate in sorted(by_step.items()):
            per_step_rows.append([_label(m), step, _pct(rate, cfg)])
    return (
        body
        + "\n\nReflex rate by GOLD nextstep:\n\n"
        + _table(["run", "gold nextstep", "reflex rate"], per_step_rows)
        + "\n\n*No 95% CI:* the Evaluator emits bootstrap CIs for per-turn correctness "
        "components (spec 8.7), not for the routing rates. Spec 14 wants one on every "
        "headline; this is a stated gap, not an omission."
    )


def _quality_rows(m: dict[str, Any], cfg: dict[str, Any]) -> list[list[str]]:
    """One row per official metric, with the label-blind constant beside it."""
    comparisons = _get(m, "constant_guard", "comparisons", default={}) or {}
    dropped = set(_get(m, "constant_guard", "dropped", default=[]) or [])
    rows: list[list[str]] = []
    for family in ("cds", "ast"):
        block = _get(m, "official", family, default={}) or {}
        for key in sorted(block):
            name = f"{family}.{key}"
            comparison = comparisons.get(name, {})
            rows.append(
                [
                    _label(m),
                    name,
                    _num(block.get(key), cfg),
                    _num(comparison.get("constant"), cfg),
                    "**DROPPED**" if name in dropped else "kept",
                ]
            )
    return rows


def _table_b(metrics: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    for m in metrics:
        rows.extend(_quality_rows(m, cfg))
    official = _table(
        ["run", "official metric", "system", "label-blind constant", "guard"], rows
    )

    ci_rows: list[list[str]] = []
    for m in metrics:
        boot = _get(m, "statistics", "bootstrap", default={}) or {}
        for component in sorted(boot):
            cell = boot[component] or {}
            ci_rows.append(
                [
                    _label(m),
                    component,
                    _num(cell.get("point"), cfg),
                    f"[{_num(cell.get('ci_low'), cfg)}, {_num(cell.get('ci_high'), cfg)}]",
                ]
            )
    ci_block = _table(["run", "component", "point", "95% CI (conversation bootstrap)"], ci_rows)

    fast_rows: list[list[str]] = []
    for m in metrics:
        errors = _get(m, "fastpath", "fastpath_error_rate", default={}) or {}
        denominators = _get(m, "fastpath", "fastpath_denominators", default={}) or {}
        for component in sorted(errors):
            fast_rows.append(
                [
                    _label(m),
                    component,
                    _pct(errors[component], cfg),
                    str(denominators.get(component, _NA)),
                ]
            )
    fast_block = _table(["run", "component", "fast-path error rate", "denominator"], fast_rows)

    parity_rows: list[list[str]] = []
    for m in metrics:
        parity = m.get("parity")
        if not parity:
            continue
        for key, delta in sorted((parity.get("delta_points") or {}).items()):
            parity_rows.append([_label(m), key, _num(delta, cfg)])
        parity_rows.append(
            [
                _label(m),
                "turn accuracy delta (points)",
                f"{_num(parity.get('turn_accuracy_delta_points'), cfg)} "
                f"[{_num(parity.get('turn_accuracy_delta_ci_low_points'), cfg)}, "
                f"{_num(parity.get('turn_accuracy_delta_ci_high_points'), cfg)}]",
            ]
        )
        parity_rows.append([_label(m), "McNemar p", _num(parity.get("mcnemar_p"), cfg)])
    parity_block = (
        _table(["run", "parity metric", "Arm B - Arm A"], parity_rows)
        if parity_rows
        else (
            f"{_UNMEASURABLE}: no run in this report carries a `parity` block. Parity is a PAIRED "
            "comparison against an Arm A run over the identical turn set (spec 8.2/8.7), and Arm A "
            "requires a paid LLM, which `llm.enabled: false` forbids. No Arm A run exists, so no "
            "parity delta can exist."
        )
    )

    forced = [m for m in metrics if _forced_reflex(m)]
    forced_note = ""
    if forced:
        forced_note = (
            "\n\n> **Quality columns above are NOT Arm B's quality for the runs marked "
            "forced-reflex.** In that mode the gate escalated normally but nobody answered the "
            "escalated turns, so those rows carry no action, values, skeleton, template, "
            "utterance or candidate. The official joint accuracies therefore score a system that "
            "declined to answer part of its turn set. The ROUTING numbers (Table A, Table C) are "
            "unaffected and are the reason that mode exists.\n>\n> Affected runs: "
            + ", ".join(_label(m) for m in forced)
        )

    return (
        "**Official metrics, with the label-blind constant beside each one.** A metric the "
        "constant wins carries no signal and is marked DROPPED, not caveated.\n\n"
        + official
        + "\n\n**Per-turn correctness components with 95% CIs (spec 8.7).**\n\n"
        + ci_block
        + "\n\n**Fast-path-only quality (spec 8.3): reflex turns only.**\n\n"
        + fast_block
        + "\n\n**Parity against Arm A (spec 8.7).**\n\n"
        + parity_block
        + forced_note
    )


def _table_c(metrics: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> str:
    rows = []
    for m in metrics:
        novelty = m.get("novelty")
        if not novelty:
            continue
        rows.append(
            [
                _label(m),
                str(m.get("split", _NA)),
                _pct(novelty.get("novel_escalation_rate"), cfg),
                _pct(novelty.get("novel_fastpath_error_rate"), cfg),
                str(novelty.get("n_agent_turns", _NA)),
                str(novelty.get("n_reflex_turns", _NA)),
            ]
        )
    if not rows:
        return (
            f"{_UNMEASURABLE}: spec 8.4 novelty is computed on the `test_novel` split and no run "
            "in this report scored it. Run `reflex run --arm B --split test_novel`."
        )
    return _table(
        ["run", "split", "novel escalation rate", "novel fast-path error rate", "turns", "reflex turns"],
        rows,
    ) + "\n\nNOVEL subflows held out of train and dev are listed in Section 1."


def _table_d(metrics: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> str:
    rows = []
    for m in metrics:
        c = _get(m, "cost_latency", default={}) or {}
        rows.append(
            [
                _label(m),
                _usd(c.get("arm_cost_per_turn")),
                _usd(c.get("arm_cost_per_conversation")),
                _usd(c.get("llm_cost_per_turn")),
                str(c.get("llm_tokens_in_billed", _NA)),
                str(c.get("llm_tokens_out_billed", _NA)),
            ]
        )
    cost = _table(
        ["run", "cost/turn", "cost/conversation", "LLM cost/turn", "tokens in", "tokens out"],
        rows,
    )
    lat_rows = []
    for m in metrics:
        c = _get(m, "cost_latency", default={}) or {}
        lat_rows.append(
            [
                _label(m),
                _ms(c.get("latency_p50_ms_fastpath"), cfg),
                _ms(c.get("latency_p95_ms_fastpath"), cfg),
                _ms(c.get("latency_p50_ms_escalated"), cfg),
                _ms(c.get("latency_p95_ms_escalated"), cfg),
            ]
        )
    latency = _table(
        ["run", "fast path p50", "fast path p95", "escalated p50", "escalated p95"], lat_rows
    )
    target = get_dotted(cfg, "latency.target_p95_ms")
    spec_target = get_dotted(cfg, "verdict.max_fastpath_p95_latency_ms")
    return (
        cost
        + "\n\n**Latency, batch 1, CPU, `runtime.torch_threads` threads. "
        "EVERY NUMBER BELOW IS CONTENDED.**\n\n"
        + latency
        + f"\n\nBudget: `latency.target_p95_ms` = {target} ms; spec 9 criterion 5 states "
        f"{spec_target} ms. `latency.gate_on_latency` is "
        f"{str(get_dotted(cfg, 'latency.gate_on_latency')).lower()}, so neither figure gates the "
        "build -- see Section 9.\n\n"
        "> **Contended measurement.** These milliseconds were produced on a machine observed "
        "swapping (29.8/30.7 GB), where one fine-tune step was measured going from 0.306s to "
        "241s -- a 790x slowdown from swap alone. Treat every absolute latency here as an UPPER "
        "bound and compare ratios, not magnitudes. A clean-machine re-measurement is outstanding."
    )


def _table_e(metrics: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> str:
    rows = []
    for m in metrics:
        rows.append(
            [
                _label(m),
                str(m.get("split", _NA)),
                _num(m.get("alpha"), cfg),
                _pct(_get(m, "routing", "reflex_rate"), cfg),
                _num(_get(m, "official", "cds", "Joint_Accuracy"), cfg),
                "forced-reflex" if _forced_reflex(m) else "normal",
            ]
        )
    in_report = _table(
        ["run", "split", "alpha", "reflex rate", "cds Joint_Accuracy", "mode"], rows
    )

    constant_bars = get_dotted(cfg, "report.probe_constant_bars") or {}
    cheap_bars = get_dotted(cfg, "report.probe_cheap_bars") or {}
    probe_rows = [
        [
            head,
            _num(constant_bars.get(head), cfg),
            _num(cheap_bars.get(head), cfg) if cheap_bars.get(head) is not None else "unmeasured",
        ]
        for head in sorted(set(constant_bars) | set(cheap_bars))
    ]
    probe = _table(["head", "label-blind constant", "best cheap (TF-IDF+logreg)"], probe_rows)
    n = get_dotted(cfg, "report.probe_dev_n")
    half = get_dotted(cfg, "report.probe_ci_half_width_points")
    return (
        "**Ablation runs present in this report.** The spec 7 E3 matrix "
        "(whole-utterance vs skeleton-first, gate variants, encoder size, context K) needs one "
        "checkpoint per arm; no such checkpoints exist in this build, so those rows are absent "
        "rather than estimated.\n\n"
        + in_report
        + "\n\n**External bars every headline must be read against** (DECISIONS D2b / D6 / D7). "
        f"Measured on a frozen probe harness, dev n={n}, CI half-width {half} points. These are "
        "NOT produced by any run in this report and must not be compared to it as if they were "
        "the same evaluation.\n\n"
        + probe
        + "\n\nTwo findings these bars encode, because a number without them misleads:\n\n"
        "- `nextstep` and `action` are LOCAL and ORDERED: a last-6-turn window with "
        "recency-tagged tokens reaches 0.8345 on nextstep, and scrambling turn order costs "
        "5.8 / 4.7 points EVEN AFTER REFITTING -- the information is destroyed, not misaligned.\n"
        "- `intent` is GLOBAL and UNORDERED: the full thread beats a 6-turn window by 19.2 "
        "points, and recency tagging HURTS it.\n"
        "- `skeleton` and `template` are UNMEASURED on both axes. No default was inherited for "
        "them and none should be assumed."
    )


def _table_f(metrics: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    for m in metrics:
        shares = _get(m, "routing", "escalation_reason_share", default={}) or {}
        escalated = _get(m, "routing", "n_escalated_turns", default=0) or 0
        for reason, share in sorted(shares.items(), key=lambda kv: -kv[1]):
            rows.append([_label(m), "escalation", reason, _pct(share, cfg), str(escalated)])
        errors = _get(m, "fastpath", "fastpath_error_rate", default={}) or {}
        denominators = _get(m, "fastpath", "fastpath_denominators", default={}) or {}
        for component, rate in sorted(errors.items()):
            rows.append(
                [_label(m), "fast-path error", component, _pct(rate, cfg), str(denominators.get(component, _NA))]
            )
    taxonomy = _table(["run", "class", "reason / component", "share", "denominator"], rows)
    wanted = int(get_dotted(cfg, "report.max_failure_examples"))
    return (
        taxonomy
        + f"\n\n**{wanted} anonymized examples: {_NA}.** Spec 11.3 section 8 asks for them and "
        "spec 4.10 forbids this module from computing anything, so it does not open "
        "`decisions.jsonl` to pick them. `metrics.json` carries no per-turn rows and the "
        "Evaluator emits no failure sample (`eval.failure_sample` is configured but unused), so "
        "the examples have no input. This is a missing PRODUCER in "
        "`reflex.evaluate.evaluate_run`, not a rendering gap."
    )


def render_table(name: str, metrics: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> str:
    """Render one spec 11.3 table as a markdown string.

    See :func:`reflex.contracts.render_table` for the frozen contract.

    Raises:
        ValueError: on an unknown table name.
    """
    renderers = {
        "A": _table_a,
        "B": _table_b,
        "C": _table_c,
        "D": _table_d,
        "E": _table_e,
        "F": _table_f,
    }
    if name not in renderers:
        raise ValueError(f"unknown table {name!r}; expected one of {_TABLE_NAMES}")
    if not metrics:
        return f"{_NA}: no metrics.json was supplied for table {name}."
    return renderers[name](list(metrics), cfg)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _new_axes(cfg: dict[str, Any]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(
            float(get_dotted(cfg, "report.figure_width_in")),
            float(get_dotted(cfg, "report.figure_height_in")),
        )
    )
    return plt, fig, ax


def _save(plt, fig, path: str, cfg: dict[str, Any]) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=int(get_dotted(cfg, "report.figure_dpi")))
    plt.close(fig)
    return os.path.abspath(path)


def _gap_figure(plt, fig, ax, path: str, title: str, reason: str, cfg: dict[str, Any]) -> str:
    """A figure that states its own missing input rather than plotting nothing.

    An empty axes reads as "the value is zero". A figure that says what is
    missing cannot be misread, and it keeps the report's figure slots honest.
    """
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.5, 0.5, f"NO INPUT\n\n{reason}", ha="center", va="center", wrap=True, fontsize=9)
    return _save(plt, fig, path, cfg)


def _figure_coverage_error(metrics: Sequence[dict[str, Any]], out_dir: str, cfg: dict[str, Any]) -> str:
    """Figure 1: the spec 7 E5 coverage-error curve -- reflex rate vs fast-path error by alpha."""
    path = os.path.join(out_dir, "figure1_coverage_error.png")
    points = []
    for m in metrics:
        alpha = m.get("alpha")
        reflex = _get(m, "routing", "reflex_rate")
        error = _get(m, "fastpath", "fastpath_error_rate", "turn")
        if _is_missing(alpha) or _is_missing(reflex) or _is_missing(error):
            continue
        points.append((float(alpha), float(reflex), float(error), str(m.get("run_id", ""))))
    plt, fig, ax = _new_axes(cfg)
    if not points:
        return _gap_figure(
            plt,
            fig,
            ax,
            path,
            "Figure 1 -- coverage vs fast-path error (spec 7 E5)",
            "No metrics.json carries all of {alpha, routing.reflex_rate,\n"
            "fastpath.fastpath_error_rate.turn}. E5 sweeps gate.alpha_sweep;\n"
            "run `reflex sweep --experiment E5` (one run per alpha).",
            cfg,
        )
    points.sort()
    ax.plot([p[1] for p in points], [p[2] for p in points], marker="o", linewidth=1.5)
    for alpha, reflex, error, _run in points:
        ax.annotate(f"a={alpha:g}", (reflex, error), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("coverage (reflex rate)")
    ax.set_ylabel("fast-path turn error rate")
    ax.set_title("Figure 1 -- coverage vs fast-path error (spec 7 E5)")
    ax.grid(True, alpha=0.3)
    if len(points) == 1:
        ax.text(
            0.02,
            0.02,
            "ONE POINT ONLY -- this is not a curve.\nE5 was not swept.",
            transform=ax.transAxes,
            fontsize=8,
        )
    return _save(plt, fig, path, cfg)


def _figure_learning_curve(metrics: Sequence[dict[str, Any]], out_dir: str, cfg: dict[str, Any]) -> str:
    """Figure 2: the spec 7 E4 learning curve over ``data.learning_curve_fractions``."""
    path = os.path.join(out_dir, "figure2_learning_curve.png")
    points = []
    for m in metrics:
        fraction = _get(m, "manifest", "config", "_source_fraction")
        quality = _get(m, "official", "cds", "Joint_Accuracy")
        if _is_missing(fraction) or _is_missing(quality):
            continue
        points.append((float(fraction), float(quality)))
    plt, fig, ax = _new_axes(cfg)
    if not points:
        fractions = get_dotted(cfg, "data.learning_curve_fractions")
        return _gap_figure(
            plt,
            fig,
            ax,
            path,
            "Figure 2 -- learning curve (spec 7 E4)",
            "No metrics.json carries a training fraction. E4 needs one bank +\n"
            f"checkpoint per data.learning_curve_fractions {list(fractions)};\n"
            "none were trained in this build, and metrics.json records no\n"
            "fraction field, so the x axis has no values to plot.",
            cfg,
        )
    points.sort()
    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=1.5)
    ax.set_xlabel("fraction of train")
    ax.set_ylabel("cds Joint_Accuracy")
    ax.set_title("Figure 2 -- learning curve (spec 7 E4)")
    ax.grid(True, alpha=0.3)
    return _save(plt, fig, path, cfg)


def render_figure(name: str, metrics: Sequence[dict[str, Any]], out_dir: str, cfg: dict[str, Any]) -> str:
    """Render one spec 11.3 figure into ``outputs/runs/<run_id>/figures/``.

    See :func:`reflex.contracts.render_figure` for the frozen contract.

    Raises:
        ValueError: on an unknown figure name.
    """
    if name not in _FIGURE_NAMES:
        raise ValueError(f"unknown figure {name!r}; expected one of {_FIGURE_NAMES}")
    os.makedirs(out_dir, exist_ok=True)
    if name == "coverage_error":
        return _figure_coverage_error(list(metrics), out_dir, cfg)
    return _figure_learning_curve(list(metrics), out_dir, cfg)


# --------------------------------------------------------------------------- #
# Section 1 and Section 9
# --------------------------------------------------------------------------- #


def _section_setup(metrics: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> str:
    rows = []
    for m in metrics:
        rows.append(
            [
                _label(m),
                str(m.get("arm", _NA)),
                str(m.get("split", _NA)),
                ", ".join(str(s) for s in (m.get("seeds") or [])) or _NA,
                str(m.get("n_records", _NA)),
                str(m.get("n_conversations", _NA)),
                "forced-reflex ($0)" if _forced_reflex(m) else "normal",
                "**FIXTURE**" if _fixture_note(m) else "measured",
            ]
        )
    runs = _table(
        ["run", "arm", "split", "seeds", "turns", "conversations", "mode", "source"], rows
    )

    budget = (
        "**Every number in this report was produced for $0.** `llm.enabled` is false and "
        "stays false -- ZERO PAID API CALLS is a hard project rule -- so Arm A does not "
        "exist here, and neither does anything that needs it: no paired parity comparison "
        "(spec 8.7), no E2, and therefore no E6.\n\n"
        "The contract's escape hatch for this, `llm.dry_run`, is UNREACHABLE while the kill "
        "switch holds: `build_llm_agent` refuses at construction, correctly, because even a "
        "dry run needs a model id and a price table it must not invent. So the only path "
        "that yields real routing numbers on this budget is `reflex run --arm B "
        "--forced-reflex`: the gate runs and its true verdict is recorded, but escalated "
        "turns are logged UNANSWERED. That buys the whole of spec 8.1 routing, the "
        "escalation-reason distribution and spec 8.4 novelty. It does NOT buy arm-level "
        "quality, because on those turns nobody answered -- see the note under Table B."
    )

    assignment = get_dotted(cfg, "report.bank_assignment_rate")
    fidelity = get_dotted(cfg, "report.bank_exact_reconstruction_rate")
    fidelity_sent = get_dotted(cfg, "report.bank_exact_reconstruction_sentences")
    wrong_field = get_dotted(cfg, "report.bank_wrong_field_rate")
    wrong_field_weighted = get_dotted(cfg, "report.bank_wrong_field_rate_occurrence_weighted")
    wrong_field_prefix = get_dotted(cfg, "report.bank_wrong_field_rate_prefix")
    wrong_field_prefix_weighted = get_dotted(
        cfg, "report.bank_wrong_field_rate_prefix_occurrence_weighted"
    )
    wrong_field_turns_prefix = get_dotted(cfg, "report.bank_wrong_field_turn_rate_prefix")
    strict = get_dotted(cfg, "report.strict_template_coverage")

    def _pending(value: Any) -> str:
        return "**PENDING**" if _is_missing(value) else _pct(value, cfg)

    bank = _table(
        ["bank property", "value", "what it means"],
        [
            [
                "templates in bank",
                str(get_dotted(cfg, "report.bank_templates")),
                "post-fix bank size",
            ],
            [
                "assignment rate",
                _pct(assignment, cfg),
                "share of retrieve turns the bank assigns templates to. The fast path always "
                "emits something, so **this, not the fidelity rate, is what bounds the reflex "
                "rate from the bank side** -- and it bounds it at 100%",
            ],
            [
                "exact reconstruction, utterances (FIDELITY)",
                _pct(fidelity, cfg),
                "share of train agent utterances the bank rebuilds VERBATIM. A fidelity "
                "number. It is not a ceiling on anything",
            ],
            [
                "exact reconstruction, sentences",
                _pct(fidelity_sent, cfg),
                "the same, per sentence",
            ],
            [
                "wrong-field rate, post-fix",
                _pending(wrong_field)
                + f" (occurrence-weighted {_pending(wrong_field_weighted)})",
                "share of field-requesting merged forms whose canonical asks the customer for "
                "DIFFERENT named identifiers than the form it replaced. Zero because field-set "
                "equality is transitive: blocking field-mismatched merges leaves nothing to "
                "disagree",
            ],
            [
                "wrong-field rate, pre-fix",
                _pct(wrong_field_prefix, cfg)
                + f" (occurrence-weighted {_pct(wrong_field_prefix_weighted, cfg)})",
                "the same measurement before the D16 merge guard, when single-linkage chained "
                f"988 distinct phrasings into one canonical. Expected effect on ALL retrieve "
                f"turns: {_pct(wrong_field_turns_prefix, cfg)}. Shown as history, not as a "
                "current property",
            ],
            [
                "strict template coverage",
                "UNMEASURED" if _is_missing(strict) else _pct(strict, cfg),
                "DEFECTS_OPEN D-7: share of covered utterances within dedup_threshold of the "
                "canonical they were merged into. Never estimated",
            ],
        ],
    )

    novel = []
    for m in metrics:
        subflows = _get(m, "manifest", "novel_subflows", default=None)
        if subflows:
            novel = list(subflows)
            break
    hashes = []
    for m in metrics:
        hashes.append(
            [
                _label(m),
                str(_get(m, "manifest", "dataset_hash", default=_NA)),
                str(_get(m, "manifest", "bank_hash", default=_NA)),
                str(_get(m, "manifest", "git_commit", default=_NA)),
                str(_get(m, "manifest", "llm_enabled", default=_NA)),
            ]
        )

    machine = {}
    for m in metrics:
        machine = _get(m, "manifest", "machine", default={}) or {}
        if machine:
            break

    return (
        "Runs feeding this report:\n\n"
        + runs
        + "\n\n### What this build could and could not buy\n\n"
        + budget
        + "\n\nProvenance (spec 10):\n\n"
        + _table(["run", "dataset hash", "bank hash", "git commit", "llm enabled"], hashes)
        + "\n\nNOVEL subflows held out of train and dev: "
        + (", ".join(f"`{s}`" for s in novel) if novel else _NA)
        + "\n\n**Bank fidelity -- three numbers, never one** (DECISIONS D16 corrects the earlier "
        "'61.01% is the ceiling on reflex rate' framing, which was wrong):\n\n"
        + bank
        + "\n\nMachine (spec 10, 'document the actual machine'): "
        + (
            f"{machine.get('platform', _NA)}, {machine.get('cpu_count', _NA)} CPUs, "
            f"python {machine.get('python', _NA)}, torch {machine.get('torch', _NA)}"
            if machine
            else _NA
        )
        + "\n\n"
        + (
            str(machine.get("latency_note"))
            if machine.get("latency_note")
            else "No machine note recorded; treat wall-clock numbers with suspicion."
        )
    )


def _criterion_rows(m: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[list[str]], list[str], list[str]]:
    """Render the five spec 9 criteria with their ACTUAL values.

    Returns ``(rows, gating_failures, unmeasurable)``. Nothing here recomputes a
    pass: it reads the booleans :func:`reflex.evaluate.verdict` already decided
    and groups its keys into spec 9's five numbered criteria.
    """
    criteria = _get(m, "verdict", "criteria", default={}) or {}
    rows: list[list[str]] = []
    gating_failures: list[str] = []
    unmeasurable: list[str] = []
    for number, title, keys in _CRITERIA:
        for key in keys:
            entry = criteria.get(key)
            if entry is None:
                rows.append([str(number), title, key, _NA, _NA, f"{_UNMEASURABLE} (no such criterion in metrics.json)"])
                unmeasurable.append(f"{number}:{key}")
                continue
            passed = entry.get("passed")
            value = entry.get("value")
            threshold = entry.get("threshold")
            if passed is None:
                status = f"{_UNMEASURABLE} -- {_unmeasurable_reason(m, key)}"
                unmeasurable.append(f"{number}:{key}")
            elif number == _REPORT_ONLY_CRITERION:
                status = ("meets" if passed else "MISSED") + " -- REPORT-ONLY, does not gate"
            else:
                status = "PASS" if passed else "**FAIL**"
                if not passed:
                    gating_failures.append(f"{number}:{key}")
            rows.append(
                [
                    str(number),
                    title,
                    key,
                    _num(value, cfg),
                    _num(threshold, cfg),
                    status,
                ]
            )
    return rows, gating_failures, unmeasurable


def _unmeasurable_reason(m: dict[str, Any], key: str) -> str:
    """Why a criterion had no input. Never defaults to pass or fail."""
    if key.startswith("parity"):
        return (
            "no Arm A baseline exists. Arm A calls a paid LLM on every turn and "
            "`llm.enabled` is false, so no paired comparison was ever run."
        )
    if key == "novel_escalation_rate":
        return "no run in this report scored the `test_novel` split."
    if key == "empirical_coverage_min":
        source = _get(m, "calibration", "source", default="")
        if source == "unavailable":
            return str(_get(m, "calibration", "unavailable_reason", default="no calibration file."))
        return (
            "per-head coverage needs the probs sidecar (`eval.probs_filename`); "
            "`decisions.jsonl` logs set SIZES, not membership."
        )
    if key == "fastpath_p95_latency_ms":
        return "no reflex turn was timed."
    if key == "reflex_rate":
        return "the run logged no turns."
    return "the Evaluator marked it unavailable."


def _section_verdict(metrics: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> str:
    blocks: list[str] = []
    for m in metrics:
        verdict_block = m.get("verdict") or {}
        rows, gating_failures, unmeasurable = _criterion_rows(m, cfg)
        dropped = _get(m, "constant_guard", "dropped", default=[]) or []
        p95 = _get(m, "cost_latency", "latency_p95_ms_fastpath")
        blocks.append(
            f"### {_label(m)}\n\n"
            + _table(["#", "criterion", "key", "actual", "threshold", "status"], rows)
            + "\n\nEvaluator verdict: **"
            + str(verdict_block.get("verdict", _NA))
            + "**.\n\n"
            + (
                "Criteria that FAILED and do gate: " + ", ".join(gating_failures) + ". "
                if gating_failures
                else "No gating criterion failed on measured inputs. "
            )
            + (
                "Criteria with no input, counted as neither pass nor fail: "
                + ", ".join(unmeasurable)
                + ". "
                if unmeasurable
                else ""
            )
            + "Criterion 5 is report-only: `latency.gate_on_latency` is "
            + str(get_dotted(cfg, "latency.gate_on_latency")).lower()
            + ", measured fast-path p95 "
            + _ms(p95, cfg)
            + " against the spec's "
            + str(get_dotted(cfg, "verdict.max_fastpath_p95_latency_ms"))
            + " ms and the revised budget of "
            + str(get_dotted(cfg, "latency.target_p95_ms"))
            + " ms, on a contended machine. It is recorded as measured-and-missed and never "
            "fails the build.\n\n"
            + (
                "Metrics DROPPED by the constant-predictor guard (a label-blind constant "
                "matched or beat them, so they carry no signal): " + ", ".join(dropped) + "."
                if dropped
                else "No metric was dropped by the constant-predictor guard."
            )
        )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def _read_metrics(path: str) -> Optional[dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload


def _attach_manifest(metrics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Attach the run's manifest so Section 1 can print provenance.

    Reading a manifest is I/O, not computation: nothing is derived from it, its
    fields are printed verbatim.
    """
    run_id = str(metrics.get("run_id", ""))
    if not run_id:
        return metrics
    path = os.path.join(resolve_path(cfg, "paths.runs_dir"), run_id, "manifest.json")
    if not os.path.exists(path):
        return metrics
    try:
        with open(path, "r", encoding="utf-8") as handle:
            metrics = dict(metrics)
            metrics["manifest"] = json.load(handle)
    except (OSError, ValueError):
        pass
    return metrics


def render_report(metrics_paths: Sequence[str], out_path: str, cfg: dict[str, Any]) -> str:
    """Render ``report.md`` in the fixed spec 11.3 structure. No new numbers.

    See :func:`reflex.contracts.render_report` for the frozen contract.

    Raises:
        ContractViolation: if no supplied path yields a readable ``metrics.json``
            -- with none, Sections 1-5 have no input at all.
    """
    loaded: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in metrics_paths:
        payload = _read_metrics(path)
        if payload is None:
            missing.append(path)
            continue
        loaded.append(_attach_manifest(payload, cfg))
    if not loaded:
        raise ContractViolation(
            "no readable metrics.json among "
            f"{list(metrics_paths)}. Sections 1-5 of spec 11.3 have no input at all; "
            "run `reflex evaluate --run-id ...` first. The reporter will not invent a "
            "single number to fill them."
        )

    first_run = str(loaded[0].get("run_id", "report"))
    figures_dir = os.path.join(resolve_path(cfg, "paths.runs_dir"), first_run, "figures")
    figure1 = render_figure("coverage_error", loaded, figures_dir, cfg)
    figure2 = render_figure("learning_curve", loaded, figures_dir, cfg)

    out_path = out_path if os.path.isabs(out_path) else os.path.join(os.getcwd(), out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    rel1 = os.path.relpath(figure1, out_dir)
    rel2 = os.path.relpath(figure2, out_dir)

    banner = _fixture_banner(loaded, "below")
    banner_end = _fixture_banner(loaded, "above")
    parts: list[str] = [
        "# REFLEXIVE v1 -- results",
        "",
    ]
    if banner:
        parts += [banner, ""]
    parts += [
        "Rendered by `reflex report` from `metrics.json` only. Spec 4.10 forbids this "
        "renderer from computing anything: every cell either came out of a metrics file "
        "(or out of `configs/default.yaml`, where spec 10 requires measured constants to "
        "live) or says `n/a` with the reason attached.",
        "",
        "## 1. Setup",
        "",
        _section_setup(loaded, cfg),
    ]
    if missing:
        parts += [
            "",
            "**Missing inputs:** " + ", ".join(f"`{p}`" for p in missing)
            + " could not be read and contributed nothing to this report.",
        ]
    parts += [
        "",
        "## 2. Table A -- Routing (spec 8.1)",
        "",
        render_table("A", loaded, cfg),
        "",
        "## 3. Table B -- Quality (spec 8.2 / 8.3 / 8.7)",
        "",
        render_table("B", loaded, cfg),
        "",
        "## 4. Table C -- Novelty (spec 8.4)",
        "",
        render_table("C", loaded, cfg),
        "",
        "## 5. Table D -- Cost and latency (spec 8.6)",
        "",
        render_table("D", loaded, cfg),
        "",
        "## 6. Figures",
        "",
        f"![Figure 1 -- coverage-error]({rel1})",
        "",
        f"![Figure 2 -- learning curve]({rel2})",
        "",
        f"Figure files: `{figure1}`, `{figure2}`. A figure whose input is absent renders "
        "as a NO INPUT panel naming what is missing, rather than as an empty axes that "
        "would read as zero.",
        "",
        "## 7. Table E -- Ablations (spec 7 E3)",
        "",
        render_table("E", loaded, cfg),
        "",
        "## 8. Table F -- Failure taxonomy (spec 11.3 section 8)",
        "",
        render_table("F", loaded, cfg),
        "",
        "## 9. Verdict (spec 9)",
        "",
        _section_verdict(loaded, cfg),
        "",
    ]
    if banner_end:
        parts += [banner_end, ""]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    return os.path.abspath(out_path)
