"""Guard the inputs ``reflex.report`` prints, and prove all nine sections render.

Spec 4.10 forbids the reporter from computing anything, which means every number
it prints was measured somewhere else and carried here by hand. That is exactly
the arrangement in which a number goes stale without anyone noticing, so the
staleness has to be tested rather than trusted.

Complements ``tests/test_config_integrity.py`` (which guards the YAML itself)
without repeating it: nothing here re-checks parsing, duplicate keys or boolean
coercion.

Each test below corresponds to a defect that was LIVE in this build:

* ``test_bank_fidelity_belongs_to_the_bank_on_disk`` -- ``report.md`` printed
  exact-reconstruction 61.01% directly beside ``bank_hash 654d796ecc51cea1``.
  61.01% belongs to the PRE-fix bank (``98c2e937bea518d6``, 4,417 templates);
  the post-fix bank the runs actually score against measures 60.60%. This is
  DECISIONS D6 exactly -- "a baseline is only a baseline WITH its configuration
  attached" -- and it is invisible to a reader because both numbers look right.
* ``test_latency_is_reported_not_gated`` / ``test_latency_threads_are_the_priced_threads``
  -- DECISIONS D3 makes latency a budget, not a gate, and spec 8.6 prices CPU
  seconds at a thread count that must be the one they were measured at.
* ``test_report_renders_every_spec_11_3_section`` -- the renderer had never been
  executed end to end against a metrics.json of the real shape.
"""

from __future__ import annotations

import json
import os

import pytest

from reflex.config import get_dotted, load_config, repo_root

CONFIG_PATH = os.path.join(repo_root(), "configs", "default.yaml")
BANK_META = os.path.join(repo_root(), "outputs", "compile", "bank", "meta.json")

#: The token the reporter prints for a spec 9 criterion that had no input at all.
_UNMEASURABLE_WORD = "UNMEASURABLE"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG_PATH)


# --------------------------------------------------------------------------- #
# The numbers the reporter carries in from elsewhere
# --------------------------------------------------------------------------- #


def test_bank_fidelity_belongs_to_the_bank_on_disk(cfg: dict) -> None:
    """The fidelity the report prints must be measured on the bank the runs use.

    The template count is the cheap, unambiguous fingerprint of which bank a
    fidelity number came from: the D16 field-set merge guard moved it 4,417 ->
    4,505 and moved exact reconstruction 61.01% -> 60.60% at the same time.
    """
    if not os.path.exists(BANK_META):
        pytest.skip("no compiled bank on disk; run `reflex compile` first")
    with open(BANK_META, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    configured = int(get_dotted(cfg, "report.bank_templates"))
    on_disk = int(meta["n_templates"])
    assert configured == on_disk, (
        f"report.bank_templates={configured} but the bank on disk holds {on_disk} templates "
        f"(bank_hash {meta.get('bank_hash')!r}). The reporter would print a fidelity rate "
        "measured on a DIFFERENT bank than every run scores against -- DECISIONS D6. "
        "Recompile or re-measure report.bank_exact_reconstruction_rate."
    )


def test_fidelity_is_not_described_as_a_ceiling(cfg: dict) -> None:
    """DECISIONS D16 retired "61.01% is the ceiling on the reflex rate".

    Every retrieve turn is assigned templates, so the fast path always emits
    something and the number bounds nothing. Three numbers must be available to
    the reporter, not one.
    """
    for key in (
        "report.bank_assignment_rate",
        "report.bank_exact_reconstruction_rate",
        "report.bank_wrong_field_rate",
        "report.bank_wrong_field_rate_prefix",
    ):
        assert get_dotted(cfg, key) is not None, f"{key} is required to print D16's three numbers"
    assert float(get_dotted(cfg, "report.bank_assignment_rate")) == 1.0, (
        "assignment rate is 1.0 by construction; if it ever is not, the 'the fast path always "
        "emits something' claim in the report is wrong and must be re-derived."
    )


def test_strict_coverage_is_unmeasured_not_estimated(cfg: dict) -> None:
    """DEFECTS_OPEN D-7: strict coverage must render UNMEASURED until measured.

    An estimated value here would be the most damaging kind of number in the
    report -- it looks measured and is not.
    """
    value = get_dotted(cfg, "report.strict_template_coverage")
    assert value is None or isinstance(value, (int, float)), (
        "report.strict_template_coverage must be null (renders UNMEASURED) or a genuinely "
        f"measured number, not {value!r}"
    )


# --------------------------------------------------------------------------- #
# DECISIONS D3: latency is a budget, not a gate
# --------------------------------------------------------------------------- #


def test_latency_is_reported_not_gated(cfg: dict) -> None:
    """Criterion 5 is recorded as measured-and-missed and never fails the build."""
    assert get_dotted(cfg, "latency.gate_on_latency") is False, (
        "DECISIONS D3: spec 9 criterion 5 was already missed at the spec's own default "
        "(34ms at K=6) and the binding constraint was encoder size, not anything a run can "
        "change. Turning this into a gate fails the build on a number nobody chose."
    )
    assert float(get_dotted(cfg, "latency.target_p95_ms")) >= float(
        get_dotted(cfg, "verdict.max_fastpath_p95_latency_ms")
    ), "latency.target_p95_ms is the revised budget and must not be stricter than spec 9's"


def test_latency_threads_are_the_priced_threads(cfg: dict) -> None:
    """Spec 8.6 must price CPU seconds at the parallelism they were measured at."""
    assert int(get_dotted(cfg, "runtime.torch_threads")) == int(
        get_dotted(cfg, "cost.inference_threads")
    ), (
        "runtime.torch_threads != cost.inference_threads: the fast path would be timed at one "
        "thread count and billed at another, and the cost-per-turn number would be fiction."
    )


def test_inference_is_cpu_only(cfg: dict) -> None:
    """Spec 10: the fast path is CPU at inference, which is what latency assumes."""
    assert str(get_dotted(cfg, "runtime.device")) == "cpu"


# --------------------------------------------------------------------------- #
# The renderer itself
# --------------------------------------------------------------------------- #


def _fixture_metrics(run_id: str, alpha: float) -> dict:
    """A metrics.json of the exact shape ``evaluate.evaluate_run`` writes.

    Deliberately hand-built rather than produced by a run: it must stay valid
    when no checkpoint exists, which is the situation the reporter has to cope
    with today.
    """
    return {
        "run_id": run_id,
        "arm": "B",
        "model_key": "strong",
        "split": "test_seen",
        "seeds": [1],
        "alpha": alpha,
        "n_records": 120,
        "n_conversations": 25,
        "official": {
            "cds": {
                "Joint_Accuracy": 0.41,
                "Turn_Accuracy": 0.55,
                "Nextstep_Accuracy": 0.80,
                "Intent_Accuracy": 0.72,
                "Action_Accuracy": 0.60,
                "Value_Accuracy": 0.51,
                "Recall_at_1": 0.30,
                "Recall_at_5": 0.55,
                "Recall_at_10": 0.68,
                "Cascading_Score": 0.44,
            },
            "ast": {"Joint_Accuracy": 0.48, "Bslot_Accuracy": 0.62, "Value_Accuracy": 0.53},
        },
        "official_notes": {"source": "wrapped utils/evaluate.py", "kb_labels": None, "arrays": {}},
        "routing": {
            "reflex_rate": 0.52,
            "reflex_rate_by_nextstep": {
                "retrieve_utterance": 0.55,
                "take_action": 0.44,
                "end_conversation": 0.71,
            },
            "containment": 0.12,
            "escalation_reason_share": {
                "low_confidence": 0.61,
                "novel": 0.18,
                "unavailable_slot": 0.14,
                "unseen_action": 0.07,
            },
            "n_agent_turns": 120,
            "n_reflex_turns": 62,
            "n_escalated_turns": 58,
            "n_conversations": 25,
        },
        "fastpath": {
            "fastpath_error_rate": {
                "nextstep": 0.11,
                "intent": 0.19,
                "action_values": 0.33,
                "utterance_recall_at_1": 0.58,
                "turn": 0.40,
            },
            "fastpath_denominators": {
                "nextstep": 62,
                "intent": 62,
                "action_values": 18,
                "utterance_recall_at_1": 33,
                "turn": 62,
            },
            "exact_template_match_rate": 0.42,
            "n_exact_template_match_applicable": 33,
            "n_reflex_turns": 62,
        },
        "cost_latency": {
            "llm_cost_per_turn": 0.0,
            "fastpath_cost_per_turn": 4.1e-07,
            "arm_cost_per_turn": 4.1e-07,
            "arm_cost_per_conversation": 1.9e-06,
            "latency_p50_ms": 71.2,
            "latency_p95_ms": 131.5,
            "latency_p50_ms_fastpath": 69.8,
            "latency_p95_ms_fastpath": 129.4,
            "latency_p50_ms_escalated": 73.0,
            "latency_p95_ms_escalated": 133.0,
            "llm_cost_usd_total": 0.0,
            "fastpath_cost_usd_total": 4.9e-05,
            "llm_tokens_in_billed": 0,
            "llm_tokens_out_billed": 0,
            "n_llm_turns": 0,
            "n_cache_hits": 0,
            "cpu_seconds": 8.8,
            "n_turns": 120,
            "n_conversations": 25,
        },
        "log_correct_mismatch": {"n_compared": 480, "mismatches": {}, "clean": True},
        "calibration": {
            "ece": {},
            "empirical_coverage": {h: None for h in ("nextstep", "intent", "action", "skeleton")},
            "alpha": None,
            "source": "unavailable",
            "unavailable_reason": "no calibration file; run `reflex calibrate` first",
        },
        "constant_guard": {
            "constant_predictor": {
                "nextstep": "retrieve_utterance",
                "intent": "storewide_query",
                "action": None,
                "value": None,
                "utterance": "first candidate (rank 0)",
            },
            "comparisons": {
                "cds.Joint_Accuracy": {"system": 0.41, "constant": 0.02, "constant_wins": False},
                "cds.Nextstep_Accuracy": {"system": 0.80, "constant": 0.72, "constant_wins": False},
                "ast.Joint_Accuracy": {"system": 0.48, "constant": 0.50, "constant_wins": True},
            },
            "dropped": ["ast.Joint_Accuracy"],
            "flagged": True,
            "note": "A label-blind constant beating a metric means the metric carries no signal.",
        },
        "statistics": {
            "bootstrap": {
                "nextstep": {"point": 0.80, "ci_low": 0.74, "ci_high": 0.86},
                "intent": {"point": 0.72, "ci_low": 0.65, "ci_high": 0.79},
                "turn": {"point": 0.55, "ci_low": 0.48, "ci_high": 0.62},
            },
            "n_bootstrap": 1000,
            "ci": 0.95,
        },
        "verdict": {
            "verdict": "PARTIAL",
            "criteria": {
                "reflex_rate": {"passed": True, "value": 0.52, "threshold": 0.50},
                "parity_cds.Joint_Accuracy": {"passed": None, "value": None, "threshold": -1.0},
                "parity_ast.Joint_Accuracy": {"passed": None, "value": None, "threshold": -1.0},
                "parity_delta_ci_low": {"passed": None, "value": None, "threshold": -2.0},
                "novel_escalation_rate": {"passed": None, "value": None, "threshold": 0.95},
                "empirical_coverage_min": {"passed": None, "value": None, "threshold": None},
                "fastpath_p95_latency_ms": {"passed": False, "value": 129.4, "threshold": 20.0},
            },
            "unavailable": [
                "empirical_coverage_min",
                "novel_escalation_rate",
                "parity_ast.Joint_Accuracy",
                "parity_cds.Joint_Accuracy",
                "parity_delta_ci_low",
            ],
            "stop_threshold": {"reflex_rate": 0.30},
            "dropped_metrics": ["ast.Joint_Accuracy"],
            "note": "Criteria with passed=null were not measurable from this run's artifacts.",
        },
        "dropped_metrics": ["ast.Joint_Accuracy"],
    }


def test_report_renders_every_spec_11_3_section(tmp_path, cfg: dict) -> None:
    """All nine spec 11.3 sections must appear, in order, from a metrics.json alone."""
    from reflex.report import render_report

    runs_dir = tmp_path / "runs"
    run_id = "20260916T000000Z-armB-strong-test_seen-s1"
    (runs_dir / run_id).mkdir(parents=True)
    metrics_path = runs_dir / run_id / "metrics.json"
    metrics_path.write_text(json.dumps(_fixture_metrics(run_id, 0.02)), encoding="utf-8")

    local = load_config(CONFIG_PATH, [f"paths.runs_dir={runs_dir}"])
    out = tmp_path / "report.md"
    written = render_report([str(metrics_path)], str(out), local)
    text = open(written, "r", encoding="utf-8").read()

    positions = []
    for number in range(1, 10):
        marker = f"\n## {number}."
        assert marker in text, f"spec 11.3 section {number} is missing from report.md"
        positions.append(text.index(marker))
    assert positions == sorted(positions), "the nine sections are not in spec 11.3 order"


def _render(tmp_path, name: str, mutate=None) -> str:
    """Render one fixture run and return the markdown."""
    from reflex.report import render_report

    runs_dir = tmp_path / name / "runs"
    run_id = "20260916T000001Z-armB-strong-test_seen-s1"
    (runs_dir / run_id).mkdir(parents=True)
    metrics = _fixture_metrics(run_id, 0.02)
    if mutate is not None:
        mutate(metrics)
    metrics_path = runs_dir / run_id / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    local = load_config(CONFIG_PATH, [f"paths.runs_dir={runs_dir}"])
    written = render_report([str(metrics_path)], str(tmp_path / name / "report.md"), local)
    return open(written, "r", encoding="utf-8").read()


def _criterion_row(text: str, key: str) -> str:
    """The section 9 table row for one criterion key."""
    for line in text.splitlines():
        if line.startswith("|") and f"| {key} |" in line:
            return line
    raise AssertionError(f"no section 9 criteria row for {key!r}")


def test_report_marks_a_constant_beaten_metric_as_dropped(tmp_path, cfg: dict) -> None:
    """A metric a label-blind constant wins is DROPPED, not caveated.

    CARRIES ITS OWN NEGATIVE CONTROL. The first version of this test asserted
    ``"DROPPED" in text`` and passed no matter what the data said, because the
    word also appears in the table's explanatory header -- a control incapable of
    failing. It now renders the SAME fixture twice, once with the constant
    winning ``ast.Joint_Accuracy`` 0.50 to 0.48 and once with it losing, and
    requires the two renders to differ.
    """
    beaten = _render(tmp_path, "beaten")

    def constant_loses(metrics: dict) -> None:
        comparison = metrics["constant_guard"]["comparisons"]["ast.Joint_Accuracy"]
        comparison["constant"] = 0.10
        comparison["constant_wins"] = False
        metrics["constant_guard"]["dropped"] = []
        metrics["constant_guard"]["flagged"] = False
        metrics["dropped_metrics"] = []
        metrics["verdict"]["dropped_metrics"] = []

    clean = _render(tmp_path, "clean", constant_loses)

    assert "**DROPPED**" in beaten, (
        "the constant beat ast.Joint_Accuracy 0.50 to 0.48 but no metric row is marked "
        "**DROPPED**; a metric a constant wins must not be reported as a result"
    )
    assert "**DROPPED**" not in clean, (
        "NEGATIVE CONTROL FAILED: the report marks a metric DROPPED even when the constant "
        "lost, so the DROPPED marking carries no information about the data"
    )
    assert "ast.Joint_Accuracy" in beaten


def test_report_never_defaults_an_unmeasured_criterion_to_pass(tmp_path, cfg: dict) -> None:
    """Spec 9: a criterion with no input is UNMEASURABLE, never a pass.

    Also carries its own negative control: the same fixture rendered with every
    criterion supplied must NOT mark those rows UNMEASURABLE.
    """
    unmeasured = _render(tmp_path, "unmeasured")

    def everything_measured(metrics: dict) -> None:
        for criterion in metrics["verdict"]["criteria"].values():
            if criterion["passed"] is None:
                criterion["passed"] = True
                criterion["value"] = 0.99
                criterion["threshold"] = 0.50
        metrics["verdict"]["unavailable"] = []
        metrics["verdict"]["verdict"] = "PASS"

    measured = _render(tmp_path, "measured", everything_measured)

    for key in ("novel_escalation_rate", "empirical_coverage_min", "parity_delta_ci_low"):
        assert _UNMEASURABLE_WORD in _criterion_row(unmeasured, key), (
            f"criterion {key} had no input and its section 9 row does not say "
            f"{_UNMEASURABLE_WORD}; an unmeasured criterion must never read as a pass"
        )
        assert _UNMEASURABLE_WORD not in _criterion_row(measured, key), (
            f"NEGATIVE CONTROL FAILED: criterion {key} is marked {_UNMEASURABLE_WORD} even "
            "when a value was supplied, so the marking carries no information"
        )

    latency_row = _criterion_row(unmeasured, "fastpath_p95_latency_ms")
    assert "REPORT-ONLY" in latency_row.upper(), (
        "criterion 5 missed its target (129.4 ms vs 20 ms) and its row must say REPORT-ONLY "
        "while latency.gate_on_latency is false, rather than showing a bare FAIL"
    )
    assert "**FAIL**" not in latency_row, (
        "criterion 5 must never render as a hard FAIL: DECISIONS D3 makes latency a budget, "
        "and latency.gate_on_latency is false"
    )


def test_render_figure_rejects_an_unknown_name(tmp_path, cfg: dict) -> None:
    """The contract specifies ValueError on an unknown figure name."""
    from reflex.report import render_figure

    with pytest.raises(ValueError):
        render_figure("not_a_figure", [_fixture_metrics("r", 0.02)], str(tmp_path), cfg)


def test_render_table_rejects_an_unknown_name(cfg: dict) -> None:
    """The contract specifies ValueError on an unknown table name."""
    from reflex.report import render_table

    with pytest.raises(ValueError):
        render_table("Z", [_fixture_metrics("r", 0.02)], cfg)
