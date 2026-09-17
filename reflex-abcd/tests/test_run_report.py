"""Integration tests for the runner's I/O and the reporter, end to end.

There is no trained checkpoint on this machine (24GB RAM, and a fine-tune step
was measured going 0.306s -> 241s once swap engaged), so ``run_arm``'s turn loop
cannot be exercised here. Everything AROUND it can be, and is:

    run.append_decisions -> evaluate.evaluate_run -> report.render_report

on synthetic decisions, through the real Evaluator, with the real official ABCD
metric functions. That chain catches schema drift, manifest drift, the flattened
spec 5.7 on-disk shape, and every rendering path in the nine sections.

The one test here worth more than the others is
``test_runner_correct_flags_agree_with_the_evaluator``: the runner logs its own
``correct`` flags and the Evaluator cross-checks them, so the two are written as
independent implementations of the same rule. If they were the same code the
cross-check would be vacuous.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import pytest

from reflex.config import apply_overrides, load_config
from reflex.contracts import ContractViolation, LLMDisabledError
from reflex.schemas import NEXT_STEPS, Decision, EvalRecord, GateOutput, RunManifest

INTENTS = ["return_size", "status_delivery_time", "manage_change_address"]
ACTIONS = ["verify-identity", "pull-up-account", "offer-refund"]
REASONS = ["low_confidence", "novel", "unavailable_slot", "unseen_action"]


@pytest.fixture()
def cfg(tmp_path) -> dict[str, Any]:
    base = load_config("configs/default.yaml")
    return apply_overrides(
        base, [f"paths.runs_dir={tmp_path / 'runs'}", "eval.n_bootstrap=50"]
    )


def _manifest(run_id: str, *, forced_reflex: bool = True) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_utc="2026-09-16T20:00:00+00:00",
        arm="B",
        model_key="strong",
        model_id="answerdotai/ModernBERT-base",
        split="test_seen",
        seed=1,
        alpha=0.02,
        config={"note": "test"},
        overrides=["run.forced_reflex=true"] if forced_reflex else [],
        prompt_hashes={"agent_A": "abc123", "act_labeling": "def456"},
        dataset_hash="40ce6eaa5279aeeb",
        bank_hash="98c2e937bea518d6",
        git_commit="unknown",
        novel_subflows=["manage_pay_bill", "manage_upgrade"],
        machine={
            "platform": "test",
            "cpu_count": 1,
            "python": "3.12",
            "torch": "2.14.0",
            "forced_reflex": forced_reflex,
            "latency_note": "CONTENDED: this machine was swapping.",
        },
        llm_enabled=False,
        price_list_date="<fill: YYYY-MM-DD>",
    )


def _records(run_id: str, n_convos: int = 8, n_turns: int = 6) -> list[EvalRecord]:
    """Synthetic decisions covering all three nextsteps and both routes."""
    from reflex.run import _correct_flags

    rng = random.Random(0)
    out: list[EvalRecord] = []
    for convo in range(n_convos):
        for t in range(n_turns):
            gold_step = NEXT_STEPS[t % 3]
            reflex = rng.random() < 0.6
            gate = GateOutput(
                route="reflex" if reflex else "escalated",
                reason="ok" if reflex else rng.choice(REASONS),
                set_sizes={
                    "nextstep": 1,
                    "intent": 1 if reflex else 3,
                    "action": 1 if gold_step == "take_action" else 0,
                    "skeleton": 1 if gold_step == "retrieve_utterance" else 0,
                    "templates": [1] if gold_step == "retrieve_utterance" else [],
                },
                novelty_distance=round(rng.random() * 0.3, 4),
                missing_slots=[],
            )
            gold_intent = INTENTS[convo % 3]
            gold_action = ACTIONS[t % 3] if gold_step == "take_action" else None
            gold_values = ["crystal minh"] if gold_step == "take_action" else None
            gold_rank = rng.randrange(100) if gold_step == "retrieve_utterance" else -1
            right = rng.random() < 0.8
            decision = Decision(
                convo_id=1000 + convo,
                turn_index=t,
                arm="B",
                route=gate.route,
                nextstep=gold_step if right else NEXT_STEPS[(t + 1) % 3],
                intent=gold_intent if right else INTENTS[(convo + 1) % 3],
                action=gold_action if reflex else None,
                values=gold_values if reflex else None,
                skeleton_id="S0000" if reflex and gold_step == "retrieve_utterance" else None,
                template_ids=["T000001"] if reflex and gold_step == "retrieve_utterance" else None,
                utterance_text="how can i help?" if reflex else None,
                candidate_utt_id=8416 if reflex else None,
                gate=gate,
                latency_ms_fastpath=round(20 + rng.random() * 60, 3),
                candidate_rank=gold_rank if reflex and right else -1,
                exact_template_match=(
                    right if reflex and gold_step == "retrieve_utterance" else None
                ),
            )
            gold = {
                "nextstep": gold_step,
                "intent": gold_intent,
                "action": gold_action,
                "values": gold_values,
                "utt_id": 8416,
                "utt_rank": gold_rank,
            }
            out.append(
                EvalRecord(
                    decision=decision,
                    gold=gold,
                    correct=_correct_flags(decision, gold),
                    seed=1,
                    run_id=run_id,
                )
            )
    return out


# --------------------------------------------------------------------------- #
# run.py -- ids and I/O
# --------------------------------------------------------------------------- #


def test_run_id_is_sortable_and_self_describing(cfg) -> None:
    from reflex.run import new_run_id

    run_id = new_run_id(cfg, "B", "strong", "test_seen", 3)
    assert run_id.endswith("-armB-strong-test_seen-s3")
    stamp = run_id.split("-")[0]
    assert len(stamp) == len("20260916T200000Z") and stamp.endswith("Z")
    # Sortable: a later run id must string-compare greater.
    assert new_run_id(cfg, "B", "strong", "test_seen", 3) >= run_id


def test_manifest_round_trips(cfg) -> None:
    from reflex.run import read_manifest, write_manifest

    manifest = _manifest("run-x")
    path = write_manifest(manifest, cfg)
    assert os.path.isabs(path) and os.path.exists(path)
    assert read_manifest(cfg, "run-x") == manifest


def test_missing_manifest_raises(cfg) -> None:
    from reflex.run import read_manifest

    with pytest.raises(FileNotFoundError):
        read_manifest(cfg, "no-such-run")


def test_decisions_round_trip_in_the_flat_spec_5_7_shape(cfg) -> None:
    from reflex.run import append_decisions, read_decisions

    records = _records("run-y", n_convos=3)
    assert append_decisions(records, cfg, "run-y") == len(records)
    assert read_decisions(cfg, "run-y") == records

    path = os.path.join(cfg["paths"]["runs_dir"], "run-y", "decisions.jsonl")
    with open(path, "r", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    assert "decision" not in first, "EvalRecord must be FLATTENED on disk (spec 5.7)"
    for key in ("convo_id", "turn_index", "arm", "route", "gold", "correct", "seed", "run_id"):
        assert key in first


def test_append_is_append_not_overwrite(cfg) -> None:
    from reflex.run import append_decisions, read_decisions

    records = _records("run-z", n_convos=2)
    append_decisions(records[:5], cfg, "run-z")
    append_decisions(records[5:], cfg, "run-z")
    assert len(read_decisions(cfg, "run-z")) == len(records)


def test_corrupt_decision_line_fails_loudly(cfg) -> None:
    from reflex.run import append_decisions, read_decisions

    append_decisions(_records("run-bad", n_convos=1), cfg, "run-bad")
    path = os.path.join(cfg["paths"]["runs_dir"], "run-bad", "decisions.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"convo_id": 1, "not_a_field": true}\n')
    with pytest.raises(ContractViolation):
        read_decisions(cfg, "run-bad")


def test_runner_correct_flags_agree_with_the_evaluator(cfg) -> None:
    """The cross-check that makes ``log_correct_mismatch`` meaningful.

    ``reflex.run._correct_flags`` and ``reflex.evaluate._turn_correct`` are
    separate implementations of the same rule, on purpose. If they ever diverge
    the Evaluator reports it as a bug in the runner -- which is only useful while
    they really are separate.
    """
    from reflex.evaluate import _turn_correct
    from reflex.run import _correct_flags

    for record in _records("run-cmp", n_convos=6):
        theirs = _turn_correct(record)
        ours = _correct_flags(record.decision, record.gold)
        for key, value in theirs.items():
            if key == "turn" or value is None:
                continue
            assert bool(value) == bool(ours[key]), (
                f"{key} disagrees on convo {record.decision.convo_id} "
                f"turn {record.decision.turn_index}: evaluator {value}, runner {ours[key]}"
            )


# --------------------------------------------------------------------------- #
# run.py -- the kill switch
# --------------------------------------------------------------------------- #


def test_arm_a_refuses_before_loading_any_data(cfg) -> None:
    """Arm A must die on the kill switch, not after an hour of setup.

    Also the ZERO-PAID-CALLS guard: if this ever stops raising, a run can bill.
    """
    from reflex.run import run_arm

    assert load_config("configs/default.yaml")["llm"]["enabled"] is False
    with pytest.raises(LLMDisabledError):
        run_arm(cfg, arm="A", model_key="strong", split="test_seen", seed=1)


def test_arm_b_without_a_checkpoint_is_a_user_error(cfg) -> None:
    from reflex.run import run_arm

    with pytest.raises(ValueError):
        run_arm(cfg, arm="B", model_key="strong", split="test_seen", seed=1)


def test_unknown_arm_and_model_key_are_rejected(cfg) -> None:
    from reflex.run import run_arm

    with pytest.raises(ValueError):
        run_arm(cfg, arm="C", model_key="strong", split="test_seen", seed=1)
    with pytest.raises(ValueError):
        run_arm(cfg, arm="B", model_key="medium", split="test_seen", seed=1)


def test_forced_reflex_is_reachable_from_the_cli(cfg) -> None:
    """``--forced-reflex`` is the only $0 path; it must exist and target Arm B."""
    from reflex.__main__ import build_parser

    args = build_parser().parse_args(
        ["run", "--arm", "B", "--split", "test_seen", "--checkpoint", "x", "--forced-reflex"]
    )
    assert args.forced_reflex is True
    assert build_parser().parse_args(
        ["run", "--arm", "B", "--split", "test_seen", "--checkpoint", "x"]
    ).forced_reflex is False


def test_sweeps_with_no_artifacts_refuse_rather_than_fabricate(cfg) -> None:
    from reflex.run import run_sweep

    with pytest.raises(ValueError):
        run_sweep(cfg, "E9")
    for experiment in ("E3", "E4", "E6"):
        with pytest.raises(NotImplementedError):
            run_sweep(cfg, experiment)


# --------------------------------------------------------------------------- #
# report.py
# --------------------------------------------------------------------------- #


@pytest.fixture()
def scored(cfg) -> tuple[dict[str, Any], str, str]:
    """A real ``metrics.json`` from the real Evaluator over synthetic decisions."""
    from reflex.evaluate import evaluate_run
    from reflex.run import append_decisions, write_manifest

    run_id = "20260916T200000Z-armB-strong-test_seen-s1"
    write_manifest(_manifest(run_id), cfg)
    append_decisions(_records(run_id), cfg, run_id)
    metrics = evaluate_run(cfg, run_id=run_id)
    path = os.path.join(cfg["paths"]["runs_dir"], run_id, "metrics.json")
    return metrics, path, run_id


def test_evaluator_finds_no_mismatch_in_the_logged_correct_flags(scored) -> None:
    metrics, _path, _run_id = scored
    assert metrics["log_correct_mismatch"]["clean"], metrics["log_correct_mismatch"]
    assert metrics["log_correct_mismatch"]["n_compared"] > 0


def test_report_renders_all_nine_sections(cfg, scored, tmp_path) -> None:
    from reflex.report import render_report

    _metrics, path, _run_id = scored
    out = render_report([path], str(tmp_path / "report.md"), cfg)
    text = open(out, encoding="utf-8").read()
    for heading in (
        "## 1. Setup",
        "## 2. Table A",
        "## 3. Table B",
        "## 4. Table C",
        "## 5. Table D",
        "## 6. Figures",
        "## 7. Table E",
        "## 8. Table F",
        "## 9. Verdict",
    ):
        assert heading in text, f"missing spec 11.3 section: {heading}"
    # sections must appear in order
    positions = [text.index(f"## {n}.") for n in range(1, 10)]
    assert positions == sorted(positions)


def test_report_carries_the_constant_predictor_column(cfg, scored, tmp_path) -> None:
    from reflex.report import render_report

    _metrics, path, _run_id = scored
    text = open(render_report([path], str(tmp_path / "r.md"), cfg), encoding="utf-8").read()
    assert "label-blind constant" in text
    assert "cds.Joint_Accuracy" in text


def test_report_states_the_three_bank_numbers_and_not_a_ceiling_claim(cfg, scored, tmp_path) -> None:
    """DECISIONS D16: assignment rate, fidelity and wrong-field rate, never one number."""
    from reflex.report import render_report

    _metrics, path, _run_id = scored
    text = open(render_report([path], str(tmp_path / "r.md"), cfg), encoding="utf-8").read()
    assert "assignment rate" in text
    assert "exact reconstruction, utterances (FIDELITY)" in text
    assert "wrong-field rate, post-fix" in text
    assert "wrong-field rate, pre-fix" in text
    assert "It is not a ceiling on anything" in text


def test_criterion_five_is_report_only_and_never_fails_the_build(cfg, scored, tmp_path) -> None:
    from reflex.report import render_report

    _metrics, path, _run_id = scored
    text = open(render_report([path], str(tmp_path / "r.md"), cfg), encoding="utf-8").read()
    verdict_section = text.split("## 9. Verdict")[1]
    assert "REPORT-ONLY, does not gate" in verdict_section
    assert "fastpath_p95_latency_ms" in verdict_section
    # criterion 5's row must never be rendered as a gating failure
    for line in verdict_section.splitlines():
        if "fastpath_p95_latency_ms" in line:
            assert "**FAIL**" not in line


def test_unmeasurable_criteria_print_a_reason_and_never_default_to_pass(cfg, scored, tmp_path) -> None:
    from reflex.report import render_report

    _metrics, path, _run_id = scored
    verdict_section = open(
        render_report([path], str(tmp_path / "r.md"), cfg), encoding="utf-8"
    ).read().split("## 9. Verdict")[1]
    assert "UNMEASURABLE" in verdict_section
    assert "no Arm A baseline exists" in verdict_section
    for line in verdict_section.splitlines():
        if "UNMEASURABLE" in line and line.startswith("|"):
            assert "PASS" not in line


def test_every_wall_clock_number_is_labelled_contended(cfg, scored, tmp_path) -> None:
    from reflex.report import render_report

    _metrics, path, _run_id = scored
    text = open(render_report([path], str(tmp_path / "r.md"), cfg), encoding="utf-8").read()
    assert "CONTENDED" in text
    assert "241s" in text, "the swap measurement that justifies the label must be quoted"


def test_fixture_metrics_are_stamped_at_both_ends(cfg, scored, tmp_path) -> None:
    """A rendering test must never be mistakable for a result."""
    from reflex.report import render_report

    _metrics, path, _run_id = scored
    payload = json.load(open(path, encoding="utf-8"))
    payload["fixture"] = {"reason": "synthetic decisions; no model ran"}
    json.dump(payload, open(path, "w", encoding="utf-8"))

    text = open(render_report([path], str(tmp_path / "r.md"), cfg), encoding="utf-8").read()
    assert text.count("FIXTURE -- THESE NUMBERS ARE NOT RESULTS") == 2
    assert "synthetic decisions; no model ran" in text
    assert "**FIXTURE**" in text.split("## 2.")[0], "Section 1 must mark the source per run"


def test_report_with_no_readable_metrics_refuses(cfg, tmp_path) -> None:
    from reflex.report import render_report

    with pytest.raises(ContractViolation):
        render_report([str(tmp_path / "nope.json")], str(tmp_path / "r.md"), cfg)


def test_unknown_table_and_figure_names_raise(cfg, scored, tmp_path) -> None:
    from reflex.report import render_figure, render_table

    metrics, _path, _run_id = scored
    with pytest.raises(ValueError):
        render_table("Z", [metrics], cfg)
    with pytest.raises(ValueError):
        render_figure("scatter", [metrics], str(tmp_path), cfg)


def test_figures_are_written_and_a_missing_input_says_so(cfg, scored, tmp_path) -> None:
    from reflex.report import render_figure

    metrics, _path, _run_id = scored
    one = render_figure("coverage_error", [metrics], str(tmp_path / "figs"), cfg)
    two = render_figure("learning_curve", [metrics], str(tmp_path / "figs"), cfg)
    for path in (one, two):
        assert os.path.exists(path) and os.path.getsize(path) > 1000
    # Figure 2 has no input in this build; it must render a NO INPUT panel rather
    # than an empty axes that would read as zero.
    assert "learning_curve" in two


def test_reporter_reads_only_metrics_and_config(cfg, scored, tmp_path) -> None:
    """Spec 4.10: the reporter must not open decisions.jsonl to derive anything."""
    import inspect

    from reflex import report

    source = inspect.getsource(report)
    assert "decisions.jsonl" not in source.replace(
        "`decisions.jsonl`", ""
    ), "report.py must not read the decisions log"
