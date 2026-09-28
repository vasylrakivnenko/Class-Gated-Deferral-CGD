"""Unit tests for the clean selection protocol and the dense-state memory guard.

SYNTHETIC DATA ONLY. Nothing here reads the ABCD corpus, loads a compiled bank
or fits anything bigger than a few hundred rows: the machine these run on has
already lost one process to OOM and one to SIGBUS, and a test suite that needs
the corpus is a test suite nobody runs.

What these prove, in the order the protocol needs them proved:

1. The internal split is GROUPED BY CONVERSATION -- no conversation is ever on
   both sides, and every row of a conversation (H5 turn, H7 sentence position,
   H4 action) lands together.
2. The split is DETERMINISTIC under a seed and independent of row order.
3. Selection NEVER READS DEV OR TEST. Proved twice: ``sweep_internal`` has no
   dev/test parameter and works with the row loader sabotaged, and ``mode_select``
   has not loaded a single held-out split at the moment the sweep runs.
4. The dense-state guard raises a named Python error instead of letting the fit
   die in native code, and its arithmetic reproduces the projections the
   coordinator measured independently.
"""

from __future__ import annotations

import argparse

import pytest

from probes import run_response_probe as RUN
from probes.featurizer_api import RenderSpec, VectorizerSpec


# --------------------------------------------------------------------------- #
# Synthetic rows: a few conversations with a learnable signal
# --------------------------------------------------------------------------- #

_INTENTS = ("refund", "shipping", "account")
_MARKER = {"refund": "refund money back charge",
           "shipping": "shipping delivery package arrive",
           "account": "password login username reset"}


def _context(convo_id: int, turn_index: int, intent: str):
    from reflex.schemas import ContextWindow

    turns = [
        "agent|hello, how can i help you today?",
        f"customer|i need help with {_MARKER[intent]}",
        f"agent|sure, let me look into the {intent} for you",
    ][: 1 + (turn_index % 3)]
    state = f"state|disclosed: flowless || actions: {intent}"
    return ContextWindow(
        convo_id=convo_id, turn_index=turn_index,
        text="\n".join(turns + [state]), turns=list(turns),
        disclosed={}, actions_so_far=[], context_hash="",
    )


def synthetic_rows(n_convos: int = 36, turns_per_convo: int = 3) -> dict:
    """``{"h5": [...], "h7": [...], "h4": [...], "spaces": ...}`` with a real signal.

    The gold label is a function of the conversation's intent, and the intent is
    visible in the context text, so a TF-IDF + linear model can actually learn
    it. That matters: a sweep over candidates that all score 0 would rank by
    tie-break and would not exercise the ranking at all.
    """
    from probes import response_labels as RL

    h5, h7, h4 = [], [], []
    for convo_id in range(1, n_convos + 1):
        intent = _INTENTS[convo_id % len(_INTENTS)]
        for turn_index in range(turns_per_convo):
            turn_id = f"train-{convo_id}-{turn_index}"
            ctx = _context(convo_id, turn_index, intent)
            h5.append(RL.H5Row(
                turn_id=turn_id, convo_id=convo_id, turn_index=turn_index, context=ctx,
                gold_skeleton_id=f"sk_{intent}", gold_acts=("ASK",),
            ))
            h7.append(RL.H7Row(
                turn_id=turn_id, convo_id=convo_id, turn_index=turn_index, position=0,
                n_positions=1, context=ctx, act="ASK",
                gold_template_id=f"tpl_{intent}_{turn_index % 2}",
            ))
        h4.append(RL.H4Row(
            turn_id=f"train-{convo_id}-a", convo_id=convo_id, turn_index=0, context=_context(
                convo_id, 0, intent),
            action="pull-up-account", action_with_position="pull-up-account",
            gold_value=intent, n_gold_values=1,
            gold_column=_INTENTS.index(intent), candidate_tokens=tuple(_INTENTS),
        ))
    return {"h5": h5, "h7": h7, "h4": h4, "spaces": _Spaces()}


class _Spaces:
    """The two attributes the protocol reads off ``LabelSpaces``."""

    bank_hash = "synthetic"
    skeleton_acts = {f"sk_{i}": ("ASK",) for i in _INTENTS}
    value_list = list(_INTENTS)

    def summary(self) -> dict:
        return {"synthetic": True}


class _Equivalence:
    """H7's near-duplicate tiers, stubbed to exact-match only."""

    def __init__(self) -> None:
        self.ids = sorted({f"tpl_{i}_{p}" for i in _INTENTS for p in (0, 1)})
        self.act = {t: "ASK" for t in self.ids}
        self.fields = {t: () for t in self.ids}

    def tier_flags(self, pred: str, gold: str) -> dict:
        same = pred == gold
        return {"exact": same, "paraphrase": same, "field_safe": same}

    def bank_facts(self) -> dict:
        return {"synthetic": True}

    def paraphrase_density(self) -> dict:
        return {"synthetic": True}


PROBE_CFG = {
    "probe": {
        "featurizer_factory": "reflex.featurize:build_featurizer",
        "vectorizer": {"analyzer": "word", "ngram_range": [1, 2], "min_df": 1,
                       "max_df": 1.0, "sublinear_tf": False, "lowercase": True,
                       "max_features": None, "classifier": "logreg", "C": 1.0,
                       "max_iter": 200, "class_weight": None},
        "vectorizer_by_head": {"h5": {"classifier": "logreg"},
                               "h7": {"classifier": "logreg"},
                               "h4": {"classifier": "logreg"}},
        "memory_guard": {"max_dense_state_gib": 2.0, "bytes_per_coefficient": 8,
                         "state_vectors_by_classifier": {"logreg": 21, "logreg_ovr": 2,
                                                         "sgd_log": 2},
                         "on_over_budget": "raise"},
        "validate_grid": [{}, {"sublinear_tf": True}],
        "arms": [{"k": 6, "tag_recency": False},
                 {"k": 6, "tag_recency": True, "recency_buckets": 3, "recency_mode": "word"}],
        "recall_ks": [1, 3],
    }
}


def _args(**over) -> argparse.Namespace:
    base = dict(heads=["h5", "h7"], out_dir="unused", cache_dir="unused", n_boot=25,
                seed=0, shuffle_seed=17, select_frac=0.25, split_seed=13,
                select_grid="vectorizers", eval_controls=False, force_uncertified=True,
                train_convos=0, train_rows=0)
    base.update(over)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# 1 + 2. The internal split: grouped by conversation, deterministic, total
# --------------------------------------------------------------------------- #


def test_no_conversation_lands_on_both_sides():
    fit, select = RUN.split_convos_grouped(range(1, 201), 0.2, seed=13)
    assert fit & select == set()
    assert fit | select == set(range(1, 201))
    assert select, "a 20% select side of 200 conversations must not be empty"


def test_every_row_of_a_conversation_moves_together():
    rows = synthetic_rows()
    ids = {r.convo_id for r in rows["h5"]}
    fit_ids, sel_ids = RUN.split_convos_grouped(ids, 0.25, seed=13)
    fit, sel = RUN.rows_for_convos(rows, fit_ids), RUN.rows_for_convos(rows, sel_ids)
    for head in ("h5", "h7", "h4"):
        on_fit = {r.convo_id for r in fit[head]}
        on_sel = {r.convo_id for r in sel[head]}
        assert on_fit & on_sel == set(), f"{head}: a conversation is on both sides"
        assert len(fit[head]) + len(sel[head]) == len(rows[head]), f"{head}: rows lost"
    # and the SAME conversations on both heads -- an H7 position may not be split
    # away from the H5 turn it belongs to.
    assert {r.convo_id for r in fit["h5"]} == {r.convo_id for r in fit["h7"]}


def test_split_is_deterministic_under_a_seed_and_order_independent():
    ids = list(range(1, 121))
    a = RUN.split_convos_grouped(ids, 0.2, seed=7)
    b = RUN.split_convos_grouped(ids, 0.2, seed=7)
    assert a == b
    # order of the input must not matter: the ids are sorted before shuffling
    shuffled = RUN.split_convos_grouped(list(reversed(ids)), 0.2, seed=7)
    assert shuffled == a
    # and a different seed must actually give a different cut
    assert RUN.split_convos_grouped(ids, 0.2, seed=8) != a


def test_split_never_empties_a_side():
    for frac in (0.0, 1.0, 0.5):
        fit, select = RUN.split_convos_grouped(range(10), frac, seed=3)
        assert fit and select, f"frac={frac} emptied a side"


# --------------------------------------------------------------------------- #
# 3. Selection never reads dev or test
# --------------------------------------------------------------------------- #


def test_sweep_internal_has_no_dev_or_test_parameter():
    import inspect

    names = set(inspect.signature(RUN.sweep_internal).parameters)
    assert not {n for n in names if "dev" in n or "test" in n}, names
    assert "rows_train" in names


def test_sweep_runs_with_the_row_loader_sabotaged(monkeypatch):
    """If the sweep loaded another split it would explode here."""

    def _boom(*_a, **_k):
        raise AssertionError("selection loaded a split; it must only see train")

    monkeypatch.setattr(RUN, "load_split_rows", _boom)
    featurizer = RUN.load_featurizer(PROBE_CFG["probe"]["featurizer_factory"])
    rows = synthetic_rows()
    candidates = RUN.selection_candidates(PROBE_CFG, "vectorizers")
    out = RUN.sweep_internal(featurizer, rows, candidates, PROBE_CFG, ["h5", "h7"],
                             select_frac=0.25, split_seed=13,
                             guard=PROBE_CFG["probe"]["memory_guard"])
    assert out["internal_split"]["disjoint"] is True
    assert out["internal_split"]["n_convos_fit"] + out["internal_split"]["n_convos_select"] == 36
    assert len(out["sweep"]) == len(candidates)
    for head in ("h5", "h7", "compose"):
        assert head in out["winners"], f"no winner chosen for {head}"
        assert out["winners"][head]["objective_value_on_train_select"] is not None
    # the winner is reported WITH its configuration (D6)
    assert out["winners"]["h5"]["vectorizer"]["classifier"] == "logreg"
    assert "arm" in out["winners"]["h5"]


def test_mode_select_loads_no_held_out_split_before_the_sweep(monkeypatch):
    """The ordering guarantee, checked on the real ``mode_select`` control flow."""
    loaded: list = []
    at_sweep_time: list = []

    sizes = {"train": 36, "dev": 12, "test_seen": 9, "test_novel": 6}

    def _fake_load(cfg, split, bank, indexer, cache_dir):
        loaded.append(split)
        rows = synthetic_rows(n_convos=sizes[split])
        return {**rows, "label_provenance": f"synthetic-{split}"}

    real_sweep = RUN.sweep_internal

    def _recording_sweep(*a, **k):
        at_sweep_time.append(list(loaded))
        return real_sweep(*a, **k)

    monkeypatch.setattr(RUN, "load_split_rows", _fake_load)
    monkeypatch.setattr(RUN, "sweep_internal", _recording_sweep)
    monkeypatch.setattr(RUN, "_load_bank", lambda cfg: object())
    monkeypatch.setattr(RUN, "_equivalence", lambda bank, cfg: _Equivalence())
    monkeypatch.setattr(RUN, "_dev_selected_config", lambda out_dir: {"available": False})
    monkeypatch.setattr(RUN, "_read_certification", lambda out_dir: {"certified": True})

    result = RUN.mode_select(_args(), PROBE_CFG, {})

    assert at_sweep_time == [["train"]], (
        f"a held-out split was loaded before selection: {at_sweep_time}")
    assert loaded[0] == "train"
    assert loaded.index("train") < loaded.index("dev")
    assert loaded.index("train") < loaded.index("test_seen")
    assert set(loaded) == {"train", "dev", "test_seen", "test_novel"}
    # the three blocks are reported separately and the novel split is never pooled
    assert result["headline_test_seen"] is not None
    assert result["free_check_dev"] is not None
    assert result["test_novel"] is not None
    assert result["headline_test_seen"] is not result["test_novel"]
    # every headline head carries its selected configuration and its constant
    h5 = result["headline_test_seen"]["h5"]
    assert h5["selected_config"]["vectorizer"]["classifier"] == "logreg"
    assert h5["constant"]["recall@1"] is not None
    assert h5["d5"]["verdict"] in {"KEEP", "UNMEASURED"} or h5["d5"]["verdict"].startswith("DROP")
    assert h5["ci_top1"]["n_clusters"] >= 1
    # compose@1 is reported on every split, with the arm it was selected on
    compose = result["headline_test_seen"]["compose"]
    assert compose["conditional"]["compose@1"] is not None
    assert compose["conditional"]["constant"] is not None
    assert compose["selected_config"]["arm"]
    # test_novel is reported on its own rows, never pooled into test_seen
    assert (result["test_novel"]["h5"]["n_eval"]
            != result["headline_test_seen"]["h5"]["n_eval"]
            or result["test_novel"]["n_convos"] != result["headline_test_seen"]["n_convos"])


def test_an_identical_config_is_not_refitted_twice(monkeypatch):
    """compose often picks the arm H5 or H7 already picked; a refit there is a full pass."""
    fits: list = []
    real_fit_score = RUN.fit_score

    def _counting(*a, **k):
        fits.append(k.get("head", "?"))
        return real_fit_score(*a, **k)

    monkeypatch.setattr(RUN, "fit_score", _counting)
    monkeypatch.setattr(RUN, "load_split_rows",
                        lambda cfg, split, bank, indexer, cache_dir: {
                            **synthetic_rows(n_convos=36 if split == "train" else 8),
                            "label_provenance": f"synthetic-{split}"})
    monkeypatch.setattr(RUN, "_load_bank", lambda cfg: object())
    monkeypatch.setattr(RUN, "_equivalence", lambda bank, cfg: _Equivalence())
    monkeypatch.setattr(RUN, "_dev_selected_config", lambda out_dir: {"available": False})
    monkeypatch.setattr(RUN, "_read_certification", lambda out_dir: {"certified": True})

    result = RUN.mode_select(_args(select_grid="arms"), PROBE_CFG, {})
    # With one shared winning config, each split fits h5 once and h7 once -- not
    # twice each because compose asked for the same thing.
    per_split_eval = [h for h in fits if h in ("h5", "h7")]
    assert result["refits_avoided"], "the memo never fired; compose refitted a known config"
    assert len(per_split_eval) > 0


# --------------------------------------------------------------------------- #
# 4. The dense-state guard
# --------------------------------------------------------------------------- #


GUARD = PROBE_CFG["probe"]["memory_guard"]


def test_guard_arithmetic_matches_the_measured_projection():
    """1,066 classes x 218k features x 8 B x 21 L-BFGS vectors."""
    est = RUN.estimate_dense_state(1066, 218_000, "logreg", GUARD)
    assert est["projected_bytes"] == 1066 * 218_000 * 8 * 21
    assert est["over_budget"] is True
    # one-vs-rest drops the history and therefore the multiplier
    ovr = RUN.estimate_dense_state(1066, 218_000, "sgd_log", GUARD)
    assert ovr["projected_bytes"] * 10 < est["projected_bytes"]


def test_guard_raises_naming_head_classes_and_features():
    vspec = VectorizerSpec(classifier="logreg")
    with pytest.raises(RUN.DenseStateTooLarge) as excinfo:
        RUN.check_dense_state("h7", 1066, 218_000, vspec, GUARD)
    message = str(excinfo.value)
    for expected in ("h7", "1066", "218000", "max_features", "memory_guard"):
        assert expected in message, f"{expected!r} missing from the guard error: {message}"


def test_guard_reports_a_max_features_that_actually_fits():
    est = RUN.estimate_dense_state(1066, 218_000, "logreg", GUARD)
    fits = RUN.estimate_dense_state(1066, est["max_features_that_would_fit"], "logreg", GUARD)
    assert fits["over_budget"] is False


def test_guard_lets_the_certified_nextstep_shape_through():
    """3-way nextstep under lbfgs must not trip the guard -- validate depends on it."""
    est = RUN.check_dense_state("nextstep", 3, 218_000, VectorizerSpec(classifier="logreg"), GUARD)
    assert est["over_budget"] is False


def test_report_mode_does_not_raise():
    cfg = {**GUARD, "on_over_budget": "report"}
    est = RUN.check_dense_state("h7", 1066, 218_000, VectorizerSpec(classifier="logreg"), cfg)
    assert est["over_budget"] is True


def test_absent_guard_config_is_a_no_op():
    assert RUN.check_dense_state("h7", 1066, 218_000,
                                 VectorizerSpec(classifier="logreg"), None)["over_budget"] is False


# --------------------------------------------------------------------------- #
# The estimator config, and the certified path it must not disturb
# --------------------------------------------------------------------------- #


def test_certified_nextstep_estimator_is_untouched():
    from sklearn.linear_model import LogisticRegression

    model = RUN._classifier(VectorizerSpec(classifier="logreg", C=1.0, max_iter=1000))
    assert isinstance(model, LogisticRegression)
    assert model.solver == "lbfgs"
    assert model.C == 1.0


def test_head_override_wins_over_a_sweep_override():
    """A sweep entry must not be able to undo the memory-safe head setting."""
    probe_cfg = {"probe": {"vectorizer": {"classifier": "logreg", "max_features": None},
                           "vectorizer_by_head": {"h7": {"classifier": "sgd_log",
                                                         "max_features": 40000}}}}
    got = RUN.vspec_for_head(probe_cfg, "h7", {"classifier": "logreg", "max_features": None})
    assert got.classifier == "sgd_log"
    assert got.max_features == 40000
    # a head with no entry keeps the base (certified) settings
    assert RUN.vspec_for_head(probe_cfg, "nextstep").classifier == "logreg"


def test_alternative_estimators_expose_predict_proba():
    """Every metric in this harness reads the ranked list, so predict_proba is the contract."""
    import numpy as np
    from scipy import sparse

    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.random((60, 12)))
    y = np.array([f"c{i % 4}" for i in range(60)])
    for kind in ("logreg", "logreg_ovr", "sgd_log"):
        model = RUN._classifier(VectorizerSpec(classifier=kind, max_iter=200), n_fit=60)
        model.fit(X, y)
        assert model.predict_proba(X[:3]).shape == (3, 4), kind
        assert len(list(model.classes_)) == 4, kind


def test_shipped_probe_yaml_fits_its_own_budget():
    """The configuration the coordinator will actually run must pass its own guard."""
    import os

    probe_cfg = RUN.load_probe_cfg(
        os.path.join(os.path.dirname(RUN.__file__), "probe.yaml"))
    guard = RUN.guard_cfg(probe_cfg)
    shapes = {"h5": 570, "h7": 1066, "h4": 226}          # H7's widest pool is ASK
    for head, n_classes in shapes.items():
        vspec = RUN.vspec_for_head(probe_cfg, head)
        assert vspec.max_features, f"{head} has no feature cap and cannot be bounded"
        RUN.check_dense_state(head, n_classes, vspec.max_features, vspec, guard)
    # and the certified base is still lbfgs at no cap
    base = RUN.vspec_for_head(probe_cfg, "nextstep")
    assert base.classifier == "logreg" and base.max_features is None


# --------------------------------------------------------------------------- #
# fit_score's additive arguments must not change what it computes
# --------------------------------------------------------------------------- #


def test_pre_rendered_texts_give_the_same_result_as_rendering():
    featurizer = RUN.load_featurizer(PROBE_CFG["probe"]["featurizer_factory"])
    rows = synthetic_rows(n_convos=12)["h5"]
    spec = RenderSpec(k=6)
    vspec = VectorizerSpec(classifier="logreg", min_df=1, max_iter=200)
    golds = [r.gold_skeleton_id for r in rows]
    a = RUN.fit_score(featurizer, rows, golds, rows, golds, spec, spec, vspec, (1,))
    texts = [featurizer.render_context(r.context, spec) for r in rows]
    b = RUN.fit_score(featurizer, rows, golds, rows, golds, spec, spec, vspec, (1,),
                      fit_texts=texts, eval_texts=texts)
    assert a["top1"] == b["top1"]
    assert a["n_features"] == b["n_features"]


def test_candidate_grid_shapes():
    full = RUN.selection_candidates(PROBE_CFG, "full")
    assert len(full) == 2 * 2
    assert len(RUN.selection_candidates(PROBE_CFG, "vectorizers")) == 2
    assert len(RUN.selection_candidates(PROBE_CFG, "arms")) == 2
    with pytest.raises(ValueError):
        RUN.selection_candidates(PROBE_CFG, "nonsense")
