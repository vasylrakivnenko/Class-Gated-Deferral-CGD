"""Gate behaviour for src/downshift/classpipe.py.

Each test pins a gate to the mistake it exists to prevent. All synthetic, no
network, no fitted models -- the gates are pure functions over predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from downshift import classpipe as CP  # noqa: E402

ROOT_CLASSPIPE = Path(__file__).resolve().parent.parent / "src" / "downshift" / "classpipe.py"


def _imbalanced(n=1000, pos_rate=0.11, seed=0):
    rng = np.random.default_rng(seed)
    gold = (rng.random(n) < pos_rate).astype(int)
    return gold


# --- metrics ---------------------------------------------------------------
def test_balanced_accuracy_punishes_the_constant_predictor():
    gold = _imbalanced()
    const = np.zeros(len(gold), int)
    assert (const == gold).mean() > 0.85          # looks great on accuracy
    assert CP.balanced_accuracy(const, gold) == pytest.approx(0.5)


def test_per_class_accuracy_is_conditioned_on_the_true_class():
    gold = np.array([0, 0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1, 0])
    pc = CP.per_class_accuracy(pred, gold)
    assert pc[0] == pytest.approx(2 / 3)
    assert pc[1] == pytest.approx(1 / 2)


def test_resolution_floor_is_set_by_the_rarest_class_not_n():
    """The mistake: quoting n=10,703 when 1,196 minority items set the precision."""
    big_balanced = np.array([0] * 5000 + [1] * 5000)
    same_n_imbalanced = np.array([0] * 9500 + [1] * 500)
    assert CP.resolution_floor(same_n_imbalanced) > 3 * CP.resolution_floor(big_balanced)


# --- G1 / G2 ---------------------------------------------------------------
def test_g1_forces_balanced_accuracy_when_a_constant_would_win():
    gold = _imbalanced()
    p = CP.profile([f"t{i}" for i in range(len(gold))], gold)
    assert p.headline_metric == "balanced_accuracy"
    assert any("G1" in n for n in p.notes)
    assert p.majority_baseline == pytest.approx(1 - gold.mean())


def test_g1_leaves_accuracy_alone_on_a_balanced_dataset():
    gold = np.array([0, 1] * 500)
    p = CP.profile([f"t{i}" for i in range(1000)], gold)
    assert p.headline_metric == "accuracy"
    assert not any("G1" in n for n in p.notes)


def test_g2_counts_every_row_in_a_duplicate_group():
    gold = np.array([0, 1, 0, 1])
    p = CP.profile(["a", "a", "b", "c"], gold)
    assert p.duplicate_rows == 2                  # both copies of "a", not one
    assert any("G2" in n for n in p.notes)


# --- G3 --------------------------------------------------------------------
def test_g3_detects_a_total_roc_order():
    """A strictly ordered pool has no complementarity -- selection cannot help."""
    rng = np.random.default_rng(0)
    gold = (rng.random(2000) < 0.3).astype(int)
    strong = np.clip(gold + rng.normal(0, 0.35, 2000), 0, 1)
    weak = np.clip(gold + rng.normal(0, 0.90, 2000), 0, 1)
    out = CP.roc_dominance({"strong": strong, "weak": weak}, gold)
    assert out["is_total_order"] is True
    assert "G3 FAIL" in out["verdict"]


def test_g3_passes_when_models_are_genuinely_complementary():
    """ROC curves that genuinely CROSS. A is flawless at the very top of its
    ranking and useless below it; B puts a few negatives at the top but separates
    far better overall. A wins at a 10% positive rate, B wins at 50%, so neither
    dominates and there is real complementarity to select between."""
    gold = np.array([0] * 500 + [1] * 500)
    a, b = np.zeros(1000), np.zeros(1000)
    a[500:750] = np.linspace(0.95, 1.00, 250)     # 250 positives ranked top
    a[:500] = np.linspace(0.40, 0.90, 500)        # every negative next
    a[750:] = np.linspace(0.00, 0.30, 250)        # the other positives, buried
    b[:50] = np.linspace(0.97, 0.99, 50)          # 50 negatives wrongly at the top
    b[50:500] = np.linspace(0.00, 0.50, 450)      # the rest of the negatives, low
    b[500:] = np.linspace(0.51, 0.95, 500)        # all positives cleanly above them

    out = CP.roc_dominance({"a": a, "b": b}, gold, rates=(0.10, 0.50))
    assert out["is_total_order"] is False
    assert "G3 pass" in out["verdict"]

    # and the crossing is real, not an artifact of the dominance bookkeeping
    top10 = lambda s: CP.per_class_accuracy((s >= np.sort(s)[-100]).astype(int), gold)
    half = lambda s: CP.per_class_accuracy((s >= np.sort(s)[-500]).astype(int), gold)
    assert top10(a)[1] > top10(b)[1]              # A better on positives at 10%
    assert half(b)[1] > half(a)[1]                # B better on positives at 50%


def test_g3_is_skipped_for_multiclass():
    gold = np.array([0, 1, 2] * 50)
    out = CP.roc_dominance({"m": np.random.default_rng(0).random(150)}, gold)
    assert out["applicable"] is False


# --- G4 --------------------------------------------------------------------
def test_g4_flags_a_pool_of_near_duplicates():
    rng = np.random.default_rng(0)
    gold = (rng.random(2000) < 0.4).astype(int)
    base = np.where(rng.random(2000) < 0.85, gold, 1 - gold)
    pool = {f"m{k}": np.where(rng.random(2000) < 0.02, 1 - base, base) for k in range(5)}
    out = CP.pool_diversity(pool, gold)
    assert out["mean_pairwise_disagreement"] < 0.10
    assert "G4 FAIL" in out["verdict"]
    # a high oracle means little when independence would give ~1.0
    assert out["oracle_if_independent"] > out["oracle_balanced"]


# --- G5 --------------------------------------------------------------------
def test_g5_flags_flat_per_class_advantage():
    """Flat advantage => the rule can only escalate everything or nothing."""
    rng = np.random.default_rng(0)
    gold = (rng.random(2000) < 0.5).astype(int)
    cheap = np.where(rng.random(2000) < 0.70, gold, 1 - gold)
    expert = np.where(rng.random(2000) < 0.80, gold, 1 - gold)
    out = CP.class_advantage(cheap, expert, gold)
    assert out["spread"] < 0.05
    assert "G5 FAIL" in out["verdict"]


def test_g5_reports_the_flip_when_the_cheap_arm_owns_a_class():
    """The regime the class-aware gate exists for: the cheap arm is better on one
    predicted class and much worse on the other."""
    # cheap says 0 on 0..799 (760 truly 0) and 1 on 800..999 (60 truly 1)
    gold = np.concatenate([np.zeros(760, int), np.ones(40, int),
                           np.zeros(140, int), np.ones(60, int)])
    cheap = np.concatenate([np.zeros(800, int), np.ones(200, int)])
    expert = gold.copy()
    expert[:240] = 1 - expert[:240]        # expert wrong on 240 of cheap's class-0 block
    expert[800:820] = 1 - expert[800:820]  # expert wrong on 20 of cheap's class-1 block
    out = CP.class_advantage(cheap, expert, gold)
    assert out["per_class"][0]["advantage"] < 0      # cheap owns the class it predicts 0 on
    assert out["per_class"][1]["advantage"] > 0      # expert owns the other
    assert out["classes_cheap_wins"] == 1
    assert out["spread"] >= 0.05
    assert "G5 pass" in out["verdict"]


# --- guarantee -------------------------------------------------------------
def test_guaranteed_coverage_separates_declining_a_class_from_breaking_the_bar():
    """Two very different outcomes that a single boolean would conflate:
    declining to serve a class at all (the guarantee working) versus serving it
    below the promised precision (the guarantee broken)."""
    rng = np.random.default_rng(0)
    n = 1200
    gold = (rng.random(n) < 0.06).astype(int)
    pred = np.where(rng.random(n) < 0.5, gold, 1 - gold)
    conf = rng.random(n) * 0.5 + 0.5
    expert = np.where(rng.random(n) < 0.8, gold, 1 - gold)
    folds = [(np.arange(0, 600), np.arange(600, n)), (np.arange(600, n), np.arange(0, 600))]
    out = CP.guaranteed_coverage(pred, conf, gold, expert, 0.95, folds)
    assert out["bar"] == 0.95
    # the rare class cannot be served at 0.95, so it is deferred entirely --
    # reported as vacuous, not silently counted as a satisfied guarantee
    assert 1 in out["vacuous_classes"]
    assert np.isnan(out["per_class"][1]["realized_precision"])
    assert out["guarantee_holds"] is True          # every class it DID serve met the bar
    assert 0.0 < out["free_share"] < 1.0


def test_guarantee_is_false_when_a_served_class_misses_the_bar():
    gold = np.array([0] * 900 + [1] * 100)
    pred = np.concatenate([np.zeros(880, int), np.ones(20, int),
                           np.zeros(40, int), np.ones(60, int)])
    conf = np.full(1000, 0.99)                     # everything looks confident
    expert = gold.copy()
    folds = [(np.arange(0, 500), np.arange(500, 1000)),
             (np.arange(500, 1000), np.arange(0, 500))]
    out = CP.guaranteed_coverage(pred, conf, gold, expert, 0.99, folds, min_support=5)
    served = [v for v in out["per_class"].values() if v["owned_share"] > 0]
    assert served, "expected at least one class to be served"
    if any(v["realized_precision"] < 0.98 for v in served):
        assert out["guarantee_holds"] is False


# --- candidate comparison --------------------------------------------------
def test_compare_candidates_holm_corrects_the_challenger_family():
    rng = np.random.default_rng(0)
    gold = (rng.random(1500) < 0.5).astype(int)
    pool = {f"m{k}": np.where(rng.random(1500) < 0.75, gold, 1 - gold) for k in range(6)}
    out = CP.compare_candidates(pool, gold)
    assert out["n_candidates"] == 6
    assert out["holm_alpha"] == pytest.approx(0.05 / 5)
    assert out["ranking"][0]["name"] == out["best"]
    assert all("significant_after_holm" in v for v in out["vs_best"].values())


# --- G7 --------------------------------------------------------------------
def test_g7_flags_a_class_whose_vote_is_usually_wrong():
    """The failure G3/G4/G5 all missed: on a 1:7.95 split a 'hate' vote was right
    41% of the time, so every reliability-based rule drifted to the majority."""
    gold = np.array([0] * 900 + [1] * 100)
    pred = gold.copy()
    pred[:150] = 1                      # 150 false positives swamp the 100 true ones
    out = CP.base_rate_dominance(pred, gold)
    assert out["min_precision"] < 0.5
    assert "G7 FAIL" in out["verdict"]


def test_g7_passes_when_every_class_vote_is_better_than_a_coin_flip():
    gold = np.array([0] * 500 + [1] * 500)
    pred = gold.copy()
    pred[:50] = 1
    pred[500:560] = 0
    out = CP.base_rate_dominance(pred, gold)
    assert out["min_precision"] > 0.5
    assert "G7 pass" in out["verdict"]


def test_g7_is_about_precision_not_recall():
    """A class can have perfect recall and still be untrustworthy to a selector."""
    gold = np.array([0] * 950 + [1] * 50)
    pred = np.concatenate([np.ones(200, int), np.zeros(750, int), np.ones(50, int)])
    assert CP.per_class_accuracy(pred, gold)[1] == pytest.approx(1.0)   # perfect recall
    assert CP.base_rate_dominance(pred, gold)["min_precision"] < 0.3    # useless vote


# --- successive halving ----------------------------------------------------
def test_stratified_rung_preserves_class_balance_and_keeps_groups_whole():
    gold = np.array([0] * 900 + [1] * 100)
    groups = np.repeat(np.arange(500), 2)          # every group has exactly 2 items
    idx = CP.stratified_rung(gold, groups, 0.2, seed=0)
    assert 0.15 < len(idx) / len(gold) < 0.30
    # minority share is roughly preserved, not wiped out
    assert 0.05 < gold[idx].mean() < 0.20
    # no group is half-in, half-out
    taken = set(idx.tolist())
    for g in set(groups[idx].tolist()):
        assert all(i in taken for i in np.where(groups == g)[0])


def test_stratified_rung_is_a_noop_at_full_fraction():
    gold = np.array([0, 1] * 50)
    groups = np.arange(100)
    assert len(CP.stratified_rung(gold, groups, 1.0)) == 100


def test_eliminate_drops_only_the_clearly_beaten():
    """Two candidates drawn at the SAME accuracy are a real tie and must both
    survive; one far below must not."""
    rng = np.random.default_rng(0)
    gold = (rng.random(3000) < 0.5).astype(int)
    strong = np.where(rng.random(3000) < 0.90, gold, 1 - gold)
    tie = np.where(rng.random(3000) < 0.90, gold, 1 - gold)     # same rate, new draw
    hopeless = np.where(rng.random(3000) < 0.55, gold, 1 - gold)
    out = CP.eliminate({"strong": strong, "tie": tie, "hopeless": hopeless}, gold)
    assert "hopeless" not in out["survivors"]
    assert out["detail"]["hopeless"]["ci"][1] < 0
    assert {"strong", "tie"} == set(out["survivors"])            # the tie is kept
    assert out["n_out"] == 2


def test_eliminate_keeps_everything_when_nothing_is_separable():
    """The rung cannot see a difference => it must not invent one."""
    rng = np.random.default_rng(1)
    gold = (rng.random(600) < 0.5).astype(int)
    base = np.where(rng.random(600) < 0.80, gold, 1 - gold)
    # each member differs from the base on five items: unmistakably a tie
    pool = {}
    for k in range(4):
        v = base.copy()
        v[rng.choice(600, 5, replace=False)] ^= 1
        pool[f"m{k}"] = v
    out = CP.eliminate(pool, gold)
    assert out["n_out"] == out["n_in"] == 4


def test_eliminate_reports_why_each_candidate_lived_or_died():
    rng = np.random.default_rng(2)
    gold = (rng.random(2000) < 0.5).astype(int)
    good = np.where(rng.random(2000) < 0.88, gold, 1 - gold)
    bad = np.where(rng.random(2000) < 0.52, gold, 1 - gold)
    out = CP.eliminate({"good": good, "bad": bad}, gold)
    assert out["detail"]["good"]["reason"] == "leader"
    assert "entirely below" in out["detail"]["bad"]["reason"]


# --- shortlist -------------------------------------------------------------
def test_label_shortlist_returns_the_nearest_labels_in_order():
    text = np.eye(3)[[0, 1, 2]].astype(float)          # three items, one-hot
    labels = np.eye(3).astype(float)                   # three labels, one-hot
    out = CP.label_shortlist(text, labels, 2)
    assert out.shape == (3, 2)
    assert list(out[:, 0]) == [0, 1, 2]                # nearest label is its own axis


def test_label_shortlist_is_a_noop_when_m_exceeds_the_label_count():
    text = np.eye(4)[:, :3].astype(float)
    labels = np.eye(3).astype(float)
    out = CP.label_shortlist(text, labels, 99)
    assert out.shape == (4, 3)


def test_label_shortlist_cuts_the_pair_count_at_large_k():
    """The whole point: cost stops scaling with the number of classes."""
    rng = np.random.default_rng(0)
    n, k, m = 500, 150, 5
    text = rng.normal(size=(n, 16)); text /= np.linalg.norm(text, axis=1, keepdims=True)
    labels = rng.normal(size=(k, 16)); labels /= np.linalg.norm(labels, axis=1, keepdims=True)
    out = CP.label_shortlist(text, labels, m)
    assert out.shape == (n, m)
    assert out.size == n * m < n * k / 20              # 150 -> 5 is a 30x cut
    assert all(len(set(row.tolist())) == m for row in out)   # no duplicate labels


# --- class-aware layer -----------------------------------------------------
def _toy_layer_inputs(n=1200, k=7, seed=0):
    """A multiclass pool: this is where passing max-probability instead of the
    full probability vector silently lands stacking at chance."""
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, k, n)
    pred, conf, proba = {}, {}, {}
    for name, acc in (("good", 0.70), ("mid", 0.55), ("weak", 0.40)):
        p = np.where(rng.random(n) < acc, gold, rng.integers(0, k, n))
        P = np.full((n, k), (1 - 0.8) / (k - 1))
        P[np.arange(n), p] = 0.8
        pred[name], conf[name], proba[name] = p, P.max(axis=1), P
    folds = [(np.arange(0, 600), np.arange(600, n)), (np.arange(600, n), np.arange(0, 600))]
    return pred, conf, proba, gold, folds


def test_class_aware_layer_stacking_is_not_at_chance_on_multiclass():
    """Regression: stacking was fed max-probability scalars, so on 7 classes it
    knew how sure each model was but not which class it picked, and scored ~1/k."""
    pred, conf, proba, gold, folds = _toy_layer_inputs(k=7)
    out = CP.class_aware_layer(pred, conf, proba, gold, folds)
    assert out["stacking"]["balanced"] > 0.4          # chance is 1/7 = 0.143
    assert out["best_single"] == "good"


def test_class_aware_layer_compares_against_the_best_single_not_the_worst():
    pred, conf, proba, gold, folds = _toy_layer_inputs()
    out = CP.class_aware_layer(pred, conf, proba, gold, folds)
    assert out["best_single_balanced"] == pytest.approx(
        CP.balanced_accuracy(pred["good"], gold))
    for k in ("selection_dcs_lca", "stacking"):
        assert "significant_after_holm" in out[k]
        assert out[k]["delta_vs_best_single"] == pytest.approx(
            out[k]["balanced"] - out["best_single_balanced"])


def test_class_aware_layer_says_ship_the_single_model_when_nothing_wins():
    pred, conf, proba, gold, folds = _toy_layer_inputs()
    out = CP.class_aware_layer(pred, conf, proba, gold, folds)
    if all(out[k]["delta_vs_best_single"] <= 0 for k in ("selection_dcs_lca", "stacking")):
        assert "ship the single model" in out["verdict"]


def test_layer_hands_the_winning_construction_downstream():
    """The free arm compared against the paid expert must be whatever actually
    won, or the pipeline under-reports its own result."""
    pred, conf, proba, gold, folds = _toy_layer_inputs(k=7)
    out = CP.class_aware_layer(pred, conf, proba, gold, folds)
    assert len(out["winner_pred"]) == len(gold)
    assert len(out["winner_conf"]) == len(gold)
    if out["winner_is_construction"]:
        assert out["winner"] in ("selection_dcs_lca", "stacking")
        assert CP.balanced_accuracy(out["winner_pred"], gold) == pytest.approx(
            out[out["winner"]]["balanced"])
    else:
        assert out["winner"] == out["best_single"]
        assert CP.balanced_accuracy(out["winner_pred"], gold) == pytest.approx(
            out["best_single_balanced"])


def test_layer_falls_back_to_the_single_model_when_no_construction_wins():
    """A construction that ties must not be shipped over the simpler model."""
    rng = np.random.default_rng(3)
    n, k = 800, 3
    gold = rng.integers(0, k, n)
    # one strong model and two copies of it: nothing to combine
    base = np.where(rng.random(n) < 0.75, gold, rng.integers(0, k, n))
    pred, conf, proba = {}, {}, {}
    for nm in ("a", "b", "c"):
        P = np.full((n, k), 0.1); P[np.arange(n), base] = 0.8
        pred[nm], conf[nm], proba[nm] = base.copy(), P.max(axis=1), P
    folds = [(np.arange(0, 400), np.arange(400, n)), (np.arange(400, n), np.arange(0, 400))]
    out = CP.class_aware_layer(pred, conf, proba, gold, folds)
    assert out["winner_is_construction"] is False
    assert "ship the single model" in out["verdict"]


# --- deferral dial ---------------------------------------------------------
def _dial_inputs(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    gold = (rng.random(n) < 0.5).astype(int)
    conf = rng.random(n) * 0.5 + 0.5
    # the free arm is wrong mostly where it is unconfident -- so the dial should work
    free = np.where(rng.random(n) < conf, gold, 1 - gold)
    expert = np.where(rng.random(n) < 0.90, gold, 1 - gold)
    folds = [(np.arange(0, 1000), np.arange(1000, n)), (np.arange(1000, n), np.arange(0, 1000))]
    return free, conf, expert, gold, folds


def test_deferral_curve_ends_are_the_two_pure_systems():
    free, conf, expert, gold, folds = _dial_inputs()
    d = CP.deferral_curve(free, conf, expert, gold, folds)
    first, last = d["points"][0], d["points"][-1]
    assert first["confidence"]["expert_share"] == pytest.approx(0.0)
    assert first["confidence"]["balanced"] == pytest.approx(d["free_alone"])
    assert last["confidence"]["expert_share"] == pytest.approx(1.0)
    assert last["confidence"]["balanced"] == pytest.approx(d["expert_alone"])


def test_deferral_curve_improves_when_the_expert_is_better():
    """Sending the free arm's least-confident traffic to a stronger expert must
    beat sending none of it."""
    free, conf, expert, gold, folds = _dial_inputs()
    d = CP.deferral_curve(free, conf, expert, gold, folds)
    at20 = next(r for r in d["points"] if r["expert_share_target"] == 0.20)
    assert at20["confidence"]["balanced"] > d["free_alone"]
    assert d["best_point"]["balanced"] >= d["free_alone"]


def test_deferral_curve_reports_marginal_gain_per_unit_spend():
    free, conf, expert, gold, folds = _dial_inputs()
    d = CP.deferral_curve(free, conf, expert, gold, folds)
    assert d["points"][0]["gain_per_10pct_spend"] is None       # nothing before 0%
    assert any(r["gain_per_10pct_spend"] is not None for r in d["points"][1:])


def test_deferral_curve_spends_budget_only_where_the_expert_helps():
    """class_aware must not escalate a class the expert is worse on, even with
    budget left over."""
    n = 2000
    gold = np.array([0] * 1000 + [1] * 1000)
    free = gold.copy(); free[:100] = 1                 # free arm near-perfect
    expert = gold.copy(); expert[1000:1600] = 0        # expert bad on class 1
    conf = np.linspace(0.5, 1.0, n)
    folds = [(np.arange(0, 1000), np.arange(1000, n)), (np.arange(1000, n), np.arange(0, 1000))]
    d = CP.deferral_curve(free, conf, expert, gold, folds, fractions=(0.5,))
    assert d["points"][0]["class_aware"]["expert_share"] < 0.5


# --- pair-scoring cost guard -----------------------------------------------
def test_pair_scoring_plan_allows_small_short_datasets():
    plan = CP.pair_scoring_plan(n=10_703, k=2, mean_chars=100)      # HateSpeech
    assert plan["run"] is True
    assert plan["shortlist_m"] is None


def test_pair_scoring_plan_skips_long_documents():
    """IMDB: only 50k pairs, but 8.3 hours of them, and both pair scorers then
    lost to the frozen bi-encoder."""
    plan = CP.pair_scoring_plan(n=25_000, k=2, mean_chars=1_300)
    assert plan["run"] is False
    assert "over" in plan["reason"]


def test_pair_scoring_plan_shortlists_instead_of_giving_up_at_large_k():
    plan = CP.pair_scoring_plan(n=22_500, k=150, mean_chars=80)     # CLINC150-ish
    assert plan["shortlist_m"] == 5
    assert plan["n_pairs"] == 22_500 * 5                            # not 22_500 * 150
    assert plan["run"] is True


def test_pair_scoring_plan_costs_tokens_not_pairs():
    short = CP.pair_scoring_plan(n=50_000, k=2, mean_chars=40)
    long_ = CP.pair_scoring_plan(n=50_000, k=2, mean_chars=4_000)
    assert short["n_pairs"] == long_["n_pairs"]
    assert long_["token_units"] > short["token_units"] * 10


def test_g5_verdict_is_scoped_to_the_selection_layer():
    """FEVER fails G5 and then wins via class-aware ALLOCATION on the dial, so
    the verdict must not read as a blanket dismissal."""
    rng = np.random.default_rng(0)
    gold = (rng.random(2000) < 0.5).astype(int)
    cheap = np.where(rng.random(2000) < 0.70, gold, 1 - gold)
    expert = np.where(rng.random(2000) < 0.80, gold, 1 - gold)
    out = CP.class_advantage(cheap, expert, gold)
    assert out["gates"] == "class-aware selection layer only"
    assert "SELECTION" in out["verdict"] and "allocation" in out["verdict"]


# --- nested selection of the operating point -------------------------------
def test_nested_best_is_not_above_the_curve_max():
    """The nested estimate must not exceed the optimistic one -- if it does, the
    nesting is leaking."""
    free, conf, expert, gold, folds = _dial_inputs()
    d = CP.deferral_curve(free, conf, expert, gold, folds)
    assert d["best_point"]["balanced"] <= d["best_point_optimistic"]["balanced"] + 1e-9
    assert d["selection_bias"] >= -1e-9


def test_nested_best_reports_what_it_picked_per_fold():
    free, conf, expert, gold, folds = _dial_inputs()
    d = CP.deferral_curve(free, conf, expert, gold, folds)
    bp = d["best_point"]
    assert len(bp["picks_per_fold"]) == len(folds)
    for pick in bp["picks_per_fold"]:
        assert pick["rule"] in CP.RULES
        assert 0.0 <= pick["budget"] <= 1.0
    assert isinstance(bp["stable"], bool)


def test_optimistic_point_declares_how_many_configurations_it_searched():
    free, conf, expert, gold, folds = _dial_inputs()
    d = CP.deferral_curve(free, conf, expert, gold, folds)
    # the count must track RULES, not a literal -- the selection bias it warns
    # about grows with every rule added to the search
    assert d["best_point_optimistic"]["n_configurations_selected_over"] == len(CP.RULES) * 8
    assert len(CP.RULES) >= 3


def test_allocate_fits_thresholds_only_on_the_training_side():
    """Swapping the held-out labels must not change the escalation choice."""
    free, conf, expert, gold, folds = _dial_inputs()
    tr, te = folds[0]
    a = CP._allocate(tr, te, "confidence", 0.2, free, conf, expert, gold)
    poisoned = gold.copy()
    poisoned[te] = 1 - poisoned[te]
    b = CP._allocate(tr, te, "confidence", 0.2, free, conf, expert, poisoned)
    assert np.array_equal(a, b)


# --- G6 enforcement --------------------------------------------------------
def test_g6_flags_a_difference_the_data_cannot_see():
    """The gate was documented from the start and never enforced, which is how
    gains of +0.0005 got reported as wins against a +/-0.0057 floor."""
    gold = np.array([0] * 3000 + [1] * 3000)
    floor = CP.resolution_floor(gold)
    small = CP.resolvable(floor / 10, gold)
    assert small["resolvable"] is False
    assert "not a difference" in small["note"]
    big = CP.resolvable(floor * 3, gold)
    assert big["resolvable"] is True and big["note"] == ""


def test_g6_floor_is_symmetric_in_sign():
    gold = np.array([0] * 500 + [1] * 500)
    f = CP.resolution_floor(gold)
    assert CP.resolvable(-f * 2, gold)["resolvable"] is True
    assert CP.resolvable(-f / 2, gold)["resolvable"] is False


def test_deferral_curve_refuses_to_claim_an_unresolvable_interior_optimum():
    """If the best point only ties the expert, beats_both_ends must be False."""
    rng = np.random.default_rng(0)
    n = 2000
    gold = (rng.random(n) < 0.5).astype(int)
    expert = np.where(rng.random(n) < 0.95, gold, 1 - gold)      # expert dominates
    free = np.where(rng.random(n) < 0.60, gold, 1 - gold)        # free arm weak
    conf = rng.random(n)
    folds = [(np.arange(0, 1000), np.arange(1000, n)), (np.arange(1000, n), np.arange(0, 1000))]
    d = CP.deferral_curve(free, conf, expert, gold, folds)
    assert d["beats_both_ends"] is False
    assert d["gain_over_expert"]["delta"] < d["gain_over_free"]["delta"]


def test_layer_will_not_ship_a_construction_inside_the_noise_floor():
    pred, conf, proba, gold, folds = _toy_layer_inputs(k=7)
    out = CP.class_aware_layer(pred, conf, proba, gold, folds)
    for k in ("selection_dcs_lca", "stacking"):
        assert "resolution" in out[k]
    if out["winner_is_construction"]:
        assert out[out["winner"]]["resolution"]["resolvable"] is True


# --- competence board / class router ---------------------------------------
def _board_folds(n, gold, k=3, seed=0):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(
        np.zeros((n, 1)), gold))


def test_board_is_keyed_on_predicted_class_not_true_class():
    """The router only knows what a model SAID. A board keyed on recall needs
    the label to decide which label to ask about, so it cannot be run."""
    gold = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = {"m": np.array([0, 0, 0, 1, 1, 1, 0, 0])}
    b = CP.competence_board(pred, gold, np.arange(8), [0, 1], prior_strength=0.0)
    # of the 5 items where m said 0, three were truly 0
    assert b["raw"]["m"][0] == pytest.approx(3 / 5)
    # of the 3 where it said 1, two were truly 1 -- NOT recall on class 1 (2/4)
    assert b["raw"]["m"][1] == pytest.approx(2 / 3)
    assert CP.per_class_accuracy(pred["m"], gold)[1] == pytest.approx(2 / 4)


def test_shrinkage_stops_a_tiny_cell_from_winning_a_class():
    """One lucky cell with 3 items must not outrank a solid model -- the 'ten
    maxima over K candidates' failure the board would otherwise invite.

    Note the index is a realistic training fold, not the 3 lucky items. Shrinking
    a cell toward an overall accuracy estimated on those same 3 items is a no-op,
    which is exactly what the first version of this test did.
    """
    n, rng = 400, np.random.default_rng(0)
    gold = rng.integers(0, 3, n)
    mediocre = np.where(rng.random(n) < 0.55, gold, (gold + 1) % 3)
    mediocre[mediocre == 2] = 0                   # never claims class 2 ...
    lucky = mediocre.copy()
    hit = np.where(gold == 2)[0][:3]
    lucky[hit] = 2                                # ... except 3 times, all right
    pred = {"lucky": lucky, "mediocre": mediocre}
    idx = np.arange(n)
    raw = CP.competence_board(pred, gold, idx, [0, 1, 2], prior_strength=0.0)
    assert raw["support"]["lucky"][2] == 3
    assert raw["raw"]["lucky"][2] == pytest.approx(1.0)      # perfect on n=3
    sh = CP.competence_board(pred, gold, idx, [0, 1, 2],
                             prior_strength=20.0)["shrunk"]
    assert sh["lucky"][2] < 0.70      # 3 lucky items cannot buy a 1.00 cell


def test_arbitration_centres_out_class_difficulty():
    """Measured on FEVER: uncentred claims sent every item to whoever named the
    easier class, scoring 0.7001 balanced against the best single model's
    0.7349. Stated directly on the boards, with no fitting in the way: A and B
    are equally reliable on class 0, B is far better on class 1, and the item
    is one where they disagree."""
    board = {"A": {0: 0.90, 1: 0.50}, "B": {0: 0.90, 1: 0.70}}
    pred = {"A": np.array([0, 0]), "B": np.array([1, 1])}
    raw_pick, _ = CP.arbitrate(pred, board, ["A", "B"], normalize="none")
    cen_pick, _ = CP.arbitrate(pred, board, ["A", "B"], normalize="class")
    # uncentred: 0.90 > 0.70, so the class-0 claim wins purely for being class 0
    assert list(raw_pick) == [0, 0]
    # centred: A is average on class 0 (+0.00), B is above average on class 1
    # (+0.10), so the better-supported claim wins
    assert list(cen_pick) == [1, 1]


def test_router_hands_over_only_the_classes_the_expert_actually_wins():
    """The whole point: 'which classes, if any' -- and the answer must be a
    subset, not all-or-nothing, when the expert is better on one class only."""
    n, rng = 900, np.random.default_rng(2)
    gold = rng.integers(0, 2, n)
    free = np.where(rng.random(n) < 0.90, gold, 1 - gold)
    # Expert far better on class 1, clearly WORSE on class 0. It has to be worse
    # rather than merely equal: free's class-1 misses land in the predicted-0
    # bucket, so an expert that merely ties on true class 0 still wins the
    # predicted-0 bucket -- and the router, which conditions on the prediction,
    # is right to take it. First version of this test asserted otherwise.
    exp = np.where(rng.random(n) < np.where(gold == 1, 0.98, 0.78), gold, 1 - gold)
    r = CP.class_router({"free": free}, exp, gold, _board_folds(n, gold))
    assert r["classes_to_expert"] == [1]
    assert 0.0 < r["expert_share"] < 1.0


def test_router_keeps_everything_free_when_the_expert_adds_nothing():
    n, rng = 900, np.random.default_rng(3)
    gold = rng.integers(0, 2, n)
    free = np.where(rng.random(n) < 0.90, gold, 1 - gold)
    exp = np.where(rng.random(n) < 0.70, gold, 1 - gold)
    r = CP.class_router({"free": free}, exp, gold, _board_folds(n, gold))
    assert r["classes_to_expert"] == []
    assert r["expert_share"] == 0.0


def test_router_respects_the_budget():
    n, rng = 900, np.random.default_rng(4)
    gold = rng.integers(0, 3, n)
    free = np.where(rng.random(n) < 0.45, gold, (gold + 1) % 3)
    exp = np.where(rng.random(n) < 0.95, gold, (gold + 1) % 3)
    wide = CP.class_router({"free": free}, exp, gold, _board_folds(n, gold),
                           budget=1.0)
    tight = CP.class_router({"free": free}, exp, gold, _board_folds(n, gold),
                            budget=0.34)
    assert wide["expert_share"] > tight["expert_share"]
    assert tight["expert_share"] <= 0.34 + 1e-9


def test_router_optimises_the_headline_metric_not_bucket_accuracy():
    """Handing a class over can raise accuracy INSIDE that class's bucket while
    lowering balanced accuracy overall, because it also changes how often the
    class gets predicted at all. Measured on HateSpeech: the expert is +25.7
    points on items the free arm calls class 1 (0.639 vs 0.382), and a
    gain-ranked router took the class and lost 1.2 points.

    Here the trap is exact: the expert answers the whole predicted-1 bucket with
    the majority label, which is 0.70 against the free arm's 0.30 on that bucket
    and takes recall on class 1 to zero.
    """
    gold = np.array([0] * 900 + [1] * 100)
    free = np.zeros(1000, int)
    free[:140] = 1                       # 140 true-0 called 1
    free[900:960] = 1                    # 60 true-1 called 1
    expert = free.copy()
    expert[free == 1] = 0                # expert answers that bucket with 0
    bucket = free == 1
    assert (expert[bucket] == gold[bucket]).mean() > (free[bucket] == gold[bucket]).mean()
    assert CP.balanced_accuracy(expert, gold) < CP.balanced_accuracy(free, gold)
    r = CP.class_router({"free": free}, expert, gold, _board_folds(1000, gold))
    assert r["classes_to_expert"] == []          # must refuse the trade
    assert r["balanced"] >= CP.balanced_accuracy(free, gold) - 1e-9


def test_per_class_conf_matches_the_dial_at_the_ends_but_not_in_between():
    """Honest statement of what the combined rule does and does not contain.

    At 0% and 100% it is the confidence dial exactly. In between it is NOT a
    superset: allocation proceeds in chunks of 2% of the training fold, class by
    class, so it cannot in general reproduce a single global sort. The earlier
    version of this test checked only the two endpoints and was quoted as
    evidence that the rule 'generalises both' -- it does not, and the oracle
    measurement in per_class_headroom() is what exposed that.
    """
    free, conf, expert, gold, folds = _dial_inputs()
    for frac in (0.0, 1.0):
        pt = CP.deferral_curve(free, conf, expert, gold, folds,
                               fractions=(frac,))["points"][0]
        assert pt["per_class_conf"]["balanced"] == pytest.approx(
            pt["confidence"]["balanced"]), f"must match the dial exactly at {frac}"
    # in between, the two rules are allowed to differ -- and the point of G8 is
    # to say whether that difference is worth anything on a given dataset
    mid = CP.deferral_curve(free, conf, expert, gold, folds,
                            fractions=(0.20,))["points"][0]
    assert set(mid) >= {"confidence", "per_class_conf"}


def test_per_class_conf_never_spends_on_a_class_it_cannot_help():
    """Budget is bought by marginal value, so a class the expert only damages
    must attract none of it however much budget is on offer."""
    n = 1200
    rng = np.random.default_rng(11)
    gold = rng.integers(0, 2, n)
    free = np.where(rng.random(n) < 0.95, gold, 1 - gold)
    conf = rng.random(n)
    expert = free.copy()
    # expert is strictly worse wherever the free arm says 1, equal elsewhere
    flip = (free == 1) & (rng.random(n) < 0.5)
    expert[flip] = 1 - expert[flip]
    folds = _board_folds(n, gold, k=3)
    esc = np.zeros(n, bool)
    for tr, te in folds:
        esc |= CP._allocate(tr, te, "per_class_conf", 0.5, free, conf, expert, gold)
    assert esc[free == 1].mean() < 0.05      # class 1 attracts almost nothing


def test_g8_reports_no_headroom_when_a_global_threshold_is_already_optimal():
    """If conf -> correctness is identical in every class, one global threshold
    is optimal and a per-class rule has nothing to win. G8 must say so, because
    three estimator 'fixes' were attempted before anyone measured this."""
    n, rng = 4000, np.random.default_rng(7)
    gold = rng.integers(0, 4, n)
    free = np.where(rng.random(n) < 0.7, gold, (gold + 1) % 4)
    conf = np.clip(rng.normal(np.where(free == gold, 0.8, 0.4), 0.12), 0, 1)
    expert = np.where(rng.random(n) < 0.85, gold, (gold + 1) % 4)
    h = CP.per_class_headroom(free, conf, expert, gold, _board_folds(n, gold), 0.20)
    assert not h["worth_it"]
    assert "G8 FAIL" in h["verdict"]


def test_g8_finds_headroom_when_one_class_needs_a_different_threshold():
    """A class whose confidences are pure noise must be escalated on sight,
    while a well-calibrated class should be escalated only at the bottom. That
    is genuine per-class structure, and G8 has to see it."""
    n, rng = 6000, np.random.default_rng(8)
    gold = rng.integers(0, 2, n)
    free = np.where(rng.random(n) < 0.75, gold, 1 - gold)
    conf = np.clip(rng.normal(np.where(free == gold, 0.8, 0.4), 0.10), 0, 1)
    # class 1's confidence carries no information at all
    noisy = free == 1
    conf[noisy] = rng.uniform(0.6, 0.9, int(noisy.sum()))
    expert = np.where(rng.random(n) < 0.92, gold, 1 - gold)
    h = CP.per_class_headroom(free, conf, expert, gold, _board_folds(n, gold), 0.20)
    assert h["headroom"] > 0


def test_reliability_cache_is_keyed_on_content_not_fold_indices():
    """Same fold indices, different free arm, must not share a cached fit --
    the cache was index-keyed first and would have returned another dataset's
    curves for anything that reuses a fold layout."""
    n, rng = 900, np.random.default_rng(21)
    gold = rng.integers(0, 2, n)
    conf = rng.random(n)
    expert = np.where(rng.random(n) < 0.9, gold, 1 - gold)
    good = np.where(rng.random(n) < 0.9, gold, 1 - gold)
    bad = np.where(rng.random(n) < 0.5, gold, 1 - gold)
    folds = _board_folds(n, gold, k=3)
    out = {}
    for name, free in (("good", good), ("bad", bad)):
        esc = np.zeros(n, bool)
        for tr, te in folds:
            esc |= CP._allocate(tr, te, "per_class_conf", 0.30, free, conf, expert, gold)
        out[name] = esc.copy()
    # a weak free arm should attract a different escalation set than a strong one
    assert not np.array_equal(out["good"], out["bad"])


def test_g8_null_absorbs_the_optimism_of_a_wider_search():
    """A per-class oracle fits K thresholds on the fold it is scored on; the
    global oracle fits one. On data with no per-class structure at all, the raw
    gap is therefore positive from search width alone -- and the shuffled-class
    null must account for essentially all of it.

    This is the bias that made the ungated gate admit the rule on 5 of 5 IMDB
    folds against a full-data verdict of FAIL.
    """
    n, rng = 3000, np.random.default_rng(31)
    gold = rng.integers(0, 6, n)
    # free arm and confidence carry NO class-specific structure
    free = np.where(rng.random(n) < 0.7, gold, (gold + 1) % 6)
    conf = np.clip(rng.normal(np.where(free == gold, 0.8, 0.4), 0.12), 0, 1)
    expert = np.where(rng.random(n) < 0.85, gold, (gold + 1) % 6)
    h = CP.per_class_headroom(free, conf, expert, gold, _board_folds(n, gold),
                              0.20, n_null=3)
    assert not h["worth_it"]
    # the null must explain most of whatever raw gap appeared
    assert h["headroom_over_null"] < h["headroom"] + 1e-12
    assert "shuffled classes buy" in h["verdict"]


def _ds_inputs(n=900, seed=41):
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, 2, n)
    emb = rng.normal(size=(n, 8))
    emb[gold == 1] += 0.9
    pred, proba = {}, {}
    for nm, acc in (("a", 0.85), ("b", 0.78), ("c", 0.70)):
        p = np.where(rng.random(n) < acc, gold, 1 - gold)
        pred[nm] = p
        pb = np.zeros((n, 2)); pb[np.arange(n), p] = 0.8; pb[np.arange(n), 1 - p] = 0.2
        proba[nm] = pb
    return pred, proba, emb, gold, _board_folds(n, gold, k=3)


def test_deslib_runs_on_the_shipped_pool_or_says_why_not():
    pred, proba, emb, gold, folds = _ds_inputs()
    r = CP.deslib_selection(pred, proba, emb, gold, folds, n_boot=300)
    if not r["available"]:
        pytest.skip(r["reason"])
    assert set(r["methods"]) >= {"DESlib LCA", "DESlib OLA", "DESlib KNORA-E"}
    assert r["baseline"] == "a"                      # the strongest candidate
    for nm, m in r["methods"].items():
        if "Oracle" not in nm:
            assert isinstance(m["significant_after_holm"], bool)


def test_deslib_oracle_is_a_ceiling_over_the_pool():
    """The Oracle picks the member that is right, so it cannot be below the best
    single model -- and it bounds what any selection rule could reach. Getting
    this wrong the first time (labels for the whole dataset passed where the
    fold's labels were wanted) raised a shape error rather than a wrong number,
    which is the lucky version of that mistake."""
    pred, proba, emb, gold, folds = _ds_inputs()
    r = CP.deslib_selection(pred, proba, emb, gold, folds, n_boot=300)
    if not r["available"]:
        pytest.skip(r["reason"])
    orc = r["methods"]["DESlib Oracle (ceiling)"]["balanced"]
    assert orc >= r["baseline_balanced"] - 1e-9
    for nm, m in r["methods"].items():
        assert m["balanced"] <= orc + 1e-9


def test_deslib_pool_sees_the_same_data_as_its_baseline():
    """The first attempt at this comparison carved DSEL out of the training
    fold, so the pool was fitted on 2/3 of what the baseline saw and part of the
    reported loss was that handicap. Frozen out-of-fold predictions mean DSEL is
    an index set, not a data cut -- the pool's predictions are byte-identical to
    the candidates the rest of the pipeline scores."""
    pred, proba, emb, gold, folds = _ds_inputs()
    r = CP.deslib_selection(pred, proba, emb, gold, folds, n_boot=300)
    if not r["available"]:
        pytest.skip(r["reason"])
    assert r["baseline_balanced"] == pytest.approx(
        CP.balanced_accuracy(pred["a"], gold))


def test_deslib_rows_carry_the_unpaired_floor_for_the_ship_decision():
    """A paired bootstrap resolves smaller effects than the unpaired floor, so a
    Holm-surviving win can still be under it -- banking77's LCA is +0.0058
    against a +/-0.0066 floor. Both numbers have to reach the caller, or the
    pipeline reports a winner it then refuses to ship with no reason given."""
    pred, proba, emb, gold, folds = _ds_inputs()
    r = CP.deslib_selection(pred, proba, emb, gold, folds, n_boot=300)
    if not r["available"]:
        pytest.skip(r["reason"])
    for nm, m in r["methods"].items():
        assert "clears_unpaired_floor" in m and "unpaired_floor" in m
        assert m["clears_unpaired_floor"] == (
            abs(m["delta_vs_best_single"]) >= m["unpaired_floor"])


def test_cheapest_equivalent_prefers_the_cheaper_indistinguishable_point():
    """Measured on FEVER: the dial peaks at 0.8192 with 75% of traffic on the
    LLM while 50% scores 0.8177 -- a gap of 0.0015 against a +/-0.0062 floor. A
    third of the bill for a difference the dataset cannot measure."""
    gold = np.repeat([0, 1], 3000)
    pts = [
        {"confidence": {"expert_share": 0.10, "balanced": 0.70}},
        {"confidence": {"expert_share": 0.50, "balanced": 0.8177}},
        {"confidence": {"expert_share": 0.75, "balanced": 0.8192}},
    ]
    r = CP.cheapest_equivalent(pts, 0.8192, gold, rules=("confidence",))
    assert r["found"]
    assert r["expert_share"] == pytest.approx(0.50)
    assert r["gives_up"] == pytest.approx(0.0015, abs=1e-6)


def test_cheapest_equivalent_does_not_cross_the_floor():
    """A point that is genuinely worse must not be sold as equivalent."""
    gold = np.repeat([0, 1], 3000)
    floor = CP.resolution_floor(gold)
    pts = [
        {"confidence": {"expert_share": 0.10, "balanced": 0.80 - 3 * floor}},
        {"confidence": {"expert_share": 0.75, "balanced": 0.80}},
    ]
    r = CP.cheapest_equivalent(pts, 0.80, gold, rules=("confidence",))
    assert r["expert_share"] == pytest.approx(0.75)


def test_confusion_counts_rows_as_truth_and_columns_as_prediction():
    """Transposing a confusion matrix swaps recall for precision and reads as a
    different system. Pin the orientation."""
    gold = np.array([0, 0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1, 0])
    c = CP.confusion(pred, gold, {0: "neg", 1: "pos"})
    assert c["matrix"] == [[2, 1], [1, 1]]      # row 0 = true neg
    assert c["recall"][0] == pytest.approx(2 / 3)
    assert c["precision"][0] == pytest.approx(2 / 3)
    assert c["off_diagonal"] == 2
    assert c["accuracy"] == pytest.approx(3 / 5)


def test_confusion_ranks_the_errors_that_matter_by_size():
    n, rng = 600, np.random.default_rng(3)
    gold = rng.integers(0, 3, n)
    pred = gold.copy()
    leak = np.where(gold == 2)[0][:40]
    pred[leak] = 1                               # class 2 leaks heavily into 1
    c = CP.confusion(pred, gold, {0: "a", 1: "b", 2: "c"})
    assert c["top_confusions"][0]["true"] == 2
    assert c["top_confusions"][0]["pred"] == 1
    assert c["top_confusions"][0]["n"] == 40
    assert 0 < c["top_confusions"][0]["share_of_true"] <= 1


def test_routing_table_reports_what_the_rule_did_with_each_cell():
    """The four cells say what SHOULD happen to those items. Without the rule's
    own decisions beside them the table is a ceiling, not a diagnosis -- you
    cannot see whether the shipped dial went and got the winnable items or spent
    the budget on the ones a call breaks."""
    gold = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    free = np.array([0, 0, 1, 1, 1, 1, 0, 0])      # right on 0,1 and 4,5
    other = np.array([0, 1, 0, 1, 1, 0, 1, 0])     # right on 0,2 and 4,6
    # cells: both = {0,4}, free-only = {1,5}, other-only = {2,6}, neither = {3,7}
    esc = np.array([False, False, True, False, False, False, False, False])
    r = CP.routing_table(free, other, gold, "the LLM", escalated=esc)
    assert (r["both_right"], r["free_only_right"],
            r["other_only_right"], r["neither_right"]) == (2, 2, 2, 2)
    # one of the two winnable items was escalated, none of the at-risk ones
    assert r["sent"]["other_only_right"]["share_sent"] == pytest.approx(0.5)
    assert r["sent"]["free_only_right"]["share_sent"] == pytest.approx(0.0)
    assert r["headroom"] == pytest.approx(0.25)
    assert r["at_risk"] == pytest.approx(0.25)


def test_routing_table_without_a_rule_is_still_a_valid_ceiling():
    gold = np.array([0, 1, 0, 1])
    free = np.array([0, 1, 1, 0])
    other = np.array([0, 0, 0, 1])
    r = CP.routing_table(free, other, gold, "the LLM")
    assert r["sent"] is None
    assert r["oracle_accuracy"] == pytest.approx(1.0)


def test_margin_equals_confidence_on_two_classes_by_construction():
    """p1 - p2 = 2*max - 1 when there are two classes: a monotone transform, so
    the ranking and therefore the escalation set are necessarily identical. Worth
    pinning, because two identical columns on the dial look like corroboration
    rather than the same rule twice."""
    rng = np.random.default_rng(5)
    p = rng.random(200)
    proba = np.column_stack([p, 1 - p])
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    sig = CP.deferral_signals(proba, pred, {"a": pred}, conf)
    assert np.corrcoef(np.argsort(sig["margin"]), np.argsort(conf))[0, 1] > 0.999


def test_margin_differs_from_confidence_once_there_are_three_classes():
    """0.55/0.44/0.01 and 0.55/0.225/0.225 have the same max probability and
    nothing else in common. That is the whole point of margin sampling."""
    proba = np.array([[0.55, 0.44, 0.01], [0.55, 0.225, 0.225]])
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    sig = CP.deferral_signals(proba, pred, {"a": pred}, conf)
    assert conf[0] == pytest.approx(conf[1])
    assert sig["margin"][0] < sig["margin"][1]      # first is the closer call


def test_committee_is_ordered_by_disagreement_then_confidence():
    """The raw count is too coarse to threshold -- at four members most items sit
    at zero disagreement and a 5% budget would escalate all of them. Confidence
    breaks the tie, exactly as the research brief specified."""
    pred = np.array([0, 0, 0, 0])
    pool = {"a": np.array([0, 1, 1, 0]), "b": np.array([0, 0, 1, 0])}
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    proba = np.column_stack([conf, 1 - conf])
    sig = CP.deferral_signals(proba, pred, pool, conf)
    order = np.argsort(sig["committee"])            # escalated first
    assert order[0] == 2                            # both members disagree
    assert order[1] == 1                            # one disagrees
    assert list(order[2:]) == [3, 0]                # no disagreement, low conf first


# --- conformal -------------------------------------------------------------
def _skewed(n=4000, k=4, seed=0):
    """A skewed label distribution at any k -- the fixture hardcoded four
    probabilities and blew up the moment a test asked for three."""
    rng = np.random.default_rng(seed)
    w = np.array([0.5 ** i for i in range(k)])
    gold = rng.choice(k, n, p=w / w.sum())
    logit = np.zeros((n, k)); logit[np.arange(n), gold] = 2.0
    logit += rng.normal(0, 1.4, (n, k))
    P = np.exp(logit); P /= P.sum(1, keepdims=True)
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(P, gold))
    return P, gold, folds


def test_mondrian_holds_per_class_where_split_conformal_does_not():
    """Split conformal guarantees the rate ON AVERAGE, which is how a rare class
    gets quietly sacrificed to hold a headline number. Measured on HateSpeech:
    marginal coverage 0.9001 against a 0.90 target while one of two classes sits
    at 0.8992. Class-conditional calibration is the fix."""
    P, gold, folds = _skewed()
    sp = CP.conformal_coverage(P, gold, folds, alpha=0.10, mondrian=False)
    mo = CP.conformal_coverage(P, gold, folds, alpha=0.10, mondrian=True)
    assert sp["marginal_coverage"] > 0.88          # the average looks fine
    assert len(sp["classes_failing"]) > 0          # while classes miss
    # per-class coverage is tighter around the target under Mondrian
    spread = lambda r: max(v["coverage"] for v in r["per_class"].values()) \
                     - min(v["coverage"] for v in r["per_class"].values())
    assert spread(mo) < spread(sp)


def test_conformal_uses_the_finite_sample_correction():
    """(n+1)/n is what turns an asymptotic statement into one that holds at the
    sample sizes a rare class actually has."""
    src = open(ROOT_CLASSPIPE, encoding="utf-8").read()
    assert "np.ceil((n + 1) * (1 - alpha)) / n" in src


def test_conformal_never_conditions_on_the_true_label_at_test_time():
    """Mondrian tests candidate label c against class c's own threshold. Keying
    on the true class would be unrunnable -- it is what the decision is for."""
    P, gold, folds = _skewed(n=800, k=3, seed=2)
    tr, te = folds[0]
    mem_a, _ = CP.conformal_sets(P, gold, tr, te, 0.10, mondrian=True)
    shuffled = gold.copy()
    # shuffled[te] is fancy indexing -- it returns a COPY, so shuffling it in
    # place scrambles a temporary and leaves `shuffled` equal to `gold`. Permute
    # and assign back, or this test passes against the very code it forbids.
    rng = np.random.default_rng(1)
    shuffled[te] = rng.permutation(gold[te])
    assert not np.array_equal(shuffled, gold), "the fixture did not scramble anything"
    mem_b, _ = CP.conformal_sets(P, shuffled, tr, te, 0.10, mondrian=True)
    assert np.array_equal(mem_a[te], mem_b[te])


# --- learned router --------------------------------------------------------
def test_router_prefers_the_expert_where_the_expert_is_actually_better():
    """The point of learning it rather than proxying: an interaction a linear
    model in these features cannot represent."""
    n, rng = 3000, np.random.default_rng(7)
    feat = rng.normal(size=(n, 4))
    region = feat[:, 0] > 0
    free_right = np.where(region, rng.random(n) < 0.45, rng.random(n) < 0.95).astype(int)
    expert_right = np.where(region, rng.random(n) < 0.92, rng.random(n) < 0.60).astype(int)
    tr, te = np.arange(0, 2000), np.arange(2000, n)
    sc = CP.router_scores(feat, free_right, expert_right, tr, te)
    # lower score = escalate; the region where the expert wins must score lower
    assert sc[te][region[te]].mean() < sc[te][~region[te]].mean()


def test_router_and_conformal_are_implemented_but_off_the_dial():
    """Both were measured for a full run and removed from the search: conformal
    is a near-duplicate of confidence as a routing signal, and the router lost to
    confidence while destabilising the operating point. Neither moved a score by
    more than its dataset's floor, and together they doubled the runtime.

    They stay implemented -- conformal because its coverage guarantee is the
    thing it is actually for, the router so the negative result can be re-checked
    without rebuilding it."""
    assert "conformal" not in CP.RULES
    assert "router" not in CP.RULES
    assert set(CP.RULES) == {"confidence", "margin", "committee",
                             "class_aware", "per_class_conf",
                                 "precision_floor"}
    # still callable, still covered by the tests above
    assert callable(CP.router_scores) and callable(CP.conformal_coverage)
    assert callable(CP.conformal_sets)


def test_g3_counts_a_tied_pair_once_not_twice():
    """Two identical candidates leave both dominance flags true. Incrementing a
    per-model counter then reported "dominating_pairs 2 of 1" -- an impossible
    count -- and flipped the verdict to "some genuine complementarity exists" for
    a duplicated model, which is what the gate exists to reject."""
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 2, 2000)
    p = np.clip(rng.normal(np.where(gold == 1, 0.7, 0.3), 0.2), 0, 1)
    r = CP.roc_dominance({"a": p, "b": p.copy()}, gold)
    assert r["dominating_pairs"] <= r["total_pairs"]
    assert r["is_total_order"] is True
    assert "G3 FAIL" in r["verdict"]


def test_g4_null_keeps_the_class_structure_it_is_testing_against():
    """A global permutation gives each null member its overall accuracy
    uniformly, so on a rare class it is stronger than any real member and
    balanced accuracy rewards it -- the null saturated at 0.9824-0.9999 across
    every committed run. Permuting within each class recovers the right
    reference: on members that genuinely are independent given the class, the
    null must land near the observed oracle rather than far above it."""
    rng = np.random.default_rng(0)
    n, K = 7000, 7
    gold = rng.choice(K, n, p=np.array([.3, .25, .2, .1, .08, .05, .02]))
    acc = {0: [.9, .5, .5], 1: [.5, .9, .5], 2: [.5, .5, .9], 3: [.6] * 3,
           4: [.4] * 3, 5: [.3] * 3, 6: [.2] * 3}
    preds = {}
    for k in range(3):
        ok = np.array([rng.random() < acc[int(g)][k] for g in gold])
        preds[f"m{k}"] = np.where(ok, gold, (gold + 1) % K)
    r = CP.pool_diversity(preds, gold, n_sim=20)
    assert abs(r["oracle_if_independent"] - r["oracle_balanced"]) < 0.05


def test_diversity_ratio_does_not_explode_when_the_pool_beats_independence():
    """The old denominator clamped at 1e-9 precisely in the good case -- members
    more complementary than independent -- and returned 5e7 into the artifact."""
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 2, 2000)
    a, b = gold.copy(), gold.copy()
    a[:200] = 1 - a[:200]
    b[200:400] = 1 - b[200:400]          # disjoint error sets
    r = CP.pool_diversity({"a": a, "b": b}, gold, n_sim=20)
    assert r["diversity_ratio"] is None or r["diversity_ratio"] < 100


def test_singleton_accuracy_reads_the_set_member_not_the_argmax():
    """Under Mondrian the threshold is per class, so the one label that clears
    its own bar is routinely NOT the model's argmax -- on this fixture they
    differ on most singletons. Scoring argmax reports the accuracy of a decision
    the conformal layer never makes."""
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(0)
    gold = np.array([0] * 550 + [1] * 100 + [2] * 550)   # class 1 rare AND hard
    P = np.array([rng.dirichlet([3.0, 0.6, 3.0]) if g == 1
                  else rng.dirichlet(np.where(np.arange(3) == g, 8.0, 1.0))
                  for g in gold])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(P, gold))
    members = np.zeros_like(P, dtype=bool)
    for tr, te in folds:
        m, _ = CP.conformal_sets(P, gold, tr, te, 0.10, mondrian=True)
        members |= m
    singles = members.sum(axis=1) == 1
    member = members[singles].argmax(axis=1)
    assert (member != P[singles].argmax(axis=1)).sum() > 0, "fixture proves nothing"
    r = CP.conformal_coverage(P, gold, folds, alpha=0.10, mondrian=True)
    assert r["accuracy_on_singletons"] == pytest.approx(
        float((member == gold[singles]).mean()))


def test_a_construction_that_clears_the_bar_is_not_vetoed_by_one_that_does_not():
    """Ranking before qualifying lets the higher-scoring construction fail Holm
    and drag a qualifying one down with it, falling back to the single model."""
    out = {"selection_dcs_lca": {"balanced": 0.81, "significant_after_holm": False,
                                 "delta_vs_best_single": 0.02,
                                 "resolution": {"resolvable": True}},
           "stacking": {"balanced": 0.80, "significant_after_holm": True,
                        "delta_vs_best_single": 0.01,
                        "resolution": {"resolvable": True}}}
    qualifies = lambda n: (out[n]["significant_after_holm"]
                           and out[n]["delta_vs_best_single"] > 0
                           and out[n]["resolution"]["resolvable"])
    ok = [n for n in ("selection_dcs_lca", "stacking") if qualifies(n)]
    assert ok == ["stacking"] and max(ok, key=lambda n: out[n]["balanced"]) == "stacking"
    src = open(ROOT_CLASSPIPE, encoding="utf-8").read()
    assert "ok = [n for n in (\"selection_dcs_lca\", \"stacking\") if qualifies(n)]" in src
    assert "winner = max((\"selection_dcs_lca\", \"stacking\")," not in src


def test_budget_split_reports_the_rule_that_ships():
    """It used to allocate with per_class_conf whatever the nested search picked,
    so on every dataset where committee or margin won the table described a rule
    nobody would deploy."""
    n = 400
    rng = np.random.default_rng(3)
    gold = rng.integers(0, 3, n)
    free = np.where(rng.random(n) < 0.7, gold, (gold + 1) % 3)
    conf = rng.random(n)
    exp = np.where(rng.random(n) < 0.9, gold, (gold + 2) % 3)
    folds = _board_folds(n, gold, k=5)
    shipped = np.zeros(n, dtype=bool)
    shipped[:40] = True                         # a mask no rule would produce
    r = CP.budget_split(free, conf, exp, gold, folds, 0.10,
                        rule="committee", escalated=shipped)
    assert r["rule"] == "committee"
    assert r["overall_share"] == pytest.approx(shipped.mean())
    total = sum(v["items"] * v["escalated"] for v in r["per_class"].values())
    assert total == pytest.approx(shipped.sum())


def test_the_inner_search_keeps_the_groups_the_outer_folds_enforce():
    """G2 groups duplicate texts so a copy cannot sit in train while its twin is
    scored. An ungrouped inner split undoes that exactly where the operating
    point is chosen."""
    import inspect
    src = inspect.getsource(CP.nested_best_operating_point)
    assert "StratifiedGroupKFold" in src
    assert "np.asarray(groups)[tr]" in src


def test_the_operating_point_is_chosen_without_gold_when_gold_does_not_exist():
    """Under --train-on llm the cascade has no gold at run time. Selecting the
    point on gold would give it information OCL's setting withholds."""
    import inspect
    src = inspect.getsource(CP.nested_best_operating_point)
    assert "sel_y = gold if fit_labels is None else np.asarray(fit_labels)" in src
    assert "balanced_accuracy(p[tr], sel_y[tr])" in src
    assert "balanced_accuracy(p[tr], gold[tr])" not in src


# --------------------------------------------------------------------------
# CPM: the bar is the expert's, not a number someone picked
# --------------------------------------------------------------------------
def _cpm_fixture(free_prec=(0.97, 0.65), n=1200, seed=0):
    """Built in terms of PREDICTED class, because that is what the rule keys on.

    `free_prec[c]` is the precision of the free arm's class-c predictions. An
    item's chance of being wrong falls linearly with its confidence, so the
    precision of the top q of a class is 1 - (1 - p)(1 - q): smooth, invertible,
    and not the degenerate "every error sits below every success" that makes any
    threshold rule look perfect.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    free_pred = np.repeat([0, 1], half)
    free_conf = rng.uniform(0.0, 1.0, n)
    gold = free_pred.copy()
    for c in (0, 1):
        idx = np.where(free_pred == c)[0]
        p_wrong = np.clip(2 * (1 - free_prec[c]) * (1 - free_conf[idx]), 0, 1)
        gold[idx[rng.random(len(idx)) < p_wrong]] = 1 - c
    return gold, free_pred, free_conf


def test_the_bar_is_read_off_the_expert_not_off_a_constant():
    """The whole borrow. A class the expert is no better at must not be shipped
    to the expert just because it fails an absolute bar."""
    gold, free_pred, free_conf = _cpm_fixture()
    # an expert that is exactly the free arm: same precision, class by class,
    # so there is nothing on any class that paying for it could buy
    expert = free_pred.copy()
    folds = _board_folds(len(gold), gold)

    absolute = CP.guaranteed_coverage(free_pred, free_conf, gold, expert, 0.90, folds)
    anchored = CP.margin_coverage(free_pred, free_conf, gold, expert, 0.10, folds)

    assert absolute["expert_share"] > 0.2, (
        "the fixture is wrong: a class at 0.65 precision should fail a 0.90 bar")
    assert anchored["expert_share"] < absolute["expert_share"] - 0.15, (
        f"anchoring did nothing: it escalated {anchored['expert_share']:.3f} "
        f"against the absolute bar's {absolute['expert_share']:.3f}, but the "
        "expert is no better than the free arm on either class")


def test_a_class_the_expert_is_better_at_is_still_escalated():
    """The rule must not simply keep everything. Where the expert genuinely
    leads, the anchored bar has to notice and pay."""
    gold, free_pred, free_conf = _cpm_fixture()
    perfect = gold.copy()
    folds = _board_folds(len(gold), gold)
    esc = np.zeros(len(gold), bool)
    for tr, te in folds:
        CP._precision_floor(tr, te, 0.02, free_pred, free_conf, perfect, gold, esc)
    weak = esc[free_pred == 1].mean()
    strong = esc[free_pred == 0].mean()
    assert weak > 0.5, (
        f"a perfect expert against a 0.65-precision class should pull most of "
        f"it; it pulled {weak:.3f}")
    assert weak > strong, (
        f"the weak class ({weak:.3f}) must escalate more than the strong one "
        f"({strong:.3f}) -- that ordering IS the class-gating")


def test_the_margin_is_not_a_budget():
    """`frac` means spend for five rules and give-away for this one. Routing it
    through the budget guards turns margin 0 -- the strictest promise there is
    -- into 'escalate nothing', which is the exact opposite."""
    gold, free_pred, free_conf = _cpm_fixture()
    perfect = gold.copy()
    tr, te = _board_folds(len(gold), gold)[0]
    tight = CP._allocate(tr, te, "precision_floor", 0.0, free_pred, free_conf,
                         perfect, gold)
    loose = CP._allocate(tr, te, "precision_floor", 1.0, free_pred, free_conf,
                         perfect, gold)
    assert tight[te].sum() > 0, "margin 0 is the strictest bar and escalated nothing"
    assert tight[te].sum() > loose[te].sum(), (
        f"margin 0 escalated {tight[te].sum()} and margin 1 escalated "
        f"{loose[te].sum()}; the promise is not being honoured monotonically")
    assert CP.PARAM_KIND["precision_floor"] == "margin"
    assert CP.PARAM_KIND["confidence"] == "budget"


def test_the_promise_is_checked_against_what_was_realized():
    """`promise_holds` is the thing being sold, so it has to be measured on the
    held-out predictions, not asserted from the fitted thresholds."""
    gold, free_pred, free_conf = _cpm_fixture()
    expert = free_pred.copy()
    folds = _board_folds(len(gold), gold)
    r = CP.margin_coverage(free_pred, free_conf, gold, expert, 0.10, folds)
    assert set(r["per_class"]) == {0, 1}
    for c, v in r["per_class"].items():
        if v["owned_share"] > 0:
            assert v["realized_precision"] >= v["bar"] - 0.05, (
                f"class {c} promised {v['bar']:.3f} and delivered "
                f"{v['realized_precision']:.3f}")
    # the bar has to move with the expert it is anchored to, not sit flat
    assert r["per_class"][0]["expert_precision"] > r["per_class"][1]["expert_precision"]
    assert r["per_class"][0]["bar"] > r["per_class"][1]["bar"]


def test_precision_floor_competes_in_the_nested_search():
    """A rule that is implemented but not admitted changes nothing. It has to
    be on the dial the operating point is chosen from."""
    assert "precision_floor" in CP.RULES
    gold, free_pred, free_conf = _cpm_fixture()
    folds = _board_folds(len(gold), gold)
    bp = CP.nested_best_operating_point(free_pred, free_conf, gold.copy(), gold,
                                        folds, (0.0, 0.1, 0.3, 1.0))
    assert all(p["param_kind"] == CP.PARAM_KIND[p["rule"]]
               for p in bp["picks_per_fold"])


# --------------------------------------------------------------------------
# soft targets: implemented, measured, off the pipeline
# --------------------------------------------------------------------------
def test_soft_targets_recover_a_teacher_that_hard_labels_cannot():
    """The point of a distribution target. Two rows with the same argmax and
    different certainty have to produce different students -- if they do not,
    the extra machinery is buying nothing over a label vector."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 6))
    W = rng.normal(size=(6, 3))
    P = np.exp(X @ W)
    P /= P.sum(axis=1, keepdims=True)

    soft = CP.soft_target_fit(X, P, l2=1e-4)
    onehot = np.eye(3)[P.argmax(axis=1)]
    hard = CP.soft_target_fit(X, onehot, l2=1e-4)

    err_soft = float(np.abs(soft.predict_proba(X) - P).mean())
    err_hard = float(np.abs(hard.predict_proba(X) - P).mean())
    assert err_soft < err_hard / 2, (
        f"soft target error {err_soft:.4f} against hard {err_hard:.4f}; the "
        "distribution is not being used")
    assert (soft.predict(X) == P.argmax(axis=1)).mean() > 0.97


def test_soft_targets_take_sparse_features():
    """The cheap arm is a TF-IDF matrix with tens of thousands of columns.
    A fitter that densifies it is a fitter that cannot be used on the one arm
    it exists for."""
    import scipy.sparse as sp
    rng = np.random.default_rng(1)
    X = (rng.random((300, 40)) < 0.1) * rng.normal(size=(300, 40))
    P = np.exp(X @ rng.normal(size=(40, 4)))
    P /= P.sum(axis=1, keepdims=True)
    dense = CP.soft_target_fit(X, P, l2=1e-3).predict_proba(X)
    sparse = CP.soft_target_fit(sp.csr_matrix(X), P, l2=1e-3).predict_proba(X)
    assert np.allclose(dense, sparse, atol=1e-4)


def test_temperature_flattens_the_target_it_is_given():
    """T is applied to the TEACHER, not to the student's output. Getting that
    backwards produces a student that is systematically over-confident and
    still passes an argmax-agreement check."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 5))
    P = np.exp(3.0 * X @ rng.normal(size=(5, 3)))
    P /= P.sum(axis=1, keepdims=True)
    sharp = CP.soft_target_fit(X, P, l2=1e-4, temperature=1.0)
    flat = CP.soft_target_fit(X, P, l2=1e-4, temperature=4.0)
    assert flat.predict_proba(X).max(axis=1).mean() < \
        sharp.predict_proba(X).max(axis=1).mean() - 0.05


def test_soft_targets_are_not_on_the_dial():
    """Measured on three datasets and inside the resolution floor on all of
    them, negative on FEVER. It stays callable and stays out of the pipeline;
    see the docstring for the numbers."""
    assert not hasattr(CP, "RULES") or "distilled" not in CP.RULES
    import inspect
    doc = inspect.getdoc(CP.soft_target_fit)
    assert "MEASURED AND NOT SHIPPED" in doc
