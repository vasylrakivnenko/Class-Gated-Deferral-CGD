"""Gate replay: the gates must be out-of-fold, must find planted structure, and must refuse to escalate when the LLM is useless."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phase2 import gate_replay as G  # noqa: E402


def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    gold = np.array(["yes" if i % 2 else "no" for i in range(n)])
    return rng, gold


def test_outer_folds_are_disjoint_and_cover_everything():
    _, gold = _synthetic()
    folds = G.outer_folds(gold, 5, 0)
    seen = np.zeros(len(gold), int)
    for tr, te in folds:
        assert len(set(tr) & set(te)) == 0
        seen[te] += 1
    assert (seen == 1).all()


def test_threshold_gate_reaches_oracle_on_separable_confidence():
    rng, gold = _synthetic()
    cheap_correct = rng.random(len(gold)) > 0.3            # 70% cheap accuracy
    p = np.where(cheap_correct, 0.99, 0.51)                # confidence perfectly separates errors
    inc_correct = np.ones(len(gold), bool)                 # LLM always right
    folds = G.outer_folds(gold, 5, 0)
    esc = G.threshold_gate_oof(p, cheap_correct, inc_correct, folds)
    casc = G._cascade_correct(esc["maxacc"], cheap_correct, inc_correct)
    assert casc.mean() == 1.0
    assert abs(esc["maxacc"].mean() - (~cheap_correct).mean()) < 1e-9   # escalates exactly the errors


def test_never_escalate_when_llm_is_useless():
    rng, gold = _synthetic()
    cheap_correct = rng.random(len(gold)) > 0.3
    p = rng.random(len(gold))
    inc_correct = np.zeros(len(gold), bool)                # LLM always wrong
    folds = G.outer_folds(gold, 5, 0)
    esc = G.threshold_gate_oof(p, cheap_correct, inc_correct, folds)
    for c in G.CRITERIA:
        assert esc[c].mean() == 0.0                         # share 0 = cheap-alone
    X = np.column_stack([p, np.ones(len(p)), np.zeros(len(p)), np.zeros(len(p))])
    esc_r = G.router_gate_oof(X, cheap_correct, inc_correct, folds, 0)
    for c in G.CRITERIA:
        assert esc_r[c].mean() == 0.0


def test_router_recovers_planted_relative_competence():
    rng, gold = _synthetic(n=600)
    f = rng.normal(size=len(gold))
    cheap_correct = f > 0                                  # cheap right on one half of feature space
    inc_correct = f < 0                                    # LLM right on the other half
    X = np.column_stack([np.full(len(f), 0.7), np.ones(len(f)), f, np.zeros(len(f))])   # signal only in 'log_len' slot
    folds = G.outer_folds(gold, 5, 0)
    esc = G.router_gate_oof(X, cheap_correct, inc_correct, folds, 0)
    casc = G._cascade_correct(esc["maxacc"], cheap_correct, inc_correct)
    assert casc.mean() > 0.95                               # near-perfect routing, out-of-fold
    assert abs(esc["maxacc"].mean() - 0.5) < 0.08


def test_frontier_endpoints_are_cheap_alone_and_llm_alone():
    rng, gold = _synthetic()
    cheap_correct = rng.random(len(gold)) > 0.25
    inc_correct = rng.random(len(gold)) > 0.4
    p = rng.random(len(gold))
    ts, sh, ac = G.threshold_frontier(p, cheap_correct, inc_correct)
    assert sh[0] == 0.0 and abs(ac[0] - cheap_correct.mean()) < 1e-12
    assert sh[-1] == 1.0 and abs(ac[-1] - inc_correct.mean()) < 1e-12
    assert (np.diff(sh) >= -1e-12).all()                   # share is monotone along the sweep


def test_ni_criterion_never_worse_than_incumbent_on_training_data():
    rng, gold = _synthetic()
    cheap_correct = rng.random(len(gold)) > 0.5            # cheap 50%
    inc_correct = rng.random(len(gold)) > 0.2              # LLM 80%
    p = rng.random(len(gold))
    ts, sh, ac = G.threshold_frontier(p, cheap_correct, inc_correct)
    i = G._choose(sh, ac, "ni", float(inc_correct.mean()))
    assert ac[i] >= inc_correct.mean() - 1e-12
