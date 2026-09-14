import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phase2 import gate_replay as G  # noqa: E402
from phase2 import gate_replay2 as G2  # noqa: E402


def test_per_class_gate_escalates_exactly_the_class_the_llm_wins():
    rng = np.random.default_rng(0); n = 400
    gold = np.array(["yes" if i % 2 else "no" for i in range(n)])
    cheap_pred = gold.copy()
    # cheap is right 60% on predicted-'yes', 95% on predicted-'no'; LLM is 90% on 'yes', 70% on 'no'
    cheap_correct = np.where(cheap_pred == "yes", rng.random(n) < 0.60, rng.random(n) < 0.95)
    inc_correct = np.where(cheap_pred == "yes", rng.random(n) < 0.90, rng.random(n) < 0.70)
    folds = G.outer_folds(gold, 5, 0)
    esc = G2.per_class_gate_oof(cheap_pred, cheap_correct, inc_correct, folds, 0.0)
    assert esc[cheap_pred == "yes"].all() and not esc[cheap_pred == "no"].any()


def test_committee_gate_reaches_oracle_when_disagreement_marks_errors():
    rng = np.random.default_rng(1); n = 400
    gold = np.array(["yes" if i % 2 else "no" for i in range(n)])
    cheap_correct = rng.random(n) > 0.3
    inc_correct = np.ones(n, bool)
    n_dis = np.where(cheap_correct, 0.0, 2.0)          # committee disagrees exactly on cheap errors
    folds = G.outer_folds(gold, 5, 0)
    esc = G.threshold_gate_oof(-n_dis, cheap_correct, inc_correct, folds)
    assert G._cascade_correct(esc["maxacc"], cheap_correct, inc_correct).mean() == 1.0
    assert abs(esc["maxacc"].mean() - (~cheap_correct).mean()) < 1e-9
