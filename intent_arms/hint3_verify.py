#!/usr/bin/env python
"""
hint3_verify.py -- independent re-derivation of the headline figures.

Deliberately uses a DIFFERENT code path from hint3_arm.py:
  * reads the persisted probability matrices from hint3_probs_{bot}.npz
    (not the in-memory arm objects),
  * hand-rolls macro-F1 / in-scope accuracy / MCC in pure numpy
    (no sklearn.metrics at all),
  * recomputes the threshold, the gate AUROC (rank-based Mann-Whitney U
    identity instead of sklearn roc_auc_score) and the error-share,
and compares everything to hint3.json. Any disagreement above 1e-9 is a bug.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OOS = "NO_NODES_DETECTED"
R = json.load(open(HERE / "hint3.json"))

fails, checks = [], 0


def chk(name, mine, theirs, tol=1e-9):
    global checks
    checks += 1
    if theirs is None and mine is None:
        return
    if theirs is None or mine is None or abs(float(mine) - float(theirs)) > tol:
        fails.append(f"{name}: mine={mine} json={theirs}")


def macro_f1_numpy(gold, pred, labels):
    """Hand-rolled macro-F1. No sklearn."""
    fs = []
    for L in labels:
        tp = int(((pred == L) & (gold == L)).sum())
        fp = int(((pred == L) & (gold != L)).sum())
        fn = int(((pred != L) & (gold == L)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        fs.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(fs))


def mcc_numpy(gold, pred):
    """Hand-rolled multiclass MCC from the confusion matrix (Gorodkin)."""
    labs = np.array(sorted(set(gold.tolist()) | set(pred.tolist())))
    ix = {l: i for i, l in enumerate(labs)}
    C = np.zeros((len(labs), len(labs)), dtype=np.float64)
    for g, p in zip(gold, pred):
        C[ix[g], ix[p]] += 1
    t = C.sum(1)   # true totals
    p = C.sum(0)   # pred totals
    n = C.sum()
    num = C.trace() * n - float(t @ p)
    den = np.sqrt((n * n - float(p @ p)) * (n * n - float(t @ t)))
    return float(num / den) if den else 0.0


def auroc_rank(score, pos):
    """AUROC via the Mann-Whitney U identity with tie-averaged ranks."""
    score = np.asarray(score, float)
    pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(score, kind="stable")
    s = score[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s), float)
    r[order] = ranks
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


print("independent re-derivation from hint3_probs_*.npz, pure-numpy metrics\n")
for bot, B in R["bots"].items():
    te = pd.read_csv(HERE / "hint3" / f"{bot}_test.csv")
    tr = pd.read_csv(HERE / "hint3" / f"{bot}_train.csv")
    gold = te.label.values.astype(object)
    ins = gold != OOS
    inscope_labels = sorted(set(gold[ins]))
    all_train = sorted(set(tr.label))
    z = np.load(HERE / f"hint3_probs_{bot}.npz", allow_pickle=True)

    # ---- constants ----
    chk(f"{bot}/const_all_oos.accuracy",
        float((np.full(len(gold), OOS) == gold).mean()),
        B["constants"]["const_all_oos"]["accuracy"])
    chk(f"{bot}/oos_rate", float((~ins).mean()), B["oos_rate"])

    for arm, A in B["arms"].items():
        P = z[f"{bot}__{arm}__test_probs"]
        cls = z[f"{bot}__{arm}__classes"].astype(object)
        pred = cls[P.argmax(1)]
        conf = P.max(1)
        # the JSON's own per-item arrays must match the npz
        chk(f"{bot}/{arm}/per_item_conf_l1",
            float(np.abs(conf - np.array(A["per_item"]["conf"])).max()), 0.0, 1e-7)
        assert list(pred) == A["per_item"]["pred_argmax"], f"{bot}/{arm} pred mismatch"
        assert A["per_item"]["gold"] == [str(g) for g in gold], f"{bot}/{arm} gold mismatch"

        for row in A["sweep_paper_grid"]:
            t = row["threshold"]
            eff = np.where(conf >= t, pred, OOS)
            chk(f"{bot}/{arm}/t{t}/acc", float((eff == gold).mean()), row["accuracy"])
            chk(f"{bot}/{arm}/t{t}/inacc",
                float((eff[ins] == gold[ins]).mean()), row["inscope_accuracy"])
            chk(f"{bot}/{arm}/t{t}/oosrec",
                float((eff[~ins] == OOS).mean()), row["oos_recall"])
            chk(f"{bot}/{arm}/t{t}/macroF1",
                macro_f1_numpy(gold, eff, inscope_labels), row["macro_f1_inscope"])
            chk(f"{bot}/{arm}/t{t}/macroF1all",
                macro_f1_numpy(gold, eff, all_train), row["macro_f1_all_train_intents"])
            chk(f"{bot}/{arm}/t{t}/mcc", mcc_numpy(gold, eff), row["mcc"], 1e-8)

    # ---- shipped row + gate ----
    S = B["shipped_row"]
    arm = S["arm"]
    P = z[f"{bot}__{arm}__test_probs"]
    cls = z[f"{bot}__{arm}__classes"].astype(object)
    pred, conf = cls[P.argmax(1)], P.max(1)
    t = S["operating_threshold"]
    eff = np.where(conf >= t, pred, OOS)
    M = S["metrics_at_operating_point"]
    chk(f"{bot}/ship/acc", float((eff == gold).mean()), M["accuracy"])
    chk(f"{bot}/ship/inacc", float((eff[ins] == gold[ins]).mean()), M["inscope_accuracy"])
    chk(f"{bot}/ship/macroF1", macro_f1_numpy(gold, eff, inscope_labels), M["macro_f1_inscope"])
    chk(f"{bot}/ship/mcc", mcc_numpy(gold, eff), M["mcc"], 1e-8)

    gA = S["gate_A_shipped_pipeline"]
    chk(f"{bot}/gateA/auroc", auroc_rank(conf, eff == gold), gA["auroc_conf_vs_correct"], 1e-8)
    gB = S["gate_B_answering_subset_inscope_only"]
    chk(f"{bot}/gateB/auroc", auroc_rank(conf[ins], pred[ins] == gold[ins]),
        gB["auroc_conf_vs_correct"], 1e-8)

    # error share, least-confident 20%, recomputed independently
    for tag, cf, co, g in [("A", conf, eff == gold, gA),
                           ("B", conf[ins], pred[ins] == gold[ins], gB)]:
        k = max(1, int(round(0.20 * len(cf))))
        o = np.argsort(cf, kind="stable")[:k]
        ne = int((~np.asarray(co)).sum())
        mine = float((~np.asarray(co)[o]).sum() / ne) if ne else None
        chk(f"{bot}/gate{tag}/errshare20", mine, g["error_share_least_confident_20pct"], 1e-9)

print(f"checks run: {checks}")
if fails:
    print(f"FAILURES: {len(fails)}")
    for f in fails[:40]:
        print("  ", f)
    raise SystemExit(1)
print("ALL INDEPENDENT RE-DERIVATIONS AGREE (tol 1e-8..1e-9)")
