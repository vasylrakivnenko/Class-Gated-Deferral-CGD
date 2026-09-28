"""Settle the OCL comparison: match the LABEL budget, not just the call budget.

The confound this removes. In neural caching / online cascade learning, `N` is
the maximum number of LLM calls -- and because a call both answers a query and
yields a training label, it is simultaneously the size of the training set. Our
cross-validated pipeline trains on 80% of the dataset at every dial position,
so at OCL's low-budget cells we were holding 1.9x to 15.4x more labels than they
were. Being higher under those conditions says nothing about the method.

Here one number buys both things, exactly as it does for them:

    N calls total = m warm-up calls + (N - m) escalations
    training labels = the m warm-up answers, and nothing else

Every training label is the LLM's own annotation, never gold, because a call
returns the LLM's answer -- that is what OCL trains on and it is what we must
train on to be compared with them. `--labels gold` runs the counterfactual.

`m` is swept and chosen on the warm-up items only. The escalation threshold is
set from CROSS-VALIDATED confidences on the warm-up set, never in-sample ones,
because a model scoring items it trained on is overconfident and would aim the
threshold at the wrong rate.

Two handicaps we accept rather than argue away, both in OCL's favour:
  - they update online, so every escalated item later becomes training data too;
    we train once on the warm-up and never again.
  - their deferral policy is learned; ours is a confidence quantile.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import _provenance as P                                   # noqa: E402
import os as _os
import reconstruct as R                                   # noqa: E402
from downshift import engine as E                         # noqa: E402
from downshift import classpipe as CP                     # noqa: E402

from sklearn.linear_model import LogisticRegression       # noqa: E402

E.EMB_CACHE = _os.path.join(P.CACHE, "emb")
from sklearn.model_selection import StratifiedKFold       # noqa: E402

# Table 1, GPT-3.5 half, transcribed from the paper. Accuracy; HateSpeech also
# carries recall on the positive class, which is the metric they report there.
OCL_CELLS = {
    "imdb":       {1300: 87.95, 3800: 92.48, 5200: 93.01},
    "hatespeech": {600: 82.66, 2700: 85.35, 4900: 83.26},
    "isear":      {1200: 60.78, 1500: 65.34, 2700: 69.75},
    "fever":      {700: 61.95, 2000: 71.86, 2800: 78.49},
}
OCL_RECALL = {"hatespeech": {600: 82.36, 2700: 77.20, 4900: 81.03}}
# The strongest baseline in their own table, for the same cells.
OEL_CELLS = {
    "imdb":       {1300: 86.73, 3800: 88.80, 5200: 89.95},
    "hatespeech": {600: 82.61, 2700: 77.48, 4900: 81.55},
    "isear":      {1200: 56.56, 1500: 60.42, 2700: 61.78},
    "fever":      {700: 61.69, 2000: 69.78, 2800: 76.67},
}
WARMUP_GRID = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)


def balanced_from_acc_recall(acc, recall_pos, n, n_pos):
    """Recover balanced accuracy from a published (accuracy, positive recall).

    Needed because HateSpeech's majority class is 88.83%, so every system in
    that column of Table 1 -- theirs at 82.66-85.35 and ours -- sits BELOW a
    constant predictor, and raw accuracy there ranks systems by how close they
    come to answering "no" every time. OCL reports accuracy and recall side by
    side on this dataset alone, which is enough to reconstruct the metric that
    does rank them.

        TP = recall_pos * n_pos
        TN = accuracy * n - TP
        balanced = (TN/n_neg + TP/n_pos) / 2
    """
    n_neg = n - n_pos
    tp = recall_pos * n_pos
    tn = acc * n - tp
    return 0.5 * (tn / n_neg + tp / n_pos)


def _fit(X, y):
    m = LogisticRegression(max_iter=2000, class_weight="balanced")
    m.fit(X, y)
    return m


def _oof_conf(X, y, seed=0, k=3):
    """Out-of-fold max-probability on the warm-up set.

    In-sample confidences from a model that memorised these rows would put the
    escalation threshold at the wrong place -- the whole point of the threshold
    is to predict doubt on items not yet seen.
    """
    out = np.zeros(len(y))
    classes = sorted(set(y.tolist()))
    if len(classes) < 2 or min(int((y == c).sum()) for c in classes) < k:
        return np.full(len(y), 0.5)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        out[te] = _fit(X[tr], y[tr]).predict_proba(X[te]).max(axis=1)
    return out


def one_stream(X, gold, expert, N, seed, labels="llm"):
    """One pass of the stream at a total budget of N calls.

    Returns the WHOLE warm-up sweep, not its maximum. Taking the max here would
    pick the split by looking at the score it produces on the very stream it is
    scored on -- the selection-on-test bias `nested_best_operating_point` exists
    to remove, reintroduced one level up. The caller picks the split on one set
    of streams and reports on another.
    """
    n = len(gold)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    y_train_src = expert if labels == "llm" else gold

    grid = {}
    for frac in WARMUP_GRID:
        m = max(20, int(round(frac * N)))
        if m > N:
            m = N
        warm = order[:m]
        rest = order[m:]
        yw = y_train_src[warm]
        if len(set(yw.tolist())) < 2:
            continue
        model = _fit(X[warm], yw)
        pred = np.empty(n, dtype=int)
        # the warm-up items were called, so they carry the LLM's answer
        pred[warm] = expert[warm]
        if len(rest):
            pr = model.predict_proba(X[rest])
            pred[rest] = model.classes_[pr.argmax(axis=1)]
            k_left = N - m
            if k_left > 0:
                conf_rest = pr.max(axis=1)
                # threshold from CROSS-VALIDATED warm-up confidences, not the
                # in-sample ones, and aimed at the share the budget can afford
                q = min(1.0, k_left / len(rest))
                thr = float(np.quantile(_oof_conf(X[warm], yw, seed), q))
                esc = rest[conf_rest <= thr][:k_left]
                pred[esc] = expert[esc]
        acc = float((pred == gold).mean())
        rec = float((pred[gold == 1] == 1).mean()) if (gold == 1).any() else float("nan")
        calls = m + max(0, min(N - m, len(rest)))
        grid[frac] = {"accuracy": acc, "recall": rec, "warmup_share": m / N,
                      "calls": int(calls), "labels_used": int(m)}
    return grid


def run(task, stream, seeds, labels):
    gold = np.asarray(stream["gold"])
    expert = np.asarray(stream["gpt3.5"]["pred"])
    texts = stream["text"]
    Xs = [E._embed(texts, e) for e in E.ENCODERS]
    X = np.hstack(Xs)
    rows = []
    half = max(1, seeds // 2)
    for N in sorted(OCL_CELLS[task]):
        grids = [one_stream(X, gold, expert, N, s, labels) for s in range(seeds)]
        grids = [g for g in grids if g]
        sel, ev = grids[:half], grids[half:] or grids[:half]
        # pick the warm-up split on the SELECTION streams only
        cand = [f for f in WARMUP_GRID if all(f in g for g in sel)]
        pick = max(cand, key=lambda f: float(np.mean([g[f]["accuracy"] for g in sel])))
        accs = [g[pick]["accuracy"] for g in ev if pick in g]
        acc = float(np.mean(accs))
        sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        rec = float(np.mean([g[pick]["recall"] for g in ev if pick in g]))
        # what the same numbers look like if you cheat, so the gap is visible
        opt = float(np.mean([max(g[f]["accuracy"] for f in g) for g in ev]))
        theirs = OCL_CELLS[task][N] / 100.0
        oel = OEL_CELLS[task][N] / 100.0
        rows.append({
            "N": N, "budget_share": N / len(gold),
            "ours_accuracy": acc, "ours_sd": sd, "ours_recall": rec,
            "optimistic_accuracy": opt, "selection_bias": opt - acc,
            "warmup_split_chosen": pick,
            "ocl_accuracy": theirs, "oel_accuracy": oel,
            "delta_vs_ocl": acc - theirs, "delta_vs_oel": acc - oel,
            "mean_labels": float(np.mean([g[pick]["labels_used"] for g in ev if pick in g])),
            "n_selection_streams": len(sel), "n_eval_streams": len(ev),
            "ocl_recall": OCL_RECALL.get(task, {}).get(N, None),
        })
    return rows


def report(task, rows, labels, n):
    L = [f"\n{'='*78}", f"{task}  n={n}  training labels = LLM calls  "
         f"(labels from {'the LLM itself' if labels == 'llm' else 'GOLD -- counterfactual'})",
         f"{'='*78}",
         f"  {'N':>6}{'budget':>9}{'ours':>9}{'sd':>7}{'OCL':>9}{'delta':>9}"
         f"{'baseline':>10}{'delta':>9}{'warm':>7}{'bias':>8}"]
    for r in rows:
        L.append(f"  {r['N']:>6}{r['budget_share']:>9.1%}{r['ours_accuracy']:>9.4f}"
                 f"{r['ours_sd']:>7.4f}{r['ocl_accuracy']:>9.4f}{r['delta_vs_ocl']:>+9.4f}"
                 f"{r['oel_accuracy']:>10.4f}{r['delta_vs_oel']:>+9.4f}"
                 f"{r['warmup_split_chosen']:>7.0%}{r['selection_bias']:>+8.4f}")
    w = sum(1 for r in rows if r["delta_vs_ocl"] > 0)
    L.append(f"  -> beats OCL on {w}/{len(rows)} cells at MATCHED labels and calls"
             f"   (warm-up split picked on {rows[0]['n_selection_streams']} held-out "
             f"streams, scored on {rows[0]['n_eval_streams']} others)")
    if task == "hatespeech":
        maj = 1 - 1196 / 10703
        L.append(f"  G1: a constant 'no' predictor scores {maj:.4f} accuracy here, "
                 f"above every system in the table -- accuracy above is not a ranking.")
        L.append(f"    {'N':>6}{'ours rec':>10}{'OCL rec':>10}{'ours bal':>10}"
                 f"{'OCL bal':>10}{'delta':>9}")
        wins = 0
        for r in rows:
            ob = balanced_from_acc_recall(r["ours_accuracy"], r["ours_recall"],
                                          10703, 1196)
            tb = balanced_from_acc_recall(r["ocl_accuracy"], r["ocl_recall"] / 100,
                                          10703, 1196)
            r["ours_balanced"], r["ocl_balanced"] = ob, tb
            r["delta_balanced"] = ob - tb
            wins += ob > tb
            L.append(f"    {r['N']:>6}{r['ours_recall']:>10.4f}"
                     f"{r['ocl_recall']/100:>10.4f}{ob:>10.4f}{tb:>10.4f}"
                     f"{ob - tb:>+9.4f}")
        L.append(f"  -> on the metric that ranks anything here: {wins}/{len(rows)}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--labels", choices=("llm", "gold"), default="llm")
    ap.add_argument("--out", default="label_matched")
    a = ap.parse_args()

    t0 = time.time()
    with P.Tee(os.path.join(P.RESULTS, a.out + ".log")):
        streams, _ = R.load_streams()
        tasks = a.tasks or list(OCL_CELLS)
        payload = {"labels": a.labels, "seeds": a.seeds, "tasks": {}}
        total_w = total_c = 0
        for t in tasks:
            rows = run(t, streams[t], a.seeds, a.labels)
            payload["tasks"][t] = rows
            print(report(t, rows, a.labels, len(streams[t]["gold"])), flush=True)
            total_w += sum(1 for r in rows if r["delta_vs_ocl"] > 0)
            total_c += len(rows)
        payload["cells_won"] = total_w
        payload["cells_total"] = total_c
        print(f"\nOVERALL: {total_w}/{total_c} cells beat OCL at matched label budget")
        P.emit(a.out, payload)
        print(f"wrote {os.path.join(P.RESULTS, a.out + '.json')} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
