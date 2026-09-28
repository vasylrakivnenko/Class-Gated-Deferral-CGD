"""Measure how semantically far compose@1 MISSES actually land from the gold
reply, split by what the LLM judge said about them.

The question: are judge-confirmed "wrong" misses also the ones that sit far
from gold in embedding space? If so, cosine-to-gold is a cheap automatic
proxy for the expensive LLM-judge call, and could gate responses directly.

Embeddings use sentence-transformers/all-MiniLM-L6-v2 -- the same model
sft/eval/redis_vector_baseline.py uses, so these numbers sit on the same
scale as the project's existing retrieval work.

SELECTION NEVER TOUCHES THE SCORED SET (D25's select protocol). The gating
threshold is the maximum of ~100 correlated F1 statistics computed against
only 22 positives, so scoring it on the rows it was chosen on reports an
optimistic in-sample maximum, not the rule's F1 on new turns. The judged rows
are therefore split BY CONVERSATION (turns of one conversation share context
and would leak across the split) into a fit half and a held-out half: the
threshold is chosen on `fit`, and the precision/recall/F1 that may be quoted
are recomputed once on `held`. Both are emitted and labelled --
`best_threshold_in_sample` is kept for continuity but is biased by
construction and must not be quoted as the gate's expected performance.

The ROC-AUC is threshold-free, involves no selection, and is unaffected.

USAGE
-----
    python sft/eval/cosine_miss_analysis.py --judge-results <results.jsonl>
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-results",
                    default="sft/eval/data/qwen3_4b_1000_judge_results.jsonl")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default="outputs/probes/response/qwen3_4b_cosine_misses.json")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed for the by-conversation fit/held split used to pick "
                         "and then score the gating threshold")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(args.judge_results, encoding="utf-8")]
    rows = [r for r in rows
            if r.get("gold_text") and r.get("predicted_text") and r.get("verdict")]
    print(f"loaded {len(rows)} judged misses with both texts", file=sys.stderr)

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(args.model)
    gold = model.encode([r["gold_text"] for r in rows], normalize_embeddings=True,
                        show_progress_bar=False)
    pred = model.encode([r["predicted_text"] for r in rows], normalize_embeddings=True,
                        show_progress_bar=False)
    cos = (gold * pred).sum(axis=1)

    for r, c in zip(rows, cos):
        r["cosine_to_gold"] = float(c)

    def stats(sel, label):
        vals = [r["cosine_to_gold"] for r in rows if sel(r)]
        if not vals:
            return None
        return {"label": label, "n": len(vals), "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals), "max": max(vals)}

    groups = [
        stats(lambda r: r["verdict"] == "wrong", "REAL miss (judge: wrong)"),
        stats(lambda r: r["verdict"] == "borderline", "borderline"),
        stats(lambda r: r["verdict"] == "appropriate", "NOT a real miss (appropriate)"),
        stats(lambda r: r["verdict"] in ("appropriate", "borderline"),
              "NOT a real miss (appropriate+borderline)"),
    ]
    groups = [g for g in groups if g]

    print(f"\n{'group':<44} {'n':>4} {'mean':>7} {'median':>7} {'stdev':>7}")
    for g in groups:
        print(f"  {g['label']:<42} {g['n']:>4} {g['mean']:>7.4f} "
              f"{g['median']:>7.4f} {g['stdev']:>7.4f}")

    # Can cosine alone separate real misses from paraphrase misses?
    from sklearn.metrics import roc_auc_score
    y = [0 if r["verdict"] == "wrong" else 1 for r in rows]
    auc = roc_auc_score(y, cos) if 0 < sum(y) < len(y) else float("nan")
    print(f"\nROC-AUC, cosine separating NOT-wrong from wrong: {auc:.4f}")

    def score_at(subset, t):
        """precision/recall/F1 of the 'flag if cosine < t' rule on `subset`."""
        tp = sum(1 for r in subset if r["verdict"] == "wrong" and r["cosine_to_gold"] < t)
        fp = sum(1 for r in subset if r["verdict"] != "wrong" and r["cosine_to_gold"] < t)
        n_wrong = sum(1 for r in subset if r["verdict"] == "wrong")
        if tp == 0 or not n_wrong:
            return None
        prec, rec = tp / (tp + fp), tp / n_wrong
        return {"threshold": t, "precision": prec, "recall": rec,
                "f1": 2 * prec * rec / (prec + rec), "flagged": tp + fp,
                "n": len(subset), "n_wrong": n_wrong}

    def sweep(subset):
        best_ = None
        for t in [i / 100 for i in range(0, 100)]:
            m = score_at(subset, t)
            if m and (best_ is None or m["f1"] > best_["f1"]):
                best_ = m
        return best_

    # Selection must not touch the scored rows. Split BY CONVERSATION -- turns
    # of one conversation share context, so a per-row split would leak.
    def convo_of(r):
        parts = str(r.get("turn_id", "")).split(":")
        return parts[1] if len(parts) > 2 else str(r.get("turn_id", ""))

    convos = sorted({convo_of(r) for r in rows})
    rng = random.Random(args.split_seed)
    rng.shuffle(convos)
    fit_convos = set(convos[: len(convos) // 2])
    fit = [r for r in rows if convo_of(r) in fit_convos]
    held = [r for r in rows if convo_of(r) not in fit_convos]

    best_in_sample = sweep(rows)   # biased by construction; kept for continuity
    best_fit = sweep(fit)
    held_out = score_at(held, best_fit["threshold"]) if best_fit else None

    if best_in_sample:
        print(f"\nIN-SAMPLE best flag-if-below threshold (OPTIMISTIC -- chosen and "
              f"scored on the same {len(rows)} rows): cos < {best_in_sample['threshold']:.2f} -> "
              f"precision {best_in_sample['precision']:.2f}, recall {best_in_sample['recall']:.2f}, "
              f"F1 {best_in_sample['f1']:.2f} ({best_in_sample['flagged']} of {len(rows)} flagged)")
    if best_fit:
        print(f"HELD-OUT: threshold cos < {best_fit['threshold']:.2f} chosen on "
              f"{len(fit)} fit rows ({len(fit_convos)} convos), scored on "
              f"{len(held)} held-out rows: " +
              (f"precision {held_out['precision']:.2f}, recall {held_out['recall']:.2f}, "
               f"F1 {held_out['f1']:.2f} ({held_out['flagged']} flagged)"
               if held_out else "no positives flagged on the held-out half"))

    out = {"model": args.model, "n": len(rows), "groups": groups,
           "roc_auc_not_wrong_vs_wrong": auc,
           "best_threshold_in_sample": (
               dict(best_in_sample,
                    note="threshold selected and scored on the same rows; "
                         "optimistically biased -- the maximum of ~100 correlated "
                         "F1 statistics. NOT the gate's expected performance.")
               if best_in_sample else None),
           "threshold_held_out": {
               "split": "by conversation",
               "split_seed": args.split_seed,
               "n_fit_rows": len(fit), "n_fit_convos": len(fit_convos),
               "n_held_rows": len(held), "n_held_convos": len(convos) - len(fit_convos),
               "selected_on_fit": best_fit,
               "scored_on_held_out": held_out,
               "note": ("threshold chosen on the fit half only and scored once on the "
                       "held-out half -- this is the unbiased estimate (one split, few "
                       "positives: unbiased but high-variance); quote this, "
                       "not best_threshold_in_sample."),
           }}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
