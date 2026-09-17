"""1-nearest-neighbor retrieval baseline: for each test_seen turn, find the
TRAIN conversation-prefix most SIMILAR (cosine, TF-IDF word 1-2gram over the
same ContextWindow.text every other arm uses) and copy that neighbor's
skeleton + templates as the prediction. No exact match required or expected --
similarity, not identity, which is the honest version of "find something like
this and copy what happened" (see DECISIONS: an exact whole-conversation match
covers 0.22% of dev turns; this is the generalizing alternative).

Scored on NEVER-SEEN data from the start (test_seen only -- no train-accuracy
step, because unlike an exact-match lookup, an honest baseline earns its
number on the split that matters), with the SAME compose@1 predicate, SAME
population (n=3985 conditional / n=8889 unconditional), and SAME bank as:
  - the learned cache (TF-IDF+logreg selector): 27.7% / 12.4%
  - Fireworks qwen3-0.6B structured arm:        24.2% / 10.9%
  - Fireworks qwen3-0.6B unconstrained arm:      21.8% /  9.8%
so all four numbers sit on one table.

WHY THIS BASELINE IS WORTH HAVING (D5's constant-predictor spirit, one level
up): a label-blind constant tells you the floor a model must clear. A
retrieval baseline tells you how much of a learned selector's score is
"a similar conversation existed in train" versus "the model learned something
past raw similarity." If 1-NN retrieval landed anywhere close to 27.7%, the
logistic regression would not be earning its keep over a much simpler method.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.retrieval_baseline --split test_seen
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_seen")
    ap.add_argument("--out", default="outputs/probes/response/retrieval_baseline.json")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args(argv)

    from reflex.config import load_config
    from reflex.compile import load_bank

    cfg = load_config()
    for ov in args.overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        cur = node[parts[-1]]
        node[parts[-1]] = type(cur)(val) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else val

    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd")
    from probes.run_response_probe import load_split_rows

    bank = load_bank(cfg)

    t0 = time.time()
    print("loading train rows (full corpus parse, shared cost)...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    print(f"  train rows loaded, {time.time()-t0:.1f}s", file=sys.stderr)
    test_rows = load_split_rows(cfg, args.split, bank, h4_indexer=None, cache_dir="")

    train_h5 = train_rows["h5"]
    test_h5 = test_rows["h5"]
    train_h7_by_turn: dict[str, list] = {}
    for r in train_rows["h7"]:
        train_h7_by_turn.setdefault(r.turn_id, []).append(r)
    test_h7_by_turn: dict[str, list] = {}
    for r in test_rows["h7"]:
        test_h7_by_turn.setdefault(r.turn_id, []).append(r)

    print(f"  train turns: {len(train_h5):,}  {args.split} turns: {len(test_h5):,}", file=sys.stderr)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    train_texts = [r.context.text for r in train_h5]
    test_texts = [r.context.text for r in test_h5]

    t1 = time.time()
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    print(f"  TF-IDF fit+transform: {time.time()-t1:.1f}s, vocab={len(vec.vocabulary_):,}", file=sys.stderr)

    t2 = time.time()
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(Xtr)
    dist, idx = nn.kneighbors(Xte)
    print(f"  1-NN search: {time.time()-t2:.1f}s", file=sys.stderr)

    n_cond = hit_cond = hit_skel_cond = 0
    hit_uncond = hit_skel_uncond = 0
    n_total = len(test_h5)
    sims = []
    per_row = []

    for i, trow in enumerate(test_h5):
        neighbor = train_h5[idx[i][0]]
        sim = float(1.0 - dist[i][0])
        sims.append(sim)

        pred_acts = list(neighbor.gold_acts)
        pred_positions = sorted(train_h7_by_turn.get(neighbor.turn_id, []), key=lambda r: r.position)
        pred_tids = [p.gold_template_id for p in pred_positions]

        gold_positions = sorted(test_h7_by_turn.get(trow.turn_id, []), key=lambda r: r.position)
        gold_acts = list(trow.gold_acts)
        gold_tids = [p.gold_template_id for p in gold_positions]
        fully_covered = bool(gold_tids) and all(gold_tids)

        skeleton_hit = pred_acts == gold_acts
        full_hit = skeleton_hit and fully_covered and pred_tids == gold_tids and bool(gold_tids)

        per_row.append({"turn_id": trow.turn_id, "nearest_train_turn_id": neighbor.turn_id,
                        "similarity": sim, "skeleton_hit": skeleton_hit, "full_hit": full_hit,
                        "fully_covered": fully_covered})

        if fully_covered:
            n_cond += 1
            if full_hit:
                hit_cond += 1
            if skeleton_hit:
                hit_skel_cond += 1
        if full_hit:
            hit_uncond += 1
        if skeleton_hit:
            hit_skel_uncond += 1

    result = {
        "method": "1-NN retrieval (cosine, TF-IDF word 1-2gram over ContextWindow.text, "
                 "fit on train, queried by test_seen -- similarity, not exact match)",
        "split": args.split,
        "n_total_turns": n_total,
        "mean_neighbor_similarity": sum(sims) / len(sims),
        "median_neighbor_similarity": sorted(sims)[len(sims) // 2],
        "conditional": {
            "n": n_cond,
            "compose@1": (hit_cond / n_cond) if n_cond else None,
            "skeleton_only": (hit_skel_cond / n_cond) if n_cond else None,
        },
        "unconditional": {
            "n": n_total,
            "compose@1": hit_uncond / n_total,
            "skeleton_only": hit_skel_uncond / n_total,
        },
        "compare_against": {
            "learned_cache_TFIDF_logreg": {"conditional": 0.2765370138017566, "unconditional": 0.09764877939025762 + 0.02632801},
            "note": "learned_cache figures from outputs/probes/response/select.json headline_test_seen; "
                   "fireworks arms from DECISIONS D25. Same population (n=3985/8889), same bank, same "
                   "compose@1 predicate throughout.",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    # fix the accidental typo-prone inline calc above; recompute cleanly
    result["compare_against"]["learned_cache_TFIDF_logreg"]["unconditional"] = 0.12397345033187085

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": per_row}, fh, indent=1)

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
