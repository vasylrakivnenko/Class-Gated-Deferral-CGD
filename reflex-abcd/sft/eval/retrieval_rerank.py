"""Retrieve-then-rerank baseline: top-10 nearest train neighbors (TF-IDF
cosine, same as retrieval_baseline.py), then a LOCAL NEURAL CROSS-ENCODER
reranks those 10 candidate responses against the query context, and the
reranker's #1 pick is scored as compose@1.

THREE NUMBERS, EACH ANSWERING A DIFFERENT QUESTION
-----------------------------------------------------
1. compose@10 (retrieval recall, no reranker): is the gold answer anywhere in
   the 10 candidates retrieval found? This is the CEILING -- no reranker can
   beat it, because reranking only reorders what retrieval already found.
2. compose@1 after rerank: what the reranker's top pick actually achieves.
3. compose@1, plain 1-NN (from retrieval_baseline.py, 5.2%/2.4%): the
   no-reranker floor. The gap between (2) and this floor is what the neural
   reranker bought; the gap between (1) and (2) is headroom the reranker left
   on the table.

RERANKER: cross-encoder/ms-marco-MiniLM-L-12-v2 (already cached locally, no
download). A CROSS-encoder, not a bi-encoder/embedding-cosine reranker --
it jointly attends to (query, candidate) as one input, which is what
"reranker" means in IR and is architecturally different from (and usually
stronger than) re-scoring by a second embedding-cosine, which would just be
retrieval with better features, not reranking. Trained on MS MARCO passage
relevance -- general web query-passage judgments, NOT on ABCD or on dialogue
acts, so it has no domain adaptation here. That mismatch is the headline risk
and is reported explicitly, not glossed over.

CANDIDATE TEXT: the natural composed utterance (sentence texts joined, in
skeleton order, NO "ACT |" prefixes) -- a cross-encoder trained on natural
query-passage pairs has no reason to understand a formatted act label, so it
is scored on what a customer would actually read.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.retrieval_rerank --split test_seen --k 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_seen")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--out", default="outputs/probes/response/retrieval_rerank.json")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N turns only")
    ap.add_argument("--rerank-batch-size", type=int, default=256)
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
    bank_text = {t.template_id: t.text_delex for t in bank.templates}

    t0 = time.time()
    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, args.split, bank, h4_indexer=None, cache_dir="")
    print(f"  rows loaded, {time.time()-t0:.1f}s", file=sys.stderr)

    train_h5 = train_rows["h5"]
    test_h5 = test_rows["h5"][: args.limit] if args.limit else test_rows["h5"]
    train_h7_by_turn: dict[str, list] = {}
    for r in train_rows["h7"]:
        train_h7_by_turn.setdefault(r.turn_id, []).append(r)
    test_h7_by_turn: dict[str, list] = {}
    for r in test_rows["h7"]:
        test_h7_by_turn.setdefault(r.turn_id, []).append(r)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    train_texts = [r.context.text for r in train_h5]
    test_texts = [r.context.text for r in test_h5]

    t1 = time.time()
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    nn = NearestNeighbors(n_neighbors=args.k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(Xtr)
    dist, idx = nn.kneighbors(Xte)
    print(f"  top-{args.k} retrieval: {time.time()-t1:.1f}s", file=sys.stderr)

    def candidate_for(train_idx: int):
        neighbor = train_h5[train_idx]
        acts = list(neighbor.gold_acts)
        positions = sorted(train_h7_by_turn.get(neighbor.turn_id, []), key=lambda r: r.position)
        tids = [p.gold_template_id for p in positions]
        texts = [bank_text.get(t, "") for t in tids]
        composed = " ".join(t for t in texts if t)
        return neighbor.turn_id, acts, tids, composed

    # ---- build all (query, candidate_text) pairs for the cross-encoder ----
    t2 = time.time()
    print(f"loading reranker {args.reranker} ...", file=sys.stderr)
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(args.reranker)
    print(f"  reranker loaded, {time.time()-t2:.1f}s", file=sys.stderr)

    all_pairs = []
    pair_owner = []  # (test_i, candidate_slot)
    candidates_by_test: list[list] = []
    for i, trow in enumerate(test_h5):
        cands = [candidate_for(j) for j in idx[i]]
        candidates_by_test.append(cands)
        for slot, (turn_id, acts, tids, composed) in enumerate(cands):
            all_pairs.append((trow.context.text, composed if composed else "(no text)"))
            pair_owner.append((i, slot))

    t3 = time.time()
    scores = reranker.predict(all_pairs, batch_size=args.rerank_batch_size, show_progress_bar=False)
    print(f"  reranked {len(all_pairs):,} pairs: {time.time()-t3:.1f}s "
          f"({len(all_pairs)/(time.time()-t3):.0f} pairs/s)", file=sys.stderr)

    scores_by_test: list[list] = [[0.0] * len(c) for c in candidates_by_test]
    for (i, slot), s in zip(pair_owner, scores):
        scores_by_test[i][slot] = float(s)

    # ---- score: compose@10 (retrieval ceiling), compose@1 after rerank ----
    n_cond = hit_cond_at1 = hit_cond_atk = 0
    hit_uncond_at1 = hit_uncond_atk = 0
    n_distinct_candidates_sum = 0
    per_row = []

    for i, trow in enumerate(test_h5):
        gold_positions = sorted(test_h7_by_turn.get(trow.turn_id, []), key=lambda r: r.position)
        gold_acts = list(trow.gold_acts)
        gold_tids = [p.gold_template_id for p in gold_positions]
        fully_covered = bool(gold_tids) and all(gold_tids)

        cands = candidates_by_test[i]
        cscores = scores_by_test[i]
        distinct = len({(tuple(acts), tuple(tids)) for _, acts, tids, _ in cands})
        n_distinct_candidates_sum += distinct

        # compose@k: is gold anywhere among the k candidates?
        hit_at_k = any(acts == gold_acts and tids == gold_tids and bool(gold_tids)
                       for _, acts, tids, _ in cands) and fully_covered

        # reranked top-1: argmax cross-encoder score
        best_slot = max(range(len(cscores)), key=lambda s: cscores[s])
        _, best_acts, best_tids, _ = cands[best_slot]
        hit_at_1 = (best_acts == gold_acts and best_tids == gold_tids
                   and bool(gold_tids) and fully_covered)

        per_row.append({"turn_id": trow.turn_id, "fully_covered": fully_covered,
                        "hit_at_1_reranked": hit_at_1, "hit_at_k_retrieval": hit_at_k,
                        "n_distinct_candidates": distinct, "best_rerank_score": cscores[best_slot]})

        if fully_covered:
            n_cond += 1
            if hit_at_1:
                hit_cond_at1 += 1
            if hit_at_k:
                hit_cond_atk += 1
        if hit_at_1:
            hit_uncond_at1 += 1
        if hit_at_k:
            hit_uncond_atk += 1

    n_total = len(test_h5)
    result = {
        "reranker_model": args.reranker,
        "reranker_kind": "cross-encoder (jointly attends query+candidate), trained on MS MARCO "
                        "passage relevance -- NOT domain-adapted to ABCD or dialogue acts",
        "candidate_text": "composed utterance from the neighbor's skeleton+templates, sentence "
                          "texts joined, NO act labels (natural text a cross-encoder can judge)",
        "split": args.split,
        "k": args.k,
        "n_total_turns": n_total,
        "mean_distinct_candidates_of_k": n_distinct_candidates_sum / n_total,
        "conditional": {
            "n": n_cond,
            f"compose@{args.k}_retrieval_ceiling": (hit_cond_atk / n_cond) if n_cond else None,
            "compose@1_after_rerank": (hit_cond_at1 / n_cond) if n_cond else None,
        },
        "unconditional": {
            "n": n_total,
            f"compose@{args.k}_retrieval_ceiling": hit_uncond_atk / n_total,
            "compose@1_after_rerank": hit_uncond_at1 / n_total,
        },
        "compare_against": {
            "learned_cache_TFIDF_logreg": {"conditional": 0.2765370138017566, "unconditional": 0.12397345033187085},
            "plain_1NN_no_rerank": {"conditional": 0.05244667503136763, "unconditional": 0.02351220609742378},
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": per_row}, fh, indent=1)

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
