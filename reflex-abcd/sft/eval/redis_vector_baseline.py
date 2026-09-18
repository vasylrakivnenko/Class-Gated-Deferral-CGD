"""Self-hosted dense-embedding retrieval baseline via RediSearch (raw Redis
Cloud instance + vector index we build and query ourselves), fixing the two
real constraints the managed LangCache probe (D31) hit:

  1. LangCache's prompt field capped queries well under 1,024 chars, forcing
     a 400-char truncation of the FULL conversation thread D2 uses everywhere
     else. Here we embed the full, untruncated context -- same representation
     TF-IDF's 1-NN baseline (D26) uses, so this is a clean head-to-head.
  2. LangCache's managed database hit "out of memory" storing past ~4,000-
     4,500 entries. This Redis Cloud instance reported 25.7MB used with no
     memory cap at the time of connection -- room for the FULL 43,159/3,985
     population, not a scaled-down probe.

EMBEDDING MODEL: sentence-transformers/all-MiniLM-L6-v2 (already cached
locally, no download) -- a real dense bi-encoder, 384-dim, cosine similarity.

VECTOR SEARCH: RediSearch FT.CREATE with a FLAT (exact, brute-force) VECTOR
field, not HNSW (approximate) -- FLAT was chosen deliberately: the LangCache
run showed real result NON-DETERMINISM between two identical runs (compose@1
2.75% vs 8.25% at the same n_train/n_test/seed), consistent with an
approximate-search backend. FLAT guarantees the true nearest neighbor every
time, same as TF-IDF's sklearn NearestNeighbors(algorithm="brute").

SAME compose@1 predicate, same fully-covered population, as D26/D31.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.redis_vector_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=0, help="0 = all fully-covered train turns")
    ap.add_argument("--n-test", type=int, default=0, help="0 = all fully-covered test_seen turns")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--index-name", default="reflex_vec_idx")
    ap.add_argument("--key-prefix", default="reflex:train:")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--flush-first", action="store_true")
    ap.add_argument("--out", default="outputs/probes/response/redis_vector_baseline.json")
    args = ap.parse_args(argv)

    import redis
    import numpy as np
    from redis.commands.search.field import VectorField, TagField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    r = redis.from_url(os.environ["REDIS_CLOUD_URL"])
    print("PING:", r.ping(), file=sys.stderr)
    if args.flush_first:
        r.flushdb()
        print("flushed db", file=sys.stderr)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import load_split_rows

    cfg = load_config()
    bank = load_bank(cfg)
    bank_text = {t.template_id: t.text_delex for t in bank.templates}

    t0 = time.time()
    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, "test_seen", bank, h4_indexer=None, cache_dir="")
    print(f"  rows loaded, {time.time()-t0:.1f}s", file=sys.stderr)

    train_h7_by_turn: dict = {}
    for row in train_rows["h7"]:
        train_h7_by_turn.setdefault(row.turn_id, []).append(row)
    test_h7_by_turn: dict = {}
    for row in test_rows["h7"]:
        test_h7_by_turn.setdefault(row.turn_id, []).append(row)

    def _fully_covered(h5_rows, h7_by_turn):
        out = []
        for row in h5_rows:
            positions = sorted(h7_by_turn.get(row.turn_id, []), key=lambda r: r.position)
            tids = [p.gold_template_id for p in positions]
            if tids and all(tids):
                out.append((row, positions, tids))
        return out

    train_covered = _fully_covered(train_rows["h5"], train_h7_by_turn)
    test_covered = _fully_covered(test_rows["h5"], test_h7_by_turn)
    if args.n_train:
        train_covered = train_covered[: args.n_train]
    if args.n_test:
        test_covered = test_covered[: args.n_test]
    print(f"  train fully-covered: {len(train_covered):,}  "
          f"test_seen fully-covered: {len(test_covered):,}", file=sys.stderr)

    print(f"loading embedding model {args.model} ...", file=sys.stderr)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"  loaded, dim={dim}", file=sys.stderr)

    # ---- (re)create the vector index ----
    try:
        r.ft(args.index_name).dropindex(delete_documents=False)
    except Exception:
        pass
    schema = (
        VectorField("embedding", "FLAT", {
            "TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE",
        }),
        TagField("turn_id"),
    )
    r.ft(args.index_name).create_index(
        schema, definition=IndexDefinition(prefix=[args.key_prefix], index_type=IndexType.HASH)
    )
    print(f"  index '{args.index_name}' created", file=sys.stderr)

    # ---- embed + store train entries ----
    t1 = time.time()
    print(f"embedding + storing {len(train_covered):,} train entries "
          f"(batch={args.batch_size})...", file=sys.stderr)
    pipe = r.pipeline(transaction=False)
    n_stored = 0
    for i in range(0, len(train_covered), args.batch_size):
        batch = train_covered[i: i + args.batch_size]
        texts = [row.context.text for row, _, _ in batch]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        for (row, positions, tids), emb in zip(batch, embs):
            acts = [p.act for p in positions]
            composed = " ".join(bank_text.get(t, "") for t in tids)
            key = f"{args.key_prefix}{row.turn_id}"
            pipe.hset(key, mapping={
                "embedding": np.asarray(emb, dtype=np.float32).tobytes(),
                "turn_id": row.turn_id,
                "template_ids": "|".join(tids),
                "acts": "|".join(acts),
                "composed_text": composed,
            })
            n_stored += 1
        pipe.execute()
        if (i // args.batch_size) % 20 == 0:
            rate = n_stored / (time.time() - t1)
            print(f"  store progress: {n_stored}/{len(train_covered)}  {rate:.0f}/s",
                  file=sys.stderr)
    print(f"  stored {n_stored} in {time.time()-t1:.1f}s", file=sys.stderr)

    info = r.ft(args.index_name).info()
    print(f"  index num_docs={info.get('num_docs')}", file=sys.stderr)

    # ---- embed + query test entries (exact/FLAT top-1) ----
    t2 = time.time()
    print(f"embedding + querying {len(test_covered):,} test_seen entries...", file=sys.stderr)
    n_scored = hit = hit_skel = 0
    sims = []
    per_row = []
    q = Query("*=>[KNN 1 @embedding $vec AS score]").sort_by("score").return_fields(
        "score", "turn_id", "template_ids", "acts", "composed_text"
    ).dialect(2)

    for i in range(0, len(test_covered), args.batch_size):
        batch = test_covered[i: i + args.batch_size]
        texts = [row.context.text for row, _, _ in batch]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        for (row, positions, gold_tids), emb in zip(batch, embs):
            gold_acts = [p.act for p in positions]
            res = r.ft(args.index_name).search(
                q, query_params={"vec": np.asarray(emb, dtype=np.float32).tobytes()}
            )
            n_scored += 1
            if not res.docs:
                per_row.append({"turn_id": row.turn_id, "matched": False})
                continue
            doc = res.docs[0]
            cosine_sim = 1.0 - float(doc.score)  # RediSearch KNN score is distance
            sims.append(cosine_sim)
            pred_tids = doc.template_ids.split("|")
            pred_acts = doc.acts.split("|")
            skel_hit = pred_acts == gold_acts
            full_hit = skel_hit and pred_tids == gold_tids and bool(gold_tids)
            if full_hit:
                hit += 1
            if skel_hit:
                hit_skel += 1
            per_row.append({"turn_id": row.turn_id, "matched": True, "similarity": cosine_sim,
                            "skeleton_hit": skel_hit, "full_hit": full_hit,
                            "context": row.context.text,
                            "gold_text": " ".join(bank_text.get(t, "") for t in gold_tids),
                            "predicted_text": doc.composed_text})
        if (i // args.batch_size) % 5 == 0:
            rate = n_scored / (time.time() - t2)
            print(f"  query progress: {n_scored}/{len(test_covered)}  {rate:.0f}/s",
                  file=sys.stderr)
    print(f"  queried {n_scored} in {time.time()-t2:.1f}s", file=sys.stderr)

    result = {
        "method": "Self-hosted RediSearch FLAT (exact) vector search, "
                 "sentence-transformers/all-MiniLM-L6-v2 embeddings, FULL untruncated context",
        "n_train_stored": n_stored,
        "n_test_queried": n_scored,
        "mean_similarity_of_matches": (sum(sims) / len(sims)) if sims else None,
        "median_similarity_of_matches": (sorted(sims)[len(sims) // 2] if sims else None),
        "compose@1": (hit / n_scored) if n_scored else None,
        "skeleton_only": (hit_skel / n_scored) if n_scored else None,
        "compare_against": {
            "learned_cache_TFIDF_logreg": {"conditional": 0.2765370138017566},
            "1NN_TFIDF_no_rerank_D26": {"conditional": 0.05244667503136763, "median_similarity": 0.47},
            "LangCache_3800_800_D31": {"compose@1": 0.0825, "median_similarity": 0.9457},
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": per_row}, fh, indent=1)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
