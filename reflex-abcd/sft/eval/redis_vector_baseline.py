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
2.75% vs 8.25% at the same n_train/n_test/seed). Comparing the two committed
copies of langcache_baseline_mid.json row by row locates that entirely in TIED
neighbours, not in approximate search: the two runs matched the identical 453
of 800 rows and agreed exactly on every row whose similarity was < 1.0 (19
full hits, 90 skeleton hits in both), while all 44 rows that flipped were at
similarity EXACTLY 1.0 -- byte-identical prompts where several stored entries
carry different answers and the search returned a different one of them.
NEITHER of those two values is "the" LangCache number: that cell is unstable
in [2.75%, 8.25%] across identical reruns.
FLAT still guarantees the true nearest DISTANCE every time, same as TF-IDF's
sklearn NearestNeighbors(algorithm="brute"); the tie-break itself is the part
that has to be pinned down. All three 1-NN arms now use the same stated rule --
lowest train turn_id (string order) among the tied candidates -- here via a
KNN window that is widened until it is no longer entirely tied (see
`_nearest_with_tie_break`), and each reports n_rows_with_tied_nearest_neighbour.

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


def _langcache_comparison(path: str) -> dict:
    """Quote the LangCache head-to-head figure FROM ITS ARTIFACT.

    This used to be a hardcoded 0.0825 copied out of D31's prose, while the
    artifact on disk for that same 3,800/800 configuration records 0.0275 (an
    identical rerun; see the module docstring). Neither is a stable point
    estimate, so quote whatever the artifact says, carry the source path, and
    say so when the artifact predates the explicit tie-break.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            summary = json.load(fh)["summary"]
    except Exception as exc:
        return {"compose@1": None, "source": f"{path} (unreadable: {exc})"}
    out = {
        "compose@1": summary.get("compose@1_of_queried"),
        "skeleton_only": summary.get("skeleton_only_of_queried"),
        "median_similarity": summary.get("median_similarity_of_matches"),
        "n_train_stored": summary.get("n_train_stored"),
        "n_test_queried": summary.get("n_test_queried"),
        "source": path,
    }
    if "tie_break" not in summary:
        out["unstable"] = (
            "artifact predates langcache_baseline.py's explicit tie-break: the answer "
            "copied on similarity==1.0 duplicate prompts was whichever entry the service "
            "listed first, so this value is one draw, not a point estimate. Measured size of "
            "the effect: two identical 3,800/800 runs scored compose@1 0.0825 and 0.0275 "
            "(skeleton-only 0.21375 and 0.11875), agreeing on every row below similarity 1.0.")
    return out


# RediSearch scores are float32 cosine distances, and the same text embedded in
# two different batches differs by ~1e-7 per component (measured locally with
# all-MiniLM-L6-v2), so exact float equality can miss true duplicates; 1e-6
# groups them.
TIE_EPS = 1e-6
KNN_WINDOW = 10


def _nearest_with_tie_break(search, n_pool, k0=KNN_WINDOW, eps=TIE_EPS):
    """Nearest neighbour under an explicit rule: lowest turn_id among ties.

    `search(k)` returns the k nearest docs (each with .score = cosine distance
    and .turn_id). With KNN 1 the winner among identical-distance
    entries (this corpus repeats its opening contexts many times over, with
    DIFFERENT gold answers) was whichever RediSearch reached first. If the
    whole window is tied the true tie set may extend past it, so widen the
    window 10x until it is not, the pool is exhausted, or the server stops
    returning more. Returns (winner_doc, best_distance, n_tied, truncated).
    """
    def _tied(docs):
        d0 = min(float(d.score) for d in docs)
        return d0, [d for d in docs if float(d.score) <= d0 + eps]

    k = min(k0, n_pool) if n_pool else k0
    docs = search(k)
    if not docs:
        return None, None, 0, False
    d0, tied = _tied(docs)
    while len(tied) >= len(docs) and len(docs) >= k and k < n_pool:
        k = min(k * 10, n_pool)
        try:
            wider = search(k)
        except Exception as exc:  # e.g. a server-side cap on K / LIMIT
            print(f"  tie-set widening to K={k} failed: {exc}", file=sys.stderr)
            break
        if len(wider) <= len(docs):
            break
        docs = wider
        d0, tied = _tied(docs)
    truncated = len(tied) >= len(docs) and len(docs) < n_pool
    winner = min(tied, key=lambda d: str(d.turn_id))
    return winner, d0, len(tied), truncated


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=0, help="0 = all fully-covered train turns")
    ap.add_argument("--n-test", type=int, default=0, help="0 = all fully-covered test_seen turns")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--index-name", default="reflex_vec_idx")
    ap.add_argument("--key-prefix", default="reflex:train:")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--flush-first", action="store_true",
                    help="wipe the WHOLE database (FLUSHDB) before storing")
    ap.add_argument("--no-flush", action="store_true",
                    help="do NOT delete existing keys under --key-prefix before storing. "
                         "The default is to delete them, because the index is defined by "
                         "prefix: hashes left by an earlier, larger run are re-indexed and "
                         "silently enlarge the candidate pool this run's score is measured over")
    ap.add_argument("--out", default="outputs/probes/response/redis_vector_baseline.json")
    ap.add_argument("--langcache-artifact",
                    default="outputs/probes/response/langcache_baseline_mid.json",
                    help="LangCache run quoted in compare_against; read from disk, "
                         "never hardcoded")
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
        pool_reset = "FLUSHDB (whole database)"
    elif not args.no_flush:
        n_deleted = 0
        del_pipe = r.pipeline(transaction=False)
        for key in r.scan_iter(match=f"{args.key_prefix}*", count=1000):
            del_pipe.delete(key)
            n_deleted += 1
            if n_deleted % 1000 == 0:
                del_pipe.execute()
        del_pipe.execute()
        pool_reset = f"deleted {n_deleted} pre-existing keys under prefix '{args.key_prefix}'"
    else:
        pool_reset = "none (--no-flush): the pool may contain entries from earlier runs"
    print(f"pool reset: {pool_reset}", file=sys.stderr)

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
    n_train_fully_covered = len(train_covered)
    n_test_fully_covered = len(test_covered)
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
    stored_by_turn: dict = {}
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
            stored_by_turn[row.turn_id] = ("|".join(tids), "|".join(acts), composed)
            n_stored += 1
        pipe.execute()
        if (i // args.batch_size) % 20 == 0:
            rate = n_stored / (time.time() - t1)
            print(f"  store progress: {n_stored}/{len(train_covered)}  {rate:.0f}/s",
                  file=sys.stderr)
    print(f"  stored {n_stored} in {time.time()-t1:.1f}s", file=sys.stderr)

    info = r.ft(args.index_name).info()
    print(f"  index num_docs={info.get('num_docs')}", file=sys.stderr)
    # The retrieval pool is what the index holds, not what this process sent.
    try:
        index_num_docs = int(float(info.get("num_docs")))
    except (TypeError, ValueError):
        index_num_docs = None
    if index_num_docs is not None and index_num_docs != n_stored:
        print(f"  WARNING: index holds {index_num_docs} docs but this run stored "
              f"{n_stored} -- the candidate pool is not the one n_train_stored names",
              file=sys.stderr)
    n_pool = index_num_docs if index_num_docs else n_stored

    # ---- embed + query test entries (exact/FLAT, explicit tie-break) ----
    t2 = time.time()
    print(f"embedding + querying {len(test_covered):,} test_seen entries...", file=sys.stderr)
    n_scored = hit = hit_skel = 0
    sims = []
    per_row = []
    n_tied_rows = n_tie_set_truncated = 0

    def _query(k):
        # Same KNN form as the original top-1 query; only K changes. Only
        # score + turn_id come back (a tie set can run to thousands of docs);
        # the winner's answer is looked up by turn_id. paging() must follow K
        # or redis-py's default LIMIT 0 10 silently caps the window.
        return Query(f"*=>[KNN {int(k)} @embedding $vec AS score]").sort_by("score") \
            .return_fields("score", "turn_id").paging(0, int(k)).dialect(2)

    for i in range(0, len(test_covered), args.batch_size):
        batch = test_covered[i: i + args.batch_size]
        texts = [row.context.text for row, _, _ in batch]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        for (row, positions, gold_tids), emb in zip(batch, embs):
            gold_acts = [p.act for p in positions]
            vec_bytes = np.asarray(emb, dtype=np.float32).tobytes()
            doc, best_dist, n_tied, truncated = _nearest_with_tie_break(
                lambda k: r.ft(args.index_name).search(
                    _query(k), query_params={"vec": vec_bytes}).docs,
                n_pool)
            n_scored += 1
            if doc is None:
                per_row.append({"turn_id": row.turn_id, "matched": False})
                continue
            if n_tied > 1:
                n_tied_rows += 1
            if truncated:
                n_tie_set_truncated += 1
            cosine_sim = 1.0 - best_dist  # RediSearch KNN score is distance
            sims.append(cosine_sim)
            nearest_turn_id = str(doc.turn_id)
            pred = stored_by_turn.get(nearest_turn_id)
            if pred is None:
                # only reachable under --no-flush: a leftover entry this run did not store
                raw = r.hmget(f"{args.key_prefix}{nearest_turn_id}",
                              "template_ids", "acts", "composed_text")
                pred = tuple((v or b"").decode("utf-8") for v in raw)
            pred_tids = pred[0].split("|")
            pred_acts = pred[1].split("|")
            skel_hit = pred_acts == gold_acts
            full_hit = skel_hit and pred_tids == gold_tids and bool(gold_tids)
            if full_hit:
                hit += 1
            if skel_hit:
                hit_skel += 1
            per_row.append({"turn_id": row.turn_id, "matched": True, "similarity": cosine_sim,
                            "nearest_train_turn_id": nearest_turn_id,
                            "n_tied_candidates": n_tied,
                            "skeleton_hit": skel_hit, "full_hit": full_hit,
                            "context": row.context.text,
                            "gold_text": " ".join(bank_text.get(t, "") for t in gold_tids),
                            "predicted_text": pred[2]})
        if (i // args.batch_size) % 5 == 0:
            rate = n_scored / (time.time() - t2)
            print(f"  query progress: {n_scored}/{len(test_covered)}  {rate:.0f}/s",
                  file=sys.stderr)
    print(f"  queried {n_scored} in {time.time()-t2:.1f}s", file=sys.stderr)

    coverage = (n_scored / n_test_fully_covered) if n_test_fully_covered else None
    compare_note = ("learned_cache_TFIDF_logreg and 1NN_TFIDF_no_rerank_D26 are over ALL "
                    "3985 fully-covered test_seen turns; the 1-NN arm searched all train turns.")
    if n_scored < n_test_fully_covered or n_stored < n_train_fully_covered:
        compare_note = (
            "POPULATIONS DIFFER -- NOT directly comparable: compose@1 here is over "
            f"{n_scored} of the {n_test_fully_covered} fully-covered test_seen turns "
            f"(coverage {coverage if coverage is not None else 0.0:.4f}) against a pool of {n_stored} of the "
            f"{n_train_fully_covered} fully-covered train turns; both are contiguous "
            "head-of-list slices (--n-test / --n-train), not random samples. " + compare_note)

    result = {
        "method": "Self-hosted RediSearch FLAT (exact) vector search, "
                 "sentence-transformers/all-MiniLM-L6-v2 embeddings, FULL untruncated context",
        "n_train_stored": n_stored,
        "n_train_fully_covered": n_train_fully_covered,
        # n_train_stored is what THIS process sent; the retrieval pool is
        # whatever the index actually held when the queries ran.
        "pool_reset_before_store": pool_reset,
        "index_num_docs_at_query_time": index_num_docs,
        "n_test_queried": n_scored,
        "n_test_fully_covered": n_test_fully_covered,
        "coverage": coverage,
        "n_rows_with_tied_nearest_neighbour": n_tied_rows,
        "n_rows_tie_set_truncated": n_tie_set_truncated,
        "tie_break": (f"lowest train turn_id (string order) among candidates within "
                      f"{TIE_EPS} of the best cosine distance"),
        "mean_similarity_of_matches": (sum(sims) / len(sims)) if sims else None,
        "median_similarity_of_matches": (sorted(sims)[len(sims) // 2] if sims else None),
        "compose@1": (hit / n_scored) if n_scored else None,
        "skeleton_only": (hit_skel / n_scored) if n_scored else None,
        "compare_against": {
            "learned_cache_TFIDF_logreg": {"conditional": 0.2765370138017566},
            "1NN_TFIDF_no_rerank_D26": {"conditional": 0.05244667503136763, "median_similarity": 0.47,
                                        "stale": "measured before the retrieval_baseline "
                                                 "nearest-neighbour tie-break fix; re-run "
                                                 "sft.eval.retrieval_baseline to refresh"},
            "LangCache_D31": _langcache_comparison(args.langcache_artifact),
            "note": compare_note,
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
