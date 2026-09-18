"""Redis LangCache semantic-cache baseline: same idea as retrieval_baseline.py
(find the most similar train conversation-prefix, copy its response) but using
LangCache's real dense-embedding vector search instead of TF-IDF cosine, to
see whether embedding quality -- not the retrieval architecture -- was the
bottleneck in D26's 1-NN result (5.2% conditional / 2.4% unconditional,
median neighbor similarity only 0.47).

WHY THIS IS COMPARABLE TO D26, NOT A DIFFERENT EXPERIMENT
------------------------------------------------------------
Same TRAIN population sampled as candidates, same test_seen queries, same
compose@1 scoring predicate (skeleton + every template id must match gold),
same "copy the single nearest neighbor's answer" prediction rule. The only
variable changed is the similarity engine: LangCache's managed embedding
model + vector search, via its real REST API, instead of local TF-IDF
1-2gram cosine.

WHY THIS RUN IS DELIBERATELY SMALL
------------------------------------
LangCache has no documented batch-store endpoint -- each stored entry is one
HTTP POST. Storing the full 43,159-turn train-covered population before
scaling would mean tens of thousands of network calls with unknown rate
limits and unknown billing, on a live credential, before any signal exists
that embedding quality even helps. This script defaults to a few hundred
stored entries and 100 queries -- enough to read the DIRECTION of the effect
(does real semantic similarity beat TF-IDF's 0.47 median / 5.2% conditional)
before committing to the API calls a full-scale run would need.

A REAL ARCHITECTURAL MISMATCH, DISCOVERED LIVE, NOT WORKED AROUND SILENTLY
----------------------------------------------------------------------------
LangCache's `prompt` field is capped at 1,024 characters ("Prompt is too
long" / "the length must be between 1 and 1024" -- confirmed via live 400
responses). Our context is the FULL conversation thread per D2's own
decision ("Context is the FULL THREAD by default"), which routinely exceeds
that on a several-turn conversation -- roughly half of a random train sample
was over the cap. LangCache is built as a single-query LLM cache (one
question in, one cached answer out), not a full-multi-turn-history cache.
This script therefore TRUNCATES context to the last `--max-prompt-chars`
characters (tail, i.e. the most recent turns) before sending, which is a
real, reported compromise -- it answers "how does LangCache do on a
recent-turns window," not "how does LangCache do on the full thread the
cache's own selector uses." Rows this drops entirely (their WHOLE context,
even truncated, still exceeds no limit -- truncation always fits) never
happens; every row is sent, just windowed. Store failures should now be near
zero; if they are not, investigate rather than assume this is the only cause.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.langcache_baseline \
        --n-train 500 --n-test 100 --similarity-threshold 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def _env():
    host = os.environ["LANGCACHE_HOST"]
    cache_id = os.environ["LANGCACHE_CACHE_ID"]
    key = os.environ["LANGCACHE_API_KEY"]
    return host, cache_id, key


def _store_one(session, base, headers, turn_id, context_text, composed_text, tids, acts):
    # This cache has no `attributes` schema configured (confirmed via a live
    # 400: "no attributes are configured for this cache"), so metadata needed
    # to score compose@1 is packed into `response` as JSON instead -- this
    # does not affect matching, since only `prompt` is embedded for search.
    payload = {"turn_id": turn_id, "composed_text": composed_text,
              "template_ids": tids, "acts": acts}
    r = session.post(
        f"{base}/entries",
        headers=headers,
        json={"prompt": context_text, "response": json.dumps(payload)},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["entryId"]


def _search_one(session, base, headers, context_text, threshold):
    r = session.post(
        f"{base}/entries/search",
        headers=headers,
        json={"prompt": context_text, "similarityThreshold": threshold},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return None
    best = max(data, key=lambda d: d["similarity"])
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--similarity-threshold", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-prompt-chars", type=int, default=1000,
                    help="LangCache caps `prompt` at 1024 chars; truncate to the "
                         "TAIL (most recent turns) of context before sending")
    ap.add_argument("--flush-first", action="store_true",
                    help="wipe the cache before storing (use if a prior run left entries)")
    ap.add_argument("--out", default="outputs/probes/response/langcache_baseline.json")
    args = ap.parse_args(argv)

    host, cache_id, key = _env()
    base = f"https://{host}/v1/caches/{cache_id}"
    headers = {"accept": "application/json", "content-type": "application/json",
              "Authorization": f"Bearer {key}"}

    session = requests.Session()
    if args.flush_first:
        session.post(f"{base}/flush", headers=headers, timeout=30)
        print("flushed cache", file=sys.stderr)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import load_split_rows

    cfg = load_config()
    bank = load_bank(cfg)
    bank_text = {t.template_id: t.text_delex for t in bank.templates}

    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, "test_seen", bank, h4_indexer=None, cache_dir="")

    train_h7_by_turn: dict = {}
    for r in train_rows["h7"]:
        train_h7_by_turn.setdefault(r.turn_id, []).append(r)
    test_h7_by_turn: dict = {}
    for r in test_rows["h7"]:
        test_h7_by_turn.setdefault(r.turn_id, []).append(r)

    def _fully_covered_turns(h5_rows, h7_by_turn):
        out = []
        for row in h5_rows:
            positions = sorted(h7_by_turn.get(row.turn_id, []), key=lambda r: r.position)
            tids = [p.gold_template_id for p in positions]
            if tids and all(tids):
                out.append((row, positions, tids))
        return out

    train_covered = _fully_covered_turns(train_rows["h5"], train_h7_by_turn)
    test_covered = _fully_covered_turns(test_rows["h5"], test_h7_by_turn)
    print(f"  train fully-covered: {len(train_covered):,}  "
          f"test_seen fully-covered: {len(test_covered):,}", file=sys.stderr)

    rng = random.Random(args.seed)
    train_sample = rng.sample(train_covered, min(args.n_train, len(train_covered)))
    test_sample = rng.sample(test_covered, min(args.n_test, len(test_covered)))

    print(f"storing {len(train_sample)} train entries "
          f"(concurrency={args.concurrency})...", file=sys.stderr)
    t0 = time.time()
    n_store_err = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {}
        for row, positions, tids in train_sample:
            acts = [p.act for p in positions]
            composed = " ".join(bank_text.get(t, "") for t in tids)
            prompt = row.context.text[-args.max_prompt_chars:]
            fut = ex.submit(_store_one, session, base, headers, row.turn_id,
                            prompt, composed, tids, acts)
            futs[fut] = row.turn_id
        n_done = 0
        for fut in as_completed(futs):
            n_done += 1
            try:
                fut.result()
            except Exception as e:
                n_store_err += 1
                print(f"  store error {futs[fut]}: {e}", file=sys.stderr)
            if n_done % 1000 == 0 or n_done == len(futs):
                rate = n_done / (time.time() - t0)
                print(f"  store progress: {n_done}/{len(futs)}  "
                      f"errors={n_store_err}  {rate:.0f}/s", file=sys.stderr)
    print(f"  stored in {time.time()-t0:.1f}s, {n_store_err} errors", file=sys.stderr)

    print(f"searching {len(test_sample)} test_seen turns "
          f"(threshold={args.similarity_threshold})...", file=sys.stderr)
    t1 = time.time()
    n_search_err = 0
    results = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {}
        for row, positions, gold_tids in test_sample:
            prompt = row.context.text[-args.max_prompt_chars:]
            fut = ex.submit(_search_one, session, base, headers,
                            prompt, args.similarity_threshold)
            futs[fut] = (row.turn_id, gold_tids, [p.act for p in positions])
        n_done = 0
        for fut in as_completed(futs):
            n_done += 1
            turn_id, gold_tids, gold_acts = futs[fut]
            try:
                results[turn_id] = (fut.result(), gold_tids, gold_acts)
            except Exception as e:
                n_search_err += 1
                print(f"  search error {turn_id}: {e}", file=sys.stderr)
            if n_done % 1000 == 0 or n_done == len(futs):
                rate = n_done / (time.time() - t1)
                print(f"  search progress: {n_done}/{len(futs)}  "
                      f"errors={n_search_err}  {rate:.0f}/s", file=sys.stderr)
    print(f"  searched in {time.time()-t1:.1f}s, {n_search_err} errors", file=sys.stderr)

    test_context_by_turn = {row.turn_id: row.context.text for row, _, _ in test_sample}
    test_gold_tids_by_turn = {row.turn_id: gold_tids for row, _, gold_tids in test_sample}

    n_scored = hit = hit_skel = 0
    sims = []
    n_no_match = 0
    per_row = []
    for turn_id, (best, gold_tids, gold_acts) in results.items():
        n_scored += 1
        gold_text = " ".join(bank_text.get(t, "") for t in test_gold_tids_by_turn.get(turn_id, []))
        if best is None:
            n_no_match += 1
            per_row.append({"turn_id": turn_id, "matched": False,
                            "context": test_context_by_turn.get(turn_id, ""),
                            "gold_text": gold_text, "predicted_text": "(no match found)"})
            continue
        sims.append(best["similarity"])
        payload = json.loads(best["response"])
        pred_tids = payload["template_ids"]
        pred_acts = payload["acts"]
        skel_hit = pred_acts == gold_acts
        full_hit = skel_hit and pred_tids == gold_tids and bool(gold_tids)
        if full_hit:
            hit += 1
        if skel_hit:
            hit_skel += 1
        per_row.append({"turn_id": turn_id, "matched": True,
                        "similarity": best["similarity"],
                        "skeleton_hit": skel_hit, "full_hit": full_hit,
                        "context": test_context_by_turn.get(turn_id, ""),
                        "gold_text": gold_text,
                        "predicted_text": payload.get("composed_text", "")})

    result = {
        "method": "LangCache managed semantic cache (real dense embedding + vector "
                 "search, via REST API) -- same 'copy nearest neighbor's answer' "
                 "rule as D26's TF-IDF 1-NN baseline",
        "n_train_stored": len(train_sample) - n_store_err,
        "n_test_queried": n_scored,
        "similarity_threshold": args.similarity_threshold,
        "n_no_match_above_threshold": n_no_match,
        "mean_similarity_of_matches": (sum(sims) / len(sims)) if sims else None,
        "median_similarity_of_matches": (sorted(sims)[len(sims) // 2] if sims else None),
        "compose@1_of_queried": (hit / n_scored) if n_scored else None,
        "skeleton_only_of_queried": (hit_skel / n_scored) if n_scored else None,
        "compare_against_D26": {
            "learned_cache_TFIDF_logreg": {"conditional": 0.2765370138017566},
            "1NN_TFIDF_no_rerank": {"conditional": 0.05244667503136763,
                                    "median_similarity": 0.47},
        },
        "caveat": ("SMALL-SCALE probe (n_train stored / n_test queried are far below "
                  "D26's full 43,159/3,985) -- meant to show DIRECTION only: does a "
                  "real embedding model find more genuinely-similar neighbors than "
                  "TF-IDF did, not a certified headline number. A larger stored "
                  "population would very likely raise the hit rate somewhat (more "
                  "candidates to match against), independent of embedding quality."),
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": per_row}, fh, indent=1)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
