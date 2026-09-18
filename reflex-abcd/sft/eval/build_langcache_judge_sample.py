"""Sample LangCache's compose@1 MISSES (matched but wrong) from
outputs/probes/response/langcache_baseline_mid.json for an LLM-judge pass,
mirroring build_llm_judge_sample.py's protocol exactly so the results are
directly comparable to D27/D27b's cache-miss judging.

WHY: LangCache's mechanical compose@1 (2.75-8.25% across two runs) looked
low enough to question whether the metric itself is too strict for a
semantic cache -- the same question D27 already asked and answered for the
learned cache's own misses. This applies the identical judge protocol to
LangCache's misses instead: full conversation context, the gold reference,
and LangCache's ACTUAL retrieved response (not a hypothetical) -- so the
"is 8% really bad" question gets the same rigor D27 already gave the cache.

USAGE
-----
    python sft/eval/build_langcache_judge_sample.py --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="outputs/probes/response/langcache_baseline_mid.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sft/eval/data/langcache_judge_sample.jsonl")
    args = ap.parse_args(argv)

    d = json.load(open(args.inp, encoding="utf-8"))
    rows = d["rows"]
    misses = [r for r in rows if r.get("matched") and not r.get("full_hit")
             and r.get("predicted_text")]

    rng = random.Random(args.seed)
    sample = rng.sample(misses, min(args.n, len(misses)))

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps({
                "turn_id": r["turn_id"],
                "context": r["context"],
                "gold_text": r["gold_text"],
                "predicted_text": r["predicted_text"],
                "similarity": r["similarity"],
                "skeleton_hit": r["skeleton_hit"],
            }, ensure_ascii=False) + "\n")

    print(json.dumps({
        "wrote": args.out,
        "n_sampled": len(sample),
        "n_misses_total": len(misses),
        "n_matched_total": sum(1 for r in rows if r.get("matched")),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
