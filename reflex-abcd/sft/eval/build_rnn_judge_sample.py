"""Sample compose@1 MISSES from the RNN-skeleton + TF-IDF-templates hybrid
(outputs/probes/response/rnn_skeleton_plus_h7.json) for the same LLM-judge
protocol used on the cache (D27/D27b), LangCache, and qwen3-0.6B.

USAGE
-----
    python sft/eval/build_rnn_judge_sample.py --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="outputs/probes/response/rnn_skeleton_plus_h7.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sft/eval/data/rnn_judge_sample.jsonl")
    args = ap.parse_args(argv)

    d = json.load(open(args.inp, encoding="utf-8"))
    rows = d["rows"]
    misses = [r for r in rows if r.get("fully_covered") and not r.get("full_hit")
             and r.get("predicted_text")]

    rng = random.Random(args.seed)
    sample = rng.sample(misses, min(args.n, len(misses)))

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps({
                "turn_id": r["turn_id"], "context": r["context"],
                "gold_text": r["gold_text"], "predicted_text": r["predicted_text"],
                "skeleton_hit": r["skeleton_hit"],
            }, ensure_ascii=False) + "\n")

    print(json.dumps({"wrote": args.out, "n_sampled": len(sample),
                      "n_misses_total": len(misses)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
