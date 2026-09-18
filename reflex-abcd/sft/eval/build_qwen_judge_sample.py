"""Sample compose@1 MISSES from the qwen3-0.6B unconstrained LLM arm (free-
text generation, no skeleton/template scaffolding), for the same LLM-judge
protocol already applied to the cache (D27/D27b) and to LangCache's misses --
same rigor, applied to this arm's "21.8% isn't good, right?" question too.

Reuses sft/eval/data/gen_unconstrained.jsonl (the raw generations) and
sft/eval/data/eval_prompts_test_seen.jsonl (context + gold, built by
build_eval_prompts.py) -- no rerun of generation needed, just resampling
misses from data already on disk.

USAGE
-----
    python sft/eval/build_qwen_judge_sample.py --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", default="sft/eval/data/gen_unconstrained.jsonl")
    ap.add_argument("--prompts", default="sft/eval/data/eval_prompts_test_seen.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sft/eval/data/qwen_judge_sample.jsonl")
    args = ap.parse_args(argv)

    gens: dict[str, str] = {}
    with open(args.generations, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            gens[d["turn_id"]] = d.get("generation", "")

    rows = []
    with open(args.prompts, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            gold = d.get("gold", [])
            fully_covered = bool(gold) and all(g["text"] for g in gold)
            if not fully_covered:
                continue
            gen_text = gens.get(d["turn_id"])
            if gen_text is None:
                continue
            gold_text = " ".join(g["text"] for g in gold)
            # crude "is this a miss" check: exact text match after normalization
            # (the real compose@1 scorer resolves via bank lookup -- this is only
            # a sampling filter, the judge doesn't care about exact match anyway)
            if gen_text.strip().lower() == gold_text.strip().lower():
                continue
            rows.append({"turn_id": d["turn_id"], "context": d["prompt"],
                        "gold_text": gold_text, "predicted_text": gen_text})

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"wrote": args.out, "n_sampled": len(sample),
                      "n_candidate_misses": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
