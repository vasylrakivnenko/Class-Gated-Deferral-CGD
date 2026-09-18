"""Sample compose@1 MISSES from the qwen3-0.6B STRUCTURED LLM arm (SFT on
skeleton+template-plan targets: "ACT | sentence" lines) -- the same LLM-judge
protocol already applied to the cache (D27/D27b), LangCache, the RNN hybrid,
and the qwen3 UNCONSTRAINED arm.

CANDIDATE TEXT: the "ACT | sentence" lines are joined into natural composed
text (act labels stripped) -- same convention as retrieval_rerank.py's
candidate_for() -- so the judge sees what a customer would actually read,
not the model's internal plan format.

USAGE
-----
    python sft/eval/build_qwen_structured_judge_sample.py --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys


def _strip_acts(generation: str) -> str:
    parts = []
    for line in generation.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            _, _, sentence = line.partition("|")
            parts.append(sentence.strip())
        else:
            parts.append(line)
    return " ".join(p for p in parts if p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", default="sft/eval/data/gen_structured.jsonl")
    ap.add_argument("--prompts", default="sft/eval/data/eval_prompts_test_seen.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sft/eval/data/qwen_structured_judge_sample.jsonl")
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
            raw_gen = gens.get(d["turn_id"])
            if raw_gen is None:
                continue
            gen_text = _strip_acts(raw_gen)
            gold_text = " ".join(g["text"] for g in gold)
            if gen_text.strip().lower() == gold_text.strip().lower():
                continue
            rows.append({"turn_id": d["turn_id"], "context": d["prompt"],
                        "gold_text": gold_text, "predicted_text": gen_text,
                        "raw_generation": raw_gen})

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
