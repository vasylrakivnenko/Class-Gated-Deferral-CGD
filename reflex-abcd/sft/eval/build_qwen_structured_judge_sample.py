"""Sample compose@1 MISSES from the qwen3-0.6B STRUCTURED LLM arm (SFT on
skeleton+template-plan targets: "ACT | sentence" lines) -- the same LLM-judge
protocol already applied to the cache (D27/D27b), LangCache, the RNN hybrid,
and the qwen3 UNCONSTRAINED arm.

CANDIDATE TEXT: the "ACT | sentence" lines are joined into natural composed
text (act labels stripped) -- same convention as retrieval_rerank.py's
candidate_for() -- so the judge sees what a customer would actually read,
not the model's internal plan format.

MISS PREDICATE: the project's ONE compose@1 predicate -- score_generations.py's
bank lookup, read back per row as `full_hit` from
sft/eval/data/gen_structured_scored.json -- NOT raw string equality against the
gold text, exactly as in build_qwen_judge_sample.py, so the two qwen arms'
judged miss pools are defined the same way. Measured on this exact data the
old string-equality filter kept 3,030 rows of which 11 (0.36%) were in fact
full_hit=true; the canonical pool is 3,020 rows (one row whose sentence equals
the gold text under a WRONG act label is a compose@1 miss and is now kept).
None of the 100 rows the old filter sampled into
qwen_structured_judge_sample.jsonl was a hit, so that judged sample is not
contaminated -- but it is no longer reproducible from this script at seed 0,
because the pool it is drawn from changed.

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
    ap.add_argument("--scored", default="sft/eval/data/gen_structured_scored.json",
                    help="score_generations.py output for THESE generations -- supplies "
                         "the canonical bank-lookup full_hit/fully_covered per row")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sft/eval/data/qwen_structured_judge_sample.jsonl")
    args = ap.parse_args(argv)

    gens: dict[str, str] = {}
    with open(args.generations, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            gens[d["turn_id"]] = d.get("generation", "")

    # Canonical compose@1 scoring for these same generations (bank lookup via
    # score_generations.py), keyed by turn_id.
    scored = {r["turn_id"]: r
              for r in json.load(open(args.scored, encoding="utf-8"))["rows"]}

    rows = []
    n_scored_hits_excluded = 0
    with open(args.prompts, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            gold = d.get("gold", [])
            raw_gen = gens.get(d["turn_id"])
            if raw_gen is None:
                continue
            # "Is this a compose@1 miss?" is answered by the ONE scorer, never by
            # string equality (same predicate as build_qwen_judge_sample.py).
            rec = scored.get(d["turn_id"])
            if rec is None or not rec.get("fully_covered"):
                continue
            if rec.get("full_hit"):
                n_scored_hits_excluded += 1
                continue
            gen_text = _strip_acts(raw_gen)
            gold_text = " ".join(g["text"] for g in gold)
            rows.append({"turn_id": d["turn_id"], "context": d["prompt"],
                        "gold_text": gold_text, "predicted_text": gen_text,
                        "raw_generation": raw_gen})

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"wrote": args.out, "n_sampled": len(sample),
                      "n_candidate_misses": len(rows),
                      "n_compose_at_1_hits_excluded": n_scored_hits_excluded,
                      "miss_predicate": "score_generations.py full_hit (bank lookup)",
                      "scored_json": args.scored}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
