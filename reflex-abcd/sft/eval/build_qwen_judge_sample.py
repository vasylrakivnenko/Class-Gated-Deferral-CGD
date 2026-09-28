"""Sample compose@1 MISSES from the qwen3-0.6B unconstrained LLM arm (free-
text generation, no skeleton/template scaffolding), for the same LLM-judge
protocol already applied to the cache (D27/D27b) and to LangCache's misses --
same rigor, applied to this arm's "21.8% isn't good, right?" question too.

Reuses sft/eval/data/gen_unconstrained.jsonl (the raw generations),
sft/eval/data/eval_prompts_test_seen.jsonl (context + gold, built by
build_eval_prompts.py) and sft/eval/data/gen_unconstrained_scored.json (the
authoritative per-row compose@1 scoring produced by score_generations.py)
-- no rerun of generation needed, just resampling misses from data already
on disk.

MISS PREDICATE: the project's ONE compose@1 predicate -- score_generations.py's
bank lookup, read back per row as `full_hit` -- NOT raw string equality against
the gold text. Raw string equality is not the metric: a generation that words
the reply differently but resolves to the gold template ids IS a compose@1 hit,
and string equality would hand it to the judge as a "miss". Measured on this
exact data, the old string-equality filter kept 3,583 rows of which 466
(13.0%) were in fact full_hit=true, and 15 of the 100 rows it sampled were
compose@1 HITS -- which inflates the arm's judged adequacy share, since the
judge is being asked "how bad are this arm's MISSES?".

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
    ap.add_argument("--scored", default="sft/eval/data/gen_unconstrained_scored.json",
                    help="score_generations.py output for THESE generations -- supplies "
                         "the canonical bank-lookup full_hit/fully_covered per row")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sft/eval/data/qwen_judge_sample.jsonl")
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
            gen_text = gens.get(d["turn_id"])
            if gen_text is None:
                continue
            # "Is this a compose@1 miss?" is answered by the ONE scorer, never by
            # string equality: a differently-worded reply that resolves to the
            # gold template ids is a HIT and must not enter the judged miss pool.
            rec = scored.get(d["turn_id"])
            if rec is None or not rec.get("fully_covered"):
                continue
            if rec.get("full_hit"):
                n_scored_hits_excluded += 1
                continue
            gold_text = " ".join(g["text"] for g in gold)
            rows.append({"turn_id": d["turn_id"], "context": d["prompt"],
                        "gold_text": gold_text, "predicted_text": gen_text})

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
