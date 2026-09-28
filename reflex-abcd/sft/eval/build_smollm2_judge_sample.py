"""Build an LLM-judge sample from the smollm2_360m_8h checkpoint's compose@1
MISSES on the 500-prompt generation run (sft/eval/data/gen_smollm2_8h.jsonl,
scored by score_generations.py into
outputs/probes/response/smollm2_8h_score_500.json), restricted to the
CONDITIONAL (fully bank-coverable) population -- the same n=191 subset the
17.28% conditional compose@1 number was computed on.

Unlike build_committee_judge_sample.py (which judges a classifier's
template-id prediction), this checkpoint emits raw free text, so the
"predicted_text" shown to the judge is the model's own generation verbatim,
not a bank-template lookup.

USAGE
-----
    python sft/eval/build_smollm2_judge_sample.py --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-json", default="outputs/probes/response/smollm2_8h_score_500.json")
    ap.add_argument("--generations", default="sft/eval/data/gen_smollm2_8h.jsonl")
    ap.add_argument("--prompts", default="sft/eval/data/eval_prompts_test_seen.jsonl")
    ap.add_argument("--out", default="sft/eval/data/smollm2_8h_judge_sample.jsonl")
    args = ap.parse_args(argv)

    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd/src")
    from reflex.config import load_config
    from reflex.compile import load_bank

    cfg = load_config()
    bank = load_bank(cfg)
    bank_text = {t.template_id: t.text_delex for t in bank.templates}

    score = json.load(open(args.score_json, encoding="utf-8"))
    rows = [r for r in score["rows"] if r.get("generated")]

    generation_by_turn = {}
    with open(args.generations, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            generation_by_turn[r["turn_id"]] = r.get("generation", "")

    prompt_by_turn = {}
    with open(args.prompts, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            prompt_by_turn[r["turn_id"]] = r["prompt"]

    print("loading test_seen rows for gold text + context...", file=sys.stderr)
    from probes.run_response_probe import load_split_rows
    test_rows = load_split_rows(cfg, "test_seen", bank, h4_indexer=None, cache_dir="")
    from collections import defaultdict
    h7_by_turn = defaultdict(list)
    for r in test_rows["h7"]:
        h7_by_turn[r.turn_id].append(r)
    context_by_turn = {}
    for r in test_rows["h5"]:
        context_by_turn[r.turn_id] = r.context.text

    misses = []
    for r in rows:
        if not r.get("fully_covered"):
            continue
        if r.get("full_hit"):
            continue
        tid = r["turn_id"]
        positions = sorted(h7_by_turn.get(tid, []), key=lambda p: p.position)
        gold_tids = [p.gold_template_id for p in positions]
        gold_text = " ".join(bank_text.get(t, "") for t in gold_tids if t)
        misses.append({
            "turn_id": tid,
            "context": context_by_turn.get(tid, prompt_by_turn.get(tid, "")),
            "gold_acts": r.get("gold_acts"),
            "gold_text": gold_text,
            "predicted_text": generation_by_turn.get(tid, "") or "(empty generation)",
        })

    rng = random.Random(args.seed)
    sample = rng.sample(misses, min(args.n, len(misses)))
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in sample:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "wrote": args.out, "n_sampled": len(sample),
        "n_conditional_misses_total": len(misses),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
