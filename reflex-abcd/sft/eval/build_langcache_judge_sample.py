"""Sample LangCache's compose@1 MISSES from
outputs/probes/response/langcache_baseline_mid.json for an LLM-judge pass,
over the SAME population the arm it is compared against is judged on: every
turn LangCache got wrong, including the ones where it returned nothing.

WHY: LangCache's mechanical compose@1 (2.75-8.25% across two runs) looked
low enough to question whether the metric itself is too strict for a
semantic cache -- the same question D27 already asked and answered for the
learned cache's own misses. This applies the same judge protocol to
LangCache's misses instead: full conversation context, the gold reference,
and LangCache's ACTUAL retrieved response (not a hypothetical) -- so the
"is 8% really bad" question gets the same rigor D27 already gave the cache.

THE DENOMINATOR IS TURNS FACED, NOT ANSWERS GIVEN. On langcache_baseline_mid
LangCache returns nothing above threshold on 347 of 800 queried turns, so 347
of its 778 misses (44.6%) are ABSTENTIONS. Sampling only from `matched` rows
silently drops them and turns the reported adequacy into "of the answers it
gave, how many were fine" -- while the learned cache it is quoted beside
never abstains (build_llm_judge_sample.py keeps its no-prediction rows via
`or "(no confident prediction)"`), so the two shares would not be the same
statistic. Abstentions are therefore INCLUDED, as explicit non-answers, and
every written row carries `matched` so the two sub-populations stay separable.
`--matched-only` still produces the answers-given view, but labels it: its
rows are written with `population: "matched_only"` and it must never be
quoted beside another arm without that label.

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
    ap.add_argument("--matched-only", action="store_true",
                    help="draw only from turns where LangCache returned a match, "
                         "excluding its 44.6%% abstentions. This is the "
                         "answers-given view, NOT the turns-faced view every other "
                         "arm is judged on -- rows are labelled population="
                         "'matched_only' and must not be quoted beside another arm.")
    args = ap.parse_args(argv)

    d = json.load(open(args.inp, encoding="utf-8"))
    rows = d["rows"]

    # Denominator is every turn the arm got wrong, abstentions included -- not
    # only the turns it chose to answer. An unmatched row has no `full_hit`,
    # `similarity` or `skeleton_hit` key at all, hence the .get()s below.
    misses = [r for r in rows if not r.get("full_hit")]
    if args.matched_only:
        misses = [r for r in misses if r.get("matched")]
    n_unmatched = sum(1 for r in misses if not r.get("matched"))

    rng = random.Random(args.seed)
    sample = rng.sample(misses, min(args.n, len(misses)))

    population = "matched_only" if args.matched_only else "all_misses"
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps({
                "turn_id": r["turn_id"],
                "context": r["context"],
                "gold_text": r["gold_text"],
                # an abstention is a non-answer the judge should see as one,
                # not a row to drop
                "predicted_text": r.get("predicted_text") or "(no match found)",
                "matched": bool(r.get("matched")),
                "population": population,
                "similarity": r.get("similarity"),
                "skeleton_hit": r.get("skeleton_hit"),
            }, ensure_ascii=False) + "\n")

    print(json.dumps({
        "wrote": args.out,
        "population": population,
        "n_sampled": len(sample),
        "n_misses_total": len(misses),
        "n_unmatched_included": n_unmatched,
        "n_unmatched_in_sample": sum(1 for r in sample if not r.get("matched")),
        "n_matched_total": sum(1 for r in rows if r.get("matched")),
        "n_rows_total": len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
