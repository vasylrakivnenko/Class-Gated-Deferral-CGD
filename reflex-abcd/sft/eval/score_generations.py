"""Score generated replies against test_seen gold, for BOTH arms, on the
SAME predicate: does the generation resolve to the right bank template(s)?

WHY ONE SCORER FOR TWO DIFFERENT OUTPUT SHAPES
------------------------------------------------
The unconstrained arm emits free text (one utterance). The structured arm
emits "ACT | sentence" lines (a skeleton plan). They cannot be compared on raw
string equality -- wording varies even when the intent is identical, and the
structured arm's lines must first be split into acts.

The fix used throughout this project (D6: a baseline is only a baseline WITH
its configuration attached) is to score BOTH arms against the bank using the
IDENTICAL matching path `reflex.train._derive_turn_labels` uses to derive
dev/test gold in the first place:

  1. split the generation into sentences (reflex.compile.split_sentences)
  2. label each sentence's act (reflex.compile.label_acts)
  3. normalize (reflex.train._normalize_for_match) and look up
     (act, normalized_text) -> template_id in the SAME bank lookup table

This makes "did the generation land on template T000038" a well-defined
question for a raw string, independent of which arm produced it. A generation
that never appears in the bank scores as a genuine miss, which is the fair
reading of an unconstrained model going off-script.

METRIC: compose@1, the SAME metric select.json's headline_test_seen reports.
  - skeleton match: the generation's act SEQUENCE == gold_acts
  - full match (compose@1): skeleton matches AND every position's resolved
    template_id == the gold template_id at that position
Both numbers are reported, never one alone, matching headline_test_seen's own
reporting convention:
  - conditional:   turns with gold text at every position
  - unconditional: all turns

DENOMINATORS ARE THE GENERATED ROWS, NOT THE PROMPT POPULATION
--------------------------------------------------------------
A turn the model was never asked to generate for is NOT evidence about the
model, so it is not scored: it stays in `rows` with generated=false but enters
no headline denominator. On a full run (gen_unconstrained/gen_structured, one
generation per prompt) that is the whole 3985 / 8889 test_seen population and
the arm is directly comparable to select.json. On a partial run (a 50/500/1000
prompt subsample) the denominators are that subsample -- dividing by 3985/8889
there understates the arm by the coverage ratio (~8x-21x). The old
full-population reading is still emitted, explicitly labelled, as
`full_population_lower_bound`.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.score_generations \
        --generations sft/eval/data/gen_unconstrained.jsonl --arm unconstrained
    PYTHONPATH=src python -m sft.eval.score_generations \
        --generations sft/eval/data/gen_structured.jsonl --arm structured
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _build_bank_lookup(bank: Any, normalize) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for template in bank.templates:
        for form in (template.surface_forms or [template.text_delex]):
            lookup.setdefault((template.act, normalize(form)), template.template_id)
    return lookup


def _resolve(text: str, cfg: dict, lookup: dict, split_sentences, label_acts, normalize
            ) -> list[tuple[str, str]]:
    """One generated utterance -> [(act, resolved_template_id_or_empty), ...]."""
    pieces = split_sentences(text, cfg)
    if not pieces:
        return []
    acts = label_acts(pieces, cfg)
    out = []
    for act, sentence in zip(acts, pieces):
        tid = lookup.get((act, normalize(sentence)), "")
        out.append((act, tid))
    return out


def _resolve_structured(text: str, cfg: dict, lookup: dict, normalize) -> list[tuple[str, str]]:
    """Structured output is already 'ACT | sentence' lines -- split on those,
    not on sentence boundaries, and treat a malformed line as an unresolved
    position with act=='' so it can never accidentally match.
    """
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            out.append(("", ""))
            continue
        act, _, sentence = line.partition("|")
        act = act.strip().upper()
        sentence = sentence.strip()
        tid = lookup.get((act, normalize(sentence)), "")
        out.append((act, tid))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", required=True,
                    help="JSONL: {turn_id, generation} per line, aligned to "
                         "sft/eval/data/eval_prompts_test_seen.jsonl")
    ap.add_argument("--prompts", default="sft/eval/data/eval_prompts_test_seen.jsonl")
    ap.add_argument("--arm", choices=("unconstrained", "structured"), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args(argv)

    from reflex.config import load_config
    from reflex.compile import load_bank, label_acts, split_sentences
    from reflex.train import _normalize_for_match

    cfg = load_config()
    for ov in args.overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        cur = node[parts[-1]]
        node[parts[-1]] = type(cur)(val) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else val

    bank = load_bank(cfg)
    lookup = _build_bank_lookup(bank, _normalize_for_match)

    gold_by_turn: dict[str, dict] = {}
    with open(args.prompts, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            gold_by_turn[d["turn_id"]] = d

    gens: dict[str, str] = {}
    with open(args.generations, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            gens[d["turn_id"]] = d.get("generation", "")

    n_total = len(gold_by_turn)
    n_generated = 0
    n_cond = 0          # conditional AND generated: gold fully covered at every position
    n_cond_full = 0     # fully covered anywhere in the prompt population (3985 on test_seen)
    hit_cond = 0
    hit_skel_cond = 0
    hit_uncond = 0
    hit_skel_uncond = 0
    per_row = []

    for turn_id, rec in gold_by_turn.items():
        gold_acts = rec["gold_acts"]
        gold = rec["gold"]
        fully_covered = bool(gold) and all(g["text"] for g in gold)
        if fully_covered:
            n_cond_full += 1

        gen_text = gens.get(turn_id)
        if gen_text is None:
            # No generation for this turn: it is NOT scored. The row is kept so the
            # prompt population stays visible, but it enters NO headline denominator
            # -- a numerator can only ever draw on generated rows, so counting it
            # would understate the arm by the coverage ratio. The full-population
            # reading (un-generated == miss) is emitted separately below as
            # `full_population_lower_bound`.
            per_row.append({"turn_id": turn_id, "convo_id": rec["convo_id"],
                            "generated": False, "skeleton_hit": False, "full_hit": False,
                            "fully_covered": fully_covered})
            continue
        n_generated += 1

        if args.arm == "unconstrained":
            resolved = _resolve(gen_text, cfg, lookup, split_sentences, label_acts, _normalize_for_match)
        else:
            resolved = _resolve_structured(gen_text, cfg, lookup, _normalize_for_match)

        gen_acts = [a for a, _ in resolved]
        skeleton_hit = gen_acts == gold_acts
        full_hit = False
        if skeleton_hit and fully_covered:
            gold_tids = [g["template_id"] for g in gold]
            gen_tids = [t for _, t in resolved]
            full_hit = gen_tids == gold_tids and all(gen_tids)

        per_row.append({"turn_id": turn_id, "convo_id": rec["convo_id"],
                        "generated": True, "skeleton_hit": skeleton_hit,
                        "full_hit": full_hit, "fully_covered": fully_covered,
                        "gen_acts": gen_acts, "gold_acts": gold_acts})

        if fully_covered:
            n_cond += 1
            if full_hit:
                hit_cond += 1
            if skeleton_hit:
                hit_skel_cond += 1
        if full_hit:
            hit_uncond += 1
        if skeleton_hit:
            hit_skel_uncond += 1

    coverage = (n_generated / n_total) if n_total else 0.0
    n_convos = len({r["convo_id"] for r in per_row if r["generated"]})
    partial = n_generated < n_total

    _select = ("outputs/probes/response/select.json :: headline_test_seen.compose "
               "-- conditional.compose@1=0.2765 (n=3985), unconditional.compose@1=0.1240 (n=8889). "
               "Scored on the SAME bank lookup and SAME (act, normalized text) matching as "
               "reflex.train._derive_turn_labels.")
    if partial:
        compare_against = (
            _select + f" POPULATIONS DIFFER -- NOT directly comparable: this arm was generated for"
            f" only {n_generated} of the {n_total} test_seen prompt turns (coverage {coverage:.4f},"
            f" a contiguous prompt-file slice, not a random sample). The rates above are over"
            f" {n_cond} conditional turns / {n_generated} turns in {n_convos} conversations -- NOT"
            f" over 3985 / 8889. To compare against the cache, recompute select.json's rate on these"
            " same turn_ids; `full_population_lower_bound` carries the 3985/8889 denominators but is"
            " a lower bound on this arm, not its rate.")
    else:
        compare_against = (
            _select + f" This arm was generated for all {n_total} prompt turns, so its conditional"
            f" (n={n_cond}) and unconditional (n={n_generated}) populations are exactly the ones"
            " select.json reports and the two arms are directly comparable.")

    result = {
        "arm": args.arm,
        # Headline denominators below are GENERATED rows, not the prompt population.
        "n_total_turns": n_total,       # context only: size of the prompts file
        "n_generated": n_generated,
        "coverage": coverage,
        "n_conversations_scored": n_convos,
        "scored_population": (f"all {n_total} test_seen prompt turns" if not partial else
                              f"a {n_generated}-turn subsample of the {n_total} test_seen prompt turns"),
        "conditional": {
            "n": n_cond,
            "n_note": "conditional turns (gold text at every position) that HAVE a generation",
            "compose@1": (hit_cond / n_cond) if n_cond else None,
            "skeleton_only": (hit_skel_cond / n_cond) if n_cond else None,
        },
        "unconditional": {
            "n": n_generated,
            "n_note": "turns that HAVE a generation",
            "compose@1": (hit_uncond / n_generated) if n_generated else None,
            "skeleton_only": (hit_skel_uncond / n_generated) if n_generated else None,
        },
        "full_population_lower_bound": {
            "note": ("every un-generated prompt turn counted as a miss. Same denominators as "
                     "select.json (3985 / 8889 on test_seen), but whenever coverage < 1.0 this is "
                     "a LOWER BOUND on the arm, not its rate."),
            "conditional": {
                "n": n_cond_full,
                "compose@1": (hit_cond / n_cond_full) if n_cond_full else None,
                "skeleton_only": (hit_skel_cond / n_cond_full) if n_cond_full else None,
            },
            "unconditional": {
                "n": n_total,
                "compose@1": (hit_uncond / n_total) if n_total else None,
                "skeleton_only": (hit_skel_uncond / n_total) if n_total else None,
            },
        },
        "compare_against": compare_against,
    }
    if partial:
        print(f"WARNING: partial run -- {n_generated}/{n_total} prompt turns have a generation "
              f"(coverage {coverage:.4f}). Headline compose@1 is over the generated subsample "
              f"only and is NOT comparable to select.json's 3985/8889 populations.", file=sys.stderr)

    out_path = args.out or args.generations.replace(".jsonl", "_scored.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": per_row}, fh, indent=1)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
