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
This is computed on the SAME conditional population (turns with gold text at
every position, n=3985) AND the unconditional population (all 8889, uncovered
counted as a miss) -- both numbers, never one alone, matching
headline_test_seen's own reporting convention.

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
    n_cond = 0          # conditional: gold fully covered at every position
    hit_cond = 0
    hit_skel_cond = 0
    hit_uncond = 0
    hit_skel_uncond = 0
    per_row = []

    for turn_id, rec in gold_by_turn.items():
        gold_acts = rec["gold_acts"]
        gold = rec["gold"]
        fully_covered = bool(gold) and all(g["text"] for g in gold)

        gen_text = gens.get(turn_id)
        if gen_text is None:
            # no generation for this turn -- counts as a miss everywhere
            per_row.append({"turn_id": turn_id, "convo_id": rec["convo_id"],
                            "generated": False, "skeleton_hit": False, "full_hit": False,
                            "fully_covered": fully_covered})
            if fully_covered:
                n_cond += 1
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

    result = {
        "arm": args.arm,
        "n_total_turns": n_total,
        "n_generated": n_generated,
        "conditional": {
            "n": n_cond,
            "compose@1": (hit_cond / n_cond) if n_cond else None,
            "skeleton_only": (hit_skel_cond / n_cond) if n_cond else None,
        },
        "unconditional": {
            "n": n_total,
            "compose@1": hit_uncond / n_total,
            "skeleton_only": hit_skel_uncond / n_total,
        },
        "compare_against": ("outputs/probes/response/select.json :: headline_test_seen.compose "
                           "-- conditional.compose@1=0.2765 (n=3985), unconditional.compose@1=0.1240 (n=8889). "
                           "Scored on the SAME bank lookup and SAME (act, normalized text) matching as "
                           "reflex.train._derive_turn_labels, so the two arms are directly comparable."),
    }
    out_path = args.out or args.generations.replace(".jsonl", "_scored.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": per_row}, fh, indent=1)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
