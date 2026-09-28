"""Decompose the train->dev drop in learned-cache template coverage.

NEW FILE. Reads only; writes JSON under outputs/probes/analysis/.
It never recompiles the bank and never fits a model. Cost is ONE corpus parse
(the same parse `probes.run_response_probe audit` does) plus sentence splitting,
delexicalization and act labelling on the non-train splits -- strictly lighter
than the audit, because it builds no ContextWindow and no H4 index.

Four label-derivation variants are run over the SAME sentences, so the deltas
between them are procedure effects and nothing else:

  A  SHIPPED   act(raw)   x  match norm(raw)    x  no sentence dropped
  B  DELEX-TEXT act(raw)  x  match norm(delex)  x  no sentence dropped
  C  +DELEX-ACT act(delex) x match norm(delex)  x  no sentence dropped
  D  TRAIN-PROC act(delex) x match norm(delex)  x  empty-delex sentences dropped,
                                                   turn denominator = all retrieve turns

A is `reflex.train._derive_turn_labels`. D is `compile.compile_bank`'s own
procedure, applied to a non-train split against the already-built bank.
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from reflex.compile import (  # noqa: E402
    _conversation_sources,
    _normalize_for_dedup,
    build_slot_registry,
    delexicalize_sentence,
    label_acts,
    load_bank,
    split_sentences,
)
from reflex.config import load_config  # noqa: E402
from reflex.data import (  # noqa: E402
    build_partitions,
    iter_agent_turns,
    load_ontology,
    load_raw_abcd,
)

OUT = os.path.join("outputs", "probes", "analysis")
SPLITS = ("dev", "test_seen", "test_novel")


def _rate(a, b):
    return (a / b) if b else None


def main() -> int:
    t0 = time.time()
    cfg = load_config(overrides=["data.context_turns_K=full"])
    ontology = load_ontology(cfg)
    registry = build_slot_registry(ontology, cfg)
    bank = load_bank(cfg)

    # (act, normalized form) -> template id, built exactly as compile_bank and
    # train._derive_turn_labels both build it.
    lookup: dict[tuple[str, str], str] = {}
    by_text: dict[str, set] = collections.defaultdict(set)
    for template in bank.templates:
        for form in template.surface_forms or [template.text_delex]:
            key = _normalize_for_dedup(form)
            lookup.setdefault((template.act, key), template.template_id)
            by_text[key].add(template.act)

    partitions = build_partitions(cfg)
    raw = load_raw_abcd(cfg)
    scenarios = {}
    for convos in raw.values():
        for convo in convos:
            scenarios[int(convo["convo_id"])] = convo.get("scenario", {}) or {}

    report = {
        "bank_hash": getattr(bank, "bank_hash", ""),
        "n_templates": len(bank.templates),
        "distinct_surface_forms": len(by_text),
        "novel_subflows": list(partitions.novel_subflows),
        "variants": {
            "A": "SHIPPED  act(raw) x norm(raw) x keep-all  == train._derive_turn_labels",
            "B": "act(raw) x norm(delex) x keep-all",
            "C": "act(delex) x norm(delex) x keep-all",
            "D": "TRAIN-PROC act(delex) x norm(delex) x drop-empty-delex, denom = all retrieve turns",
        },
        "splits": {},
    }

    for split in SPLITS:
        partition = getattr(partitions, split)
        turn_rows = []           # one per retrieve turn
        raw_flat, delex_flat = [], []
        sources_cache = {}
        for convo_id, turn_index, turn in iter_agent_turns(partition):
            if turn.nextstep != "retrieve_utterance":
                continue
            if convo_id not in sources_cache:
                sources_cache[convo_id] = _conversation_sources(
                    scenarios.get(int(convo_id), {}), partition[convo_id], registry, cfg
                )
            sources = sources_cache[convo_id]
            pieces = split_sentences(turn.text, cfg)
            delex = [delexicalize_sentence(s, sources, registry, cfg)[0] for s in pieces]
            row = {
                "turn_id": f"{split}:{convo_id}:{turn_index}",
                "intent": str(turn.intent),
                "n_chars": len(turn.text),
                "raw": pieces,
                "delex": delex,
                "off": len(raw_flat),
            }
            turn_rows.append(row)
            raw_flat.extend(pieces)
            delex_flat.extend(delex)

        # Act labelling: once over the raw sentences (what the shipped dev path
        # sees) and once over the non-empty delexicalized ones (what compile sees).
        acts_raw = label_acts(raw_flat, cfg)
        keep_idx = [i for i, d in enumerate(delex_flat) if d]
        acts_delex_kept = label_acts([delex_flat[i] for i in keep_idx], cfg)
        acts_delex = [""] * len(delex_flat)
        for i, a in zip(keep_idx, acts_delex_kept):
            acts_delex[i] = a

        stats = {v: {"pos": 0, "pos_gold": 0, "turns": 0, "full": 0} for v in "ABCD"}
        miss_reason = collections.Counter()
        act_flip = 0
        empty_delex = 0
        delex_changed_text = 0
        by_len = collections.Counter()
        cov_len = {"A": collections.Counter(), "D": collections.Counter()}
        by_int = collections.Counter()
        cov_int = {"A": collections.Counter(), "D": collections.Counter()}
        n_chars_total = 0
        # rescued / lost sets, A vs D, at the POSITION level (aligned on raw index)
        a_miss_d_hit = 0
        a_hit_d_miss = 0

        for row in turn_rows:
            off, pieces, delex = row["off"], row["raw"], row["delex"]
            n = len(pieces)
            n_chars_total += row["n_chars"]
            gold = {v: [] for v in "ABCD"}
            for j in range(n):
                i = off + j
                ar, ad, rt, dt = acts_raw[i], acts_delex[i], pieces[j], delex[j]
                nr, nd = _normalize_for_dedup(rt), _normalize_for_dedup(dt) if dt else ""
                if not dt:
                    empty_delex += 1
                elif nd != nr:
                    delex_changed_text += 1
                if dt and ad != ar:
                    act_flip += 1
                gold["A"].append(lookup.get((ar, nr), ""))
                gold["B"].append(lookup.get((ar, nd), "") if dt else "")
                gold["C"].append(lookup.get((ad, nd), "") if dt else "")
            gold["D"] = [gold["C"][j] for j in range(n) if delex[j]]

            for v in "ABC":
                stats[v]["pos"] += n
                stats[v]["pos_gold"] += sum(1 for g in gold[v] if g)
                if n:
                    stats[v]["turns"] += 1
                    stats[v]["full"] += int(all(gold[v]))
            stats["D"]["pos"] += len(gold["D"])
            stats["D"]["pos_gold"] += sum(1 for g in gold["D"] if g)
            stats["D"]["turns"] += 1                      # compile counts EVERY retrieve turn
            full_d = bool(gold["D"]) and all(gold["D"])
            stats["D"]["full"] += int(full_d)

            full_a = bool(gold["A"]) and all(gold["A"])
            b = min(n, 5)
            by_len[b] += 1
            cov_len["A"][b] += int(full_a)
            cov_len["D"][b] += int(full_d)
            by_int[row["intent"]] += 1
            cov_int["A"][row["intent"]] += int(full_a)
            cov_int["D"][row["intent"]] += int(full_d)

            for j in range(n):
                ha, hd = bool(gold["A"][j]), bool(gold["C"][j]) and bool(delex[j])
                a_miss_d_hit += int(hd and not ha)
                a_hit_d_miss += int(ha and not hd)
                if not hd and delex[j]:
                    nd = _normalize_for_dedup(delex[j])
                    if nd in by_text:
                        miss_reason["text_in_bank_under_another_act"] += 1
                    else:
                        miss_reason["text_absent_from_bank"] += 1
                elif not delex[j]:
                    miss_reason["delexicalized_to_empty"] += 1

        report["splits"][split] = {
            "n_convos": len(partition),
            "n_retrieve_turns": len(turn_rows),
            "n_raw_sentences": len(raw_flat),
            "sentences_per_turn": _rate(len(raw_flat), len(turn_rows)),
            "mean_turn_chars": _rate(n_chars_total, len(turn_rows)),
            "delex": {
                "sentences_delexicalized_to_empty": empty_delex,
                "sentences_whose_text_changed": delex_changed_text,
                "sentences_whose_act_flipped": act_flip,
            },
            "variants": {
                v: {
                    "n_positions": s["pos"],
                    "n_positions_with_gold": s["pos_gold"],
                    "sentence_coverage": _rate(s["pos_gold"], s["pos"]),
                    "n_turns": s["turns"],
                    "turns_fully_covered": s["full"],
                    "turn_coverage": _rate(s["full"], s["turns"]),
                }
                for v, s in stats.items()
            },
            "position_flips_A_to_D": {"A_miss_D_hit": a_miss_d_hit, "A_hit_D_miss": a_hit_d_miss},
            "D_miss_reasons": dict(miss_reason),
            "by_n_sentences": {
                str(k): {
                    "turns": by_len[k],
                    "A": _rate(cov_len["A"][k], by_len[k]),
                    "D": _rate(cov_len["D"][k], by_len[k]),
                }
                for k in sorted(by_len)
            },
            "by_intent": {
                k: {"turns": by_int[k], "A": _rate(cov_int["A"][k], by_int[k]),
                    "D": _rate(cov_int["D"][k], by_int[k])}
                for k in sorted(by_int)
            },
        }
        print(f"[{split}] done  t={time.time() - t0:.0f}s  "
              f"A={_rate(stats['A']['full'], stats['A']['turns']):.4f}  "
              f"D={_rate(stats['D']['full'], stats['D']['turns']):.4f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "coverage_gap.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print("wrote", path, f"in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
