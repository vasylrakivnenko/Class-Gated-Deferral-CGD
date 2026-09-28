"""How much of the learned cache's coverage rests on rare templates, train vs dev.

NEW FILE. Reads only. Companion to coverage_gap.py.

`compile.min_template_count: 2` cannot be lowered without a recompile, so the
templates it already dropped are not on disk and the m=1 point is unmeasurable
here. What IS measurable is the curve ABOVE the current threshold: post-hoc
remove every bank template whose occurrence count is < m and recompute coverage,
for train (from labels/train.jsonl, exact) and for the held-out splits (from the
train-procedure labels, variant D of coverage_gap.py).

Caveat stated rather than hidden: `count` is the POST-merge occurrence count of a
deduped template, so m here thresholds merged templates, not the pre-dedup rows
`min_template_count` actually filters. A real recompile at a different
min_template_count would also change which rows merge. This is an elasticity
probe, not a simulation of that recompile.
"""

from __future__ import annotations

import collections
import json
import os
import sys

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
from reflex.data import build_partitions, iter_agent_turns, load_ontology, load_raw_abcd  # noqa: E402

OUT = os.path.join("outputs", "probes", "analysis")
THRESHOLDS = (2, 3, 4, 5, 8, 16, 32)


def curve(positions, counts, turn_of):
    """positions: list of matched template id ("" == miss). Returns per-threshold rates."""
    out = {}
    turns = collections.defaultdict(list)
    for tid, t in zip(positions, turn_of):
        turns[t].append(tid)
    n_turns = len(turns)
    for m in THRESHOLDS:
        ok = [bool(t) and counts.get(t, 0) >= m for t in positions]
        full = sum(
            1 for rows in turns.values()
            if rows and all(bool(t) and counts.get(t, 0) >= m for t in rows)
        )
        out[str(m)] = {
            "sentence_coverage": sum(ok) / len(positions) if positions else None,
            "turn_coverage": full / n_turns if n_turns else None,
        }
    return out


def main() -> int:
    cfg = load_config(overrides=["data.context_turns_K=full"])
    ontology = load_ontology(cfg)
    registry = build_slot_registry(ontology, cfg)
    bank = load_bank(cfg)
    counts = {t.template_id: int(t.count) for t in bank.templates}
    lookup = {}
    for t in bank.templates:
        for form in t.surface_forms or [t.text_delex]:
            lookup.setdefault((t.act, _normalize_for_dedup(form)), t.template_id)

    report = {"thresholds": list(THRESHOLDS), "splits": {}}

    # --- train: exact, straight out of the labels compile wrote --------------
    pos, turn_of = [], []
    n_turns_train = 0
    for line in open(os.path.join("outputs", "compile", "labels", "train.jsonl")):
        row = json.loads(line)
        if row["nextstep"] != "retrieve_utterance":
            continue
        n_turns_train += 1
        for tid in row["template_ids"] or []:
            pos.append(tid)
            turn_of.append(row["turn_id"])
    report["splits"]["train"] = {
        "n_positions": len(pos), "n_turns": n_turns_train, "curve": curve(pos, counts, turn_of),
        "hit_count_histogram": {},
    }
    hist = collections.Counter()
    for tid in pos:
        if tid:
            c = counts.get(tid, 0)
            hist["2" if c == 2 else "3-4" if c <= 4 else "5-9" if c <= 9 else
                 "10-49" if c <= 49 else "50+"] += 1
    report["splits"]["train"]["hit_count_histogram"] = dict(hist)

    # --- held-out splits: train procedure (variant D) -------------------------
    partitions = build_partitions(cfg)
    raw = load_raw_abcd(cfg)
    scenarios = {}
    for convos in raw.values():
        for convo in convos:
            scenarios[int(convo["convo_id"])] = convo.get("scenario", {}) or {}

    for split in ("dev", "test_seen", "test_novel"):
        partition = getattr(partitions, split)
        flat, owner, cache = [], [], {}
        n_turns = 0
        for convo_id, turn_index, turn in iter_agent_turns(partition):
            if turn.nextstep != "retrieve_utterance":
                continue
            n_turns += 1
            if convo_id not in cache:
                cache[convo_id] = _conversation_sources(
                    scenarios.get(int(convo_id), {}), partition[convo_id], registry, cfg)
            tid = f"{split}:{convo_id}:{turn_index}"
            for s in split_sentences(turn.text, cfg):
                d = delexicalize_sentence(s, cache[convo_id], registry, cfg)[0]
                if d:
                    flat.append(d)
                    owner.append(tid)
        acts = label_acts(flat, cfg)
        pos = [lookup.get((a, _normalize_for_dedup(t)), "") for a, t in zip(acts, flat)]
        hist = collections.Counter()
        for tid in pos:
            if tid:
                c = counts.get(tid, 0)
                hist["2" if c == 2 else "3-4" if c <= 4 else "5-9" if c <= 9 else
                     "10-49" if c <= 49 else "50+"] += 1
        report["splits"][split] = {
            "n_positions": len(pos), "n_turns": n_turns,
            "curve": curve(pos, counts, owner),
            "hit_count_histogram": dict(hist),
        }
        print(split, "done", flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "min_count_elasticity.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
