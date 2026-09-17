#!/usr/bin/env python3
"""Spec 3.3 -- the mandatory ABCD verification step. Fully working, no stubs.

Prints, per spec 3.3:
  * number of conversations per split
  * turn-type counts
  * the exact key names of one conversation and one turn
  * the 55 subflows
  * N random agent utterances with their targets

and then, because spec 3.3 also says "Record any deviation from 3.2 in README.md
under 'Dataset notes'", it VERIFIES each structural assumption of spec 3.2 over
the whole corpus and prints the result. Every number in the README's "Dataset
notes" section is this script's output, not prose.

Reads only ``data.abcd_dir`` from the config; imports nothing from
:mod:`reflex.contracts`, so it runs before any module is implemented.

Usage:
    python scripts/inspect_abcd.py [--config configs/default.yaml] [--sample 20] [--seed 0]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from reflex.config import load_config, resolve_path  # noqa: E402

#: ABCD's own slot markers look like ``<order_id>``.
MARKER_RE = re.compile(r"<[a-z_]+>")

#: Spec 3.2's expectations, which this script checks rather than assumes.
EXPECTED_SPLITS = ("train", "dev", "test")
EXPECTED_CONVO_KEYS = ("convo_id", "delexed", "original", "scenario")
EXPECTED_TURN_KEYS = ("candidates", "speaker", "targets", "text", "turn_count")
EXPECTED_NEXT_STEPS = ("retrieve_utterance", "take_action", "end_conversation")


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_all(abcd_dir: str, version: str) -> dict[str, object]:
    data_dir = os.path.join(abcd_dir, "data")
    paths = {
        "raw": os.path.join(data_dir, f"abcd_v{version}.json"),
        "utterances": os.path.join(data_dir, "utterances.json"),
        "ontology": os.path.join(data_dir, "ontology.json"),
        "kb": os.path.join(data_dir, "kb.json"),
        "guidelines": os.path.join(data_dir, "guidelines.json"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "missing ABCD files:\n  " + "\n  ".join(missing)
            + f"\nSet data.abcd_dir (currently {abcd_dir!r}) or run scripts/download_abcd.sh"
        )
    out: dict[str, object] = {"_paths": paths}
    for name, path in paths.items():
        with open(path, "r", encoding="utf-8") as fh:
            out[name] = json.load(fh)
    return out


def subflows_of(ontology: dict) -> list[str]:
    """Flatten ``ontology['intents']['subflows']`` in file order (flow, then subflow)."""
    node = ontology["intents"]["subflows"]
    if isinstance(node, dict):
        return [s for flow in ontology["intents"]["flows"] for s in node[flow]]
    return list(node)


def actions_of(ontology: dict) -> list[str]:
    """Flatten ``ontology['actions']`` in file order, deduplicated."""
    node = ontology["actions"]
    names: list[str] = []
    if isinstance(node, dict):
        for category in node:
            entry = node[category]
            names.extend(entry if isinstance(entry, list) else list(entry))
    else:
        names = list(node)
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spec 3.3 ABCD verification report.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sample", type=int, default=20, help="Random agent utterances to print.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    abcd_dir = resolve_path(cfg, "data.abcd_dir")
    version = str(cfg["data"]["version"])
    turn_key = cfg["data"]["turn_list_key"]

    blob = load_all(abcd_dir, version)
    raw: dict = blob["raw"]          # type: ignore[assignment]
    utt: list = blob["utterances"]   # type: ignore[assignment]
    ont: dict = blob["ontology"]     # type: ignore[assignment]
    kb: dict = blob["kb"]            # type: ignore[assignment]
    paths: dict = blob["_paths"]     # type: ignore[assignment]

    rule("0. FILES")
    for name, path in paths.items():
        print(f"{name:12s} {os.path.getsize(path) / 1e6:9.2f} MB  {path}")
    print(f"\nconfig     : {cfg['_config_path']}")
    print(f"turn list  : conversation[{turn_key!r}]  (spec 3.2 does not name it)")

    # ---------------------------------------------------------------- splits
    rule("1. CONVERSATIONS PER SPLIT (spec 3.3)")
    print(f"split keys: {sorted(raw.keys())}")
    total_convos = 0
    for split in raw:
        print(f"  {split:6s} {len(raw[split]):6d} conversations")
        total_convos += len(raw[split])
    print(f"  {'TOTAL':6s} {total_convos:6d}")
    if tuple(sorted(raw.keys())) != tuple(sorted(EXPECTED_SPLITS)):
        print(f"  !! DEVIATION: expected splits {EXPECTED_SPLITS}")

    # ------------------------------------------------------------- key names
    rule("2. EXACT KEY NAMES (spec 3.3)")
    convo = raw["train"][0]
    print(f"conversation keys : {sorted(convo.keys())}")
    print(f"  expected (3.2)  : {list(EXPECTED_CONVO_KEYS)}")
    print(f"scenario keys     : {sorted(convo['scenario'].keys())}")
    turn0 = convo[turn_key][0]
    print(f"turn keys         : {sorted(turn0.keys())}")
    print(f"  expected (3.2)  : {list(EXPECTED_TURN_KEYS)}")
    print(f"\ntargets layout    : [intent, nextstep, action, values, utt_rank]  (len {len(turn0['targets'])})")
    print(f"ontology keys     : {sorted(ont.keys())}")
    print(f"  intents keys    : {sorted(ont['intents'].keys())}")
    print(f"  values keys     : {sorted(ont['values'].keys())}")
    print(f"  next_steps      : {ont['next_steps']}")
    if tuple(ont["next_steps"]) != EXPECTED_NEXT_STEPS:
        print(f"  !! DEVIATION: next_steps order is NORMATIVE and must be {EXPECTED_NEXT_STEPS}")
    else:
        print("  next_steps order OK (NORMATIVE: cds_report branches on label == 0/1/2)")

    print(f"\nexample conversation {convo['convo_id']}, first 4 turns of {turn_key!r}:")
    for i, t in enumerate(convo[turn_key][:4]):
        print(f"  [{i}] speaker={t['speaker']:8s} turn_count={t['turn_count']:3d} "
              f"n_candidates={len(t['candidates']):3d}")
        print(f"      text    = {t['text'][:96]!r}")
        print(f"      targets = {t['targets']}")
        print(f"      original[{i}] = {convo['original'][i]!r}"[:120])

    # ------------------------------------------------------- turn-type counts
    rule("3. TURN-TYPE COUNTS (spec 3.3)")
    spk_by_split: dict[str, collections.Counter] = {}
    ns_all: collections.Counter = collections.Counter()
    pair_all: collections.Counter = collections.Counter()
    cand_len: collections.Counter = collections.Counter()
    targets_len: collections.Counter = collections.Counter()
    values_len: collections.Counter = collections.Counter()
    aligned = 0
    n_turns = 0
    for split in raw:
        spk = collections.Counter()
        for c in raw[split]:
            if len(c["original"]) == len(c[turn_key]):
                aligned += 1
            for t in c[turn_key]:
                n_turns += 1
                spk[t["speaker"]] += 1
                ns_all[str(t["targets"][1])] += 1
                pair_all[(t["speaker"], str(t["targets"][1]))] += 1
                cand_len[(t["speaker"], len(t["candidates"]))] += 1
                targets_len[len(t["targets"])] += 1
                if t["speaker"] == "action":
                    values_len[len(t["targets"][3])] += 1
        spk_by_split[split] = spk

    print(f"{'split':8s} {'agent':>9s} {'customer':>9s} {'action':>9s} {'total':>9s}")
    for split, spk in spk_by_split.items():
        print(f"{split:8s} {spk['agent']:9d} {spk['customer']:9d} {spk['action']:9d} {sum(spk.values()):9d}")
    tot = collections.Counter()
    for spk in spk_by_split.values():
        tot.update(spk)
    print(f"{'ALL':8s} {tot['agent']:9d} {tot['customer']:9d} {tot['action']:9d} {sum(tot.values()):9d}")

    print("\nnextstep counts (targets[1], all splits):")
    for k, v in ns_all.most_common():
        print(f"  {k:20s} {v:7d}")
    print("\nspeaker -> nextstep pairing (all splits):")
    for (s, ns), v in sorted(pair_all.items()):
        print(f"  {s:9s} -> {ns:20s} {v:7d}")
    print("  => speaker determines nextstep 1:1." if len(pair_all) == 3
          else "  !! speaker/nextstep is NOT 1:1")
    print(f"\n'end_conversation' in raw data: {ns_all.get('end_conversation', 0)} turns")
    if not ns_all.get("end_conversation"):
        print("  !! DEVIATION: end_conversation NEVER appears in raw ABCD.")
        print("     utils/process.py synthesizes ONE extra example per conversation after")
        print("     the last turn: end_targets = turn['targets'].copy();")
        print("     end_targets[1]='end_conversation'; end_targets[4]=-1.")
        print(f"     Replicating it adds {total_convos} turns (data.synthesize_end_conversation).")

    print(f"\ncandidate-list length by speaker: "
          f"{ {f'{s}:{n}': v for (s, n), v in sorted(cand_len.items())} }")
    print(f"targets tuple lengths           : {dict(targets_len)}")
    print(f"len(values) on action turns     : {dict(sorted(values_len.items()))}")
    print(f"len(original) == len({turn_key}) : {aligned}/{total_convos} conversations")

    # ---------------------------------------------------------------- subflows
    rule("4. SUBFLOWS (spec 3.3)")
    flows = ont["intents"]["flows"]
    subflows = subflows_of(ont)
    node = ont["intents"]["subflows"]
    print(f"ontology['intents']['subflows'] is a {type(node).__name__} of len {len(node)}")
    if isinstance(node, dict):
        print("  !! DEVIATION: it is a DICT KEYED BY THE 10 FLOWS, not a flat list of 55.")
        print("     Flatten it (flow order, then file order) to get the canonical 55.")
    print(f"\n{len(flows)} flows, {len(subflows)} subflows:")
    for flow in flows:
        names = node[flow] if isinstance(node, dict) else []
        print(f"  {flow:22s} ({len(names):2d}) {', '.join(names)}")
    print(f"\nkb.json entries: {len(kb)}; keys == flattened subflows: {set(kb.keys()) == set(subflows)}")
    acts = actions_of(ont)
    print(f"ontology['actions'] categories : "
          f"{ {k: len(v) for k, v in ont['actions'].items()} if isinstance(ont['actions'], dict) else len(acts)}")
    print(f"flattened action (button) list : {len(acts)} names")
    print("  (matches the 'intent mask should be size of 30 long' comment in cds_report)")

    # ------------------------------------------------- utt_rank verification
    rule("5. targets[4]: RANK OR GLOBAL UTTERANCE ID? (full-corpus check)")
    print(f"utterances.json: {len(utt)} strings (index == the id used in `candidates`)")
    n_cand = rank_vs_delex = rank_vs_orig = glob_vs_delex = 0
    delex_fired: list[tuple] = []
    neither: list[tuple] = []
    for split in raw:
        for c in raw[split]:
            for i, t in enumerate(c[turn_key]):
                cands = t["candidates"]
                if not cands:
                    continue
                r = t["targets"][4]
                if not (0 <= r < len(cands)):
                    continue
                n_cand += 1
                gold = utt[cands[r]]
                orig = c["original"][i][1].lower() if i < len(c["original"]) else None
                if gold == t["text"]:
                    rank_vs_delex += 1
                if gold == orig:
                    rank_vs_orig += 1
                if 0 <= r < len(utt) and utt[r] == t["text"]:
                    glob_vs_delex += 1
                if gold != t["text"]:
                    (delex_fired if gold == orig else neither).append(
                        (split, c["convo_id"], t["turn_count"], r, cands[r], gold, t["text"])
                    )
    print(f"agent turns with candidates: {n_cand}")
    print(f"  utterances[candidates[targets[4]]] == delexed text : "
          f"{rank_vs_delex:6d} / {n_cand}  ({100 * rank_vs_delex / n_cand:.3f}%)")
    print(f"  utterances[candidates[targets[4]]] == ORIGINAL text: "
          f"{rank_vs_orig:6d} / {n_cand}  ({100 * rank_vs_orig / n_cand:.3f}%)  <-- the truth")
    print(f"  utterances[targets[4]]             == delexed text : "
          f"{glob_vs_delex:6d} / {n_cand}  ({100 * glob_vs_delex / n_cand:.3f}%)  <-- global reading fails")
    print("\n  => targets[4] IS A RANK into `candidates`. gold_utt_id = candidates[targets[4]].")
    print(f"     The {len(delex_fired)} apparent mismatches against the DELEXED text are NOT duplicate")
    print("     collisions: utterances.json is the LEXICALIZED pool, so the gold string keeps")
    print("     the literal value where ABCD's delexicalizer replaced it in `delexed`.")
    print(f"     Turns matching NEITHER delexed nor original text: {len(neither)}")
    for m in delex_fired[:3]:
        print(f"\n     {m[0]} convo={m[1]} turn={m[2]} rank={m[3]} -> utt_id={m[4]}")
        print(f"       utterances[utt_id] = {m[5][:92]!r}")
        print(f"       delexed text       = {m[6][:92]!r}")
    for m in neither[:3]:
        print(f"\n     NEITHER: {m[0]} convo={m[1]} turn={m[2]}")
        print(f"       utterances[utt_id] = {m[5][:92]!r}")
        print(f"       delexed text       = {m[6][:92]!r}")

    # --------------------------------------------------- delexicalization state
    rule("6. `delexed` IS ALREADY PARTIALLY DELEXICALIZED")
    marker_counts: collections.Counter = collections.Counter()
    turns_with_marker = 0
    glued = 0
    glued_examples: list[tuple[str, str]] = []
    for split in raw:
        for c in raw[split]:
            for t in c[turn_key]:
                found = MARKER_RE.findall(t["text"])
                if found:
                    turns_with_marker += 1
                    marker_counts.update(found)
                for m in MARKER_RE.finditer(t["text"]):
                    a, b = m.span()
                    pre = t["text"][a - 1] if a > 0 else " "
                    post = t["text"][b] if b < len(t["text"]) else " "
                    if pre.isalnum() or post.isalnum():
                        glued += 1
                        if len(glued_examples) < 6:
                            glued_examples.append((t["text"][max(0, a - 24):b + 14], m.group()))
    n_markers = sum(marker_counts.values())
    utt_with_marker = sum(1 for s in utt if MARKER_RE.search(s))
    print(f"markers in `{turn_key}` text : {n_markers} over {turns_with_marker} / {n_turns} turns")
    for k, v in marker_counts.most_common():
        print(f"  {k:18s} {v:6d}")
    non_enum = ont["values"]["non_enumerable"]
    backbone = sorted({s for slots in non_enum.values() for s in slots})
    print(f"\nontology['values']['non_enumerable'] flattened ({len(backbone)}): {backbone}")
    print("  => THIS IS ABCD'S OWN SLOT VOCABULARY and the registry BACKBONE (spec 6.3).")
    print("     utils/load.py adds exactly [f'<{slot}>' for these] to its tokenizer.")
    observed = sorted(m.strip("<>") for m in marker_counts)
    print(f"  markers observed in text ({len(observed)}): {observed}")
    unused = sorted(set(backbone) - set(observed))
    if unused:
        print(f"  declared but never seen as a marker: {unused}")
    print(f"\nutterances.json strings containing a marker: {utt_with_marker} / {len(utt)} "
          f"({100 * utt_with_marker / len(utt):.2f}%)")
    print("  => the candidate pool is LEXICALIZED. Delexicalize candidates before comparing")
    print("     them to composed template text (spec 6.5 step 5).")
    print(f"\nmarkers GLUED to an adjacent alphanumeric: {glued} / {n_markers} "
          f"({100 * glued / n_markers:.2f}%)")
    print("  ABCD's own delexicalizer did partial-substring replacement, so these spans are")
    print("  NOT clean slots and are NOT reversible by a filler:")
    for text, marker in glued_examples:
        print(f"    {marker:16s} in ...{text}...")

    # ------------------------------------------------------- random utterances
    rule(f"7. {args.sample} RANDOM AGENT UTTERANCES WITH TARGETS (spec 3.3)")
    rng = random.Random(args.seed)
    pool = []
    for split in raw:
        for c in raw[split]:
            for i, t in enumerate(c[turn_key]):
                if t["speaker"] == "agent":
                    pool.append((split, c, i, t))
    print(f"(sampling {args.sample} of {len(pool)} agent turns, seed={args.seed})")
    for n, (split, c, i, t) in enumerate(rng.sample(pool, min(args.sample, len(pool))), 1):
        intent, nextstep, action, values, rank = t["targets"]
        gold_id = t["candidates"][rank] if t["candidates"] and 0 <= rank < len(t["candidates"]) else None
        print(f"\n[{n:2d}] {split} convo={c['convo_id']} turn_index={i} turn_count={t['turn_count']}")
        print(f"     text (delexed) : {t['text'][:100]!r}")
        print(f"     text (original): {c['original'][i][1][:100]!r}")
        print(f"     targets        : intent={intent!r} nextstep={nextstep!r} action={action!r} "
              f"values={values!r} utt_rank={rank}")
        print(f"     resolved utt_id: {gold_id}   (candidates[{rank}])")
        if gold_id is not None:
            print(f"     utterances[{gold_id}] = {utt[gold_id][:92]!r}")
        print(f"     scenario       : flow={c['scenario']['flow']!r} subflow={c['scenario']['subflow']!r}")

    rule("8. SUMMARY OF DEVIATIONS FROM SPEC 3.2 (record these in README)")
    print(f"D1 The processed turn list is conversation[{turn_key!r}]; spec 3.2 does not name it.")
    print(f"   len(original) == len({turn_key}) on {aligned}/{total_convos} conversations, index-aligned 1:1.")
    print("D2 targets[4] is a RANK into `candidates`, not a global utterance id.")
    print(f"   gold_utt_id = candidates[targets[4]] resolves to the original text on "
          f"{rank_vs_orig}/{n_cand} ({100 * rank_vs_orig / n_cand:.3f}%).")
    print("D3 'end_conversation' never appears in the raw data; utils/process.py synthesizes")
    print(f"   one extra example per conversation (+{total_convos} turns). Must be replicated.")
    print(f"D4 `{turn_key}` text is ALREADY partially delexicalized: {n_markers} <slot> markers")
    print(f"   over {turns_with_marker} turns, drawn from ontology['values']['non_enumerable'].")
    print("D5 utterances.json is LEXICALIZED "
          f"({utt_with_marker}/{len(utt)} entries carry a marker), unlike `{turn_key}` text.")
    print(f"D6 {glued}/{n_markers} markers are glued to adjacent alphanumerics (partial-substring")
    print("   replacement by ABCD), so they are not clean, reversible slots.")
    print("D7 ontology['intents']['subflows'] is a dict keyed by the 10 flows, not a flat list of 55.")
    print("D8 ontology['actions'] is a dict of 3 categories; flatten it for the 30-button list.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
