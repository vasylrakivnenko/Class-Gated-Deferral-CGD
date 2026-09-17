"""Build the test_seen prompt set for the structured arm's eval, reusing the
SAME row-building path the probe used to produce `headline_test_seen`.

Why reuse rather than re-derive: `probes.run_response_probe.load_split_rows` ->
`probes.response_labels.build_rows` is what scored H5/H7/compose@1 on
`test_seen` (outputs/probes/response/select.json). Building a second,
independent gold-derivation path here would risk a silent mismatch with the
number this arm is being compared against -- exactly the D6 failure mode this
project keeps re-finding. One row-building path, two consumers (the probe's
TF-IDF fit, and this LLM eval).

Unlike sft/build_structured_sft_dataset.py (train-only, refuses other splits
because it joins directly against labels/train.jsonl), this script uses
`load_split_rows`, which already handles the dev/test derivation asymmetry
(reflex.train._derive_turn_labels; see DECISIONS and response_labels.py's
`dev_gold_provenance`).

Output: one JSON record per TURN (not per sentence position -- H7Rows are
joined back onto their parent H5Row by turn_id), with the prompt text and the
gold skeleton+templates needed to score compose@1 exactly the way
response_metrics.compose_at_1 does.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.build_eval_prompts --split test_seen --out sft/eval/data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_seen")
    ap.add_argument("--out", default="sft/eval/data")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args(argv)

    from reflex.config import load_config, get_dotted
    from reflex.compile import load_bank

    cfg = load_config()
    for ov in args.overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        cur = node[parts[-1]]
        node[parts[-1]] = type(cur)(val) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else val

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from probes.run_response_probe import load_split_rows
    import probes.response_labels as RL

    bank = load_bank(cfg)
    bank_text = {t.template_id: t.text_delex for t in bank.templates}
    rows = load_split_rows(cfg, args.split, bank, h4_indexer=None, cache_dir="")

    h5_by_turn = {r.turn_id: r for r in rows["h5"]}
    h7_by_turn: dict[str, list] = {}
    for r in rows["h7"]:
        h7_by_turn.setdefault(r.turn_id, []).append(r)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"eval_prompts_{args.split}.jsonl")

    n = 0
    n_no_gold_skeleton = 0
    with open(path, "w", encoding="utf-8") as fh:
        for turn_id, h5row in h5_by_turn.items():
            positions = sorted(h7_by_turn.get(turn_id, []), key=lambda r: r.position)
            gold_acts = list(h5row.gold_acts)
            gold_template_ids = [p.gold_template_id for p in positions]
            gold_lines = []
            for act, tid in zip(gold_acts, gold_template_ids):
                text = bank_text.get(tid, "")
                gold_lines.append({"act": act, "template_id": tid, "text": text})

            rec = {
                "turn_id": turn_id,
                "convo_id": h5row.convo_id,
                "turn_index": h5row.turn_index,
                "prompt": h5row.context.text,
                "gold_skeleton_id": h5row.gold_skeleton_id,
                "gold_acts": gold_acts,
                "gold": gold_lines,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if h5row.gold_skeleton_id is None:
                n_no_gold_skeleton += 1

    print(json.dumps({
        "wrote": path,
        "n_turns": n,
        "n_no_gold_skeleton": n_no_gold_skeleton,
        "note": ("n_no_gold_skeleton turns have an act tuple unseen in train (H5 OOV) -- "
                 "they still get a prompt+gold_acts (for the compose@1 skeleton-match check) "
                 "but no gold_skeleton_id."),
        "label_provenance": rows.get("label_provenance"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
