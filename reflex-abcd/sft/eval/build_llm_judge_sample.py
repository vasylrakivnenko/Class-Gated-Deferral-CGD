"""Sample compose@1 MISSES from the cache's own winning selector (the same
config select.json certified as headline_test_seen), pair each miss with its
gold answer, and write judge-ready records for an LLM-as-judge pass.

WHY THIS EXISTS
----------------
compose@1 is a strict mechanical match: the cache's predicted skeleton AND
every predicted template id must equal gold, or it counts as a miss. A miss
that way can still be a perfectly reasonable reply (different valid phrasing,
a benign reordering) or a genuinely wrong one (wrong action, misleading). This
script does not change or re-measure compose@1 -- it explains what the misses
LOOK like, using an LLM judge shown the gold answer as a reference, per user
request.

HOW THE PREDICTION IS REPRODUCED
---------------------------------
select.json's `selection.winners.compose` is the exact (arm, vectorizer)
config `mode_select` fit H5 and H7 with to produce headline_test_seen. This
script reuses that recorded config with run_response_probe's own `_h5_arm` /
`_h7_arm` (refit on full train, predicted on test_seen) so the "cache's
predicted answer" here is not a new model -- it is a byte-for-byte
reproduction of the certified selector, just kept instead of stripped
(mode_select strips `_pred_by_turn` / `_pred_by_position` before writing
select.json, because per-turn predictions are not part of the certified
summary -- see run_response_probe.py's `mode_select`, "compose FIRST: it
reads the per-turn predictions that are stripped below").

SAMPLE
------
Turns are drawn from the CONDITIONAL population (gold fully bank-coverable --
the same n=3985 headline_test_seen scores), restricted to compose@1 MISSES,
stratified by gold act-sequence tuple so the sample is not just repeats of
whichever failure mode is most common, seeded for reproducibility.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.build_llm_judge_sample --n 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-json", default="outputs/probes/response/select.json")
    ap.add_argument("--out", default="sft/eval/data/llm_judge_sample.jsonl")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args(argv)

    from reflex.config import load_config

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
    from probes.run_response_probe import (
        load_probe_cfg, load_split_rows, _load_bank, _h5_arm, _h7_arm,
        vspec_for_head, guard_cfg,
    )
    from probes.featurizer_api import load_featurizer, RenderSpec

    probe_cfg = load_probe_cfg("probes/probe.yaml")
    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])
    bank = _load_bank(cfg)
    bank_text = {t.template_id: t.text_delex for t in bank.templates}
    memory_guard = guard_cfg(probe_cfg)
    ks = tuple(probe_cfg["probe"]["recall_ks"])

    sel = json.load(open(args.select_json, encoding="utf-8"))
    compose_win = sel["selection"]["winners"]["compose"]
    print(f"reusing certified config: {compose_win['arm']} "
          f"overrides={compose_win['vectorizer_overrides']}", file=sys.stderr)

    print("loading rows (train + test_seen)...", file=sys.stderr)
    rows_train = load_split_rows(cfg, "train", bank, None, "")
    rows_test = load_split_rows(cfg, "test_seen", bank, None, "")

    spec = RenderSpec(**compose_win["arm_detail"])
    vs_h5 = vspec_for_head(probe_cfg, "h5", compose_win["vectorizer_overrides"])
    vs_h7 = vspec_for_head(probe_cfg, "h7", compose_win["vectorizer_overrides"])

    print("refitting H5 (skeleton) on full train, predicting test_seen...", file=sys.stderr)
    h5 = _h5_arm(featurizer, rows_train["h5"], rows_test["h5"], spec, vs_h5, ks,
                n_boot=1, seed=args.seed, shuffle_seed=0, guard=memory_guard, controls=False)
    print("refitting H7 (template, per-act) on full train, predicting test_seen...", file=sys.stderr)
    from probes.run_response_probe import _equivalence
    equivalence = _equivalence(bank, cfg)
    h7 = _h7_arm(featurizer, rows_train["h7"], rows_test["h7"], spec, vs_h7, ks,
                n_boot=1, seed=args.seed, shuffle_seed=0, equivalence=equivalence,
                guard=memory_guard, controls=False)

    pred_sk = h5["_pred_by_turn"]
    gold_sk = h5["_gold_by_turn"]
    pred_tp = h7["_pred_by_position"]

    h5_by_turn = {r.turn_id: r for r in rows_test["h5"]}
    h7_by_turn: dict[str, list] = defaultdict(list)
    for r in rows_test["h7"]:
        h7_by_turn[r.turn_id].append(r)

    misses = []
    hits = 0
    n_cond = 0
    for turn_id, h5row in h5_by_turn.items():
        positions = sorted(h7_by_turn.get(turn_id, []), key=lambda r: r.position)
        gold_tids = [p.gold_template_id for p in positions]
        fully_covered = bool(gold_tids) and all(gold_tids)
        if not fully_covered:
            continue
        n_cond += 1

        pred_tids = [pred_tp.get((turn_id, p.position), "") for p in positions]
        skeleton_ok = pred_sk.get(turn_id) == gold_sk.get(turn_id) and gold_sk.get(turn_id) is not None
        full_hit = skeleton_ok and pred_tids == gold_tids and all(pred_tids)
        if full_hit:
            hits += 1
            continue

        gold_text = " ".join(bank_text.get(t, "") for t in gold_tids if t)
        pred_text = " ".join(bank_text.get(t, "") for t in pred_tids if t) or "(no confident prediction)"
        misses.append({
            "turn_id": turn_id,
            "convo_id": h5row.convo_id,
            "context": h5row.context.text,
            "gold_acts": list(h5row.gold_acts),
            "gold_text": gold_text,
            "predicted_text": pred_text,
            "skeleton_ok": skeleton_ok,
        })

    print(f"conditional n={n_cond}  hits={hits}  misses={len(misses)}  "
          f"compose@1={hits/n_cond:.4f}", file=sys.stderr)

    by_acts: dict[tuple, list] = defaultdict(list)
    for m in misses:
        by_acts[tuple(m["gold_acts"])].append(m)

    rng = random.Random(args.seed)
    groups = list(by_acts.items())
    rng.shuffle(groups)
    sample = []
    gi = 0
    while len(sample) < min(args.n, len(misses)) and groups:
        key, bucket = groups[gi % len(groups)]
        if bucket:
            sample.append(bucket.pop(rng.randrange(len(bucket))))
        if not bucket:
            groups.pop(gi % len(groups))
            if not groups:
                break
        else:
            gi += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in sample:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "wrote": args.out,
        "n_sampled": len(sample),
        "n_conditional_misses_total": len(misses) + sum(len(b) for _, b in by_acts.items() if b not in sample),
        "conditional_n": n_cond,
        "conditional_compose@1": hits / n_cond if n_cond else None,
        "n_distinct_act_sequences_in_sample": len({tuple(r["gold_acts"]) for r in sample}),
        "certified_config_reused": compose_win["arm"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
