"""Recompute the certified cache's compose@1 with the TEMPLATE match relaxed
from exact template_id equality to the existing paraphrase-equivalence check
(embedding cosine similarity between the predicted and gold template's
canonical text), at a chosen threshold -- default 0.95, per user request
(the bank's own dedup used 0.92; 0.95 is stricter).

Skeleton match is NOT relaxed -- only the per-position TEMPLATE match, via
`TemplateEquivalence.paraphrase` (probes/response_metrics.py), reusing the
SAME embedding model the bank's own dedup step used (see D33's note on
paraphrase_density: 0.92 already only weakly captures true synonymy, e.g.
"can i have your account id?" vs "may i have your account id?" scores 0.880
and would NOT pass even the looser 0.92 threshold -- so a HIGHER 0.95
threshold will find even fewer near-misses than that, not more).

USES the exact certified config from select.json's compose winner -- refits
H5+H7 on full train, predicts on test_seen, same protocol as every other
number in this file.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.cache_paraphrase_relaxed --threshold 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--select-json", default="outputs/probes/response/select.json")
    ap.add_argument("--out", default="outputs/probes/response/cache_paraphrase_relaxed.json")
    args = ap.parse_args(argv)

    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd")
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import (
        load_split_rows, vspec_for_head, load_probe_cfg, _h5_arm, _h7_arm, _equivalence,
    )
    from probes.featurizer_api import load_featurizer, RenderSpec

    cfg = load_config()
    bank = load_bank(cfg)

    probe_cfg = load_probe_cfg("probes/probe.yaml")
    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])

    sel = json.load(open(args.select_json, encoding="utf-8"))
    compose_win = sel["selection"]["winners"]["compose"]
    spec = RenderSpec(**compose_win["arm_detail"])
    vs_h5 = vspec_for_head(probe_cfg, "h5", compose_win["vectorizer_overrides"])
    vs_h7 = vspec_for_head(probe_cfg, "h7", compose_win["vectorizer_overrides"])
    print(f"reusing certified config: {compose_win['arm']} "
          f"overrides={compose_win['vectorizer_overrides']}", file=sys.stderr)

    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, "test_seen", bank, h4_indexer=None, cache_dir="")
    ks = tuple(probe_cfg["probe"]["recall_ks"])

    print("refitting H5 (skeleton) on full train, predicting test_seen...", file=sys.stderr)
    h5 = _h5_arm(featurizer, train_rows["h5"], test_rows["h5"], spec, vs_h5, ks,
                n_boot=1, seed=0, shuffle_seed=0, controls=False)
    print("refitting H7 (template) on full train, predicting test_seen...", file=sys.stderr)
    equivalence = _equivalence(bank, cfg)
    h7 = _h7_arm(featurizer, train_rows["h7"], test_rows["h7"], spec, vs_h7, ks,
                n_boot=1, seed=0, shuffle_seed=0, equivalence=equivalence, controls=False)

    equivalence.threshold = args.threshold
    print(f"paraphrase threshold set to {args.threshold} (bank's own dedup used 0.92)",
          file=sys.stderr)

    pred_sk = h5["_pred_by_turn"]
    gold_sk = h5["_gold_by_turn"]
    pred_tp = h7["_pred_by_position"]

    test_h7_by_turn: dict = defaultdict(list)
    for r in test_rows["h7"]:
        test_h7_by_turn[r.turn_id].append(r)

    n_cond = hit_exact_cond = hit_relaxed_cond = 0
    n_total = hit_exact_uncond = hit_relaxed_uncond = 0
    n_positions_relaxed_only = 0  # positions that needed paraphrase, not exact

    for turn_id, positions in test_h7_by_turn.items():
        positions = sorted(positions, key=lambda p: p.position)
        gold_tids = [p.gold_template_id for p in positions]
        fully_covered = bool(gold_tids) and all(gold_tids)
        skel_ok = pred_sk.get(turn_id) == gold_sk.get(turn_id) and gold_sk.get(turn_id) is not None

        pred_tids = [pred_tp.get((turn_id, p.position), "") for p in positions]

        # A position whose gold_template_id is "" has NO surviving bank
        # template, so it can never be hit: the canonical predicate
        # (probes/run_response_probe.py `_composed._ok`) guards with
        # `if not gold ... return 0.0`. Without that guard the "" default
        # below compares equal to the "" gold and every uncovered turn scored
        # as a hit, inflating the UNCONDITIONAL column (the conditional one is
        # gated on fully_covered and was unaffected).
        ok_positions = all(g and p == g for p, g in zip(pred_tids, gold_tids))
        exact_hit = skel_ok and fully_covered and ok_positions

        relaxed_hit = skel_ok and fully_covered
        rescued_here = 0
        if relaxed_hit:
            for pred_t, gold_t in zip(pred_tids, gold_tids):
                if pred_t == gold_t:
                    continue
                if equivalence.paraphrase(pred_t, gold_t):
                    rescued_here += 1
                    continue
                relaxed_hit = False
                break
        if relaxed_hit:
            # only positions in turns that SURVIVED rescued anything
            n_positions_relaxed_only += rescued_here

        n_total += 1
        if exact_hit:
            hit_exact_uncond += 1
        if relaxed_hit:
            hit_relaxed_uncond += 1
        if fully_covered:
            n_cond += 1
            if exact_hit:
                hit_exact_cond += 1
            if relaxed_hit:
                hit_relaxed_cond += 1

    result = {
        "method": "certified cache (TF-IDF+logreg), template match relaxed from exact "
                 f"to paraphrase-equivalence (cosine >= {args.threshold}), skeleton match "
                 "unchanged (still exact)",
        "paraphrase_threshold": args.threshold,
        "conditional": {
            "n": n_cond,
            "compose@1_exact": (hit_exact_cond / n_cond) if n_cond else None,
            "compose@1_relaxed": (hit_relaxed_cond / n_cond) if n_cond else None,
            "miss_rate_exact": 1 - (hit_exact_cond / n_cond) if n_cond else None,
            "miss_rate_relaxed": 1 - (hit_relaxed_cond / n_cond) if n_cond else None,
        },
        "unconditional": {
            "n": n_total,
            "compose@1_exact": hit_exact_uncond / n_total if n_total else None,
            "compose@1_relaxed": hit_relaxed_uncond / n_total if n_total else None,
            "miss_rate_exact": 1 - hit_exact_uncond / n_total if n_total else None,
            "miss_rate_relaxed": 1 - hit_relaxed_uncond / n_total if n_total else None,
        },
        "n_positions_rescued_by_paraphrase_only": n_positions_relaxed_only,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
