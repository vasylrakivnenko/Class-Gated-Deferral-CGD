"""Pair the full-train RNN's skeleton predictions (rnn_skeleton_dryrun.py
--save-predictions) with the certified per-act TF-IDF H7 template
classifiers -- the same combination that took the n-gram model from 10.5%
to 20.2% compose@1. This is the RNN analogue of ngram_skeleton_plus_h7.py.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.rnn_skeleton_plus_h7 \
        --predictions outputs/probes/response/rnn_skeleton_full_predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_seen")
    ap.add_argument("--predictions", required=True,
                    help="turn_id -> predicted_skeleton_id JSON from rnn_skeleton_dryrun.py")
    ap.add_argument("--select-json", default="outputs/probes/response/select.json")
    ap.add_argument("--out", default="outputs/probes/response/rnn_skeleton_plus_h7.json")
    args = ap.parse_args(argv)

    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd")
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import load_split_rows, vspec_for_head, load_probe_cfg
    from probes.featurizer_api import load_featurizer, RenderSpec
    from sft.eval.build_llm_judge_sample import _fit_per_act_h7, _predict_template_for_act

    cfg = load_config()
    bank = load_bank(cfg)

    probe_cfg = load_probe_cfg("probes/probe.yaml")
    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])

    sel = json.load(open(args.select_json, encoding="utf-8"))
    compose_win = sel["selection"]["winners"]["compose"]
    spec = RenderSpec(**compose_win["arm_detail"])
    vs_h7 = vspec_for_head(probe_cfg, "h7", compose_win["vectorizer_overrides"])
    bank_text = {t.template_id: t.text_delex for t in bank.templates}
    print(f"reusing certified config: {compose_win['arm']} "
          f"overrides={compose_win['vectorizer_overrides']}", file=sys.stderr)

    rnn_preds = json.load(open(args.predictions, encoding="utf-8"))
    print(f"loaded {len(rnn_preds)} RNN skeleton predictions", file=sys.stderr)

    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, args.split, bank, h4_indexer=None, cache_dir="")
    spaces = train_rows["spaces"]
    skeleton_acts = spaces.skeleton_acts

    test_h7_by_turn: dict = defaultdict(list)
    for r in test_rows["h7"]:
        test_h7_by_turn[r.turn_id].append(r)

    print("fitting per-act H7 classifiers on full train...", file=sys.stderr)
    fitted_by_act = _fit_per_act_h7(featurizer, train_rows["h7"], spec, vs_h7)

    test_context_by_turn: dict = {}
    for r in test_rows["h7"]:
        if r.turn_id not in test_context_by_turn:
            test_context_by_turn[r.turn_id] = featurizer.render_context(r.context, spec)

    n_cond = hit_cond = hit_skel_cond = 0
    n_total = hit_uncond = hit_skel_uncond = 0
    per_row = []

    for row in test_rows["h5"]:
        positions = sorted(test_h7_by_turn.get(row.turn_id, []), key=lambda p: p.position)
        gold_tids = tuple(p.gold_template_id for p in positions)
        fully_covered = bool(gold_tids) and all(gold_tids)

        pred_sk = rnn_preds.get(row.turn_id)
        pred_acts = skeleton_acts.get(pred_sk, ()) if pred_sk else ()
        skel_hit = pred_sk is not None and pred_sk == row.gold_skeleton_id

        context_text = test_context_by_turn.get(row.turn_id, "")
        pred_tids = tuple(
            _predict_template_for_act(fitted_by_act, act, context_text)
            for act in pred_acts
        )
        full_hit = (fully_covered and pred_tids == gold_tids
                   and bool(gold_tids) and all(pred_tids))

        per_row.append({"turn_id": row.turn_id, "fully_covered": fully_covered,
                        "predicted_skeleton": pred_sk, "skeleton_hit": skel_hit,
                        "full_hit": full_hit,
                        "context": row.context.text,
                        "gold_text": " ".join(bank_text.get(t, "") for t in gold_tids),
                        "predicted_text": " ".join(bank_text.get(t, "") for t in pred_tids if t)
                                          or "(no confident prediction)"})

        n_total += 1
        if full_hit:
            hit_uncond += 1
        if skel_hit:
            hit_skel_uncond += 1
        if fully_covered:
            n_cond += 1
            if full_hit:
                hit_cond += 1
            if skel_hit:
                hit_skel_cond += 1

    result = {
        "method": "RNN (GRU, full-train, end-to-end) skeleton prediction + certified "
                 "per-act TF-IDF H7 template classifiers",
        "split": args.split,
        "conditional": {"n": n_cond, "compose@1": (hit_cond / n_cond) if n_cond else None,
                       "skeleton_only": (hit_skel_cond / n_cond) if n_cond else None},
        "unconditional": {"n": n_total, "compose@1": hit_uncond / n_total if n_total else None,
                          "skeleton_only": hit_skel_uncond / n_total if n_total else None},
        "compare_against": {
            "learned_cache_TFIDF_logreg_D25": {"conditional": 0.2765370138017566, "unconditional": 0.12397345033187085},
            "ngram_skeleton_plus_h7": {"conditional": 0.2015056461731493, "unconditional": 0.09033637079536506},
            "H5_skeleton_accuracy_D25": {"conditional": 0.432},
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": per_row}, fh, indent=1)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
