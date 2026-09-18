"""Combine the n-gram skeleton Markov model (ngram_skeleton_baseline.py) with
the certified TF-IDF H7 per-act template classifier, instead of pairing the
n-gram skeleton prediction with a crude "modal template" completion.

WHY: ngram_skeleton_baseline.py showed skeleton-only accuracy (42.0%) nearly
matching the trained TF-IDF+logreg classifier's H5 (43.2%), using pure
sequence-frequency counting -- no text at all. But its full compose@1
(10.5%) was far below the cache's 27.7%, because pairing a predicted
skeleton with its single most-common train template-tuple ignores the
actual conversation-specific content at each position. This script keeps
the n-gram model's skeleton prediction (the part that worked nearly as well
as text-based learning) and replaces the crude completion step with the
SAME certified per-act H7 template classifiers used elsewhere in this
project (reused via sft.eval.build_llm_judge_sample's _fit_per_act_h7 /
_predict_template_for_act, fit on the exact winning config select.json
certified) -- i.e., "predict the abstract flow from structure alone, then
fill in the specific wording from context, same as the trained pipeline
does for its own predicted skeleton."

Same compose@1 predicate, same conditional/unconditional populations, as
every other baseline in this project.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.ngram_skeleton_plus_h7 --max-order 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_seen")
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--select-json", default="outputs/probes/response/select.json")
    ap.add_argument("--out", default="outputs/probes/response/ngram_skeleton_plus_h7.json")
    args = ap.parse_args(argv)

    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd")
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import (
        load_split_rows, vspec_for_head, _load_bank as _lb,
    )
    from probes.featurizer_api import load_featurizer, RenderSpec
    from probes.run_response_probe import load_probe_cfg
    from sft.eval.build_llm_judge_sample import _fit_per_act_h7, _predict_template_for_act

    cfg = load_config()
    bank = load_bank(cfg)

    probe_cfg = load_probe_cfg("probes/probe.yaml")
    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])

    sel = json.load(open(args.select_json, encoding="utf-8"))
    compose_win = sel["selection"]["winners"]["compose"]
    spec = RenderSpec(**compose_win["arm_detail"])
    vs_h7 = vspec_for_head(probe_cfg, "h7", compose_win["vectorizer_overrides"])
    print(f"reusing certified config: {compose_win['arm']} "
          f"overrides={compose_win['vectorizer_overrides']}", file=sys.stderr)

    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, args.split, bank, h4_indexer=None, cache_dir="")
    spaces = train_rows["spaces"]
    skeleton_acts = spaces.skeleton_acts

    train_h7_by_turn: dict = defaultdict(list)
    for r in train_rows["h7"]:
        train_h7_by_turn[r.turn_id].append(r)
    test_h7_by_turn: dict = defaultdict(list)
    for r in test_rows["h7"]:
        test_h7_by_turn[r.turn_id].append(r)

    # ---- n-gram skeleton model (same as ngram_skeleton_baseline.py) ----
    train_by_convo: dict = defaultdict(list)
    for r in train_rows["h5"]:
        if r.gold_skeleton_id is not None:
            train_by_convo[r.convo_id].append((r.turn_index, r.gold_skeleton_id))
    for convo in train_by_convo.values():
        convo.sort()

    counts: dict = {n: defaultdict(Counter) for n in range(1, args.max_order + 1)}
    order0 = Counter()
    for convo in train_by_convo.values():
        seq = [sk for _, sk in convo]
        for i, sk in enumerate(seq):
            order0[sk] += 1
            for n in range(1, args.max_order + 1):
                if i >= n:
                    counts[n][tuple(seq[i - n: i])][sk] += 1

    def predict_skeleton(history: list):
        for n in range(args.max_order, 0, -1):
            if len(history) < n:
                continue
            dist = counts[n].get(tuple(history[-n:]))
            if dist:
                return dist.most_common(1)[0][0]
        return order0.most_common(1)[0][0] if order0 else None

    # ---- certified per-act H7 template classifiers, fit once on full train ----
    print("fitting per-act H7 classifiers on full train...", file=sys.stderr)
    fitted_by_act = _fit_per_act_h7(featurizer, train_rows["h7"], spec, vs_h7)

    test_context_by_turn = {}
    for r in test_rows["h7"]:
        if r.turn_id not in test_context_by_turn:
            test_context_by_turn[r.turn_id] = featurizer.render_context(r.context, spec)

    # ---- score ----
    test_by_convo: dict = defaultdict(list)
    for r in test_rows["h5"]:
        test_by_convo[r.convo_id].append(r)
    for convo in test_by_convo.values():
        convo.sort(key=lambda r: r.turn_index)

    n_cond = hit_cond = hit_skel_cond = 0
    n_total = hit_uncond = hit_skel_uncond = 0
    per_row = []

    for convo_id, rows in test_by_convo.items():
        history: list = []
        for row in rows:
            positions = sorted(test_h7_by_turn.get(row.turn_id, []), key=lambda p: p.position)
            gold_tids = tuple(p.gold_template_id for p in positions)
            fully_covered = bool(gold_tids) and all(gold_tids)

            pred_sk = predict_skeleton(history)
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
                            "full_hit": full_hit})

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

            if row.gold_skeleton_id is not None:
                history.append(row.gold_skeleton_id)

    result = {
        "method": "n-gram skeleton Markov model + certified per-act TF-IDF H7 template "
                 "classifiers (replaces the modal-template completion in "
                 "ngram_skeleton_baseline.py with the same text-based selector D25 used)",
        "split": args.split,
        "max_order": args.max_order,
        "conditional": {"n": n_cond, "compose@1": (hit_cond / n_cond) if n_cond else None,
                       "skeleton_only": (hit_skel_cond / n_cond) if n_cond else None},
        "unconditional": {"n": n_total, "compose@1": hit_uncond / n_total if n_total else None,
                          "skeleton_only": hit_skel_uncond / n_total if n_total else None},
        "compare_against": {
            "learned_cache_TFIDF_logreg_D25": {"conditional": 0.2765370138017566, "unconditional": 0.12397345033187085},
            "ngram_skeleton_plus_modal_template": {"conditional": 0.10514429109159347, "unconditional": 0.04713691078861514},
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
