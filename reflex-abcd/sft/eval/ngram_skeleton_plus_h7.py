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


# ---------------------------------------------------------------------------
# The learned cache's (certified H5) skeleton@1 bar, carried WITH its population.
# These are three DIFFERENT denominators and must never be stacked in one column:
#   n=3985 -- the template-fully-covered turns this file calls "conditional".
#             Source: outputs/probes/response/recall_at_k.json, block
#             "ALL (conditional)", skeleton_recall@1 = 0.5821831869510665.
#   n=8858 -- every test_seen turn that HAS a gold skeleton. This is H5's own
#             eval set (probes/run_response_probe.py: rows filtered on
#             gold_skeleton_id). Source: outputs/probes/response/select.json
#             headline_test_seen.h5 accuracy.recall@1 = 0.431700158049221,
#             n_eval = 8858.
#   n=8889 -- all test_seen turns, i.e. this file's "unconditional" population.
#             The 31 turns carrying no gold skeleton can never be hit, so the
#             n=8858 rate rescales exactly: 0.431700158049221 * 8858 / 8889
#             = 3824 hits / 8889 = 0.43019462256721785.
# The entry that used to live here was {"conditional": 0.432} -- the n=8858
# rate filed under the n=3985 label, which is what made the n-gram look 1.2
# points behind the cache and the RNN look 9 points ahead of it. On matched
# populations the cache leads on every one of the three.
H5_SKELETON_BAR = {
    "conditional_n3985": 0.5821831869510665,
    "gold_skeleton_n8858": 0.431700158049221,
    "unconditional_n8889": 0.43019462256721785,
    "label_blind_constant_n8858": 0.2630390607360578,
    "compare_like_with_like": (
        "conditional_n3985 <-> conditional.skeleton_only; "
        "gold_skeleton_n8858 <-> skeleton_population.skeleton_only; "
        "unconditional_n8889 <-> unconditional.skeleton_only"
    ),
    "sources": [
        "outputs/probes/response/recall_at_k.json -> blocks['ALL (conditional)'].skeleton_recall@1 (n=3985)",
        "outputs/probes/response/select.json -> headline_test_seen.h5.accuracy['recall@1'] (n_eval=8858)",
    ],
}
# ---------------------------------------------------------------------------


def _argmax(counter):
    """Deterministic argmax over a Counter.

    Counter.most_common resolves a tie by INSERTION order, i.e. by the order
    train rows happened to be parsed, so a tied prediction would depend on row
    order rather than on any stated rule. Break ties lexicographically on the
    key instead -- the same total-order key src/reflex/arm_b0.py already uses
    for its constant label.
    """
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_seen")
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--history-mode", choices=("free", "gold"), default="free",
                    help="free (default): condition on the model's OWN previous predictions. "
                         "gold: teacher-force on the test conversation's true skeletons -- the "
                         "pre-2026-09-21 behaviour, kept only to reproduce the old figure.")
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
                return _argmax(dist)
        return _argmax(order0) if order0 else None

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
    n_skel = hit_skel_skelpop = 0
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
            # Denominator note: n_skel counts turns that HAVE a gold skeleton
            # (expected 8858) -- the population the certified H5 is scored on,
            # distinct from both the 3,985 fully-covered and the 8,889 total.
            if row.gold_skeleton_id is not None:
                n_skel += 1
                if skel_hit:
                    hit_skel_skelpop += 1
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

            # See ngram_skeleton_baseline.py for why this defaults to the model's own
            # prediction: teacher-forcing on the test conversation's gold skeletons is
            # information a deployed system does not have, and it is the sole source of
            # this arm's apparent skill.
            if args.history_mode == "gold":
                if row.gold_skeleton_id is not None:
                    history.append(row.gold_skeleton_id)
            elif pred_sk is not None:
                history.append(pred_sk)

    result = {
        "method": "n-gram skeleton Markov model + certified per-act TF-IDF H7 template "
                 "classifiers (replaces the modal-template completion in "
                 "ngram_skeleton_baseline.py with the same text-based selector D25 used)",
        "split": args.split,
        "max_order": args.max_order,
        "history_mode": args.history_mode,
        "history_mode_note": (
            "free: the model conditions on its OWN previous predictions, which is all a "
            "deployed system has. gold: teacher-forced on the test conversation's true "
            "skeleton ids -- a DIAGNOSTIC only; those labels do not exist at inference and "
            "the arms this is compared against receive text alone. Free-running, this "
            "model scores exactly the label-blind constant (0.3139272271016311 conditional)."
        ),
        "conditional": {"n": n_cond, "compose@1": (hit_cond / n_cond) if n_cond else None,
                       "skeleton_only": (hit_skel_cond / n_cond) if n_cond else None},
        "unconditional": {"n": n_total, "compose@1": hit_uncond / n_total if n_total else None,
                          "skeleton_only": hit_skel_uncond / n_total if n_total else None},
        "skeleton_population": {
            "n": n_skel,
            "skeleton_only": (hit_skel_skelpop / n_skel) if n_skel else None,
            "note": "turns with a gold skeleton -- the SAME population the certified "
                    "H5's 0.4317 is measured over (select.json h5.n_eval=8858). This is "
                    "the only skeleton-only number directly comparable to it.",
        },
        "compare_against": {
            "learned_cache_TFIDF_logreg_D25": {"conditional": 0.2765370138017566, "unconditional": 0.12397345033187085},
            "ngram_skeleton_plus_modal_template": {
                "conditional": 0.10514429109159347, "unconditional": 0.04713691078861514,
                "stale": "measured before the argmax tie-break fix; re-run "
                         "sft.eval.ngram_skeleton_baseline to refresh",
            },
            "H5_skeleton_accuracy_D25": H5_SKELETON_BAR,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": per_row}, fh, indent=1)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
