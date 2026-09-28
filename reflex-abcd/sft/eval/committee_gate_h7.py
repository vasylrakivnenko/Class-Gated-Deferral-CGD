"""Same committee-disagreement gate as committee_gate.py, but scored on
full compose@1 (H7 template step layered on the primary's predicted
skeleton) instead of skeleton-only accuracy -- the real headline metric,
not just the upstream signal.

Reuses the certified per-act H7 classifiers (_fit_per_act_h7 /
_predict_template_for_act from build_llm_judge_sample.py) applied to the
PRIMARY's (H5 TF-IDF+logreg) predicted skeleton -- same "predict the
abstract flow from structure, then fill in wording from context" pattern
ngram_skeleton_plus_h7.py and rnn_skeleton_plus_h7.py use for their own
predicted skeletons.

WHICH H5 THIS IS: the metric here is compose@1, so BOTH heads are refit with
select.json's COMPOSE winner -- exactly what the certified compose@1 path does
(probes/run_response_probe.py fits h5 and h7 at the compose winner). That is
why compose@1_conditional here reproduces headline_test_seen.compose. It also
means the primary is the COMPOSE-PATH H5 ({'C': 4.0, 'sublinear_tf': True}),
NOT the certified H5 head ({'min_df': 2, 'sublinear_tf': True}) that
committee_gate.py uses. So `skeleton_accuracy` below is the compose-path H5's
skeleton accuracy and must not be read as headline_test_seen.h5.accuracy
.recall@1, and the n_dis buckets here are not row-for-row the buckets of
committee_gate.json. The config actually fit is stamped into the summary as
`primary_config`.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.committee_gate_h7
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def _argmax(counter):
    """Deterministic argmax. Counter.most_common breaks a tie by INSERTION order,
    i.e. by the order train rows happened to be parsed, so a tied prediction would
    depend on row order rather than on any stated rule. Break ties lexicographically
    instead -- the same total-order key sft/eval/ngram_skeleton_baseline.py and
    src/reflex/arm_b0.py use. (Ported here 2026-09-21: the two standalone n-gram
    baselines were fixed on 09-19 but these committee sites were not, so the SAME
    model was behaving two different ways depending on which script ran it --
    698 of 8,889 predictions differ between the rules.)"""
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


HISTORY_MODE_DEFAULT = "free"  # see the history comment in the turn loop


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd")
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import load_split_rows, vspec_for_head, load_probe_cfg, _classifier
    from probes.featurizer_api import load_featurizer, RenderSpec
    from sft.eval.build_llm_judge_sample import _fit_per_act_h7, _predict_template_for_act
    import numpy as np

    cfg = load_config()
    bank = load_bank(cfg)
    probe_cfg = load_probe_cfg("probes/probe.yaml")
    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])

    sel = json.load(open("outputs/probes/response/select.json", encoding="utf-8"))
    # Deliberately the COMPOSE winner for both heads (see module docstring):
    # compose@1 is certified at this config, so the H5 fit here is the
    # compose-path H5, not the certified H5 head.
    compose_win = sel["selection"]["winners"]["compose"]
    spec = RenderSpec(**compose_win["arm_detail"])
    vs_h5 = vspec_for_head(probe_cfg, "h5", compose_win["vectorizer_overrides"])
    vs_h7 = vspec_for_head(probe_cfg, "h7", compose_win["vectorizer_overrides"])

    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, "test_seen", bank, h4_indexer=None, cache_dir="")
    spaces = train_rows["spaces"]
    skeleton_acts = spaces.skeleton_acts

    # ---- PRIMARY: H5 (TF-IDF+logreg), predicted skeleton + confidence ----
    print("refitting H5 (primary skeleton classifier, COMPOSE-winner config)...", file=sys.stderr)
    fit = [r for r in train_rows["h5"] if r.gold_skeleton_id]
    ev = [r for r in test_rows["h5"] if r.gold_skeleton_id]
    fit_texts = [featurizer.render_context(r.context, spec) for r in fit]
    eval_texts = [featurizer.render_context(r.context, spec) for r in ev]
    vectorizer = featurizer.make_vectorizer(vs_h5)
    X_fit = vectorizer.fit_transform(fit_texts)
    X_eval = vectorizer.transform(eval_texts)
    model = _classifier(vs_h5, n_fit=len(fit_texts))
    model.fit(X_fit, [r.gold_skeleton_id for r in fit])
    classes = list(model.classes_)
    proba = model.predict_proba(X_eval)
    top1_idx = np.argmax(proba, axis=1)
    primary_pred = {r.turn_id: classes[i] for r, i in zip(ev, top1_idx)}
    primary_conf = {r.turn_id: float(proba[j, i]) for j, (r, i) in enumerate(zip(ev, top1_idx))}
    gold_by_turn = {r.turn_id: r.gold_skeleton_id for r in ev}

    # ---- certified per-act H7 classifiers, shared across the "H7 layer" ----
    print("fitting per-act H7 classifiers on full train...", file=sys.stderr)
    fitted_by_act = _fit_per_act_h7(featurizer, train_rows["h7"], spec, vs_h7)

    test_h7_by_turn = defaultdict(list)
    for r in test_rows["h7"]:
        test_h7_by_turn[r.turn_id].append(r)
    test_context_by_turn = {}
    for r in test_rows["h7"]:
        if r.turn_id not in test_context_by_turn:
            test_context_by_turn[r.turn_id] = featurizer.render_context(r.context, spec)

    def full_hit_for(turn_id: str, pred_sk: str | None) -> tuple[bool, bool]:
        """Returns (fully_covered, full_hit) for a given predicted skeleton."""
        positions = sorted(test_h7_by_turn.get(turn_id, []), key=lambda p: p.position)
        gold_tids = tuple(p.gold_template_id for p in positions)
        fully_covered = bool(gold_tids) and all(gold_tids)
        pred_acts = skeleton_acts.get(pred_sk, ()) if pred_sk else ()
        context_text = test_context_by_turn.get(turn_id, "")
        pred_tids = tuple(_predict_template_for_act(fitted_by_act, act, context_text)
                          for act in pred_acts)
        full_hit = (fully_covered and pred_tids == gold_tids
                   and bool(gold_tids) and all(pred_tids))
        return fully_covered, full_hit

    # ---- committee members: RNN (saved) + n-gram (refit here, fast) ----
    rnn_pred = json.load(open("outputs/probes/response/rnn_skeleton_full_predictions.json",
                              encoding="utf-8"))

    print("fitting n-gram skeleton model...", file=sys.stderr)
    train_by_convo = defaultdict(list)
    for r in train_rows["h5"]:
        if r.gold_skeleton_id:
            train_by_convo[r.convo_id].append((r.turn_index, r.gold_skeleton_id))
    for c in train_by_convo.values():
        c.sort()
    MAX_ORDER = 4
    counts = {n: defaultdict(Counter) for n in range(1, MAX_ORDER + 1)}
    order0 = Counter()
    for convo in train_by_convo.values():
        seq = [sk for _, sk in convo]
        for i, sk in enumerate(seq):
            order0[sk] += 1
            for n in range(1, MAX_ORDER + 1):
                if i >= n:
                    counts[n][tuple(seq[i - n:i])][sk] += 1

    def predict_ngram(history):
        for n in range(MAX_ORDER, 0, -1):
            if len(history) < n:
                continue
            dist = counts[n].get(tuple(history[-n:]))
            if dist:
                return _argmax(dist)
        return _argmax(order0) if order0 else None

    test_by_convo = defaultdict(list)
    for r in test_rows["h5"]:
        test_by_convo[r.convo_id].append(r)
    for c in test_by_convo.values():
        c.sort(key=lambda r: r.turn_index)
    history_mode = HISTORY_MODE_DEFAULT
    ngram_pred = {}
    for convo_id, rows in test_by_convo.items():
        history = []
        for row in rows:
            ngram_pred[row.turn_id] = predict_ngram(history)
            # 2026-09-21: was `history.append(row.gold_skeleton_id)` -- teacher forcing on
            # the TEST conversation's gold skeleton ids. Those are LABELS, not observable
            # input: they are the very thing the system predicts, and no deployed gate can
            # have them. Free-running, this n-gram scores exactly the label-blind constant
            # (0.3139272271016311 conditional), so the committee member it provides was
            # contributing the answer key rather than a second opinion.
            if history_mode == "gold":
                history.append(row.gold_skeleton_id)
            elif ngram_pred[row.turn_id] is not None:
                history.append(ngram_pred[row.turn_id])

    # ---- score: primary's compose@1, bucketed by committee disagreement ----
    print("scoring primary compose@1 per committee bucket...", file=sys.stderr)
    rows_out = []
    for r in ev:
        tid = r.turn_id
        p_primary = primary_pred[tid]
        p_rnn = rnn_pred.get(tid)
        p_ngram = ngram_pred.get(tid)
        members = [m for m in (p_rnn, p_ngram) if m is not None]
        n_dis = sum(1 for m in members if m != p_primary)
        fully_covered, full_hit = full_hit_for(tid, p_primary)
        skel_hit = p_primary == gold_by_turn[tid]
        rows_out.append({
            "turn_id": tid, "n_dis": n_dis, "fully_covered": fully_covered,
            "full_hit": full_hit, "skeleton_hit": skel_hit,
            "primary_conf": primary_conf[tid],
        })

    def bucket_stats(rs):
        cond = [r for r in rs if r["fully_covered"]]
        n, ncond = len(rs), len(cond)
        return {
            "n": n, "n_conditional": ncond,
            "skeleton_accuracy": (sum(1 for r in rs if r["skeleton_hit"]) / n) if n else None,
            "compose@1_unconditional": (sum(1 for r in rs if r["full_hit"]) / n) if n else None,
            "compose@1_conditional": (sum(1 for r in cond if r["full_hit"]) / ncond) if ncond else None,
        }

    n = len(rows_out)
    buckets = {
        "n_dis=0 (unanimous)": [r for r in rows_out if r["n_dis"] == 0],
        "n_dis=1": [r for r in rows_out if r["n_dis"] == 1],
        "n_dis=2 (both disagree)": [r for r in rows_out if r["n_dis"] == 2],
    }
    result = {
        "method": "committee-disagreement gate, primary compose@1 (H7 layered on primary's "
                 "predicted skeleton) bucketed by committee (RNN, n-gram) disagreement count",
        "n": n,
        "primary_config": {
            "source": "outputs/probes/response/select.json :: selection.winners.compose "
                      "(applied to BOTH h5 and h7, as the certified compose@1 path does)",
            "arm": compose_win["arm"],
            "arm_detail": compose_win["arm_detail"],
            "vectorizer_overrides": compose_win["vectorizer_overrides"],
            "estimator": str(vs_h5.classifier),
            "skeleton_accuracy_is": "the COMPOSE-PATH H5's skeleton accuracy, NOT the "
                                    "certified H5 head's headline_test_seen.h5.accuracy"
                                    ".recall@1 (that head has its own winner; see "
                                    "committee_gate.json)",
        },
        "overall": bucket_stats(rows_out),
        "by_disagreement_count": {
            k: {**bucket_stats(v), "share": len(v) / n} for k, v in buckets.items()
        },
    }
    with open("outputs/probes/response/committee_gate_h7.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": rows_out}, fh, indent=1)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
