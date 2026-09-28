"""Committee-disagreement confidence gate, adapted from
legalbench_map/phase2/gate_replay2.py's `committee_conf` gate to our
skeleton-selection task.

PRIMARY: the certified cache's H5 (TF-IDF+logreg) skeleton classifier --
this is the only member with a real calibrated predict_proba, since it's
the only one that's a logreg over TF-IDF features.

COMMITTEE (2 members, each independently trained/derived, no shared
parameters with the primary):
  - RNN skeleton classifier (already trained; predictions reused from
    outputs/probes/response/rnn_skeleton_full_predictions.json)
  - n-gram (order 1-4 backoff Markov) skeleton predictor, refit here (fast,
    pure counting, no ML)

GATE SCORE (same formula as gate_replay2.py's committee_conf):
    score = n_disagreeing_with_primary + (1 - p_pred)
Higher score = escalate to the big LLM. We don't fit a threshold here (no
labeled "LLM would get this right" ground truth to learn one against) --
instead we report the SAME diagnostic gate_replay2.py leads with: accuracy
split by discrete disagreement count, which is the number that decides
whether this whole approach is worth pursuing before any threshold-fitting.

METRIC: skeleton accuracy against gold_skeleton_id (not full compose@1) --
the three predictors are only directly comparable at the skeleton level
(H7 template selection is a separate, shared downstream step that isn't
being contested here).

USAGE
-----
    PYTHONPATH=src python -m sft.eval.committee_gate
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
    from probes.run_response_probe import load_split_rows, vspec_for_head, load_probe_cfg
    from probes.featurizer_api import load_featurizer, RenderSpec

    cfg = load_config()
    bank = load_bank(cfg)
    probe_cfg = load_probe_cfg("probes/probe.yaml")
    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])

    sel = json.load(open("outputs/probes/response/select.json", encoding="utf-8"))
    # The primary IS the certified H5 head, so it must be refit with H5's OWN
    # train-select winner ({'min_df': 2, 'sublinear_tf': True}). Using the
    # compose winner's overrides ({'C': 4.0, 'sublinear_tf': True}) here fits a
    # DIFFERENT classifier, whose skeleton accuracy is not select.json's
    # headline_test_seen.h5 number even though both are measured on the same
    # 8,858 gold-skeleton rows.
    h5_win = sel["selection"]["winners"]["h5"]
    spec = RenderSpec(**h5_win["arm_detail"])
    vs_h5 = vspec_for_head(probe_cfg, "h5", h5_win["vectorizer_overrides"])

    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, "test_seen", bank, h4_indexer=None, cache_dir="")

    # ---- PRIMARY: refit H5 (TF-IDF+logreg), keep predict_proba ----
    print("refitting H5 (certified TF-IDF+logreg) for primary + confidence...", file=sys.stderr)
    fit = [r for r in train_rows["h5"] if r.gold_skeleton_id]
    ev = [r for r in test_rows["h5"] if r.gold_skeleton_id]
    fit_texts = [featurizer.render_context(r.context, spec) for r in fit]
    eval_texts = [featurizer.render_context(r.context, spec) for r in ev]
    vectorizer = featurizer.make_vectorizer(vs_h5)
    X_fit = vectorizer.fit_transform(fit_texts)
    X_eval = vectorizer.transform(eval_texts)
    from probes.run_response_probe import _classifier
    model = _classifier(vs_h5, n_fit=len(fit_texts))
    model.fit(X_fit, [r.gold_skeleton_id for r in fit])
    classes = list(model.classes_)
    proba = model.predict_proba(X_eval)
    import numpy as np
    top1_idx = np.argmax(proba, axis=1)
    primary_pred = {r.turn_id: classes[i] for r, i in zip(ev, top1_idx)}
    primary_conf = {r.turn_id: float(proba[j, i]) for j, (r, i) in enumerate(zip(ev, top1_idx))}
    gold_by_turn = {r.turn_id: r.gold_skeleton_id for r in ev}

    # ---- committee member 1: RNN (already trained, load saved predictions) ----
    rnn_pred = json.load(open("outputs/probes/response/rnn_skeleton_full_predictions.json",
                              encoding="utf-8"))

    # ---- committee member 2: n-gram backoff Markov, refit here (fast) ----
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

    # ---- build committee agreement table ----
    rows_out = []
    for r in ev:
        tid = r.turn_id
        p_primary = primary_pred[tid]
        p_rnn = rnn_pred.get(tid)
        p_ngram = ngram_pred.get(tid)
        members = [m for m in (p_rnn, p_ngram) if m is not None]
        n_dis = sum(1 for m in members if m != p_primary)
        score = n_dis + (1.0 - primary_conf[tid])
        gold = gold_by_turn[tid]
        rows_out.append({
            "turn_id": tid, "gold": gold,
            "primary_pred": p_primary, "primary_conf": primary_conf[tid],
            "rnn_pred": p_rnn, "ngram_pred": p_ngram,
            "n_dis": n_dis, "score": score,
            "primary_correct": p_primary == gold,
        })

    n = len(rows_out)
    agree0 = [r for r in rows_out if r["n_dis"] == 0]
    dis1 = [r for r in rows_out if r["n_dis"] == 1]
    dis2 = [r for r in rows_out if r["n_dis"] == 2]

    def acc(rs):
        return (sum(1 for r in rs if r["primary_correct"]) / len(rs)) if rs else float("nan")

    # Same config, rows and protocol as select.json's certified H5 headline, so
    # the two SHOULD be one number. That is CHECKED here, not assumed: they only
    # agree when select.json was produced by the same code/data state as this
    # run, and a silent gap between them is exactly the defect this guards.
    overall = acc(rows_out)
    sel_h5 = (sel.get("headline_test_seen") or {}).get("h5") or {}
    sel_r1 = (sel_h5.get("accuracy") or {}).get("recall@1")
    comparable = isinstance(sel_r1, (int, float)) and sel_h5.get("n_eval") == n
    hit_delta = round((overall - sel_r1) * n) if comparable else None
    if hit_delta != 0:
        print(f"WARNING: primary skeleton accuracy {overall:.6f} (n={n}) does NOT match "
              f"select.json headline_test_seen.h5 recall@1={sel_r1!r} "
              f"(n_eval={sel_h5.get('n_eval')!r}); hit delta={hit_delta!r} rows. select.json "
              f"was produced by a different code/data state -- re-run the probe before "
              f"quoting either number next to the other.", file=sys.stderr)

    result = {
        "method": "committee-disagreement gate (adapted from legalbench_map/phase2/gate_replay2.py), "
                 "primary=certified H5 TF-IDF+logreg, committee={RNN skeleton, n-gram order1-4 backoff}",
        "n": n,
        "primary_config": {
            "source": "outputs/probes/response/select.json :: selection.winners.h5 "
                      "(the certified H5 head's OWN train-select winner, not the "
                      "compose winner) -- so primary_skeleton_accuracy_overall is "
                      "the same quantity as headline_test_seen.h5.accuracy.recall@1; "
                      "whether the two VALUES agree is recorded in vs_select_json",
            "arm": h5_win["arm"],
            "arm_detail": h5_win["arm_detail"],
            "vectorizer_overrides": h5_win["vectorizer_overrides"],
            "estimator": str(vs_h5.classifier),
        },
        "vs_select_json": {
            "select_json_h5_recall@1": sel_r1, "select_json_n_eval": sel_h5.get("n_eval"),
            "this_run": overall, "hit_delta_rows": hit_delta, "matches": hit_delta == 0,
        },
        "primary_skeleton_accuracy_overall": overall,
        "by_disagreement_count": {
            "n_dis=0 (unanimous)": {"n": len(agree0), "share": len(agree0) / n, "primary_accuracy": acc(agree0)},
            "n_dis=1": {"n": len(dis1), "share": len(dis1) / n, "primary_accuracy": acc(dis1)},
            "n_dis=2 (both disagree)": {"n": len(dis2), "share": len(dis2) / n, "primary_accuracy": acc(dis2)},
        },
    }
    with open("outputs/probes/response/committee_gate.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": rows_out}, fh, indent=1)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
