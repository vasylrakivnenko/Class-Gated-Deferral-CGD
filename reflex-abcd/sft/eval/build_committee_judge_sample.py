"""Build an LLM-judge sample from the committee-gate's UNANIMOUS-bucket
compose@1 MISSES (all 3 methods -- primary H5, RNN, n-gram -- agreed on the
same skeleton, but the mechanical exact-match metric still scored it wrong).
This is the most informative slice: if unanimous agreement is a genuinely
strong signal, its remaining "misses" should skew toward near-miss
paraphrases rather than genuinely bad answers -- worth checking before
trusting the 34.2% conditional compose@1 number on that bucket as a ceiling.

Refits H5 (primary skeleton) + per-act H7 (shared template layer) once more
(same certified config as committee_gate_h7.py) since that script only
saved hit/miss flags, not predicted text.

TWO CONTEXT STRINGS, DELIBERATELY. The featurizer's rendering (every token
prefixed `r0|`..`r3|` by recency) is a CLASSIFIER input and is what the H7
models are scored against. The JUDGE is shown the raw conversation text, the
same string build_llm_judge_sample.py writes (`h5row.context.text`), because
a judge reading "agent|r3|how r3|may r3|i r3|help r3|you?" is not being asked
the same question as a judge reading "agent|how may i help you?" -- and this
sample's adequacy share is quoted beside that one.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.build_committee_judge_sample --n 100
"""

from __future__ import annotations

import argparse
import json
import random
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history-mode", choices=("free", "gold"),
                    default=HISTORY_MODE_DEFAULT,
                    help="free (default): the n-gram committee member conditions on its OWN "
                         "previous predictions. gold: teacher-forced on the test gold "
                         "skeletons -- the pre-2026-09-21 behaviour, a diagnostic only.")
    ap.add_argument("--out", default="sft/eval/data/committee_judge_sample.jsonl")
    ap.add_argument("--buckets", default="0",
                    help="comma-separated committee disagreement counts to include "
                         "(default 0 = unanimous only; '0,1,2' = every turn)")
    ap.add_argument("--restrict-turns", default="",
                    help="path to a JSON list of turn_ids to restrict to (e.g. the "
                         "turns another arm was also scored on, for a matched comparison)")
    args = ap.parse_args(argv)
    keep_buckets = {int(x) for x in args.buckets.split(",") if x.strip() != ""}
    restrict = None
    if args.restrict_turns:
        restrict = set(json.load(open(args.restrict_turns, encoding="utf-8")))

    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd")
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import load_split_rows, vspec_for_head, load_probe_cfg, _classifier
    from probes.featurizer_api import load_featurizer, RenderSpec
    from sft.eval.build_llm_judge_sample import _fit_per_act_h7, _predict_template_for_act
    import numpy as np

    cfg = load_config()
    bank = load_bank(cfg)
    bank_text = {t.template_id: t.text_delex for t in bank.templates}
    probe_cfg = load_probe_cfg("probes/probe.yaml")
    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])

    sel = json.load(open("outputs/probes/response/select.json", encoding="utf-8"))
    compose_win = sel["selection"]["winners"]["compose"]
    spec = RenderSpec(**compose_win["arm_detail"])
    vs_h5 = vspec_for_head(probe_cfg, "h5", compose_win["vectorizer_overrides"])
    vs_h7 = vspec_for_head(probe_cfg, "h7", compose_win["vectorizer_overrides"])

    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, "test_seen", bank, h4_indexer=None, cache_dir="")
    spaces = train_rows["spaces"]
    skeleton_acts = spaces.skeleton_acts

    print("refitting H5 (primary skeleton classifier)...", file=sys.stderr)
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

    print("fitting per-act H7 classifiers on full train...", file=sys.stderr)
    fitted_by_act = _fit_per_act_h7(featurizer, train_rows["h7"], spec, vs_h7)

    test_h7_by_turn = defaultdict(list)
    for r in test_rows["h7"]:
        test_h7_by_turn[r.turn_id].append(r)
    test_context_by_turn = {}
    for r in test_rows["h7"]:
        if r.turn_id not in test_context_by_turn:
            test_context_by_turn[r.turn_id] = featurizer.render_context(r.context, spec)
    # Raw, untagged conversation text -- the judge's input. Kept separate from
    # the recency-tagged rendering above, which is the classifier's input only.
    raw_context_by_turn = {r.turn_id: r.context.text for r in test_rows["h5"]}

    print("loading committee (RNN + n-gram) predictions for the unanimous filter...", file=sys.stderr)
    rnn_pred = json.load(open("outputs/probes/response/rnn_skeleton_full_predictions.json",
                              encoding="utf-8"))
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
    history_mode = args.history_mode
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

    print("scoring + collecting unanimous misses...", file=sys.stderr)
    candidates = []
    for r in ev:
        tid = r.turn_id
        p_primary = primary_pred[tid]
        members = [m for m in (rnn_pred.get(tid), ngram_pred.get(tid)) if m is not None]
        n_dis = sum(1 for m in members if m != p_primary)
        if n_dis not in keep_buckets:
            continue
        if restrict is not None and tid not in restrict:
            continue

        positions = sorted(test_h7_by_turn.get(tid, []), key=lambda p: p.position)
        gold_tids = tuple(p.gold_template_id for p in positions)
        fully_covered = bool(gold_tids) and all(gold_tids)
        if not fully_covered:
            continue

        pred_acts = skeleton_acts.get(p_primary, ()) if p_primary else ()
        context_text = test_context_by_turn.get(tid, "")
        pred_tids = tuple(_predict_template_for_act(fitted_by_act, act, context_text)
                          for act in pred_acts)
        full_hit = (pred_tids == gold_tids and bool(gold_tids) and all(pred_tids))
        if full_hit:
            continue  # only misses

        candidates.append({
            # judge-visible context is the RAW text, not the r-tagged rendering
            # that `context_text` (the classifier's input) carries
            "turn_id": tid, "context": raw_context_by_turn.get(tid, ""),
            "gold_text": " ".join(bank_text.get(t, "") for t in gold_tids),
            "predicted_text": " ".join(bank_text.get(t, "") for t in pred_tids if t),
        })

    rng = random.Random(args.seed)
    sample = rng.sample(candidates, min(args.n, len(candidates)))
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"wrote": args.out, "n_sampled": len(sample),
                      "buckets": sorted(keep_buckets),
                      "restricted_to_turns": len(restrict) if restrict else None,
                      "n_misses_total": len(candidates)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
