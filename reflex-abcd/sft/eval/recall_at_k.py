"""Measure the cache's recall@k -- how often the correct answer is somewhere in
its top-k candidates, rather than at rank 1.

This bounds the ceiling of any rerank-style approach that hands a downstream
model the cache's shortlist: the model can only pick a right answer that is
actually in the list.

Reports, for k = 1/5/10/20, overall and split by committee-disagreement bucket:

  skeleton recall@k -- is the gold skeleton among H5's top-k?
                       (verifies D25's 90.2% figure, recorded there as unverified)
  compose  recall@k -- take each of H5's top-k skeletons, complete it with the
                       per-act H7 argmax templates (exactly what the real
                       decoder emits), and ask whether ANY of those k complete
                       responses exactly equals gold.

Also reports the oracle-skeleton template rate: hand the decoder the GOLD
skeleton and ask whether its argmax templates match gold. That isolates how
much of the compose gap is skeleton selection versus template selection.

USAGE
-----
    PYTHONPATH=src:. python -m sft.eval.recall_at_k
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-k", type=int, default=20)
    ap.add_argument("--out", default="outputs/probes/response/recall_at_k.json")
    ap.add_argument("--gate", default="outputs/probes/response/committee_gate_h7.json")
    args = ap.parse_args(argv)
    KS = [k for k in (1, 5, 10, 20) if k <= args.max_k]

    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd")
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import load_split_rows, vspec_for_head, load_probe_cfg, _classifier
    from probes.featurizer_api import load_featurizer, RenderSpec
    from sft.eval.build_llm_judge_sample import _fit_per_act_h7
    import numpy as np

    cfg = load_config()
    bank = load_bank(cfg)
    probe_cfg = load_probe_cfg("probes/probe.yaml")
    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])

    sel = json.load(open("outputs/probes/response/select.json", encoding="utf-8"))
    win = sel["selection"]["winners"]["compose"]
    spec = RenderSpec(**win["arm_detail"])
    vs_h5 = vspec_for_head(probe_cfg, "h5", win["vectorizer_overrides"])
    vs_h7 = vspec_for_head(probe_cfg, "h7", win["vectorizer_overrides"])

    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, "test_seen", bank, h4_indexer=None, cache_dir="")
    skeleton_acts = train_rows["spaces"].skeleton_acts

    print("fitting H5 (skeleton)...", file=sys.stderr)
    fit = [r for r in train_rows["h5"] if r.gold_skeleton_id]
    ev = [r for r in test_rows["h5"] if r.gold_skeleton_id]
    vectorizer = featurizer.make_vectorizer(vs_h5)
    X_fit = vectorizer.fit_transform([featurizer.render_context(r.context, spec) for r in fit])
    eval_texts = [featurizer.render_context(r.context, spec) for r in ev]
    X_eval = vectorizer.transform(eval_texts)
    model = _classifier(vs_h5, n_fit=X_fit.shape[0])
    model.fit(X_fit, [r.gold_skeleton_id for r in fit])
    classes = np.array(model.classes_)
    proba = model.predict_proba(X_eval)
    topk_idx = np.argsort(-proba, axis=1)[:, :args.max_k]
    ranked = {r.turn_id: list(classes[topk_idx[i]]) for i, r in enumerate(ev)}
    gold_skel = {r.turn_id: r.gold_skeleton_id for r in ev}
    ctx_by_turn = {r.turn_id: eval_texts[i] for i, r in enumerate(ev)}

    print("fitting per-act H7 (templates)...", file=sys.stderr)
    fitted_by_act = _fit_per_act_h7(featurizer, train_rows["h7"], spec, vs_h7)

    # Batch the H7 argmax per act across ALL eval turns at once: the decoder's
    # choice depends only on (act, context), never on which skeleton proposed
    # the act, so one transform per act replaces ~max_k lookups per turn.
    turn_ids = [r.turn_id for r in ev]
    texts_in_order = [ctx_by_turn[t] for t in turn_ids]
    argmax_tpl: dict = {}
    for act, entry in fitted_by_act.items():
        if not entry:
            continue
        vec_a, mdl_a = entry
        Xa = vec_a.transform(texts_in_order)
        pa = mdl_a.predict_proba(Xa)
        cls_a = np.array(mdl_a.classes_)
        best = cls_a[np.argmax(pa, axis=1)]
        argmax_tpl[act] = {t: best[i] for i, t in enumerate(turn_ids)}
        print(f"  act {act}: {len(cls_a)} classes", file=sys.stderr)

    def compose_for(skel, tid):
        acts = skeleton_acts.get(skel, ()) if skel else ()
        if not acts:
            return None
        out = []
        for a in acts:
            t = argmax_tpl.get(a, {}).get(tid, "")
            if not t:
                return None
            out.append(t)
        return tuple(out)

    gold_tpl_by_turn = defaultdict(list)
    for r in test_rows["h7"]:
        gold_tpl_by_turn[r.turn_id].append(r)

    bucket = {r["turn_id"]: r["n_dis"]
              for r in json.load(open(args.gate, encoding="utf-8"))["rows"]}

    print("scoring recall@k...", file=sys.stderr)
    recs = []
    for tid in turn_ids:
        positions = sorted(gold_tpl_by_turn.get(tid, []), key=lambda p: p.position)
        gold_tids = tuple(p.gold_template_id for p in positions)
        if not gold_tids or not all(gold_tids):
            continue  # conditional population only
        rk = ranked[tid]
        skel_rank = rk.index(gold_skel[tid]) + 1 if gold_skel[tid] in rk else None
        comp_rank = None
        for i, s in enumerate(rk):
            if compose_for(s, tid) == gold_tids:
                comp_rank = i + 1
                break
        recs.append({"turn_id": tid, "n_dis": bucket.get(tid),
                     "skeleton_rank": skel_rank, "compose_rank": comp_rank,
                     "oracle_template_hit": compose_for(gold_skel[tid], tid) == gold_tids})

    def block(rows, label):
        n = len(rows)
        if not n:
            return None
        d = {"label": label, "n": n}
        for k in KS:
            d[f"skeleton_recall@{k}"] = sum(
                1 for r in rows if r["skeleton_rank"] and r["skeleton_rank"] <= k) / n
            d[f"compose_recall@{k}"] = sum(
                1 for r in rows if r["compose_rank"] and r["compose_rank"] <= k) / n
        d["oracle_skeleton_template_rate"] = sum(
            1 for r in rows if r["oracle_template_hit"]) / n
        return d

    blocks = [block(recs, "ALL (conditional)")]
    for b, name in ((0, "unanimous (n_dis=0)"), (1, "n_dis=1"), (2, "both disagree (n_dis=2)")):
        blocks.append(block([r for r in recs if r["n_dis"] == b], name))
    blocks = [b for b in blocks if b]

    hdr = f"{'bucket':<26}{'n':>6}" + "".join(f"{'sk@'+str(k):>8}" for k in KS) \
          + "".join(f"{'cmp@'+str(k):>9}" for k in KS) + f"{'oracle-tpl':>12}"
    print("\n" + hdr)
    for b in blocks:
        line = f"{b['label']:<26}{b['n']:>6}"
        line += "".join(f"{b[f'skeleton_recall@{k}']*100:>7.1f}%" for k in KS)
        line += "".join(f"{b[f'compose_recall@{k}']*100:>8.1f}%" for k in KS)
        line += f"{b['oracle_skeleton_template_rate']*100:>11.1f}%"
        print(line)

    json.dump({"max_k": args.max_k, "blocks": blocks}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
