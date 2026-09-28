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

CORRECTED (previously a real bug): `_h7_arm`'s own `_pred_by_position` is
keyed by (turn_id, position) but computed WITHIN EACH ROW'S OWN GOLD ACT'S
per-act pool -- `_h7_arm` partitions both fit and eval rows by `r.act`, and
for an H7Row that is the GOLD act at that position, never whatever act a
mispredicted skeleton would actually imply. For a turn whose predicted
skeleton is WRONG (skeleton_ok=False), the real deployed decoder
(`select.py`'s `_score_templates`, which scores `(query, act_id)` using
only the query and the act id, no gold dependency) would score candidates
within the PREDICTED act's pool at each position, not the gold act's pool.
Reusing `_h7_arm`'s gold-conditioned `_pred_by_position` for a wrong-skeleton
row therefore showed an easier, non-deployable hypothetical text -- "what the
selector would say if the act at this position were the correct one" -- not
what the system actually outputs when it gets the skeleton wrong. Fixed here
by fitting one classifier PER ACT (`_fit_per_act_h7`, mirroring `_h7_arm`'s
own per-act fit loop) and, for a wrong-skeleton row, scoring each position
against the act the PREDICTED skeleton assigns there (looked up via
`spaces.skeleton_acts[pred_skeleton_id]`), not the row's own gold act. A
skeleton_ok=True row is unaffected: predicted and gold acts coincide there by
definition, so the old and new text are identical.

SAMPLE
------
Turns are drawn from the CONDITIONAL population (gold fully bank-coverable --
the same n=3985 headline_test_seen scores), restricted to compose@1 MISSES,
stratified by gold act-sequence tuple so the sample is not just repeats of
whichever failure mode is most common, seeded for reproducibility.

THE STRATIFICATION IS NOT FREE, SO EVERY ROW CARRIES ITS WEIGHT. Allocation
is round-robin over act-sequence groups -- roughly equal per stratum, NOT
proportional to population. ('ASK',) is 31.4% of the conditional population
(1251/3985) but gets ~3 of 100 rows, so an UNWEIGHTED verdict share over this
sample is an average over failure MODES, not over turns, and is not the same
statistic as the simple-random samples every other arm in the five-arm
comparison uses (build_langcache_judge_sample.py, build_rnn_judge_sample.py,
build_qwen_judge_sample.py, build_qwen_structured_judge_sample.py,
build_smollm2_judge_sample.py all use plain `rng.sample`).

The stratification is kept as the default -- it is what makes rare failure
modes visible at n=100 -- and each sampled row instead carries
`stratum_weight`, the standard design weight (stratum's share of the MISS pool
the sample is drawn from / rows drawn from that stratum; the per-row
`stratum_population_share` records that share). run_llm_judge.py reads it and
reports `share_*_weighted` beside the unweighted share. The weighted share is
the statistic comparable to the other arms -- BUT equal allocation makes it a
very noisy one: a handful of single-act strata hold most of the miss pool and
get 1-4 rows each, so the weighted share rests on roughly a dozen rows
(run_llm_judge.py prints the Kish effective n; ~16 of 100 on the D27b sample).
For a cross-arm number with real precision use `--design simple_random`, which
draws exactly as the other arms do (plain `rng.sample`, no weights needed).

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
from collections import Counter, defaultdict


def _fit_per_act_h7(featurizer, train_h7_rows, spec, vspec):
    """One (vectorizer, model) per act, mirroring `_h7_arm`'s own per-act fit
    loop exactly (same skip condition: fewer than 2 gold classes -> unfit),
    but the fitted objects are KEPT so any (row, act) pair can be scored --
    not just a row against its own gold act's pool.
    """
    from probes.run_response_probe import _classifier

    by_act_fit: dict = defaultdict(list)
    for r in train_h7_rows:
        if r.gold_template_id and r.act:
            by_act_fit[r.act].append(r)

    fitted: dict = {}
    for act, fit_rows in by_act_fit.items():
        golds = [r.gold_template_id for r in fit_rows]
        if len(set(golds)) < 2:
            fitted[act] = None
            continue
        texts = [featurizer.render_context(r.context, spec) for r in fit_rows]
        vectorizer = featurizer.make_vectorizer(vspec)
        X = vectorizer.fit_transform(texts)
        model = _classifier(vspec, n_fit=len(texts))
        model.fit(X, golds)
        fitted[act] = (vectorizer, model)
    return fitted


def _predict_template_for_act(fitted: dict, act: str, context_text: str) -> str:
    """What the real decoder would pick at one position, given the act the
    PREDICTED skeleton assigns there (not necessarily the row's gold act).
    Empty string means this act has fewer than 2 fit classes -- the same
    'unsupported' condition `_h7_arm` itself skips, not a new failure mode.
    """
    import numpy as np

    entry = fitted.get(act)
    if not entry:
        return ""
    vectorizer, model = entry
    X = vectorizer.transform([context_text])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    return classes[int(np.argmax(proba))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-json", default="outputs/probes/response/select.json")
    ap.add_argument("--out", default="sft/eval/data/llm_judge_sample.jsonl")
    ap.add_argument("--design", choices=("stratified", "simple_random"), default="simple_random",
                    help="stratified (default, D27/D27b): equal-allocation round-robin over "
                         "gold act-sequence strata, rows carry stratum_weight. simple_random: "
                         "plain rng.sample over the misses, the design every other arm's "
                         "builder uses -- the one to use for a cross-arm comparison.")
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

    print("fitting one H7 classifier PER ACT (for wrong-skeleton rows, to score "
          "the act the PREDICTED skeleton actually implies at each position, "
          "not the row's gold act)...", file=sys.stderr)
    spaces = rows_train["spaces"]
    skeleton_acts = spaces.skeleton_acts
    fitted_by_act = _fit_per_act_h7(featurizer, rows_train["h7"], spec, vs_h7)

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

        if not skeleton_ok:
            # The real decoder's own act at each position: what the PREDICTED
            # skeleton implies, not the row's gold act (the D27-correction
            # fix -- see module docstring "CORRECTED").
            pred_skeleton_id = pred_sk.get(turn_id)
            predicted_acts = skeleton_acts.get(pred_skeleton_id, ()) if pred_skeleton_id else ()
            context_text = featurizer.render_context(h5row.context, spec)
            real_pred_tids = [
                _predict_template_for_act(fitted_by_act, act, context_text)
                for act in predicted_acts
            ]
        else:
            real_pred_tids = pred_tids

        gold_text = " ".join(bank_text.get(t, "") for t in gold_tids if t)
        pred_text = " ".join(bank_text.get(t, "") for t in real_pred_tids if t) or "(no confident prediction)"
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
    # Snapshot stratum populations BEFORE the round-robin below pops from the
    # buckets -- these are the weights' numerators.
    stratum_size = {k: len(v) for k, v in by_acts.items()}

    rng = random.Random(args.seed)
    groups = list(by_acts.items())
    rng.shuffle(groups)
    sample = []
    gi = 0
    if args.design == "simple_random":
        # Same design as every other arm's builder; self-weighting, so no
        # stratum_weight is written and run_llm_judge.py's plain share applies.
        sample = rng.sample(misses, min(args.n, len(misses)))
        groups = []
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

    # Design weights. Allocation is ~equal per stratum, not proportional, so a
    # sampled row stands for stratum_size/n_drawn_from_that_stratum turns. The
    # divisor is the number of rows actually drawn from the stratum, which the
    # round-robin does NOT hold constant (1 to 4 per stratum on the D27b sample:
    # small strata are exhausted early) -- omitting it would leave the estimate
    # biased toward the under-drawn strata.
    # 2026-09-21: the DEFAULT flipped from "stratified" to "simple_random".
    # Equal allocation over the 47 act-sequence strata makes this arm's judged
    # share an average over failure MODES while every other arm's is an average
    # over TURNS, and the two were being ranked against each other. Measured on
    # one fixed 200-row sample of the same arm and judge, the estimator alone
    # moved p_appropriate from 0.6850 (turn-level) to 0.4835 (equal allocation).
    # The weights below make the stratified design recoverable, but with 1-4 rows
    # per stratum its Kish effective n is ~16, so it cannot carry a headline;
    # simple_random is what a cross-arm comparison needs.
    drawn_per_stratum = Counter(tuple(m["gold_acts"]) for m in sample)
    n_misses_total = len(misses)
    for row in (sample if args.design == "stratified" else []):
        key = tuple(row["gold_acts"])
        row["stratum_size"] = stratum_size[key]
        row["stratum_population_share"] = stratum_size[key] / n_misses_total
        row["n_drawn_from_stratum"] = drawn_per_stratum[key]
        row["stratum_weight"] = (
            stratum_size[key] / n_misses_total / drawn_per_stratum[key]
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in sample:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "wrote": args.out,
        "n_sampled": len(sample),
        "n_conditional_misses_total": len(misses),
        "conditional_n": n_cond,
        "conditional_compose@1": hits / n_cond if n_cond else None,
        "n_distinct_act_sequences_in_sample": len({tuple(r["gold_acts"]) for r in sample}),
        "n_distinct_act_sequences_in_population": len(stratum_size),
        "sampling_design": ("equal-allocation round-robin over gold act-sequence strata; "
                           "rows carry stratum_weight -- use run_llm_judge.py's "
                           "share_*_weighted to compare against simple-random arms"
                           if args.design == "stratified" else
                           "simple random sample of the misses (self-weighting; same "
                           "design as the other arms' builders)"),
        "sum_stratum_weight": (sum(r["stratum_weight"] for r in sample)
                               if args.design == "stratified" else None),
        "certified_config_reused": compose_win["arm"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
