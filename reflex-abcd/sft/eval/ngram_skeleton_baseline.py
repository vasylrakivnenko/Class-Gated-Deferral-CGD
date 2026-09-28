"""N-gram Markov model over SKELETON SEQUENCES, not text -- a different kind
of retrieval baseline than TF-IDF/LangCache/RediSearch (D26/D31), which all
match on similarity of the raw conversation text. This one is symbolic:
each retrieve_utterance turn's skeleton_id is treated as a token, and
classic n-gram frequency counting (with backoff to lower orders when a
context is unseen) predicts the next skeleton from the last N skeleton_ids
in THIS SAME conversation -- catching conversations that follow the same
ABSTRACT FLOW (greet -> pull-up-account -> verify-identity -> ask-issue)
even when the literal wording is completely different, which text-similarity
methods cannot see by construction.

HISTORY MODE -- READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SCRIPT
--------------------------------------------------------------------
Until 2026-09-21 this script conditioned on the GOLD skeleton_ids of earlier
turns in the test conversation, justified as follows:

    "Every other turn-level measurement in this project (H5/H7/H4, D25)
     scores each turn independently against the TRUE preceding
     context/state, not a chain of the model's own predictions -- this
     keeps errors from compounding into an unmeasurable moving target."

That argument does not hold for THIS model, and the distinction is the whole
ballgame. H5/H7/H4 condition on the prior conversation TEXT, which genuinely
exists at inference time. This model conditions on prior skeleton IDS, which
are LABELS -- the very thing the system is asked to predict. Feeding them in
is not "scoring each turn independently against the true state", it is
handing the model the answer key for every turn but the current one. The
cache it is benchmarked against gets text only.

Measured 2026-09-21: free-running, this model's conditional skeleton@1 is
0.3139272271016311 -- BIT-IDENTICAL to the label-blind constant. Its own
predicted history never matches a training context, so backoff falls through
to the order-0 prior on essentially every turn. Teacher-forced it scored
0.4281. The entire apparent skill of this arm was the gold labels.

So --history-mode defaults to `free`. `gold` is retained to reproduce the
pre-2026-09-21 figure and must be labelled a teacher-forced diagnostic
wherever it appears, never compared against an arm that gets text only.

SKELETON -> FULL COMPOSE@1
----------------------------
The n-gram model predicts a skeleton_id only. To score compose@1 (skeleton
AND every template id), it is paired with the MODAL template-id tuple for
that skeleton in train (the most frequent complete template-tuple choice
train ever made for that skeleton) -- a simple, deterministic completion
step, not a second learned model.

BACKOFF
-------
For each test position, try the highest order (default 4) context first;
if that exact skeleton sequence was never seen in train, back off to
order-1 lower, down to order 0 (train's unconditional skeleton frequency --
the same thing D5's label-blind constant already measures).

USAGE
-----
    PYTHONPATH=src python -m sft.eval.ngram_skeleton_baseline --max-order 4
"""

from __future__ import annotations

import argparse
import json
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
                    help="free (default): condition on the model's OWN previous predictions, "
                         "which is all a deployed system has. gold: teacher-force on the test "
                         "conversation's true skeletons -- the pre-2026-09-21 behaviour, kept "
                         "only to reproduce the old figure. Free-running this model scores "
                         "exactly the label-blind constant.")
    ap.add_argument("--out", default="outputs/probes/response/ngram_skeleton_baseline.json")
    args = ap.parse_args(argv)

    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd")
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import load_split_rows

    cfg = load_config()
    bank = load_bank(cfg)
    bank_text = {t.template_id: t.text_delex for t in bank.templates}

    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, args.split, bank, h4_indexer=None, cache_dir="")

    train_h7_by_turn: dict = defaultdict(list)
    for r in train_rows["h7"]:
        train_h7_by_turn[r.turn_id].append(r)
    test_h7_by_turn: dict = defaultdict(list)
    for r in test_rows["h7"]:
        test_h7_by_turn[r.turn_id].append(r)

    # ---- per-conversation ordered skeleton sequences (train) ----
    train_by_convo: dict = defaultdict(list)
    for r in train_rows["h5"]:
        if r.gold_skeleton_id is not None:
            train_by_convo[r.convo_id].append((r.turn_index, r.gold_skeleton_id))
    for convo in train_by_convo.values():
        convo.sort()

    # ---- n-gram counts, orders 1..max_order, plus order-0 unconditional ----
    counts: dict = {n: defaultdict(Counter) for n in range(1, args.max_order + 1)}
    order0 = Counter()
    for convo in train_by_convo.values():
        seq = [sk for _, sk in convo]
        for i, sk in enumerate(seq):
            order0[sk] += 1
            for n in range(1, args.max_order + 1):
                if i >= n:
                    ctx = tuple(seq[i - n: i])
                    counts[n][ctx][sk] += 1

    def predict_skeleton(history: list) -> tuple:
        """Backoff prediction: returns (predicted_skeleton, order_used)."""
        for n in range(args.max_order, 0, -1):
            if len(history) < n:
                continue
            ctx = tuple(history[-n:])
            dist = counts[n].get(ctx)
            if dist:
                return _argmax(dist), n
        if order0:
            return _argmax(order0), 0
        return None, -1

    # ---- modal template-tuple per skeleton (train) ----
    train_h5_by_turn = {r.turn_id: r for r in train_rows["h5"]}
    skeleton_template_votes: dict = defaultdict(Counter)
    for turn_id, positions in train_h7_by_turn.items():
        row = train_h5_by_turn.get(turn_id)
        if row is None or row.gold_skeleton_id is None:
            continue
        tids = tuple(p.gold_template_id for p in sorted(positions, key=lambda p: p.position))
        if tids and all(tids):
            skeleton_template_votes[row.gold_skeleton_id][tids] += 1
    modal_templates = {sk: _argmax(votes) for sk, votes in skeleton_template_votes.items()}
    n_tied_modal = sum(1 for votes in skeleton_template_votes.values()
                       if len(votes) > 1
                       and sorted(votes.values(), reverse=True)[0]
                       == sorted(votes.values(), reverse=True)[1])

    # ---- score on the test split ----
    test_by_convo: dict = defaultdict(list)
    for r in test_rows["h5"]:
        test_by_convo[r.convo_id].append(r)
    for convo in test_by_convo.values():
        convo.sort(key=lambda r: r.turn_index)

    n_cond = hit_cond = hit_skel_cond = 0
    n_total = hit_uncond = hit_skel_uncond = 0
    n_skel = hit_skel_skelpop = 0
    order_used_counter = Counter()
    per_row = []

    for convo_id, rows in test_by_convo.items():
        history: list = []
        for row in rows:
            positions = sorted(test_h7_by_turn.get(row.turn_id, []), key=lambda p: p.position)
            gold_tids = tuple(p.gold_template_id for p in positions)
            fully_covered = bool(gold_tids) and all(gold_tids)

            pred_sk, order_used = predict_skeleton(history)
            order_used_counter[order_used] += 1
            pred_tids = modal_templates.get(pred_sk, ())

            skel_hit = pred_sk is not None and pred_sk == row.gold_skeleton_id
            full_hit = fully_covered and tuple(pred_tids) == gold_tids and bool(gold_tids)
            per_row.append({"turn_id": row.turn_id, "convo_id": convo_id,
                            "fully_covered": fully_covered, "order_used": order_used,
                            "predicted_skeleton": pred_sk, "skeleton_hit": skel_hit,
                            "full_hit": full_hit})

            n_total += 1
            # Denominator note: n_skel counts turns that HAVE a gold skeleton
            # (expected 8858) -- the population the certified H5 is scored on.
            # It is neither the 3,985 template-fully-covered turns nor all
            # 8,889 test turns, so it gets its own counter and its own key.
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

            # What the model is allowed to condition the NEXT turn on.
            #
            # free (default): its own prediction, which is all a deployed system has.
            # gold: the test conversation's true skeleton -- teacher forcing. This was
            #   the only behaviour until 2026-09-21 and it is why this arm looked
            #   skilful: free-running, its conditional skeleton@1 is
            #   0.3139272271016311, BIT-IDENTICAL to the label-blind constant, because
            #   its own predicted history never matches a training context and backoff
            #   falls through to the order-0 prior. Kept only so the old published
            #   figure stays reproducible; it is a diagnostic, never a headline.
            if args.history_mode == "gold":
                if row.gold_skeleton_id is not None:
                    history.append(row.gold_skeleton_id)
            elif pred_sk is not None:
                history.append(pred_sk)

    result = {
        "method": "n-gram Markov model over SKELETON SEQUENCES (not text), backoff "
                 f"order {args.max_order}..0, paired with each skeleton's modal train "
                 "template-tuple for full compose@1 scoring",
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
        "n_skeletons_with_tied_modal_tuple": n_tied_modal,
        "n_skeletons_with_modal_tuple": len(modal_templates),
        "order_used_distribution": dict(order_used_counter),
        "compare_against": {
            "learned_cache_TFIDF_logreg": {"conditional": 0.2765370138017566, "unconditional": 0.12397345033187085},
            "H5_skeleton_accuracy_D25": H5_SKELETON_BAR,
            "1NN_TFIDF_D26": {"conditional": 0.05244667503136763},
        },
        "note": ("order_used=0 means backoff hit the unconditional train skeleton "
                "frequency -- the SAME thing as the skeleton label-blind constant. "
                "A high share at order 0 means the n-gram signal rarely fired."),
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "rows": per_row}, fh, indent=1)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
