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

WHY GOLD PRIOR SKELETONS, NOT PREDICTED ONES
-----------------------------------------------
Every other turn-level measurement in this project (H5/H7/H4, D25) scores
each turn independently against the TRUE preceding context/state, not a
chain of the model's own predictions -- this keeps errors from compounding
into an unmeasurable moving target. This script follows that same
convention: a test turn's n-gram context is the GOLD skeleton_ids of
earlier turns in its own conversation, not this model's own prior guesses.

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_seen")
    ap.add_argument("--max-order", type=int, default=4)
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
                return dist.most_common(1)[0][0], n
        if order0:
            return order0.most_common(1)[0][0], 0
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
    modal_templates = {sk: votes.most_common(1)[0][0]
                       for sk, votes in skeleton_template_votes.items()}

    # ---- score on the test split ----
    test_by_convo: dict = defaultdict(list)
    for r in test_rows["h5"]:
        test_by_convo[r.convo_id].append(r)
    for convo in test_by_convo.values():
        convo.sort(key=lambda r: r.turn_index)

    n_cond = hit_cond = hit_skel_cond = 0
    n_total = hit_uncond = hit_skel_uncond = 0
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
        "method": "n-gram Markov model over SKELETON SEQUENCES (not text), backoff "
                 f"order {args.max_order}..0, paired with each skeleton's modal train "
                 "template-tuple for full compose@1 scoring",
        "split": args.split,
        "max_order": args.max_order,
        "conditional": {"n": n_cond, "compose@1": (hit_cond / n_cond) if n_cond else None,
                       "skeleton_only": (hit_skel_cond / n_cond) if n_cond else None},
        "unconditional": {"n": n_total, "compose@1": hit_uncond / n_total if n_total else None,
                          "skeleton_only": hit_skel_uncond / n_total if n_total else None},
        "order_used_distribution": dict(order_used_counter),
        "compare_against": {
            "learned_cache_TFIDF_logreg": {"conditional": 0.2765370138017566, "unconditional": 0.12397345033187085},
            "H5_skeleton_accuracy_D25": {"conditional": 0.432, "constant": 0.2632},
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
