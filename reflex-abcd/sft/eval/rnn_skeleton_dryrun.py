"""DRY RUN: an RNN (GRU) encoder trained END TO END on stripped/lowercased/
stemmed/tokenized text, feeding a skeleton (H5) classification head -- the
untested condition D7 itself flagged: every neural arm tried so far used a
FROZEN, mean-pooled encoder, which destroys recency and word order. A
trainable sequence model, by construction, keeps order and can learn its
own recency weighting instead of losing it to mean-pooling.

WHY INPUT-ONLY PREPROCESSING
------------------------------
Stemming/lowercasing/tokenizing applies to the CONTEXT (input) only. The
model is a CLASSIFIER (predicts a skeleton_id, like H5 already does), not a
text generator, so there is no "output text" to mangle by stemming --
unlike a generative RNN, which would have to emit real English, not stems.

WHY 5,000 EXAMPLES, NOT THE FULL 43,159
------------------------------------------
Explicit dry run per request: cheap, fast (CPU, minutes not hours), to see
whether a trainable sequence encoder is even in the right ballpark before
committing to a full-scale training run. The label space (which skeletons
are even learnable) and the label-blind constant are BOTH computed on this
same 5,000-row subset, not the full train set, so the comparison is fair to
what this dry run actually saw (D5/D6 discipline: a baseline's constant must
match what it was fit on).

STEMMER: a small rule-based suffix stripper (no nltk in this venv, and a
dry run does not need real Porter-stemmer precision) -- ing/ed/ly/es/s
suffix stripping with short-word guards. Documented as approximate, not
claimed as a real Porter/Snowball stemmer.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.rnn_skeleton_dryrun --n-train 5000 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
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
        "conditional_n3985 <-> skeleton_accuracy.conditional.rnn; "
        "gold_skeleton_n8858 <-> skeleton_accuracy.skeleton_population.rnn; "
        "unconditional_n8889 <-> skeleton_accuracy.unconditional.rnn"
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


_SUFFIXES = ("ing", "edly", "ed", "ies", "es", "ly", "s")


def _stem(tok: str) -> str:
    if len(tok) <= 4:
        return tok
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def preprocess(text: str) -> list[str]:
    """lowercase -> tokenize -> stem. Applied to INPUT context only."""
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    return [_stem(t) for t in tokens]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="outputs/probes/response/rnn_skeleton_dryrun.json")
    ap.add_argument("--save-predictions", default=None,
                    help="optional path to write {turn_id: predicted_skeleton_id} JSON, "
                         "for a downstream skeleton+template combo")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    torch.manual_seed(args.seed)

    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd")
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import load_split_rows

    cfg = load_config()
    bank = load_bank(cfg)

    t0 = time.time()
    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, "test_seen", bank, h4_indexer=None, cache_dir="")
    print(f"  loaded, {time.time()-t0:.1f}s", file=sys.stderr)

    train_h5 = [r for r in train_rows["h5"] if r.gold_skeleton_id is not None]
    rng = random.Random(args.seed)
    train_sample = rng.sample(train_h5, min(args.n_train, len(train_h5)))
    print(f"  train sample: {len(train_sample)}", file=sys.stderr)

    # ---- label space + label-blind constant, computed on THIS subset ----
    label_counts = Counter(r.gold_skeleton_id for r in train_sample)
    labels = sorted(label_counts)
    label_to_idx = {sk: i for i, sk in enumerate(labels)}
    constant_skeleton = _argmax(label_counts)
    print(f"  label space: {len(labels)} skeletons", file=sys.stderr)

    # ---- vocab from stemmed tokens in the train subset ----
    tok_counts = Counter()
    train_tokens = []
    for r in train_sample:
        toks = preprocess(r.context.text)
        train_tokens.append(toks)
        tok_counts.update(toks)
    vocab = ["<pad>", "<unk>"] + [w for w, _ in tok_counts.most_common(args.vocab_size)]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    print(f"  vocab: {len(vocab)} tokens", file=sys.stderr)

    def encode(tokens: list[str]) -> list[int]:
        ids = [word_to_idx.get(t, 1) for t in tokens[: args.max_len]]
        if len(ids) < args.max_len:
            ids = ids + [0] * (args.max_len - len(ids))
        return ids

    class SkeletonDataset(Dataset):
        def __init__(self, rows, token_lists):
            self.x = [encode(t) for t in token_lists]
            self.y = [label_to_idx[r.gold_skeleton_id] for r in rows]

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return torch.tensor(self.x[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.long)

    train_ds = SkeletonDataset(train_sample, train_tokens)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    class GRUClassifier(nn.Module):
        def __init__(self, vocab_size, emb_dim, hidden_dim, n_labels):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, n_labels)

        def forward(self, x):
            e = self.emb(x)
            _, h = self.gru(e)
            return self.fc(h.squeeze(0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRUClassifier(len(vocab), args.emb_dim, args.hidden_dim, len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"training on {device}, {len(labels)}-way, {args.epochs} epochs...", file=sys.stderr)
    t1 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_correct = 0
        n_seen = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n_correct += (logits.argmax(1) == yb).sum().item()
            n_seen += xb.size(0)
        print(f"  epoch {epoch+1}/{args.epochs}  loss={total_loss/n_seen:.4f}  "
              f"train_acc={n_correct/n_seen:.4f}", file=sys.stderr)
    print(f"  trained in {time.time()-t1:.1f}s", file=sys.stderr)

    # ---- evaluate on test_seen ----
    test_h7_by_turn: dict = defaultdict(list)
    for r in test_rows["h7"]:
        test_h7_by_turn[r.turn_id].append(r)

    model.eval()
    n_cond = hit_cond = hit_cond_const = 0
    n_uncond = hit_uncond = hit_uncond_const = 0
    n_skel = hit_skel = hit_skel_const = 0
    predictions: dict = {}
    with torch.no_grad():
        for row in test_rows["h5"]:
            positions = test_h7_by_turn.get(row.turn_id, [])
            gold_tids = [p.gold_template_id for p in positions]
            fully_covered = bool(gold_tids) and all(gold_tids)
            toks = preprocess(row.context.text)
            x = torch.tensor([encode(toks)], dtype=torch.long).to(device)
            logits = model(x)
            pred_idx = int(logits.argmax(1).item())
            pred_sk = labels[pred_idx]
            predictions[row.turn_id] = pred_sk

            skel_hit = pred_sk == row.gold_skeleton_id
            const_hit = constant_skeleton == row.gold_skeleton_id

            n_uncond += 1
            # Denominator note: n_skel counts turns that HAVE a gold skeleton
            # (expected 8858) -- the population the certified H5's 0.4317 is
            # scored on. Neither the 3,985 fully-covered turns nor all 8,889.
            if row.gold_skeleton_id is not None:
                n_skel += 1
                if skel_hit:
                    hit_skel += 1
                if const_hit:
                    hit_skel_const += 1
            if skel_hit:
                hit_uncond += 1
            if const_hit:
                hit_uncond_const += 1
            if fully_covered:
                n_cond += 1
                if skel_hit:
                    hit_cond += 1
                if const_hit:
                    hit_cond_const += 1

    if args.save_predictions:
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        with open(args.save_predictions, "w", encoding="utf-8") as fh:
            json.dump(predictions, fh)
        print(f"  wrote predictions: {args.save_predictions}", file=sys.stderr)

    result = {
        "method": "GRU encoder trained END-TO-END (not frozen) on stripped/lowercased/"
                 "stemmed/tokenized INPUT text -> skeleton (H5) classification head",
        "n_train_sample": len(train_sample),
        "n_labels_learnable": len(labels),
        "vocab_size": len(vocab),
        "epochs": args.epochs,
        "skeleton_accuracy": {
            "conditional": {"n": n_cond, "rnn": (hit_cond / n_cond) if n_cond else None,
                            "label_blind_constant_same_5k_fit": (hit_cond_const / n_cond) if n_cond else None},
            "unconditional": {"n": n_uncond, "rnn": hit_uncond / n_uncond if n_uncond else None,
                              "label_blind_constant_same_5k_fit": hit_uncond_const / n_uncond if n_uncond else None},
            "skeleton_population": {
                "n": n_skel,
                "rnn": (hit_skel / n_skel) if n_skel else None,
                "label_blind_constant_same_5k_fit": (hit_skel_const / n_skel) if n_skel else None,
                "note": "turns with a gold skeleton -- the SAME population the certified "
                        "H5's 0.4317 is measured over (select.json h5.n_eval=8858). This is "
                        "the only skeleton accuracy directly comparable to it.",
            },
        },
        "seed": args.seed,
        "compare_against": {
            "H5_TFIDF_logreg_full_train_D25": H5_SKELETON_BAR,
            "ngram_skeleton_full_train": {
                "conditional_n3985": 0.41957340025094103,
                "stale": "measured before the n-gram argmax tie-break fix; re-run "
                         "sft.eval.ngram_skeleton_baseline to refresh",
            },
            "H5_label_blind_constant_full_train_D25": {
                "gold_skeleton_n8858": 0.2630390607360578,
                "note": "select.json headline_test_seen.h5.constant['recall@1'], measured "
                        "over the 8,858 gold-skeleton turns -- NOT over the 3,985 "
                        "fully-covered turns, and so not comparable to "
                        "conditional.label_blind_constant_same_5k_fit.",
            },
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
