"""Per-act RNN (GRU) template classifiers, trained end to end on stemmed/
lowercased/tokenized context -- the RNN analogue of the certified per-act
TF-IDF+logreg H7 classifiers, so the RNN skeleton model (rnn_skeleton_dryrun.py)
can be paired with a template step trained the SAME way, instead of the
TF-IDF/RNN hybrid in rnn_skeleton_plus_h7.py.

HONEST EXPECTATION, STATED BEFORE RUNNING: only 9 acts exist, but the two
largest (ASK: 1,066 classes / 19,020 rows; ACK: 1,020 classes / 15,983 rows)
average roughly 18-20 examples PER CLASS -- a much harder regime for a
from-scratch sequence model than the 570-way SKELETON task (which had far
more examples per class). This may well underperform the already-tuned
TF-IDF+logreg H7 classifiers; the point is to find out, not to assume.

USAGE
-----
    PYTHONPATH=src python -m sft.eval.rnn_h7_template \
        --skeleton-predictions outputs/probes/response/rnn_skeleton_full_predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn_skeleton_dryrun import preprocess  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skeleton-predictions", required=True)
    ap.add_argument("--vocab-size", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument("--emb-dim", type=int, default=96)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-class-rows", type=int, default=2)
    ap.add_argument("--out", default="outputs/probes/response/rnn_h7_template.json")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    sys.path.insert(0, "/Users/vasyl/zadumai/reflex-abcd")
    from reflex.config import load_config
    from reflex.compile import load_bank
    from probes.run_response_probe import load_split_rows

    cfg = load_config()
    bank = load_bank(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    print("loading rows...", file=sys.stderr)
    train_rows = load_split_rows(cfg, "train", bank, h4_indexer=None, cache_dir="")
    test_rows = load_split_rows(cfg, "test_seen", bank, h4_indexer=None, cache_dir="")
    spaces = train_rows["spaces"]
    skeleton_acts = spaces.skeleton_acts

    by_act_fit: dict = defaultdict(list)
    for r in train_rows["h7"]:
        if r.gold_template_id and r.act:
            by_act_fit[r.act].append(r)

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

    class ActDataset(Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return torch.tensor(self.x[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.long)

    fitted_by_act: dict = {}
    for act, rows in sorted(by_act_fit.items(), key=lambda kv: -len(kv[1])):
        golds = [r.gold_template_id for r in rows]
        if len(set(golds)) < 2:
            fitted_by_act[act] = None
            continue
        t1 = time.time()
        token_lists = [preprocess(r.context.text) for r in rows]
        tok_counts = Counter(t for toks in token_lists for t in toks)
        vocab = ["<pad>", "<unk>"] + [w for w, _ in tok_counts.most_common(args.vocab_size)]
        word_to_idx = {w: i for i, w in enumerate(vocab)}

        def encode(tokens):
            ids = [word_to_idx.get(t, 1) for t in tokens[: args.max_len]]
            if len(ids) < args.max_len:
                ids = ids + [0] * (args.max_len - len(ids))
            return ids

        labels = sorted(set(golds))
        label_to_idx = {tid: i for i, tid in enumerate(labels)}
        x = [encode(t) for t in token_lists]
        y = [label_to_idx[g] for g in golds]

        model = GRUClassifier(len(vocab), args.emb_dim, args.hidden_dim, len(labels)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        dl = DataLoader(ActDataset(x, y), batch_size=args.batch_size, shuffle=True)

        print(f"training act={act}  n={len(rows)}  classes={len(labels)}  "
              f"vocab={len(vocab)} ...", file=sys.stderr)
        for epoch in range(args.epochs):
            model.train()
            total_loss, n_correct, n_seen = 0.0, 0, 0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * xb.size(0)
                n_correct += (logits.argmax(1) == yb).sum().item()
                n_seen += xb.size(0)
            print(f"  [{act}] epoch {epoch+1}/{args.epochs}  loss={total_loss/n_seen:.4f}  "
                  f"train_acc={n_correct/n_seen:.4f}", file=sys.stderr)
        print(f"  [{act}] trained in {time.time()-t1:.1f}s", file=sys.stderr)
        fitted_by_act[act] = (model, word_to_idx, labels)

    def predict_template(act: str, context_text: str) -> str:
        entry = fitted_by_act.get(act)
        if not entry:
            return ""
        model, word_to_idx, labels = entry
        toks = preprocess(context_text)
        ids = [word_to_idx.get(t, 1) for t in toks[: args.max_len]]
        if len(ids) < args.max_len:
            ids = ids + [0] * (args.max_len - len(ids))
        model.eval()
        with torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long).to(device)
            logits = model(x)
            idx = int(logits.argmax(1).item())
        return labels[idx]

    # ---- score, using the RNN's own predicted skeletons ----
    rnn_sk_preds = json.load(open(args.skeleton_predictions, encoding="utf-8"))
    test_h7_by_turn: dict = defaultdict(list)
    for r in test_rows["h7"]:
        test_h7_by_turn[r.turn_id].append(r)

    n_cond = hit_cond = hit_skel_cond = 0
    n_total = hit_uncond = hit_skel_uncond = 0
    for row in test_rows["h5"]:
        positions = sorted(test_h7_by_turn.get(row.turn_id, []), key=lambda p: p.position)
        gold_tids = tuple(p.gold_template_id for p in positions)
        fully_covered = bool(gold_tids) and all(gold_tids)

        pred_sk = rnn_sk_preds.get(row.turn_id)
        pred_acts = skeleton_acts.get(pred_sk, ()) if pred_sk else ()
        skel_hit = pred_sk is not None and pred_sk == row.gold_skeleton_id

        pred_tids = tuple(predict_template(act, row.context.text) for act in pred_acts)
        full_hit = (fully_covered and pred_tids == gold_tids
                   and bool(gold_tids) and all(pred_tids))

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

    result = {
        "method": "RNN (GRU) skeleton prediction + RNN (GRU) per-act template classifiers "
                 "(fully RNN-based pipeline, not the TF-IDF/RNN hybrid)",
        "conditional": {"n": n_cond, "compose@1": (hit_cond / n_cond) if n_cond else None,
                       "skeleton_only": (hit_skel_cond / n_cond) if n_cond else None},
        "unconditional": {"n": n_total, "compose@1": hit_uncond / n_total if n_total else None,
                          "skeleton_only": hit_skel_uncond / n_total if n_total else None},
        "compare_against": {
            "learned_cache_TFIDF_logreg_D25": {"conditional": 0.2765370138017566, "unconditional": 0.12397345033187085},
            "rnn_skeleton_plus_certified_h7": {"conditional": 0.2506900878293601, "unconditional": 0.11238609517381033},
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
