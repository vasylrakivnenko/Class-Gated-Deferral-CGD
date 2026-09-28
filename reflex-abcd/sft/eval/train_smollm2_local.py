"""Full fine-tune (all weights, not LoRA) SmolLM2-360M on the UNCONSTRAINED
arm's exact SFT data (sft/data/abcd_train_lexical.jsonl), run entirely local
(no Fireworks account, no hourly GPU billing) on Apple Silicon MPS.

WHY THIS ARM EXISTS
--------------------
Every prior neural arm this session either froze the encoder (H1-H7 heads) or
was a LoRA adapter on qwen3-0.6B served through Fireworks (hourly-billed,
dedicated GPU only). SmolLM2-360M is small enough (361M params) that a FULL
fine-tune -- every weight updated, no LoRA -- is cheap on a single machine.
The open question this answers: does full-tuning a smaller model close, match,
or fall short of LoRA-tuning a larger one (qwen3-0.6B) on the same task.

SAME PROTOCOL AS THE QWEN UNCONSTRAINED ARM, so the two are comparable:
  - identical training data: sft/data/abcd_train_lexical.jsonl (messages
    format, ContextWindow.text prompt, lexicalized utterance target -- see
    sft/build_sft_dataset.py's own docstring for why each choice was made)
  - identical system prompt (baked into that data's `messages[0]`)
  - identical eval prompts: sft/eval/data/eval_prompts_test_seen.jsonl
  - identical scorer: sft/eval/score_generations.py (compose@1 against the
    bank, both conditional and unconditional populations)

WHAT "full fine-tune" MEANS HERE
---------------------------------
No PEFT/LoRA import. `model.parameters()` are ALL trainable. A held-out 3% of
train (deterministic split, seed-derived) is used for early-stopping /
best-checkpoint selection on token-level loss -- this is loop control only,
same as spec 6.4's dev_selection_score, not a reported metric.

USAGE
-----
    /Users/vasyl/zadumai/.venv/bin/python sft/eval/train_smollm2_local.py \
        --out-dir sft/eval/checkpoints/smollm2_360m_full \
        --epochs 3 --batch-size 8 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time


def _load_messages(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


_UNVERIFIED_DEV_WARNING = (
    "WARNING: DEV HOLDOUT NOT VERIFIED CLEAN. {why} The fixed holdout only "
    "protects a resume chain in which EVERY run used it. A checkpoint trained "
    "before the holdout existed (or under another --seed/--dev-frac/--train) "
    "took its training rows from the head of the same shuffle this run's dev "
    "rows come from, so this run's dev_loss -- and the best-checkpoint choice "
    "made on it -- may be measured on rows the checkpoint already trained on. "
    "Only retraining from --base-model gives a clean dev signal.")


def _prior_best_dev(resume_from: str, dev_split: dict) -> tuple[float, bool]:
    """(best_dev_loss to start a --resume-from run at, dev holdout verified clean).

    The second value is True only when the resumed checkpoint recorded the SAME
    dev_split AND was itself verified clean, i.e. no run in the chain can have
    trained on this run's dev rows. Otherwise _UNVERIFIED_DEV_WARNING is printed.

    Carried forward from the checkpoint being resumed ONLY when that run held
    out the SAME dev rows; two losses measured on different dev sets are not
    comparable. Without this, `best_dev` starts at inf on every resume, so the
    first epoch always "improves" and overwrites the checkpoint whether or not
    it actually got better.
    """
    path = os.path.join(resume_from, "train_history.json")
    try:
        with open(path, encoding="utf-8") as fh:
            prior = json.load(fh)
    except (OSError, ValueError) as e:
        print(f"resume: no usable {path} ({e}); best_dev starts at inf",
              file=sys.stderr)
        print(_UNVERIFIED_DEV_WARNING.format(
            why=f"{path} is unreadable, so what {resume_from} trained on is unknown."),
            file=sys.stderr)
        return float("inf"), False
    prior_split = prior.get("dev_split")
    prior_best = prior.get("best_dev_loss")
    if prior_split != dev_split:
        print(f"resume: {path} recorded dev_split={prior_split!r}, this run has "
              f"{dev_split!r} -- losses are not comparable, best_dev starts at inf",
              file=sys.stderr)
        print(_UNVERIFIED_DEV_WARNING.format(
            why=f"{resume_from} did not hold out this run's dev rows."), file=sys.stderr)
        return float("inf"), False
    clean = prior.get("dev_holdout_verified_clean") is True
    if not clean:
        print(_UNVERIFIED_DEV_WARNING.format(
            why=f"{resume_from} held out the same dev rows but was itself resumed "
                f"from a checkpoint that was not verified clean."), file=sys.stderr)
    if not isinstance(prior_best, (int, float)) or prior_best != prior_best:
        print(f"resume: {path} has no usable best_dev_loss; best_dev starts at inf",
              file=sys.stderr)
        return float("inf"), clean
    print(f"resume: carrying best_dev_loss={prior_best:.4f} forward from {resume_from}",
          file=sys.stderr)
    return float(prior_best), clean


def render_prompt(system: str, user: str) -> str:
    """Plain-text frame for SmolLM2-360M base (no chat template). MUST stay
    byte-identical between training (this file) and generation
    (generate_smollm2_local.py) -- the model only ever sees this exact shape."""
    return f"{system}\n\n{user}\nAgent:"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="sft/data/abcd_train_lexical.jsonl")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--resume-from", default=None,
                    help="path to a checkpoint dir saved by this script -- when "
                         "set, model+tokenizer load from here instead of "
                         "--base-model, continuing training rather than "
                         "reinitializing. The dev holdout is carved off the "
                         "full shuffled corpus BEFORE --smoke/--row-offset "
                         "windowing, so it is the same rows at the same --seed "
                         "in every run, and a resume is never evaluated on rows "
                         "an earlier run trained on PROVIDED every run in the "
                         "chain used this holdout. That is NOT true of a "
                         "checkpoint trained before the holdout existed (its "
                         "train_history.json has no dev_split): resuming one "
                         "prints a contamination warning and records "
                         "dev_holdout_verified_clean=false. A larger --smoke is "
                         "still a strict superset of a smaller prior run's "
                         "training rows, so it adds new rows on top of ones "
                         "already seen -- that is training-side reuse, not dev "
                         "contamination.")
    ap.add_argument("--out-dir", default="sft/eval/checkpoints/smollm2_360m_full")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--dev-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, train on only this many rows (pipeline smoke "
                         "test). Counted in the TRAIN pool, i.e. after the "
                         "fixed dev holdout is removed, so --smoke rows are "
                         "all training rows.")
    ap.add_argument("--row-offset", type=int, default=0,
                    help="skip this many TRAIN-pool rows (post-shuffle, "
                         "post-dev-holdout, same fixed seed) "
                         "before taking --smoke rows -- e.g. --smoke 776 "
                         "--row-offset 776 grabs the NEXT 776 rows, disjoint "
                         "from a prior --smoke 776 --row-offset 0 run, for a "
                         "true continuation rather than reseeing the same data.")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"],
                    help="'auto' picks mps if available. MPS on this machine "
                         "(torch 2.14 / macOS 26.6.1) was found to hang "
                         "unpredictably -- traced to a stall inside Metal's own "
                         "command-buffer semaphore wait, not this script -- so "
                         "'cpu' is the reliable fallback, just slower per step.")
    ap.add_argument("--cpu-threads", type=int, default=10)
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    if device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    print(f"device={device}", file=sys.stderr)

    load_from = args.resume_from or args.base_model
    print(f"loading tokenizer+model: {load_from}"
          f"{' (RESUMING, not reinitializing)' if args.resume_from else ''}",
          file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(load_from)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bf16, not fp32: a timing probe found fp32-on-MPS wildly unstable for this
    # model (10s -> 158s on back-to-back identical-shape steps, almost
    # certainly GPU memory pressure/paging), while bf16 held steady at 1-8s/step
    # across the same shapes. AdamW master weights/grads stay bf16 too (no
    # mixed-precision optimizer here) -- fine for a from-scratch SFT run of
    # this size, and the whole point is to stay usable on a laptop.
    # bf16 on MPS was the fast/stable choice (see the note above); CPU has no
    # native bf16 matmul path, so fp32 is what the earlier CPU timing probe
    # actually validated (~2-3s/step at batch 4) -- bf16-on-CPU would just be
    # emulated and slower, not smaller.
    model_dtype = torch.bfloat16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(load_from, dtype=model_dtype)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total={n_params:,} trainable={n_trainable:,} (full fine-tune)",
          file=sys.stderr)

    rows = _load_messages(args.train)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    # Dev is held out of the FULL shuffled corpus BEFORE any --smoke/--row-offset
    # windowing, so it is the same rows in every run at this seed. Splitting
    # after the window (the old order) made dev the head of whatever slice this
    # run took, so a --resume-from run with a larger --smoke evaluated on rows
    # the checkpoint had already trained on.
    n_dev = max(1, int(len(rows) * args.dev_frac))
    dev_rows, pool = rows[:n_dev], rows[n_dev:]
    if args.smoke:
        train_rows = pool[args.row_offset: args.row_offset + args.smoke]
    else:
        train_rows = pool
    dev_split = {"scheme": "head of the full shuffled corpus, before --smoke/--row-offset",
                 "train_path": args.train, "n_corpus": len(rows),
                 "seed": args.seed, "dev_frac": args.dev_frac, "n_dev": len(dev_rows)}
    print(f"n_train={len(train_rows)} n_dev={len(dev_rows)} "
          f"(dev held out of the full {len(rows)}-row corpus before windowing)",
          file=sys.stderr)

    def encode(example: dict) -> dict:
        msgs = example["messages"]
        assistant_text = msgs[-1]["content"]
        prompt_text = render_prompt(msgs[0]["content"], msgs[1]["content"])
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tok(" " + assistant_text, add_special_tokens=False)["input_ids"]
        eos = [tok.eos_token_id]
        input_ids = prompt_ids + target_ids + eos
        labels = [-100] * len(prompt_ids) + target_ids + eos
        input_ids = input_ids[: args.max_len]
        labels = labels[: args.max_len]
        return {"input_ids": input_ids, "labels": labels}

    class SFTDataset(Dataset):
        def __init__(self, raw_rows):
            self.examples = [encode(r) for r in raw_rows]

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            return self.examples[i]

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = tok.pad_token_id
        input_ids, attn, labels = [], [], []
        for b in batch:
            n_pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n_pad)
            attn.append([1] * len(b["input_ids"]) + [0] * n_pad)
            labels.append(b["labels"] + [-100] * n_pad)
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(attn, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    print("tokenizing...", file=sys.stderr)
    train_ds = SFTDataset(train_rows)
    dev_ds = SFTDataset(dev_rows)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_dl) // args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(1, int(0.03 * total_steps)),
        num_training_steps=max(1, total_steps))

    def eval_dev() -> float:
        model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for input_ids, attn, labels in dev_dl:
                input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                total_loss += out.loss.item()
                n_batches += 1
        model.train()
        return total_loss / max(1, n_batches)

    os.makedirs(args.out_dir, exist_ok=True)
    best_dev, dev_clean = (_prior_best_dev(args.resume_from, dev_split)
                           if args.resume_from else (float("inf"), True))
    saved_any = False
    t0 = time.time()
    step = 0
    history = []

    model.train()
    for epoch in range(args.epochs):
        opt.zero_grad()
        for i, (input_ids, attn, labels) in enumerate(train_dl):
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"epoch={epoch} step={step}/{total_steps} "
                          f"loss={out.loss.item():.4f} elapsed={elapsed:.0f}s",
                          file=sys.stderr)
        dev_loss = eval_dev()
        history.append({"epoch": epoch, "dev_loss": dev_loss,
                        "elapsed_s": time.time() - t0})
        print(f"epoch={epoch} done dev_loss={dev_loss:.4f} "
              f"elapsed={time.time() - t0:.0f}s", file=sys.stderr)
        if dev_loss < best_dev:
            best_dev = dev_loss
            saved_any = True
            model.save_pretrained(args.out_dir)
            tok.save_pretrained(args.out_dir)
            print(f"  new best (dev_loss={dev_loss:.4f}), saved to {args.out_dir}",
                  file=sys.stderr)

    if not saved_any:
        print(f"WARNING: no epoch beat best_dev_loss={best_dev:.4f} (carried "
              f"forward from {args.resume_from}), so NO model was written to "
              f"{args.out_dir} -- the resumed checkpoint is still the best one.",
              file=sys.stderr)
    with open(os.path.join(args.out_dir, "train_history.json"), "w", encoding="utf-8") as fh:
        # smoke/row_offset/seed/dev_frac are part of the result: without them the
        # training window of a resume chain cannot be checked after the fact.
        json.dump({"base_model": args.base_model, "loaded_from": load_from,
                   "resumed": bool(args.resume_from), "n_train": len(train_rows),
                   "n_dev": len(dev_rows), "epochs": args.epochs,
                   "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                   "lr": args.lr, "best_dev_loss": best_dev,
                   "saved_checkpoint": saved_any,
                   "seed": args.seed, "dev_frac": args.dev_frac,
                   "smoke": args.smoke, "row_offset": args.row_offset,
                   "dev_split": dev_split,
                   "dev_holdout_verified_clean": dev_clean,
                   "total_time_s": time.time() - t0, "history": history}, fh, indent=2)
    print(f"done. best_dev_loss={best_dev:.4f} total_time={time.time() - t0:.0f}s "
          f"checkpoint={args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
