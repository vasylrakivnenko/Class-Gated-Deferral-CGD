"""3-class sentiment fine-tune of ModernBERT-base on Apple Silicon (MPS).
Verified on: M4 Pro / macOS 26.6 / Python 3.14.3 / torch 2.14.0 / transformers 5.17.0
Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 python finetune_modernbert.py
"""
import os
# MUST be set before torch import: some ops still lack Metal kernels.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

MODEL, NUM_LABELS, SEQ, NTRAIN = "answerdotai/ModernBERT-base", 3, 128, 2000

ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
train = ds["train"].shuffle(seed=0).select(range(NTRAIN))
eval_ = ds["validation"].select(range(1000))

tok = AutoTokenizer.from_pretrained(MODEL)

# PITFALL 1 (MPS graph cache): Metal compiles a kernel per unique tensor shape and
# never evicts. Variable-length batches blow up unified memory. Pad to a FIXED length
# so every batch is the same shape. (Cost: some wasted compute on short texts.)
def tok_fn(b):
    return tok(b["text"], truncation=True, max_length=SEQ, padding="max_length")

train = train.map(tok_fn, batched=True, remove_columns=["text"])
eval_ = eval_.map(tok_fn, batched=True, remove_columns=["text"])

# PITFALL 2 (attention): flash-attn has no Apple Silicon build. In transformers v5
# ModernBERT no longer defaults to FA2, but pass sdpa explicitly to be safe.
# PITFALL 3 (dtype): load in fp32. fp16 on ModernBERT classification heads is a
# reported source of NaN / stuck-at-chance training (HF issue #38720).
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=NUM_LABELS, attn_implementation="sdpa", dtype=torch.float32,
)
# PITFALL 4: `reference_compile` was REMOVED in transformers v5. On v5 passing it
# raises TypeError. Only pass reference_compile=False on transformers 4.48-4.5x.

def metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro")}

args = TrainingArguments(
    output_dir="./out",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=5e-5,
    warmup_steps=0.1,          # v5: float in [0,1) = ratio. `warmup_ratio` was REMOVED.
    weight_decay=0.01,
    eval_strategy="epoch",     # v5: `evaluation_strategy` was REMOVED.
    save_strategy="no",
    logging_steps=50,
    bf16=False,                # keep fp32 on MPS; bf16 needs macOS 14+ and buys little here
    dataloader_num_workers=0,  # >0 is slower on MPS
    report_to="none",
    seed=42,
)
# Trainer auto-detects MPS via torch.backends.mps.is_available(); `use_mps_device`
# was REMOVED in v5. Do not pass it.

trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=eval_,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
print("device:", trainer.args.device)
trainer.train()
print(trainer.evaluate())
