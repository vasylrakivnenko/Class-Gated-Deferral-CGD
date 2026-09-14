"""
finetune_encoder candidate for the legalbench_map CV harness.

Approach
--------
A fresh `answerdotai/ModernBERT-base` encoder + randomly-initialized linear
classification head (AutoModelForSequenceClassification) is fine-tuned
end-to-end, from the public pretrained checkpoint, INSIDE every single
`fit_predict_proba` call: 3 epochs, lr 3e-5, AdamW, batch size BATCH_SIZE.
This is intentional and matches the harness spec for this candidate: the
classification head's shape/weights depend on that fold's exact
`classes`/`y_train`, so there is no encoder-only artifact that can be
computed once and reused across folds/repeats the way a frozen-embedding
candidate would. Because of that, this candidate does NOT keep a
(text, label) -> artifact cache under `cache_dir` the way the general
harness docstring asks other candidates to -- there is nothing fold-
independent to cache. The only thing `cache_dir` is used for here is
redirecting the HuggingFace model/tokenizer download cache to
`<cache_dir>/hf_models`, so the ~600MB ModernBERT-base checkpoint is
downloaded once and reused (not re-fetched) across all 15 calls for a task
and across tasks/runs, and to point HF's tokenizer-parallelism env var to a
safe default.

Device selection
----------------
CPU is the default. If `torch.cuda.is_available()` is True, CUDA is used;
else if `torch.backends.mps.is_available()` is True (Apple Silicon), MPS is
used; else CPU. Detection happens fresh on every call.

Determinism
-----------
At the top of every `fit_predict_proba` call: `random.seed(seed)`,
`np.random.seed(seed)`, and `torch.manual_seed(seed)` (plus
`torch.cuda.manual_seed_all(seed)` when CUDA is the selected device) are
set before the pretrained checkpoint is loaded (so the randomly-initialized
classifier head's init is also seed-determined) and before the training
DataLoader's shuffle generator is built (also seeded from `seed`,
`num_workers=0` so there is no multiprocessing nondeterminism). On CPU this
makes the whole call reproducible run-to-run: same seed + same inputs ->
byte-identical probabilities. On MPS or CUDA the same seeding calls are
still made (best-effort determinism) but PyTorch's own GPU/MPS kernels
(e.g. scaled-dot-product-attention reductions) are not guaranteed
bit-identical across runs/hardware the way CPU float ops are, so exact
byte-identical output on MPS/CUDA is NOT guaranteed -- only intended and
seeded as far as PyTorch allows. `torch.use_deterministic_algorithms(True)`
is deliberately NOT set, because ModernBERT's attention path can fall back
to sdpa kernels that do not all have deterministic implementations
registered, and forcing it would crash rather than degrade.

Truncation stats
-----------------
Texts are truncated at `MAX_LENGTH` tokens (module constant, currently
512 -- a practical cap chosen for CPU/MPS fine-tuning speed; ModernBERT's
architecture itself supports up to 8192, but training at that length on
this hardware, up to 15x per task, is not tractable). On every call, before
truncating, `X_test` is tokenized once without truncation to count how many
examples exceed `MAX_LENGTH`. This is accumulated (not overwritten) across
every call seen so far for a given `task_key` into a running
`[truncated_count, total_count]` in the module-level dict
`_TRUNCATION_COUNTS`, and the resulting ratio is written to the
module-level dict:

    TRUNCATION_STATS[task_key] = cumulative_truncated / cumulative_total

so that after the harness finishes all folds/repeats for a task, the
integration code can read back the true X_test truncation fraction for
that task via:

    from candidates.finetune_encoder import TRUNCATION_STATS
    TRUNCATION_STATS["<task_key>"]  # float in [0, 1]

Interface
---------
See the harness's fixed candidate interface docstring for the full
contract. In short: `fit_predict_proba(X_train, y_train, X_test, classes,
*, seed, task_key, cache_dir) -> np.ndarray` of shape
`(len(X_test), len(classes))`, rows summing to 1.0, columns in the exact
order of `classes` (including classes absent from `y_train`, which receive
near-zero probability via ordinary softmax cross-entropy training -- no
special-casing needed: a class that never appears as a training target
still gets its logit suppressed indirectly by the softmax normalization
term on every other example's loss).
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME = "finetune_encoder"

MODEL_NAME = "answerdotai/ModernBERT-base"
# Overridable via env so a long-document experiment can raise the cap without
# touching the fixed candidate interface. Defaults are the values every prior
# run used, so behaviour is byte-identical unless the env vars are set.
# FINETUNE_MAX_LENGTH: ModernBERT supports up to 8192; 512 is the practical
# cap for CPU/MPS speed on short clauses (see module docstring).
import os as _os
MAX_LENGTH = int(_os.environ.get("FINETUNE_MAX_LENGTH", "512"))
# FINETUNE_LEN_BUCKET > 0 rounds each batch's padded length UP to a multiple
# of this value (capped at MAX_LENGTH). Off by default (0) = pad to the
# batch's longest sequence, exactly as before. Why it exists: on MPS every
# distinct padded shape triggers a Metal graph compilation; with dynamic
# padding at 2048 tokens that was ~120 compiles per epoch and a 40-minute
# first fold. Bucketing to 256 caps it at 8 shapes for <=12% padding waste.
LEN_BUCKET = int(_os.environ.get("FINETUNE_LEN_BUCKET", "0"))
EPOCHS = 3  # spec default for real runs; the __main__ self-test overrides
LR = 3e-5
TRAIN_BATCH_SIZE = int(_os.environ.get("FINETUNE_TRAIN_BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(_os.environ.get("FINETUNE_EVAL_BATCH_SIZE", "16"))

# TRUNCATION_STATS[task_key] -> cumulative fraction of X_test examples
# across every fit_predict_proba call seen so far for that task_key whose
# token count exceeded MAX_LENGTH before truncation. See module docstring.
TRUNCATION_STATS = {}
# internal running [truncated_count, total_count] per task_key backing the
# ratios above; not part of the public contract.
_TRUNCATION_COUNTS = {}


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _update_truncation_stats(task_key, texts, tokenizer):
    if not texts:
        return
    lengths = tokenizer(list(texts), truncation=False)["input_ids"]
    truncated = sum(1 for ids in lengths if len(ids) > MAX_LENGTH)
    counts = _TRUNCATION_COUNTS.setdefault(task_key, [0, 0])
    counts[0] += truncated
    counts[1] += len(texts)
    TRUNCATION_STATS[task_key] = counts[0] / counts[1] if counts[1] else 0.0


class _TextDataset(Dataset):
    """Holds raw (text, label_idx) pairs; tokenization happens per-batch in
    the collate function so padding is dynamic (to the batch's longest
    sequence, capped at MAX_LENGTH) rather than always padding to
    MAX_LENGTH -- cheaper for the common case of short legal clauses."""

    def __init__(self, texts, label_idx=None):
        self.texts = texts
        self.label_idx = label_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        label = None if self.label_idx is None else self.label_idx[i]
        return self.texts[i], label


def _make_collate(tokenizer):
    def collate(batch):
        texts = [b[0] for b in batch]
        if LEN_BUCKET > 0:
            enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)
            longest = max(len(ids) for ids in enc["input_ids"])
            target = min(MAX_LENGTH, ((longest + LEN_BUCKET - 1) // LEN_BUCKET) * LEN_BUCKET)
            enc = tokenizer.pad(enc, padding="max_length", max_length=target, return_tensors="pt")
        else:
            enc = tokenizer(
                texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
        if batch[0][1] is not None:
            enc["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return enc

    return collate


def fit_predict_proba(X_train, y_train, X_test, classes, *, seed, task_key, cache_dir):
    # --- determinism (see module docstring for CPU vs MPS/CUDA caveat) ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _select_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    hf_cache_dir = os.path.join(cache_dir, "hf_models")
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=hf_cache_dir)

    _update_truncation_stats(task_key, X_test, tokenizer)

    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    # Fresh checkpoint + fresh randomly-initialized classifier head every
    # call, per the harness spec for this candidate (fold-specific model).
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
        cache_dir=hf_cache_dir,
    )
    model.to(device)

    y_idx = [label2id[y] for y in y_train]
    train_ds = _TextDataset(list(X_train), y_idx)
    collate = _make_collate(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(TRAIN_BATCH_SIZE, max(1, len(train_ds))),
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for _epoch in range(EPOCHS):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()

    model.eval()
    test_ds = _TextDataset(list(X_test), None)
    test_loader = DataLoader(
        test_ds,
        batch_size=min(EVAL_BATCH_SIZE, max(1, len(test_ds))),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(classes)))
    probs = probs.astype(np.float64)

    # Free the fold-specific model/optimizer before returning -- this
    # candidate is called up to 15x per task and each call allocates a
    # fresh ~150M-param model, so releasing device memory between calls
    # matters (especially on MPS, which shares unified memory with the OS).
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return probs


if __name__ == "__main__":
    import tempfile

    # Self-test only: 2 epochs to keep this fast. The real run (driven by
    # the harness) uses EPOCHS = 3 as set above, per spec.
    EPOCHS = 2

    classes = ["alpha", "beta", "gamma"]
    topics = {
        "alpha": ["cat", "dog", "bird", "fish", "rabbit", "hamster", "turtle", "parrot"],
        "beta": ["car", "truck", "bicycle", "train", "airplane", "boat", "scooter", "bus"],
        "gamma": ["apple", "banana", "orange", "grape", "mango", "peach", "pear", "plum"],
    }

    rng = random.Random(0)
    X_train, y_train = [], []
    for cls in classes:
        for i in range(13):
            word = topics[cls][i % len(topics[cls])]
            X_train.append(f"this text is example {i} about a {word}, which is class {cls}")
            y_train.append(cls)
    # trim to exactly ~40
    X_train, y_train = X_train[:40], y_train[:40]

    X_test = []
    expected = []
    for i in range(10):
        cls = classes[i % len(classes)]
        word = topics[cls][(i + 3) % len(topics[cls])]
        X_test.append(f"held out example {i} mentioning a {word}")
        expected.append(cls)

    with tempfile.TemporaryDirectory() as tmp_cache:
        result = fit_predict_proba(
            X_train,
            y_train,
            X_test,
            classes,
            seed=0,
            task_key="selftest_finetune_encoder",
            cache_dir=tmp_cache,
        )

    assert result.shape == (10, 3), f"bad shape: {result.shape}"
    row_sums = result.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"rows do not sum to 1: {row_sums}"
    assert np.all(result >= 0.0), "negative probabilities"

    print("shape:", result.shape)
    print("row sums:", row_sums)
    print("TRUNCATION_STATS:", TRUNCATION_STATS)
    print("self-test OK")
