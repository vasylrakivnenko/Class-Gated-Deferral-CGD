"""Non-generative baselines: the rows that stop the chart from lying.

A cost-vs-accuracy chart containing only LLMs answers the question "which LLM?"
when the buyer's real question is "do I need an LLM at all?". For a fixed label
set and a few thousand examples -- ticket routing, intent, sentiment -- the
answer is frequently no, and the honest chart has to be able to say so.

Four rows, cheapest first:

  majority   Always predict the most common class. The floor. On this dataset it
             scores 61.2%, so any model below that line loses to a constant.
  tfidf      Bag of n-grams + logistic regression. Trains in under a second on
             CPU. Reported at ~86.7% (5-fold CV) on this dataset, which is
             higher than several prompted small LLMs -- that is the finding, not
             a bug, and burying it would be the dishonest move.
  frozen     Sentence embeddings (never updated) + logistic regression. Seconds,
             no GPU, and in independent measurement it beat a fine-tuned encoder
             at n=2000. Omit it and the chart overstates the case for fine-tuning.
  finetuned  A small encoder actually trained on the labels. The upsell -- but it
             has to earn its place against the three rows above it.

All four are priced from measured throughput on a cheap CPU instance rather than
per-token, because that is how they would really be served.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from .evaluate import EvalResult
from .stats import accuracy_ci

# HF Inference Endpoints, intel-spr x2 (2 vCPU / 4 GB): $0.067/hr.
# https://huggingface.co/docs/inference-endpoints/support/pricing
CPU_INSTANCE_USD_PER_HOUR = 0.067

# These rows all run locally on the user's own machine, so their marginal cost
# per classification is genuinely zero -- there is no per-call bill to pay.
# We still MEASURE throughput and report the hypothetical hosted rate in each
# row's note, because "what would this cost me if I served it?" is a real
# question. But the chart's cost axis is what you actually pay, and charging
# a laptop $0.00006/1k invents a bill that does not exist while stretching a
# log axis across four extra empty decades to plot it.
LOCAL_IS_FREE = True


def _cost_per_1k(items_per_second: float) -> float:
    """Dollars per 1,000 classifications at a measured CPU throughput.

    Returns exactly 0.0 for infinite throughput (a constant lookup, no model
    invoked) rather than the vanishingly small but nonzero float that
    multiplying by CPU_INSTANCE_USD_PER_HOUR would otherwise produce.
    """
    if items_per_second <= 0 or items_per_second == float("inf"):
        return 0.0
    seconds_per_1k = 1000.0 / items_per_second
    return seconds_per_1k / 3600.0 * CPU_INSTANCE_USD_PER_HOUR


def _result(key: str, label: str, preds: list[str], gold: list[str],
            train_seconds: float, items_per_second: float, note: str) -> EvalResult:
    correct = np.array([p == g for p, g in zip(preds, gold)], dtype=bool)
    hosted = _cost_per_1k(items_per_second)
    if hosted > 0:
        note = (f"{note} Runs locally at ~{items_per_second:,.0f} items/s, so $0 marginal; "
                f"served on a small CPU instance it would be about ${hosted:.5f}/1k.")
    return EvalResult(
        model_key=key, label=label, split="test", n=len(gold),
        correct=correct, predictions=preds, gold=gold,
        accuracy=accuracy_ci(correct),
        mean_in_tokens=0.0, mean_out_tokens=0.0,
        total_in_tokens=0, total_out_tokens=0,
        # This is FIT time, not time to classify the test split. The LLM rows
        # put concurrent-batch duration in the same field and the page renders
        # both under one "Wall time" label, which invited a reader to compare
        # 1,347s of GPU training against 28s of 251 parallel API calls. The
        # value is kept (it is the honest cost of training) and disambiguated
        # by `timing_basis`.
        wall_seconds=train_seconds, timing_basis="fit", parse_failures=0, errors=0,
        cost_per_1k_calls=0.0 if LOCAL_IS_FREE else hosted,
        instruction=note,
    )


def _texts_labels(examples) -> tuple[list[str], list[str]]:
    return [e.sentence for e in examples], [e.label for e in examples]


def run_majority(train, test) -> EvalResult:
    """Always answer the most common training label.

    Cost is set to exactly 0.0, not derived from `_cost_per_1k` with a large
    fake throughput. Feeding that helper a "very fast" proxy (1e9 items/sec)
    still multiplies by `CPU_INSTANCE_USD_PER_HOUR`, so it returns something
    like 1.86e-11 -- nonzero. That number then became the smallest "positive"
    cost on the chart, which set the log-axis floor at ~1.5e-12 and rendered
    every tick as nonsense ($0.0000000001 and friends). A constant lookup has
    no realistic serving cost at all; say so directly instead of computing a
    number that looks precise and is actually a units artifact.
    """
    _, y_train = _texts_labels(train)
    _, y_test = _texts_labels(test)
    # `max(set(y_train), ...)` iterated a set of strings, so on a TIE the winner
    # followed set iteration order, which depends on PYTHONHASHSEED -- verified:
    # a three-way tie returned a different label under each of three seeds. The
    # honesty floor has to be reproducible, so ties break lexicographically.
    # Counter also drops the O(classes x rows) rescan `list.count` did per label.
    counts = Counter(y_train)
    winner = max(sorted(counts), key=counts.__getitem__)
    return _result("majority", "Always predict majority class",
                   [winner] * len(y_test), y_test, 0.0, float("inf"),
                   note=f"Constant prediction: '{winner}'. Free, instant, and the bar every model must clear.")


def run_tfidf(train, test, seed: int = 0) -> EvalResult:
    """Character+word n-gram TF-IDF into logistic regression.

    Character n-grams matter on financial text: the signal lives in tokens like
    "EUR", "mn", "%", "rose", "down", which word-level vectorisers fragment
    across the many number formats in this corpus.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union

    X_train, y_train = _texts_labels(train)
    X_test, y_test = _texts_labels(test)

    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2),
    )
    # class_weight balanced: 61% of this dataset is neutral, and an unweighted
    # fit happily collapses toward predicting it for everything.
    clf = make_pipeline(features, LogisticRegression(max_iter=2000, C=4.0,
                                                     class_weight="balanced",
                                                     random_state=seed))
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_seconds = time.time() - t0

    t1 = time.time()
    preds = list(clf.predict(X_test))
    infer = time.time() - t1

    return _result("tfidf-logreg", "TF-IDF + logistic regression", preds, y_test,
                   train_seconds, len(X_test) / max(infer, 1e-6),
                   note=f"Trained on {len(X_train)} examples in {train_seconds:.1f}s. No GPU, no API.")


def run_frozen_embeddings(train, test, model_name: str = "intfloat/multilingual-e5-small",
                          seed: int = 0) -> EvalResult:
    """Frozen sentence embeddings + logistic regression. No weights updated."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    X_train, y_train = _texts_labels(train)
    X_test, y_test = _texts_labels(test)

    encoder = SentenceTransformer(model_name)
    t0 = time.time()
    # e5 models are trained with these prefixes and lose accuracy without them.
    emb_train = encoder.encode([f"query: {t}" for t in X_train],
                               batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced", random_state=seed)
    clf.fit(emb_train, y_train)
    train_seconds = time.time() - t0

    t1 = time.time()
    emb_test = encoder.encode([f"query: {t}" for t in X_test],
                              batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    preds = list(clf.predict(emb_test))
    infer = time.time() - t1

    return _result("frozen-embed", "Frozen embeddings + logreg", preds, y_test,
                   train_seconds, len(X_test) / max(infer, 1e-6),
                   note=f"{model_name}, weights frozen. Trained head on {len(X_train)} examples in {train_seconds:.1f}s.")


def run_finetuned_encoder(train, test, model_name: str = "jhu-clsp/ettin-encoder-68m",
                          epochs: int = 3, batch_size: int = 16, lr: float = 5e-5,
                          max_length: int = 128, seed: int = 0,
                          labels: tuple[str, ...] | None = None) -> EvalResult:
    """Fine-tune a small encoder end to end.

    A plain torch loop rather than `Trainer`: transformers v5 renamed or removed
    most of the arguments every tutorial passes (`reference_compile` is gone and
    raises, `evaluation_strategy` is now `eval_strategy`, `use_mps_device` is
    gone), so the hand-rolled loop is both faster here and far less likely to
    break on the next minor release.

    `labels` is the task's declared label set. It used to be derived as
    `sorted(set(y_train) | set(y_test))`, i.e. the classification head's output
    space was sized and ordered by reading the HELD-OUT SPLIT -- in the one
    module whose premise is that the test set is never touched, and for the one
    row the product sells as the upsell. Pass `task.labels`; falling back to the
    train split alone is the honest default, since a class with no training
    examples is one this model genuinely cannot predict.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    X_train, y_train = _texts_labels(train)
    X_test, y_test = _texts_labels(test)
    labels = sorted(labels) if labels is not None else sorted(set(y_train))
    to_id = {l: i for i, l in enumerate(labels)}

    torch.manual_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels)).to(device)

    def encode(texts):
        enc = tok(texts, truncation=True, padding="max_length",
                  max_length=max_length, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    ids, mask = encode(X_train)
    y = torch.tensor([to_id[l] for l in y_train])
    loader = DataLoader(TensorDataset(ids, mask, y), batch_size=batch_size, shuffle=True)

    # Class weights for the 13/61/25 imbalance, same reasoning as the TF-IDF row.
    counts = torch.tensor([max(y_train.count(l), 1) for l in labels], dtype=torch.float)
    weights = (counts.sum() / (len(labels) * counts)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(len(loader) * epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps,
                                                pct_start=0.1, anneal_strategy="linear")

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for b_ids, b_mask, b_y in loader:
            opt.zero_grad()
            out = model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device))
            loss = loss_fn(out.logits, b_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    train_seconds = time.time() - t0

    model.eval()
    t_ids, t_mask = encode(X_test)
    preds: list[str] = []
    t1 = time.time()
    with torch.no_grad():
        for i in range(0, len(X_test), 64):
            out = model(input_ids=t_ids[i:i + 64].to(device),
                        attention_mask=t_mask[i:i + 64].to(device))
            preds.extend(labels[j] for j in out.logits.argmax(-1).tolist())
    infer = time.time() - t1

    return _result("finetuned-encoder", f"Fine-tuned encoder ({model_name.split('/')[-1]})",
                   preds, y_test, train_seconds, len(X_test) / max(infer, 1e-6),
                   note=f"{model_name} fine-tuned on {len(X_train)} examples, "
                        f"{epochs} epochs, {train_seconds:.0f}s on {device}.")
