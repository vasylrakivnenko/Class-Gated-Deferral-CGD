#!/usr/bin/env python
"""
hint3_arm.py -- the "free arm" measurement on HINT3 v1 full (Arora et al.,
EMNLP 2020 Insights Workshop, arXiv:2009.13833): three real production chatbot
intent datasets (sofmattress, curekart, powerplay11) from three Indian companies.

Task shape: OPEN-SET intent detection. `NO_NODES_DETECTED` (OOS) occurs only in
test, never in train. Train on in-scope intents only; decide per test item
whether to answer or to say "no intent".

=============================================================================
DESIGN DECISIONS, AND THE MEASUREMENT THAT FORCED EACH  (house style)
=============================================================================

D1. Threshold comparison is `score >= t`, not `score > t`.
    FORCED BY: reproducing the authors' own 15 result files
    (results/{platform}_{bot}.csv, 3 bots x 5 platforms x 9 thresholds x 5
    metrics = 675 numbers) from their own per-item preds/. With `>=` the max
    absolute deviation over all 675 numbers is 2.22e-16 (float noise); with `>`
    it is 4.5e-2. So `>=` is the authors' operator and my whole metric stack is
    now validated against their reference implementation.

D2. The published platform numbers are treated as PAIRED SAME-ITEMS, not merely
    "published on the authors' protocol".
    FORCED BY: `pd.DataFrame.equals` -> our hint3/{bot}_{train,test}.csv are
    byte-identical to the repo's dataset/v1/{train,test}/, AND for all 5
    platforms x 3 bots the preds/ files match the test CSV row-for-row in
    order on both `sentence` and `label` (397/991/983 rows, 0 NaN preds).
    Because the items and their order are identical, per-item McNemar tests
    against Dialogflow / LUIS / RASA / BERT / Haptik are legitimate. This is
    stronger evidence than the usual published-baseline comparison, and it
    cost nothing.

D3. Macro-F1 is averaged over in-scope intents that have non-zero support in
    test gold, not over all train intents.
    FORCED BY: curekart trains 28 intents but only 21 of them ever appear in
    test (7 have zero test support); powerplay11 and sofmattress each have 1
    such intent. Averaging over zero-support classes injects a hard 0.0 per
    empty class and would report a macro-F1 depressed by a factor of 21/28
    for reasons that have nothing to do with the model. Both variants are
    recorded in the JSON (`macro_f1_inscope` vs `macro_f1_all_train_intents`).

D4. Overall accuracy and OOS recall are reported but FLAGGED as
    constant-dominated; the headline metrics are in-scope accuracy, macro-F1
    and MCC.
    FORCED BY: the all-OOS constant predictor (label-blind, identifies zero
    intents) scores 41.8 / 54.4 / 72.0 % overall accuracy and 1.000 OOS recall
    on the three bots. It therefore *wins* overall accuracy outright on
    powerplay11 against every method in this study, published platforms
    included, and wins OOS recall trivially everywhere. In-scope accuracy,
    macro-F1 and MCC are all exactly 0 for any constant predictor, so they are
    the metrics that carry information. Project rule: drop or flag any metric a
    label-blind constant wins.

D5. The rejection threshold CANNOT be tuned on train in any way that reflects
    the accuracy/OOS-recall trade-off.
    FORCED BY: train contains 0 out-of-scope rows (verified: 0/328, 0/600,
    0/471). With no OOS examples in any CV fold, every rejection is a pure loss
    on train, so a CV objective of "maximise fold accuracy" is monotone in
    lowering t and selects the degenerate t=0 (never reject). This is a
    property of the HINT3 protocol, not a bug in the sweep.
    RESOLUTION: (a) the full threshold sweep is the primary result, exactly as
    the paper does it; (b) for a single shipped operating point we select t by
    an OOS-free rule -- t = the q-th quantile of out-of-fold in-scope
    confidence, i.e. "reject the q least-confident fraction of in-scope
    traffic" (a target in-scope rejection rate, standard selective
    prediction). q=0.10 is fixed a priori as primary; q=0.20/0.30 also
    reported; (c) the test-optimal t is reported separately and labelled
    OPTIMISTIC / TUNED-ON-TEST everywhere it appears.

D6. Intents with 1-2 training examples are KEPT, not dropped.
    FORCED BY: powerplay11 has 14 intents with <3 train examples, 7 of them
    with exactly 1. Those intents own 27 of the 275 in-scope test items (14
    items for the 7 singleton intents). Dropping the intents would make those
    27 items unanswerable by construction -- a silent 9.8% ceiling cut on
    in-scope test items disguised as a modelling choice. They are kept and the
    per-intent breakdown reports how they actually do.

D7. Model/arm selection for the "row we would ship" is by mean out-of-fold CV
    macro-F1 on TRAIN, never by test score.
    FORCED BY: project rule that the cascade gate must be computed on the row
    we would actually ship, plus the prior error of computing it on the
    weakest row (which reversed a verdict). Test scores of every arm are
    reported regardless, so the reader can check the CV choice was right.

D8. Encoder epoch count is CV-selected over {3, 10, 20}; the spec's 3 is also
    reported as its own arm.
    FORCED BY: at batch 16, 3 epochs is 63 / 114 / 90 optimizer steps total on
    the three bots -- a 68M-parameter encoder with a freshly initialised
    classifier head (the HF load report confirms classifier.{weight,bias} are
    newly initialised). Both numbers are reported so the choice is auditable.

D9. Tokenisation truncates at max_length=256 but pads dynamically to the batch
    maximum.
    FORCED BY: the longest sentence in any split is 73 words (means 4.2-6.9
    words, p95 <= 17). Padding to a fixed 256 is arithmetically identical under
    the attention mask and ~10x slower. Same numbers, less compute.

D10. Full probability matrices are persisted per arm to hint3_probs.npz, and
    per-item {pred, gold, conf} to hint3.json.
    FORCED BY: this project's standing known limitation -- an earlier encoder
    runner kept only argmax and discarded the softmax distribution, so its
    strongest row could not be asked how confident it was, and every chart in
    the project is stuck on the TF-IDF row. All four arms here keep the full
    distribution.

D11. `sublinear_tf=True` is applied to BOTH the word and the char_wb block.
    The spec attaches it to the word block; the measured difference between
    the two readings is recorded in the JSON under
    `tfidf_sublinear_char_ablation` so the ambiguity is resolved by number
    rather than by guess.

D12. The cascade gate is computed under TWO explicit correctness definitions
    because on this task shape they disagree, and reporting only one would be
    a hidden choice:
      A (shipped pipeline): correct = (thresholded prediction == gold), i.e.
        the output we would actually ship, OOS rejections included.
      B (answering subset): correct = (argmax == gold) among gold-in-scope
        items only, i.e. "when the bot does answer, does confidence track
        correctness" -- the definition comparable to the project's banking77
        0.905 / ToxicChat 0.902 / CUAD 0.797 reference rows.
    Definition A is mechanically non-monotone in confidence on an OOS-heavy
    set (low confidence -> predicts OOS -> correct exactly when gold is OOS,
    which is 72% of powerplay11), so a low AUROC under A is a statement about
    the task, not about the confidence signal. Both are reported. Calibration
    gap is deliberately NOT computed as a gate (ranking correctness and being
    calibrated are different properties).

D13. No paid API call is made anywhere in this file. Every number is from
    local sklearn/torch or from the authors' own committed prediction files.
    Zero spend.

SEED = 0 everywhere.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import FeatureUnion

SEED = 0
OOS = "NO_NODES_DETECTED"
BOTS = ["sofmattress", "curekart", "powerplay11"]
PLATFORMS = ["dialogflow", "luis", "rasa", "bert", "haptik"]
THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]          # paper grid
FINE_T = [round(x, 3) for x in np.arange(0.0, 1.0001, 0.01)]     # curve grid
IRR_TARGETS = [0.10, 0.20, 0.30]      # in-scope rejection rate targets (D5)
PRIMARY_IRR = 0.10
CV_K = 5
EPOCH_GRID = [3, 10, 20]
SPEC_EPOCHS = 3

HERE = Path(__file__).resolve().parent
DATA = HERE / "hint3"
REPO = HERE / "hint3_repo" / "HINT3-master"
CACHE = HERE / "hint3_encoder_cache"
CACHE.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load(bot):
    tr = pd.read_csv(DATA / f"{bot}_train.csv")
    te = pd.read_csv(DATA / f"{bot}_test.csv")
    assert (tr.label == OOS).sum() == 0, "train must be in-scope only"
    return tr, te


def norm_text(s):
    import re
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def contamination(tr, te):
    ex = set(tr.sentence.astype(str))
    nm = set(tr.sentence.map(norm_text))
    te_s = te.sentence.astype(str)
    te_n = te.sentence.map(norm_text)
    ins = (te.label != OOS).values
    hit_e = te_s.isin(ex).values
    hit_n = te_n.isin(nm).values
    return {
        "test_rows": int(len(te)),
        "exact_overlap_n": int(hit_e.sum()),
        "exact_overlap_pct": round(100 * float(hit_e.mean()), 3),
        "normalized_overlap_n": int(hit_n.sum()),
        "normalized_overlap_pct": round(100 * float(hit_n.mean()), 3),
        "exact_overlap_inscope_n": int(hit_e[ins].sum()),
        "exact_overlap_inscope_of": int(ins.sum()),
        "exact_overlap_oos_n": int(hit_e[~ins].sum()),
        "exact_overlap_oos_of": int((~ins).sum()),
        "train_internal_dup_exact": int(len(tr) - tr.sentence.nunique()),
        "train_internal_dup_normalized": int(len(tr) - tr.sentence.map(norm_text).nunique()),
    }


# --------------------------------------------------------------------------
# metrics  (validated against the authors' implementation -- D1)
# --------------------------------------------------------------------------
def effective_pred(pred_argmax, conf, t):
    """Answer only above confidence t, else say OOS. `>=` per D1."""
    return np.where(np.asarray(conf) >= t, np.asarray(pred_argmax), OOS)


def metrics_at(gold, pred_argmax, conf, t, inscope_labels, all_train_labels):
    gold = np.asarray(gold)
    eff = effective_pred(pred_argmax, conf, t)
    ins = gold != OOS
    n_ans = int((eff != OOS).sum())
    return {
        "threshold": float(t),
        "accuracy": float((eff == gold).mean()),                       # FLAGGED D4
        "inscope_accuracy": float((eff[ins] == gold[ins]).mean()),
        "oos_recall": float((eff[~ins] == OOS).mean()) if (~ins).sum() else None,  # FLAGGED D4
        "macro_f1_inscope": float(f1_score(gold, eff, labels=inscope_labels,
                                           average="macro", zero_division=0)),
        "macro_f1_all_train_intents": float(f1_score(gold, eff, labels=all_train_labels,
                                                     average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, eff, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(gold, eff)),
        "answered_n": n_ans,
        "answered_frac": float(n_ans / len(gold)),
    }


def constants(tr, te, inscope_labels, all_train_labels):
    """The two label-blind constants (D4)."""
    gold = te.label.values
    ins = gold != OOS
    out = {}
    # all-OOS constant
    eff = np.full(len(gold), OOS)
    out["const_all_oos"] = {
        "accuracy": float((eff == gold).mean()),
        "inscope_accuracy": float((eff[ins] == gold[ins]).mean()),
        "oos_recall": 1.0,
        "macro_f1_inscope": float(f1_score(gold, eff, labels=inscope_labels,
                                           average="macro", zero_division=0)),
        "macro_f1_all_train_intents": float(f1_score(gold, eff, labels=all_train_labels,
                                                     average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, eff, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(gold, eff)) if len(set(eff)) > 1 else 0.0,
        "identifies_intents": 0,
    }
    # majority in-scope intent constant
    maj = tr.label.value_counts().idxmax()
    eff = np.full(len(gold), maj)
    out["const_majority_intent"] = {
        "label": str(maj),
        "train_share": float((tr.label == maj).mean()),
        "accuracy": float((eff == gold).mean()),
        "inscope_accuracy": float((eff[ins] == gold[ins]).mean()),
        "oos_recall": 0.0,
        "macro_f1_inscope": float(f1_score(gold, eff, labels=inscope_labels,
                                           average="macro", zero_division=0)),
        "macro_f1_all_train_intents": float(f1_score(gold, eff, labels=all_train_labels,
                                                     average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, eff, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(gold, eff)) if len(set(eff)) > 1 else 0.0,
        "identifies_intents": 1,
    }
    return out


# --------------------------------------------------------------------------
# CV folds: stratified round-robin, tolerates singleton classes (D6)
# --------------------------------------------------------------------------
def strat_folds(y, k=CV_K, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    folds = np.zeros(len(y), dtype=int)
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        folds[idx] = (np.arange(len(idx)) + int(rng.integers(k))) % k
    return folds


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def tfidf_vec(sublinear_char=True):
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 sublinear_tf=sublinear_char)),
    ])


def fit_lr(X, y, max_iter):
    clf = LogisticRegression(max_iter=max_iter, C=4.0,
                             class_weight="balanced", random_state=SEED)
    clf.fit(X, y)
    return clf


def arm_majority(tr, te):
    maj = tr.label.value_counts().idxmax()
    p = float((tr.label == maj).mean())
    classes = np.array(sorted(tr.label.unique()))
    P_te = np.zeros((len(te), len(classes)))
    P_te[:, list(classes).index(maj)] = p
    P_cv = np.zeros((len(tr), len(classes)))
    P_cv[:, list(classes).index(maj)] = p
    return dict(classes=classes,
                test_pred=np.full(len(te), maj), test_conf=np.full(len(te), p),
                cv_pred=np.full(len(tr), maj), cv_conf=np.full(len(tr), p),
                test_probs=P_te, cv_probs=P_cv)


def arm_tfidf(tr, te, sublinear_char=True):
    folds = strat_folds(tr.label.values)
    classes = np.array(sorted(tr.label.unique()))
    cls_ix = {c: i for i, c in enumerate(classes)}
    P_cv = np.zeros((len(tr), len(classes)))
    for f in range(CV_K):
        m = folds != f
        v = tfidf_vec(sublinear_char)
        Xtr = v.fit_transform(tr.sentence[m].astype(str))
        Xva = v.transform(tr.sentence[~m].astype(str))
        clf = fit_lr(Xtr, tr.label[m].values, 2000)
        pr = clf.predict_proba(Xva)
        for j, c in enumerate(clf.classes_):
            P_cv[np.where(~m)[0], cls_ix[c]] = pr[:, j]
    v = tfidf_vec(sublinear_char)
    Xtr = v.fit_transform(tr.sentence.astype(str))
    Xte = v.transform(te.sentence.astype(str))
    clf = fit_lr(Xtr, tr.label.values, 2000)
    pr = clf.predict_proba(Xte)
    P_te = np.zeros((len(te), len(classes)))
    for j, c in enumerate(clf.classes_):
        P_te[:, cls_ix[c]] = pr[:, j]
    return _pack(classes, P_cv, P_te)


_E5 = {}


def e5_embed(texts):
    from sentence_transformers import SentenceTransformer
    if "m" not in _E5:
        _E5["m"] = SentenceTransformer("intfloat/multilingual-e5-small")
    return _E5["m"].encode([f"query: {t}" for t in texts],
                           normalize_embeddings=True, batch_size=64,
                           show_progress_bar=False)


def arm_frozen(tr, te):
    Etr = e5_embed(tr.sentence.astype(str).tolist())
    Ete = e5_embed(te.sentence.astype(str).tolist())
    folds = strat_folds(tr.label.values)
    classes = np.array(sorted(tr.label.unique()))
    cls_ix = {c: i for i, c in enumerate(classes)}
    P_cv = np.zeros((len(tr), len(classes)))
    for f in range(CV_K):
        m = folds != f
        clf = fit_lr(Etr[m], tr.label[m].values, 3000)
        pr = clf.predict_proba(Etr[~m])
        for j, c in enumerate(clf.classes_):
            P_cv[np.where(~m)[0], cls_ix[c]] = pr[:, j]
    clf = fit_lr(Etr, tr.label.values, 3000)
    pr = clf.predict_proba(Ete)
    P_te = np.zeros((len(te), len(classes)))
    for j, c in enumerate(clf.classes_):
        P_te[:, cls_ix[c]] = pr[:, j]
    return _pack(classes, P_cv, P_te)


def _pack(classes, P_cv, P_te):
    return dict(classes=classes,
                test_pred=classes[P_te.argmax(1)], test_conf=P_te.max(1),
                cv_pred=classes[P_cv.argmax(1)], cv_conf=P_cv.max(1),
                test_probs=P_te, cv_probs=P_cv)


# ---------------------------- fine-tuned encoder --------------------------
MODEL_ID = "jhu-clsp/ettin-encoder-68m"


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _train_encoder(texts, labels, classes, eval_sets, epochs_ckpt, seed=SEED):
    """Train once; return {epochs: {name: prob_matrix}} at each checkpoint."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dev = _device()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    cls_ix = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_ix[l] for l in labels])
    max_ep = max(epochs_ckpt)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(classes)).to(dev)

    def collate(batch):
        ts = [b[0] for b in batch]
        enc = tok(ts, return_tensors="pt", padding=True, truncation=True,
                  max_length=256)                                    # D9
        enc["labels"] = torch.tensor([b[1] for b in batch])
        return enc

    items = list(zip([str(t) for t in texts], y.tolist()))
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(items, batch_size=16, shuffle=True, collate_fn=collate, generator=g)

    cnt = np.bincount(y, minlength=len(classes)).astype(np.float64)
    w = np.where(cnt > 0, len(y) / (len(classes) * np.maximum(cnt, 1)), 0.0)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(dev))

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=5e-5, total_steps=max_ep * len(dl),
        pct_start=0.1, anneal_strategy="linear")

    out = {}
    for ep in range(1, max_ep + 1):
        model.train()
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            lab = batch.pop("labels")
            logits = model(**batch).logits
            loss = lossf(logits, lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if ep in epochs_ckpt:
            model.eval()
            snap = {}
            for name, ts in eval_sets.items():
                ps = []
                with torch.no_grad():
                    for i in range(0, len(ts), 64):
                        enc = tok([str(x) for x in ts[i:i + 64]], return_tensors="pt",
                                  padding=True, truncation=True, max_length=256)
                        enc = {k: v.to(dev) for k, v in enc.items()}
                        ps.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
                snap[name] = np.concatenate(ps) if ps else np.zeros((0, len(classes)))
            out[ep] = snap
    del model
    return out


def arm_encoder(bot, tr, te):
    """CV for epoch selection + OOF confidences, then final fits. Cached."""
    cf = CACHE / f"{bot}_encoder.npz"
    classes = np.array(sorted(tr.label.unique()))
    if cf.exists():
        z = np.load(cf, allow_pickle=True)
        log(f"  [{bot}] encoder: cache hit")
        return {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("cv_")}, \
               {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("te_")}, classes

    folds = strat_folds(tr.label.values)
    P_cv = {ep: np.zeros((len(tr), len(classes))) for ep in EPOCH_GRID}
    for f in range(CV_K):
        m = folds != f
        t0 = time.time()
        res = _train_encoder(tr.sentence[m].tolist(), tr.label[m].tolist(), classes,
                             {"va": tr.sentence[~m].tolist()}, EPOCH_GRID)
        for ep in EPOCH_GRID:
            P_cv[ep][np.where(~m)[0]] = res[ep]["va"]
        log(f"  [{bot}] encoder CV fold {f} done in {time.time()-t0:.0f}s")

    t0 = time.time()
    res = _train_encoder(tr.sentence.tolist(), tr.label.tolist(), classes,
                         {"te": te.sentence.tolist()}, EPOCH_GRID)
    P_te = {ep: res[ep]["te"] for ep in EPOCH_GRID}
    log(f"  [{bot}] encoder final fit done in {time.time()-t0:.0f}s")

    np.savez_compressed(cf, **{f"cv_{ep}": P_cv[ep] for ep in EPOCH_GRID},
                        **{f"te_{ep}": P_te[ep] for ep in EPOCH_GRID})
    return P_cv, P_te, classes


# --------------------------------------------------------------------------
# threshold selection (D5) and the gate (D12)
# --------------------------------------------------------------------------
def pick_threshold_trainonly(cv_conf, irr):
    """t = irr-quantile of out-of-fold in-scope confidence. No OOS needed."""
    return float(np.quantile(np.asarray(cv_conf), irr))


def error_share_low_conf(conf, correct, frac=0.20):
    """Share of ALL errors that fall in the least-confident `frac` of items."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n_err = int((~correct).sum())
    if n_err == 0:
        return None
    k = max(1, int(round(frac * len(conf))))
    order = np.argsort(conf, kind="stable")          # least confident first
    return float((~correct[order[:k]]).sum() / n_err)


def gate(conf, correct):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    out = {"n": int(len(conf)), "n_errors": int((~correct).sum()),
           "base_accuracy": float(correct.mean())}
    if 0 < correct.sum() < len(correct) and len(np.unique(conf)) > 1:
        out["auroc_conf_vs_correct"] = float(roc_auc_score(correct, conf))
    else:
        out["auroc_conf_vs_correct"] = None
        out["auroc_note"] = ("degenerate: constant confidence" if len(np.unique(conf)) <= 1
                             else "degenerate: all items correct or all wrong")
    out["error_share_least_confident_20pct"] = error_share_low_conf(conf, correct, 0.20)
    out["random_reference_20pct"] = 0.20
    out["gate_auroc_pass"] = (out["auroc_conf_vs_correct"] is not None
                              and out["auroc_conf_vs_correct"] >= 0.75)
    es = out["error_share_least_confident_20pct"]
    out["gate_error_share_pass"] = es is not None and es > 0.35
    out["gate_overall_pass"] = bool(out["gate_auroc_pass"] and out["gate_error_share_pass"])
    return out


def escalation_dial(conf, correct, fracs=(0.0, 0.1, 0.2, 0.3, 0.4)):
    """Escalate the least-confident `f`; report accuracy on the KEPT slice."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(conf, kind="stable")
    rows = []
    for f in fracs:
        k = int(round(f * len(conf)))
        kept = order[k:]
        rows.append({"escalated_frac": float(f), "escalated_n": int(k),
                     "kept_n": int(len(kept)),
                     "kept_accuracy": float(correct[kept].mean()) if len(kept) else None})
    return rows


def band_router(conf, correct, frac=0.20):
    """Diagnostic for the non-monotone case (D12): escalate the middle band
    of confidence instead of the tail, sized to the same budget."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n_err = int((~correct).sum())
    if n_err == 0:
        return None
    k = max(1, int(round(frac * len(conf))))
    order = np.argsort(conf, kind="stable")
    best = None
    for start in range(0, len(conf) - k + 1):
        sel = order[start:start + k]
        share = float((~correct[sel]).sum() / n_err)
        if best is None or share > best["error_share"]:
            best = {"error_share": share, "band_start_rank": int(start),
                    "band_conf_lo": float(conf[order[start]]),
                    "band_conf_hi": float(conf[order[start + k - 1]]),
                    "n": int(k)}
    best["note"] = ("band position chosen on test -> OPTIMISTIC; diagnostic only, "
                    "shows whether errors are concentrated anywhere at all")
    return best


# --------------------------------------------------------------------------
# published platforms (paired, D2)
# --------------------------------------------------------------------------
def published(bot, te, inscope_labels, all_train_labels):
    out = {}
    for plat in PLATFORMS:
        p = pd.read_csv(REPO / "preds" / f"{plat}_{bot}.csv")
        assert len(p) == len(te)
        assert (p.sentence.values == te.sentence.values).all()
        assert (p.label.values == te.label.values).all()
        pred = p.predicted_node.values
        conf = p.predicted_node_score.values.astype(float)
        sweep = [metrics_at(te.label.values, pred, conf, t, inscope_labels, all_train_labels)
                 for t in THRESHOLDS]
        fine = [metrics_at(te.label.values, pred, conf, t, inscope_labels, all_train_labels)
                for t in FINE_T]
        out[plat] = {
            "aligned_paired_same_items": True,
            "sweep_paper_grid": sweep,
            "sweep_fine": fine,
            "best_on_test_accuracy": max(fine, key=lambda r: r["accuracy"]),
            "best_on_test_macro_f1_inscope": max(fine, key=lambda r: r["macro_f1_inscope"]),
            "best_on_test_mcc": max(fine, key=lambda r: r["mcc"]),
            "note_best_is_optimistic": "threshold chosen on test; optimistic, applied symmetrically to all methods",
            "per_item": {"pred": [str(x) for x in pred], "conf": [float(x) for x in conf]},
        }
    return out


def verify_against_authors():
    """D1: reproduce the authors' own results/ from their preds/."""
    worst = 0.0
    for bot in BOTS:
        te = pd.read_csv(DATA / f"{bot}_test.csv")
        for plat in PLATFORMS:
            p = pd.read_csv(REPO / "preds" / f"{plat}_{bot}.csv")
            pub = pd.read_csv(REPO / "results" / f"{plat}_{bot}.csv")
            mine = []
            for t in THRESHOLDS:
                eff = effective_pred(p.predicted_node.values,
                                     p.predicted_node_score.values.astype(float), t)
                g = te.label.values
                ins = g != OOS
                mine.append([(eff == g).mean(),
                             f1_score(g, eff, average="weighted", zero_division=0),
                             (eff[ins] == g[ins]).mean(),
                             (eff[~ins] == OOS).mean(),
                             matthews_corrcoef(g, eff)])
            ref = pub[["Accuracy", "Weighted F1", "Inscope Accuracy", "OOS Recall", "MCC"]].values
            worst = max(worst, float(np.abs(np.array(mine) - ref).max()))
    return worst


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    results = {
        "meta": {
            "dataset": "HINT3 v1 full (Arora et al., EMNLP 2020 Insights Workshop, arXiv:2009.13833)",
            "repo": "https://github.com/hellohaptik/HINT3 (master)",
            "seed": SEED,
            "paid_api_calls": 0,
            "device": _device(),
            "protocol": "open-set: train on in-scope intents only; answer only if max prob >= t, else NO_NODES_DETECTED",
            "threshold_operator": ">=",
            "cv_k": CV_K,
            "irr_targets": IRR_TARGETS,
            "primary_irr": PRIMARY_IRR,
            "encoder_epoch_grid": EPOCH_GRID,
        },
        "bots": {},
    }

    log("verifying metric stack against the authors' own results/ files (D1)")
    dev = verify_against_authors()
    results["meta"]["metric_stack_max_abs_deviation_from_published_results_files"] = dev
    results["meta"]["metric_stack_validated"] = bool(dev < 1e-9)
    log(f"  max abs deviation over 675 published numbers: {dev:.3e}")
    assert dev < 1e-9, "metric stack does not reproduce the authors' numbers"

    for bot in BOTS:
        log(f"=== {bot} ===")
        tr, te = load(bot)
        gold = te.label.values
        ins_mask = gold != OOS
        inscope_labels = sorted(set(gold[ins_mask]))          # D3
        all_train_labels = sorted(set(tr.label))
        vc = tr.label.value_counts()

        B = {
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_train_intents": int(tr.label.nunique()),
            "n_test_inscope": int(ins_mask.sum()),
            "n_test_oos": int((~ins_mask).sum()),
            "oos_rate": float((~ins_mask).mean()),
            "labels_per_intent": {"median": float(vc.median()), "min": int(vc.min()),
                                  "max": int(vc.max())},
            "intents_with_lt3_train_ex": int((vc < 3).sum()),
            "intents_with_1_train_ex": int((vc == 1).sum()),
            "test_items_in_lt3_intents": int(pd.Series(gold[ins_mask]).isin(vc[vc < 3].index).sum()),
            "test_items_in_singleton_intents": int(pd.Series(gold[ins_mask]).isin(vc[vc == 1].index).sum()),
            "train_intents_absent_from_test": int(len(set(all_train_labels) - set(inscope_labels))),
            "test_inscope_intents_absent_from_train": int(len(set(inscope_labels) - set(all_train_labels))),
            "contamination": contamination(tr, te),
            "constants": constants(tr, te, inscope_labels, all_train_labels),
            "arms": {},
        }

        # ---- arms ----
        armdata = {}
        log("  arm: majority")
        armdata["majority"] = arm_majority(tr, te)
        log("  arm: tfidf_lr")
        armdata["tfidf_lr"] = arm_tfidf(tr, te, sublinear_char=True)
        log("  arm: frozen_e5_lr")
        armdata["frozen_e5_lr"] = arm_frozen(tr, te)
        log("  arm: ettin_encoder")
        P_cv_ep, P_te_ep, enc_classes = arm_encoder(bot, tr, te)

        # epoch selection on TRAIN CV only (D8)
        ep_cv = {}
        for ep in EPOCH_GRID:
            pc = enc_classes[P_cv_ep[ep].argmax(1)]
            ep_cv[ep] = float(f1_score(tr.label.values, pc,
                                       labels=all_train_labels, average="macro", zero_division=0))
        best_ep = max(EPOCH_GRID, key=lambda e: ep_cv[e])
        B["encoder_epoch_selection"] = {"cv_macro_f1_by_epochs": ep_cv,
                                        "selected_epochs": int(best_ep),
                                        "selected_on": "train CV out-of-fold macro-F1 (no test)"}
        armdata[f"ettin_encoder_{SPEC_EPOCHS}ep"] = _pack(
            enc_classes, P_cv_ep[SPEC_EPOCHS], P_te_ep[SPEC_EPOCHS])
        if best_ep != SPEC_EPOCHS:
            armdata[f"ettin_encoder_cv{best_ep}ep"] = _pack(
                enc_classes, P_cv_ep[best_ep], P_te_ep[best_ep])

        # D11 ablation
        alt = arm_tfidf(tr, te, sublinear_char=False)
        B["tfidf_sublinear_char_ablation"] = {
            "sublinear_on_both_inscope_acc_at_t0": float(
                (armdata["tfidf_lr"]["test_pred"][ins_mask] == gold[ins_mask]).mean()),
            "sublinear_word_only_inscope_acc_at_t0": float(
                (alt["test_pred"][ins_mask] == gold[ins_mask]).mean()),
        }

        # ---- per-arm metrics ----
        probs_to_save = {}
        for name, a in armdata.items():
            cvp, cvc = a["cv_pred"], a["cv_conf"]
            cv_macro = float(f1_score(tr.label.values, cvp, labels=all_train_labels,
                                      average="macro", zero_division=0))
            cv_acc = float((cvp == tr.label.values).mean())
            ts = {}
            for irr in IRR_TARGETS:
                t = pick_threshold_trainonly(cvc, irr)
                ts[f"irr_{irr:.2f}"] = {
                    "threshold": t,
                    "selected_on": "train out-of-fold confidence quantile (OOS-free, D5)",
                    "metrics": metrics_at(gold, a["test_pred"], a["test_conf"], t,
                                          inscope_labels, all_train_labels),
                }
            fine = [metrics_at(gold, a["test_pred"], a["test_conf"], t,
                               inscope_labels, all_train_labels) for t in FINE_T]
            B["arms"][name] = {
                "cv_macro_f1_train_oof": cv_macro,
                "cv_accuracy_train_oof": cv_acc,
                "no_rejection_t0": metrics_at(gold, a["test_pred"], a["test_conf"], 0.0,
                                              inscope_labels, all_train_labels),
                "sweep_paper_grid": [metrics_at(gold, a["test_pred"], a["test_conf"], t,
                                                inscope_labels, all_train_labels)
                                     for t in THRESHOLDS],
                "sweep_fine": fine,
                "train_selected_operating_points": ts,
                "best_on_test_accuracy_OPTIMISTIC": max(fine, key=lambda r: r["accuracy"]),
                "best_on_test_macro_f1_OPTIMISTIC": max(fine, key=lambda r: r["macro_f1_inscope"]),
                "best_on_test_mcc_OPTIMISTIC": max(fine, key=lambda r: r["mcc"]),
                "per_item": {
                    "pred_argmax": [str(x) for x in a["test_pred"]],
                    "gold": [str(x) for x in gold],
                    "conf": [float(x) for x in a["test_conf"]],
                },
            }
            probs_to_save[f"{bot}__{name}__test_probs"] = a["test_probs"]
            probs_to_save[f"{bot}__{name}__classes"] = a["classes"]

        # ---- shipped row, chosen on train CV only (D7) ----
        cand = [k for k in B["arms"] if k != "majority"]
        shipped = max(cand, key=lambda k: B["arms"][k]["cv_macro_f1_train_oof"])
        a = armdata[shipped]
        t_star = pick_threshold_trainonly(a["cv_conf"], PRIMARY_IRR)
        eff = effective_pred(a["test_pred"], a["test_conf"], t_star)

        corr_A = (eff == gold)                                   # shipped pipeline
        corr_B = (a["test_pred"][ins_mask] == gold[ins_mask])     # answering subset
        conf_B = np.asarray(a["test_conf"])[ins_mask]

        B["shipped_row"] = {
            "arm": shipped,
            "selected_on": "highest train out-of-fold CV macro-F1 among non-constant arms (no test)",
            "operating_threshold": t_star,
            "operating_threshold_selected_on": f"train OOF confidence {PRIMARY_IRR:.0%} quantile (D5)",
            "metrics_at_operating_point": metrics_at(gold, a["test_pred"], a["test_conf"],
                                                     t_star, inscope_labels, all_train_labels),
            "gate_A_shipped_pipeline": gate(a["test_conf"], corr_A),
            "gate_B_answering_subset_inscope_only": gate(conf_B, corr_B),
            "gate_definitions": {
                "A": "correct = (thresholded pred == gold) over all test items; the output we would ship",
                "B": "correct = (argmax == gold) over gold-in-scope items only; comparable to banking77/ToxicChat/CUAD reference rows",
            },
            "escalation_dial_A_shipped_pipeline": escalation_dial(a["test_conf"], corr_A),
            "escalation_dial_B_answering_subset": escalation_dial(conf_B, corr_B),
            "band_router_diagnostic_A": band_router(a["test_conf"], corr_A),
            "framing": "coverage/cost result only; this project measured out-of-fold that a cascade never beats the better base model",
        }

        # per-intent breakdown for the tiny intents (D6)
        small = sorted(vc[vc < 3].index)
        pi = []
        for lab in small:
            m = gold == lab
            if m.sum() == 0:
                continue
            pi.append({"intent": str(lab), "train_ex": int(vc[lab]), "test_items": int(m.sum()),
                       "argmax_correct": int((a["test_pred"][m] == lab).sum()),
                       "kept_at_operating_t": int((eff[m] == lab).sum())})
        B["tiny_intent_breakdown_shipped_row"] = pi

        B["published_platforms"] = published(bot, te, inscope_labels, all_train_labels)

        # paired McNemar, shipped row vs each platform, each at its own test-optimal t
        mc = {}
        for plat in PLATFORMS:
            pf = B["published_platforms"][plat]
            bt = pf["best_on_test_macro_f1_inscope"]["threshold"]
            p = pd.read_csv(REPO / "preds" / f"{plat}_{bot}.csv")
            eff_p = effective_pred(p.predicted_node.values,
                                   p.predicted_node_score.values.astype(float), bt)
            c_p = (eff_p == gold)
            our_best_t = B["arms"][shipped]["best_on_test_macro_f1_OPTIMISTIC"]["threshold"]
            c_o = (effective_pred(a["test_pred"], a["test_conf"], our_best_t) == gold)
            b = int((c_o & ~c_p).sum())
            c = int((~c_o & c_p).sum())
            from scipy.stats import binomtest
            pv = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
            mc[plat] = {"ours_right_theirs_wrong": b, "theirs_right_ours_wrong": c,
                        "mcnemar_exact_p": float(pv),
                        "ours_acc": float(c_o.mean()), "theirs_acc": float(c_p.mean()),
                        "ours_t": float(our_best_t), "theirs_t": float(bt),
                        "note": "each method at its own test-optimal macro-F1 threshold -> optimistic for BOTH, symmetric"}
        B["paired_mcnemar_vs_published"] = mc

        results["bots"][bot] = B
        np.savez_compressed(HERE / f"hint3_probs_{bot}.npz", **probs_to_save)
        log(f"  shipped row = {shipped}  t*={t_star:.4f}")

    with open(HERE / "hint3.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    log(f"wrote {HERE/'hint3.json'}")
    return results


if __name__ == "__main__":
    main()
