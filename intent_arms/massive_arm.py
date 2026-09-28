"""MASSIVE (arXiv:2204.08582) -- the free arm outside English.

60-intent virtual-assistant utterance classification, 51 languages (52 locale
directories ship on the parquet branch; see below). This is the project's first
NON-ENGLISH measurement arm: every suite measured so far is English, while the
frozen-embedding row already runs a multilingual model, so the open question is
whether the free arm survives a script change and a resource-level change, and
where it breaks.

Run as:  massive_arm.py cheap | encoder | gate | all

---------------------------------------------------------------------------
DESIGN DECISIONS, AND THE MEASUREMENT THAT FORCED EACH (house style)
---------------------------------------------------------------------------

* OFFICIAL PER-LOCALE SPLITS, never a pooled random split. MASSIVE's locales
  are parallel translations of one utterance pool sharing an `id`, so a random
  pooled split puts a Spanish utterance in train and its German twin in test.
  This project has already shipped two suites burned by that exact class of
  error (FinBen headlines 71.5%, BeaverTails 99.8% of test rows sharing a
  prompt with train), so it was checked rather than assumed. MEASURED in
  massive_contam.py: 16,521 distinct ids (= 11,514 train + 2,033 dev + 2,974
  test), every id present in all 52 locales, and **0 ids whose partition
  disagrees between locales (0.0000%)**. The official split is exactly
  id-aligned, so a per-locale run is automatically free of the cross-lingual
  twin leak and no custom id-holdout is needed. Any cross-lingual transfer run
  inherits the same guarantee for free, because it reuses these partitions.

* `utt` IS THE TEXT, never `annot_utt`. `annot_utt` carries inline slot
  annotations ("wake me up at [time : nine am]"), and the bracketed slot NAME
  is a label-correlated feature the deployed model would not have. Using it is
  the same feature leak that once read as 99-100% accuracy in this project.

* INTENT NAMES COME FROM THE PARQUET SCHEMA METADATA, not a hand-written list.
  `intent` is an int64; the 60 names live under
  huggingface.info.features.intent.names. A hand-written column list is what
  produced the diversity_1-6 leak, so the mapping is read from the file.

* 13 LOCALES, PRE-COMMITTED BEFORE ANY RESULT WAS SEEN. Four arms x 51 locales
  is not affordable, so breadth is bought on the two cheap arms only. The list
  spans writing system and resource level, which are the two axes that could
  plausibly break a free arm:
      en-US  Latin, English          -- the anchor, comparable to every other
                                        row in the project
      de-DE  Latin, high-resource
      es-ES  Latin, high-resource
      fr-FR  Latin, high-resource
      ru-RU  Cyrillic, high-resource
      zh-CN  Han, NO WHITESPACE word boundaries
      ja-JP  Han+kana, NO WHITESPACE word boundaries
      ko-KR  Hangul, agglutinative
      ar-SA  Arabic, right-to-left, rich morphology
      hi-IN  Devanagari
      th-TH  Thai, NO WHITESPACE word boundaries
      sw-KE  Latin, LOW-resource
      am-ET  Ge'ez script, LOW-resource
  Three no-whitespace scripts and two low-resource languages are deliberate:
  they are the pre-registered predictions this arm exists to test. Nothing was
  added after seeing a result; any later addition is labelled as such.

* THE WORD/CHAR HALVES OF TF-IDF ARE SCORED SEPARATELY, not just the union.
  The pre-registered mechanism claim is that a word 1-2gram vectorizer must
  collapse on zh-CN / ja-JP / th-TH, because those scripts have no whitespace
  and sklearn's default token pattern splits on whitespace -- so a "word" is a
  whole clause and almost every feature is hapax. Reporting only the union
  would let the char_wb half silently rescue it and turn a mechanism finding
  into an assertion. Halves are run for all 13 locales, not only the three
  that motivated it, so the contrast has controls.

* CONSTANT-PREDICTOR GUARD on every row (project rule). 60 intents are not
  balanced; the majority intent is carried per locale and any metric a
  label-blind constant wins is flagged.

* BOTH HEADLINE METRICS. MASSIVE's own metric is intent accuracy, so that is
  primary for comparability with the paper's Table 8. Macro-F1 over 60 intents
  is reported alongside because the intents are unbalanced and accuracy hides
  the tail. The smallest per-intent test count is recorded so the resolution of
  any per-class claim is visible.

* PER-ITEM PREDICTIONS **AND** CONFIDENCES PERSISTED FOR EVERY ARM. This
  project carries a standing known limitation: an earlier encoder runner kept
  only logits.argmax(-1) and threw the distribution away, so its strongest row
  could not be asked how sure it was and every chart in the project is stuck on
  the TF-IDF row. Every arm here returns (pred, gold, conf) per item, including
  the encoders, specifically so that limitation does not propagate.

* TWO FINE-TUNED ENCODERS, and the reason is BLURB. `jhu-clsp/ettin-encoder-68m`
  is the project's standard encoder and it is ENGLISH. PRE-REGISTERED
  PREDICTION, recorded here before the run: it will hold up on en-US, degrade
  on Latin-script non-English, and collapse on non-Latin scripts (zh-CN, th-TH)
  because its tokenizer will shatter unseen scripts into bytes/UNK. BLURB is
  the precedent -- there the project's cheap arm lost by 33 points to
  PubMedBERT and the true finding was "we picked the wrong free model for this
  domain", not "free loses here". To avoid repeating that mistake in the
  opposite direction, `intfloat/multilingual-e5-small` is ALSO fine-tuned on
  the same locales. That choice is deliberate: it is the same checkpoint the
  frozen arm uses, so frozen-vs-fine-tuned isolates the head-vs-backbone
  question with the representation held fixed.

* CALIBRATION GAP IS NOT A GATE. Ranking correctness and being calibrated are
  different properties; the gate is AUROC(confidence -> correct) >= ~0.75 plus
  share of all errors in the least-confident 20% well above 20%. The gate is
  evaluated on the row we would actually SHIP (the most accurate arm available
  for that locale), because evaluating it on the weakest row once reversed a
  verdict in this project.

* ZERO PAID API CALLS. Every arm here is local. The published mT5/XLM-R figures
  come from the paper's Table 8 via curl, and are labelled
  published-on-authors'-protocol -- weaker evidence than a paired same-items
  run, which this arm does not have.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "AmazonScience/massive"
REV = "refs/convert/parquet"
S = "/private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad"
SEED = 0

# Pre-committed, in the docstring above, before any result was seen.
LOCALES = ["en-US", "de-DE", "es-ES", "fr-FR", "ru-RU", "zh-CN", "ja-JP",
           "ko-KR", "ar-SA", "hi-IN", "th-TH", "sw-KE", "am-ET"]
NO_WHITESPACE = {"zh-CN", "ja-JP", "th-TH"}
ANCHOR = "en-US"
# Encoder locales: the anchor, one Latin non-English control, and the two
# no-whitespace / non-Latin cases the prediction is about.
ENC_LOCALES = ["en-US", "de-DE", "zh-CN", "th-TH"]
ENGLISH_ENCODER = "jhu-clsp/ettin-encoder-68m"
MULTILINGUAL_ENCODER = "intfloat/multilingual-e5-small"
FROZEN_MODEL = "intfloat/multilingual-e5-small"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def intent_names() -> list[str]:
    """The 60 names, read from parquet schema metadata rather than typed out."""
    p = hf_hub_download(REPO, f"{ANCHOR}/train/0000.parquet",
                        repo_type="dataset", revision=REV)
    info = json.loads(pq.read_schema(p).metadata[b"huggingface"])
    names = info["info"]["features"]["intent"]["names"]
    assert len(names) == 60, f"expected 60 intents, got {len(names)}"
    return names


def load(loc: str, part: str):
    p = hf_hub_download(REPO, f"{loc}/{part}/0000.parquet",
                        repo_type="dataset", revision=REV)
    t = pq.read_table(p, columns=["id", "utt", "intent"]).to_pydict()
    return t["id"], t["utt"], np.asarray(t["intent"], dtype=np.int64)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def score(gold, pred) -> dict:
    from sklearn.metrics import f1_score
    gold, pred = np.asarray(gold), np.asarray(pred)
    return {"accuracy": float((gold == pred).mean()),
            "macro_f1": float(f1_score(gold, pred, average="macro",
                                       zero_division=0))}


def constant_row(train_y, test_y) -> dict:
    """Majority intent learned on train, applied to test: the label-blind guard."""
    maj = int(np.bincount(train_y, minlength=60).argmax())
    pred = np.full(len(test_y), maj)
    m = score(test_y, pred)
    m["predicts_intent"] = maj
    return m


def gate(gold, pred, conf) -> dict:
    """AUROC(conf -> correct) and the share of all errors in the least-confident
    20%. Calibration gap is deliberately NOT computed: it is not the gate."""
    from sklearn.metrics import roc_auc_score
    gold, pred, conf = np.asarray(gold), np.asarray(pred), np.asarray(conf, float)
    correct = (gold == pred).astype(int)
    n, n_err = len(gold), int((1 - correct).sum())
    out = {"n": n, "n_errors": n_err, "accuracy": float(correct.mean())}
    if n_err == 0 or n_err == n:
        out.update(auroc=None, err_share_low20=None, note="degenerate")
        return out
    out["auroc"] = float(roc_auc_score(correct, conf))
    # least-confident 20%, ties broken by index for determinism (SEED=0 world)
    order = np.lexsort((np.arange(n), conf))
    k = int(round(0.20 * n))
    out["err_share_low20"] = float((1 - correct[order[:k]]).sum() / n_err)
    out["k_low20"] = k
    # escalation dial: accuracy on the KEPT slice after escalating the least
    # confident f. Framed as coverage/cost, never as beating an LLM.
    dial = {}
    for f in (0.0, 0.10, 0.20, 0.30, 0.40):
        kk = int(round(f * n))
        kept = order[kk:]
        dial[f"{int(f*100)}%"] = {"escalated": kk,
                                  "coverage": float(len(kept) / n),
                                  "accuracy_on_kept": float(correct[kept].mean())}
    out["escalation_dial"] = dial
    return out


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def tfidf_arm(tr_x, tr_y, te_x, variant: str):
    """variant: 'union' (project setting), 'word' or 'char' (the halves)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion

    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec = {"word": word, "char": char,
           "union": FeatureUnion([("w", word), ("c", char)])}[variant]
    Xtr = vec.fit_transform(tr_x)
    Xte = vec.transform(te_x)
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                             random_state=SEED).fit(Xtr, tr_y)
    P = clf.predict_proba(Xte)
    pred = clf.classes_[P.argmax(1)]
    return pred, P.max(1), {"n_features": int(Xtr.shape[1])}


def embed(texts, model):
    """e5 wants a 'query: ' prefix and L2-normalised output."""
    return model.encode([f"query: {t}" for t in texts],
                        batch_size=256, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)


def frozen_arm(tr_x, tr_y, te_x, model):
    from sklearn.linear_model import LogisticRegression
    Etr, Ete = embed(tr_x, model), embed(te_x, model)
    clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced",
                             random_state=SEED).fit(Etr, tr_y)
    P = clf.predict_proba(Ete)
    pred = clf.classes_[P.argmax(1)]
    return pred, P.max(1), {"dim": int(Etr.shape[1])}


def encoder_arm(tr_x, tr_y, te_x, model_name: str):
    """Fine-tuned encoder. Keeps the FULL softmax, not just argmax -- that is
    the standing limitation this arm exists not to repeat."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=60).to(dev)

    def enc(xs):
        # max_length=256 is the TRUNCATION cap, per protocol. Padding is to the
        # longest sequence present rather than a flat 256: these are single
        # utterances (en-US ~10 pieces, worst case am-ET under ettin ~77), so
        # padding to 256 would have burned ~25x the compute for a result the
        # attention mask makes identical.
        b = tok(list(xs), truncation=True, max_length=256, padding="longest",
                return_tensors="pt")
        return b["input_ids"], b["attention_mask"]

    itr, mtr = enc(tr_x)
    ite, mte = enc(te_x)
    ytr = torch.tensor(tr_y, dtype=torch.long)
    dl = DataLoader(TensorDataset(itr, mtr, ytr), batch_size=16, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED))

    cnt = np.bincount(tr_y, minlength=60).astype(np.float64)
    w = np.where(cnt > 0, len(tr_y) / (60 * np.maximum(cnt, 1)), 0.0)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(dev))
    opt = torch.optim.AdamW(mdl.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 3
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=5e-5, total_steps=epochs * len(dl), pct_start=0.1,
        anneal_strategy="linear")

    mdl.train()
    for ep in range(epochs):
        tot = 0.0
        for ii, mm, yy in dl:
            opt.zero_grad()
            out = mdl(input_ids=ii.to(dev), attention_mask=mm.to(dev))
            loss = lossf(out.logits, yy.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        print(f"      epoch {ep+1}/{epochs} loss {tot/len(dl):.4f}", flush=True)

    mdl.eval()
    probs = []
    with torch.no_grad():
        for a in range(0, len(te_x), 64):
            out = mdl(input_ids=ite[a:a + 64].to(dev),
                      attention_mask=mte[a:a + 64].to(dev))
            probs.append(torch.softmax(out.logits.float(), -1).cpu().numpy())
    P = np.concatenate(probs)
    return P.argmax(1), P.max(1), {"device": dev, "epochs": epochs}


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def tokenisation_probe(names):
    """Evidence for the whitespace claim, independent of any model: how many
    whitespace tokens does an utterance have, per locale."""
    probe = {}
    for loc in LOCALES:
        _, x, _ = load(loc, "train")
        wc = np.array([len(t.split()) for t in x])
        ch = np.array([len(t) for t in x])
        probe[loc] = {"mean_whitespace_tokens": float(wc.mean()),
                      "mean_chars": float(ch.mean()),
                      "chars_per_token": float(ch.mean() / wc.mean())}
    return probe


def run_cheap(out_path):
    from sentence_transformers import SentenceTransformer
    names = intent_names()
    t0 = time.time()
    res = {"intent_names": names, "locales_run": LOCALES,
           "published_note": "mT5/XLM-R figures are published-on-authors'-protocol "
                             "(arXiv:2204.08582 Table 8), NOT paired same-items.",
           "per_locale": {}}
    res["tokenisation_probe"] = tokenisation_probe(names)
    print("tokenisation probe (mean whitespace tokens per train utterance):")
    for loc, v in res["tokenisation_probe"].items():
        tag = "  <- no-whitespace script" if loc in NO_WHITESPACE else ""
        print(f"  {loc}: {v['mean_whitespace_tokens']:5.2f} tokens, "
              f"{v['mean_chars']:5.1f} chars, "
              f"{v['chars_per_token']:5.2f} chars/token{tag}")

    print(f"\nloading {FROZEN_MODEL} ...", flush=True)
    st = SentenceTransformer(FROZEN_MODEL)

    for loc in LOCALES:
        _, tr_x, tr_y = load(loc, "train")
        te_id, te_x, te_y = load(loc, "test")
        cnt = np.bincount(te_y, minlength=60)
        seen = cnt[cnt > 0]
        L = {"n_train": len(tr_x), "n_test": len(te_x),
             "n_intents_in_test": int((cnt > 0).sum()),
             "smallest_intent_test_count": int(seen.min()),
             "largest_intent_test_count": int(seen.max()),
             "gold": [int(v) for v in te_y], "test_ids": list(te_id),
             "arms": {}}
        L["constant"] = constant_row(tr_y, te_y)
        L["arms"]["majority"] = dict(L["constant"],
                                     pred=[L["constant"]["predicts_intent"]] * len(te_y),
                                     conf=[1.0] * len(te_y))
        for variant in ("union", "word", "char"):
            pred, conf, meta = tfidf_arm(tr_x, tr_y, te_x, variant)
            m = score(te_y, pred)
            L["arms"][f"tfidf_{variant}"] = dict(
                m, **meta, pred=[int(v) for v in pred],
                conf=[float(v) for v in conf])
            print(f"  {loc} tfidf_{variant:5}: acc {m['accuracy']:.4f} "
                  f"macroF1 {m['macro_f1']:.4f} feats {meta['n_features']:>7,} "
                  f"({time.time()-t0:,.0f}s)", flush=True)
        pred, conf, meta = frozen_arm(tr_x, tr_y, te_x, st)
        m = score(te_y, pred)
        L["arms"]["frozen_e5"] = dict(m, **meta, pred=[int(v) for v in pred],
                                      conf=[float(v) for v in conf])
        print(f"  {loc} frozen_e5   : acc {m['accuracy']:.4f} "
              f"macroF1 {m['macro_f1']:.4f}   [majority {L['constant']['accuracy']:.4f}] "
              f"({time.time()-t0:,.0f}s)", flush=True)
        res["per_locale"][loc] = L
        json.dump(res, open(out_path, "w"))
    print(f"\nwrote {out_path} ({time.time()-t0:,.0f}s)")


def run_encoders(path):
    res = json.load(open(path))
    t0 = time.time()
    for loc in ENC_LOCALES:
        _, tr_x, tr_y = load(loc, "train")
        _, te_x, te_y = load(loc, "test")
        for tag, mname in (("encoder_ettin_en", ENGLISH_ENCODER),
                           ("encoder_e5_multi", MULTILINGUAL_ENCODER)):
            if tag in res["per_locale"][loc]["arms"]:
                print(f"  {loc} {tag}: already present, skipping")
                continue
            print(f"  {loc} {tag} ({mname}) ...", flush=True)
            pred, conf, meta = encoder_arm(tr_x, tr_y, te_x, mname)
            m = score(te_y, pred)
            res["per_locale"][loc]["arms"][tag] = dict(
                m, **meta, model=mname, pred=[int(v) for v in pred],
                conf=[float(v) for v in conf])
            print(f"  {loc} {tag}: acc {m['accuracy']:.4f} "
                  f"macroF1 {m['macro_f1']:.4f} ({time.time()-t0:,.0f}s)", flush=True)
            json.dump(res, open(path, "w"))
    print(f"\nencoders done ({time.time()-t0:,.0f}s)")


def run_gate(path):
    """Gate the row we would actually SHIP: the most accurate arm per locale.

    SHIPPABLE excludes `majority` (label-blind) and the two TF-IDF HALVES.
    The halves exist as diagnostics for the whitespace/combining-mark
    mechanism; `tfidf_union` is the project's standard configuration and the
    row comparable to every other suite. Stated because the choice is not
    free: char-only edges out the union on a few locales, but by 0.3-0.5
    points, which is inside the +-1.8-point 95% binomial interval at
    n=2,974, so nothing is being hidden by the rule. `best_any_arm` is
    recorded alongside so the decision is auditable.
    """
    res = json.load(open(path))
    pub = json.load(open(os.path.join(S, "massive_published.json")))
    shippable = lambda a: a not in ("majority", "tfidf_word", "tfidf_char")
    for loc, L in res["per_locale"].items():
        gold = L["gold"]
        n = L["n_test"]
        for a, v in L["arms"].items():
            # 95% binomial (Wald) half-width on accuracy, so "gap" claims are
            # readable against the resolution of a 2,974-item test split.
            p = v["accuracy"]
            v["acc_ci95_halfwidth"] = float(1.96 * (p * (1 - p) / n) ** 0.5)
        best = max((a for a in L["arms"] if shippable(a)),
                   key=lambda a: L["arms"][a]["accuracy"])
        L["ship_arm"] = best
        L["best_any_arm"] = max((a for a in L["arms"] if a != "majority"),
                                key=lambda a: L["arms"][a]["accuracy"])
        L["gates"] = {}
        for a, v in L["arms"].items():
            if a == "majority":
                continue
            L["gates"][a] = gate(gold, v["pred"], v["conf"])
        g = L["gates"][best]
        L["ship_gate"] = g
        L["published"] = pub.get(loc)
        ok = (g["auroc"] is not None and g["auroc"] >= 0.75
              and g["err_share_low20"] is not None and g["err_share_low20"] > 0.20)
        L["ship_gate_passes"] = bool(ok)

    # Fold the audit + probe figures into the single machine-readable artefact,
    # so massive.json alone answers "was this split contaminated" without
    # needing the four side files.
    def side(name):
        p = os.path.join(S, name)
        return json.load(open(p)) if os.path.exists(p) else None

    c = side("massive_contam.json") or {}
    res["contamination"] = {
        k: c.get(k) for k in
        ("n_locale_dirs", "not_in_paper_table", "n_distinct_ids",
         "ids_in_all_locales", "ids_in_all_locales_share",
         "ids_partition_disagree", "ids_partition_disagree_share",
         "en_us_ids", "en_us_partition_agrees_everywhere",
         "en_us_partition_agrees_share")}
    res["contamination"]["within_locale_dupes"] = {
        l: c.get("within_locale_dupes", {}).get(l) for l in res["per_locale"]}
    res["contamination"]["within_locale_dupes_all_locales"] = \
        c.get("within_locale_dupes")
    # Cross-lingual zero-shot lives under its own key per locale, NOT inside
    # `arms`, so it can never be picked as a ship row: it is a different
    # protocol (one classifier trained on en-US only) and is not comparable to
    # the per-locale supervised arms.
    z = side("massive_zeroshot.json")
    if z:
        for loc, v in z.items():
            if loc in res["per_locale"]:
                res["per_locale"][loc]["zeroshot_frozen_e5"] = v
                res["per_locale"][loc]["zeroshot_frozen_e5"]["gate"] = gate(
                    v["gold"], v["pred"], v["conf"])
    res["published_baselines"] = side("massive_published.json")
    res["tokenizer_fertility_probe"] = side("massive_tokprobe.json")
    res["word_tokenisation_probe"] = side("massive_wordprobe.json")
    res["token_pattern_fix_experiment"] = side("massive_tokenfix.json")
    json.dump(res, open(path, "w"))
    print(f"{'locale':8} {'ship arm':18} {'acc':>7} {'AUROC':>7} {'err@low20':>10}  gate")
    for loc, L in res["per_locale"].items():
        g = L["ship_gate"]
        print(f"{loc:8} {L['ship_arm']:18} {g['accuracy']:>7.4f} "
              f"{g['auroc']:>7.3f} {g['err_share_low20']:>10.1%}  "
              f"{'PASS' if L['ship_gate_passes'] else 'FAIL'}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    p = os.path.join(S, "massive.json")
    if what in ("cheap", "all"):
        run_cheap(p)
    if what in ("encoder", "all"):
        run_encoders(p)
    if what in ("gate", "all"):
        run_gate(p)
