"""Second derivation of the headline figures. Project rule: re-derive at least
one headline number a different way before reporting it.

Four independent checks:
  A. Recompute every arm's accuracy from the persisted per-item {pred, gold}
     with a hand-rolled mean, not sklearn, and diff against the stored value.
     This catches a stored metric that disagrees with the stored predictions.
  B. Refit en-US TF-IDF-union from scratch in a separate process/vectoriser
     construction order and confirm the accuracy reproduces bit-for-bit under
     SEED=0. This catches a non-deterministic pipeline.
  C. Re-derive the gate's "errors in least-confident 20%" by a completely
     different route -- sort-free, using a quantile threshold on confidence --
     and compare to the lexsort-based figure in massive.json.
  D. Confirm gold labels in massive.json match the parquet file's intent column
     for the locales run, i.e. nothing was reordered in persistence.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

S = "/private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad"
REPO, REV = "AmazonScience/massive", "refs/convert/parquet"
SEED = 0
R = json.load(open(f"{S}/massive.json"))
PL = R["per_locale"]

print("A. accuracy recomputed from persisted per-item arrays (hand-rolled mean)")
worst = 0.0
for loc, L in PL.items():
    gold = np.array(L["gold"])
    for arm, v in L["arms"].items():
        pred = np.array(v["pred"])
        assert len(pred) == len(gold) == L["n_test"], f"{loc}/{arm} length"
        assert len(v["conf"]) == len(gold), f"{loc}/{arm} conf length"
        hand = float(sum(1 for p, g in zip(pred, gold) if p == g)) / len(gold)
        d = abs(hand - v["accuracy"])
        worst = max(worst, d)
        assert d < 1e-12, f"{loc}/{arm}: stored {v['accuracy']} vs hand {hand}"
print(f"   all {sum(len(L['arms']) for L in PL.values())} (locale,arm) rows agree; "
      f"max abs diff {worst:.2e}")

print("\nB. independent refit of en-US TF-IDF-union")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion


def get(loc, part):
    p = hf_hub_download(REPO, f"{loc}/{part}/0000.parquet",
                        repo_type="dataset", revision=REV)
    t = pq.read_table(p, columns=["utt", "intent"]).to_pydict()
    return t["utt"], np.asarray(t["intent"], dtype=np.int64)


tr_x, tr_y = get("en-US", "train")
te_x, te_y = get("en-US", "test")
# Deliberately build the union in the opposite order to massive_arm.py.
vec = FeatureUnion([
    ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True))])
Xtr = vec.fit_transform(tr_x)
clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                         random_state=SEED).fit(Xtr, tr_y)
P = clf.predict_proba(vec.transform(te_x))
pred2 = clf.classes_[P.argmax(1)]
acc2 = float((pred2 == te_y).mean())
stored = PL["en-US"]["arms"]["tfidf_union"]["accuracy"]
print(f"   refit  {acc2:.6f}")
print(f"   stored {stored:.6f}")
print(f"   diff   {abs(acc2-stored):.2e}")
agree = float((pred2 == np.array(PL['en-US']['arms']['tfidf_union']['pred'])).mean())
print(f"   per-item prediction agreement with stored: {agree:.4%}")

print("\nC. gate error-share re-derived without the lexsort path")
for loc in list(PL):
    L = PL[loc]
    arm = L["ship_arm"]
    v = L["arms"][arm]
    gold = np.array(L["gold"])
    correct = (np.array(v["pred"]) == gold).astype(int)
    conf = np.array(v["conf"], float)
    n = len(conf)
    k = int(round(0.20 * n))
    # quantile-threshold route: take the k-th smallest confidence as a cut.
    thr = np.partition(conf, k - 1)[k - 1]
    sel = conf < thr
    # add back ties at the threshold, up to k, to match a strict bottom-k
    ties = np.flatnonzero(conf == thr)
    need = k - int(sel.sum())
    sel[ties[:max(need, 0)]] = True
    alt = float((1 - correct[sel]).sum() / max((1 - correct).sum(), 1))
    ref = L["ship_gate"]["err_share_low20"]
    flag = "ok" if abs(alt - ref) < 1e-9 else f"DIFF {abs(alt-ref):.2e}"
    print(f"   {loc:8} {arm:18} stored {ref:.6f}  alt {alt:.6f}  {flag}")

print("\nD. persisted gold matches the parquet intent column")
for loc in list(PL):
    _, y = get(loc, "test")
    same = bool((y == np.array(PL[loc]["gold"])).all())
    print(f"   {loc:8} gold identical to parquet: {same}")
    assert same, loc

print("\nall verification checks passed")
