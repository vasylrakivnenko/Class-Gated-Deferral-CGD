"""Turn the combining-mark hypothesis into a result.

massive_wordprobe.py showed sklearn's default token_pattern r"(?u)\\b\\w\\w+\\b"
rejects 88.5% of hi-IN and 80.2% of th-TH whitespace tokens, because Devanagari
and Thai vowel signs are Unicode Mn/Mc and \\w does not match them. That
predicts a word-only TF-IDF should recover most of its lost ground under a
pattern that admits marks. Predicted BEFORE running: hi-IN word-only rises from
0.5797 toward the char-only 0.8211; th-TH rises from 0.7095 toward 0.8090.

The replacement pattern is r"[^\\s]{2,}" -- split on whitespace only, keep
anything 2+ non-space characters. Deliberately crude: the point is to isolate
the mark-dropping cause, not to tune a vectorizer. Locales without combining
marks are included as controls; they should barely move.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

S = "/private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad"
REPO, REV = "AmazonScience/massive", "refs/convert/parquet"
SEED = 0
FIX = r"[^\s]{2,}"
# the two affected locales, plus controls that should not move
LOCS = ["hi-IN", "th-TH", "en-US", "ru-RU", "zh-CN"]


def get(loc, part):
    p = hf_hub_download(REPO, f"{loc}/{part}/0000.parquet",
                        repo_type="dataset", revision=REV)
    t = pq.read_table(p, columns=["utt", "intent"]).to_pydict()
    return t["utt"], np.asarray(t["intent"], dtype=np.int64)


def run(tr_x, tr_y, te_x, te_y, pattern):
    kw = {} if pattern is None else {"token_pattern": pattern}
    V = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, **kw)
    Xtr = V.fit_transform(tr_x)
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                             random_state=SEED).fit(Xtr, tr_y)
    pred = clf.classes_[clf.predict_proba(V.transform(te_x)).argmax(1)]
    return (float((pred == te_y).mean()),
            float(f1_score(te_y, pred, average="macro", zero_division=0)),
            int(Xtr.shape[1]))


def main():
    R = json.load(open(f"{S}/massive.json"))
    out = {}
    print(f"{'locale':8} {'word default':>13} {'word mark-aware':>16} "
          f"{'delta':>7} {'char-only':>10} {'union':>8} {'feats def':>10} {'feats fix':>10}")
    for loc in LOCS:
        tr_x, tr_y = get(loc, "train")
        te_x, te_y = get(loc, "test")
        d_acc, d_f1, d_n = run(tr_x, tr_y, te_x, te_y, None)
        f_acc, f_f1, f_n = run(tr_x, tr_y, te_x, te_y, FIX)
        stored = R["per_locale"][loc]["arms"]["tfidf_word"]["accuracy"]
        assert abs(d_acc - stored) < 1e-12, f"{loc}: default refit {d_acc} != stored {stored}"
        ch = R["per_locale"][loc]["arms"]["tfidf_char"]["accuracy"]
        un = R["per_locale"][loc]["arms"]["tfidf_union"]["accuracy"]
        out[loc] = {"word_default_acc": d_acc, "word_default_macro_f1": d_f1,
                    "word_default_features": d_n,
                    "word_markaware_acc": f_acc, "word_markaware_macro_f1": f_f1,
                    "word_markaware_features": f_n,
                    "delta_acc": f_acc - d_acc,
                    "char_only_acc": ch, "union_acc": un}
        print(f"{loc:8} {d_acc:>13.4f} {f_acc:>16.4f} {(f_acc-d_acc)*100:>+7.1f} "
              f"{ch:>10.4f} {un:>8.4f} {d_n:>10,} {f_n:>10,}")
    json.dump(out, open(f"{S}/massive_tokenfix.json", "w"), indent=1)
    print(f"\n(default-pattern refits reproduce massive.json exactly, which is "
          f"also a determinism check)")
    print(f"wrote {S}/massive_tokenfix.json")


if __name__ == "__main__":
    main()
