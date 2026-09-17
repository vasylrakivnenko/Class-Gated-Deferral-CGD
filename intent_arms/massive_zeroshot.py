"""Cross-lingual ZERO-SHOT transfer: train on en-US only, test on the other 12.

Why this is worth running and why it is safe. The paper's Table 8 carries
zero-shot columns (train on en-US, test on every other locale) for mT5 and
XLM-R, so this is the one setting on this suite where a published number and a
free arm can be put side by side on the SAME protocol rather than on different
ones. And the leak risk that normally kills cross-lingual transfer is already
measured away: massive_contam.py found the official partition is exactly
id-aligned across all 52 locales (0 ids disagree), so en-US TRAIN ids and
locale-xx TEST ids are disjoint by construction. This script asserts that
rather than trusting it.

The arm is the frozen multilingual embedder + logistic regression, i.e. exactly
the project's cheap arm, with a single classifier trained once on en-US and
applied unchanged to 12 other languages. Nothing is refit per locale -- that is
what makes it zero-shot.

Predicted before running: this should be the one place on MASSIVE where the
free arm can actually beat a published baseline, because published zero-shot
intent accuracy falls a long way on distant locales (XLM-R zero-shot is 44.8 on
ja-JP and 46.6 on sw-KE against 88.3 full-data on en-US) whereas a frozen
sentence embedder is trained for cross-lingual alignment in the first place.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

S = "/private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad"
REPO, REV = "AmazonScience/massive", "refs/convert/parquet"
SEED = 0
MODEL = "intfloat/multilingual-e5-small"
LOCALES = ["en-US", "de-DE", "es-ES", "fr-FR", "ru-RU", "zh-CN", "ja-JP",
           "ko-KR", "ar-SA", "hi-IN", "th-TH", "sw-KE", "am-ET"]


def load(loc, part):
    p = hf_hub_download(REPO, f"{loc}/{part}/0000.parquet",
                        repo_type="dataset", revision=REV)
    t = pq.read_table(p, columns=["id", "utt", "intent"]).to_pydict()
    return t["id"], t["utt"], np.asarray(t["intent"], dtype=np.int64)


def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score

    t0 = time.time()
    st = SentenceTransformer(MODEL)

    def emb(xs):
        return st.encode([f"query: {t}" for t in xs], batch_size=256,
                         normalize_embeddings=True, show_progress_bar=False,
                         convert_to_numpy=True)

    tr_id, tr_x, tr_y = load("en-US", "train")
    tr_ids = set(tr_id)
    Etr = emb(tr_x)
    clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced",
                             random_state=SEED).fit(Etr, tr_y)
    print(f"trained once on en-US train ({len(tr_x):,} rows, "
          f"{time.time()-t0:,.0f}s)\n")

    pub = json.load(open(f"{S}/massive_published.json"))
    sup = json.load(open(f"{S}/massive.json"))["per_locale"]
    out = {}
    print(f"{'locale':8} {'zero-shot':>10} {'macroF1':>8} {'AUROC':>7} "
          f"{'supervised':>11} {'drop':>7} {'pub XLMR zero':>14} {'vs pub':>8}")
    for loc in LOCALES:
        te_id, te_x, te_y = load(loc, "test")
        # the guarantee, asserted not assumed
        assert not (set(te_id) & tr_ids), f"{loc}: test id overlaps en-US train"
        P = clf.predict_proba(emb(te_x))
        pred = clf.classes_[P.argmax(1)]
        conf = P.max(1)
        acc = float((pred == te_y).mean())
        mf1 = float(f1_score(te_y, pred, average="macro", zero_division=0))
        corr = (pred == te_y).astype(int)
        auroc = float(roc_auc_score(corr, conf)) if 0 < corr.sum() < len(corr) else None
        s = sup[loc]["arms"]["frozen_e5"]["accuracy"]
        pz = (pub.get(loc) or {}).get("XLMR_zero")
        pz = pz["acc"] / 100 if pz else None
        out[loc] = {"zeroshot_accuracy": acc, "zeroshot_macro_f1": mf1,
                    "zeroshot_auroc": auroc,
                    "supervised_frozen_accuracy": s,
                    "drop_vs_supervised": acc - s,
                    "published_xlmr_zeroshot": pz,
                    "vs_published_xlmr_zeroshot": (acc - pz) if pz else None,
                    "pred": [int(v) for v in pred],
                    "conf": [float(v) for v in conf],
                    "gold": [int(v) for v in te_y]}
        print(f"{loc:8} {acc:>10.4f} {mf1:>8.4f} "
              f"{'-' if auroc is None else f'{auroc:>7.3f}'} {s:>11.4f} "
              f"{(acc-s)*100:>+7.1f} "
              f"{'-' if pz is None else f'{pz:>14.4f}'} "
              f"{'-' if pz is None else f'{(acc-pz)*100:>+8.1f}'}")

    non_en = [l for l in LOCALES if l != "en-US"]
    wins = [l for l in non_en if out[l]["vs_published_xlmr_zeroshot"]
            and out[l]["vs_published_xlmr_zeroshot"] > 0]
    print(f"\nzero-shot: beats published XLM-R zero-shot on {len(wins)} of "
          f"{len(non_en)} non-English locales: {', '.join(wins)}")
    json.dump(out, open(f"{S}/massive_zeroshot.json", "w"))
    print(f"wrote {S}/massive_zeroshot.json ({time.time()-t0:,.0f}s)")


if __name__ == "__main__":
    main()
