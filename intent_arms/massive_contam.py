"""MASSIVE contamination / id-alignment audit. Runs before any modelling.

Why this file exists at all: MASSIVE's 51 locales are PARALLEL TRANSLATIONS of
one utterance pool, so the same underlying example carries the same `id` in
every language. Pooling locales and splitting at random would put a Spanish
utterance in train and its German twin in test, and a multilingual encoder
reads straight through that. Two of this project's four existing suites were
already burned by this exact class of error (71.5% and 99.8% of test rows
sharing a prompt with train), so nothing here is assumed -- every claim below
is a measured figure.

Three questions, each answered with a number:
  1. Is the official partition id-ALIGNED across locales? i.e. is an id that is
     in en-US train in train for every locale that has it? If yes, using the
     official per-locale split is automatically safe against the cross-lingual
     twin leak, and no custom id-holdout is needed for the per-locale runs.
  2. Do all locales share one id set, or do locales differ in coverage?
  3. WITHIN a locale, how much exact / near-duplicate train-test overlap on
     `utt` is there? Reported whatever it turns out to be.

Note on locale count: the parquet branch carries 52 locale directories, one
more than the 51 languages in the paper's Table 8. The extra one is identified
and excluded from published-baseline comparisons rather than silently matched.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "AmazonScience/massive"
REV = "refs/convert/parquet"
S = "/private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad"
OUT = os.path.join(S, "massive_contam.json")
SEED = 0
PARTS = ("train", "validation", "test")


def locales():
    fs = [x for x in list_repo_files(REPO, repo_type="dataset", revision=REV)
          if x.endswith(".parquet") and not x.startswith("all")]
    return sorted({x.split("/")[0] for x in fs})


def path(loc, part):
    return hf_hub_download(REPO, f"{loc}/{part}/0000.parquet",
                           repo_type="dataset", revision=REV)


def read(loc, part, cols=("id", "partition", "intent", "utt")):
    return pq.read_table(path(loc, part), columns=list(cols)).to_pydict()


def main():
    t0 = time.time()
    locs = locales()
    pub = json.load(open(os.path.join(S, "massive_published.json")))
    extra = [l for l in locs if l not in pub]
    print(f"locale dirs on parquet branch: {len(locs)}")
    print(f"locales in paper Table 8: {len(pub)}")
    print(f"present in data but NOT in the paper's table: {extra}")

    # ---- 1 & 2: id -> partition, per locale -------------------------------
    id_part = defaultdict(dict)      # id -> {locale: partition}
    counts = {}
    for loc in locs:
        for part in PARTS:
            d = read(loc, part, cols=("id", "partition"))
            ids, ps = d["id"], d["partition"]
            counts[(loc, part)] = len(ids)
            # The validation/ directory carries partition == "dev" in the column,
            # not "validation" -- measured, not assumed; an earlier assert here
            # fired on af-ZA/validation and that is how we found it.
            want = {"validation": "dev"}.get(part, part)
            assert set(ps) == {want}, (
                f"{loc}/{part}: partition column is {set(ps)}, expected {want}")
            for i in ids:
                id_part[i][loc] = part
        print(f"  {loc}: " + " ".join(f"{p}={counts[(loc,p)]}" for p in PARTS),
              flush=True)

    all_ids = sorted(id_part)
    n_full = sum(1 for i in all_ids if len(id_part[i]) == len(locs))
    misaligned = [i for i in all_ids if len(set(id_part[i].values())) > 1]
    print(f"\ndistinct ids across all locales: {len(all_ids):,}")
    print(f"ids present in all {len(locs)} locales: {n_full:,} "
          f"({n_full/len(all_ids):.4%})")
    print(f"ids whose partition DISAGREES between locales: {len(misaligned):,} "
          f"({len(misaligned)/len(all_ids):.4%})")

    # en-US as the reference: for each of its ids, does every other locale agree?
    en_ids = {i: p for i, p in ((i, id_part[i].get("en-US")) for i in all_ids)
              if p is not None}
    agree = sum(1 for i, p in en_ids.items()
                if all(v == p for v in id_part[i].values()))
    print(f"en-US ids: {len(en_ids):,}; of these, partition agrees in every "
          f"locale that has the id: {agree:,} ({agree/len(en_ids):.4%})")

    per_locale_cov = {l: sum(1 for i in all_ids if l in id_part[i]) for l in locs}
    print(f"\nper-locale id coverage: min {min(per_locale_cov.values()):,} "
          f"({min(per_locale_cov, key=per_locale_cov.get)}), "
          f"max {max(per_locale_cov.values()):,}")

    # ---- 3: within-locale train/test duplicate overlap --------------------
    from sklearn.feature_extraction.text import TfidfVectorizer

    def norm(s):
        return re.sub(r"\s+", " ", s.strip().lower())

    dup = {}
    check_locs = [l for l in locs]
    for loc in check_locs:
        tr = read(loc, "train", cols=("id", "intent", "utt"))
        te = read(loc, "test", cols=("id", "intent", "utt"))
        tr_u, te_u = tr["utt"], te["utt"]
        # id overlap within a locale must be zero by construction; verify.
        id_ov = len(set(tr["id"]) & set(te["id"]))
        trn = {norm(u) for u in tr_u}
        exact = [j for j, u in enumerate(te_u) if norm(u) in trn]
        # label-conflicting exact dupes: same normalised text, different intent
        tr_lbl = defaultdict(set)
        for u, y in zip(tr_u, tr["intent"]):
            tr_lbl[norm(u)].add(y)
        conflict = sum(1 for j in exact
                       if te["intent"][j] not in tr_lbl[norm(te_u[j])])
        # near-duplicates: char_wb 4-gram cosine, max over train, >= 0.90
        V = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4),
                            sublinear_tf=True, min_df=1)
        Xtr = V.fit_transform([norm(u) for u in tr_u])
        Xte = V.transform([norm(u) for u in te_u])
        mx = np.zeros(Xte.shape[0], dtype=np.float32)
        for a in range(0, Xte.shape[0], 512):
            blk = (Xte[a:a + 512] @ Xtr.T).toarray()
            mx[a:a + 512] = blk.max(axis=1)
        near90 = int((mx >= 0.90).sum())
        near95 = int((mx >= 0.95).sum())
        dup[loc] = {"n_train": len(tr_u), "n_test": len(te_u),
                    "train_test_id_overlap": id_ov,
                    "exact_dup_test_rows": len(exact),
                    "exact_dup_share": len(exact) / len(te_u),
                    "exact_dup_label_conflicts": conflict,
                    "near_dup_ge_0.90": near90,
                    "near_dup_ge_0.90_share": near90 / len(te_u),
                    "near_dup_ge_0.95": near95,
                    "near_dup_ge_0.95_share": near95 / len(te_u),
                    "mean_max_sim": float(mx.mean())}
        print(f"  {loc}: exact {len(exact):>4} ({len(exact)/len(te_u):6.2%}) "
              f"conflicts {conflict:>3}  near>=.90 {near90:>4} "
              f"({near90/len(te_u):6.2%})  idov {id_ov}", flush=True)

    res = {"n_locale_dirs": len(locs), "locales": locs,
           "not_in_paper_table": extra,
           "n_distinct_ids": len(all_ids),
           "ids_in_all_locales": n_full,
           "ids_in_all_locales_share": n_full / len(all_ids),
           "ids_partition_disagree": len(misaligned),
           "ids_partition_disagree_share": len(misaligned) / len(all_ids),
           "en_us_ids": len(en_ids),
           "en_us_partition_agrees_everywhere": agree,
           "en_us_partition_agrees_share": agree / len(en_ids),
           "per_locale_id_coverage": per_locale_cov,
           "row_counts": {f"{l}/{p}": counts[(l, p)] for l in locs for p in PARTS},
           "within_locale_dupes": dup}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT} ({time.time()-t0:,.0f}s)")


if __name__ == "__main__":
    main()
