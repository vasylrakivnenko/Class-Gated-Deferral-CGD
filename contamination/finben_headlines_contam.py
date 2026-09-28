"""FinBen Headlines contamination audit: 71.5% of the test set is in train.

WHAT IS CLAIMED. SCOPE.md says "FinBen headlines: 71.5% of test rows have a
prompt that appears in train." RESEARCH_BRIEF.md raises it as a reportable
dataset finding. This file is the evidence.

WHY IT MATTERS. `flare-headlines` is nine stacked binary sub-tasks over gold
commodity headlines. 1,632 of each sub-task's 2,283 test rows carry a headline
that also appears verbatim in train, at 99.7-100% label agreement -- so 71.5%
of this benchmark's test set is answerable by lookup. Any score on the full
test split is therefore part memorisation, which is why this project reports a
leak-free score on the 651 remaining rows per sub-task alongside it. It also
explains the fine-tuned model the FinBen paper reports at 0.97: it was trained
on a split containing 71.5% of the test headlines.

TWO STEPS, IN ORDER. The sub-task a row belongs to is not a column in train --
`label_type` is populated only in test. It has to be recovered from the
instruction text, and a hand-written sub-task list is exactly the kind of
shortcut that produced this project's `diversity_1-6` feature leak. So the
vocabulary is read from the dataset's OWN `label_type` column, and step 1
proves the recovery is sound (it must reproduce `label_type` on all 20,547
test rows, with no row matching zero or multiple sub-tasks) before step 2 is
allowed to quote a leak rate per sub-task.

PROVENANCE. Recovered verbatim from `headlines_verify.py` and
`headlines_leak.py`, written 2026-09-14 in a /tmp scratchpad destroyed by an
OOM reboot on 2026-09-16 and restored here from the session transcript. The
measurement logic is unchanged; the pin, fingerprints, JSON artifact and the
assertion against the published numbers are new.

    .venv/bin/python contamination/finben_headlines_contam.py
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import _provenance as prov  # noqa: E402

REPO = "TheFinAI/flare-headlines"
# Pinned. `main` at the time of the original audit and of this restoration.
REVISION = "39e4f9ba3515a7cf464ed07ece80a4f7f4189134"

# The published claim, asserted on every run.
EXPECTED = {
    "derivation": {
        "n_test_rows": 20547,
        "test_rows_agreeing_with_label_type": 20547,
        "test_rows_matching_no_subtask": 0,
        "test_rows_matching_multiple_subtasks": 0,
        "test_rows_with_unparsable_headline": 0,
        "train_rows_with_unparsable_headline": 0,
    },
    "leakage": {
        "total_test_rows": 20547,
        "total_leaked_rows": 14688,
        "per_subtask": {
            "Asset Comparision":  {"n_train": 7988, "n_test": 2283, "leaked": 1632, "label_agreement_pct": 99.9},
            "Direction Constant": {"n_train": 7988, "n_test": 2283, "leaked": 1632, "label_agreement_pct": 99.8},
            "Direction Down":     {"n_train": 7988, "n_test": 2283, "leaked": 1632, "label_agreement_pct": 99.8},
            "Direction Up":       {"n_train": 7988, "n_test": 2283, "leaked": 1632, "label_agreement_pct": 99.7},
            "FutureNews":         {"n_train": 7988, "n_test": 2283, "leaked": 1632, "label_agreement_pct": 100.0},
            "FuturePrice":        {"n_train": 7988, "n_test": 2283, "leaked": 1632, "label_agreement_pct": 100.0},
            "PastNews":           {"n_train": 7988, "n_test": 2283, "leaked": 1632, "label_agreement_pct": 100.0},
            "PastPrice":          {"n_train": 7988, "n_test": 2283, "leaked": 1632, "label_agreement_pct": 100.0},
            "Price or Not":       {"n_train": 7988, "n_test": 2283, "leaked": 1632, "label_agreement_pct": 100.0},
        },
    },
    "within_train_duplicates": {
        "subtask": "Direction Up",
        "headlines_appearing_more_than_once": 159,
        "headlines_with_conflicting_labels": 8,
    },
}

_TEXT = re.compile(r"\nText:\s*(.*?)\s*\nAnswer:\s*$", re.S)


def split_q(q: str):
    """Split a flare prompt into (instruction, headline)."""
    m = _TEXT.search(q)
    return (q[:m.start()].strip(), m.group(1)) if m else (q, None)


def derive(instr: str, vocab):
    """Which sub-task does this instruction name? Longest match wins.

    Longest-match matters: "Price or Not" contains no other name, but a naive
    substring scan over an instruction mentioning several would pick the wrong
    one. Step 1 checks that exactly one name matches every row.
    """
    hits = [v for v in vocab if v in instr]
    return (max(hits, key=len) if hits else None), len(hits)


def derivation_check(ds, vocab) -> dict:
    """Step 1: prove the recovered sub-task labels are the dataset's own."""
    stats = {}
    for sp in ("train", "test"):
        nohead = none = multi = 0
        got = Counter()
        for q in ds[sp]["query"]:
            i, h = split_q(q)
            if h is None:
                nohead += 1
            v, n = derive(i, vocab)
            none += n == 0
            multi += n > 1
            got[v] += 1
        stats[sp] = {"rows": len(ds[sp]), "unparsable_headline": nohead,
                     "matched_no_subtask": none, "matched_multiple_subtasks": multi,
                     "derived_counts": dict(sorted(got.items(), key=lambda kv: str(kv[0])))}
        print(f"\n{sp}: rows={len(ds[sp])} unparsed_headline={nohead} "
              f"no_subtask={none} multi_hit={multi}")
        print("  derived:", stats[sp]["derived_counts"])

    agree = sum(1 for q, lt in zip(ds["test"]["query"], ds["test"]["label_type"])
                if derive(split_q(q)[0], vocab)[0] == lt)
    n_test = len(ds["test"])
    print(f"\ntest derivation agrees with the dataset's own label_type column: "
          f"{agree}/{n_test}")

    tr_h = {split_q(q)[1] for q in ds["train"]["query"]}
    te_h = {split_q(q)[1] for q in ds["test"]["query"]}
    print(f"unique headlines: train={len(tr_h):,} test={len(te_h):,} "
          f"overlap={len(tr_h & te_h):,}")

    return {"vocabulary_from_label_type": list(vocab),
            "n_test_rows": n_test,
            "test_rows_agreeing_with_label_type": agree,
            "test_rows_matching_no_subtask": stats["test"]["matched_no_subtask"],
            "test_rows_matching_multiple_subtasks": stats["test"]["matched_multiple_subtasks"],
            "test_rows_with_unparsable_headline": stats["test"]["unparsable_headline"],
            "train_rows_with_unparsable_headline": stats["train"]["unparsable_headline"],
            "unique_headlines_train": len(tr_h),
            "unique_headlines_test": len(te_h),
            "unique_headlines_shared": len(tr_h & te_h),
            "per_split": stats}


def leakage(ds, vocab) -> dict:
    """Step 2: per sub-task, how many test headlines appear verbatim in train?"""
    rows = defaultdict(list)
    for sp in ("train", "test"):
        for q, a in zip(ds[sp]["query"], ds[sp]["answer"]):
            i, h = split_q(q)
            rows[(sp, derive(i, vocab)[0])].append((h, a))

    print(f"\n{'sub-task':20} {'n_tr':>6} {'n_te':>6} {'te_in_tr':>9} "
          f"{'%leak':>7} {'lbl_agree':>10}")
    per, tot_leak, tot = {}, 0, 0
    for st in vocab:
        tr = dict(rows[("train", st)])
        te = rows[("test", st)]
        inside = [(h, a) for h, a in te if h in tr]
        agree = sum(1 for h, a in inside if tr[h] == a)
        tot_leak += len(inside)
        tot += len(te)
        per[st] = {"n_train": len(rows[("train", st)]), "n_test": len(te),
                   "leaked": len(inside), "leak_free": len(te) - len(inside),
                   "leak_share": len(inside) / len(te),
                   "label_agreement_on_leaked": agree,
                   "label_agreement_pct": round(100 * agree / max(len(inside), 1), 1)}
        print(f"{st:20} {per[st]['n_train']:>6} {len(te):>6} {len(inside):>9} "
              f"{per[st]['leak_share']:>6.1%} {per[st]['label_agreement_pct']:>9.1f}%")
    print(f"{'TOTAL':20} {'':>6} {tot:>6} {tot_leak:>9} {tot_leak/tot:>6.1%}")

    # Supplementary, not part of the published claim: a model that also sees
    # the validation split has an even larger lookup table available.
    valid_h = {split_q(q)[1] for q in ds["valid"]["query"]} if "valid" in ds else set()
    train_h = {split_q(q)[1] for q in ds["train"]["query"]}
    te_pairs = [(split_q(q)[1], derive(split_q(q)[0], vocab)[0])
                for q in ds["test"]["query"]]
    in_tr_or_va = sum(1 for h, _ in te_pairs if h in train_h or h in valid_h)
    print(f"\nsupplementary: test rows whose headline is in train OR valid: "
          f"{in_tr_or_va:,}/{len(te_pairs):,} = {in_tr_or_va/len(te_pairs):.1%}")

    return {"per_subtask": per,
            "total_test_rows": tot,
            "total_leaked_rows": tot_leak,
            "total_leak_share": tot_leak / tot,
            "leak_free_rows_per_subtask": per[vocab[0]]["leak_free"],
            "supplementary_test_rows_in_train_or_valid": in_tr_or_va,
            "supplementary_test_rows_in_train_or_valid_share": in_tr_or_va / len(te_pairs)}


def within_train_duplicates(ds, vocab, subtask: str = "Direction Up") -> dict:
    """Are there duplicate headlines inside train, and do they self-contradict?"""
    tr = [(split_q(q)[1], a) for q, a in zip(ds["train"]["query"], ds["train"]["answer"])
          if derive(split_q(q)[0], vocab)[0] == subtask]
    c = Counter(h for h, _ in tr)
    d = defaultdict(set)
    for h, a in tr:
        d[h].add(a)
    dup = sum(1 for v in c.values() if v > 1)
    conflict = sum(1 for v in d.values() if len(v) > 1)
    print(f"\ntrain dup headlines ({subtask}): {dup} headlines appear >1x")
    print(f"train headlines with CONFLICTING labels ({subtask}): {conflict}")
    return {"subtask": subtask,
            "headlines_appearing_more_than_once": dup,
            "headlines_with_conflicting_labels": conflict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--revision", default=REVISION,
                    help="HuggingFace dataset revision to audit (default: the pin)")
    ap.add_argument("--no-verify", action="store_true",
                    help="measure without asserting against the published claim")
    args = ap.parse_args()

    t0 = time.time()
    with prov.Tee(f"{prov.RESULTS}/finben_headlines_contam.log"):
        from datasets import load_dataset

        pin = prov.dataset_pin(REPO, args.revision)
        env = prov.environment()
        print(f"dataset : {REPO} @ {args.revision}")
        print(f"files   : {len(pin['snapshot_files'])} in snapshot")
        for rel, meta in pin["snapshot_files"].items():
            print(f"          {meta['sha256'][:16]}  {meta['bytes']:>12,}  {rel}")
        print(f"env     : python {env['python']}, datasets "
              f"{env['packages']['datasets']}, commit {env['git_commit'][:8]}"
              f"{' (dirty)' if env['git_dirty'] else ''}")

        ds = load_dataset(REPO, revision=args.revision)
        fp = {sp: prov.content_fingerprint({"query": ds[sp]["query"],
                                            "answer": ds[sp]["answer"]})
              for sp in sorted(ds)}
        print("\ncontent fingerprints (query+answer, row order):")
        for sp, h in fp.items():
            print(f"          {h[:32]}  {sp} ({len(ds[sp]):,} rows)")

        # The sub-task vocabulary comes from the dataset's own label_type
        # column, never from a hand-written list.
        vocab = sorted({v for v in ds["test"]["label_type"] if v})
        print(f"\nsub-task vocabulary, read from the dataset's label_type column "
              f"({len(vocab)}): {vocab}")

        derivation = derivation_check(ds, vocab)
        if derivation["test_rows_agreeing_with_label_type"] != derivation["n_test_rows"]:
            print("\n!! sub-task recovery does not reproduce label_type on every test\n"
                  "   row. The per-sub-task leak rates below would be unsound, so\n"
                  "   they are not reported.")
            return prov.finish("finben_headlines_contam",
                               {"audit": "finben_headlines_contamination",
                                "dataset": pin, "environment": env,
                                "derivation": derivation},
                               ["derivation.test_rows_agreeing_with_label_type"], t0)

        measured = {"derivation": derivation,
                    "leakage": leakage(ds, vocab),
                    "within_train_duplicates": within_train_duplicates(ds, vocab)}
        mismatches = [] if args.no_verify else prov.verify(measured, EXPECTED)

        payload = {"audit": "finben_headlines_contamination",
                   "claim_backed": [
                       "SCOPE.md: FinBen headlines: 71.5% of test rows have a prompt that appears in train",
                       "CASCADE_DIAL.md: FinBen at 71.5%",
                       "RESEARCH_BRIEF.md: train/test prompt overlap in FinBen (we measure 71.5% of test rows)",
                   ],
                   "dataset": pin,
                   "content_fingerprints": fp,
                   "environment": env,
                   "verified_against_published_claim": not args.no_verify,
                   **measured}
        return prov.finish("finben_headlines_contam", payload, mismatches, t0)


if __name__ == "__main__":
    raise SystemExit(main())
