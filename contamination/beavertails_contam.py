"""BeaverTails contamination audit: is the official split prompt-disjoint? (No.)

WHAT IS CLAIMED. SCOPE.md says "BeaverTails: 99.8% of test rows share a prompt
with train (distinct responses, so pairs do not repeat)", and that "across
texts appearing more than once, 27.7% disagree on is_safe" and 23.9% on
non_violent_unethical_behavior. This file is the evidence for all three.

WHY IT MATTERS. BeaverTails pairs each red-team PROMPT with responses from
several different models and labels each (prompt, response) PAIR. The official
splits divide those pairs, not the prompts, so the same prompt -- "how do I
hotwire a car" -- appears in train and in test with different responses. The
label is a judgement about the RESPONSE, but the prompt has to be in the model
input for that judgement to be well posed. A classifier can therefore learn
prompt -> category (hotwiring implies theft, near enough regardless of what the
response says) and score well on the official split without ever reading the
response. Any number on the official split is an upper bound contaminated by
that shortcut, which is why this project reports a prompt-disjoint split
instead.

Both the 330k and 30k splits are checked, because the published baselines
(LoRA-Guard, WildGuard, GuardReasoner, MD-Judge) all evaluate on 30k.

The duplicate check is the second half. Dedup collapses 330k_train to 99,734
unique prompt+response texts, every one of them appearing at least three
times. If those copies disagree on their labels then (a) dedup keeps an
arbitrary one of several contradictory labels and (b) there is a label-noise
ceiling no classifier can pass. Both have to be known before any F1 on this
suite can be read.

PROVENANCE. Recovered verbatim from `bt_leakcheck.py` and `bt_dupcheck.py`,
written 2026-09-14 in a /tmp scratchpad destroyed by an OOM reboot on
2026-09-16 and restored here from the session transcript. The measurement
logic is unchanged; the pin, fingerprints, JSON artifact and the assertion
against the published numbers are new.

    .venv/bin/python contamination/beavertails_contam.py
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import _provenance as prov  # noqa: E402

REPO = "PKU-Alignment/BeaverTails"
# Pinned. `main` at the time of the original audit and of this restoration.
REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"
SPLIT_PAIRS = (("330k_train", "330k_test"), ("30k_train", "30k_test"))

# The published claim, asserted on every run. Integer counts only: the
# percentages in the docs are derived from these, so pinning the counts pins
# the percentages without any float tolerance.
EXPECTED = {
    "leakage": {
        "330k": {"n_test_rows": 33396,
                 "unique_train_prompts": 16188,
                 "unique_test_prompts": 7766,
                 "shared_prompts": 7759,
                 "test_rows_whose_prompt_is_in_train": 33324,
                 "test_rows_whose_pair_is_in_train": 0},
        "30k": {"n_test_rows": 3021,
                "unique_train_prompts": 7766,
                "unique_test_prompts": 2526,
                "shared_prompts": 2518,
                "test_rows_whose_prompt_is_in_train": 2997,
                "test_rows_whose_pair_is_in_train": 12},
    },
    "duplicates": {
        "unique_texts": 99734,
        "texts_appearing_more_than_once": 99734,
        "copies_per_text": {"3": 99360, "4": 54, "6": 242, "9": 51,
                            "12": 20, "15": 3, "18": 3, "21": 1},
        "copies_agree_on_full_category_vector": 46953,
        "copies_agree_on_is_safe": 72142,
        # disagreement rate among repeated texts, percent, as published
        "disagree_pct": {
            "animal_abuse": 1.0,
            "child_abuse": 0.7,
            "controversial_topics,politics": 5.8,
            "discrimination,stereotype,injustice": 9.0,
            "drug_abuse,weapons,banned_substance": 5.2,
            "financial_crime,property_crime,theft": 8.1,
            "hate_speech,offensive_language": 10.7,
            "misinformation_regarding_ethics,laws_and_safety": 3.4,
            "non_violent_unethical_behavior": 23.9,
            "privacy_violation": 3.8,
            "self_harm": 0.8,
            "sexually_explicit,adult_content": 3.1,
            "terrorism,organized_crime": 1.9,
            "violence,aiding_and_abetting,incitement": 16.2,
            "is_safe": 27.7,
        },
    },
}


def leakage(ds) -> dict:
    """Exact-string prompt overlap between each official train and test split.

    Exact match, not fuzzy: the claim is that literally the same prompt string
    recurs, which is the strongest and least arguable form of the finding.
    """
    out = {}
    for tr_name, te_name in SPLIT_PAIRS:
        key = tr_name.split("_")[0]
        tr_prompt, tr_resp = ds[tr_name]["prompt"], ds[tr_name]["response"]
        te_prompt, te_resp = ds[te_name]["prompt"], ds[te_name]["response"]

        ptr = set(tr_prompt)
        pte = set(te_prompt)
        ov_rows = sum(1 for p in te_prompt if p in ptr)
        pairs_tr = set(zip(tr_prompt, tr_resp))
        pair_ov = sum(1 for pr in zip(te_prompt, te_resp) if pr in pairs_tr)

        n = len(te_prompt)
        out[key] = {
            "n_train_rows": len(tr_prompt),
            "n_test_rows": n,
            "unique_train_prompts": len(ptr),
            "unique_test_prompts": len(pte),
            "shared_prompts": len(ptr & pte),
            "test_rows_whose_prompt_is_in_train": ov_rows,
            "test_rows_whose_prompt_is_in_train_share": ov_rows / n,
            "test_rows_whose_pair_is_in_train": pair_ov,
            "test_rows_whose_pair_is_in_train_share": pair_ov / n,
            "responses_per_train_prompt": len(pairs_tr) / max(len(ptr), 1),
        }
        r = out[key]
        print(f"\n=== {tr_name} / {te_name} ===")
        print(f"  unique prompts: train {len(ptr):,}  test {len(pte):,}  "
              f"shared {len(ptr & pte):,}")
        print(f"  TEST ROWS whose prompt appears in train: {ov_rows:,}/{n:,} "
              f"= {r['test_rows_whose_prompt_is_in_train_share']:.1%}")
        print(f"  TEST ROWS whose (prompt,response) PAIR appears in train: "
              f"{pair_ov:,}/{n:,} = {r['test_rows_whose_pair_is_in_train_share']:.1%}")
        print(f"  responses per prompt: train {r['responses_per_train_prompt']:.1f}")
    return out


def duplicates(ds) -> dict:
    """Do the repeated prompt+response texts in 330k_train agree on their labels?"""
    cats = sorted(ds["330k_train"][0]["category"].keys())
    by_text = defaultdict(list)
    for p, r, cat, safe in zip(ds["330k_train"]["prompt"],
                               ds["330k_train"]["response"],
                               ds["330k_train"]["category"],
                               ds["330k_train"]["is_safe"]):
        t = f"{(p or '').strip()}\n\n{(r or '').strip()}"
        by_text[t].append((tuple(bool(cat[c]) for c in cats), bool(safe)))

    multi = {t: v for t, v in by_text.items() if len(v) > 1}
    copies = {str(k): v for k, v in sorted(Counter(len(v) for v in by_text.values()).items())}
    print(f"\nunique texts: {len(by_text):,}; texts appearing >1x: {len(multi):,}")
    print("copies per text:", copies)

    full_agree = sum(1 for v in multi.values() if len({x[0] for x in v}) == 1)
    safe_agree = sum(1 for v in multi.values() if len({x[1] for x in v}) == 1)
    print(f"\nof the {len(multi):,} repeated texts:")
    print(f"  all copies identical on the full {len(cats)}-vector: {full_agree:,} "
          f"({full_agree/len(multi):.1%})")
    print(f"  all copies identical on is_safe:            {safe_agree:,} "
          f"({safe_agree/len(multi):.1%})")

    # Per-label: how often do copies of one text disagree, and how much of the
    # label mass survives a majority vote? maxvote acc is the ceiling any
    # classifier faces on repeated texts even with perfect knowledge.
    print(f"\n{'category':50} {'disagree%':>10} {'maxvote acc':>12}")
    disagree_pct, maxvote = {}, {}
    for i, c in enumerate(cats + ["is_safe"]):
        dis, frac = 0, []
        for v in multi.values():
            vals = [x[0][i] for x in v] if c != "is_safe" else [x[1] for x in v]
            if len(set(vals)) > 1:
                dis += 1
            frac.append(max(vals.count(True), vals.count(False)) / len(vals))
        disagree_pct[c] = round(100 * dis / len(multi), 1)
        maxvote[c] = round(100 * sum(frac) / len(frac), 1)
        print(f"{c:50} {disagree_pct[c]:>9.1f}% {maxvote[c]:>11.1f}%")

    return {"categories": cats,
            "unique_texts": len(by_text),
            "texts_appearing_more_than_once": len(multi),
            "copies_per_text": copies,
            "copies_agree_on_full_category_vector": full_agree,
            "copies_agree_on_is_safe": safe_agree,
            "disagree_pct": disagree_pct,
            "majority_vote_accuracy_pct": maxvote}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--revision", default=REVISION,
                    help="HuggingFace dataset revision to audit (default: the pin)")
    ap.add_argument("--no-verify", action="store_true",
                    help="measure without asserting against the published claim")
    args = ap.parse_args()

    t0 = time.time()
    with prov.Tee(f"{prov.RESULTS}/beavertails_contam.log"):
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
        fp = {sp: prov.content_fingerprint(
                  {"prompt": ds[sp]["prompt"], "response": ds[sp]["response"]})
              for sp in sorted({s for pair in SPLIT_PAIRS for s in pair})}
        print("\ncontent fingerprints (prompt+response, row order):")
        for sp, h in fp.items():
            print(f"          {h[:32]}  {sp} ({len(ds[sp]):,} rows)")

        measured = {"leakage": leakage(ds), "duplicates": duplicates(ds)}
        mismatches = [] if args.no_verify else prov.verify(measured, EXPECTED)

        payload = {"audit": "beavertails_contamination",
                   "claim_backed": [
                       "SCOPE.md: BeaverTails: 99.8% of test rows share a prompt with train",
                       "SCOPE.md: 27.7% disagree on is_safe, 23.9% on non_violent_unethical_behavior",
                       "CASCADE_DIAL.md: BeaverTails at 99.8%",
                   ],
                   "dataset": pin,
                   "content_fingerprints": fp,
                   "environment": env,
                   "verified_against_published_claim": not args.no_verify,
                   **measured}
        return prov.finish("beavertails_contam", payload, mismatches, t0)


if __name__ == "__main__":
    raise SystemExit(main())
