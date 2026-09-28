#!/usr/bin/env python3
"""Score the filled-in act_audit_sample.md.

Estimates the occurrence-weighted per-sentence accuracy of
`compile.act_labeler: rules_plus_embed` over the bank, with a 95% CI.

Design (this is the D-6 fix): within each of 4 strata, 30 sentence OCCURRENCES
were drawn i.i.d. with probability proportional to template count. Repeat draws
are kept as a `draws` multiplicity instead of being deduped away, so the
per-stratum mean is an unbiased estimate of that stratum's occurrence-weighted
accuracy.

Overall  p = sum_s W_s * p_s.  The interval is NOT Wald: on n=30 per stratum a
Wald SE is identically zero at 30/30 and puts its upper limit outside [0,1] at
29/30, so a perfect stratum would print a zero-width 95% interval on 120 draws
(a stratified nonparametric bootstrap degenerates in exactly the same way).
Each stratum gets a Wilson score interval instead (30/30 -> [0.886, 1.000]) and
the overall interval is the weight-combined [sum_s W_s*lo_s, sum_s W_s*hi_s].

Run:  python3 outputs/act_audit/score_act_audit.py
"""
import json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
key = json.load(open(os.path.join(HERE, "act_audit_key.json")))
rows = {r["n"]: r for r in key["rows"]}

filled = {}
row_re = re.compile(r"^\|\s*(\d{3})\s*\|(.*)\|\s*([A-Z]+)\s*\|([^|]*)\|([^|]*)\|\s*$")
for line in open(os.path.join(HERE, "act_audit_sample.md")):
    m = row_re.match(line.rstrip("\n"))
    if m:
        filled[int(m.group(1))] = (m.group(4).strip().lower(), m.group(5).strip().upper())

missing = [n for n in rows if n not in filled or filled[n][0] == ""]
if missing:
    sys.exit(f"{len(missing)} of {len(rows)} rows have no verdict: {missing[:12]}...\n"
             f"Fill every row. A partly-filled sheet has no defined sampling distribution.")
bad = {n: v for n, (v, _) in filled.items() if v not in ("y", "n", "?")}
if bad:
    sys.exit(f"unrecognised verdicts (use y / n / ?): {bad}")

# -- acquiescence control: the decoy labels are deliberately wrong ---------- #
decoys = [n for n, r in rows.items() if r["decoy"]]
caught = [n for n in decoys if filled[n][0] == "n"]
rate = len(caught) / len(decoys)
print(f"acquiescence control: {len(caught)}/{len(decoys)} corrupted labels rejected ({rate:.0%})")
if rate < 0.75:
    sys.exit("REFUSING TO REPORT AN ACCURACY.\n"
             "Fewer than 3 in 4 corrupted labels were caught, so the sheet measures\n"
             "agreement with the printed label, not correctness. Re-review, deciding the\n"
             "act yourself BEFORE reading the label column.")

def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: stays inside [0,1] and stays non-degenerate at
    k == n, where a Wald SE is identically zero (30/30 -> [0.886, 1.000])."""
    ph, d = k / n, 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run(ambiguous_is_wrong: bool) -> tuple[float, float, float, dict]:
    p, lo, hi, per = 0.0, 0.0, 0.0, {}
    for s, meta in key["strata"].items():
        num = den = 0
        for n, r in rows.items():
            if r["decoy"] or r["stratum"] != s:
                continue
            v = filled[n][0]
            den += r["draws"]
            if v == "y" or (v == "?" and not ambiguous_is_wrong):
                num += r["draws"]
        ps = num / den
        lo_s, hi_s = _wilson(num, den)
        per[s] = (ps, den, meta["weight"], lo_s, hi_s)
        p += meta["weight"] * ps
        # combine the per-stratum Wilson limits rather than a Wald SE: den is 30
        # in every stratum, and a stratum at 30/30 contributes zero Wald variance.
        lo += meta["weight"] * lo_s
        hi += meta["weight"] * hi_s
    return p, lo, hi, per

for label, flag in (("ambiguous counted WRONG (lower bound)", True),
                    ("ambiguous counted RIGHT (upper bound)", False)):
    p, lo, hi, per = run(flag)
    print(f"\n{label}")
    print(f"  act-label accuracy c = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
          f"  (weight-combined Wilson, not Wald)")
    for s in sorted(per):
        ps, den, w, lo_s, hi_s = per[s]
        print(f"    {s}  w={w:.3f}  n={den:>3}  p={ps:.3f} [{lo_s:.3f}, {hi_s:.3f}]"
              f"   {key['strata'][s]['name']}")

wrong = [(rows[n]["stratum"], rows[n]["act"], filled[n][1], rows[n]["text"])
         for n in rows if not rows[n]["decoy"] and filled[n][0] == "n"]
if wrong:
    print(f"\nconfusions found ({len(wrong)} distinct sentences):")
    for s, was, should, txt in sorted(wrong):
        print(f"  [{s}] {was} -> {should or '?'}   {txt[:70]}")

p, lo, hi, _ = run(True)
# Figures below are READ from the artifacts, not typed: the bank total and the
# stratum-B weight are already in `key`; the eval-frame gap comes from
# coverage_gap.json (variant A == train._derive_turn_labels, the shipped path).
_cg_path = os.path.normpath(os.path.join(HERE, "..", "probes", "analysis", "coverage_gap.json"))
try:
    _cg = json.load(open(_cg_path))["splits"]
    _va = _cg["test_seen"]["variants"]["A"]
    _npos, _ngold = int(_va["n_positions"]), int(_va["n_positions_with_gold"])
    _flips = (int(_cg["test_seen"]["delex"]["sentences_whose_act_flipped"]),
              int(_cg["dev"]["delex"]["sentences_whose_act_flipped"]))
    _gap = (f"where {(_npos - _ngold) / _npos:.1%} of positions ({_npos - _ngold:,} of {_npos:,} on\n"
            f"  test_seen) have no bank row at all, and coverage_gap.json records {_flips[0]} test_seen /\n"
            f"  {_flips[1]} dev sentences whose act flips between raw and delexicalized input.")
    _tail = f"the {(_npos - _ngold) / _npos:.1%} unbanked eval tail"
except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
    _gap = (f"where a large share of positions have no bank row at all\n"
            f"  (see {_cg_path}; could not read it here: {exc}).")
    _tail = "the unbanked eval tail"
print("\n---")
print("WHAT `c` IS -- and what it is NOT -- before attaching it to an H5 number `a`:")
print(f"  c = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}] is the labeller's occurrence-weighted")
print(f"  accuracy over the {key['total_bank_occurrences']:,} BANKED train occurrences: 65.01% of the 89,623 train")
print("  sentences per outputs/compile/compile_summary.md (the rest were dropped by")
print("  compile.min_template_count=2 and dedup, so they were never eligible to be")
print("  sampled), judged on DELEXICALIZED template text.")
print("  H5's gold comes from a DIFFERENT population: train._derive_turn_labels act-labels")
print(f"  the RAW sentences of the eval split, {_gap}")
print(f"  The bank frame gives the unvalidated embed tier (stratum B) {key['strata']['B']['weight']:.1%} of the occurrence")
print(f"  weight; its share of {_tail} has never been measured. The SIGN and")
print("  SIZE of the resulting bias on the eval population have not been measured, so c is a")
print("  frame-mismatched estimate, not a measurement of the labeller that produced H5's gold.")
print("  Mean 1.260 sentences per retrieve turn (89,623 / 71,133, compile_summary.md), so over")
print("  the BANK frame the exact-skeleton gold is correct at roughly c^1.26.")
print("  [a + c_skel - 1, min(a, c_skel)] is a valid Frechet bound only when `a` and")
print("  `c_skel` are measured over the SAME population,")
print("  which they are not -- quote it as indicative, never as H5's true-correctness")
print("  interval, and never feed c in as if it were a point estimate for that population.")

# D5's bar is the constant APPLIED to the eval split, not the train prior share
# of S0000 (0.2556, compile_summary.md) -- read it from the published artifact.
_sel = os.path.normpath(os.path.join(HERE, "..", "probes", "response", "select.json"))
try:
    _bar = float(json.load(open(_sel))["headline_test_seen"]["h5"]["constant"]["recall@1"])
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"  Report H5 beside the label-blind constant from {_sel}")
    print(f"  (headline_test_seen.h5.constant.recall@1) -- could not read it here: {exc}")
else:
    print(f"  Report H5 beside the label-blind constant {_bar:.4f} on test_seen (always")
    print("  predict S0000=[ASK]): D5 compares against the constant APPLIED to the eval")
    print("  split, not against 0.2556, which is only S0000's share of the TRAIN prior.")
