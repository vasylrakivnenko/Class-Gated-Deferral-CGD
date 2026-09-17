#!/usr/bin/env python3
"""Score the filled-in act_audit_sample.md.

Estimates the occurrence-weighted per-sentence accuracy of
`compile.act_labeler: rules_plus_embed` over the bank, with a 95% CI.

Design (this is the D-6 fix): within each of 4 strata, 30 sentence OCCURRENCES
were drawn i.i.d. with probability proportional to template count. Repeat draws
are kept as a `draws` multiplicity instead of being deduped away, so the
per-stratum mean is an unbiased estimate of that stratum's occurrence-weighted
accuracy and its variance is the ordinary binomial one on n=30 draws.

Overall  p = sum_s W_s * p_s ,  Var = sum_s W_s^2 * p_s(1-p_s)/n_s.

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

def run(ambiguous_is_wrong: bool) -> tuple[float, float, dict]:
    p, var, per = 0.0, 0.0, {}
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
        per[s] = (ps, den, meta["weight"])
        p += meta["weight"] * ps
        var += meta["weight"] ** 2 * ps * (1 - ps) / den
    return p, math.sqrt(var), per

for label, flag in (("ambiguous counted WRONG (lower bound)", True),
                    ("ambiguous counted RIGHT (upper bound)", False)):
    p, se, per = run(flag)
    print(f"\n{label}")
    print(f"  act-label accuracy c = {p:.3f}  95% CI [{max(0,p-1.96*se):.3f}, {min(1,p+1.96*se):.3f}]")
    for s in sorted(per):
        ps, den, w = per[s]
        print(f"    {s}  w={w:.3f}  n={den:>3}  p={ps:.3f}   {key['strata'][s]['name']}")

wrong = [(rows[n]["stratum"], rows[n]["act"], filled[n][1], rows[n]["text"])
         for n in rows if not rows[n]["decoy"] and filled[n][0] == "n"]
if wrong:
    print(f"\nconfusions found ({len(wrong)} distinct sentences):")
    for s, was, should, txt in sorted(wrong):
        print(f"  [{s}] {was} -> {should or '?'}   {txt[:70]}")

p, se, _ = run(True)
print("\n---")
print("Attach to any H5 number `a` (agreement with these gold labels):")
print(f"  H5 gold per-sentence act labels are correct at c = {p:.3f} (95% CI +/-{1.96*se:.3f}).")
print("  Mean 1.260 sentences per retrieve turn, so the exact-skeleton gold is correct at")
print("  roughly c^1.26; H5's true correctness lies in [a + c_skel - 1, min(a, c_skel)].")
print("  Report H5 beside the label-blind constant 0.2556 (always predict S0000=[ASK]).")
