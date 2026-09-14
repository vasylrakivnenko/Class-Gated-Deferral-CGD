"""Paired comparison of two cheap candidates from their FILE A per-item dumps.

Same --seed => identical stratified fold partitions => every (repeat, item_idx)
pairs exactly. Reports per repeat and mean: accuracy of each, McNemar mid-p on
the discordant items, non-inferiority (delta=1pt) of the challenger vs the
reference, and keep rates. Refuses to compare if text_sha256 or gold disagree.

    python phase2/compare_cheap.py --task T --ref-file A.csv --new-file B.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT)); sys.path.insert(0, "/Users/vasyl/zadumai/src")
from phase2.analyze import FILE_A_COLS, _ni, _paired  # noqa: E402


def load(path):
    a = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(a.columns) == FILE_A_COLS, path
    a["item_idx"] = a["item_idx"].astype(int); a["repeat"] = a["repeat"].astype(int); a["kept"] = a["kept"].astype(int)
    return a.sort_values(["repeat", "item_idx"]).reset_index(drop=True)


def compare(ref_path, new_path, n_boot=2000, seed=0, delta=0.01) -> dict:
    r, n = load(ref_path), load(new_path)
    assert (r["task"].iloc[0] == n["task"].iloc[0]), "different tasks"
    # Same seed => identical fold partition for a given repeat index, so a
    # 1-repeat run pairs exactly with repeat 0 of a 3-repeat run. Intersect.
    common = sorted(set(r["repeat"]) & set(n["repeat"]))
    assert common, "no common repeat indices"
    if len(common) < max(r["repeat"].nunique(), n["repeat"].nunique()):
        print(f"  (pairing on common repeats {common} only)")
    r = r[r["repeat"].isin(common)].reset_index(drop=True)
    n = n[n["repeat"].isin(common)].reset_index(drop=True)
    assert len(r) == len(n) and (r["repeat"].to_numpy() == n["repeat"].to_numpy()).all() \
        and (r["item_idx"].to_numpy() == n["item_idx"].to_numpy()).all(), "row alignment differs"
    assert (r["text_sha256"].to_numpy() == n["text_sha256"].to_numpy()).all(), "text_sha256 mismatch"
    assert (r["gold"].to_numpy() == n["gold"].to_numpy()).all(), "gold mismatch"
    out = {"task": r["task"].iloc[0], "ref": r["candidate"].iloc[0], "new": n["candidate"].iloc[0], "per_repeat": {}}
    for rep in sorted(r["repeat"].unique()):
        rr, nn = r[r["repeat"] == rep], n[n["repeat"] == rep]
        cr = (rr["pred"].to_numpy() == rr["gold"].to_numpy()); cn = (nn["pred"].to_numpy() == nn["gold"].to_numpy())
        out["per_repeat"][int(rep)] = {
            "ref_acc": float(cr.mean()), "new_acc": float(cn.mean()),
            "ref_keep": float(rr["kept"].mean()), "new_keep": float(nn["kept"].mean()),
            "mcnemar_new_vs_ref": _paired(cn, cr, n_boot, seed + int(rep)),
            "ni_new_vs_ref": _ni(cn, cr, delta, n_boot, seed + int(rep)),
        }
    reps = list(out["per_repeat"].values())
    out["mean"] = {
        "ref_acc": float(np.mean([x["ref_acc"] for x in reps])), "new_acc": float(np.mean([x["new_acc"] for x in reps])),
        "ref_keep": float(np.mean([x["ref_keep"] for x in reps])), "new_keep": float(np.mean([x["new_keep"] for x in reps])),
        "diff": float(np.mean([x["mcnemar_new_vs_ref"]["diff"] for x in reps])),
        "diff_ci_lo": float(np.mean([x["mcnemar_new_vs_ref"]["diff_ci_lo"] for x in reps])),
        "diff_ci_hi": float(np.mean([x["mcnemar_new_vs_ref"]["diff_ci_hi"] for x in reps])),
        "p_mean": float(np.mean([x["mcnemar_new_vs_ref"]["p_value"] for x in reps])),
        "ni_passes": float(np.mean([float(x["ni_new_vs_ref"]["passes"]) for x in reps])),
        "n_discordant": float(np.mean([x["mcnemar_new_vs_ref"]["n_discordant"] for x in reps])),
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--ref-file", required=True); ap.add_argument("--new-file", required=True)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    o = compare(a.ref_file, a.new_file, a.n_boot, a.seed); m = o["mean"]
    print(f"{o['task']}: {o['new']} vs {o['ref']} (paired, {len(o['per_repeat'])} repeats)")
    print(f"  acc  new {100*m['new_acc']:.1f}%  ref {100*m['ref_acc']:.1f}%   diff {100*m['diff']:+.1f}pp "
          f"CI[{100*m['diff_ci_lo']:+.1f},{100*m['diff_ci_hi']:+.1f}]  McNemar p={m['p_mean']:.3f}  "
          f"discordant={m['n_discordant']:.0f}  NI(new>=ref-1pt) passes={m['ni_passes']:.2f}")
    print(f"  keep new {100*m['new_keep']:.1f}%  ref {100*m['ref_keep']:.1f}%")
    for rep, x in o["per_repeat"].items():
        print(f"    repeat {rep}: new {100*x['new_acc']:.1f} ref {100*x['ref_acc']:.1f} p={x['mcnemar_new_vs_ref']['p_value']:.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
