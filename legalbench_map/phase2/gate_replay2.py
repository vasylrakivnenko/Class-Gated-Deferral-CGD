"""Gate replay 2: two signals the base classifier does NOT already have.

  committee   escalate where OTHER cheap models disagree with the primary one
              (query-by-committee). Signal = number of disagreeing committee
              members (0..K); variant 'b' adds (1 - p_pred) as a tie-break.
              Threshold chosen on training folds, applied out-of-fold.
  per-class   your original spec: on the training folds, for each class the
              primary model PREDICTS, compare incumbent vs cheap accuracy on
              that subset; escalate the whole predicted class out-of-fold iff
              the incumbent is better by > margin (0 or 2 points).

Both reuse gate_replay's loading, folds, evaluation and criteria. Also
reports, per pair, the diagnostic that decides everything: on the items
where the committee disagrees, who is actually right more often?
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
_HERE = Path(__file__).resolve().parent; _PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT)); sys.path.insert(0, "/Users/vasyl/zadumai/src")
from phase2.gate_replay import (  # noqa: E402
    FILE_A_COLS, CRITERIA, _cascade_correct, _mean_num, _json_safe, evaluate_gate, load_pair, outer_folds,
    threshold_gate_oof,
)

COMMITTEE = ["tfidf_logreg", "embed_small_logreg", "embed_base_logreg", "embed_large_logreg"]


def load_committee(per_item_dirs, task, primary, n, text_sha) -> dict[str, dict[int, np.ndarray]]:
    """{candidate: {repeat: pred array}} for every committee member except the primary."""
    out = {}
    for cand in COMMITTEE:
        if cand == primary:
            continue
        path = next((Path(d) / f"{task}__{cand}.csv" for d in per_item_dirs if (Path(d) / f"{task}__{cand}.csv").exists()), None)
        if path is None:
            continue
        a = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(a.columns) == FILE_A_COLS
        a["item_idx"] = a["item_idx"].astype(int); a["repeat"] = a["repeat"].astype(int)
        reps = {}
        for r, g in a.groupby("repeat"):
            g = g[g["item_idx"] < n].sort_values("item_idx")
            assert g["item_idx"].tolist() == list(range(n))
            assert (g["text_sha256"].to_numpy() == np.asarray(text_sha)).all(), f"{cand}: sha mismatch"
            reps[int(r)] = g["pred"].to_numpy()
        out[cand] = reps
    return out


def per_class_gate_oof(cheap_pred, cheap_correct, inc_correct, folds, margin: float) -> np.ndarray:
    """Escalate a PREDICTED class out-of-fold iff, on the training folds, the
    incumbent beats the cheap model on that predicted class by > margin."""
    esc = np.zeros(len(cheap_pred), dtype=bool)
    for tr, te in folds:
        for c in np.unique(cheap_pred[tr]):
            m = tr[cheap_pred[tr] == c]
            adv = float(inc_correct[m].mean() - cheap_correct[m].mean())
            if adv > margin:
                esc[te[cheap_pred[te] == c]] = True
    return esc


def analyze(pair, committee, *, folds_k, seed, n_boot) -> dict:
    gold, ic, cost1k = pair["gold"], pair["inc_correct"], pair["inc_cost_per_1k"]
    per_repeat, diag = {}, []
    for r, rep in pair["repeats"].items():
        cc, cp, p = rep["cheap_correct"], rep["cheap_pred"], rep["p_pred"]
        folds = outer_folds(gold, folds_k, seed + r)
        members = [committee[c][r] for c in committee if r in committee[c]]
        n_dis = np.sum([m != cp for m in members], axis=0) if members else np.zeros(len(cp))
        gates = {
            "conformal": {"": rep["kept"] == 0},
            "confidence": threshold_gate_oof(p, cc, ic, folds),
            "committee": threshold_gate_oof(-n_dis.astype(float), cc, ic, folds),
            "committee_conf": threshold_gate_oof(-(n_dis + (1.0 - p)), cc, ic, folds),
            "perclass_any": {"": per_class_gate_oof(cp, cc, ic, folds, 0.0)},
            "perclass_2pt": {"": per_class_gate_oof(cp, cc, ic, folds, 0.02)},
            "oracle": {"": ~cc},
        }
        res = {"cheap_accuracy": float(cc.mean()), "incumbent_accuracy": float(ic.mean()), "gates": {}}
        for g, variants in gates.items():
            for crit, esc in variants.items():
                res["gates"][g if not crit else f"{g}_{crit}"] = evaluate_gate(esc, cc, ic, cost1k, n_boot, seed + r)
        # diagnostics: who is right where the committee disagrees / agrees; committee majority vote
        dis = n_dis >= 1
        votes = np.array([cp] + members)  # shape (K+1, n)
        maj = np.array([max(set(col), key=list(col).count) for col in votes.T]) if members else cp
        res["diag"] = {
            "n_committee": len(members),
            "disagree_rate": float(dis.mean()),
            "cheap_acc_on_disagree": float(cc[dis].mean()) if dis.any() else float("nan"),
            "inc_acc_on_disagree": float(ic[dis].mean()) if dis.any() else float("nan"),
            "cheap_acc_on_agree": float(cc[~dis].mean()) if (~dis).any() else float("nan"),
            "inc_acc_on_agree": float(ic[~dis].mean()) if (~dis).any() else float("nan"),
            "majority_vote_accuracy": float((maj == gold).mean()),
        }
        per_repeat[str(r)] = res
    mean = _mean_num(list(per_repeat.values()))
    g = mean["gates"]; gap = g["oracle"]["accuracy"] - g["conformal"]["accuracy"]
    for k in g:
        g[k]["oracle_gap_recovered"] = float((g[k]["accuracy"] - g["conformal"]["accuracy"]) / gap) if gap > 1e-12 else float("nan")
    return {"task": pair["task"], "model_key": pair["model_key"], "candidate": pair["candidate"], "n": pair["n"],
            "committee": [c for c in committee], "per_repeat": per_repeat, "mean_over_repeats": mean}


def _pct(x): return f"{100*x:.1f}%"

def render(results) -> str:
    L = ["# Gate replay 2 -- committee disagreement and per-class ownership (offline, $0, out-of-fold)", "",
         "All gates OOF (5-fold over items, rule fitted without the item), mean of 3 cheap repeats. "
         "'ni' = cheapest training-fold point with accuracy >= incumbent-alone; 'maxacc' = training-fold max accuracy. "
         "committee = other cheap models (tfidf / bge-small / bge-base / bge-large minus the primary).", "",
         "## Who is right where the committee disagrees? (the diagnostic that decides it)", "",
         "| task | incumbent | primary | committee | disagree rate | on DISAGREE: cheap / LLM | on AGREE: cheap / LLM | committee majority vote (no LLM) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: (x["task"], x["model_key"])):
        d = r["mean_over_repeats"]["diag"]; m = r["mean_over_repeats"]
        L.append(f"| {r['task']} | {r['model_key']} | {r['candidate']} ({_pct(m['cheap_accuracy'])}) | {', '.join(r['committee'])} "
                 f"| {_pct(d['disagree_rate'])} | {_pct(d['cheap_acc_on_disagree'])} / {_pct(d['inc_acc_on_disagree'])} "
                 f"| {_pct(d['cheap_acc_on_agree'])} / {_pct(d['inc_acc_on_agree'])} | {_pct(d['majority_vote_accuracy'])} |")
    L += ["", "## Gates: accuracy @ LLM share (OOF)", "",
          "| task | incumbent | cheap | LLM | conformal | confidence-ni | committee-ni | committee-maxacc | committee+conf-maxacc | per-class any | per-class 2pt | oracle | best gate vs cheap (diff, p) |",
          "|" + "---|" * 13]
    for r in sorted(results, key=lambda x: (x["task"], x["model_key"])):
        m = r["mean_over_repeats"]; g = m["gates"]
        def gs(k): return f"{_pct(g[k]['accuracy'])} @ {_pct(g[k]['llm_share'])}"
        cands = ["confidence_ni", "committee_ni", "committee_maxacc", "committee_conf_maxacc", "perclass_any", "perclass_2pt"]
        best = max(cands, key=lambda k: g[k]["accuracy"])
        b = g[best]
        L.append(f"| {r['task']} | {r['model_key']} | {_pct(m['cheap_accuracy'])} | {_pct(m['incumbent_accuracy'])} | {gs('conformal')} "
                 f"| {gs('confidence_ni')} | {gs('committee_ni')} | {gs('committee_maxacc')} | {gs('committee_conf_maxacc')} "
                 f"| {gs('perclass_any')} | {gs('perclass_2pt')} | {gs('oracle')} "
                 f"| {best}: {100*b['mcnemar_vs_cheap']['diff']:+.1f}pp, p={b['mcnemar_vs_cheap']['p_value']:.3f} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-item-dirs", nargs="+", default=[str(_PROJECT / "results_phase2_cv" / "per_item"), str(_PROJECT / "results_cheap_v2_large" / "per_item")])
    ap.add_argument("--incumbent-dirs", nargs="+", default=[str(_PROJECT / f"results_stage{i}" / "incumbent") for i in (1, 2, 3)])
    ap.add_argument("--configs-dir", default=str(_PROJECT / "results_phase2_cv" / "configs"))
    ap.add_argument("--out-dir", default=str(_PROJECT / "results_gate2"))
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    results = []
    for f in sorted(p for d in a.incumbent_dirs for p in Path(d).glob("*__*.csv")):
        pair = load_pair(Path(a.per_item_dirs[0]), f, Path(a.configs_dir))
        committee = load_committee(a.per_item_dirs, pair["task"], pair["candidate"], pair["n"], pair["text_sha256"])
        res = analyze(pair, committee, folds_k=a.folds, seed=a.seed, n_boot=a.n_boot)
        with open(out / f"{pair['task']}__{pair['model_key']}.json", "w") as fh:
            json.dump(_json_safe(res), fh, indent=1)
        results.append(res)
    (out / "gate2_summary.md").write_text(render(results))
    print((out / "gate2_summary.md").read_text())
    return 0

if __name__ == "__main__":
    sys.exit(main())
