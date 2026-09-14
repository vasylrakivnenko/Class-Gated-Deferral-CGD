"""Gate replay: can a better gate recover the oracle gap? Offline, $0.

Stages 1-3 measured, on 9 (task x incumbent) pairs, that the deployed
conformal gate escalates 2-3x more items than a perfect gate would and that
the LLM is usually no better than the cheap model on what it escalates. This
script asks whether the GATE is the problem, using only data already on disk:

  FILE A  <per-item-dir>/<task>__<best_candidate>.csv   cheap OOF preds, 3 repeats
  FILE B  <incumbent-dir>/<task>__<model>.csv            incumbent preds, 1 pass

Gates (every routing decision is made by a rule fitted WITHOUT that item --
5-fold out-of-fold over items, per cheap repeat):
  conformal  kept==1 from FILE A. The deployed gate, re-derived as a check.
  threshold  escalate if p_pred < t. t chosen on the training folds.
  router     two logistic regressions P(LLM right | x) and P(cheap right | x)
             on x = [p_pred, set_size, log(len), pred-class] (+ a bge-small
             text embedding unless --no-embeddings); escalate if
             P_llm - P_cheap > m, with m chosen by an inner 3-fold CV on the
             training folds. NOTE: training the router needs per-item LLM
             labels on a calibration set -- in deployment that is one LLM
             pass over a few hundred items, a real (small) cost.
  oracle     escalate exactly the cheap model's errors. The ceiling.

Two selection criteria on the training folds, reported separately:
  maxacc  maximize cascade accuracy
  ni      cheapest operating point whose cascade accuracy >= incumbent-alone
          accuracy on the same training items (never worse than the LLM,
          then as cheap as possible)

Outputs (under --out-dir): <task>__<model>.json, gate_summary.md,
gate_frontiers.png. Every accuracy here is OOF over items and averaged over
the 3 cheap repeats; the incumbent is a single temperature-0 pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, "/Users/vasyl/zadumai/src")

from core.data import load_registry, load_task_data, parse_task_spec  # noqa: E402
from phase2.analyze import (  # noqa: E402
    FILE_A_COLS, FILE_B_COLS, _ni, _paired, add_lenient_columns, read_best_candidate,
)

logger = logging.getLogger("legalbench_map.phase2.gate_replay")

M_GRID = np.round(np.linspace(-0.5, 0.5, 41), 4)
CRITERIA = ("maxacc", "ni")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_pair(per_item_dir: Path, incumbent_path: Path, configs_dir: Path) -> dict:
    """Join FILE A (all repeats) with FILE B on item_idx. Returns arrays."""
    b = pd.read_csv(incumbent_path, dtype=str, keep_default_na=False)
    assert list(b.columns) == FILE_B_COLS, f"FILE B columns differ: {incumbent_path}"
    task, model_key = b["task"].iloc[0], b["model_key"].iloc[0]
    candidate = read_best_candidate(configs_dir, task)
    a = pd.read_csv(per_item_dir / f"{task}__{candidate}.csv", dtype=str, keep_default_na=False)
    assert list(a.columns) == FILE_A_COLS, "FILE A columns differ"
    b["item_idx"] = b["item_idx"].astype(int)
    b = b.sort_values("item_idx").reset_index(drop=True)
    n = len(b)
    assert b["item_idx"].tolist() == list(range(n)), "FILE B must cover item_idx 0..n-1"
    assert not b["raw_output"].str.startswith("<error:").any(), "API-error rows present; rerun the runner first"
    classes = sorted(set(a["gold"]))
    b = add_lenient_columns(b, classes)
    gold = b["gold"].str.strip().str.lower().to_numpy()
    inc_pred = b["pred_lenient"].to_numpy()
    inc_correct = inc_pred == gold
    inc_cost_per_1k = float(b["cost_usd"].astype(float).mean()) * 1000.0

    a["item_idx"] = a["item_idx"].astype(int)
    a["repeat"] = a["repeat"].astype(int)
    repeats = {}
    for r, g in a.groupby("repeat"):
        g = g[g["item_idx"] < n].sort_values("item_idx").reset_index(drop=True)
        assert g["item_idx"].tolist() == list(range(n)), f"FILE A repeat {r} does not cover every item"
        assert (g["text_sha256"].to_numpy() == b["text_sha256"].to_numpy()).all(), "text_sha256 mismatch A vs B"
        assert (g["gold"].to_numpy() == gold).all(), "gold mismatch A vs B"
        repeats[int(r)] = {
            "cheap_pred": g["pred"].to_numpy(),
            "cheap_correct": g["pred"].to_numpy() == gold,
            "p_pred": g["p_pred"].astype(float).to_numpy(),
            "set_size": g["set_size"].astype(int).to_numpy(),
            "kept": g["kept"].astype(int).to_numpy(),
        }
    return {"task": task, "model_key": model_key, "candidate": candidate, "classes": classes, "n": n,
            "gold": gold, "inc_pred": inc_pred, "inc_correct": inc_correct,
            "inc_cost_per_1k": inc_cost_per_1k, "text_sha256": b["text_sha256"].tolist(), "repeats": repeats}


def load_texts(registry_path: Path, task: str, expected_sha: list[str]) -> list[str]:
    entry = next(e for e in load_registry(registry_path) if e["task"] == task)
    td = load_task_data(entry)
    got = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in td.X_test]
    assert got[: len(expected_sha)] == expected_sha, f"{task}: loaded texts do not match FILE B text_sha256"
    return list(td.X_test[: len(expected_sha)])


def embed_texts(texts: list[str], task: str, cache_dir: Path, model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(("\n".join(texts) + model_name).encode("utf-8")).hexdigest()[:16]
    path = cache_dir / f"emb_{task}_{key}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = np.asarray(model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)
    np.save(path, emb)
    return emb


def scalar_features(rep: dict, texts: list[str] | None, classes: list[str]) -> np.ndarray:
    n = len(rep["p_pred"])
    log_len = np.array([math.log(1 + len(t)) for t in texts]) if texts is not None else np.zeros(n)
    pred_class = np.array([classes.index(p) if p in classes else -1 for p in rep["cheap_pred"]], dtype=float)
    return np.column_stack([rep["p_pred"], rep["set_size"].astype(float), log_len, pred_class])


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------

def outer_folds(gold: np.ndarray, k: int, seed: int):
    """Stratified k-fold over items. Returns list of (train_idx, test_idx)."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(gold)), gold)]


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def _cascade_correct(escalate: np.ndarray, cheap_correct: np.ndarray, inc_correct: np.ndarray) -> np.ndarray:
    return np.where(escalate, inc_correct, cheap_correct)


def _choose(shares: np.ndarray, accs: np.ndarray, criterion: str, inc_acc_train: float) -> int:
    """Index of the chosen operating point among candidates (on TRAINING data)."""
    if criterion == "maxacc":
        best = accs.max()
        cands = np.where(accs >= best - 1e-12)[0]
        return int(cands[np.argmin(shares[cands])])
    if criterion == "ni":
        ok = np.where(accs >= inc_acc_train - 1e-12)[0]
        if len(ok) == 0:  # cannot happen: share==1 reproduces the incumbent exactly
            ok = np.arange(len(shares))
        return int(ok[np.argmin(shares[ok])])
    raise ValueError(criterion)


def threshold_frontier(p: np.ndarray, cheap_correct: np.ndarray, inc_correct: np.ndarray):
    """All operating points of 'escalate if p_pred < t' on the given items."""
    ts = np.concatenate([[-np.inf], np.unique(p), [np.inf]])   # -inf: never escalate; inf: always
    shares, accs = [], []
    for t in ts:
        esc = p < t
        shares.append(float(esc.mean()))
        accs.append(float(_cascade_correct(esc, cheap_correct, inc_correct).mean()))
    return ts, np.array(shares), np.array(accs)


def threshold_gate_oof(p, cheap_correct, inc_correct, folds) -> dict[str, np.ndarray]:
    """OOF escalate decisions per criterion; t chosen on training folds only."""
    out = {c: np.zeros(len(p), dtype=bool) for c in CRITERIA}
    for tr, te in folds:
        ts, sh, ac = threshold_frontier(p[tr], cheap_correct[tr], inc_correct[tr])
        inc_acc_tr = float(inc_correct[tr].mean())
        for c in CRITERIA:
            t = ts[_choose(sh, ac, c, inc_acc_tr)]
            out[c][te] = p[te] < t
    return out


def _fit_prob(X_tr, y_tr, X_te, seed: int) -> np.ndarray:
    """P(y=1|x) on X_te from a standardized L2 logistic regression; constant if y_tr is single-class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y_tr = np.asarray(y_tr, dtype=int)
    if y_tr.min() == y_tr.max():
        return np.full(len(X_te), float(y_tr[0]))
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.5, max_iter=5000, random_state=seed).fit(sc.transform(X_tr), y_tr)
    return clf.predict_proba(sc.transform(X_te))[:, 1]


def router_gate_oof(X, cheap_correct, inc_correct, folds, seed: int) -> dict[str, np.ndarray]:
    """OOF escalate decisions per criterion. Inner 3-fold CV on each training
    fold chooses the margin m; models are then refit on the full training fold
    and applied to the held-out fold."""
    from sklearn.model_selection import StratifiedKFold
    out = {c: np.zeros(len(cheap_correct), dtype=bool) for c in CRITERIA}
    for fi, (tr, te) in enumerate(folds):
        # inner OOF probabilities on the training fold, to choose m honestly
        inner_pl = np.zeros(len(tr)); inner_pc = np.zeros(len(tr))
        strat = (cheap_correct[tr].astype(int) * 2 + inc_correct[tr].astype(int))
        n_inner = 3 if np.bincount(strat).min() >= 3 else 2
        try:
            inner = list(StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + fi).split(X[tr], strat))
        except ValueError:
            inner = list(StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + fi).split(X[tr], np.zeros(len(tr), int)))
        for itr, ite in inner:
            inner_pl[ite] = _fit_prob(X[tr][itr], inc_correct[tr][itr], X[tr][ite], seed)
            inner_pc[ite] = _fit_prob(X[tr][itr], cheap_correct[tr][itr], X[tr][ite], seed)
        edge_tr = inner_pl - inner_pc
        shares = np.array([float((edge_tr > m).mean()) for m in M_GRID])
        accs = np.array([float(_cascade_correct(edge_tr > m, cheap_correct[tr], inc_correct[tr]).mean()) for m in M_GRID])
        inc_acc_tr = float(inc_correct[tr].mean())
        # refit on the whole training fold, apply to held-out
        pl = _fit_prob(X[tr], inc_correct[tr], X[te], seed)
        pc = _fit_prob(X[tr], cheap_correct[tr], X[te], seed)
        edge_te = pl - pc
        for c in CRITERIA:
            m = M_GRID[_choose(shares, accs, c, inc_acc_tr)]
            out[c][te] = edge_te > m
    return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate_gate(escalate, cheap_correct, inc_correct, inc_cost_per_1k, n_boot, seed, delta=0.01) -> dict:
    casc = _cascade_correct(escalate, cheap_correct, inc_correct)
    share = float(np.mean(escalate))
    return {
        "accuracy": float(casc.mean()),
        "llm_share": share,
        "cost_per_1k_items_usd": share * inc_cost_per_1k,
        "mcnemar_vs_cheap": _paired(casc, cheap_correct, n_boot, seed),
        "ni_vs_cheap": _ni(casc, cheap_correct, delta, n_boot, seed),
        "mcnemar_vs_incumbent": _paired(casc, inc_correct, n_boot, seed),
        "ni_vs_incumbent": _ni(casc, inc_correct, delta, n_boot, seed),
    }


def _mean_num(dicts: list[dict]) -> dict:
    """Mean of numeric leaves across repeats (nested dicts), else first value."""
    out = {}
    for k in dicts[0]:
        vals = [d[k] for d in dicts]
        if isinstance(vals[0], dict):
            out[k] = _mean_num(vals)
        elif isinstance(vals[0], (int, float, bool, np.floating, np.integer, np.bool_)) and not isinstance(vals[0], str):
            out[k] = float(np.mean([float(v) for v in vals]))
        else:
            out[k] = vals[0]
    return out


def analyze_pair(pair: dict, texts: list[str] | None, emb: np.ndarray | None, *, folds_k: int, seed: int, n_boot: int) -> dict:
    gold, inc_correct, cost1k = pair["gold"], pair["inc_correct"], pair["inc_cost_per_1k"]
    per_repeat = {}
    frontier_pts = []
    for r, rep in pair["repeats"].items():
        cc = rep["cheap_correct"]
        folds = outer_folds(gold, folds_k, seed + r)
        X = scalar_features(rep, texts, pair["classes"])
        if emb is not None:
            X = np.column_stack([X, emb])
        gates = {
            "conformal": {"": rep["kept"] == 0},
            "threshold": threshold_gate_oof(rep["p_pred"], cc, inc_correct, folds),
            "router": router_gate_oof(X, cc, inc_correct, folds, seed + r),
            "oracle": {"": ~cc},
        }
        res = {"cheap_accuracy": float(cc.mean()), "incumbent_accuracy": float(inc_correct.mean()), "gates": {}}
        for gname, variants in gates.items():
            for crit, esc in variants.items():
                key = gname if not crit else f"{gname}_{crit}"
                res["gates"][key] = evaluate_gate(esc, cc, inc_correct, cost1k, n_boot, seed + r)
        ts, sh, ac = threshold_frontier(rep["p_pred"], cc, inc_correct)   # in-sample curve, chart only
        frontier_pts.append((sh.tolist(), ac.tolist()))
        per_repeat[str(r)] = res
    mean = _mean_num(list(per_repeat.values()))
    g = mean["gates"]
    gap = g["oracle"]["accuracy"] - g["conformal"]["accuracy"]
    for key in list(g):
        g[key]["oracle_gap_recovered"] = float((g[key]["accuracy"] - g["conformal"]["accuracy"]) / gap) if gap > 1e-12 else float("nan")
    return {"task": pair["task"], "model_key": pair["model_key"], "candidate": pair["candidate"], "n": pair["n"],
            "incumbent_cost_per_1k_usd": cost1k, "folds": folds_k, "seed": seed, "n_boot": n_boot,
            "router_features": "p_pred,set_size,log_len,pred_class" + (",bge-small-384" if emb is not None else ""),
            "per_repeat": per_repeat, "mean_over_repeats": mean, "frontier_in_sample": frontier_pts}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _pct(x): return f"{100*x:.1f}%"

def render_summary(results: list[dict]) -> str:
    L = ["# Gate replay -- can a better gate recover the oracle gap? (offline, $0)", "",
         "All gate numbers are out-of-fold over items (5-fold, rule fitted without the item), mean of 3 cheap repeats. "
         "Incumbent = single temperature-0 pass (lenient scoring). 'gap recovered' = (gate - conformal) / (oracle - conformal). "
         "'ni' = cheapest point whose training-fold accuracy >= incumbent-alone; 'maxacc' = training-fold max accuracy.", "",
         "| task | incumbent | cheap | LLM | conformal acc / share | threshold-ni acc / share | router-ni acc / share | router-maxacc acc / share | oracle acc / share | gap recovered: thr-ni / router-ni / router-maxacc | router-maxacc vs cheap (diff, p) | router-maxacc vs LLM (diff, p) | $/1k LLM -> router-ni |",
         "|" + "---|" * 13]
    for r in sorted(results, key=lambda x: (x["task"], x["model_key"])):
        m = r["mean_over_repeats"]; g = m["gates"]
        def gs(k): return f"{_pct(g[k]['accuracy'])} / {_pct(g[k]['llm_share'])}"
        rm = g["router_maxacc"]
        L.append(f"| {r['task']} | {r['model_key']} | {_pct(m['cheap_accuracy'])} | {_pct(m['incumbent_accuracy'])} "
                 f"| {gs('conformal')} | {gs('threshold_ni')} | {gs('router_ni')} | {gs('router_maxacc')} | {gs('oracle')} "
                 f"| {g['threshold_ni']['oracle_gap_recovered']:+.2f} / {g['router_ni']['oracle_gap_recovered']:+.2f} / {rm['oracle_gap_recovered']:+.2f} "
                 f"| {100*rm['mcnemar_vs_cheap']['diff']:+.1f}pp, p={rm['mcnemar_vs_cheap']['p_value']:.3f} "
                 f"| {100*rm['mcnemar_vs_incumbent']['diff']:+.1f}pp, p={rm['mcnemar_vs_incumbent']['p_value']:.3f} "
                 f"| {r['incumbent_cost_per_1k_usd']:.3f} -> {g['router_ni']['cost_per_1k_items_usd']:.3f} |")
    L += ["", "Router features: " + results[0]["router_features"] + ". Training the router needs per-item LLM labels on a "
          "calibration set (one LLM pass over a few hundred items) -- a small real deployment cost the conformal gate does not have.", ""]
    return "\n".join(L)


def render_chart(results: list[dict], path: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = sorted({r["task"] for r in results}); models = sorted({r["model_key"] for r in results})
    fig, axes = plt.subplots(len(tasks), len(models), figsize=(4.2 * len(models), 3.6 * len(tasks)), squeeze=False)
    for i, t in enumerate(tasks):
        for j, mk in enumerate(models):
            ax = axes[i][j]
            r = next((x for x in results if x["task"] == t and x["model_key"] == mk), None)
            if r is None:
                ax.set_axis_off(); ax.set_title(f"{mk}\n(not run on {t})", fontsize=7); continue
            m = r["mean_over_repeats"]; g = m["gates"]
            for sh, ac in r["frontier_in_sample"]:
                ax.plot(sh, ac, color="#bbbbbb", lw=0.8, zorder=1)
            ax.scatter([0], [m["cheap_accuracy"]], marker="s", color="black", s=40, zorder=3, label="cheap alone")
            ax.scatter([1], [m["incumbent_accuracy"]], marker="s", color="#7f7f7f", s=40, zorder=3, label="LLM alone")
            ax.scatter([g["conformal"]["llm_share"]], [g["conformal"]["accuracy"]], marker="D", color="#d62728", s=45, zorder=4, label="conformal (deployed)")
            ax.scatter([g["threshold_ni"]["llm_share"]], [g["threshold_ni"]["accuracy"]], marker="^", color="#ff7f0e", s=45, zorder=4, label="threshold-ni (OOF)")
            ax.scatter([g["router_ni"]["llm_share"]], [g["router_ni"]["accuracy"]], marker="o", color="#1f77b4", s=45, zorder=5, label="router-ni (OOF)")
            ax.scatter([g["router_maxacc"]["llm_share"]], [g["router_maxacc"]["accuracy"]], marker="o", facecolors="none", edgecolors="#1f77b4", s=60, zorder=5, label="router-maxacc (OOF)")
            ax.scatter([g["oracle"]["llm_share"]], [g["oracle"]["accuracy"]], marker="*", color="#2ca02c", s=90, zorder=6, label="oracle")
            ax.set_xlim(-0.03, 1.03); ax.set_title(f"{t[:34]}\nx {mk}", fontsize=8)
            ax.set_xlabel("LLM share", fontsize=8); ax.set_ylabel("accuracy", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Gate replay: accuracy vs LLM share -- grey = in-sample threshold frontier; markers = out-of-fold gates", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _json_safe(o):
    if isinstance(o, dict): return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-item-dir", default=str(_PROJECT / "results_phase2_cv" / "per_item"))
    ap.add_argument("--incumbent-dirs", nargs="+", default=[str(_PROJECT / f"results_stage{i}" / "incumbent") for i in (1, 2, 3)])
    ap.add_argument("--configs-dir", default=str(_PROJECT / "results_phase2_cv" / "configs"))
    ap.add_argument("--registry", default=str(_PROJECT / "data" / "task_registry.json"))
    ap.add_argument("--out-dir", default=str(_PROJECT / "results_gate"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for d in args.incumbent_dirs for p in Path(d).glob("*__*.csv"))
    results, text_cache, emb_cache = [], {}, {}
    for f in files:
        pair = load_pair(Path(args.per_item_dir), f, Path(args.configs_dir))
        t = pair["task"]
        if t not in text_cache:
            text_cache[t] = load_texts(Path(args.registry), t, pair["text_sha256"])
            emb_cache[t] = None if args.no_embeddings else embed_texts(text_cache[t], t, out / "emb_cache")
        res = analyze_pair(pair, text_cache[t], emb_cache[t], folds_k=args.folds, seed=args.seed, n_boot=args.n_boot)
        with open(out / f"{t}__{pair['model_key']}.json", "w") as fh:
            json.dump(_json_safe(res), fh, indent=1)
        g = res["mean_over_repeats"]["gates"]
        logger.warning("%s x %s: conformal %.3f@%.2f  thr-ni %.3f@%.2f  router-ni %.3f@%.2f  router-max %.3f@%.2f  oracle %.3f@%.2f",
                       t, pair["model_key"], g["conformal"]["accuracy"], g["conformal"]["llm_share"],
                       g["threshold_ni"]["accuracy"], g["threshold_ni"]["llm_share"], g["router_ni"]["accuracy"], g["router_ni"]["llm_share"],
                       g["router_maxacc"]["accuracy"], g["router_maxacc"]["llm_share"], g["oracle"]["accuracy"], g["oracle"]["llm_share"])
        results.append(res)
    (out / "gate_summary.md").write_text(render_summary(results))
    render_chart(results, out / "gate_frontiers.png")
    print((out / "gate_summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
