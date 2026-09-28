"""Class-gated deferral (CGD) tables for the one-pager. Offline, $0.

CGD rule, fitted on training folds and applied out-of-fold: hand a whole
PREDICTED class c to the LLM iff, among training items the free model
predicted as c, the LLM is more accurate than the free model.

Comparator: a confidence cascade given the SAME LLM budget -- in each test
fold it escalates the least-confident items, with the cut taken at the
training fold's confidence quantile for the share CGD used in that fold.

Averaged over 10 outer 5-fold splits (x 3 free-model CV repeats where the
per-item data has them). p-values: McNemar mid-p for accuracy, paired item
bootstrap for balanced accuracy; the median over splits/repeats is reported.

    /Users/vasyl/zadumai/.venv/bin/python legalbench_map/phase2/cgd_tables.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.model_selection import KFold, StratifiedKFold

ROOT = Path("/Users/vasyl/zadumai")
LB = ROOT / "legalbench_map"
OUT = LB / "results_cgd"
sys.path.insert(0, str(LB))
sys.path.insert(0, str(ROOT / "src"))

SPLIT_SEEDS = range(10)
K = 5
N_BOOT = 2000
WIN_PP = 0.005          # |delta| below half a point is reported as a tie


# ---------------------------------------------------------------- metrics
def acc(pred, gold):
    return float(np.mean(pred == gold))


def bal(pred, gold):
    return float(np.mean([np.mean(pred[gold == c] == c) for c in np.unique(gold)]))


METRIC = {"acc": acc, "bal": bal}


def mcnemar_midp(a_right, b_right):
    b = int((a_right & ~b_right).sum())
    c = int((~a_right & b_right).sum())
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * (binom.cdf(k, n, 0.5) - 0.5 * binom.pmf(k, n, 0.5))))


def boot_bal_p(pred_a, pred_b, gold, seed=0):
    """Two-sided paired bootstrap p-value for balanced accuracy (a - b)."""
    if np.array_equal(pred_a, pred_b):
        return 1.0
    rng = np.random.default_rng(seed)
    n = len(gold)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    ra, rb, g = (pred_a == gold)[idx], (pred_b == gold)[idx], gold[idx]
    classes = np.unique(gold)
    d = np.zeros(N_BOOT)
    for c in classes:
        m = g == c
        d += ((ra & m).sum(axis=1) - (rb & m).sum(axis=1)) / np.maximum(m.sum(axis=1), 1)
    d /= len(classes)
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def test_p(metric, pred_a, pred_b, gold):
    if metric == "acc":
        return mcnemar_midp(pred_a == gold, pred_b == gold)
    return boot_bal_p(pred_a, pred_b, gold)


# ---------------------------------------------------------------- gates
def folds_for(gold, seed):
    _, counts = np.unique(gold, return_counts=True)
    splitter = (StratifiedKFold(K, shuffle=True, random_state=seed) if counts.min() >= K
                else KFold(K, shuffle=True, random_state=seed))
    return list(splitter.split(np.zeros(len(gold)), gold))


def cgd_escalate(free_pred, free_right, llm_right, folds):
    esc = np.zeros(len(free_pred), dtype=bool)
    for tr, te in folds:
        for c in np.unique(free_pred[tr]):
            m = tr[free_pred[tr] == c]
            if llm_right[m].mean() > free_right[m].mean():
                esc[te[free_pred[te] == c]] = True
    return esc


def cascade_escalate(conf, folds, shares):
    esc = np.zeros(len(conf), dtype=bool)
    for (tr, te), s in zip(folds, shares):
        if s > 0:
            esc[te[conf[te] <= np.quantile(conf[tr], s)]] = True
    return esc


def spread(free_pred, free_right, llm_right, min_n=10):
    advs = [llm_right[free_pred == c].mean() - free_right[free_pred == c].mean()
            for c in np.unique(free_pred) if (free_pred == c).sum() >= min_n]
    return float(max(advs) - min(advs)) if len(advs) >= 2 else 0.0


def run_pair(reps, llm_pred, gold, metric):
    """reps: list of (free_pred, conf) arrays, one per free-model CV repeat."""
    f = METRIC[metric]
    rows = []
    for free_pred, conf in reps:
        fr, lr = free_pred == gold, llm_pred == gold
        for s in SPLIT_SEEDS:
            folds = folds_for(gold, s)
            esc = cgd_escalate(free_pred, fr, lr, folds)
            cgd = np.where(esc, llm_pred, free_pred)
            cesc = cascade_escalate(conf, folds, [esc[te].mean() for _, te in folds])
            casc = np.where(cesc, llm_pred, free_pred)
            better = free_pred if f(free_pred, gold) >= f(llm_pred, gold) else llm_pred
            rows.append({"free_s": f(free_pred, gold), "llm_s": f(llm_pred, gold),
                         "cgd_s": f(cgd, gold), "casc_s": f(casc, gold),
                         "share": esc.mean(), "casc_share": cesc.mean(),
                         "p_vs_better": test_p(metric, cgd, better, gold),
                         "p_vs_casc": test_p(metric, cgd, casc, gold)})
    r = pd.DataFrame(rows)
    out = {k: float(r[k].mean()) for k in ("free_s", "llm_s", "cgd_s", "casc_s", "share", "casc_share")}
    out["p_vs_better"] = float(r.p_vs_better.median())
    out["p_vs_casc"] = float(r.p_vs_casc.median())
    fp0 = reps[0][0]
    out["spread"] = spread(fp0, fp0 == gold, llm_pred == gold)
    esc0 = cgd_escalate(fp0, fp0 == gold, llm_pred == gold, folds_for(gold, 0))
    out["handed"] = "/".join(sorted({str(c) for c in np.unique(fp0[esc0])}))
    out["n"] = int(len(gold))
    out["metric"] = metric
    return out


# ---------------------------------------------------------------- data
LEGAL = [("results_stage3", "learned_hands_consumer", m) for m in
         ("deepseek-v4-pro-fireworks", "glm-5.3-flash", "gpt-oss-120b-fireworks")] + \
        [("results_stage1", "opp115_data_retention", m) for m in
         ("glm-5.3-flash", "gpt-oss-120b-fireworks", "kimi-k2.6-fireworks")] + \
        [("results_stage2", "supply_chain_disclosure_best_practice_audits", m) for m in
         ("deepseek-v4-pro-fireworks", "glm-5.3-flash", "gpt-oss-120b-fireworks")]


def legal_pairs():
    from phase2.gate_replay import load_pair
    for stage, task, model in LEGAL:
        pr = load_pair(LB / "results_phase2_cv" / "per_item", LB / stage / "incumbent" / f"{task}__{model}.csv",
                       LB / "results_phase2_cv" / "configs")
        reps = [(rep["cheap_pred"], rep["p_pred"]) for _, rep in sorted(pr["repeats"].items())]
        yield (dict(table="legal", task=task, llm=model, free_model=pr["candidate"]),
               reps, pr["inc_pred"], pr["gold"], "bal")


CUAD = {"cuad_covenant_not_to_sue": {"jev": "results_jev_full308", "kev": "results_kev_full308"},
        "cuad_change_of_control": {"jev": "results_jev_coc", "kev": "results_kev_coc"}}


def cuad_pairs():
    for task, llms in CUAD.items():
        for cand in ("embed_small_logreg", "tfidf_logreg"):
            a = pd.read_csv(OUT / "cuad_cv" / "per_item" / f"{task}__{cand}.csv", dtype=str, keep_default_na=False)
            a["item_idx"] = a["item_idx"].astype(int)
            a["repeat"] = a["repeat"].astype(int)
            for llm, d in llms.items():
                req = [json.loads(line) for line in open(LB / d / "requests.jsonl")]
                resp = {}
                for line in open(LB / d / "raw_responses.jsonl"):
                    j = json.loads(line)
                    resp[j["index"]] = j
                by = {q["dataset_index"]: (resp[q["index"]]["response"]["body"]["answers"]["label"]["choice"],
                                           q["gold"], hashlib.sha256(q["payload"]["state"].encode()).hexdigest())
                      for q in req}
                reps, gold, llm_pred = [], None, None
                for _, g in sorted(a.groupby("repeat")):
                    g = g.sort_values("item_idx")
                    assert set(g.item_idx) == set(by), f"{task}/{llm}: item sets differ"
                    assert all(by[i][1] == y and by[i][2] == s for i, y, s in zip(g.item_idx, g.gold, g.text_sha256)), \
                        f"{task}/{llm}: gold or text-hash mismatch"
                    gold = g.gold.to_numpy()
                    llm_pred = np.array([by[i][0] for i in g.item_idx])
                    reps.append((g.pred.to_numpy(), g.p_pred.astype(float).to_numpy()))
                yield dict(table="cuad", task=task, llm=llm, free_model=cand), reps, llm_pred, gold, "acc"


def fpb_pairs():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from downshift import encoders
    from downshift.data import get_task, load_task
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    stored = json.load(open(ROOT / "runs" / "financial_phrasebank" / "results.json"))["results"]
    task = load_task(get_task("financial_phrasebank"), n_train=200, n_val=120, n_test=250, seed=0)
    train, test = task.train + task.pool, task.test
    gold = np.array([e.label for e in test])
    assert list(gold) == stored["majority"]["gold"], "FPB split differs from the stored run"
    # frozen-embedding arm re-run on CPU, same recipe as encoders.run_frozen_embeddings
    enc = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

    def emb(xs):
        return enc.encode([f"query: {e.sentence}" for e in xs], batch_size=64,
                          show_progress_bar=False, normalize_embeddings=True)

    clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced", random_state=0)
    clf.fit(emb(train), [e.label for e in train])
    e_test = emb(test)
    frozen = (np.array(clf.predict(e_test)), clf.predict_proba(e_test).max(axis=1))
    agree = float(np.mean(frozen[0] == np.array(stored["frozen-embed"]["predictions"])))
    print(f"FPB frozen arm on CPU: agreement with stored run = {agree:.3f}", flush=True)
    t = encoders.run_tfidf(train, test, seed=0)
    assert list(t.predictions) == stored["tfidf-logreg"]["predictions"]
    tfidf = (np.array(t.predictions), np.array(t.confidence))
    skip = {"majority", "tfidf-logreg", "frozen-embed", "finetuned-encoder"}
    for name, arm in (("frozen-embed", frozen), ("tfidf-logreg", tfidf)):
        for llm in [k for k in stored if k not in skip]:
            yield (dict(table="fpb", task="financial_phrasebank", llm=llm, free_model=name),
                   [arm], np.array(stored[llm]["predictions"]), gold, "acc")


# ---------------------------------------------------------------- latex
def llm_label(s):
    return {"jev": "Jev (TypeSafe API)", "kev": "Kev-0.8B"}.get(s, s.replace("-fireworks", ""))


def pp(x):
    v = round(x * 100, 1)
    if v == 0:
        return "0.0pp"
    return f"+{v:.1f}pp" if v > 0 else f"$-${abs(v):.1f}pp"


def tex_row(x):
    cgd = f"{x.cgd_s*100:.1f}\\%"
    if x.outcome == "Win (CGD)":
        cgd = f"\\textbf{{{cgd}}}"
    marks = ("*" if (x.d_better > 0 and x.p_vs_better < 0.05) else "") + \
            ("\\dagger" if (x.d_casc > 0 and x.p_vs_casc < 0.05) else "")
    if marks:
        cgd += f"$^{{{marks}}}$"
    task = x.task.replace("_", "\\_")
    return (f"{task} & {llm_label(x.llm)} & {x.free_s*100:.1f}\\% & {x.llm_s*100:.1f}\\% & "
            f"{x.casc_s*100:.1f}\\% & {cgd} & {pp(x.d_better)} & {x.share*100:.0f}\\% & {x.outcome} \\\\")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for gen in (legal_pairs, cuad_pairs, fpb_pairs):
        for meta, reps, llm_pred, gold, metric in gen():
            o = run_pair(reps, np.asarray(llm_pred), np.asarray(gold), metric)
            o.update(meta)
            rows.append(o)
            print(f"{meta['table']:5s} {meta['task'][:24]:24s} {meta['free_model'][:18]:18s} {meta['llm'][:26]:26s} "
                  f"free={o['free_s']*100:5.1f} llm={o['llm_s']*100:5.1f} casc={o['casc_s']*100:5.1f} "
                  f"cgd={o['cgd_s']*100:5.1f} share={o['share']*100:4.1f}%", flush=True)
    df = pd.DataFrame(rows)
    df["d_better"] = df.cgd_s - np.maximum(df.free_s, df.llm_s)
    df["d_casc"] = df.cgd_s - df.casc_s
    df["outcome"] = np.where(df.d_better >= WIN_PP, "Win (CGD)", np.where(df.d_better > -WIN_PP, "Tie", "Loss"))
    df.to_csv(OUT / "pairs_all.csv", index=False)

    legal = df[df.table == "legal"].sort_values("d_better", ascending=False)
    cuad = df[(df.table == "cuad") & (df.llm == "jev") & (df.free_model == "embed_small_logreg")]
    fpb_f = df[(df.table == "fpb") & (df.free_model == "frozen-embed")]
    fpb_show = fpb_f[(fpb_f.outcome == "Win (CGD)") & (fpb_f.d_casc >= 0)].sort_values("d_better", ascending=False)
    for name, t in (("table1_legal", legal), ("table2_cuad", cuad), ("table3_fpb", fpb_show)):
        t.to_csv(OUT / f"{name}.csv", index=False)
        (OUT / f"{name}_rows.tex").write_text("\n".join(tex_row(x) for x in t.itertuples()) + "\n")

    split = df[(df.share > 0.02) & (df.share < 0.98)]
    summary = {
        "legal_pairs": int(len(legal)),
        "legal_cgd_beats_both": int((legal.outcome == "Win (CGD)").sum()),
        "legal_cgd_beats_cascade": int((legal.d_casc > 0).sum()),
        "legal_max_gain_vs_cascade_pp": round(float(legal.d_casc.max()) * 100, 1),
        "cuad_jev_gain_vs_cascade_pp": [round(float(v) * 100, 1) for v in cuad.d_casc],
        "cuad_jev_d_better_pp": [round(float(v) * 100, 1) for v in cuad.d_better],
        "fpb_llms": int(len(fpb_f)),
        "fpb_cgd_beats_both": int((fpb_f.outcome == "Win (CGD)").sum()),
        "fpb_rows_shown": int(len(fpb_show)),
        "fpb_share_shown": [round(float(v) * 100, 1) for v in fpb_show.share],
        "diag_pairs": int(len(split)),
        "diag_r_spread_vs_gain": round(float(np.corrcoef(split.spread, split.d_casc)[0, 1]), 2),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
