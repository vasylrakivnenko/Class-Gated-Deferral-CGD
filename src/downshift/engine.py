"""The fixed pipeline: Dataset in, artifact out.

Nothing in this module knows where a dataset came from. It takes a `Dataset`
(see `downshift/dataset.py`), runs the candidate pool, the gates, the
constructions, DESlib and -- if the dataset carries expert predictions -- the
cascade, and returns one artifact shaped the same way every time.

This is the part that should stop changing. It is edited when a gate is wrong,
a statistic is wrong, or a candidate is added to the pool for every dataset --
not to accommodate a source. Sources are adapted into `Dataset` instead.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from . import classpipe as CP
from .dataset import Dataset

# bge-base dropped deliberately: small/base/large is one family at three sizes,
# so it comes out ROC-ordered by capacity on every dataset and G3 can never do
# anything but fail. small (cheapest encoder) and large (best) bracket the range;
# the middle rung told us nothing.
ENCODERS = ("BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5")
NLI_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
# Measured and dropped from the default pool, not untried: last of the real
# candidates on every dataset it saw (ISEAR 0.4181, FEVER 0.5210, IMDB
# 0.6562), and ablation showed it contributes 0.0015 to the stacking win
# while NLI carries it. Still available with --with-reranker.
RERANK_MODEL = "BAAI/bge-reranker-base"
# Above this many classes, pair-scoring every (item, label) stops being
# affordable, so shortlist with the bi-encoder first and only score the
# top few. CLINC150 goes from ~3.4M forward passes to ~113k.
SHORTLIST_OVER_K = 10
SHORTLIST_M = 5
RUNGS = (0.2, 0.5, 1.0)
# Set by the caller (a runner knows where its cache lives; the engine does not).
EMB_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "downshift", "emb")

def _tfidf():
    """The program's standard free arm; the char half carries non-space scripts."""
    return Pipeline([("f", FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              sublinear_tf=True, min_df=2))])),
        ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def _fingerprint(texts) -> str:
    """Content address for a corpus. Keying a cache on (model, row count) alone
    means two datasets that happen to share a row count silently read each
    other's vectors, and nothing downstream can detect it -- the array has the
    right shape and the wrong meaning. Hash what was actually encoded."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(len(texts)).encode())
    for t in texts:
        h.update(b"\x00")
        h.update(str(t).encode("utf-8", "replace"))
    return h.hexdigest()


def _cache_load(path):
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except (ValueError, OSError, EOFError):
        # a run killed mid-write leaves a truncated .npy that would otherwise be
        # trusted forever; drop it and recompute
        os.remove(path)
        return None


def _cache_save(path, arr):
    """Write through a temp file in the same directory. `np.save` straight to
    the final name leaves a half-written array behind if the process dies."""
    tmp = f"{path}.{os.getpid()}.tmp.npy"   # .npy, or np.save appends its own
    np.save(tmp, arr)
    os.replace(tmp, path)


def _embed(texts, name):
    os.makedirs(EMB_CACHE, exist_ok=True)
    path = os.path.join(EMB_CACHE,
                        f"{name.split('/')[-1]}_{_fingerprint(texts)}.npy")
    hit = _cache_load(path)
    if hit is not None:
        return hit
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    E = SentenceTransformer(name, device=dev).encode(
        list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    _cache_save(path, E)
    return E


def _pair_scores(texts, label_names: dict[int, str], task: str, model_name: str,
                 tag: str, max_length: int = 256, shortlist_emb=None):
    """Entailment logit for (premise=text, hypothesis="This example involves X.")
    for every label X -- one column per class.

    Why this candidate exists: every other model in the pool embeds the text and
    fits a linear head, so they order themselves by encoder capacity and G3 has
    nothing to judge. NLI reaches the answer a different way, which is the only
    honest route to a pool that is not a capacity ladder.

    One template shape for every task and every label, per the convention in
    legalbench_map/candidates/zeroshot_nli.py -- only the label string changes,
    so there is nothing task-specific to tune.
    """
    os.makedirs(EMB_CACHE, exist_ok=True)
    path = os.path.join(EMB_CACHE, f"{tag}_{task}_{_fingerprint(texts)}.npy")
    hit = _cache_load(path)
    if hit is not None:
        return hit
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()
    # Read the entailment index off the config rather than assuming it is 0.
    # NLI heads expose an explicit entailment class; a reranker has a single
    # relevance logit. Read which off the config rather than assuming.
    n_out = int(getattr(mdl.config, "num_labels", 1))
    ent = 0 if n_out == 1 else next(
        (i for i, l in mdl.config.id2label.items()
         if "entail" in str(l).lower() and "not" not in str(l).lower()), 0)
    order = sorted(label_names)
    hyps = [f"This example involves {label_names[c]}." for c in order]
    n, k = len(texts), len(order)

    # Retrieve-then-rerank: at large k, score only the labels the bi-encoder
    # already thinks are plausible. Unscored cells are filled with the minimum
    # observed score, which is what "the retriever did not surface this label"
    # means to the head downstream.
    if shortlist_emb is not None and k > SHORTLIST_OVER_K:
        from sentence_transformers import SentenceTransformer
        H = SentenceTransformer(ENCODERS[0], device=dev).encode(
            hyps, normalize_embeddings=True, show_progress_bar=False)
        short = CP.label_shortlist(shortlist_emb, H, SHORTLIST_M)
        pairs = [(i, int(c)) for i in range(n) for c in short[i]]
        print(f"    {tag} {task}: shortlisted {k} labels -> {SHORTLIST_M}, "
              f"{len(pairs):,} pairs instead of {n*k:,}", flush=True)
    else:
        pairs = [(i, c) for i in range(n) for c in range(k)]

    scored = np.full((n, k), np.nan, dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(pairs), 64):
            chunk = pairs[a:a + 64]
            enc = tok([texts[i] for i, _ in chunk], [hyps[c] for _, c in chunk],
                      truncation=True, padding=True, max_length=max_length,
                      return_tensors="pt").to(dev)
            v = mdl(**enc).logits[:, ent].float().cpu().numpy()
            for (i, c), s_ in zip(chunk, v):
                scored[i, c] = s_
            if a % 12800 == 0:
                print(f"    {tag} {task}: {a:,}/{len(pairs):,} pairs", flush=True)
    out = np.where(np.isnan(scored), np.nanmin(scored), scored)
    _cache_save(path, out)
    return out


def sweep(texts, gold, groups, label_names: dict[int, str], seed: int = 0,
          folds_k: int = 5, task: str = "",
          with_reranker: bool = False, fit_y=None):
    """Out-of-fold predictions and probabilities for every candidate.

    Grouped folds (gate G2): duplicate texts share a group id, so a copy can
    never sit in train while its twin is scored in test.
    """
    folds = list(StratifiedGroupKFold(folds_k, shuffle=True,
                                      random_state=seed).split(texts, gold, groups))
    # What the candidates are allowed to learn from. Gold by default; the LLM's
    # own annotations when reproducing OCL's information setting, where gold
    # does not exist at run time. Evaluation is against gold either way.
    y = gold if fit_y is None else fit_y
    classes = np.array(sorted(set(gold.tolist())))
    pred, conf, pos, proba = {}, {}, {}, {}

    col = {c: j for j, c in enumerate(classes)}

    def _collect(name, fit_predict):
        pr = np.zeros(len(gold), dtype=int)
        cf = np.zeros(len(gold))
        pp = np.zeros(len(gold))
        pb = np.zeros((len(gold), len(classes)))
        for tr, te in folds:
            p_, pr_, seen = fit_predict(tr, te)
            pr[te] = p_
            cf[te] = pr_.max(axis=1)
            pp[te] = pr_[:, -1] if pr_.shape[1] == 2 else pr_.max(axis=1)
            # Place each column where its CLASS lives, not where the estimator
            # happened to put it. A fold whose training split is missing a rare
            # class -- or, under --train-on llm, a class the teacher never
            # predicts -- yields fewer columns than the global class list, and
            # positional assignment would either crash or silently file class 5
            # under class 4 for the rest of the pipeline.
            for j, c in enumerate(seen):
                pb[te, col[int(c)]] = pr_[:, j]
        pred[name], conf[name], pos[name], proba[name] = pr, cf, pp, pb

    _collect("tfidf+logreg", lambda tr, te: (
        lambda m: (m.predict(texts[te]), m.predict_proba(texts[te]), m.classes_)
    )(_tfidf().fit(texts[tr], y[tr])))
    for enc in ENCODERS:
        E = _embed(texts, enc)
        _collect(enc.split("/")[-1] + "+logreg", lambda tr, te, E=E: (
            lambda m: (m.predict(E[te]), m.predict_proba(E[te]), m.classes_)
        )(LogisticRegression(max_iter=3000, class_weight="balanced").fit(E[tr], y[tr])))
    if label_names:
        # Both score (text, label-sentence) pairs jointly rather than embedding
        # the text alone -- the only mechanism in the pool that does, which is
        # what stops G3 from seeing a pure capacity ladder. NLI is trained for
        # entailment (a native fit); the reranker for query-document relevance
        # (a stretch), so expect it to trail on accuracy and earn its place, if
        # at all, on disagreement.
        plan = CP.pair_scoring_plan(len(texts), len(label_names),
                                    float(np.mean([len(s_) for s_ in texts])))
        print(f"    pair-scoring: {plan['reason']}", flush=True)
        scorers = [("nli", NLI_MODEL)] + ([("rerank", RERANK_MODEL)] if with_reranker else [])
        for tag, mdl_name in (scorers if plan["run"] else []):
            F = _pair_scores(texts, label_names, task, mdl_name, tag,
                             shortlist_emb=_embed(texts, ENCODERS[0]))
            _collect(f"{tag}+logreg", lambda tr, te, F=F: (
                lambda m: (m.predict(F[te]), m.predict_proba(F[te]), m.classes_)
            )(LogisticRegression(max_iter=3000, class_weight="balanced").fit(F[tr], y[tr])))
    maj = int(np.bincount(y, minlength=int(max(classes.max(), y.max())) + 1).argmax())
    pred["constant-majority"] = np.full(len(gold), maj)
    conf["constant-majority"] = np.ones(len(gold))
    pos["constant-majority"] = np.zeros(len(gold))
    pbm = np.zeros((len(gold), len(classes)))
    if maj in col:                      # the teacher can favour a class gold lacks
        pbm[:, col[maj]] = 1.0
    proba["constant-majority"] = pbm
    return pred, conf, pos, proba, folds


def analyse(data: Dataset, seed: int = 0, drop=(), with_reranker: bool = False,
            train_on: str = "gold") -> dict:
    """One dataset in, one artifact out. The only entry point a runner needs."""
    task, texts, gold = data.name, data.texts, data.gold
    expert = data.expert
    label_names = data.label_names
    if train_on == "llm" and expert is None:
        raise ValueError(
            f"{task}: --train-on llm needs expert predictions, and this dataset "
            "carries none. The adapter decides whether a teacher exists.")
    groups_map = {t: i for i, t in enumerate(dict.fromkeys(texts.tolist()))}
    groups = np.array([groups_map[t] for t in texts])
    prof = CP.profile(texts, gold)
    fit_y = expert if train_on == "llm" else None
    pred, conf, pos, proba, folds = sweep(texts, gold, groups, label_names,
                                          seed=seed, task=task,
                                          with_reranker=with_reranker, fit_y=fit_y)
    for d in drop:
        for store in (pred, conf, pos, proba):
            store.pop(d, None)
    if drop:
        print(f"  ablation: dropped {list(drop)}", flush=True)

    model_only = {k: v for k, v in pred.items() if k != "constant-majority"}
    cmp_ = CP.compare_candidates(pred, gold, seed=seed)
    best = cmp_["best"]
    g3 = CP.roc_dominance({k: pos[k] for k in model_only}, gold)
    g4 = CP.pool_diversity(model_only, gold)
    # Without per-item expert predictions the cascade half of the pipeline has
    # nothing to measure, but the pool half does -- which is the whole question
    # on a dataset whose interest is its class count rather than its teacher.
    has_expert = expert is not None
    g5 = CP.class_advantage(pred[best], expert, gold) if has_expert else {
        "spread": float("nan"), "gates": "class-aware selection layer only",
        "verdict": "G5 n/a: no expert predictions for this dataset"}
    g7 = CP.base_rate_dominance(pred[best], gold)

    build_ensemble = (not g3.get("is_total_order", False)
                      and g4["mean_pairwise_disagreement"] >= 0.10
                      and (not has_expert or g5["spread"] >= 0.05)
                      and g7["min_precision"] >= 0.5)
    keep = [k for k in pred if k != "constant-majority"]
    layer = (CP.class_aware_layer({k: pred[k] for k in keep}, {k: conf[k] for k in keep},
                                  {k: proba[k] for k in keep}, gold, folds, seed=seed,
                                  fit_labels=fit_y)
             if build_ensemble else None)

    # The free arm we ship -- and therefore the one we compare against the paid
    # expert and run the coverage rule on -- is the winning construction when
    # one wins, otherwise the best single model.
    ds = CP.deslib_selection({k: pred[k] for k in keep}, {k: proba[k] for k in keep},
                             _embed(texts, ENCODERS[-1]), gold, folds,
                             fit_labels=fit_y, seed=seed)
    free_name = layer["winner"] if layer else best
    free_pred = layer["winner_pred"] if layer else pred[best]
    free_conf = layer["winner_conf"] if layer else conf[best]
    # Taken here, with its siblings, because the block below strips these keys
    # off `layer` before the artifact is built. Reading winner_proba after that
    # returned None and silently fell back to the best single model -- and only
    # on runs where a construction actually wins, which is exactly the run it
    # corrupts. ISEAR ships stacking, so every margin, committee and conformal
    # number there was computed from bge-large while free_pred was stacking's.
    winner_proba = np.asarray(layer["winner_proba"] if layer else proba[best])
    # A construction that is reported as a winner and then not shipped is the
    # bug that once put "stacking wins" three lines above a comparison of
    # bge-large. DESlib is a candidate for the free arm on the same terms as our
    # own layer: it must beat the best single model, survive Holm, AND clear the
    # resolution floor -- a Holm-surviving half-point under the floor is a hint,
    # and hints do not get shipped.
    ds_ship = None
    if ds.get("available"):
        elig = [(n, m) for n, m in ds["methods"].items()
                if "Oracle" not in n and m["delta_vs_best_single"] > 0
                and m.get("significant_after_holm")
                and m.get("clears_unpaired_floor")]
        if elig:
            ds_ship = max(elig, key=lambda kv: kv[1]["balanced"])
    if layer:
        # arrays are for the stages below, not for the artifact
        layer = {k: v for k, v in layer.items()
                 if k not in ("winner_pred", "winner_conf", "winner_proba")}
    provenance = {"source": data.source, "dataset_notes": list(data.notes)}
    _oracle_pred = ds.pop("oracle_pred", None)     # used below, not shipped
    if not has_expert:
        conf_block = CP.confusion(free_pred, gold, label_names)
        conf_block["configuration"] = f"{free_name}, no LLM (none available)"
        conf_block["expert_share"] = 0.0
        # No teacher, so the paired table is against the pool's own ceiling:
        # what SELECTION could win rather than what escalation could.
        orc = _oracle_pred if ds.get("available") else None
        conf_block["routing"] = (
            CP.routing_table(free_pred, np.asarray(orc), gold,
                             "the best pool member")
            if orc is not None else None)
        if conf_block["routing"]:
            _rt = conf_block["routing"]
            # No teacher, so no routing rule is shipped -- the free arm answers
            # everything. "captured 0%" would read as a rule that got nothing;
            # there is no rule. And the pool Oracle is right whenever ANY member
            # is, so the free-right/other-wrong cell is empty by construction
            # rather than by merit. Both are said, not implied.
            _rt["captured"] = None
            _rt["no_rule_reason"] = (
                "no LLM stream for this dataset, so nothing is routed: the free arm "
                "answers 100% of traffic")
            _rt["risk_cell_is_structural"] = True
        # The competence board needs no expert -- it is a property of the pool.
        # It was living inside the cascade half only by accident of where
        # class_router() computes it.
        cls_all = sorted({int(c) for c in gold.tolist()})
        board = CP.competence_board({k: pred[k] for k in keep}, gold,
                                    np.arange(len(gold)), cls_all)
        return {"task": task, "trained_on": train_on, "profile": prof.to_dict(),
                "dataset": data.describe(), **provenance,
                "confusion": conf_block, "competence_board": board,
                "label_names": label_names,
                "candidates": cmp_, "G3_roc_dominance": g3,
                "G4_pool_diversity": g4, "G5_class_advantage": g5,
                "G7_base_rate_dominance": g7, "class_aware_layer": layer,
                "free_arm_shipped": free_name, "deslib": ds, "expert": None,
                "recommendation": {
                    "best_free_model": best, "free_arm_shipped": free_name,
                    "build_class_aware_layer": bool(build_ensemble),
                    "deslib_ships": ds_ship[0] if ds_ship else None,
                    "why": "no expert predictions -- pool half only"}}
    # Signals for the rules that are not plain max-probability. The winning
    # construction hands back its own probability vector, so margin is taken on
    # the arm that actually ships rather than on whichever model happens to be
    # nearby.
    signals = CP.deferral_signals(winner_proba, free_pred,
                                  {k: pred[k] for k in keep}, free_conf)
    # The router sees what every other rule sees separately: the pool's full
    # probability vectors plus the derived signals. Its targets are the two arms'
    # own out-of-fold outcomes, fitted inside each training fold.
    feat = np.column_stack(
        [np.asarray(proba[n]).reshape(len(gold), -1) for n in keep]
        + [free_conf.reshape(-1, 1), signals["margin"].reshape(-1, 1),
           signals["committee"].reshape(-1, 1), free_pred.reshape(-1, 1)])
    signals["router_ctx"] = {
        "features": feat,
        "free_right": (free_pred == (fit_y if fit_y is not None else gold)).astype(int),
        "expert_right": (expert == (fit_y if fit_y is not None else gold)).astype(int),
        "kind": "hgb", "seed": seed}
    signals["conformal_ctx"] = {"proba": np.asarray(winner_proba),
                                "alpha": 0.10, "mondrian": True}
    # Does a conformal guarantee actually hold out of fold, and does making it
    # class-conditional fix the rare-class failure the empirical threshold had?
    conformal = {m: CP.conformal_coverage(np.asarray(winner_proba), gold, folds,
                                          alpha=0.10, mondrian=(m == "mondrian"))
                 for m in ("split", "mondrian")}
    vs = CP.versus_expert(free_pred, expert, gold, seed=seed)
    cov = [CP.guaranteed_coverage(free_pred, free_conf, gold, expert, bar, folds)
           for bar in (0.80, 0.90, 0.95)]
    # The same question with the bar anchored to the expert's own per-class
    # precision instead of to a fixed number. Both are reported because they are
    # different promises, and the comparison between them is the point: an
    # absolute bar overpays on classes the expert is bad at and underpays on
    # classes it is good at, and only side by side is that visible.
    marg = [CP.margin_coverage(free_pred, free_conf, gold, expert, m, folds)
            for m in (0.00, 0.05, 0.10, 0.20)]
    dial = CP.deferral_curve(free_pred, free_conf, expert, gold, folds, seed=seed,
                             fit_labels=fit_y, signals=signals, groups=groups)
    # One board over every free candidate, then a per-class decision on the
    # expert. Ungated on purpose: this is a routing rule, not a selection layer,
    # so it answers "which classes, if any" even where the gates say no.
    board = CP.class_router({k: pred[k] for k in keep}, expert, gold, folds,
                            fit_labels=fit_y)
    for k in ("pred", "free_pred"):
        board.pop(k, None)
    head = CP.per_class_headroom(free_pred, free_conf, expert, gold, folds,
                                 dial["best_point"]["expert_share"])
    # The matrix describes the configuration the Verdict quotes -- the nested
    # operating point -- not the free arm and not the expert. A confusion matrix
    # for a different arm than the headline number is worse than none.
    bp = dial["best_point"]
    _esc = np.asarray(bp.pop("escalated"))
    # Same reasoning for the per-class spend table: hand it the mask the nested
    # search produced, so it reports where the SHIPPED rule sent the money
    # rather than where a fixed favourite would have.
    _picks = [p_["rule"] for p_ in bp["picks_per_fold"]]
    split = CP.budget_split(free_pred, free_conf, expert, gold, folds,
                            bp["expert_share"], fit_labels=fit_y,
                            rule="/".join(sorted(set(_picks))), signals=signals,
                            escalated=_esc)
    conf_block = CP.confusion(np.asarray(bp.pop("pred")), gold, label_names)
    conf_block["configuration"] = (
        f"{free_name} + LLM on {bp['expert_share']:.1%} of traffic, "
        f"rule chosen per fold ({'/'.join(sorted({p['rule'] for p in bp['picks_per_fold']}))})")
    conf_block["expert_share"] = bp["expert_share"]
    conf_block["routing"] = CP.routing_table(free_pred, expert, gold, "the LLM",
                                             escalated=_esc)
    _rt = conf_block["routing"]
    _span = _rt["oracle_accuracy"] - _rt["free_accuracy"]
    # How much of the routing headroom the shipped dial actually took. The
    # ceiling is what a perfect router would score; without this the two numbers
    # sit next to each other and nobody does the division.
    _rt["captured"] = (float((conf_block["accuracy"] - _rt["free_accuracy"]) / _span)
                       if _span > 1e-12 else None)
    return {"task": task, "trained_on": train_on, "profile": prof.to_dict(),
            "dataset": data.describe(), **provenance,
            "confusion": conf_block,
            "label_names": label_names, "candidates": cmp_,
            "G3_roc_dominance": g3, "G4_pool_diversity": g4,
            "G5_class_advantage": g5, "G7_base_rate_dominance": g7,
            "class_aware_layer": layer, "free_arm_shipped": free_name,
            "class_router": board, "budget_split": split,
            "G8_per_class_headroom": head, "deslib": ds,
            "versus_expert": vs, "deferral_curve": dial,
            "guaranteed_coverage": cov, "margin_coverage": marg,
            "conformal": conformal,
            "recommendation": {
                "best_free_model": best,
                "free_arm_shipped": free_name,
                "build_class_aware_layer": bool(build_ensemble),
                "why": ("all gates open -- a class-aware layer has something to work with"
                        if build_ensemble else
                        "; ".join(v for v in (g3.get("verdict", ""), g4["verdict"],
                                              g5["verdict"], g7["verdict"]) if "FAIL" in v))}}


def _render_board(board, label_names=None) -> str:
    """P(correct | this model PREDICTS class c), one row per class.

    Transposed relative to the two-class version: with 77 intents the classes
    have to be rows or the table is unreadable. Sorted by how much the best and
    worst candidate differ on that class, because a class where every candidate
    scores the same is a class no selection rule can act on -- the spread is the
    quantity that decides whether selection has anything to do.
    """
    sh, sup, raw = board["shrunk"], board["support"], board["raw"]
    names = list(sh)
    classes = sorted(next(iter(sh.values())).keys())
    rows = []
    for c in classes:
        vals = [sh[m][c] for m in names]
        rows.append((float(max(vals) - min(vals)), c))
    rows.sort(reverse=True)
    L = ["\n  competence board -- P(correct | this model PREDICTS class c), "
         "by class",
         "    " + f"{'class':<30}{'n':>6}" + "".join(f"{m.split('+')[0][:11]:>12}"
                                                     for m in names)
         + f"{'spread':>9}{'best':>14}"]
    for spread, c in rows:
        nm = (label_names or {}).get(c, str(c))
        n_c = max(sup[m][c] for m in names)
        best = max(names, key=lambda m: sh[m][c])
        L.append("    " + f"{str(nm)[:29]:<30}{n_c:>6}"
                 + "".join(f"{sh[m][c]:>12.3f}" for m in names)
                 + f"{spread:>9.3f}{best.split('+')[0][:13]:>14}")
    wins = {m: sum(1 for c in classes
                   if max(names, key=lambda z: sh[z][c]) == m) for m in names}
    L.append("    classes owned (highest shrunk competence): "
             + ", ".join(f"{m.split('+')[0]} {v}" for m, v in
                         sorted(wins.items(), key=lambda kv: -kv[1])))
    sp = [s_ for s_, _ in rows]
    L.append(f"    spread across classes: median {float(np.median(sp)):.3f}, "
             f"max {max(sp):.3f}, min {min(sp):.3f}")
    # A candidate that never wins a class widens every spread without offering
    # anything to select. On banking77 the NLI row owns 0 of 77 and drags the
    # median from 0.029 to 0.101 -- a 3.5x overstatement of how much there is
    # for a selection rule to act on.
    live = [m for m in names if wins[m] > 0]
    if live and len(live) < len(names):
        sp2 = [max(sh[m][c] for m in live) - min(sh[m][c] for m in live)
               for c in classes]
        dead = [m.split("+")[0] for m in names if wins[m] == 0]
        L.append(f"    excluding {', '.join(dead)} (owns 0 classes): "
                 f"median {float(np.median(sp2)):.3f}, max {max(sp2):.3f}"
                 f"   <- the spread a selection rule can actually use")
    return "\n".join(L)


def render(r) -> str:
    p, c = r["profile"], r["candidates"]
    L = [f"\n{'='*78}", f"{r['task']} [trained on {r['trained_on']}]  "
         f"n={p['n']}  classes={p['n_classes']}  "
         f"imbalance 1:{p['imbalance_ratio']:.2f}", "=" * 78]
    for n in p["notes"]:
        L.append(f"  ! {n}")
    L.append(f"  headline metric: {p['headline_metric']}  "
             f"(resolution +/-{p['resolution_half_width']:.4f})")
    L.append(f"\n  candidates (balanced accuracy):")
    for row in c["ranking"]:
        tag = "  <- best" if row["name"] == c["best"] else ""
        L.append(f"    {row['name']:<28}{row['balanced']:.4f}{tag}")
    L.append(f"\n  {r['G3_roc_dominance'].get('verdict', 'G3 n/a (multiclass)')}")
    L.append(f"  {r['G4_pool_diversity']['verdict']}")
    L.append(f"    oracle {r['G4_pool_diversity']['oracle_balanced']:.4f} vs "
             f"independence null {r['G4_pool_diversity']['oracle_if_independent']:.4f}")
    L.append(f"  {r['G5_class_advantage']['verdict']}")
    L.append(f"  {r['G7_base_rate_dominance']['verdict']}")
    if r.get("class_aware_layer"):
        cl = r["class_aware_layer"]
        L.append(f"\n  class-aware layer (vs {cl['best_single']} "
                 f"{cl['best_single_balanced']:.4f}):")
        for k in ("selection_dcs_lca", "stacking"):
            d = cl[k]
            L.append(f"    {k:<20}{d['balanced']:.4f}  {d['delta_vs_best_single']:+.4f} "
                     f"CI [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] p={d['p']:.3f} "
                     f"holm={d['significant_after_holm']}")
        L.append(f"    -> {cl['verdict']}")
    ds = r.get("deslib")
    if ds and ds.get("available"):
        L.append(f"\n  DESlib on the same pool (baseline: {ds['baseline']} "
                 f"{ds['baseline_balanced']:.4f})")
        for nm, m in ds["methods"].items():
            tag = "" if "Oracle" in nm else (
                f"  holm={m['significant_after_holm']}"
                + ("" if m.get("clears_unpaired_floor", True)
                   else f"  [under the +/-{m['unpaired_floor']:.4f} unpaired floor]"))
            L.append(f"    {nm:<26}{m['balanced']:>8.4f}"
                     f"{m['delta_vs_best_single']:>+9.4f} "
                     f"CI [{m['ci'][0]:+.4f},{m['ci'][1]:+.4f}] p={m['p']:.3f}{tag}")
        L.append(f"    -> {ds['verdict']}")
        won = [n for n, m in ds["methods"].items()
               if "Oracle" not in n and m["delta_vs_best_single"] > 0
               and m.get("significant_after_holm")]
        shipped = (r.get("recommendation") or {}).get("deslib_ships")
        if won and not shipped:
            L.append(f"       not shipped: {', '.join(won)} beat the baseline on a "
                     f"paired test but stay inside the unpaired resolution floor")
    elif ds:
        L.append(f"\n  DESlib: {ds.get('reason')}")
    if r.get("expert", True) is None:
        b_ = r.get("competence_board")
        if b_:
            L.append(_render_board(b_, r.get("label_names")))
        L.append("\n  no expert predictions -- cascade stages skipped")
        L.append(f"  >> ship: {r['free_arm_shipped']}")
        return "\n".join(L)
    v = r["versus_expert"]
    L.append(f"\n  {r['free_arm_shipped']} vs expert LLM: {v['cheap_balanced']:.4f} vs "
             f"{v['expert_balanced']:.4f}  delta {v['balanced_delta']:+.4f} "
             f"CI [{v['balanced_ci'][0]:+.4f},{v['balanced_ci'][1]:+.4f}] p={v['balanced_p']:.3f}")
    L.append(f"  non-inferiority at 2pts: {v['non_inferiority']['verdict']}")
    d = r["deferral_curve"]
    L.append(f"\n  deferral dial -- balanced accuracy by share sent to the LLM")
    # every rule in RULES, not a hardcoded three -- a rule that is in the search
    # but not in the table is a rule nobody checks
    cols = [r for r in CP.RULES if r in d["points"][0]]
    L.append(f"    {'LLM share':>10}"
             + "".join(f"{c.replace('_','-'):>16}" for c in cols)
             + f"{'gain/+10% spend':>18}")
    for row in d["points"]:
        g = row["gain_per_10pct_spend"]
        L.append(f"    {row[cols[0]]['expert_share']:>10.1%}"
                 + "".join(f"{row[c]['balanced']:>16.4f}" for c in cols)
                 + f"{('  --' if g is None else f'{g:+.4f}'):>18}")
    # Two rules that score the same at every budget are the same rule here, and
    # saying so is cheaper than letting a reader treat them as two agreeing
    # pieces of evidence. On a binary task margin is 2*max - 1, a monotone
    # transform of confidence, so it cannot rank anything differently.
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if all(abs(r[cols[i]]["balanced"] - r[cols[j]]["balanced"]) < 1e-12
                   for r in d["points"]):
                why = (" (on 2 classes margin is a monotone transform of confidence,"
                       " so it cannot order items differently)"
                       if {cols[i], cols[j]} == {"confidence", "margin"} else "")
                L.append(f"    note: {cols[i]} and {cols[j]} are identical at every"
                         f" budget{why}")
    bo, bp = d["best_point_optimistic"], d["best_point"]
    L.append(f"    curve max (OPTIMISTIC -- chosen by looking at its own score, "
             f"{bo['n_configurations_selected_over']} configs): "
             f"{bo['balanced']:.4f} at {bo['expert_share']:.1%} via {bo['rule']}")
    rules = {p_['rule'] for p_ in bp['picks_per_fold']}
    L.append(f"    NESTED (rule+budget picked on training folds only): "
             f"{bp['balanced']:.4f} at {bp['expert_share']:.1%} LLM  "
             f"[selection bias {d['selection_bias']:+.4f}]")
    L.append(f"      per_class_conf admitted by train-fold G8 in "
             f"{bp.get('per_class_conf_admitted', 0)}/{len(bp['picks_per_fold'])} folds")
    L.append(f"      per-fold picks: {'/'.join(sorted(rules))}, "
             f"budgets {[p_['budget'] for p_ in bp['picks_per_fold']]}, "
             f"stable={bp['stable']}")
    L.append(f"    reference: free alone {d['free_alone']:.4f}, "
             f"LLM alone {d['expert_alone']:.4f}")
    ce = d.get("cheapest_equivalent")
    if ce and ce.get("found") and ce["expert_share"] < bp["expert_share"] - 1e-9:
        L.append(f"    CHEAPEST EQUIVALENT: {ce['note']}"
                 f"  -- {bp['expert_share']-ce['expert_share']:.0%} less traffic "
                 f"to the LLM for a difference this dataset cannot measure")
    cf = r.get("conformal")
    if cf:
        L.append("\n  conformal prediction sets at 90% target coverage")
        L.append(f"    {'variant':<10}{'marginal':>10}{'worst class':>13}"
                 f"{'classes under':>15}{'singletons':>12}{'set size':>10}")
        for nm in ("split", "mondrian"):
            c_ = cf[nm]
            L.append(f"    {nm:<10}{c_['marginal_coverage']:>10.4f}"
                     f"{c_['worst_class_coverage']:>13.4f}"
                     f"{len(c_['classes_failing']):>10}/{len(c_['per_class']):<4}"
                     f"{c_['singleton_share']:>12.1%}{c_['mean_set_size']:>10.2f}")
        sp, mo = cf["split"], cf["mondrian"]
        L.append(f"    -> {'Mondrian holds for every class' if mo['holds_everywhere'] else f'''Mondrian still misses {len(mo["classes_failing"])} class(es)'''}"
                 f"; split conformal misses {len(sp['classes_failing'])}"
                 f" while its marginal coverage reads {sp['marginal_coverage']:.4f}")
    h = r.get("G8_per_class_headroom")
    if h:
        L.append(f"    {h['verdict']}")
        L.append(f"      oracle global {h['oracle_global']:.4f} vs oracle "
                 f"per-class {h['oracle_per_class']:.4f}")
    bs = r.get("budget_split")
    if bs:
        L.append(f"    board+dial at a {bs['budget']:.0%} budget "
                 f"({bs['overall_share']:.1%} actually spent) -- per predicted class:")
        for c, row in bs["per_class"].items():
            L.append(f"      class {c}: {row['escalated']:>6.1%} of "
                     f"{row['items']:>6d} items escalated  "
                     f"({row['share_of_spend']:.0%} of the spend)")

    b_ = r["class_router"]
    bd, sup = b_["board"]["shrunk"], b_["board"]["support"]
    cls = sorted(next(iter(bd.values())).keys())
    L.append(f"\n  competence board -- P(correct | this model PREDICTS class c)")
    L.append("    " + f"{'candidate':<30}" + "".join(f"{('c'+str(c)):>9}" for c in cls))
    for m in bd:
        L.append("    " + f"{m:<30}"
                 + "".join(f"{bd[m][c]:>9.3f}" for c in cls))
    L.append("    " + f"{'(items claimed, all models)':<30}"
             + "".join(f"{sup[next(iter(sup))][c]:>9d}" for c in cls))
    for nm, row in b_["routable_rows"].items():
        L.append("    " + f"{'-> ' + nm:<30}" + "".join(f"{row[c]:>9.3f}" for c in cls))
    L.append(f"    free board (best claim wins the item): {b_['free_board_balanced']:.4f}"
             f"   +expert on {len(b_['classes_to_expert'])}/{len(cls)} classes: "
             f"{b_['balanced']:.4f}  at {b_['expert_share']:.1%} LLM")
    L.append(f"    classes handed to the LLM: {b_['classes_to_expert'] or 'none'}"
             f"   per fold {b_['picks_per_fold']}  stable={b_['stable']}")
    for tag, k in (("vs free board", "vs_free_board"), ("vs LLM alone", "vs_expert")):
        e = b_[k]
        L.append(f"    {tag}: {e['delta']:+.4f}"
                 + ("" if e["resolvable"] else f"  [{e['note']}]"))
    L.append(f"    beats BOTH ends by a resolvable margin? {d['beats_both_ends']}"
             f"  (vs free {d['gain_over_free']['delta']:+.4f}, "
             f"vs LLM {d['gain_over_expert']['delta']:+.4f}, "
             f"floor +/-{d['gain_over_free']['resolution_half_width']:.4f})")
    for k in ("gain_over_free", "gain_over_expert"):
        if d[k]["note"]:
            L.append(f"      ! {d[k]['note']}")
    L.append(f"\n  per-class guaranteed coverage (free share | guarantee holds):")
    for cv in r["guaranteed_coverage"]:
        L.append(f"    absolute bar {cv['bar']:.2f}  free {cv['free_share']:>6.1%}  "
                 f"balanced {cv['balanced']:.4f}  holds={cv['guarantee_holds']}")
    for cv in r.get("margin_coverage", []):
        L.append(f"    within {cv['margin']:.0%} of LLM   free {cv['free_share']:>6.1%}  "
                 f"balanced {cv['balanced']:.4f}  holds={cv['promise_holds']}"
                 f"  deferred classes {len(cv['deferred_classes'])}")
    rec = r["recommendation"]
    L.append(f"\n  >> ship: {rec['free_arm_shipped']}")
    L.append(f"  >> build class-aware layer: {rec['build_class_aware_layer']}")
    L.append(f"     {rec['why']}")
    return "\n".join(L)


