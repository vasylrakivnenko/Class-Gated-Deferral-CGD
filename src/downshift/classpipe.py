"""Class-aware AutoML: pick a free classifier, and decide whether anything
cleverer than "pick the best one" is worth building on this dataset.

Why this exists. A long run of one-off experiments on OCL's HateSpeech
benchmark produced a string of apparent wins that each dissolved on contact
with a check: raw accuracy beat a constant predictor, per-class routing beat
nothing, an ensemble tied within noise. Every one of those was catchable up
front by a cheap diagnostic. This module turns those diagnostics into gates
that run before the expensive stage, so the answer arrives in seconds instead
of a session.

The gates ARE the product. In order:

  G1 majority baseline   -- if a constant predictor beats the models on the
                            headline metric, that metric ranks nothing. Report
                            balanced accuracy instead, and say so.
  G2 duplicate texts     -- copies straddling a CV fold leak. Group them.
  G3 ROC dominance       -- if the candidate pool is a total order (each model
                            better than the next on EVERY class at EVERY matched
                            operating point) there is nothing to select between.
                            Ship the best single model and stop.
  G4 pool diversity      -- an oracle over K members inflates with K. Compare it
                            against the oracle you would get if members were
                            independent. Near-identical members => no headroom.
  G5 per-class spread    -- the class-aware SELECTION layer thresholds
                            (expert_acc - cheap_acc) per predicted class. If that
                            is flat, selection has nothing to act on. Scoped to
                            selection only: budget-constrained class-aware
                            ALLOCATION on the deferral dial is a different rule
                            and is reported whatever G5 says.
  G6 resolution          -- on an imbalanced set, balanced accuracy is limited by
                            the MINORITY count, not n. Differences below that
                            floor are not differences.
  G8 per-class headroom -- fit BOTH allocation rules on the held-out fold and
                            compare. If a per-class threshold cannot beat a
                            global one even with the answers in hand, no
                            estimator will, and the extra parameters are cost
                            with no ceiling to reach. Recomputed inside the
                            training folds when it gates rather than reports.
  G7 base-rate dominance -- P(correct | predicts c) per class. A class whose vote
                            is worse than a coin flip cannot be trusted by any
                            selection rule, however the rule is built. G3/G4/G5
                            all opened on a 1:7.95 dataset where every class-aware
                            rule then collapsed; this is what they were missing.

Separately from the gates, class_router() answers the operational question in
one table: score every candidate AND the paid expert per predicted class, let
each candidate's own claim compete, then hand over whole classes where the
expert is reliably better. Its output is a class list -- none of them, two of
seven, or all of them -- which is the form the decision actually takes. It is
ungated on purpose: it is a routing rule, not a selection layer, so it reports
even where the gates say a selection layer is pointless.

Only if G3/G4/G5/G7 open do we fit the class-aware layer, and it is then tested
against the best single model with a paired bootstrap and Holm correction --
because "beats the best single model" is the only comparison that justifies the
extra machinery. Separately, pair_scoring_plan() decides whether the expensive
pair-scoring candidates are worth running at all on a given dataset's size,
class count and document length.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .stats import holm_bonferroni, mcnemar_test, non_inferiority_test

# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def per_class_accuracy(pred: np.ndarray, gold: np.ndarray) -> dict[int, float]:
    """Accuracy within each TRUE class -- what de Gibert et al. report, and the
    only per-class view that a majority-class drifter cannot game."""
    return {int(c): float((pred[gold == c] == c).mean())
            for c in sorted(set(gold.tolist()))}


def balanced_accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    pc = per_class_accuracy(pred, gold)
    return float(np.mean(list(pc.values()))) if pc else float("nan")


def _class_boot(pred_a, pred_b, gold, n_boot=10_000, seed=0):
    """Paired bootstrap on the balanced-accuracy DIFFERENCE, resampling within
    each class. Stratifying matters: a plain item bootstrap under-weights the
    minority class, which is the class that sets this metric's precision.
    Kept here rather than in stats.py because it is class-metric specific.
    """
    classes = sorted(set(gold.tolist()))
    idx = [np.where(gold == c)[0] for c in classes]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        da = db = 0.0
        for ii in idx:
            s = rng.choice(ii, len(ii), replace=True)
            da += (pred_a[s] == gold[s]).mean()
            db += (pred_b[s] == gold[s]).mean()
        out[b] = (da - db) / len(idx)
    lo, hi = np.quantile(out, [0.025, 0.975])
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p))


def resolvable(delta: float, gold: np.ndarray) -> dict:
    """G6. Is a reported difference bigger than what this dataset can measure?

    Listed as a gate from the start and never enforced, which is how a set of
    "interior optimum beats both ends" claims got reported on gains of +0.0000,
    +0.0005, +0.0003 and +0.0022 against floors of 0.0102, 0.0057, 0.0062 and
    0.0032. A number the data cannot resolve is not a result, and nothing in the
    pipeline was saying so.
    """
    floor = resolution_floor(gold)
    ok = abs(delta) >= floor
    return {"delta": float(delta), "resolution_half_width": floor,
            "resolvable": bool(ok),
            "note": "" if ok else
                    f"G6: {delta:+.4f} is inside the +/-{floor:.4f} this dataset "
                    f"can resolve -- not a difference"}


def resolution_floor(gold: np.ndarray) -> float:
    """Half-width that balanced accuracy can resolve, set by the RAREST class.

    The mistake this prevents: quoting n=10,703 when the metric's precision is
    governed by the 1,196 minority items, then reporting a 0.15-point "win".
    """
    counts = [int((gold == c).sum()) for c in sorted(set(gold.tolist()))]
    return float(np.sqrt(0.25 / min(counts)) / np.sqrt(len(counts)))


# --------------------------------------------------------------------------
# G1 / G2 -- dataset profile
# --------------------------------------------------------------------------
@dataclass
class Profile:
    n: int
    n_classes: int
    class_counts: dict[int, int]
    majority_class: int
    majority_baseline: float
    imbalance_ratio: float
    duplicate_rows: int
    duplicate_share: float
    resolution_half_width: float
    headline_metric: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def profile(texts, gold: np.ndarray) -> Profile:
    gold = np.asarray(gold)
    counts = {int(c): int((gold == c).sum()) for c in sorted(set(gold.tolist()))}
    maj = max(counts, key=counts.get)
    base = counts[maj] / len(gold)
    seen: dict[str, int] = {}
    for t in texts:
        seen[t] = seen.get(t, 0) + 1
    dup = sum(v for v in seen.values() if v > 1)
    notes = []
    # G1: the check that would have caught the retracted HateSpeech claim.
    if base >= 0.65:
        notes.append(
            f"G1: a constant '{maj}' predictor scores {base:.4f} accuracy. Raw "
            f"accuracy ranks systems by closeness to degenerate prediction on this "
            f"dataset -- headline metric forced to balanced accuracy.")
    if dup:
        notes.append(f"G2: {dup} rows ({dup/len(gold):.2%}) share text with another "
                     f"row; CV is grouped so copies cannot straddle a fold.")
    return Profile(
        n=len(gold), n_classes=len(counts), class_counts=counts, majority_class=maj,
        majority_baseline=base,
        imbalance_ratio=max(counts.values()) / max(1, min(counts.values())),
        duplicate_rows=dup, duplicate_share=dup / len(gold),
        resolution_half_width=resolution_floor(gold),
        headline_metric="balanced_accuracy" if base >= 0.65 else "accuracy",
        notes=notes)


# --------------------------------------------------------------------------
# G3 / G4 -- is an ensemble even possible on this pool?
# --------------------------------------------------------------------------
def roc_dominance(prob: dict[str, np.ndarray], gold: np.ndarray,
                  rates=(0.10, 0.15, 0.20, 0.25, 0.30)) -> dict:
    """G3. At matched positive rate, does one model beat another on EVERY class?

    A per-class 'flip' between two models -- A better on the negative class, B on
    the positive -- is the signature of a threshold difference, not complementary
    competence, and no routing rule can monetise it. Only meaningful for binary.
    """
    names = list(prob)
    if len(set(gold.tolist())) != 2:
        return {"applicable": False, "reason": "binary only"}
    wins = {n: 0 for n in names}
    ordered = 0
    comparisons = 0
    for a in names:
        for b in names:
            if a >= b:
                continue
            dom_a = dom_b = True
            for r in rates:
                k = max(1, int(r * len(gold)))
                pa = (prob[a] >= np.sort(prob[a])[-k]).astype(int)
                pb = (prob[b] >= np.sort(prob[b])[-k]).astype(int)
                ca, cb = per_class_accuracy(pa, gold), per_class_accuracy(pb, gold)
                if not all(ca[c] >= cb[c] for c in ca):
                    dom_a = False
                if not all(cb[c] >= ca[c] for c in cb):
                    dom_b = False
            comparisons += 1
            # Count the PAIR once when it is ordered either way. Incrementing a
            # per-model counter double-counts a tie -- two identical candidates
            # left both dom_a and dom_b true, so a one-pair pool reported
            # "dominating_pairs 2 of 1" and the verdict read "some genuine
            # complementarity exists" for a duplicated model, the exact inverse.
            if dom_a or dom_b:
                ordered += 1
                if dom_a:
                    wins[a] += 1
                if dom_b:
                    wins[b] += 1
    total_pairs = len(names) * (len(names) - 1) // 2
    chain = ordered == total_pairs and total_pairs > 0
    return {"applicable": True, "total_pairs": total_pairs,
            "dominating_pairs": int(ordered), "is_total_order": bool(chain),
            "per_model_wins": {k: int(v) for k, v in wins.items()},
            "verdict": ("G3 FAIL: the pool is a total ROC order -- every per-class "
                        "difference is a threshold artifact. Ship the best single "
                        "model; selection cannot help."
                        if chain else
                        "G3 pass: at least one pair is not ROC-ordered, so some "
                        "genuine complementarity exists.")}


def _diversity_ratio(obs, null, P, gold, names):
    best = max(balanced_accuracy(P[k], gold) for k in range(len(names)))
    denom = null - best
    if denom <= 1e-9:
        return None            # an independent pool buys nothing over its best member
    return float((obs - best) / denom)


def pool_diversity(pred: dict[str, np.ndarray], gold: np.ndarray,
                   n_sim: int = 200, seed: int = 0) -> dict:
    """G4. Oracle over K members against the oracle K INDEPENDENT members would
    give. A high oracle means little if the members are near-duplicates."""
    names = list(pred)
    P = np.array([pred[n] for n in names])
    cor = P == gold[None, :]
    obs = balanced_accuracy(np.where(cor.any(axis=0), gold, -1), gold)
    # Permute WITHIN each class, not across the whole column. A global shuffle
    # gives every null member its overall accuracy uniformly, which on a rare
    # class makes it far stronger than any real member -- and balanced accuracy
    # is exactly the metric that rewards that, so the null saturated: 0.9824 to
    # 0.9999 on all five datasets, which is the tell. Measured on a synthetic
    # 7-class pool whose members genuinely are independent given the class, the
    # global null reads 0.9290 against an observed oracle of 0.8268 while the
    # stratified null reads 0.8242 -- it recovers the right answer. The gate
    # itself keys on disagreement, so this corrected a reported number rather
    # than a decision, but it is a number the README quoted.
    rng = np.random.default_rng(seed)
    classes = sorted(set(gold.tolist()))
    idx = {c: np.where(gold == c)[0] for c in classes}
    sims = []
    for _ in range(n_sim):
        Q = np.empty_like(cor)
        for k in range(len(names)):
            for c in classes:
                ii = idx[c]
                Q[k, ii] = rng.permutation(cor[k, ii])
        sims.append(balanced_accuracy(np.where(Q.any(axis=0), gold, -1), gold))
    null = float(np.mean(sims))
    dis = [float((P[a] != P[b]).mean()) for a in range(len(names))
           for b in range(a + 1, len(names))]
    mean_dis = float(np.mean(dis)) if dis else 0.0
    return {"oracle_balanced": obs, "oracle_if_independent": null,
            "mean_pairwise_disagreement": mean_dis,
            # Headroom the pool has beyond its best member, as a fraction of the
            # headroom an independent pool would have. The old form divided by
            # max(1e-9, null - obs + 1e-9), which clamps precisely when obs > null
            # -- members MORE complementary than independent, the good case -- and
            # returned 5e7 there. Reported as None instead: "no independent
            # reference left to divide by" is a statement; 50,000,000 is not.
            "diversity_ratio": _diversity_ratio(obs, null, P, gold, names),
            "verdict": ("G4 FAIL: members agree "
                        f"{1-mean_dis:.1%} of the time -- one classifier repeated. "
                        "Selection has almost nothing to choose between."
                        if mean_dis < 0.10 else
                        f"G4 pass: members disagree on {mean_dis:.1%} of items.")}


# --------------------------------------------------------------------------
# G5 -- does the class-aware rule have anything to act on?
# --------------------------------------------------------------------------
def class_advantage(cheap_pred: np.ndarray, expert_pred: np.ndarray,
                    gold: np.ndarray) -> dict:
    """G5. Per PREDICTED class, expert accuracy minus cheap accuracy -- the `adv`
    the class-aware gate thresholds. Flat across classes => the rule degenerates
    to all-or-nothing."""
    rows = {}
    for c in sorted(set(cheap_pred.tolist())):
        m = cheap_pred == c
        if not m.any():
            continue
        rows[int(c)] = {
            "share": float(m.mean()),
            "cheap_acc": float((cheap_pred[m] == gold[m]).mean()),
            "expert_acc": float((expert_pred[m] == gold[m]).mean()),
        }
        rows[int(c)]["advantage"] = rows[int(c)]["expert_acc"] - rows[int(c)]["cheap_acc"]
    adv = [r["advantage"] for r in rows.values()]
    spread = float(max(adv) - min(adv)) if adv else 0.0
    flips = sum(1 for a in adv if a < 0)
    return {"per_class": rows, "spread": spread, "classes_cheap_wins": flips,
            # Scope matters: this gates the class-aware SELECTION layer, which
            # needs a per-class signal to choose between models. It says nothing
            # about budget-constrained class-aware ALLOCATION on the deferral
            # dial, which is a different rule -- FEVER fails this gate and then
            # produces its best operating point via exactly that allocation.
            "gates": "class-aware selection layer only",
            "verdict": ("G5 FAIL: per-class advantage is flat "
                        f"(spread {spread:.4f}); a class-aware SELECTION layer has "
                        "nothing to act on. Budget-constrained class-aware "
                        "allocation is a separate rule, reported on the dial "
                        "regardless."
                        if spread < 0.05 else
                        f"G5 pass: spread {spread:.4f} across {len(adv)} classes, "
                        f"cheap arm wins {flips}.")}


def label_shortlist(text_emb: np.ndarray, label_emb: np.ndarray, m: int) -> np.ndarray:
    """Top-m candidate labels per item by bi-encoder cosine -- the 'retrieve'
    half of retrieve-then-rerank.

    Pair-scoring candidates (NLI, cross-encoder rerankers) cost one forward pass
    per (item, label), which is fine at k=2 and impossible at k=150: CLINC150
    would need ~3.4M passes. Shortlisting first turns that into m passes per
    item regardless of k, and the embeddings it needs are already computed for
    the bi-encoder candidates, so the retrieve half is free.

    Both inputs must be L2-normalised, so the dot product is cosine.
    """
    m = int(min(m, label_emb.shape[0]))
    sim = text_emb @ label_emb.T
    top = np.argpartition(-sim, m - 1, axis=1)[:, :m] if m < sim.shape[1] \
        else np.tile(np.arange(sim.shape[1]), (sim.shape[0], 1))
    # order within the shortlist so the output is deterministic
    rows = np.arange(len(top))[:, None]
    return np.take_along_axis(top, np.argsort(-sim[rows, top], axis=1), axis=1)


def pair_scoring_plan(n: int, k: int, mean_chars: float, max_token_units: int = 4_000_000,
                      shortlist_over_k: int = 10, shortlist_m: int = 5,
                      max_length: int = 256) -> dict:
    """Should the pair-scoring candidates run here, and with what shortlist?

    Cost is one forward pass per (item, label), so it scales with n*k AND with
    document length. Shortlisting fixes the k term; nothing fixes the length
    term. IMDB is the case that forced this: n=25,000 two-class long reviews is
    only 50,000 pairs -- comfortably under any pair-count limit -- but at 256
    tokens each it ran for over eight hours, and both pair-scoring candidates
    then scored below the frozen bi-encoder anyway.

    Budget is measured in token-units (pairs x truncated length) rather than
    pairs, because that is what actually costs time.
    """
    m = shortlist_m if k > shortlist_over_k else None
    eff_k = m or k
    pairs = n * eff_k
    tokens = int(min(mean_chars / 4, max_length))
    units = pairs * max(tokens, 1)
    ok = units <= max_token_units
    return {"run": bool(ok), "shortlist_m": m, "n_pairs": int(pairs),
            "token_units": int(units), "budget": int(max_token_units),
            "reason": (f"{pairs:,} pairs x ~{tokens} tokens = {units:,} units, "
                       + ("within" if ok else "over")
                       + f" the {max_token_units:,} budget"
                       + ("" if ok else " -- skipped; pair scorers are not worth "
                                        "hours on a long-document task"))}


def stratified_rung(gold: np.ndarray, groups: np.ndarray, frac: float,
                    seed: int = 0) -> np.ndarray:
    """Item indices for one successive-halving rung.

    Two constraints the naive `sample frac of rows` version breaks:
      * class proportions are preserved, because on an imbalanced set a plain
        subsample leaves so few minority items that the rung cannot rank
        candidates at all -- 20% of HateSpeech is ~239 hate items;
      * duplicate GROUPS stay whole, for the same reason folds are grouped.
    """
    if frac >= 1.0:
        return np.arange(len(gold))
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    gstart = np.searchsorted(groups[order], np.unique(groups))
    members = np.split(order, gstart[1:])
    # a group's class is the class most of its items carry
    gclass = np.array([np.bincount(gold[m]).argmax() for m in members])
    picked = []
    for c in np.unique(gclass):
        idx = np.where(gclass == c)[0]
        rng.shuffle(idx)
        take = max(1, int(round(frac * len(idx))))
        picked.extend(idx[:take])
    return np.sort(np.concatenate([members[i] for i in picked]))


def eliminate(pred: dict[str, np.ndarray], gold: np.ndarray, seed: int = 0,
              n_boot: int = 2000) -> dict:
    """Successive-halving elimination, conservative by design.

    Standard halving keeps the top half by point estimate. That is exactly the
    noise-driven decision the gates exist to prevent: on an early rung the
    interval is wide, so the eventual winner can be discarded on a difference
    the rung cannot see. Here a candidate is dropped only if its balanced-
    accuracy interval against the current leader lies ENTIRELY below zero.
    Ties survive to the next rung, so the schedule shrinks more slowly and never
    throws away something it could not actually distinguish.
    """
    names = sorted(pred)
    scores = {n: balanced_accuracy(pred[n], gold) for n in names}
    leader = max(names, key=lambda n: scores[n])
    detail, survivors = {}, [leader]
    for n in names:
        if n == leader:
            detail[n] = {"balanced": scores[n], "ci": [0.0, 0.0], "survives": True,
                         "reason": "leader"}
            continue
        lo, hi, _ = _class_boot(pred[n], pred[leader], gold, n_boot=n_boot, seed=seed)
        beaten = hi < 0
        detail[n] = {"balanced": scores[n], "ci": [lo, hi], "survives": not beaten,
                     "reason": "interval entirely below the leader" if beaten
                               else "indistinguishable from the leader"}
        if not beaten:
            survivors.append(n)
    return {"leader": leader, "survivors": sorted(survivors), "detail": detail,
            "n_in": len(names), "n_out": len(survivors)}


def base_rate_dominance(pred: np.ndarray, gold: np.ndarray) -> dict:
    """G7. P(correct | the model PREDICTS class c) -- the quantity a reliability-
    based rule actually consults, as opposed to per-class recall.

    Added after G3/G4/G5 all opened on a 1:7.95 dataset where every class-aware
    rule then collapsed. The reason was not pool structure, which is all those
    gates see. It was that a "hate" vote was right only 41% of the time while a
    "noHate" vote was right 95% of the time, so any rule that resolves
    disagreements by reliability picks the majority class almost every time and
    drifts to degenerate prediction.

    A class whose vote is more often wrong than right cannot be trusted by a
    selection rule, however the rule is built.
    """
    prec = {}
    for c in sorted(set(gold.tolist())):
        m = pred == c
        prec[int(c)] = float((gold[m] == c).mean()) if m.any() else float("nan")
    vals = [v for v in prec.values() if not np.isnan(v)]
    worst = min(vals) if vals else float("nan")
    return {"precision_by_predicted_class": prec, "min_precision": worst,
            "spread": float(max(vals) - worst) if vals else float("nan"),
            "verdict": ("G7 FAIL: a vote for the weakest class is right only "
                        f"{worst:.1%} of the time. Reliability-based selection will "
                        "avoid that class and drift to the majority."
                        if worst < 0.5 else
                        f"G7 pass: every class's vote is right more often than not "
                        f"(worst {worst:.1%}).")}


# --------------------------------------------------------------------------
# The competence board -- one table, every candidate x every predicted class
# --------------------------------------------------------------------------
def competence_board(pred: dict[str, np.ndarray], gold: np.ndarray,
                     idx: np.ndarray, classes: list[int],
                     prior_strength: float = 20.0) -> dict:
    """P(correct | this candidate PREDICTS class c), one cell per (model, class).

    Conditioned on the PREDICTED class, not the true one, because that is the
    only thing a router knows at decision time. Keying on recall instead --
    "how often does it catch class c" -- builds a rule you cannot actually run,
    since it needs the label to decide which label to ask for.

    Cells are shrunk toward the model's own overall accuracy with a
    `prior_strength` pseudo-count. Without it, picking a per-class winner is
    taking one maximum per class over K candidates: on a class with 100 items
    and +/-5 points of noise, the argmax is noise. Shrinkage makes a cell earn
    its deviation with support.
    """
    idx = np.asarray(idx)
    board, support, raw = {}, {}, {}
    for m, p in pred.items():
        pm, gm = p[idx], gold[idx]
        overall = float((pm == gm).mean()) if len(idx) else 0.0
        board[m], support[m], raw[m] = {}, {}, {}
        for c in classes:
            sel = pm == c
            n_c = int(sel.sum())
            hits = int((gm[sel] == c).sum())
            support[m][c] = n_c
            raw[m][c] = float(hits / n_c) if n_c else float("nan")
            denom = n_c + prior_strength
            # a class this model never predicts, with no prior to fall back on,
            # has no competence to report -- score it 0 rather than dividing by 0
            board[m][c] = float((hits + prior_strength * overall) / denom) \
                if denom > 0 else 0.0
    return {"shrunk": board, "raw": raw, "support": support}


def arbitrate(pred: dict[str, np.ndarray], board: dict[int, float],
              order: list[str], normalize: str = "class"
              ) -> tuple[np.ndarray, np.ndarray]:
    """Each candidate states a claim; the best-supported claim wins the item.

    This is what replaces a per-class winner table. "Model A owns class 3, model
    B owns class 5" leaves the item where A says 3 and B says 5 undecided, and
    the tie-break then needs its own machinery (pairwise conflict counts are
    k^2 cells, mostly empty). Scoring each model's own claim by its competence
    for the class it actually predicted resolves that item with the numbers
    already in the board, and reduces to the per-class winner whenever the
    models agree.

    `normalize="class"` centres each class column on its mean across candidates
    before comparing. Without it the contest is unfair in a specific way: a
    claim of "class 0" beats a claim of "class 1" whenever class 0 is simply
    the easier class for everyone, so the item goes to whoever guessed the easy
    class rather than to whoever is better. That maximises raw accuracy and
    drifts to the easy class -- measured on FEVER, raw claims scored 0.7001
    balanced against the best single model's 0.7349. Centring asks the question
    that was intended: which candidate is unusually good at the class it named?

    Returns (winning prediction, winning claim score).
    """
    names = list(order)
    n = len(pred[names[0]])
    top = max(max(int(c) for c in pred[m]) for m in names)
    centre = {}
    if normalize == "class":
        for c in board[names[0]]:
            centre[int(c)] = float(np.mean([board[m][c] for m in names]))
    score = np.empty((len(names), n))
    for i, m in enumerate(names):
        lut = np.zeros(top + 1)
        for c, v in board[m].items():
            if 0 <= int(c) <= top:
                lut[int(c)] = v - centre.get(int(c), 0.0)
        # tie-break toward the earlier (better overall) candidate
        score[i] = lut[pred[m]] - i * 1e-12
    win = score.argmax(axis=0)
    out = np.empty(n, dtype=int)
    for i, m in enumerate(names):
        sel = win == i
        out[sel] = pred[m][sel]
    return out, score.max(axis=0)


def class_router(pred: dict[str, np.ndarray], expert_pred: np.ndarray,
                 gold: np.ndarray, folds, budget: float = 1.0,
                 margin_z: float = 1.0, prior_strength: float = 20.0,
                 normalize: str = "class", fit_labels=None) -> dict:
    """One board over every free candidate, then a per-class LLM decision.

    Stage 1 arbitrates among the free candidates. Stage 2 asks, for each class
    the stage-1 winner predicts, whether the expert is reliably better on that
    class -- and if so hands the whole class over, cheapest gain first until the
    budget runs out.

    The expert is scored on `P(expert correct | the FREE winner predicted c)`,
    not on its own predicted class. Its own class is unknowable before paying
    for the call, so a board row keyed on it describes the expert but cannot
    route to it. Both are reported; only the first is used.

    Everything is fitted on training folds and applied out-of-fold, so the
    escalation set for an item is never chosen by a rule that saw it.

    `margin_z` no longer gates the decision -- the greedy step below selects on
    the headline metric directly, so a per-class paired z-test on raw accuracy
    would be testing a different quantity than the one being optimised. It is
    kept because the per-class `take` flag it produces is worth reading next to
    the greedy choice: where the two disagree, the class is one whose bucket
    accuracy and whose contribution to balanced accuracy point opposite ways,
    which is exactly the HateSpeech case. Stability across folds is the guard
    that replaced it, and it is reported.
    """
    gold = np.asarray(gold)
    y = gold if fit_labels is None else np.asarray(fit_labels)
    classes = sorted({int(c) for c in gold.tolist()})
    names = list(pred)

    final = np.empty(len(gold), dtype=int)
    free_only = np.empty(len(gold), dtype=int)
    escalated = np.zeros(len(gold), dtype=bool)
    per_fold = []

    for tr, te in folds:
        tr, te = np.asarray(tr), np.asarray(te)
        cb = competence_board(pred, y, tr, classes, prior_strength)
        order = sorted(names, key=lambda m: -float((pred[m][tr] == y[tr]).mean()))
        win_all, _ = arbitrate(pred, cb["shrunk"], order, normalize)
        free_only[te] = win_all[te]

        # paired per-class gain for the expert, measured on the training fold
        gains = {}
        for c in classes:
            sel = (win_all[tr] == c)
            n_c = int(sel.sum())
            if not n_c:
                gains[c] = {"gain": -1.0, "se": 0.0, "n": 0, "take": False}
                continue
            fw = win_all[tr][sel] == y[tr][sel]
            ew = expert_pred[tr][sel] == y[tr][sel]
            b = int((ew & ~fw).sum())      # expert saves it
            d = int((fw & ~ew).sum())      # expert breaks it
            gain = (b - d) / n_c
            se = float(np.sqrt(b + d)) / n_c
            gains[c] = {"gain": float(gain), "se": se, "n": n_c,
                        "share": float(n_c / len(tr)),
                        "free_acc": float(fw.mean()),
                        "expert_acc": float(ew.mean()),
                        "take": bool(gain > margin_z * se)}

        # Greedy on the HEADLINE metric, not on within-bucket accuracy.
        # Ranking by raw paired gain optimises accuracy inside the escalated
        # bucket, which is a different objective: on HateSpeech the expert is
        # +25.7 points on items the free arm calls class 1 (0.639 vs 0.382), so
        # a gain-ranked router hands the class over -- and loses 1.2 points of
        # balanced accuracy, because the escalation also shifts how often that
        # class is predicted at all. Same shape as the class-difficulty trap in
        # arbitrate(), one stage later.
        room = budget * len(te)
        taken: list[int] = []
        mask_tr = np.zeros(len(tr), dtype=bool)

        def _score(m):
            return balanced_accuracy(np.where(m, expert_pred[tr], win_all[tr]),
                                     y[tr])

        cur = _score(mask_tr)
        remaining = [c for c in classes if gains[c]["n"]]
        while remaining:
            best_c, best_s = None, cur
            for c in remaining:
                sc = _score(mask_tr | (win_all[tr] == c))
                if sc > best_s:
                    best_s, best_c = sc, c
            if best_c is None:
                break
            remaining.remove(best_c)
            pool = te[win_all[te] == best_c]
            if len(pool) == 0 or len(pool) > room:
                continue
            mask_tr |= (win_all[tr] == best_c)
            cur = best_s
            escalated[pool] = True
            room -= len(pool)
            taken.append(best_c)
        per_fold.append({"classes_to_expert": sorted(taken),
                         "gains": {int(k): v for k, v in gains.items()}})
        final[te] = np.where(escalated[te], expert_pred[te], win_all[te])

    picks = [tuple(f["classes_to_expert"]) for f in per_fold]
    # The two rows that decide, averaged over folds. Keyed on the class the FREE
    # winner named, which is the only thing known before paying for a call --
    # unlike the expert's own-class precision, which describes it but cannot
    # route to it. On ISEAR the two disagree in sign on class 1.
    routable = {"FREE winner | it says c": {}, "EXPERT | free says c": {}}
    for c in classes:
        fa = [f["gains"][c]["free_acc"] for f in per_fold if f["gains"][c]["n"]]
        ea = [f["gains"][c]["expert_acc"] for f in per_fold if f["gains"][c]["n"]]
        routable["FREE winner | it says c"][c] = float(np.mean(fa)) if fa else float("nan")
        routable["EXPERT | free says c"][c] = float(np.mean(ea)) if ea else float("nan")
    board_full = competence_board(
        {**pred, "EXPERT(own class)": expert_pred}, y,
        np.arange(len(gold)), classes, prior_strength)
    bal = balanced_accuracy(final, gold)
    return {
        "balanced": bal,
        "free_board_balanced": balanced_accuracy(free_only, gold),
        "expert_balanced": balanced_accuracy(expert_pred, gold),
        "expert_share": float(escalated.mean()),
        "classes_to_expert": sorted({c for f in per_fold
                                     for c in f["classes_to_expert"]}),
        "stable": len(set(picks)) == 1,
        "picks_per_fold": [list(p) for p in picks],
        "per_fold": per_fold,
        "board": board_full,
        "routable_rows": routable,
        "pred": final,
        "free_pred": free_only,
        "vs_free_board": resolvable(bal - balanced_accuracy(free_only, gold), gold),
        "vs_expert": resolvable(bal - balanced_accuracy(expert_pred, gold), gold),
    }


# --------------------------------------------------------------------------
# DESlib -- the classic dynamic-selection baselines, on our own pool
# --------------------------------------------------------------------------
def _deslib_shim():
    """DESlib 0.3.7 calls BaseEstimator._validate_data, removed in sklearn 1.7.

    Restored as a thin forwarder so the library's own LCA/OLA/KNORA-E run
    unmodified. The point is to be beaten by DESlib, not by a reimplementation
    of it.
    """
    import sklearn.base
    from sklearn.utils.validation import validate_data as _vd
    if not hasattr(sklearn.base.BaseEstimator, "_validate_data"):
        def _validate_data(self, X="no_validation", y="no_validation", **kw):
            return _vd(self, X=X, y=y, **kw)
        sklearn.base.BaseEstimator._validate_data = _validate_data


def _frozen_pool(pred, proba, classes):
    """Each candidate as an estimator DESlib can call, backed by the
    out-of-fold predictions the pipeline already produced.

    The alternative -- refitting every candidate inside one shared feature
    matrix -- would have to reduce TF-IDF and reshape the NLI scorer, so DESlib
    would be selecting among different models than the ones we ship. Freezing
    the real predictions keeps the pool identical to the pool under test.

    It also removes a fairness bug from the first attempt at this comparison.
    DESlib needs a DSEL for competence estimation, and carving it out of the
    training fold left the pool fitted on 2/3 of what the single-model baseline
    saw; the reported loss was partly that handicap. Here DSEL is only an index
    set over predictions that are already out-of-fold, so both sides see the
    same data.

    X carries the row index in its last column; a companion kNN ignores that
    column and measures neighbourhoods in embedding space instead.
    """
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _Frozen(BaseEstimator, ClassifierMixin):
        def __init__(self, name="", p=None, pb=None, cls=None):
            self.name, self.p, self.pb, self.cls = name, p, pb, cls

        def fit(self, X, y=None):
            self.classes_ = np.asarray(self.cls)
            return self

        def _rows(self, X):
            return np.asarray(X)[:, -1].astype(int)

        def predict(self, X):
            return self.p[self._rows(X)]

        def predict_proba(self, X):
            return self.pb[self._rows(X)]

    out = []
    for nm in pred:
        e = _Frozen(nm, pred[nm], proba[nm], classes)
        e.fit(np.zeros((1, 1)))
        out.append(e)
    return out


def _index_blind_knn(emb, k):
    from sklearn.neighbors import NearestNeighbors

    class _KNN:
        def __init__(self, n_neighbors=k, **kw):
            self.n_neighbors = n_neighbors

        def fit(self, X, y=None):
            self.idx_ = np.asarray(X)[:, -1].astype(int)
            n = min(self.n_neighbors, len(self.idx_))
            self._nn = NearestNeighbors(n_neighbors=n).fit(emb[self.idx_])
            return self

        def kneighbors(self, X, n_neighbors=None, return_distance=True):
            q = emb[np.asarray(X)[:, -1].astype(int)]
            n = min(n_neighbors or self.n_neighbors, len(self.idx_))
            d, i = self._nn.kneighbors(q, n, return_distance=True)
            return (d, i) if return_distance else i

    return _KNN


def deslib_selection(pred: dict[str, np.ndarray], proba: dict[str, np.ndarray],
                     region_emb: np.ndarray, gold: np.ndarray, folds,
                     fit_labels=None, k: int = 7, seed: int = 0,
                     n_boot: int = 4000) -> dict:
    """LCA, OLA, KNORA-E and the Oracle ceiling over our own candidate pool.

    Dynamic classifier selection is the classic form of the idea this project
    started from -- per input, trust the model most reliable in that region --
    and DESlib is its reference implementation. Beating our own per-class rule
    means nothing if the library's versions do better, so they run on the same
    pool, the same folds and the same data, and are tested against the best
    single candidate with a paired bootstrap and Holm correction.
    """
    try:
        _deslib_shim()
        from deslib.dcs import LCA, OLA
        from deslib.des import KNORAE
        from deslib.static import Oracle
    except ImportError as e:                                  # pragma: no cover
        return {"available": False, "reason": f"deslib not installed ({e})"}

    gold = np.asarray(gold)
    y = gold if fit_labels is None else np.asarray(fit_labels)
    classes = np.array(sorted(set(gold.tolist())))
    names = [n for n in pred if n != "constant-majority"]
    pool = _frozen_pool({n: pred[n] for n in names},
                        {n: proba[n] for n in names}, classes)
    X = np.arange(len(gold), dtype=float).reshape(-1, 1)
    KNN = _index_blind_knn(region_emb, k)

    out = {n: np.zeros(len(gold), dtype=int) for n in
           ("DESlib LCA", "DESlib OLA", "DESlib KNORA-E", "DESlib Oracle (ceiling)")}
    for tr, te in folds:
        tr, te = np.asarray(tr), np.asarray(te)
        for label, ctor in (("DESlib LCA", LCA), ("DESlib OLA", OLA),
                            ("DESlib KNORA-E", KNORAE)):
            m = ctor(pool_classifiers=pool, k=k, knn_classifier=KNN,
                     random_state=seed)
            m.fit(X[tr], y[tr])
            out[label][te] = m.predict(X[te])
        o = Oracle(pool_classifiers=pool)
        o.fit(X[tr], y[tr])
        out["DESlib Oracle (ceiling)"][te] = o.predict(X[te], gold[te])

    best = max(names, key=lambda n: balanced_accuracy(pred[n], gold))
    rows, pvals = {}, {}
    for label, p in out.items():
        lo, hi, pv = _class_boot(p, pred[best], gold, n_boot=n_boot, seed=seed)
        d = balanced_accuracy(p, gold) - balanced_accuracy(pred[best], gold)
        rows[label] = {"balanced": balanced_accuracy(p, gold),
                       "delta_vs_best_single": d, "ci": [lo, hi], "p": pv,
                       # A paired bootstrap resolves smaller effects than the
                       # unpaired floor G6 quotes -- that is why we pair. Both
                       # are reported because a Holm-surviving win that is still
                       # under the floor is a hint, not a result.
                       "clears_unpaired_floor": bool(abs(d) >= resolution_floor(gold)),
                       "unpaired_floor": resolution_floor(gold)}
        if "Oracle" not in label:
            pvals[label] = pv
    for label, ok in zip(pvals, holm_bonferroni(list(pvals.values()))):
        rows[label]["significant_after_holm"] = bool(ok)
    beat = [l for l, r in rows.items()
            if "Oracle" not in l and r["delta_vs_best_single"] > 0
            and r.get("significant_after_holm")]
    return {"available": True, "k": k, "baseline": best,
            "baseline_balanced": balanced_accuracy(pred[best], gold),
            # the Oracle's per-item answers, so a dataset with no teacher can
            # still be shown the paired table -- what SELECTION could win, in
            # place of what escalation could
            "oracle_pred": out["DESlib Oracle (ceiling)"],
            "methods": rows, "beats_best_single": sorted(beat),
            "verdict": (f"DESlib beats the best single model ({best}) with "
                        + ", ".join(beat) if beat else
                        f"no DESlib method beats the best single model ({best})")}


def class_aware_layer(pred: dict[str, np.ndarray], conf: dict[str, np.ndarray],
                      proba: dict[str, np.ndarray], gold: np.ndarray, folds,
                      seed: int = 0, fit_labels=None) -> dict:
    """Fit the layer the gates authorised, and test whether it earned its keep.

    Two constructions, both fitted out-of-fold:
      selection -- per (model, predicted class, confidence bin) reliability,
                   trust the vote with the highest estimated reliability. This is
                   DCS-LCA with the competence region defined by confidence
                   rather than a kNN neighbourhood; the classic version lives in
                   DESlib and is the baseline to beat.
      stacking  -- one logistic regression over every model's FULL per-class
                   probability vector. Subsumes per-class routing and cannot
                   make the base-rate error, because it learns from calibrated
                   scores. `conf` (max probability) is the right input for the
                   confidence bins above and the wrong one here: on a 7-class
                   task it tells the meta-learner how sure each model was while
                   hiding which class it picked, which lands at chance.

    Both are compared against the best single model. That is the only comparison
    that justifies the extra machinery -- beating the worst member is free.
    """
    from sklearn.linear_model import LogisticRegression

    y = gold if fit_labels is None else fit_labels
    names = sorted(pred)
    best = max(names, key=lambda n: balanced_accuracy(pred[n], gold))
    Pm = np.array([pred[n] for n in names])
    Cm = np.array([conf[n] for n in names])
    edges = np.array([0.0, .6, .7, .8, .9, 1.01])
    bin_of = lambda v: np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)

    n_cls = int(max(gold.max(), y.max())) + 1
    sel = np.zeros(len(gold), dtype=int)
    sel_conf = np.zeros(len(gold))
    sel_proba = np.zeros((len(gold), n_cls))
    for tr, te in folds:
        rel = np.zeros((len(names), int(max(gold.max(), y.max())) + 1, len(edges) - 1))
        for k in range(len(names)):
            for c in sorted(set(gold.tolist())):
                base = (y[tr][Pm[k][tr] == c] == c).mean() if (Pm[k][tr] == c).any() else 0.0
                for b in range(len(edges) - 1):
                    m = (Pm[k][tr] == c) & (bin_of(Cm[k][tr]) == b)
                    rel[k, c, b] = (y[tr][m] == c).mean() if m.sum() >= 25 else base
        bt = bin_of(Cm[:, te])
        scores = np.array([[rel[k, Pm[k][te][j], bt[k, j]] for j in range(len(te))]
                           for k in range(len(names))])
        chosen = np.argmax(scores, axis=0)
        sel[te] = Pm[chosen, te]
        sel_conf[te] = Cm[chosen, te]
        for j, t_ in enumerate(te):
            pb = np.asarray(proba[names[chosen[j]]][t_]).ravel()
            sel_proba[t_, :len(pb)] = pb

    X = np.column_stack([np.asarray(proba[n]).reshape(len(gold), -1)
                         for n in names])
    stack = np.zeros(len(gold), dtype=int)
    stack_conf = np.zeros(len(gold))
    stack_proba = np.zeros((len(gold), n_cls))
    for tr, te in folds:
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        pr_ = m.predict_proba(X[te])
        stack[te] = m.classes_[pr_.argmax(axis=1)]
        stack_conf[te] = pr_.max(axis=1)
        for i_, c_ in enumerate(m.classes_):
            stack_proba[te, int(c_)] = pr_[:, i_]

    out = {"best_single": best,
           "best_single_balanced": balanced_accuracy(pred[best], gold)}
    # Downstream stages compare the free arm against the paid expert. That has
    # to be whichever construction actually won, not the best single model, or
    # the pipeline under-reports its own result.
    built = {"selection_dcs_lca": (sel, sel_conf, sel_proba),
             "stacking": (stack, stack_conf, stack_proba)}
    ps = []
    for nm, p in (("selection_dcs_lca", sel), ("stacking", stack)):
        lo, hi, pv = _class_boot(p, pred[best], gold, seed=seed)
        out[nm] = {"balanced": balanced_accuracy(p, gold),
                   "per_class": per_class_accuracy(p, gold),
                   "delta_vs_best_single": balanced_accuracy(p, gold)
                                           - balanced_accuracy(pred[best], gold),
                   "ci": [lo, hi], "p": pv}
        ps.append(pv)
    for nm, keep in zip(("selection_dcs_lca", "stacking"), holm_bonferroni(ps)):
        out[nm]["significant_after_holm"] = bool(keep)
        out[nm]["resolution"] = resolvable(out[nm]["delta_vs_best_single"], gold)
    # Qualify first, then rank. Taking the higher-scoring construction and only
    # then asking whether IT cleared the bar lets a construction that scores
    # 0.0003 more and fails Holm veto one that scores less and passes -- the
    # pipeline would fall back to the best single model while a construction had
    # in fact earned its keep. Holm already controls the family-wise error rate
    # across both tests, so ranking the survivors is the coherent order.
    qualifies = lambda n: (out[n]["significant_after_holm"]
                           and out[n]["delta_vs_best_single"] > 0
                           and out[n]["resolution"]["resolvable"])
    ok = [n for n in ("selection_dcs_lca", "stacking") if qualifies(n)]
    won = bool(ok)
    winner = (max(ok, key=lambda n: out[n]["balanced"]) if ok
              else max(("selection_dcs_lca", "stacking"),
                       key=lambda n: out[n]["balanced"]))
    out["winner"] = winner if won else best
    out["winner_is_construction"] = bool(won)
    out["winner_pred"] = built[winner][0] if won else pred[best]
    out["winner_conf"] = built[winner][1] if won else conf[best]
    out["winner_proba"] = built[winner][2] if won else np.asarray(proba[best])
    out["verdict"] = (
        f"{winner} beats the best single model by "
        f"{out[winner]['delta_vs_best_single']:+.4f} and survives Holm"
        if won else f"no construction beats {best}; ship the single model")
    return out


# --------------------------------------------------------------------------
# free arm versus a paid expert
# --------------------------------------------------------------------------
def versus_expert(cheap_pred: np.ndarray, expert_pred: np.ndarray, gold: np.ndarray,
                  margin: float = 0.02, seed: int = 0) -> dict:
    """Paired comparison plus the non-inferiority test, which is the question the
    dial actually answers: can the free arm serve this traffic?"""
    lo, hi, p = _class_boot(cheap_pred, expert_pred, gold, seed=seed)
    mc = mcnemar_test(cheap_pred == gold, expert_pred == gold, seed=seed)
    ni = non_inferiority_test(cheap_pred == gold, expert_pred == gold,
                              margin=margin, seed=seed)
    return {"cheap_balanced": balanced_accuracy(cheap_pred, gold),
            "expert_balanced": balanced_accuracy(expert_pred, gold),
            "balanced_delta": balanced_accuracy(cheap_pred, gold)
                              - balanced_accuracy(expert_pred, gold),
            "balanced_ci": [lo, hi], "balanced_p": p,
            "mcnemar_accuracy": mc.to_dict(), "non_inferiority": ni.to_dict()}


def _allocate(tr, te, rule: str, frac: float, free_pred, free_conf, expert_pred,
              gold, fit_labels=None, signals=None) -> np.ndarray:
    """Escalation mask over `te`. Every threshold is fitted on `tr` only.

    `fit_labels` is the signal the rule is allowed to learn from. With gold it is
    gold. In OCL's setting there is no gold at run time, so it is the expert's
    own annotations -- and then per-class 'advantage' degenerates into per-class
    DISAGREEMENT, because the expert is right by definition. That degeneracy is
    the honest consequence of the setting, not a bug in the rule.
    """
    y = gold if fit_labels is None else fit_labels
    esc = np.zeros(len(gold), dtype=bool)
    # Before the budget guards, deliberately. `frac` is a spend for every other
    # rule and a precision MARGIN for this one, so the guards read backwards
    # here: margin 0 is the strictest promise and escalates the most, margin 1
    # drops the bar to zero and escalates nothing. Routing it through
    # `if frac <= 0: return esc` would silently turn the strictest setting into
    # a no-op -- the one configuration whose whole purpose is to spend.
    if rule == "precision_floor":
        return _precision_floor(tr, te, frac, free_pred, free_conf, expert_pred,
                                y, esc)
    if frac <= 0:
        return esc
    if frac >= 1:
        esc[te] = True
        return esc
    if rule == "router":
        ctx = (signals or {}).get("router_ctx")
        if ctx is None:
            return esc
        sc = router_scores(ctx["features"], ctx["free_right"], ctx["expert_right"],
                           tr, te, ctx.get("kind", "hgb"), ctx.get("seed", 0))
        thr = np.quantile(sc[te], frac) if len(te) else 0.0
        esc[te[sc[te] <= thr]] = True
        return esc
    if rule == "conformal":
        ctx = (signals or {}).get("conformal_ctx")
        if ctx is None:
            return esc
        mem, _ = conformal_sets(ctx["proba"], y, tr, te,
                                alpha=ctx.get("alpha", 0.10),
                                mondrian=ctx.get("mondrian", True))
        # a bigger prediction set is a more ambiguous item; confidence orders
        # within a set size, which is otherwise far too coarse to threshold
        size = mem.sum(axis=1).astype(float)
        k = mem.shape[1]
        sc = (k - size) + np.asarray(free_conf, dtype=float)
        thr = np.quantile(sc[te], frac) if len(te) else 0.0
        esc[te[sc[te] <= thr]] = True
        return esc
    if rule in ("confidence", "margin", "committee"):
        # one mechanism, three signals: escalate the lowest-scoring share, with
        # the threshold taken on the training fold only
        score = free_conf if rule == "confidence" else (signals or {}).get(rule)
        if score is None:
            return esc
        score = np.asarray(score, dtype=float)
        thr = np.quantile(score[tr], frac)
        esc[te[score[te] <= thr]] = True
        return esc
    if rule == "per_class_conf":
        return _per_class_conf(tr, te, frac, free_pred, free_conf, expert_pred,
                               y, esc)
    adv = {}
    for c in sorted(set(free_pred[tr].tolist())):
        m = free_pred[tr] == c
        adv[c] = ((expert_pred[tr][m] == y[tr][m]).mean()
                  - (free_pred[tr][m] == y[tr][m]).mean()) if m.any() else -1.0
    budget = frac * len(te)
    for c in sorted(adv, key=lambda k: -adv[k]):
        if budget <= 0 or adv[c] <= 0:
            break
        pool = te[free_pred[te] == c]
        if len(pool) <= budget:                          # whole class fits
            esc[pool] = True
            budget -= len(pool)
        else:                                            # part of it, worst first
            esc[pool[np.argsort(free_conf[pool])[:int(budget)]]] = True
            budget = 0
    return esc


def _free_reliability(conf: np.ndarray, correct: np.ndarray,
                      cls: np.ndarray | None = None,
                      prior_strength: float = 20.0):
    """P(the free arm is right | its confidence, the class it named), as monotone
    curves shrunk toward one global curve.

    Per class, not global, and the reason is measurable rather than a matter of
    taste. On the synthetic benchmark used to size this rule, P(free correct |
    conf) inside a single confidence band runs from 0.388 to 0.810 across seven
    classes -- because even with an identical conf-given-correctness likelihood,
    the posterior moves with each class's own base rate. That spread IS the
    signal a per-class threshold exists to exploit. A global curve throws it
    away and leaves the rule strictly worse than the dial it was meant to beat,
    which is exactly what the first two attempts at this measured.

    Each class curve is pulled toward the global one by `prior_strength`
    pseudo-counts, so a class with 40 items borrows almost everything and a
    class with 4,000 borrows almost nothing. Isotonic because reliability rising
    with confidence is a real prior that costs nothing to impose.
    """
    from sklearn.isotonic import IsotonicRegression

    def _fit(x, yy):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(x, yy.astype(float))
        return iso

    glob = _fit(conf, correct)
    if cls is None:
        return lambda q, c=None: glob.predict(q)
    per, n_by = {}, {}
    for c in sorted(set(cls.tolist())):
        m = cls == c
        n_by[int(c)] = int(m.sum())
        per[int(c)] = _fit(conf[m], correct[m]) if m.sum() >= 2 else None

    def predict(q, c):
        g = glob.predict(q)
        iso, n_c = per.get(int(c)), n_by.get(int(c), 0)
        if iso is None:
            return g
        w = n_c / (n_c + prior_strength)
        return w * iso.predict(q) + (1.0 - w) * g

    return predict


_REL_CACHE: dict = {}


def _rel_cache_key(tr, prior_strength, *arrays):
    """Content-addressed, not index-addressed.

    Keying on the fold indices alone is wrong: the same `tr` can be handed a
    different free arm -- another candidate, another ablation, `--train-on llm`
    swapping the fit labels -- and the cached curves would silently belong to
    the wrong data. Hashing the arrays is O(n) memcmp-speed against work that is
    O(n log n) per class, so correctness here is nearly free.
    """
    return (np.asarray(tr).tobytes(), float(prior_strength),
            tuple(np.ascontiguousarray(a).tobytes() for a in arrays))


_CPM_CACHE: dict = {}


def _cpm_curves(tr, free_pred, free_conf, expert_pred, y, min_support: int):
    """Everything CPM needs from a training fold that does NOT depend on the
    margin: per class, the items sorted best-confidence-first, the running
    precision down that order, and the expert's own precision on the class.

    Split out because the margin is the swept parameter. Recomputing the sort
    and the cumulative precision once per margin would repeat O(n log n) work
    eight times per fold for an answer that cannot change.
    """
    ckey = _rel_cache_key(tr, min_support, free_pred, free_conf, expert_pred, y)
    hit = _CPM_CACHE.get(ckey)
    if hit is not None:
        return hit
    pooled = float((expert_pred[tr] == y[tr]).mean())
    curves = {}
    for c in sorted(set(free_pred[tr].tolist())):
        pos = np.where(free_pred[tr] == c)[0]
        if len(pos) == 0:
            continue
        order = pos[np.argsort(-free_conf[tr][pos], kind="stable")]
        n = np.arange(1, len(order) + 1)
        running = np.cumsum(y[tr][order] == c) / n
        # The anchor is the expert's PRECISION on this class -- P(truth is c |
        # the expert says c) -- not its accuracy on the slice the free arm calls
        # c. Branch precision and backbone precision have to be the same
        # quantity or the comparison is between two different questions.
        em = expert_pred[tr] == c
        anchor = float((y[tr][em] == c).mean()) if em.any() else pooled
        curves[c] = (order, running, n, anchor)
    hit = (curves, pooled)
    if len(_CPM_CACHE) > 256:
        _CPM_CACHE.clear()
    _CPM_CACHE[ckey] = hit
    return hit


def _precision_floor(tr, te, margin, free_pred, free_conf, expert_pred, y, esc,
                     min_support: int = 30):
    """Class Precision Margin: keep what clears a bar set BY THE EXPERT'S OWN
    precision on that class, escalate the rest. The share is an output.

    Every other rule on the dial is handed a budget and asked where to spend it.
    This one is handed a quality promise -- "no class may fall more than
    `margin` below what the paid model achieves on it" -- and the spend is
    whatever keeping that promise costs. On a dataset where the free arm is
    already as good as the expert the promise is free and nothing escalates; on
    one where it is not, the spend rises on its own.

    That is not a cosmetic reframing. `guaranteed_coverage` asks the same
    question against an ABSOLUTE bar, and an absolute bar is wrong in both
    directions at once. On HateSpeech the free arm scores 0.8460 against the
    expert's 0.8331, yet a 0.90 bar hands 22% of traffic to the weaker model and
    takes the result down to 0.8302 -- paying to get worse, because the bar has
    no idea what the expert can do. In the other direction a 0.90 bar on a class
    where the expert reaches 0.99 keeps free predictions that are nine points
    behind the thing they are replacing.

    Anchoring per class fixes both, and it cannot ask for the impossible: a bar
    of (1 - margin) x expert precision is by construction reachable whenever the
    free arm matches the expert, so a class is never declared vacuous merely for
    being hard.

    Under `--train-on llm` the labels are the expert's own, so its precision is
    1.0 on every class and the bar degenerates to the absolute (1 - margin).
    Same degeneracy the class-advantage rules carry in that setting, same
    reason, and it is reported rather than hidden.
    """
    curves, _ = _cpm_curves(tr, free_pred, free_conf, expert_pred, y, min_support)
    for c, (order, running, n, anchor) in curves.items():
        bar = (1.0 - margin) * anchor
        # The LARGEST prefix that still clears the bar -- equivalently the
        # smallest threshold, which is what CPM asks for: maximise what the free
        # arm owns subject to the per-class promise.
        ok = np.where((running >= bar) & (n >= min_support))[0]
        pool = te[free_pred[te] == c]
        if len(ok) == 0:
            esc[pool] = True                 # nothing on this class clears it
            continue
        thr = float(free_conf[tr][order[ok[-1]]])
        esc[pool[free_conf[pool] < thr]] = True
    # A class the free arm never predicts on `tr` has no threshold to apply; it
    # goes to the expert rather than being owned on no evidence.
    unseen = np.setdiff1d(np.unique(free_pred[te]), np.fromiter(curves, dtype=int))
    if len(unseen):
        esc[te[np.isin(free_pred[te], unseen)]] = True
    return esc


def _per_class_conf(tr, te, frac, free_pred, free_conf, expert_pred, y, esc,
                    prior_strength: float = 20.0):
    """The board and the dial combined: a separate confidence threshold per
    predicted class, with the budget allocated across classes by marginal value.

    The two rules it generalises are its own corner cases. Give every class the
    same threshold and it is the confidence dial; drive every threshold to 0 or
    1 and it is whole-class hand-over. Neither corner is usually optimal: the
    board cannot say "the worst tenth of class 0", and the dial cannot say
    "class 3 is worth twice as much per call as class 5".

    The value of escalating one item is split into a part that varies by class
    and a part that does not:

        v(i) = ( expert_right[c] - free_right(conf_i) ) * W[c]

    `expert_right[c]` is the board's shrunk P(expert correct | the free arm says
    c). `free_right(q, c)` is a per-class isotonic curve shrunk toward the global
    one. `W[c]` is the mean balanced-accuracy weight of an item predicted c.

    Both terms are per class, and the second one is where the signal actually
    lives: a global free-reliability curve measured 0.388 to 0.810 across seven
    classes inside one confidence band, and using it cost more than the noise it
    was meant to remove.

    The first version scored chunks by their REALISED paired outcome on the
    training fold, which is K noisy estimates rather than one, and it lost
    out-of-fold at matched spend on synthetic data built so that per-class
    thresholds should win: -0.0021 at K=7 and -0.0146 at K=20, scaling with the
    class count rather than with n. Shrinking that realised value toward the
    pooled value was a no-op for an arithmetic reason -- with equal chunk sizes
    (sum + a*g)/(chunk + a) is monotone in sum, so the argmax cannot move. The
    fix had to change what is estimated, not how hard it is pulled.
    """
    classes = sorted(set(free_pred[tr].tolist()))
    labels = sorted(set(y[tr].tolist()))
    K = len(labels)
    tot = {t: max(1, int((y[tr] == t).sum())) for t in labels}

    # Everything below depends on `tr` alone, not on the budget, so the eight
    # budget points of a dial share one fit. Without this the per-class isotonic
    # curves are refitted 8x per fold and the four-dataset run goes from about
    # three minutes to over half an hour -- and a gate you run less often is a
    # worse gate.
    ckey = _rel_cache_key(tr, prior_strength, free_pred, free_conf,
                          expert_pred, y)
    cached = _REL_CACHE.get(ckey)
    if cached is None:
        free_ok = (free_pred[tr] == y[tr])
        exp_ok = (expert_pred[tr] == y[tr])
        pooled = float(exp_ok.mean())
        rel = _free_reliability(free_conf[tr], free_ok, free_pred[tr], prior_strength)
        expert_right, W = {}, {}
        item_w = np.array([1.0 / (K * tot[int(t)]) for t in y[tr]])
        for c in classes:
            m = free_pred[tr] == c
            n_c = int(m.sum())
            expert_right[c] = float((int(exp_ok[m].sum()) + prior_strength * pooled)
                                    / (n_c + prior_strength))
            W[c] = float(item_w[m].mean()) if n_c else 0.0
        order = {}
        value = np.zeros(len(tr))
        for c in classes:
            pos = np.where(free_pred[tr] == c)[0]
            order[c] = pos[np.argsort(free_conf[tr][pos], kind="stable")]
            value[pos] = (expert_right[c]
                          - np.asarray(rel(free_conf[tr][pos], c))) * W[c]
        if len(_REL_CACHE) > 256:
            _REL_CACHE.clear()
        _REL_CACHE[ckey] = (order, value)
        cached = (order, value)
    order, value = cached
    taken = {c: 0 for c in classes}

    chunk = max(1, int(0.02 * len(tr)))
    room = int(frac * len(tr))
    while room > 0:
        best_c, best_v = None, 0.0
        for c in classes:
            nxt = order[c][taken[c]:taken[c] + min(chunk, room)]
            if len(nxt) == 0:
                continue
            v = float(value[nxt].mean())          # modelled value per call
            if v > best_v:
                best_v, best_c = v, c
        if best_c is None:                        # nothing left worth buying
            break
        step = len(order[best_c][taken[best_c]:taken[best_c] + min(chunk, room)])
        taken[best_c] += step
        room -= step

    # turn each class's count into a confidence threshold, then apply it to `te`
    for c in classes:
        k = taken[c]
        if k == 0:
            continue
        thr = float(free_conf[tr][order[c][k - 1]])
        pool = te[(free_pred[te] == c) & (free_conf[te] <= thr)]
        esc[pool] = True
    return esc


def conformal_sets(proba: np.ndarray, gold: np.ndarray, tr, te, alpha: float = 0.10,
                   mondrian: bool = False) -> tuple[np.ndarray, dict]:
    """Split conformal prediction sets, optionally class-conditional (Mondrian).

    A threshold picked because 0.7 looked about right is a number someone chose.
    Conformal derives it instead: score every calibration item by how badly the
    model handled it (`1 - p(true label)`), take the quantile that corresponds to
    the error rate you asked for, and a label is in the set when it scores under
    it. Under exchangeability the true label is in the set at least `1 - alpha`
    of the time -- whether or not the model is calibrated, and with a
    finite-sample correction rather than asymptotically.

    The plain version guarantees that rate ON AVERAGE, which is exactly how a
    rare class gets quietly sacrificed to hold a headline number. `mondrian=True`
    calibrates a separate threshold per class, so the guarantee holds FOR EVERY
    CLASS. That is the machinery this project has been approximating with an
    empirical per-class threshold that does not survive out of fold -- HateSpeech
    realised 0.866 against a 0.90 bar.

    Testing candidate label c against class c's own threshold is what makes
    Mondrian runnable: it conditions on the label being *considered*, never on
    the true label, which is not available when the decision is made.

    Returns a boolean (n_items x n_classes) membership matrix and the thresholds.
    """
    proba = np.asarray(proba, dtype=float)
    gold = np.asarray(gold)
    tr, te = np.asarray(tr), np.asarray(te)
    k = proba.shape[1]
    # nonconformity of the TRUE label, on the calibration split
    cal = 1.0 - proba[tr, gold[tr]]

    def _q(scores):
        n = len(scores)
        if n == 0:
            return 1.0
        # the (n+1) correction is what turns an asymptotic statement into a
        # finite-sample one; without it small calibration sets under-cover
        lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(scores, lvl, method="higher"))

    if mondrian:
        qhat = {}
        for c in range(k):
            m = gold[tr] == c
            qhat[c] = _q(cal[m]) if m.any() else _q(cal)
        thr = np.array([qhat[c] for c in range(k)])
    else:
        thr = np.full(k, _q(cal))

    members = np.zeros((len(gold), k), dtype=bool)
    members[te] = (1.0 - proba[te]) <= thr[None, :]
    return members, {"alpha": alpha, "mondrian": mondrian,
                     "thresholds": thr.tolist()}


def conformal_coverage(proba: np.ndarray, gold: np.ndarray, folds,
                       alpha: float = 0.10, mondrian: bool = False) -> dict:
    """Does the guarantee actually hold, out of fold, for every class?

    The point of running this rather than trusting the theory: the theory is
    conditional on exchangeability, our folds are GROUPED to keep duplicate texts
    together, and a rare class can have a calibration slice too small for the
    finite-sample correction to rescue. So the coverage is measured, per class,
    on held-out items -- which is where the previous empirical threshold failed.
    """
    proba = np.asarray(proba, dtype=float)
    gold = np.asarray(gold)
    k = proba.shape[1]
    members = np.zeros((len(gold), k), dtype=bool)
    for tr, te in folds:
        m, _ = conformal_sets(proba, gold, tr, te, alpha, mondrian)
        members |= m
    covered = members[np.arange(len(gold)), gold]
    sizes = members.sum(axis=1)
    per_class = {}
    for c in range(k):
        sel = gold == c
        if not sel.any():
            continue
        per_class[int(c)] = {"n": int(sel.sum()),
                             "coverage": float(covered[sel].mean()),
                             "mean_set_size": float(sizes[sel].mean()),
                             "holds": bool(covered[sel].mean() >= 1 - alpha)}
    worst = min(per_class.values(), key=lambda r: r["coverage"]) if per_class else None
    singles = sizes == 1
    return {
        "alpha": alpha, "mondrian": mondrian, "target": 1 - alpha,
        "marginal_coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_share": float(singles.mean()),
        # The set's single member, NOT the model's argmax. Under Mondrian the
        # threshold is per class, so the one label that clears its own bar is
        # routinely not the highest-scoring one -- on a skewed 3-class fixture
        # they differ on 32 of 39 singletons. Reading argmax here would report
        # the accuracy of a decision the conformal layer never makes.
        "accuracy_on_singletons": float((members[singles].argmax(axis=1)
                                         == gold[singles]).mean()) if singles.any() else float("nan"),
        "per_class": per_class,
        "classes_failing": sorted(c for c, r in per_class.items() if not r["holds"]),
        "worst_class_coverage": worst["coverage"] if worst else float("nan"),
        "holds_everywhere": all(r["holds"] for r in per_class.values()) if per_class else False,
    }


_ROUTER_CACHE: dict = {}


def router_scores(features: np.ndarray, free_right: np.ndarray,
                  expert_right: np.ndarray, tr, te, kind: str = "hgb",
                  seed: int = 0) -> np.ndarray:
    """Learn who will be right, instead of guessing from a proxy.

    Every other rule on the dial is a stand-in for "the expert will do better on
    this item": confidence, margin, disagreement, class membership. This
    estimates it. Two models, `P(free arm right | x)` and `P(expert right | x)`,
    and the score is the difference -- low means the expert is the better bet.

    Gradient boosting rather than the two logistic regressions the research brief
    described, because the signal is an interaction: a low-confidence item in a
    class the expert is bad at should stay, and a fairly confident item in a
    class the expert dominates should go. A linear model in these features
    cannot represent that, and it is exactly the shape the whole per-class line
    of work has been circling. `HistGradientBoosting` rather than XGBoost only to
    avoid a dependency -- it is the same family and ships with scikit-learn.

    Fitted on `tr`, applied to `te`, and cached on the content of `tr` because
    the fit does not depend on the budget -- eight budget points otherwise refit
    the same pair of models eight times.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    tr, te = np.asarray(tr), np.asarray(te)
    key = (kind, int(seed), np.asarray(tr).tobytes(),
           np.ascontiguousarray(features).tobytes(),
           np.asarray(free_right).tobytes(), np.asarray(expert_right).tobytes())
    hit = _ROUTER_CACHE.get(key)
    if hit is None:
        def _fit(y):
            yy = np.asarray(y)[tr].astype(int)
            if len(set(yy.tolist())) < 2:          # one arm is perfect on this fold
                return None, float(yy.mean())
            m = (HistGradientBoostingClassifier(
                    max_depth=3, max_iter=120, learning_rate=0.1,
                    l2_regularization=1.0, random_state=seed)
                 if kind == "hgb" else
                 LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(features[tr], yy)
            return m, None
        hit = (_fit(free_right), _fit(expert_right))
        if len(_ROUTER_CACHE) > 64:
            _ROUTER_CACHE.clear()
        _ROUTER_CACHE[key] = hit
    (mf, cf), (me, ce) = hit
    pf = np.full(len(te), cf) if mf is None else mf.predict_proba(features[te])[:, 1]
    pe = np.full(len(te), ce) if me is None else me.predict_proba(features[te])[:, 1]
    out = np.zeros(len(free_right))
    out[te] = pf - pe                     # low = the expert is the better bet
    return out


def deferral_signals(winner_proba: np.ndarray, winner_pred: np.ndarray,
                     pool_pred: dict[str, np.ndarray],
                     winner_conf: np.ndarray) -> dict[str, np.ndarray]:
    """Per-item scores for the rules that are not plain max-probability.

    Both come back oriented like confidence -- LOW means escalate first -- so
    `_allocate` thresholds them identically and the dial stays one mechanism.

    margin      p1 - p2 on the shipped arm's own probability vector. "How sure"
                and "how much surer than the runner-up" are different questions:
                a 0.55/0.44 call and a 0.55/0.15/0.15/0.15 call have the same
                max probability and nothing else in common. Margin sampling is
                one of the two policies Cache & Distil uses, so without it the
                dial was being compared against a baseline it did not contain.

    committee   how many pool members disagree with the shipped arm, tie-broken
                by its confidence -- the query-by-committee signal, the other
                Cache & Distil policy. The integer count alone is far too coarse
                to threshold: at four members most items sit at zero
                disagreement, and a 5% budget would escalate every one of them.
                Scoring `(K - n_disagree) + confidence` keeps disagreement
                dominant and lets confidence order within it, which is the
                tie-break the research brief already specified.
    """
    P = np.asarray(winner_proba)
    if P.ndim == 2 and P.shape[1] >= 2:
        top2 = np.sort(P, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
    else:                                        # nothing to take a margin over
        margin = np.asarray(winner_conf, dtype=float)
    members = [p for n, p in pool_pred.items() if n != "constant-majority"]
    if members:
        n_dis = np.sum([np.asarray(p) != np.asarray(winner_pred) for p in members],
                       axis=0).astype(float)
        committee = (len(members) - n_dis) + np.asarray(winner_conf, dtype=float)
    else:
        committee = np.asarray(winner_conf, dtype=float)
    return {"margin": margin, "committee": committee}


# Measured, not assumed. `conformal` and `router` were on this list for one full
# run across five datasets and came off it:
#
#   conformal  a near-duplicate of confidence as a ROUTING signal -- it won
#              HateSpeech's nested pick in all five folds and scored 0.0005 less
#              than confidence would have. Its value is the per-class coverage
#              guarantee, which conformal_coverage() reports and which does not
#              need a seat on the dial.
#   router     P(free right) - P(expert right) from a gradient-boosted pair. Lost
#              to plain confidence at every budget on HateSpeech (0.8483 against
#              0.8538 at 5%), and where it did win a nested pick it destabilised
#              it -- FEVER's selection bias went +0.0009 -> +0.0037 and IMDB lost
#              a stable 10% operating point for budgets scattered over 0.2-0.5.
#
# Neither changed any score by more than its dataset's resolution floor, while
# the two of them took the four-dataset run from 165s to 324s. Both remain
# implemented and tested; they are simply not in the search.
RULES = ("confidence", "margin", "committee", "class_aware", "per_class_conf",
         "precision_floor")

# What the swept parameter means for each rule. Every rule but one reads it as a
# budget -- the share of traffic to escalate. `precision_floor` reads it as the
# fraction of the expert's per-class precision it is willing to give up, and its
# share comes out the other end. Recorded so a pick of 0.20 is never read as a
# 20% spend when it means a 20% quality give-away.
PARAM_KIND = {r: ("margin" if r == "precision_floor" else "budget") for r in RULES}


def nested_best_operating_point(free_pred, free_conf, expert_pred, gold, folds,
                                fractions, inner_k: int = 3, seed: int = 0,
                                fit_labels=None, signals=None, groups=None) -> dict:
    """The honest answer to "what should we deploy, and what will it score?".

    Reporting max(rule, budget) over the out-of-fold curve selects among 16
    configurations and then scores the winner on the same data it was chosen
    with. Thresholds inside each rule were already fitted on training folds, so
    the curve itself is sound -- but the pick of WHICH point is not, and the
    resulting headline is biased upward. RESEARCH_BRIEF.md 7 records the same
    trap against this project's own gate table ("the headline is a max over
    gates").

    Here the choice of rule and budget is made by an inner CV inside each outer
    fold's training data, then applied once to that fold's held-out items. The
    number this returns is what running the PROCEDURE on new data would give,
    which is lower than the max of the curve and is the one to quote.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    # The OUTER folds are grouped because duplicate texts share a group id (gate
    # G2). An ungrouped inner split undoes that exactly where the configuration
    # is chosen: a text's twin sits in inner-train while it is scored in
    # inner-test, so a rule that rewards memorised items looks better than it is.
    # The outer score stays honest either way; the PICK does not.
    def _inner(tr):
        if groups is None:
            return list(StratifiedKFold(inner_k, shuffle=True,
                                        random_state=seed).split(tr, gold[tr]))
        return list(StratifiedGroupKFold(inner_k, shuffle=True, random_state=seed)
                    .split(tr, gold[tr], np.asarray(groups)[tr]))

    # What the inner search is allowed to score itself against. Under --train-on
    # llm there is no gold at run time, so selecting the operating point on gold
    # would hand the cascade information OCL's setting does not have and inflate
    # every cell of that comparison. Evaluation is still against gold; only this
    # choice is made blind.
    sel_y = gold if fit_labels is None else np.asarray(fit_labels)

    esc_all = np.zeros(len(gold), dtype=bool)
    picks, admitted = [], []
    for tr, te in folds:
        inner = _inner(tr)
        # G8, computed INSIDE this fold's training data. The oracle version in
        # per_class_headroom() reads held-out labels, which is fine for a report
        # and fatal for a gate -- it would pick the rule using the very items it
        # is about to be scored on. Here the headroom that admits per_class_conf
        # to the search is measured on training folds only.
        inner_folds = [(tr[a], tr[b]) for a, b in inner]
        adm = list(RULES)
        if "per_class_conf" in adm:
            h = per_class_headroom(free_pred, free_conf, expert_pred, gold,
                                   inner_folds, frac=0.20)
            if not h["worth_it"]:
                adm.remove("per_class_conf")
        best_cfg, best_score = (adm[0], 0.0), -np.inf
        for rule in adm:
            for frac in fractions:
                esc_in = np.zeros(len(gold), dtype=bool)
                for itr, ite in inner:
                    esc_in |= _allocate(tr[itr], tr[ite], rule, frac, free_pred,
                                        free_conf, expert_pred, gold, fit_labels,
                                        signals)
                p = np.where(esc_in, expert_pred, free_pred)
                sc = balanced_accuracy(p[tr], sel_y[tr])
                if sc > best_score:
                    best_score, best_cfg = sc, (rule, frac)
        picks.append(best_cfg)
        admitted.append(sorted(adm))
        esc_all |= _allocate(tr, te, best_cfg[0], best_cfg[1], free_pred, free_conf,
                             expert_pred, gold, fit_labels, signals)
    final = np.where(esc_all, expert_pred, free_pred)
    chosen = [r for r, _ in picks]
    return {"balanced": balanced_accuracy(final, gold),
            # the caller needs these to report a confusion matrix for the
            # configuration the headline actually quotes, rather than for some
            # other arm that happens to be lying around
            "pred": final, "escalated": esc_all,
            "expert_share": float(esc_all.mean()),
            "rules_admitted_per_fold": admitted,
            "per_class_conf_admitted": sum(
                1 for a in admitted if "per_class_conf" in a),
            # `budget` is kept under its old name so nothing downstream
            # breaks, but `param_kind` says what the number is: a share of
            # traffic for five of the rules, a precision give-away for
            # precision_floor, whose spend is an output rather than an input.
            "picks_per_fold": [{"rule": r, "budget": f,
                                "param_kind": PARAM_KIND.get(r, "budget")}
                               for r, f in picks],
            "rule_agreement": max(chosen.count(r) for r in RULES) / len(chosen),
            "stable": len(set(picks)) == 1}


def per_class_headroom(free_pred, free_conf, expert_pred, gold, folds,
                       frac: float = 0.20, n_null: int = 3, seed: int = 0) -> dict:
    """G8. Is there anything for a PER-CLASS threshold to win, at this budget?

    Both rules are fitted on the held-out fold itself -- an oracle neither can
    be built as -- so the gap is the most a per-class rule could ever capture,
    with estimation error removed. If the oracle gap is zero or negative, no
    estimator fixes it and the extra parameters are pure cost.

    This exists because three attempts to "fix" the per-class rule were made
    before anyone measured whether there was anything to fix. On the synthetic
    benchmark it was tuned against, the oracle gap is +0.0021 at K=7 and
    -0.0073 at K=20 -- so the rule was being asked to capture headroom that was
    not there, and at K=20 to beat a global sort the greedy cannot even express.
    Chunked per-class allocation is not a superset of a global threshold at
    intermediate budgets, only at 0% and 100%.
    """
    def _global(idx):
        k = int(frac * len(idx))
        e = np.zeros(len(gold), dtype=bool)
        e[idx[np.argsort(free_conf[idx], kind="stable")[:k]]] = True
        return e

    def _per_class(idx, group=None):
        grp = free_pred if group is None else group
        e = np.zeros(len(gold), dtype=bool)
        labels = sorted(set(gold[idx].tolist()))
        K = len(labels)
        tot = {t: max(1, int((gold[idx] == t).sum())) for t in labels}
        w = np.array([(float(expert_pred[j] == gold[j]) - float(free_pred[j] == gold[j]))
                      / (K * tot[int(gold[j])]) for j in idx])
        order, taken = {}, {}
        for c in sorted(set(grp[idx].tolist())):
            pos = np.where(grp[idx] == c)[0]
            order[c] = pos[np.argsort(free_conf[idx][pos], kind="stable")]
            taken[c] = 0
        chunk = max(1, int(0.02 * len(idx)))
        room = int(frac * len(idx))
        while room > 0:
            best_c, best_v = None, 0.0
            for c in order:
                nxt = order[c][taken[c]:taken[c] + min(chunk, room)]
                if len(nxt) == 0:
                    continue
                v = float(w[nxt].mean())
                if v > best_v:
                    best_v, best_c = v, c
            if best_c is None:
                break
            step = len(order[best_c][taken[best_c]:taken[best_c] + min(chunk, room)])
            taken[best_c] += step
            room -= step
        for c in order:
            if taken[c]:
                e[idx[order[c][:taken[c]]]] = True
        return e

    eg = np.zeros(len(gold), dtype=bool)
    ep = np.zeros(len(gold), dtype=bool)
    covered = []
    for _, te in folds:
        eg |= _global(te)
        ep |= _per_class(te)
        covered.append(np.asarray(te))
    # Score ONLY the items these folds actually cover. When the gate runs inside
    # a training fold the folds span `tr`, not the whole dataset, and the
    # uncovered items score identically under both arms -- which dilutes the gap
    # toward zero and, measured on IMDB, let the rule in on 4 of 5 folds while
    # the full-data oracle said there was no headroom at all.
    cov = np.unique(np.concatenate(covered)) if covered else np.arange(len(gold))
    g = balanced_accuracy(np.where(eg, expert_pred, free_pred)[cov], gold[cov])
    p = balanced_accuracy(np.where(ep, expert_pred, free_pred)[cov], gold[cov])
    gap = p - g
    floor = resolution_floor(gold[cov])

    # The per-class arm fits K thresholds on the very fold it is scored on while
    # the global arm fits one, so part of any gap is just the wider search
    # winning on noise -- and that optimism grows as the fold shrinks. Measured:
    # on IMDB the full-data oracle says +0.0005 against a +/-0.0032 floor, while
    # the same statistic computed inside training folds admitted the rule in 5
    # of 5. Same shape as the bias nested_best_operating_point() exists to
    # remove, so it gets the same treatment G4 gives its oracle: a null.
    #
    # The null keeps every threshold and every budget, and destroys only the
    # class structure, by permuting which group each item belongs to.
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(max(1, n_null)):
        shuffled = free_pred.copy()
        rng.shuffle(shuffled)
        en = np.zeros(len(gold), dtype=bool)
        for _, te in folds:
            en |= _per_class(np.asarray(te), group=shuffled)
        nulls.append(balanced_accuracy(
            np.where(en, expert_pred, free_pred)[cov], gold[cov]) - g)
    null_gap = float(np.mean(nulls))
    excess = gap - null_gap
    return {"budget": float(frac), "oracle_global": g, "oracle_per_class": p,
            "headroom": float(gap), "null_headroom": null_gap,
            "headroom_over_null": float(excess),
            "resolution_half_width": floor, "items_scored": int(len(cov)),
            "worth_it": bool(excess >= floor),
            "verdict": (f"G8 pass: a per-class threshold gains {gap:+.4f} at a "
                        f"{frac:.0%} budget with the answers in hand, {excess:+.4f} "
                        f"of it beyond what shuffled classes buy, above the "
                        f"+/-{floor:.4f} floor."
                        if excess >= floor else
                        f"G8 FAIL: with the answers in hand a per-class threshold "
                        f"gains {gap:+.4f} at a {frac:.0%} budget, but shuffled "
                        f"classes buy {null_gap:+.4f} of that -- only {excess:+.4f} "
                        f"is real, inside the +/-{floor:.4f} this dataset resolves. "
                        f"No estimator recovers headroom that is not there.")}


def confusion(pred: np.ndarray, gold: np.ndarray, label_names=None,
              top_confusions: int = 8) -> dict:
    """Counts, plus the off-diagonal cells worth reading.

    Balanced accuracy says how much is wrong; this says what it is wrong ABOUT,
    which is the part that decides whether an error matters. A moderation
    classifier that misses hate and one that over-flags safe text score the same
    and are not the same product.

    `top_confusions` is there because a 77-class matrix is 5,929 cells and the
    eye cannot rank them. Off-diagonal mass is listed largest first, as a share
    of the true class, so "this class leaks into that one" is readable without
    hunting the grid.
    """
    classes = sorted({int(c) for c in np.concatenate([gold, pred]).tolist()})
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    m = np.zeros((k, k), dtype=int)
    for g, p in zip(gold.tolist(), pred.tolist()):
        m[idx[int(g)], idx[int(p)]] += 1
    support = m.sum(axis=1)
    name = (lambda c: str((label_names or {}).get(c, c)))
    pairs = []
    for i, ti in enumerate(classes):
        for j, pj in enumerate(classes):
            if i == j or not m[i, j]:
                continue
            pairs.append({"true": ti, "pred": pj, "n": int(m[i, j]),
                          "share_of_true": float(m[i, j] / max(1, support[i])),
                          "label": f"{name(ti)} \u2192 {name(pj)}"})
    pairs.sort(key=lambda r: -r["n"])
    return {"classes": classes, "matrix": m.tolist(),
            "support": support.tolist(),
            "recall": [float(m[i, i] / s) if s else float("nan")
                       for i, s in enumerate(support)],
            "precision": [float(m[i, i] / m[:, i].sum()) if m[:, i].sum() else float("nan")
                          for i in range(k)],
            "accuracy": float(np.trace(m) / max(1, m.sum())),
            "balanced": balanced_accuracy(pred, gold),
            "top_confusions": pairs[:top_confusions],
            "off_diagonal": int(m.sum() - np.trace(m))}


def routing_table(free_pred: np.ndarray, other_pred: np.ndarray,
                  gold: np.ndarray, other_name: str = "LLM",
                  escalated: np.ndarray | None = None) -> dict:
    """The 2x2 that decides whether routing can buy anything: who gets it right.

    A k x k class confusion says what a single model is wrong about. It does not
    say what a CASCADE can do, and at 77 classes it is 5,929 cells nobody reads.
    This is the paired table instead -- McNemar's table -- and it is the same
    shape whether the dataset has two classes or a hundred:

                      other right     other wrong
        free right    no action       escalating LOSES this
        free wrong    escalating WINS both wrong, no rule helps

    Only one cell is the prize. `free wrong & other right` is the whole ceiling
    on routing, and `free right & other wrong` is what a rule pays for being
    wrong about which items to send. The difference between them is the most any
    router can add, before a single threshold is chosen.

    Pass `escalated` -- the shipped rule's own decision per item -- and each cell
    also reports what the rule did with it. That is the difference between a
    ceiling and a diagnosis: the prize cell says how much was winnable, the share
    beside it says how much the rule actually went and got.
    """
    gold = np.asarray(gold)
    fr = free_pred == gold
    orr = other_pred == gold
    n = len(gold)
    masks = {"both_right": fr & orr, "free_only_right": fr & ~orr,
             "other_only_right": ~fr & orr, "neither_right": ~fr & ~orr}
    both, free_only, other_only, neither = (int(m.sum()) for m in masks.values())
    sent = {}
    if escalated is not None:
        escalated = np.asarray(escalated, dtype=bool)
        for k, m in masks.items():
            c = int(m.sum())
            sent[k] = {"n_sent": int((m & escalated).sum()),
                       "share_sent": float((m & escalated).sum() / c) if c else 0.0}
    return {
        "other_name": other_name, "n": n, "sent": sent or None,
        "both_right": both, "free_only_right": free_only,
        "other_only_right": other_only, "neither_right": neither,
        "free_accuracy": float(fr.mean()), "other_accuracy": float(orr.mean()),
        # the ceiling: keep every item the free arm gets right, and take every
        # item only the other arm gets right
        "oracle_accuracy": float((both + free_only + other_only) / n),
        "headroom": float(other_only / n),
        "at_risk": float(free_only / n),
        "net_ceiling": float((other_only - free_only) / n),
        "unreachable": float(neither / n),
        "verdict": (
            f"routing can add at most {other_only/n:.1%} (items only {other_name} gets "
            f"right) and risks {free_only/n:.1%} (items only the free arm gets right); "
            f"{neither/n:.1%} is out of reach for both"),
    }


def cheapest_equivalent(points, best_balanced: float, gold: np.ndarray,
                        rules=RULES) -> dict:
    """The least LLM spend whose score is within the noise floor of the best.

    Reporting the argmax of the dial answers "what is the highest number" and
    never asks "is anything cheaper just as good". On FEVER the argmax is 0.8192
    at 75% of traffic while 50% scores 0.8177 -- a gap of 0.0015 against a
    +/-0.0062 floor, so a third of the bill buys a difference the dataset cannot
    measure. G6 already refuses to call that a difference when comparing
    systems; it should refuse just as firmly when comparing operating points.
    """
    floor = resolution_floor(gold)
    best = None
    for row in points:
        for r in rules:
            if r not in row:
                continue
            cell = row[r]
            if best_balanced - cell["balanced"] <= floor:
                cand = (cell["expert_share"], -cell["balanced"], r)
                if best is None or cand < best:
                    best = cand
    if best is None:
        return {"found": False}
    share, negbal, rule = best
    bal = -negbal
    return {"found": True, "expert_share": float(share), "balanced": float(bal),
            "rule": rule, "resolution_half_width": floor,
            "gives_up": float(best_balanced - bal),
            "note": (f"{share:.1%} of traffic scores {bal:.4f} via {rule}, "
                     f"{best_balanced - bal:.4f} BELOW the best point and inside "
                     f"the +/-{floor:.4f} this dataset resolves")}


def budget_split(free_pred, free_conf, expert_pred, gold, folds, frac,
                 fit_labels=None, rule: str = "per_class_conf",
                 signals=None, escalated=None) -> dict:
    """Where the SHIPPED rule actually sent the money, per predicted class.

    The board's whole appeal was that you could read the decision off it. A
    per-class threshold keeps that, at finer grain: not "class 3 goes to the
    LLM" but "the least-confident 34% of class 3 does, and class 1 gets
    nothing".

    `rule` has to be the one the nested search picked, not a fixed favourite.
    Hardcoding per_class_conf meant that on every dataset where committee or
    margin won -- IMDB and HateSpeech among them -- this table described the
    spend of a rule nobody was going to deploy, under a heading claiming
    otherwise.
    """
    if escalated is not None:
        # The mask the nested search actually produced, where each fold applied
        # its own pick. Re-deriving it from one rule at one budget would give a
        # different set of items than the configuration being reported.
        esc = np.asarray(escalated, dtype=bool)
    else:
        esc = np.zeros(len(gold), dtype=bool)
        for tr, te in folds:
            esc |= _allocate(tr, te, rule, frac, free_pred, free_conf,
                             expert_pred, gold, fit_labels, signals)
    rows = {}
    for c in sorted(set(free_pred.tolist())):
        m = free_pred == c
        rows[int(c)] = {"items": int(m.sum()), "escalated": float(esc[m].mean()),
                        "share_of_spend": float(esc[m].sum() / max(1, esc.sum()))}
    return {"budget": float(frac), "rule": rule,
            "overall_share": float(esc.mean()), "per_class": rows}


def deferral_curve(free_pred: np.ndarray, free_conf: np.ndarray,
                   expert_pred: np.ndarray, gold: np.ndarray, folds,
                   fractions=(0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0),
                   seed: int = 0, fit_labels=None, signals=None,
                   groups=None) -> dict:
    """Accuracy as a function of how much traffic the paid expert sees.

    The precision-bar rule answers "what can the free arm own at 90% precision".
    That is a different question from "we have budget for 10% of calls, where
    should they go", and only the second one lets you read a cost/quality
    trade-off off the page. Both ends are operating points: 0% is the free arm
    alone, 100% is the expert alone, and the interesting answer is usually
    neither.

    Two allocation rules at every budget, both fitted on training folds:
      confidence  -- escalate the least-confident x% of traffic. The cascade
                     dial; ignores which class the item is in.
      per_class_conf -- the board and the dial combined: one confidence
                     threshold per predicted class, budget split across classes
                     by marginal value per call. Contains the other two as
                     corner cases.
      class_aware -- rank PREDICTED classes by the expert's measured advantage
                     on them and spend the budget on the best classes first,
                     least-confident first within a class. This is the only form
                     of the class-aware rule that takes a budget, and it reduces
                     to the margin rule when the budget is unconstrained.
      precision_floor -- the only rule here that is not given a budget at all.
                     It is given a promise -- no class falls more than `m` below
                     the expert's own precision on that class -- and the spend
                     is whatever keeping it costs. Its column of the curve is
                     therefore indexed by margin, not by share, which is why
                     `expert_share_target` does not describe it and the realized
                     `expert_share` beside each cell does.

    Two summaries come back, and they differ for a reason. `best_point_optimistic`
    is the max of the curve -- useful for reading the shape, biased upward as a
    headline because the point was chosen by looking at its own score.
    `best_point` is nested: rule and budget chosen inside each fold's training
    data, applied once to its held-out items. Quote the second one.
    """
    out = {"free_alone": balanced_accuracy(free_pred, gold),
           "expert_alone": balanced_accuracy(expert_pred, gold), "points": []}
    for frac in fractions:
        esc = {r: np.zeros(len(gold), dtype=bool) for r in RULES}
        for tr, te in folds:
            for r in RULES:
                esc[r] |= _allocate(tr, te, r, frac, free_pred, free_conf,
                                    expert_pred, gold, fit_labels, signals)
        row = {"expert_share_target": frac}
        for r in RULES:
            final = np.where(esc[r], expert_pred, free_pred)
            row[r] = {"expert_share": float(esc[r].mean()),
                      "balanced": balanced_accuracy(final, gold)}
        out["points"].append(row)
    # marginal value of the next slice of spend, so the knee is visible
    pts = out["points"]
    for i, row in enumerate(pts):
        if i == 0:
            row["gain_per_10pct_spend"] = None
            continue
        d_acc = row["confidence"]["balanced"] - pts[i - 1]["confidence"]["balanced"]
        d_spend = row["confidence"]["expert_share"] - pts[i - 1]["confidence"]["expert_share"]
        row["gain_per_10pct_spend"] = float(d_acc / d_spend * 0.10) if d_spend > 1e-9 else None
    best = max(pts, key=lambda r: max(r[x]["balanced"] for x in RULES))
    win = max(RULES, key=lambda x: best[x]["balanced"])
    out["best_point_optimistic"] = {
        "expert_share": best[win]["expert_share"],
        "balanced": best[win]["balanced"],
        "rule": win,
        "n_configurations_selected_over": len(RULES) * len(fractions)}
    out["cheapest_equivalent"] = cheapest_equivalent(
        pts, out["best_point_optimistic"]["balanced"], gold)
    out["best_point"] = nested_best_operating_point(
        free_pred, free_conf, expert_pred, gold, folds, fractions, seed=seed,
        signals=signals,
        fit_labels=fit_labels, groups=groups)
    out["selection_bias"] = (out["best_point_optimistic"]["balanced"]
                             - out["best_point"]["balanced"])
    # G6 applied where it matters: an operating point is only interesting if its
    # advantage over BOTH pure strategies is one the data can actually see.
    bp = out["best_point"]
    out["gain_over_free"] = resolvable(bp["balanced"] - out["free_alone"], gold)
    out["gain_over_expert"] = resolvable(bp["balanced"] - out["expert_alone"], gold)
    out["beats_both_ends"] = bool(out["gain_over_free"]["resolvable"]
                                  and out["gain_over_expert"]["resolvable"]
                                  and out["gain_over_free"]["delta"] > 0
                                  and out["gain_over_expert"]["delta"] > 0)
    return out


def guaranteed_coverage(pred: np.ndarray, conf: np.ndarray, gold: np.ndarray,
                        expert_pred: np.ndarray, bar: float,
                        folds, min_support: int = 30) -> dict:
    """Own what clears a per-class PRECISION bar, defer the rest to the expert.

    Thresholds are fitted on training folds only. The realized precision is
    reported next to the requested bar on purpose: on a rare class the empirical
    threshold does not transfer, and that failure is the finding, not a bug.
    """
    keep = np.zeros(len(gold), bool)
    for tr, te in folds:
        for c in sorted(set(pred.tolist())):
            m = pred[tr] == c
            if not m.any():
                continue
            thr = np.inf
            for t in np.sort(np.unique(conf[tr][m])):
                mm = m & (conf[tr] >= t)
                if mm.sum() >= min_support and (gold[tr][mm] == c).mean() >= bar:
                    thr = t
                    break
            keep[te[(pred[te] == c) & (conf[te] >= thr)]] = True
    final = np.where(keep, pred, expert_pred)
    realized, vacuous = {}, []
    for c in sorted(set(pred.tolist())):
        k = keep & (pred == c)
        if not k.any():
            vacuous.append(int(c))
        realized[int(c)] = {"owned_share": float(k.mean()),
                            "realized_precision": float((gold[k] == c).mean()) if k.any() else float("nan")}
    served = [v for v in realized.values() if not np.isnan(v["realized_precision"])]
    return {"bar": bar, "free_share": float(keep.mean()),
            "expert_share": float(1 - keep.mean()),
            "balanced": balanced_accuracy(final, gold),
            "per_class": realized,
            # A class the rule declines to serve at all is the guarantee working,
            # not failing -- it defers rather than promising what it cannot keep.
            # But that is a different outcome from serving everything at the bar,
            # so it is reported separately instead of hiding inside one boolean.
            "vacuous_classes": vacuous,
            "guarantee_holds": bool(served) and all(
                v["realized_precision"] >= bar - 0.01 for v in served)}


def margin_coverage(pred: np.ndarray, conf: np.ndarray, gold: np.ndarray,
                    expert_pred: np.ndarray, margin: float,
                    folds, min_support: int = 30) -> dict:
    """`guaranteed_coverage`, with the bar anchored to the expert instead of to
    a number someone picked.

    Reported side by side with the absolute-bar version on purpose. The two
    answer different customer questions -- "90% precision, full stop" versus
    "never more than 10% worse than the model you are paying for" -- and only
    the second one is always keepable, because the second one cannot ask the
    free arm to beat a class the expert itself cannot do.

    `promise_holds` is the claim being sold, so it is checked against the
    realized per-class numbers rather than asserted from the fitted thresholds.
    """
    keep = np.zeros(len(gold), bool)
    bars = {}
    for tr, te in folds:
        esc = np.zeros(len(gold), bool)
        _precision_floor(tr, te, margin, pred, conf, expert_pred, gold, esc,
                         min_support=min_support)
        keep[te] = ~esc[te]
        curves, pooled = _cpm_curves(tr, pred, conf, expert_pred, gold, min_support)
        for c, (_, _, _, anchor_c) in curves.items():
            bars.setdefault(int(c), []).append((1.0 - margin) * anchor_c)
    final = np.where(keep, pred, expert_pred)
    realized, shortfall = {}, []
    for c in sorted(set(pred.tolist())):
        k = keep & (pred == c)
        em = expert_pred == c
        bar = float(np.mean(bars[int(c)])) if int(c) in bars else float("nan")
        got = float((gold[k] == c).mean()) if k.any() else float("nan")
        realized[int(c)] = {
            "owned_share": float(k.mean()),
            "bar": bar,
            "expert_precision": float((gold[em] == c).mean()) if em.any() else float("nan"),
            "realized_precision": got}
        if k.any() and not np.isnan(bar) and got < bar - 0.01:
            shortfall.append(int(c))
    return {"margin": margin, "free_share": float(keep.mean()),
            "expert_share": float(1 - keep.mean()),
            "balanced": balanced_accuracy(final, gold),
            "per_class": realized,
            # Classes the rule declines outright. Under an absolute bar this is
            # the common outcome on a hard class; under an anchored bar it means
            # the free arm genuinely trails the expert there, which is the
            # finding the whole method is looking for.
            "deferred_classes": [c for c, v in realized.items()
                                 if v["owned_share"] == 0.0],
            "classes_below_bar": shortfall,
            "promise_holds": not shortfall}

def soft_target_fit(X, P, l2: float = 1.0, temperature: float = 1.0,
                    max_iter: int = 300, sample_weight=None):
    """Multinomial logistic regression fitted against a probability MATRIX.

    sklearn takes a label vector, so a teacher's distribution has to be
    collapsed to its argmax before a student can be trained on it -- which
    throws away exactly the thing worth keeping. Two items the teacher calls
    class 3 with 0.99 and with 0.34 become the same training row, and the
    student learns to be equally certain of both. Hinton's soft target keeps the
    difference, and with it the inter-class structure: "this is a 3, and if it
    were not, it would be a 5" is a stronger lesson than "this is a 3".

    `temperature` softens the teacher before fitting. The conventional T^2 on
    the loss exists to keep a soft term commensurate with a hard-label term when
    both are present; there is no hard term here, so it would only rescale the
    regularizer, and it is folded into `l2` instead of applied silently.

    Returns an object with `predict_proba`/`predict`/`classes_` so it is a
    drop-in wherever a fitted sklearn classifier is expected. `X` may be dense
    or scipy-sparse.

    MEASURED AND NOT SHIPPED, like `router` and `conformal` below. Distillation
    is the dominant lever in CalexNet's own ablation (Aperstein & Apartsin 2025,
    arXiv:2509.08318), so it was worth a run: bge-large+logreg as teacher,
    tfidf+logreg as student, teacher probabilities taken from an inner CV inside
    each training fold so they are neither in-sample nor leaked from the outer
    test fold.

        dataset      hard student   best soft   delta    floor
        isear            0.6062       0.6100   +0.0038  +/-0.0057
        hatespeech       0.7789       0.7873   +0.0084  +/-0.0102
        fever            0.6829       0.6828   -0.0001  +/-0.0062

    Positive twice, negative once, and never outside the resolution floor. The
    teacher leads the student by 0.052 to 0.080 on these three and distillation
    recovers at most a seventh of that.

    The reason is structural, not a tuning failure. CalexNet's branch reads the
    backbone's OWN intermediate feature map, so "this is a 3 and otherwise a 5"
    is a sentence the student's features can express. Ours reads sparse word and
    character n-grams while the teacher reads a 1024-d neural embedding; the
    inter-class geometry the soft target carries has no coordinates in the
    student's space. Temperature does not fix that -- T=4 was worse than T=1 on
    all three.

    Kept because the negative result should be re-checkable, and because a
    teacher that DOES emit a distribution -- an LLM API returning logprobs,
    which OCL's parsed-label annotations are not -- is the case this was built
    for and the case we have not been able to test.
    """
    from scipy.optimize import minimize
    from scipy.special import softmax as _softmax

    P = np.asarray(P, dtype=float)
    if temperature != 1.0:
        with np.errstate(divide="ignore"):
            P = _softmax(np.log(np.clip(P, 1e-12, None)) / temperature, axis=1)
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    # The cheap arm's features are a sparse TF-IDF matrix with tens of thousands
    # of columns, so the bias is carried separately rather than appended as a
    # column of ones -- densifying X to bolt one column on would be the single
    # most expensive thing in the fit, for a constant.
    n, d = X.shape
    k = P.shape[1]
    # Without this the comparison against a `class_weight="balanced"` sklearn
    # baseline is not a comparison: the soft student would be optimising plain
    # accuracy while the arm it is measured against optimises the balanced kind,
    # and any difference would be the weighting rather than the distillation.
    sw = (np.ones(n) if sample_weight is None
          else np.asarray(sample_weight, dtype=float))
    sw = sw * (n / sw.sum())

    def obj(w):
        W = w[:d * k].reshape(d, k)
        b = w[d * k:]
        Z = X @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(Z).sum(axis=1))
        loss = float((sw * (lse - (P * Z).sum(axis=1))).sum()) / n
        D = sw[:, None] * (np.exp(Z - lse[:, None]) - P) / n
        gW = np.asarray(X.T @ D)
        gb = D.sum(axis=0)
        # the bias is not penalised; penalising it just shrinks the base rates
        return (loss + 0.5 * l2 * float((W ** 2).sum()),
                np.concatenate([(gW + l2 * W).ravel(), gb]))

    res = minimize(obj, np.zeros(d * k + k), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter})
    W = res.x[:d * k].reshape(d, k)
    b = res.x[d * k:]

    class _Fitted:
        classes_ = np.arange(k)
        n_iter_ = int(res.nit)

        def predict_proba(self, Z):
            A = Z @ W + b
            A -= A.max(axis=1, keepdims=True)
            E = np.exp(A)
            return E / E.sum(axis=1, keepdims=True)

        def predict(self, Z):
            return self.predict_proba(Z).argmax(axis=1)

    return _Fitted()

# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def compare_candidates(pred: dict[str, np.ndarray], gold: np.ndarray,
                       seed: int = 0) -> dict:
    """Rank candidates, then Holm-correct every challenger against the leader.

    The correction is the difference between "our best of seven beat it" and a
    claim that survives review.
    """
    scored = sorted(((n, balanced_accuracy(p, gold)) for n, p in pred.items()),
                    key=lambda kv: -kv[1])
    best_name = scored[0][0]
    challengers = [n for n, _ in scored[1:]]
    tests = {n: _class_boot(pred[n], pred[best_name], gold, seed=seed)
             for n in challengers}
    flags = holm_bonferroni([tests[n][2] for n in challengers]) if challengers else []
    return {"ranking": [{"name": n, "balanced": b} for n, b in scored],
            "best": best_name,
            "vs_best": {n: {"delta": balanced_accuracy(pred[n], gold)
                                     - balanced_accuracy(pred[best_name], gold),
                            "ci": [tests[n][0], tests[n][1]], "p": tests[n][2],
                            "significant_after_holm": bool(f)}
                        for n, f in zip(challengers, flags)},
            "n_candidates": len(pred),
            "holm_alpha": 0.05 / max(1, len(challengers))}
