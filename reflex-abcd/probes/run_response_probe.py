"""CLI for the H5 / H7 / H4 measurement protocol. Run by the COORDINATOR.

    python -m probes.run_response_probe conformance
    python -m probes.run_response_probe audit      --split dev
    python -m probes.run_response_probe validate
    python -m probes.run_response_probe measure    --heads h5,h7,h4
    python -m probes.run_response_probe select     --heads h5,h7,h4

MODES
-----
``conformance``  Checks the featurizer against ``reflex.select._render_variant``.
                 Needs NO corpus. Run it first; a failure is a wiring bug.

``audit``        Label-space and coverage facts only -- no model, no accuracy.
                 Emits every denominator and every ceiling the metrics are
                 quoted against. Safe to run before anything is trusted.

``validate``     THE REPRODUCTION CHECK. Re-measures a known D7 cell and refuses
                 to certify the harness unless it lands in band. See below.

``measure``      Runs the arms for the requested heads, with the ``nat->nat`` /
                 ``nat->shuf`` / ``shuf->shuf`` controls and clustered bootstrap
                 CIs. Refuses to emit numbers unless ``validate`` has certified
                 this harness, unless ``--force-uncertified`` is passed, and
                 stamps ``certified: false`` on the artifact when it is.
                 **Every number it emits is a DEV number, on a grid a reader then
                 picks from by eye -- see ``select``.**

``select``       THE CLEAN SELECTION PROTOCOL. Splits TRAIN internally, grouped
                 by conversation and seeded; sweeps the grid on that split only;
                 refits the winner on all of train; scores ``test_seen`` ONCE.
                 ``dev`` comes back as a free held-out consistency check (it took
                 no part in selection here) and ``test_novel`` separately, never
                 pooled. Use this for any number that is going to be quoted.

ESTIMATORS AND MEMORY -- READ BEFORE CHANGING THE SOLVER
--------------------------------------------------------
nextstep is 3-way and lbfgs/multinomial is free there; that is the CERTIFIED path
and it does not change. H5/H7/H4 are 226 to 1,066-way, and an lbfgs history at
that width is 21 dense ``(n_classes x n_features)`` arrays -- ~36 GiB for H7's
ASK pool alone on a 24 GB machine, which is why ``measure --heads h5,h7,h4`` died
with SIGBUS and no traceback. Their estimator and feature cap are config, in
``probe.vectorizer_by_head``, and :func:`check_dense_state` refuses a fit that
would not fit rather than letting it die in native code.

WHY THE REPRODUCTION CHECK IS SHAPED THE WAY IT IS -- READ THIS BEFORE TRUSTING
A VALIDATION RESULT
-------------------------------------------------------------------------------
The target is DECISIONS D7's twice-verified cell: nextstep, recency-tagged,
k6 window, MAXB=3, dev accuracy 0.8345 (agent) / 0.8351 (maintainer).

**The harness that produced it is not in this repository.** Its digest
``08ad053a37a67cc36aeec16e0a02306c`` appears only in DECISIONS.md and
configs/default.yaml; no file in the tree computes it. So four things that
materially move the number are UNRECORDED and are therefore knobs here, not
constants:

1. **The row set.** D7 quotes "38,795 train / 13,284 dev rows". The current bank
   compiles 105,672 agent-side train turns from 7,467 conversations, so 38,795 is
   a SUBSET and the subsetting rule is not written down anywhere. ``--train-convos``
   / ``--train-rows`` exist for this and the sweep tries the recorded row counts.
2. **The TF-IDF and logreg settings** (``min_df``, ``sublinear_tf``, ``C``,
   ``class_weight``). D7 says only "word 1-2gram TF-IDF + logreg, same C".
3. **Whether the STATE LINE was in the rendered text.** ``build_context`` appends
   it and D13 measured its token cost, but whether the probe fed it is unstated.
   ``--state-line include|exclude``.
4. **How the speaker prefix interacts with word-mode tagging.**
   ``select._tag_turn`` emits ``speaker|r0|tok r0|tok ...`` -- the speaker once,
   outside the tagging. A probe that tagged the speaker token too would have a
   different vocabulary.

``validate`` sweeps the declared grid and reports EVERY cell against the band,
plus the supporting cells D7 also records (plain full 0.7855, plain k6
0.8062-0.8071, tagged full 0.8281-0.8297, and the k6 MAXB row 1/2/3/4 =
.8307/.8345/.8351/.8318). Landing the headline cell alone is weak evidence;
reproducing the SHAPE of the MAXB curve is much stronger, because the curve is
what a wrong rendering would distort.

**If no cell lands in band, the harness is NOT certified and no H5/H7/H4 number
may be quoted from it.** That is the honest outcome: it would mean D7's cell is
not reproducible from what the repo records, which is itself a finding worth
more than an uncertified accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Optional, Sequence

from probes import response_labels as RL
from probes import response_metrics as RM
from probes.featurizer_api import RenderSpec, VectorizerSpec, conformance_report, load_featurizer


# --------------------------------------------------------------------------- #
# Probe config -- kept OUT of configs/default.yaml, which the coordinator owns
# --------------------------------------------------------------------------- #

PROBE_CFG_DEFAULT = os.path.join(os.path.dirname(__file__), "probe.yaml")


def load_probe_cfg(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_project_cfg(overrides: Sequence[str]) -> dict:
    """Load ``configs/default.yaml`` READ-ONLY, with ``--set`` overrides applied.

    This harness never writes to that file: the coordinator owns it exclusively
    and DECISIONS records a previous run losing a config block to exactly that
    race.
    """
    from reflex.config import load_config
    return load_config(overrides=list(overrides) or None)


# --------------------------------------------------------------------------- #
# Row cache -- the corpus parse is the dominant cost (DEFECTS_OPEN D-10)
# --------------------------------------------------------------------------- #


def _cache_key(cfg: dict, split: str, bank_hash: str) -> str:
    from reflex.data import dataset_hash
    payload = json.dumps(
        {"dataset": dataset_hash(cfg), "split": split, "bank": bank_hash,
         "K": cfg.get("data", {}).get("context_turns_K")},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_split_rows(cfg: dict, split: str, bank: Any, h4_indexer: Any, cache_dir: str) -> dict:
    """Build (or reuse) the probe rows for one split.

    ``build_partitions`` + ``load_raw_abcd`` parse the whole corpus and that is
    the expensive step; every arm reuses ONE call. Contexts are kept in memory
    rather than pickled, because a ContextWindow per agent-side turn is cheap
    next to the raw object graph that produced it.
    """
    from reflex.compile import load_turn_labels
    from reflex.data import build_partitions, iter_agent_turns
    from reflex.train import _derive_turn_labels

    partitions = build_partitions(cfg)
    try:
        labels = list(load_turn_labels(split, cfg))
        provenance = f"labels/{split}.jsonl (written by compile.compile_bank)"
    except FileNotFoundError:
        labels = _derive_turn_labels(getattr(partitions, split), bank, cfg, split)
        provenance = "reflex.train._derive_turn_labels (NO delexicalization -- see probes.response_labels)"
    rows = RL.build_rows(partitions, split, bank, labels, cfg, h4_indexer=h4_indexer)
    rows["label_provenance"] = provenance
    rows["partitions"] = partitions
    # The PARENT population of rows["h4"]: build_rows drops every take_action turn
    # with an empty gold value list, so len(rows["h4"]) is not the take_action count
    # and must not be reported under that name. Counted here because build_rows
    # does not return it. Cheap: no context is built, the partition is already
    # parsed.
    rows["n_take_action_turns"] = sum(
        1 for _, _, turn in iter_agent_turns(getattr(partitions, split))
        if turn.nextstep == "take_action"
    )
    # Same for rows["h5"]: build_rows drops every retrieve turn that has no
    # TurnLabel BEFORE the arm sees it, so len(rows["h5"]) is "retrieve turns with
    # a label", not "every retrieve turn". Count the parent population directly.
    rows["n_retrieve_turns"] = sum(
        1 for _, _, turn in iter_agent_turns(getattr(partitions, split))
        if turn.nextstep == "retrieve_utterance"
    )
    return rows


def _n_unlabelled(rows: dict) -> Any:
    """Retrieve turns build_rows dropped for having no TurnLabel (one H5Row per
    labelled retrieve turn), or None when the split's turns were not counted."""
    n_all = rows.get("n_retrieve_turns")
    return None if n_all is None else int(n_all) - len(rows.get("h5") or [])


# --------------------------------------------------------------------------- #
# One fit/score pass
# --------------------------------------------------------------------------- #


class DenseStateTooLarge(MemoryError):
    """The estimator's dense parameter state would not fit the declared budget.

    Raised BEFORE the fit, so the failure is a Python exception naming the head,
    the class count and the feature count -- not a SIGBUS in native code with no
    traceback, which is what `measure --heads h5,h7,h4` did before this guard
    existed (verified twice by the coordinator: RSS 3.0 -> 7.1 GB, exit 138).
    """


def estimate_dense_state(n_classes: int, n_features: int, classifier: str,
                         guard_cfg: Optional[dict]) -> dict:
    """Bytes of dense parameter state ``classifier`` will hold at ``n_classes x n_features``.

    The multiplier is the number of dense ``(n_classes, n_features)`` arrays the
    solver keeps live at once, and it is the whole story here:

    * ``logreg`` is lbfgs/multinomial. SciPy's L-BFGS keeps ``2m + 1 = 21``
      correction vectors at the default ``m = 10``, each the full size of the
      coefficient array. That 21x is why H7 died: ASK is 1,066 classes and, at
      218k features, one vector is 1.86 GB, so the history alone asks for ~39 GB
      on a 24 GB machine.
    * ``logreg_ovr`` and ``sgd_log`` are one-vs-rest. They still materialise the
      full ``(n_classes, n_features)`` coefficient matrix -- that floor is
      unavoidable for any linear model over this label space -- but they hold no
      history, so the multiplier drops to ~2 and the same pool needs ~1.9 GB
      instead of ~39 GB.

    The multipliers live in ``probe.memory_guard.state_vectors_by_classifier``
    (spec 10: no numeric threshold in Python). Nothing here is measured RSS; it
    is the parameter-array arithmetic, which is the part that scales with the
    label space and is what overran.
    """
    guard_cfg = guard_cfg or {}
    itemsize = int(guard_cfg.get("bytes_per_coefficient", 8))
    vectors = int((guard_cfg.get("state_vectors_by_classifier") or {}).get(classifier, 1))
    budget_gib = float(guard_cfg.get("max_dense_state_gib", 0.0))
    total = int(n_classes) * int(n_features) * itemsize * vectors
    budget = int(budget_gib * (1024 ** 3))
    per_class = max(int(n_classes) * itemsize * vectors, 1)
    return {
        "classifier": classifier,
        "n_classes": int(n_classes),
        "n_features": int(n_features),
        "state_vectors": vectors,
        "bytes_per_coefficient": itemsize,
        "projected_bytes": total,
        "projected_gib": total / (1024 ** 3),
        "budget_gib": budget_gib,
        "over_budget": bool(budget > 0 and total > budget),
        "max_features_that_would_fit": (budget // per_class) if budget > 0 else None,
    }


def check_dense_state(head: str, n_classes: int, n_features: int,
                      vspec: VectorizerSpec, guard_cfg: Optional[dict]) -> dict:
    """Refuse a fit whose dense state exceeds ``probe.memory_guard``.

    ``on_over_budget: raise`` (the default) is deliberate. Silently shrinking the
    representation would change the number without changing the report, which is
    the D6 failure mode; the error instead names the exact config key and the
    ``max_features`` value that WOULD fit, so the remedy is a recorded config
    change rather than a hidden one.
    """
    est = estimate_dense_state(n_classes, n_features, str(vspec.classifier), guard_cfg)
    est["head"] = head
    if not est["over_budget"]:
        return est
    if str((guard_cfg or {}).get("on_over_budget", "raise")) != "raise":
        return est
    raise DenseStateTooLarge(
        f"{head}: refusing to fit. classifier={vspec.classifier!r} at "
        f"{est['n_classes']} classes x {est['n_features']} features would hold "
        f"{est['state_vectors']} dense arrays = {est['projected_gib']:.2f} GiB, over the "
        f"probe.memory_guard.max_dense_state_gib budget of {est['budget_gib']:.2f} GiB. "
        f"Remedies, in probes/probe.yaml: set probe.vectorizer_by_head.{head}.max_features "
        f"<= {est['max_features_that_would_fit']}, or set "
        f"probe.vectorizer_by_head.{head}.classifier to a one-vs-rest kind "
        f"('sgd_log' / 'logreg_ovr', which drop the lbfgs history), or raise "
        f"probe.memory_guard.max_dense_state_gib if the machine really has the room. "
        f"This guard exists because the unguarded fit died with SIGBUS and no traceback."
    )


def _classifier(vspec: VectorizerSpec, n_fit: Optional[int] = None):
    """The estimator named by ``vspec.classifier``.

    ``logreg`` is the CERTIFIED path and is untouched: ``validate`` reproduces
    D7's nextstep cell with it, and nextstep is 3-way so the lbfgs history costs
    nothing there. The other two kinds exist because H5/H7/H4 are not 3-way and
    lbfgs cannot hold their coefficient history (see :class:`DenseStateTooLarge`).

    **They are different estimators and the number must be quoted with the one
    that produced it (D6).** ``logreg_ovr`` keeps the logistic loss and the exact
    solver but fits one-vs-rest instead of multinomial; ``sgd_log`` keeps the
    logistic loss but optimises it by SGD, so it is also approximate. All three
    expose ``predict_proba``, which is the only interface the H5/H7/H4 metrics
    require -- per-act micro top-1, recall@5/@20, the three equivalence tiers and
    both denominators all read the ranked list, not the coefficients.
    """
    from sklearn.linear_model import LogisticRegression

    kind = str(vspec.classifier)
    if kind == "logreg":
        # NOTE: `multi_class` was REMOVED in scikit-learn 1.9 (pinned in
        # requirements.txt); passing it raises TypeError. Multinomial is the default
        # for lbfgs. random_state is set even though lbfgs is deterministic, because
        # spec 10 requires determinism to be stated rather than inferred.
        return LogisticRegression(
            C=vspec.C, max_iter=vspec.max_iter, class_weight=vspec.class_weight,
            solver="lbfgs", random_state=0,
        )
    if kind == "logreg_ovr":
        # liblinear lost its own multiclass path in scikit-learn 1.9 ("does not
        # support multiclass classification (n_classes >= 3) ... wrap the
        # estimator in a OneVsRestClassifier"), so the wrap is required, not a
        # preference. Exact solver, logistic loss, no solver history.
        from sklearn.multiclass import OneVsRestClassifier
        return OneVsRestClassifier(
            LogisticRegression(
                C=vspec.C, max_iter=vspec.max_iter, class_weight=vspec.class_weight,
                solver="liblinear", random_state=0,
            )
        )
    if kind == "sgd_log":
        from sklearn.linear_model import SGDClassifier
        alpha = getattr(vspec, "alpha", None)
        if alpha is None:
            # The standard correspondence between LogisticRegression's C and
            # SGD's alpha: C scales the loss, alpha scales the penalty per
            # sample. n_fit is required for it, so a caller that cannot supply
            # one must pin `alpha` in the config instead of getting a silent
            # default.
            if not n_fit:
                raise ValueError(
                    "sgd_log needs either an explicit `alpha` in the vectorizer config "
                    "or n_fit, to convert C into alpha = 1 / (C * n_fit)"
                )
            alpha = 1.0 / (float(vspec.C) * float(n_fit))
        return SGDClassifier(
            loss="log_loss", alpha=float(alpha), max_iter=int(vspec.max_iter),
            class_weight=vspec.class_weight, random_state=0,
        )
    raise ValueError(
        f"unsupported classifier {vspec.classifier!r}; expected 'logreg' (lbfgs multinomial, "
        f"the certified nextstep path), 'logreg_ovr' or 'sgd_log'"
    )


def guard_cfg(probe_cfg: dict) -> dict:
    """``probe.memory_guard``, or ``{}`` (no guard) when the key is absent."""
    return dict((probe_cfg.get("probe") or {}).get("memory_guard") or {})


def vspec_for_head(probe_cfg: dict, head: str, overrides: Optional[dict] = None) -> VectorizerSpec:
    """The vectorizer+estimator for one head: base, then sweep override, then head override.

    The HEAD override is applied LAST and therefore wins. That ordering is the
    memory contract: ``probe.vectorizer_by_head`` is what keeps a 1,066-class
    pool inside the budget, and a sweep entry must not be able to undo it and
    reintroduce the SIGBUS. ``nextstep`` has no entry, so ``validate`` keeps the
    certified base settings exactly.
    """
    probe = probe_cfg.get("probe") or {}
    merged = dict(probe.get("vectorizer") or {})
    merged.update(dict(overrides or {}))
    merged.update(dict((probe.get("vectorizer_by_head") or {}).get(head) or {}))
    return VectorizerSpec(**merged)


def fit_score(
    featurizer: Any,
    fit_rows: Sequence[Any],
    fit_golds: Sequence[str],
    eval_rows: Sequence[Any],
    eval_golds: Sequence[str],
    fit_spec: RenderSpec,
    eval_spec: RenderSpec,
    vspec: VectorizerSpec,
    ks: Sequence[int],
    *,
    head: str = "unnamed",
    guard: Optional[dict] = None,
    fit_texts: Optional[Sequence[str]] = None,
    eval_texts: Optional[Sequence[str]] = None,
) -> dict:
    """Fit on ``fit_rows`` rendered under ``fit_spec``; score ``eval_rows`` under ``eval_spec``.

    The two specs are separate so one function covers all three controls:
    ``nat->nat`` (equal), ``nat->shuf`` (eval shuffled only) and ``shuf->shuf``
    (both shuffled -- the REFIT control, which is the actual experiment; D7 is
    explicit that ``nat->shuf`` alone shows only a train/test mismatch).

    The keyword-only arguments are additive and default to the original
    behaviour. ``head`` and ``guard`` feed :func:`check_dense_state`, which
    refuses a fit that cannot fit in memory instead of letting it SIGBUS.
    ``fit_texts`` / ``eval_texts`` let a caller that sweeps several vectorizer
    settings over ONE rendering pay the render cost once; passing them is exactly
    equivalent to letting this function render, and the caller must have rendered
    under the same specs.
    """
    import numpy as np

    fit_texts = ([featurizer.render_context(r.context, fit_spec) for r in fit_rows]
                 if fit_texts is None else list(fit_texts))
    eval_texts = ([featurizer.render_context(r.context, eval_spec) for r in eval_rows]
                  if eval_texts is None else list(eval_texts))

    vectorizer = featurizer.make_vectorizer(vspec)
    X_fit = vectorizer.fit_transform(fit_texts)
    X_eval = vectorizer.transform(eval_texts)

    # The guard runs here and not earlier because n_features is a property of the
    # FITTED vocabulary; n_classes is known from the labels.
    memory = check_dense_state(head, len(set(fit_golds)), int(X_fit.shape[1]), vspec, guard)

    model = _classifier(vspec, n_fit=len(fit_texts))
    model.fit(X_fit, list(fit_golds))
    classes = list(model.classes_)
    proba = model.predict_proba(X_eval)
    order = np.argsort(-proba, axis=1)

    max_k = max(ks) if ks else 1
    ranked = [[classes[j] for j in order[i, :max_k]] for i in range(order.shape[0])]
    per_row = {k: [] for k in ks}
    for ranks, gold in zip(ranked, eval_golds):
        flags = RM.hits_at_k(ranks, gold, ks)
        for k in ks:
            per_row[k].append(1.0 if flags[k] else 0.0)

    return {
        "top1": [row[0] for row in ranked],
        "ranked": ranked,
        "per_row_hits": per_row,
        "n_features": int(X_fit.shape[1]),
        "n_classes": len(classes),
        "n_fit": len(fit_texts),
        "n_eval": len(eval_texts),
        "fit_texts_sample": fit_texts[:3],
        "estimator": str(vspec.classifier),
        "dense_state": memory,
    }


# --------------------------------------------------------------------------- #
# validate -- reproduce a known D7 cell
# --------------------------------------------------------------------------- #

#: DECISIONS D7, "the ship configuration, verified twice" plus the MAXB table.
#: Each entry is (label, RenderSpec kwargs, target lo, target hi). The bands are
#: the two independently measured values where D7 gives both, and +-0.001 around
#: a single recorded value otherwise -- NOT a tolerance chosen to make the check
#: pass, but the spread the record itself contains.
D7_CELLS = (
    ("nextstep tagged k6 MAXB=3 (THE headline cell)",
     dict(k=6, tag_recency=True, recency_buckets=3), 0.8345, 0.8351),
    ("nextstep plain k6",
     dict(k=6), 0.8062, 0.8071),
    ("nextstep tagged full MAXB=3",
     dict(k="full", tag_recency=True, recency_buckets=3), 0.8281, 0.8297),
    ("nextstep plain full",
     dict(k="full"), 0.7845, 0.7865),
    ("nextstep tagged k6 MAXB=1",
     dict(k=6, tag_recency=True, recency_buckets=1), 0.8297, 0.8317),
    ("nextstep tagged k6 MAXB=2",
     dict(k=6, tag_recency=True, recency_buckets=2), 0.8335, 0.8355),
    ("nextstep tagged k6 MAXB=4",
     dict(k=6, tag_recency=True, recency_buckets=4), 0.8308, 0.8328),
)

#: The cheapest structural check in the whole harness and it needs no target:
#: a 6-turn window has only 6 positions, so k6 at MAXB 6 / 8 / 12 must be
#: byte-identical. D7 records exactly this sanity check.
D7_IDENTITY_CELLS = (6, 8, 12)


def mode_validate(args, probe_cfg: dict, cfg: dict) -> dict:
    from reflex.compile import load_bank

    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])
    conf = conformance_report(featurizer)
    if not conf["ok"] and not args.force_uncertified:
        return {"certified": False, "stage": "conformance", "conformance": conf,
                "reason": "the featurizer does not reproduce reflex.select._render_variant; "
                          "fix the wiring before measuring anything"}

    bank = load_bank(cfg)
    indexer = None
    rows_train = load_split_rows(cfg, "train", bank, indexer, args.cache_dir)
    rows_dev = load_split_rows(cfg, "dev", bank, indexer, args.cache_dir)

    # nextstep rows: every agent-side turn. Reuse the H5/H4 row builders' contexts
    # by rebuilding a flat list keyed on the same partitions, so the reproduction
    # runs on the SAME context objects the response heads will run on.
    def _nextstep_rows(rows, split):
        from reflex.data import build_context, iter_agent_turns, load_raw_abcd, turn_key
        partitions = rows["partitions"]
        raw = load_raw_abcd(cfg)
        scenarios = {int(c["convo_id"]): c.get("scenario", {}) or {}
                     for convos in raw.values() for c in convos}
        out = []
        for convo_id, turn_index, turn in iter_agent_turns(getattr(partitions, split)):
            ctx = build_context(getattr(partitions, split)[convo_id], turn_index,
                                scenarios.get(int(convo_id), {}), cfg)
            out.append(type("R", (), {
                "context": ctx, "convo_id": int(convo_id),
                "turn_id": turn_key(split, convo_id, turn_index),
                "gold": str(turn.nextstep)})())
        return out

    train_rows = _nextstep_rows(rows_train, "train")
    dev_rows = _nextstep_rows(rows_dev, "dev")
    if args.train_convos:
        keep, seen = [], set()
        for row in train_rows:
            if row.convo_id not in seen and len(seen) >= args.train_convos:
                continue
            seen.add(row.convo_id)
            keep.append(row)
        train_rows = keep
    if args.train_rows:
        train_rows = train_rows[: args.train_rows]

    grid = probe_cfg["probe"]["validate_grid"]
    results = []
    best = None
    for vs in grid:
        vspec = VectorizerSpec(**{**probe_cfg["probe"]["vectorizer"], **vs})
        cells = []
        for label, spec_kwargs, lo, hi in D7_CELLS:
            spec = RenderSpec(**spec_kwargs)
            out = fit_score(
                featurizer, train_rows, [r.gold for r in train_rows],
                dev_rows, [r.gold for r in dev_rows], spec, spec, vspec, (1,),
                head="nextstep", guard=guard_cfg(probe_cfg),
            )
            acc = RM.accuracy([bool(x) for x in out["per_row_hits"][1]])
            ci = RM.bootstrap_ci(out["per_row_hits"][1], [r.convo_id for r in dev_rows],
                                 n_boot=args.n_boot, seed=args.seed)
            cells.append({
                "cell": label, "spec": spec.label(), "accuracy": acc,
                "target_lo": lo, "target_hi": hi,
                "in_band": acc is not None and lo <= acc <= hi,
                "ci": ci, "n_fit": out["n_fit"], "n_eval": out["n_eval"],
                "n_features": out["n_features"],
            })
        headline = cells[0]
        score = sum(1 for c in cells if c["in_band"])
        row = {"vectorizer": vspec.as_dict(), "cells": cells,
               "cells_in_band": score, "headline_in_band": headline["in_band"]}
        results.append(row)
        # Rank by the headline cell first, then by curve coverage. This is the rule
        # that produced every certified artifact on disk, so it is kept deliberately:
        # `certified` below gates on best["headline_in_band"], and ranking by anything
        # else picks a winner the gate then rejects (a half-applied change did exactly
        # that -- it made a re-run write certified:false with a different vectorizer).
        #
        # It is NOT a good rule. The headline band is 0.0006 wide while that cell's own
        # clustered CI half-width is ~0.007, so the headline cannot really carry the
        # ranking, and the winner it picks reproduces fewer of D7's 7 cells than another
        # candidate does. That gap is now reported as max_cells_in_band /
        # max_cells_vectorizer instead of being invisible, and probe.min_cells_in_band
        # can enforce a floor. Changing the RANKING re-certifies a different vectorizer
        # and moves every published headline, so it is a deliberate decision, not a fix.
        if best is None or (row["headline_in_band"], score) > (best["headline_in_band"], best["cells_in_band"]):
            best = row

    identity = {
        b: featurizer.render_context(dev_rows[0].context,
                                     RenderSpec(k=6, tag_recency=True, recency_buckets=b))
        for b in D7_IDENTITY_CELLS
    }
    identity_ok = len(set(identity.values())) == 1

    # Optional floor on curve reproduction. probe.yaml does not set it today, so
    # the default is "no floor" and the verdict is unchanged for a config that
    # already reproduces the curve; set probe.min_cells_in_band to require one.
    min_cells = (probe_cfg.get("probe") or {}).get("min_cells_in_band")
    cells_ok = True if min_cells is None else bool(best and best["cells_in_band"] >= int(min_cells))
    certified = bool(best and best["headline_in_band"] and identity_ok and cells_ok)
    return {
        "certified": certified,
        "stage": "validate",
        "identity_check_k6_maxb_6_8_12": identity_ok,
        "n_cells": len(D7_CELLS),
        "best_cells_in_band": (best or {}).get("cells_in_band"),
        # Curve coverage of the BEST-COVERING candidate, which is not necessarily the
        # certified one. If max_cells_in_band > best_cells_in_band, the certified config
        # reproduces D7's curve worse than another candidate in the same grid.
        "max_cells_in_band": max((r["cells_in_band"] for r in results), default=None),
        "max_cells_vectorizer": (max(results, key=lambda r: r["cells_in_band"])["vectorizer"]
                                 if results else None),
        "min_cells_in_band": min_cells,
        "best": best,
        "all": results,
        "conformance": conf,
        "n_train_rows": len(train_rows),
        "n_dev_rows": len(dev_rows),
        "d7_recorded_row_counts": {"train": 38795, "dev": 13284},
        "interpretation": (
            "CERTIFIED means this harness reproduced D7's twice-verified nextstep cell. "
            "NOT certified means one of two things and the artifact does not distinguish "
            "them: either this harness renders/vectorizes differently from the lost one, "
            "or D7's cell is not reproducible from what the repository records. Either way "
            "no H5/H7/H4 number may be quoted until it is resolved."
        ) if not certified else (
            "CERTIFIED. The reproduction cell landed in band and the MAXB 6/8/12 identity "
            "holds. Quote the row set and vectorizer settings with every number (D6). "
            "Read best_cells_in_band / n_cells with the boolean: certification turns on the "
            "headline cell, but the curve is how much of D7 this harness actually reproduces."
        ),
    }


# --------------------------------------------------------------------------- #
# measure -- the actual H5 / H7 / H4 arms
# --------------------------------------------------------------------------- #


def _arm_specs(probe_cfg: dict) -> list:
    """The window x tagging grid each head is measured on.

    D2b and D7 found the best window and the best order-sensitivity differ PER
    HEAD and point in opposite directions, and both records say H5/H7/H4 must be
    measured rather than inherit a default. So the grid is the same one those
    two decisions were taken on -- it is the only way the new heads' answers are
    comparable to the established ones.
    """
    return [RenderSpec(**entry) for entry in probe_cfg["probe"]["arms"]]


def _h5_arm(featurizer, train_rows, dev_rows, spec, vspec, ks, n_boot, seed, shuffle_seed,
            *, guard=None, controls=True, n_retrieve_all=None):
    fit = [r for r in train_rows if r.gold_skeleton_id]
    ev = [r for r in dev_rows if r.gold_skeleton_id]
    shuf = RenderSpec(**{**spec.as_dict(), "shuffle": "within_window", "shuffle_seed": shuffle_seed})

    nat = fit_score(featurizer, fit, [r.gold_skeleton_id for r in fit],
                    ev, [r.gold_skeleton_id for r in ev], spec, spec, vspec, ks,
                    head="h5", guard=guard)
    nat_shuf = shuf_shuf = None
    if controls:
        nat_shuf = fit_score(featurizer, fit, [r.gold_skeleton_id for r in fit],
                             ev, [r.gold_skeleton_id for r in ev], spec, shuf, vspec, ks,
                             head="h5", guard=guard)
        shuf_shuf = fit_score(featurizer, fit, [r.gold_skeleton_id for r in fit],
                              ev, [r.gold_skeleton_id for r in ev], shuf, shuf, vspec, ks,
                              head="h5", guard=guard)

    clusters = [r.convo_id for r in ev]
    const = RM.constant_recall_at_k([r.gold_skeleton_id for r in fit],
                                    [r.gold_skeleton_id for r in ev], ks)
    acc = {f"recall@{k}": RM.accuracy([bool(x) for x in nat["per_row_hits"][k]]) for k in ks}
    ci1 = RM.bootstrap_ci(nat["per_row_hits"][1], clusters, n_boot=n_boot, seed=seed)

    # `ev` dropped every turn whose gold act-tuple is not in the train skeleton
    # bank (gold_skeleton_id is None for an OOV tuple), so `acc` is CONDITIONAL on
    # a reachable gold. Report both denominators, as _h7_arm and _composed do.
    # An OOV turn can never be a hit, so the rescale below is EXACT.
    # `dev_rows` is itself already conditional: build_rows dropped every retrieve
    # turn with no TurnLabel. `n_retrieve_all` is the COUNTED parent population
    # (load_split_rows); an unlabelled turn cannot be a hit either, so the rescale
    # stays exact. None (caller did not count) falls back to len(dev_rows).
    n_with_label = len(dev_rows)
    n_unconditional = n_retrieve_all if n_retrieve_all is not None else n_with_label
    gold_scale = (len(ev) / n_unconditional) if n_unconditional else 0.0

    # near-miss shape: right acts wrong order / right arity
    gold_acts = {r.turn_id: r.gold_acts for r in ev}
    shape = {"act_multiset": [], "arity": []}
    skeleton_acts = getattr(featurizer, "_skeleton_acts", None)
    for row, pred in zip(ev, nat["top1"]):
        pa = (skeleton_acts or {}).get(pred)
        ga = gold_acts.get(row.turn_id) or ()
        shape["act_multiset"].append(
            1.0 if pa is not None and sorted(pa) == sorted(ga) else 0.0)
        shape["arity"].append(1.0 if pa is not None and len(pa) == len(ga) else 0.0)

    return {
        "_pred_by_turn": {r.turn_id: p for r, p in zip(ev, nat["top1"])},
        "_gold_by_turn": {r.turn_id: r.gold_skeleton_id for r in ev},
        "head": "h5", "spec": spec.label(), "n_fit": len(fit), "n_eval": len(ev),
        "vectorizer": vspec.as_dict(), "dense_state": nat["dense_state"],
        "denominators": {
            "conditional_n": len(ev),
            "unconditional_n": n_unconditional,
            "unconditional_n_source": (
                "counted: every retrieve_utterance turn in the partition"
                if n_retrieve_all is not None else
                "NOT counted: len(rows) = retrieve turns that have a TurnLabel"),
            "n_turns_with_a_label": n_with_label,
            "turns_dropped_no_label": n_unconditional - n_with_label,
            "turns_dropped_gold_skeleton_oov": n_with_label - len(ev),
            "gold_present_rate": gold_scale,
            "note": "conditional_n is turns whose gold act-tuple exists in the train "
                    "skeleton bank. `accuracy` is over that subset; `unconditional` is "
                    "over unconditional_n (see unconditional_n_source), counting a turn "
                    "with no label and a turn with an OOV gold as misses.",
        },
        "accuracy": acc, "ci_top1": ci1, "constant": const,
        "unconditional": {
            "n": n_unconditional,
            "accuracy": {k: (v * gold_scale if v is not None else None)
                         for k, v in acc.items()},
            "constant": {k: (v * gold_scale if v is not None else None)
                         for k, v in const.items()},
            "note": "turns with no label or whose gold skeleton is OOV counted as misses "
                    "(exact, not an approximation: such a turn cannot be a hit).",
        },
        "d5": RM.d5_verdict(acc["recall@1"], const["recall@1"], ci1["lo"]),
        "controls": {
            "nat_to_nat": acc["recall@1"],
            "nat_to_shuf": RM.accuracy([bool(x) for x in nat_shuf["per_row_hits"][1]]),
            "shuf_to_shuf_REFIT": RM.accuracy([bool(x) for x in shuf_shuf["per_row_hits"][1]]),
            "order_effect_points_paired": RM.paired_bootstrap_delta(
                nat["per_row_hits"][1], shuf_shuf["per_row_hits"][1], clusters,
                n_boot=n_boot, seed=seed),
            "note": "shuf_to_shuf is the experiment; nat_to_shuf alone shows only a "
                    "train/test mismatch (DECISIONS D7).",
        } if controls else {
            "skipped": "order controls not run on this pass",
            "note": "D7's order finding is measured by `measure` on dev; the clean-protocol "
                    "eval pass reports the accuracy and its constant, not the controls.",
        },
        "near_miss_shape": {k: RM.accuracy([bool(x) for x in v]) for k, v in shape.items()},
        "recall_at_10_warning": (
            "a label-blind constant reaches recall@10 ~0.82 on this label prior. Do NOT "
            "quote H5 at k >= 5."
        ),
    }


def _h7_arm(featurizer, train_rows, dev_rows, spec, vspec, ks, n_boot, seed,
            shuffle_seed, equivalence, *, guard=None, controls=True,
            n_retrieve_unlabelled=None):
    """One logreg PER ACT, because that is the pool select._score_templates ranks in."""
    by_act_fit = defaultdict(list)
    for r in train_rows:
        if r.gold_template_id and r.act:
            by_act_fit[r.act].append(r)
    by_act_ev = defaultdict(list)
    for r in dev_rows:
        if r.gold_template_id and r.act:
            by_act_ev[r.act].append(r)

    shuf = RenderSpec(**{**spec.as_dict(), "shuffle": "within_window", "shuffle_seed": shuffle_seed})
    per_act: dict = {}
    micro_hits = {k: [] for k in ks}
    micro_clusters: list = []
    micro_tiers = {"exact": [], "paraphrase": [], "field_safe": []}
    micro_shuf: list = []
    micro_nat: list = []
    dense_states: dict = {}
    pred_by_position: dict = {}
    field_gold_rows = 0
    wrong_field_rows = 0

    for act in sorted(by_act_ev):
        fit, ev = by_act_fit.get(act, []), by_act_ev[act]
        if len(set(r.gold_template_id for r in fit)) < 2 or not ev:
            per_act[act] = {"skipped": "fewer than 2 gold classes in the fit split",
                            "n_fit": len(fit), "n_eval": len(ev)}
            continue
        nat = fit_score(featurizer, fit, [r.gold_template_id for r in fit],
                        ev, [r.gold_template_id for r in ev], spec, spec, vspec, ks,
                        head="h7", guard=guard)
        ss = None
        if controls:
            ss = fit_score(featurizer, fit, [r.gold_template_id for r in fit],
                           ev, [r.gold_template_id for r in ev], shuf, shuf, vspec, ks,
                           head="h7", guard=guard)
        clusters = [r.convo_id for r in ev]
        dense_states[act] = nat["dense_state"]
        const = RM.constant_recall_at_k([r.gold_template_id for r in fit],
                                        [r.gold_template_id for r in ev], ks)
        acc = {f"recall@{k}": RM.accuracy([bool(x) for x in nat["per_row_hits"][k]]) for k in ks}
        ci1 = RM.bootstrap_ci(nat["per_row_hits"][1], clusters, n_boot=n_boot, seed=seed)

        tiers = {"exact": [], "paraphrase": [], "field_safe": []}
        for row, pred in zip(ev, nat["top1"]):
            pred_by_position[(row.turn_id, row.position)] = pred
            flags = equivalence.tier_flags(pred, row.gold_template_id)
            for name in tiers:
                tiers[name].append(1.0 if flags[name] else 0.0)
            if equivalence.fields.get(row.gold_template_id):
                field_gold_rows += 1
                if not flags["field_safe"]:
                    wrong_field_rows += 1

        per_act[act] = {
            "n_candidates": sum(1 for t in equivalence.ids if equivalence.act[t] == act),
            "n_fit": len(fit), "n_eval": len(ev),
            "accuracy": acc, "ci_top1": ci1, "constant": const,
            "tiers": {name: RM.accuracy([bool(x) for x in v]) for name, v in tiers.items()},
            "d5": RM.d5_verdict(acc["recall@1"], const["recall@1"], ci1["lo"]),
            "shuf_to_shuf_REFIT": (
                RM.accuracy([bool(x) for x in ss["per_row_hits"][1]]) if ss else None),
        }
        for k in ks:
            micro_hits[k].extend(nat["per_row_hits"][k])
        micro_clusters.extend(clusters)
        micro_nat.extend(nat["per_row_hits"][1])
        if ss:
            micro_shuf.extend(ss["per_row_hits"][1])
        for name in micro_tiers:
            micro_tiers[name].extend(tiers[name])

    n_all_positions = len(dev_rows)
    n_with_gold = sum(1 for r in dev_rows if r.gold_template_id)
    n_scored = len(micro_hits[1])
    micro_acc = {f"recall@{k}": RM.accuracy([bool(x) for x in micro_hits[k]]) for k in ks}
    const_micro = RM.constant_recall_at_k_grouped(
        [(r.act, r.gold_template_id) for r in train_rows if r.gold_template_id and r.act],
        [(r.act, r.gold_template_id) for r in dev_rows if r.gold_template_id and r.act],
        ks,
    )
    ci1 = RM.bootstrap_ci(micro_hits[1], micro_clusters, n_boot=n_boot, seed=seed)
    # EXACT, not an approximation: a position with no gold, and a position in an
    # act that could not be fit, can never be a hit, so unconditional hits are
    # exactly the conditional hits over the wider denominator.
    scale = (n_scored / n_all_positions) if n_all_positions else 0.0

    return {
        "_pred_by_position": pred_by_position,
        "head": "h7", "spec": spec.label(),
        "denominators": {
            "conditional_n": n_scored,
            "positions_with_a_gold": n_with_gold,
            "positions_dropped_for_an_unfittable_act": n_with_gold - n_scored,
            "unconditional_n": n_all_positions,
            # build_rows emits positions only for retrieve turns that HAVE a
            # TurnLabel (a turn's sentence count is read off its label), so
            # unconditional_n cannot include an unlabelled turn's positions. The
            # count of such turns is reported so that gap is visible, not assumed
            # to be zero. None = the caller did not count it.
            "retrieve_turns_without_a_label": n_retrieve_unlabelled,
            "gold_present_rate": scale,
            "note": "the conditional denominator is optimistic by construction: "
                    "compile.min_template_count dropped exactly the rare phrasings. "
                    "unconditional_n is every sentence position of the retrieve turns "
                    "that have a label; retrieve_turns_without_a_label says how many "
                    "turns that leaves out.",
        },
        "conditional": {"micro": micro_acc, "ci_top1": ci1,
                        "constant": const_micro["micro"],
                        "constant_macro": const_micro["macro"],
                        "d5": RM.d5_verdict(micro_acc["recall@1"],
                                            const_micro["micro"]["recall@1"], ci1["lo"])},
        "unconditional": {
            "micro": {f"recall@{k}": (v * scale if v is not None else None)
                      for k, v in micro_acc.items()},
            "constant": {f"recall@{k}": (v * scale if v is not None else None)
                         for k, v in const_micro["micro"].items()},
            "note": "positions with no gold counted as misses.",
        },
        "tiers_conditional": {name: RM.accuracy([bool(x) for x in v])
                              for name, v in micro_tiers.items()},
        "wrong_field_rate": {
            "n_field_requesting_gold_rows": field_gold_rows,
            "n_wrong_field": wrong_field_rows,
            "rate_over_field_requesting_rows": (
                wrong_field_rows / field_gold_rows) if field_gold_rows else None,
            "rate_over_all_retrieve_positions": (
                wrong_field_rows / n_all_positions) if n_all_positions else None,
            "note": "D16's number. Quote BOTH denominators: the first says how often the "
                    "head gets fields wrong when fields are at stake; the second is the "
                    "share of all turns a customer would feel it on.",
        },
        "controls": {
            "shuf_to_shuf_REFIT_micro": RM.accuracy([bool(x) for x in micro_shuf]),
            "order_effect_points_paired": RM.paired_bootstrap_delta(
                micro_nat, micro_shuf, micro_clusters, n_boot=n_boot, seed=seed),
        } if controls else {"skipped": "order controls not run on this pass"},
        "vectorizer": vspec.as_dict(),
        "dense_state_by_act": dense_states,
        "per_act": per_act,
        "equivalence_bank_facts": equivalence.bank_facts(),
        "paraphrase_tier_density": equivalence.paraphrase_density(),
    }


def _h4_arm(featurizer, train_rows, dev_rows, spec, vspec, ks, n_boot, seed,
            shuffle_seed, value_list_len, *, guard=None, controls=True,
            n_take_action_all=None):
    """H4 in INDEX space; the constant is the modal COLUMN, not the modal string."""
    fit = [r for r in train_rows if r.gold_column >= 0]
    ev = [r for r in dev_rows if r.gold_column >= 0]
    if len(set(r.gold_column for r in fit)) < 2 or not ev:
        return {"head": "h4", "spec": spec.label(),
                "skipped": "fewer than 2 resolvable gold columns"}

    shuf = RenderSpec(**{**spec.as_dict(), "shuffle": "within_window", "shuffle_seed": shuffle_seed})
    nat = fit_score(featurizer, fit, [str(r.gold_column) for r in fit],
                    ev, [str(r.gold_column) for r in ev], spec, spec, vspec, ks,
                    head="h4", guard=guard)
    ss = None
    if controls:
        ss = fit_score(featurizer, fit, [str(r.gold_column) for r in fit],
                       ev, [str(r.gold_column) for r in ev], shuf, shuf, vspec, ks,
                       head="h4", guard=guard)
    clusters = [r.convo_id for r in ev]
    acc = {f"recall@{k}": RM.accuracy([bool(x) for x in nat["per_row_hits"][k]]) for k in ks}
    ci1 = RM.bootstrap_ci(nat["per_row_hits"][1], clusters, n_boot=n_boot, seed=seed)

    const_col = RM.constant_recall_at_k([str(r.gold_column) for r in fit],
                                        [str(r.gold_column) for r in ev], ks)
    const_str = RM.constant_recall_at_k([r.gold_value for r in fit],
                                        [r.gold_value for r in ev], ks)
    bar = max(
        [v for v in (const_col["recall@1"], const_str["recall@1"]) if v is not None] or [0.0]
    )

    # string-level: decode the predicted column back to a string per row
    def _decode(column: int, row) -> str:
        column = int(column)
        if column < value_list_len:
            return f"<enumerable:{column}>"
        offset = column - value_list_len
        return row.candidate_tokens[offset] if offset < len(row.candidate_tokens) else ""
    string_hits = [
        1.0 if _decode(int(pred), row) == _decode(row.gold_column, row) else 0.0
        for row, pred in zip(ev, nat["top1"])
    ]

    return {
        "head": "h4", "spec": spec.label(),
        "vectorizer": vspec.as_dict(), "dense_state": nat["dense_state"],
        "denominators": {
            # dev_rows is rows["h4"], which response_labels.build_rows already
            # filtered to take_action turns that HAVE a gold value list -- ~31% of
            # take_action turns in ABCD v1.1 have an empty one and never become a
            # row. The old name `n_take_action_turns` for len(dev_rows) therefore
            # overstated the population it covered; response_labels.audit names the
            # same set `n_take_action_turns_with_values`.
            "n_take_action_turns": n_take_action_all,
            "n_take_action_turns_with_values": len(dev_rows),
            "n_take_action_turns_dropped_no_values": (
                None if n_take_action_all is None else n_take_action_all - len(dev_rows)),
            "n_resolvable": len(ev),
            "resolvable_rate_over_turns_with_values": (
                len(ev) / len(dev_rows) if dev_rows else None),
            "resolvable_rate_over_all_take_action": (
                len(ev) / n_take_action_all if n_take_action_all else None),
            "scorable_turn_ids": {r.turn_id for r in ev},
            "note": "resolvability depends on the WINDOW (the copy tier reads context.turns). "
                    "Use response_metrics.align_h4_denominator before comparing arms. "
                    "A take_action turn with no gold value has nothing to resolve, so "
                    "resolvable_rate_over_turns_with_values is the rate H4 is answerable on; "
                    "the over_all_take_action rate is the share of the whole take_action "
                    "population H4 covers. n_take_action_turns is None when the caller did "
                    "not pass the split's true take_action count.",
        },
        "index_space": {"accuracy": acc, "ci_top1": ci1},
        "string_space": {"accuracy@1": RM.accuracy([bool(x) for x in string_hits])},
        "constant": {
            "modal_column": const_col, "modal_string": const_str,
            "bar_used": bar,
            "note": "the bar is the LARGER of the two. A copy head's columns are token "
                    "positions and gold values cluster at similar positions, so the modal "
                    "COLUMN can be far stronger than the modal STRING; using the string bar "
                    "alone would understate it.",
        },
        "d5": RM.d5_verdict(acc["recall@1"], bar, ci1["lo"]),
        "controls": {
            "shuf_to_shuf_REFIT": RM.accuracy([bool(x) for x in ss["per_row_hits"][1]]),
            "order_effect_points_paired": RM.paired_bootstrap_delta(
                nat["per_row_hits"][1], ss["per_row_hits"][1], clusters,
                n_boot=n_boot, seed=seed),
        } if controls else {"skipped": "order controls not run on this pass"},
        "position_ceiling": {
            "n_multi_value_turns": sum(1 for r in dev_rows if r.n_gold_values > 1),
            "note": "train.value_position='first' supervises values[0] only. Positions b/c "
                    "of verify-identity and validate-purchase are unmeasurable by H4 as "
                    "defined -- a ceiling on joint accuracy, not a model failure.",
        },
    }


def mode_measure(args, probe_cfg: dict, cfg: dict) -> dict:
    from reflex.compile import load_bank

    if not args.force_uncertified:
        stamp = _read_certification(args.out_dir)
        if not stamp.get("certified"):
            return {
                "refused": True,
                "reason": (
                    "this harness is not certified. Run `validate` first: it reproduces "
                    "DECISIONS D7's nextstep k6/MAXB=3 cell (0.8345-0.8351). Pass "
                    "--force-uncertified to measure anyway; every number will be stamped "
                    "certified:false and must be quoted that way."
                ),
                "certification": stamp,
            }

    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])
    bank = load_bank(cfg)
    equivalence = RM.TemplateEquivalence(bank, cfg)

    indexer = None
    if "h4" in args.heads:
        indexer = _build_h4_indexer(cfg, probe_cfg)

    rows_train = load_split_rows(cfg, "train", bank, indexer, args.cache_dir)
    rows_dev = load_split_rows(cfg, "dev", bank, indexer, args.cache_dir)
    spaces = rows_dev["spaces"]

    # the featurizer gets the skeleton act tuples for the H5 near-miss shape
    try:
        setattr(featurizer, "_skeleton_acts", spaces.skeleton_acts)
    except Exception:
        pass

    vspec = VectorizerSpec(**probe_cfg["probe"]["vectorizer"])
    # Per-head estimator + feature cap. The base `vectorizer` block is the
    # CERTIFIED nextstep setting and stays lbfgs/multinomial; H5/H7/H4 have label
    # spaces two to three orders of magnitude wider and cannot hold an lbfgs
    # history at any feature count this corpus produces, so they take their own
    # entry from probe.vectorizer_by_head. See DenseStateTooLarge.
    head_vspecs = {h: vspec_for_head(probe_cfg, h) for h in ("h5", "h7", "h4")}
    memory_guard = guard_cfg(probe_cfg)
    ks = tuple(probe_cfg["probe"]["recall_ks"])
    out: dict = {
        "certified": bool(_read_certification(args.out_dir).get("certified")),
        "bank_hash": spaces.bank_hash,
        "label_spaces": spaces.summary(),
        "label_provenance": {"train": rows_train["label_provenance"],
                             "dev": rows_dev["label_provenance"]},
        "normalizer_drift": RL.check_normalizer_drift(),
        "dev_gold_provenance": RL.dev_gold_provenance(),
        "audit_dev": RL.audit(rows_dev, cfg, "dev"),
        "featurizer_fingerprint": featurizer.fingerprint(),
        "vectorizer": vspec.as_dict(),
        # Same key name as mode_select. measure runs no sweep, so as-run == the
        # probe.yaml per-head spec here; the config block keeps its own name,
        # probe.vectorizer_by_head.
        "vectorizer_as_run_by_head": {h: v.as_dict() for h, v in head_vspecs.items()},
        "memory_guard": memory_guard,
        "estimator_note": (
            "H5/H7/H4 do NOT use the certified nextstep estimator. nextstep is 3-way so "
            "lbfgs/multinomial costs nothing there; these heads are 226 to 1,066-way and an "
            "lbfgs history at that width does not fit in RAM (ASK alone projects to ~36 GiB). "
            "vectorizer_as_run_by_head records the estimator and the feature cap each number "
            "was produced with -- quote them with the number (D6)."
        ),
        "arms": [],
    }

    h4_denominators: dict = {}
    for spec in _arm_specs(probe_cfg):
        arm: dict = {"spec": spec.label(), "spec_detail": spec.as_dict()}
        sample_rows = rows_dev["h5"][:500]
        sample = [featurizer.render_context(r.context, spec) for r in sample_rows]
        shuffled_spec = RenderSpec(**{**spec.as_dict(),
                                      "shuffle": "within_window",
                                      "shuffle_seed": args.shuffle_seed})
        sample_shuf = [featurizer.render_context(r.context, shuffled_spec) for r in sample_rows]
        min_headroom = float(probe_cfg["probe"].get("min_order_headroom", 0.01))
        probe_vec = featurizer.make_vectorizer(vspec)
        probe_vec.fit_transform(sample)
        arm["order_headroom"] = RM.order_sensitivity_headroom(
            probe_vec, sample, sample_shuf, min_headroom=min_headroom)
        # The figure above is the BASE (nextstep) vectorizer's. The H5/H7/H4
        # numbers in this arm come from head_vspecs, whose feature caps differ, so
        # each head's order control is bounded by ITS OWN vectorizer's headroom.
        arm["order_headroom"]["vectorizer"] = "base probe.vectorizer (NOT a head's)"
        arm["order_headroom_by_head"] = {}
        for h in ("h5", "h7", "h4"):
            if h not in args.heads:
                continue
            head_vec = featurizer.make_vectorizer(head_vspecs[h])
            head_vec.fit_transform(sample)
            arm["order_headroom_by_head"][h] = RM.order_sensitivity_headroom(
                head_vec, sample, sample_shuf, min_headroom=min_headroom)

        if "h5" in args.heads:
            arm["h5"] = _h5_arm(featurizer, rows_train["h5"], rows_dev["h5"], spec,
                                head_vspecs["h5"], ks, args.n_boot, args.seed,
                                args.shuffle_seed, guard=memory_guard,
                                n_retrieve_all=rows_dev.get("n_retrieve_turns"))
        if "h7" in args.heads:
            arm["h7"] = _h7_arm(featurizer, rows_train["h7"], rows_dev["h7"], spec,
                                head_vspecs["h7"], ks, args.n_boot, args.seed,
                                args.shuffle_seed, equivalence, guard=memory_guard,
                                n_retrieve_unlabelled=_n_unlabelled(rows_dev))
        if "h4" in args.heads:
            arm["h4"] = _h4_arm(featurizer, rows_train["h4"], rows_dev["h4"], spec,
                                head_vspecs["h4"], ks, args.n_boot, args.seed,
                                args.shuffle_seed, len(spaces.value_list),
                                guard=memory_guard,
                                n_take_action_all=rows_dev.get("n_take_action_turns"))
            if "denominators" in arm["h4"]:
                h4_denominators[spec.label()] = arm["h4"]["denominators"].pop(
                    "scorable_turn_ids")

        if "h5" in arm and "h7" in arm:
            arm["composed"] = _composed(arm, rows_train, rows_dev, args, seed=args.seed)

        for head in ("h5", "h7"):
            for private in ("_pred_by_turn", "_gold_by_turn", "_pred_by_position"):
                arm.get(head, {}).pop(private, None)
        out["arms"].append(arm)

    if h4_denominators:
        aligned = RM.align_h4_denominator(h4_denominators)
        aligned.pop("intersection", None)
        out["h4_denominator_alignment"] = aligned
    return out


def _composed(arm: dict, rows_train: dict, rows_dev: dict, args, seed: int) -> dict:
    """compose@1 -- the metric that actually describes the reflex.

    The predicted skeleton is exact AND every sentence position's top-1 template
    is exact, i.e. the turn's whole gold template tuple is recovered. That is
    what ``Selection.exact_template_match`` reports, and it is the response
    path's headline because it has the most headroom over a constant: the
    constant recovers a whole tuple far less often than it recovers any single
    position.
    """
    pred_sk = arm["h5"].get("_pred_by_turn", {})
    gold_sk = arm["h5"].get("_gold_by_turn", {})
    pred_tp = arm["h7"].get("_pred_by_position", {})

    turn_rows: dict = defaultdict(list)
    gold_tp: dict = {}
    clusters: dict = {}
    for row in rows_dev["h7"]:
        turn_rows[row.turn_id].append(row.position)
        gold_tp[(row.turn_id, row.position)] = row.gold_template_id
        clusters[row.turn_id] = row.convo_id

    all_turns = list(turn_rows)
    fully_covered = [t for t in all_turns if all(gold_tp[(t, p)] for p in turn_rows[t])]

    def _ok(turn_id: str) -> float:
        if pred_sk.get(turn_id) != gold_sk.get(turn_id) or gold_sk.get(turn_id) is None:
            return 0.0
        for position in turn_rows[turn_id]:
            gold = gold_tp[(turn_id, position)]
            if not gold or pred_tp.get((turn_id, position)) != gold:
                return 0.0
        return 1.0

    cond = [_ok(t) for t in fully_covered]
    uncond = [_ok(t) for t in all_turns]

    fit_tuples = []
    train_turn_rows: dict = defaultdict(list)
    for row in rows_train["h7"]:
        train_turn_rows[row.turn_id].append((row.position, row.gold_template_id))
    for turn_id, items in train_turn_rows.items():
        ids = [t for _, t in sorted(items)]
        if all(ids):
            fit_tuples.append(tuple(ids))
    eval_tuples = [
        tuple(gold_tp[(t, p)] for p in sorted(turn_rows[t])) for t in fully_covered
    ]
    const = RM.constant_recall_at_k(
        [str(t) for t in fit_tuples], [str(t) for t in eval_tuples], (1,)
    )
    const_uncond = (
        (const["recall@1"] or 0.0) * len(fully_covered) / len(all_turns)
    ) if all_turns else None

    ci = RM.bootstrap_ci(cond, [clusters[t] for t in fully_covered],
                         n_boot=args.n_boot, seed=seed)
    return {
        "conditional": {
            "n": len(fully_covered),
            "compose@1": RM.accuracy([bool(x) for x in cond]),
            "constant": const["recall@1"],
            "ci": ci,
            "d5": RM.d5_verdict(RM.accuracy([bool(x) for x in cond]),
                                const["recall@1"], ci["lo"]),
        },
        "unconditional": {
            "n": len(all_turns),
            "compose@1": RM.accuracy([bool(x) for x in uncond]),
            "constant": const_uncond,
            "note": "turns with any uncovered sentence counted as misses.",
        },
        "note": (
            "compose@1 requires the skeleton AND every template position. It is the "
            "lowest-constant metric in this harness and therefore the cleanest headline "
            "for the response path; it is also where an H5 arity error shows up, because "
            "a wrong arity changes how many templates are demanded."
        ),
    }


def _build_h4_indexer(cfg: dict, probe_cfg: dict):
    """H4's column index. Uses models.ModernBERT.value_target_index by default."""
    from reflex.data import load_ontology
    mode = str(probe_cfg["probe"].get("h4_indexer", "model"))
    ontology = load_ontology(cfg)
    if mode == "mirror":
        from transformers import AutoTokenizer
        from reflex.config import get_dotted
        tokenizer = AutoTokenizer.from_pretrained(get_dotted(cfg, "model.encoder"))
        return RL.H4TargetIndexer(
            model=None, tokenizer=tokenizer, ontology=ontology,
            value_context_len=int(get_dotted(cfg, "model.value_context_len")),
            mirror_only=True,
        )
    from reflex.compile import load_bank
    from reflex.models import build_model
    model = build_model(cfg, load_bank(cfg), ontology)
    return RL.H4TargetIndexer(model=model, ontology=ontology, mirror_only=False)


# --------------------------------------------------------------------------- #
# select -- the clean selection protocol (DECISIONS D6, again)
# --------------------------------------------------------------------------- #
#
# WHAT IS WRONG WITH THE CURRENT FLOW
# -----------------------------------
# ``validate`` sweeps six vectorizer settings and keeps the one that scores best
# on DEV; ``measure`` then reports DEV numbers for seven arms, and a reader picks
# the best arm off that table by eye. Both axes are therefore chosen on the same
# split the number is quoted from, and any number quoted that way is
# optimistically biased by the selection itself. This is D6 in a new costume -- a
# baseline is only a baseline WITH its configuration attached, and a number
# chosen on the data it is reported on is not a clean estimate of anything.
#
# THE PROTOCOL
# ------------
# 1. Split TRAIN internally, GROUPED BY CONVERSATION, deterministically under a
#    seed. Never by row: turns in one conversation share the scenario, the
#    customer and the intent, so a row-level split leaks the answer across the
#    boundary and the selection would be made on a contaminated score.
# 2. Sweep the grid on that internal split ONLY. Fit on train-fit, score on
#    train-select, pick the winner there. dev and test are not loaded yet -- not
#    read late, not loaded at all, which is the only version of "did not peek"
#    that can be checked mechanically.
# 3. Refit the winner on the FULL train set and evaluate ONCE on ``test_seen``.
# 4. Report ``dev`` beside it as a FREE held-out check. Under this protocol dev
#    was never used for selection, so it is a consistency read -- labelled as
#    such, never as the headline.
# 5. Report ``test_novel`` separately and never pooled with test_seen. It holds
#    out 5 subflows entirely, it is small, and its clustered interval is wide.
#
# WHAT THIS PROTOCOL DOES NOT FIX
# -------------------------------
# The BANK and the train labels were compiled from the whole of train, including
# the conversations that land in train-select. train-select is therefore not
# clean in the way test_seen is, and its ABSOLUTE level is optimistic. What it is
# clean enough for is RANKING: the bank is identical across every candidate in
# the sweep, so it cannot favour one candidate over another. The headline number
# is taken on test_seen, which the bank never saw.
#
# WHAT IT COSTS, IN FIT PASSES
# ----------------------------
# Counted in TRAIN-PASSES -- one render + vectorize + fit + score over the full
# train row set of one head. These are counts, not wall-clock: no H5/H7/H4 fit
# has ever completed on this corpus (`measure` only ever died with SIGBUS), so
# there is no measured second to scale them by.
#
#   measure (7 arms x 3 order controls)                  21.0 per head
#   select --select-grid full   (42 candidates x 0.8     33.6 sweep
#     of train, + 1 refit per held-out split)           + 3.0 to 9.0 eval
#                                                      = 36.6 to 42.6  -> 1.7-2.0x
#   select --select-grid vectorizers (6 candidates)       7.8  -> 0.37x
#   select --select-grid arms        (7 candidates)       8.6  -> 0.41x
#
# The eval range is the compose@1 winner: when it picks the arm H5 and H7 already
# picked, the memo in mode_select reuses their fits and the eval cost is 3; when
# it picks a different arm, two more fits per held-out split are needed.
# ``--eval-controls`` triples the eval term. Rendering is NOT multiplied by the
# grid: the sweep renders once per ARM and reuses it across every vectorizer
# setting, so a 42-candidate sweep pays 7 render passes, not 42.
#
# So the clean protocol costs roughly TWICE a full `measure` run, and it buys a
# number that can be quoted. The cheap variants exist because they still fix the
# defect on one axis: `vectorizers` is the axis `validate` was selecting on.


def split_convos_grouped(convo_ids: Sequence[Any], select_frac: float, seed: int) -> tuple:
    """Deterministically cut conversation ids into (fit, select).

    Grouping is the whole point: the unit is the CONVERSATION, so every row of a
    conversation -- every H5 turn, every H7 sentence position, every H4 action --
    lands on one side. Sorting before shuffling makes the result independent of
    the order rows arrive in, so the split is a function of (ids, frac, seed) and
    nothing else.
    """
    import random

    ids = sorted({int(c) for c in convo_ids})
    if len(ids) < 2:
        return set(ids), set()
    shuffled = list(ids)
    random.Random(int(seed)).shuffle(shuffled)
    n_select = int(round(len(ids) * float(select_frac)))
    n_select = max(1, min(len(ids) - 1, n_select))
    return set(shuffled[n_select:]), set(shuffled[:n_select])


def rows_for_convos(rows: dict, convo_ids: set) -> dict:
    """Subset a ``{"h5": [...], "h7": [...], "h4": [...]}`` bundle to some conversations."""
    out = {head: [r for r in rows.get(head, []) if int(r.convo_id) in convo_ids]
           for head in ("h5", "h7", "h4")}
    out["spaces"] = rows.get("spaces")
    return out


def selection_candidates(probe_cfg: dict, grid: str) -> list:
    """The (arm, vectorizer-override) pairs the sweep ranks.

    Both axes were previously settled on dev -- the vectorizer by ``validate``'s
    best-on-dev rule, the arm by reading ``measure``'s dev table -- so both
    belong in the sweep. ``--select-grid`` narrows it when the full cross product
    is too expensive; whichever was used is recorded with the result, because a
    winner is only a winner over the set it was chosen from (D6).
    """
    probe = probe_cfg.get("probe") or {}
    arms = [RenderSpec(**entry) for entry in probe["arms"]]
    vectorizers = [dict(entry) for entry in probe["validate_grid"]]
    reference_arm = RenderSpec(**((probe.get("selection") or {}).get(
        "reference_arm") or probe["arms"][0]))
    if grid == "full":
        # Arm OUTERMOST so the sweep renders once per arm and reuses it across
        # every vectorizer setting; the render pass is not the dominant cost but
        # it is not free either.
        return [(a, v) for a in arms for v in vectorizers]
    if grid == "vectorizers":
        return [(reference_arm, v) for v in vectorizers]
    if grid == "arms":
        return [(a, {}) for a in arms]
    raise ValueError(f"unknown --select-grid {grid!r}; expected full|vectorizers|arms")


def _render_for(featurizer, rows: Sequence[Any], spec: RenderSpec, cache: dict, key: str) -> list:
    """Render once per (rows, spec) and reuse across every vectorizer setting."""
    ck = (key, spec.label())
    if ck not in cache:
        cache[ck] = [featurizer.render_context(r.context, spec) for r in rows]
    return cache[ck]


def _compose_flags(pred_sk: dict, gold_sk: dict, pred_tp: dict, h7_rows: Sequence[Any]) -> dict:
    """compose@1 flags without the bootstrap -- the ranking objective for the sweep.

    Same predicate as :func:`_composed`: the skeleton is exact AND every sentence
    position's top-1 template is exact.
    """
    turn_positions: dict = defaultdict(list)
    gold_tp: dict = {}
    clusters: dict = {}
    for row in h7_rows:
        turn_positions[row.turn_id].append(row.position)
        gold_tp[(row.turn_id, row.position)] = row.gold_template_id
        clusters[row.turn_id] = row.convo_id

    def _ok(turn_id: str) -> float:
        if gold_sk.get(turn_id) is None or pred_sk.get(turn_id) != gold_sk.get(turn_id):
            return 0.0
        for position in turn_positions[turn_id]:
            gold = gold_tp[(turn_id, position)]
            if not gold or pred_tp.get((turn_id, position)) != gold:
                return 0.0
        return 1.0

    all_turns = list(turn_positions)
    covered = [t for t in all_turns if all(gold_tp[(t, p)] for p in turn_positions[t])]
    return {
        "conditional": RM.accuracy([bool(_ok(t)) for t in covered]),
        "unconditional": RM.accuracy([bool(_ok(t)) for t in all_turns]),
        "n_conditional": len(covered),
        "n_unconditional": len(all_turns),
    }


def score_candidate(featurizer, rows_fit: dict, rows_sel: dict, spec: RenderSpec,
                    probe_cfg: dict, overrides: dict, heads: Sequence[str],
                    guard: Optional[dict], cache: dict) -> dict:
    """Fit on train-fit, score on train-select. ONE fit per head -- no controls.

    The order controls are a separate question (D7) measured by ``measure`` on
    dev; running them here would triple the sweep for a number selection does not
    read. Every objective is reported beside its label-blind constant, fit on
    train-fit and applied to train-select, so a candidate that only beats another
    candidate without beating a constant is visible at selection time (D5).
    """
    result: dict = {
        "arm": spec.label(), "arm_detail": spec.as_dict(),
        "vectorizer_overrides": dict(overrides), "objectives": {},
    }

    if "h5" in heads:
        fit = [r for r in rows_fit["h5"] if r.gold_skeleton_id]
        sel = [r for r in rows_sel["h5"] if r.gold_skeleton_id]
        if fit and sel:
            vs = vspec_for_head(probe_cfg, "h5", overrides)
            out = fit_score(
                featurizer, fit, [r.gold_skeleton_id for r in fit],
                sel, [r.gold_skeleton_id for r in sel], spec, spec, vs, (1,),
                head="h5", guard=guard,
                fit_texts=_render_for(featurizer, fit, spec, cache, "h5_fit"),
                eval_texts=_render_for(featurizer, sel, spec, cache, "h5_sel"),
            )
            const = RM.constant_recall_at_k([r.gold_skeleton_id for r in fit],
                                            [r.gold_skeleton_id for r in sel], (1,))
            h5_value = RM.accuracy([bool(x) for x in out["per_row_hits"][1]])
            h5_all = len(rows_sel["h5"])
            result["objectives"]["h5"] = {
                "metric": "recall@1 (skeleton top-1) on train-select",
                "value": h5_value,
                "constant": const["recall@1"], "n_fit": out["n_fit"], "n_eval": out["n_eval"],
                "n_features": out["n_features"], "n_classes": out["n_classes"],
                # `value` is CONDITIONAL on a gold skeleton in the bank, like the h7
                # objective below. The denominator is the same for every candidate,
                # so the ranking does not depend on which one is read.
                "conditional_n": out["n_eval"], "unconditional_n": h5_all,
                "unconditional_value": (
                    h5_value * (out["n_eval"] / h5_all)
                    if h5_value is not None and h5_all else None),
            }
            result["_h5_pred"] = {r.turn_id: p for r, p in zip(sel, out["top1"])}
            result["_h5_gold"] = {r.turn_id: r.gold_skeleton_id for r in sel}

    if "h7" in heads:
        by_act_fit: dict = defaultdict(list)
        for r in rows_fit["h7"]:
            if r.gold_template_id and r.act:
                by_act_fit[r.act].append(r)
        by_act_sel: dict = defaultdict(list)
        for r in rows_sel["h7"]:
            if r.gold_template_id and r.act:
                by_act_sel[r.act].append(r)
        vs = vspec_for_head(probe_cfg, "h7", overrides)
        micro: list = []
        pred_tp: dict = {}
        fit_pairs, sel_pairs = [], []
        n_positions_all = len(rows_sel["h7"])
        for act in sorted(by_act_sel):
            fit, sel = by_act_fit.get(act, []), by_act_sel[act]
            if len(set(r.gold_template_id for r in fit)) < 2 or not sel:
                continue
            out = fit_score(
                featurizer, fit, [r.gold_template_id for r in fit],
                sel, [r.gold_template_id for r in sel], spec, spec, vs, (1,),
                head="h7", guard=guard,
                fit_texts=_render_for(featurizer, fit, spec, cache, f"h7_fit_{act}"),
                eval_texts=_render_for(featurizer, sel, spec, cache, f"h7_sel_{act}"),
            )
            micro.extend(out["per_row_hits"][1])
            for row, pred in zip(sel, out["top1"]):
                pred_tp[(row.turn_id, row.position)] = pred
            fit_pairs.extend((r.act, r.gold_template_id) for r in fit)
            sel_pairs.extend((r.act, r.gold_template_id) for r in sel)
        if micro:
            const = RM.constant_recall_at_k_grouped(fit_pairs, sel_pairs, (1,))
            value = RM.accuracy([bool(x) for x in micro])
            scale = len(micro) / n_positions_all if n_positions_all else 0.0
            result["objectives"]["h7"] = {
                "metric": "conditional micro recall@1 (per-act pools) on train-select",
                "value": value, "constant": const["micro"]["recall@1"],
                "conditional_n": len(micro), "unconditional_n": n_positions_all,
                "unconditional_value": (value * scale) if value is not None else None,
                "n_acts_fit": len(set(a for a, _ in sel_pairs)),
            }
            result["_h7_pred"] = pred_tp

    if "h4" in heads:
        fit = [r for r in rows_fit["h4"] if r.gold_column >= 0]
        sel = [r for r in rows_sel["h4"] if r.gold_column >= 0]
        if len(set(r.gold_column for r in fit)) >= 2 and sel:
            vs = vspec_for_head(probe_cfg, "h4", overrides)
            out = fit_score(
                featurizer, fit, [str(r.gold_column) for r in fit],
                sel, [str(r.gold_column) for r in sel], spec, spec, vs, (1,),
                head="h4", guard=guard,
                fit_texts=_render_for(featurizer, fit, spec, cache, "h4_fit"),
                eval_texts=_render_for(featurizer, sel, spec, cache, "h4_sel"),
            )
            col = RM.constant_recall_at_k([str(r.gold_column) for r in fit],
                                          [str(r.gold_column) for r in sel], (1,))
            string = RM.constant_recall_at_k([r.gold_value for r in fit],
                                             [r.gold_value for r in sel], (1,))
            h4_value = RM.accuracy([bool(x) for x in out["per_row_hits"][1]])
            h4_all = len(rows_sel["h4"])
            result["objectives"]["h4"] = {
                "metric": "index-space recall@1 on train-select",
                "value": h4_value,
                "constant": max([v for v in (col["recall@1"], string["recall@1"])
                                 if v is not None] or [0.0]),
                "n_fit": out["n_fit"], "n_eval": out["n_eval"],
                # CONDITIONAL on an in-window gold column. unconditional_n is the
                # take_action turns that carry a value (build_rows already dropped
                # the value-less ones), NOT every take_action turn.
                "conditional_n": out["n_eval"], "unconditional_n": h4_all,
                "unconditional_value": (
                    h4_value * (out["n_eval"] / h4_all)
                    if h4_value is not None and h4_all else None),
            }

    if "_h5_pred" in result and "_h7_pred" in result:
        flags = _compose_flags(result["_h5_pred"], result["_h5_gold"],
                               result["_h7_pred"], rows_sel["h7"])
        result["objectives"]["compose"] = {
            "metric": "conditional compose@1 on train-select",
            "value": flags["conditional"],
            "unconditional_value": flags["unconditional"],
            "conditional_n": flags["n_conditional"],
            "unconditional_n": flags["n_unconditional"],
            "constant": None,
            "constant_note": "the compose@1 constant is computed on the EVAL split at "
                             "report time by _composed; the sweep ranks candidates, and "
                             "the constant is identical across them because it does not "
                             "read the model.",
        }
    for private in ("_h5_pred", "_h5_gold", "_h7_pred"):
        result.pop(private, None)
    return result


def sweep_internal(featurizer, rows_train: dict, candidates: Sequence[tuple],
                   probe_cfg: dict, heads: Sequence[str], select_frac: float,
                   split_seed: int, guard: Optional[dict] = None) -> dict:
    """Rank every candidate on an internal, conversation-grouped split of TRAIN.

    **This function takes TRAIN and nothing else.** There is no dev parameter and
    no test parameter, and it calls no loader, so "selection never reads dev or
    test" is a property of the signature rather than a promise in a docstring.
    """
    convo_ids = {int(r.convo_id) for head in ("h5", "h7", "h4") for r in rows_train.get(head, [])}
    fit_ids, sel_ids = split_convos_grouped(convo_ids, select_frac, split_seed)
    rows_fit = rows_for_convos(rows_train, fit_ids)
    rows_sel = rows_for_convos(rows_train, sel_ids)

    table: list = []
    cache: dict = {}
    last_arm = None
    for spec, overrides in candidates:
        if last_arm is not None and spec.label() != last_arm:
            cache.clear()          # renderings are per-arm; drop them when the arm changes
        last_arm = spec.label()
        table.append(score_candidate(featurizer, rows_fit, rows_sel, spec, probe_cfg,
                                     overrides, heads, guard, cache))
    cache.clear()

    winners: dict = {}
    for head in list(heads) + (["compose"] if {"h5", "h7"} <= set(heads) else []):
        scored = [(i, row) for i, row in enumerate(table)
                  if (row["objectives"].get(head) or {}).get("value") is not None]
        if not scored:
            continue
        # Deterministic tie-break: first candidate in grid order wins a tie.
        best_i, best = min(scored, key=lambda item: (-item[1]["objectives"][head]["value"],
                                                     item[0]))
        obj = best["objectives"][head]
        winners[head] = {
            "candidate_index": best_i,
            "arm": best["arm"], "arm_detail": best["arm_detail"],
            "vectorizer_overrides": best["vectorizer_overrides"],
            # compose@1 spans two heads, and H5 and H7 have different per-head
            # estimator/feature-cap entries, so it reports BOTH rather than
            # picking one and implying the other matched it.
            "vectorizer": (
                vspec_for_head(probe_cfg, head, best["vectorizer_overrides"]).as_dict()
                if head != "compose" else
                {h: vspec_for_head(probe_cfg, h, best["vectorizer_overrides"]).as_dict()
                 for h in ("h5", "h7")}),
            "objective": obj["metric"],
            "objective_value_on_train_select": obj["value"],
            "objective_constant_on_train_select": obj.get("constant"),
            "beats_constant_on_train_select": (
                None if obj.get("constant") is None
                else bool(obj["value"] > obj["constant"])),
            "n_candidates_ranked": len(scored),
            "spread_points": (
                (max(r["objectives"][head]["value"] for _, r in scored)
                 - min(r["objectives"][head]["value"] for _, r in scored)) * 100),
        }

    # Coverage of the select side: a class the fit side never saw cannot be hit,
    # and an unstratified conversation split produces some. Reported, not fixed:
    # stratifying by subflow would make the split a function of the labels.
    def _unseen(head: str, attr: str) -> dict:
        fit_classes = {getattr(r, attr) for r in rows_fit.get(head, []) if getattr(r, attr)}
        sel_rows = [r for r in rows_sel.get(head, []) if getattr(r, attr)]
        unseen = [r for r in sel_rows if getattr(r, attr) not in fit_classes]
        return {"n_select_rows": len(sel_rows), "n_unreachable": len(unseen),
                "unreachable_rate": (len(unseen) / len(sel_rows)) if sel_rows else None}

    return {
        "internal_split": {
            "grouped_by": "conversation",
            "seed": int(split_seed),
            "select_frac_requested": float(select_frac),
            "n_convos_total": len(convo_ids),
            "n_convos_fit": len(fit_ids),
            "n_convos_select": len(sel_ids),
            "select_frac_actual": (len(sel_ids) / len(convo_ids)) if convo_ids else None,
            "n_rows": {head: {"fit": len(rows_fit.get(head, [])),
                              "select": len(rows_sel.get(head, []))}
                       for head in ("h5", "h7", "h4")},
            "disjoint": not (fit_ids & sel_ids),
            "note": "grouped by conversation, never by row: turns in one conversation share "
                    "scenario, customer and intent, so a row-level split leaks.",
        },
        "select_side_class_coverage": {
            "h5_skeleton": _unseen("h5", "gold_skeleton_id"),
            "h7_template": _unseen("h7", "gold_template_id"),
        },
        "sweep": table,
        "winners": winners,
    }


def _dev_selected_config(out_dir: str) -> dict:
    """What ``validate`` picked on DEV -- the thing this protocol is replacing."""
    try:
        with open(os.path.join(out_dir, "validate.json"), "r", encoding="utf-8") as handle:
            recorded = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"available": False,
                "reason": "outputs/probes/response/validate.json not readable"}
    best = recorded.get("best") or {}
    cells = best.get("cells") or []
    headline = [
        {"vectorizer": row["vectorizer"],
         "headline_cell_dev_accuracy": (row.get("cells") or [{}])[0].get("accuracy"),
         "cells_in_band": row.get("cells_in_band")}
        for row in (recorded.get("all") or [])
    ]
    values = [row["headline_cell_dev_accuracy"] for row in headline
              if row["headline_cell_dev_accuracy"] is not None]
    ci = (cells[0].get("ci") or {}) if cells else {}
    return {
        "available": True,
        "selected_vectorizer": best.get("vectorizer"),
        "selection_rule": "argmax over (headline cell in D7's band, cells in band) on DEV",
        "dev_grid": headline,
        "dev_headline_spread_points": ((max(values) - min(values)) * 100) if values else None,
        "dev_clustered_ci_half_width_points": ci.get("half_width_points"),
        "note": "this is the configuration the OLD protocol chose, and it chose it on the "
                "same split it then reported from.",
    }


def _load_bank(cfg: dict):
    """Seam: the compiled bank. Separate so tests can run the protocol without one."""
    from reflex.compile import load_bank
    return load_bank(cfg)


def _equivalence(bank: Any, cfg: dict):
    """Seam: H7's near-duplicate tiers."""
    return RM.TemplateEquivalence(bank, cfg)


def mode_select(args, probe_cfg: dict, cfg: dict) -> dict:
    """Sweep on an internal split of train, refit the winner, score test_seen ONCE."""
    if not args.force_uncertified:
        stamp = _read_certification(args.out_dir)
        if not stamp.get("certified"):
            return {"refused": True, "certification": stamp,
                    "reason": "this harness is not certified; run `validate` first."}

    featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])
    bank = _load_bank(cfg)
    equivalence = _equivalence(bank, cfg)
    indexer = _build_h4_indexer(cfg, probe_cfg) if "h4" in args.heads else None
    memory_guard = guard_cfg(probe_cfg)
    ks = tuple(probe_cfg["probe"]["recall_ks"])

    # ---- phase 1: selection. TRAIN only; dev and test are not loaded yet. ----
    rows_train = load_split_rows(cfg, "train", bank, indexer, args.cache_dir)
    spaces = rows_train["spaces"]
    try:
        setattr(featurizer, "_skeleton_acts", spaces.skeleton_acts)
    except Exception:
        pass
    candidates = selection_candidates(probe_cfg, args.select_grid)
    selection = sweep_internal(featurizer, rows_train, candidates, probe_cfg, args.heads,
                               args.select_frac, args.split_seed, guard=memory_guard)
    selection["grid"] = {
        "mode": args.select_grid, "n_candidates": len(candidates),
        "arms": sorted({spec.label() for spec, _ in candidates}),
        "vectorizer_overrides": [dict(v) for _, v in candidates][: len(candidates)],
    }

    dev_choice = _dev_selected_config(args.out_dir)
    comparison: dict = {"dev_selection": dev_choice, "per_head": {}}
    for head, win in selection["winners"].items():
        chosen = dev_choice.get("selected_vectorizer")
        differs = None
        if chosen:
            shared = {k: win["vectorizer"].get(k) for k in ("min_df", "sublinear_tf", "C")}
            devside = {k: chosen.get(k) for k in ("min_df", "sublinear_tf", "C")}
            differs = shared != devside
        comparison["per_head"][head] = {
            "internal_winner_arm": win["arm"],
            "internal_winner_vectorizer_overrides": win["vectorizer_overrides"],
            "dev_winner_vectorizer": {k: (chosen or {}).get(k)
                                      for k in ("min_df", "sublinear_tf", "C")},
            "differs_from_dev_choice": differs,
        }
    comparison["note"] = (
        "A difference here IS the evidence that the old protocol was biased: the two "
        "protocols saw the same grid and disagreed about the winner, and only one of them "
        "chose on data it does not report from. Note the two selections also optimise "
        "different objectives -- validate ranked by nextstep's distance from D7's recorded "
        "band, this ranks each response head on its own metric -- so a difference is "
        "expected and is not by itself a measure of the bias."
    )

    # ---- phase 2: evaluation. Now, and only now, the held-out splits load. ----
    eval_rows = {name: load_split_rows(cfg, name, bank, indexer, args.cache_dir)
                 for name in ("test_seen", "dev", "test_novel")}
    controls = bool(args.eval_controls)

    report: dict = {}
    reuse_log: dict = {}
    for split_name, rows_eval in eval_rows.items():
        # One (head, arm, vectorizer-override) config is fitted at most ONCE per
        # split. compose@1 is selected over its own shared arm, which is often
        # the same arm H5 or H7 chose; without the memo that config would be
        # refitted, and a refit here is a full pass over train.
        memo: dict = {}

        def _arm_at(head: str, win: dict) -> dict:
            key = (head, win["arm"], json.dumps(win["vectorizer_overrides"], sort_keys=True))
            if key in memo:
                reuse_log[f"{split_name}:{head}"] = "refit avoided (same config as another head)"
                return memo[key]
            spec = RenderSpec(**win["arm_detail"])
            vs = vspec_for_head(probe_cfg, head, win["vectorizer_overrides"])
            if head == "h5":
                built = _h5_arm(featurizer, rows_train["h5"], rows_eval["h5"], spec, vs, ks,
                                args.n_boot, args.seed, args.shuffle_seed,
                                guard=memory_guard, controls=controls,
                                n_retrieve_all=rows_eval.get("n_retrieve_turns"))
            elif head == "h7":
                built = _h7_arm(featurizer, rows_train["h7"], rows_eval["h7"], spec, vs, ks,
                                args.n_boot, args.seed, args.shuffle_seed, equivalence,
                                guard=memory_guard, controls=controls,
                                n_retrieve_unlabelled=_n_unlabelled(rows_eval))
            elif head == "h4":
                built = _h4_arm(featurizer, rows_train["h4"], rows_eval["h4"], spec, vs, ks,
                                args.n_boot, args.seed, args.shuffle_seed,
                                len(spaces.value_list), guard=memory_guard, controls=controls,
                                n_take_action_all=rows_eval.get("n_take_action_turns"))
            else:
                raise ValueError(f"no arm for head {head!r}")
            memo[key] = built
            return built

        block: dict = {"n_convos": len({r.convo_id for r in rows_eval["h5"]}),
                       "label_provenance": rows_eval["label_provenance"]}

        # compose FIRST: it reads the per-turn predictions that are stripped below.
        compose_win = selection["winners"].get("compose")
        if compose_win and {"h5", "h7"} <= set(args.heads):
            pair = {h: _arm_at(h, compose_win) for h in ("h5", "h7")}
            block["compose"] = _composed(pair, rows_train, rows_eval, args, seed=args.seed)
            block["compose"]["selected_config"] = {
                "arm": compose_win["arm"], "arm_detail": compose_win["arm_detail"],
                "vectorizer": compose_win["vectorizer"],
            }
            block["compose"]["selected_on"] = "train-select (internal, conversation-grouped)"

        for head in args.heads:
            win = selection["winners"].get(head)
            if not win:
                continue
            block[head] = _arm_at(head, win)
            block[head]["selected_on"] = "train-select (internal, conversation-grouped)"
            block[head]["selected_config"] = {
                "arm": win["arm"], "arm_detail": win["arm_detail"],
                "vectorizer": win["vectorizer"],
            }

        for entry in list(memo.values()):
            for private in ("_pred_by_turn", "_gold_by_turn", "_pred_by_position"):
                entry.pop(private, None)
        if "h4" in block and "denominators" in block["h4"]:
            block["h4"]["denominators"].pop("scorable_turn_ids", None)
        report[split_name] = block

    return {
        "protocol": "clean-selection (train-internal grouped split -> refit -> test_seen once)",
        "certified": bool(_read_certification(args.out_dir).get("certified")),
        "bank_hash": spaces.bank_hash,
        "label_spaces": spaces.summary(),
        "featurizer_fingerprint": featurizer.fingerprint(),
        "memory_guard": memory_guard,
        # Two different things, and the old single key "vectorizer_by_head" was the
        # BASE one while claiming to be the record of what ran (D6). The base is
        # probe.yaml before the sweep; as-run is base + the head's winning override,
        # i.e. exactly the spec each headline number was produced with.
        "vectorizer_base_by_head": {h: vspec_for_head(probe_cfg, h).as_dict()
                                    for h in ("h5", "h7", "h4")},
        "vectorizer_as_run_by_head": {
            **{h: vspec_for_head(
                probe_cfg, h,
                (selection["winners"].get(h) or {}).get("vectorizer_overrides") or {}).as_dict()
               if selection["winners"].get(h) else None
               for h in ("h5", "h7", "h4")},
            # compose@1 is NOT produced by the per-head winners above: it refits
            # h5 and h7 with the COMPOSE winner's overrides, which can differ.
            "compose": (
                {h: vspec_for_head(
                    probe_cfg, h,
                    selection["winners"]["compose"].get("vectorizer_overrides") or {}).as_dict()
                 for h in ("h5", "h7")}
                if selection["winners"].get("compose") else None),
        },
        "selection": selection,
        "refits_avoided": reuse_log,
        "winner_vs_dev_choice": comparison,
        "headline_test_seen": report.get("test_seen"),
        "free_check_dev": report.get("dev"),
        "test_novel": report.get("test_novel"),
        "reading_order": {
            "headline": "headline_test_seen -- selected on train-select, refit on all of "
                        "train, scored ONCE here. This is the number to quote.",
            "free_check": "free_check_dev -- dev took no part in selection under this "
                          "protocol, so it is a free consistency read. It is NOT a second "
                          "headline and must not be quoted as one; it is also the split the "
                          "OLD protocol selected on, so it is the one most likely to look "
                          "flattering.",
            "novel": "test_novel -- 5 subflows held out ENTIRELY. Never pool it with "
                     "test_seen: they answer different questions (known intent vs unseen "
                     "intent). It is small, so its conversation-clustered interval is wide; "
                     "read the interval, not the point, and quote n_clusters with it.",
            "constants": "every headline number carries its label-blind constant and a D5 "
                         "verdict. A metric the constant wins is DROPPED, not caveated.",
            "config": "every number carries the arm and the vectorizer/estimator it was "
                      "produced with (D6): read it off that head's "
                      "selected_config.vectorizer, or off vectorizer_as_run_by_head, "
                      "which is the same thing collected in one place. Its h5/h7/h4 "
                      "entries cover the PER-HEAD headlines only; the compose headline "
                      "refits h5 and h7 with the compose winner's own overrides, recorded "
                      "as its `compose` entry and at "
                      "headline_test_seen.compose.selected_config.vectorizer -- it is NOT "
                      "the h5 and h7 entries side by side. "
                      "vectorizer_base_by_head is probe.yaml BEFORE the sweep and is not "
                      "what produced any number here. H5/H7/H4 do not use the certified "
                      "nextstep estimator.",
        },
        "known_limits": {
            "bank_saw_train_select": (
                "the bank and labels/train.jsonl were compiled from ALL of train, "
                "train-select included, so train-select's LEVEL is optimistic. Its ranking "
                "is still usable because the bank is identical across every candidate. The "
                "headline is on test_seen, which the bank never saw."
            ),
            "test_gold_provenance": (
                "test_seen/test_novel golds come from reflex.train._derive_turn_labels, "
                "which does NOT delexicalize -- the same asymmetry the dev numbers carry. "
                "It biases H7 coverage DOWNWARD and is bounded by the 49 of 4,417 templates "
                "that bear a slot."
            ),
            "selection_objective": (
                "each head was selected on its OWN metric, because D2b and D7 both record "
                "that the best window differs per head. The winners are therefore allowed "
                "to disagree, and compose@1 is selected jointly over one shared arm."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Certification stamp
# --------------------------------------------------------------------------- #


def _cert_path(out_dir: str) -> str:
    return os.path.join(out_dir, "certification.json")


def _read_certification(out_dir: str) -> dict:
    try:
        with open(_cert_path(out_dir), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"certified": False, "reason": "validate has not been run"}


def _write(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=str)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode",
                        choices=["conformance", "audit", "validate", "measure", "select"])
    parser.add_argument("--probe-config", default=PROBE_CFG_DEFAULT)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="project config override, e.g. --set data.context_turns_K=full")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--heads", default="h5,h7,h4")
    parser.add_argument("--out-dir", default=os.path.join("outputs", "probes", "response"))
    parser.add_argument("--cache-dir", default=os.path.join("outputs", "probes", "cache"))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=17)
    parser.add_argument("--train-convos", type=int, default=0,
                        help="subset train to this many conversations (D7's row set is unrecorded)")
    parser.add_argument("--train-rows", type=int, default=0)
    parser.add_argument("--force-uncertified", action="store_true")
    # ---- select: the clean selection protocol ----
    parser.add_argument("--select-frac", type=float, default=0.2,
                        help="share of TRAIN CONVERSATIONS held out internally for "
                             "hyperparameter selection (select mode)")
    parser.add_argument("--split-seed", type=int, default=13,
                        help="seed for the internal conversation-grouped split. Reuses the "
                             "project's partition-seed convention; any value works, it just "
                             "has to be recorded with the result")
    parser.add_argument("--select-grid", default="full",
                        choices=["full", "vectorizers", "arms"],
                        help="full = arms x validate_grid (both axes were previously chosen "
                             "on dev); vectorizers / arms sweep one axis only")
    parser.add_argument("--eval-controls", action="store_true",
                        help="also run the nat->shuf / shuf->shuf order controls on the "
                             "held-out splits. Triples the evaluation cost; D7's order "
                             "finding is already measured by `measure` on dev")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.heads = [h.strip() for h in args.heads.split(",") if h.strip()]

    probe_cfg = load_probe_cfg(args.probe_config)
    started = time.time()

    if args.mode == "conformance":
        featurizer = load_featurizer(probe_cfg["probe"]["featurizer_factory"])
        result = conformance_report(featurizer)
        path = _write(result, os.path.join(args.out_dir, "conformance.json"))
    else:
        cfg = load_project_cfg(args.overrides)
        if args.mode == "audit":
            from reflex.compile import load_bank
            bank = load_bank(cfg)
            indexer = _build_h4_indexer(cfg, probe_cfg) if "h4" in args.heads else None
            rows = load_split_rows(cfg, args.split, bank, indexer, args.cache_dir)
            result = {
                "audit": RL.audit(rows, cfg, args.split),
                "label_spaces": rows["spaces"].summary(),
                "label_provenance": rows["label_provenance"],
                "normalizer_drift": RL.check_normalizer_drift(),
                "dev_gold_provenance": RL.dev_gold_provenance(),
            }
            path = _write(result, os.path.join(args.out_dir, f"audit_{args.split}.json"))
        elif args.mode == "validate":
            result = mode_validate(args, probe_cfg, cfg)
            path = _write(result, os.path.join(args.out_dir, "validate.json"))
            # The stamp carries the EVIDENCE, not just the boolean: downstream
            # readers (_read_certification, select.json's "certified") saw a bare
            # true and could not tell 1-of-7 cells reproduced from 7-of-7.
            _best = result.get("best") or {}
            _write({"certified": result["certified"],
                    "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "best": _best.get("vectorizer"),
                    "cells_in_band": _best.get("cells_in_band"),
                    "n_cells": len(D7_CELLS),
                    "headline_in_band": _best.get("headline_in_band"),
                    "cells": [{"cell": c.get("cell"), "accuracy": c.get("accuracy"),
                               "band": [c.get("target_lo"), c.get("target_hi")],
                               "in_band": c.get("in_band")}
                              for c in (_best.get("cells") or [])],
                    "n_train_rows": result.get("n_train_rows"),
                    "n_dev_rows": result.get("n_dev_rows")},
                   _cert_path(args.out_dir))
        elif args.mode == "select":
            result = mode_select(args, probe_cfg, cfg)
            path = _write(result, os.path.join(args.out_dir, "select.json"))
        else:
            result = mode_measure(args, probe_cfg, cfg)
            path = _write(result, os.path.join(args.out_dir, "measure.json"))

    print(json.dumps({"mode": args.mode, "wrote": path,
                      "elapsed_s": round(time.time() - started, 1),
                      "certified": result.get("certified"),
                      "refused": result.get("refused", False)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
