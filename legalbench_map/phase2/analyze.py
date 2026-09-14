"""
phase2/analyze.py -- paired cascade analysis (cheap conformal gate + LLM incumbent).

Reads
  FILE A  <per-item-dir>/<task>__<candidate>.csv      (Phase 1 cheap-model OOF dump)
  FILE B  <incumbent-dir>/<task>__<model_key>.csv     (Phase 2 incumbent single pass)
joins them on the PRIMARY KEY (task, item_idx), asserts text_sha256 (and gold)
agree on every joined row, and writes
  <out-dir>/<task>__<model_key>.md / .json     one per (task, incumbent model)
  <out-dir>/summary.md                         one table row per (task, model)

No network, no API calls, no key. Pure numpy/pandas + downshift.stats.

=====================================================================
ASYMMETRY (stated, not hidden)
=====================================================================
The incumbent side is ONE temperature-0 pass over the test split. The cheap
side has `repeats` (normally 3) independent out-of-fold passes from Phase 1's
repeated stratified 5-fold CV. Every paired statistic is therefore computed
once per cheap repeat r -- pairing the cheap model's repeat-r OOF prediction
for each item with the SINGLE incumbent prediction for that same item -- and
then reported per repeat and as the mean across repeats. We chose this
asymmetry (re-running the incumbent 3x buys nothing at temperature 0 and
costs money); we did not hide it.

=====================================================================
LABELS
=====================================================================
  measured        incumbent numbers from this run's FILE B
  CV-estimated    cheap-model numbers from Phase 1's out-of-fold predictions
  published-2023  data/published_scores.csv (LegalBench paper), contrast only

=====================================================================
VERDICT RULES (also written verbatim into summary.md)
=====================================================================
See VERDICT_RULES_TEXT below.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
_DOWNSHIFT_SRC = "/Users/vasyl/zadumai/src"
if _DOWNSHIFT_SRC not in sys.path:
    sys.path.insert(0, _DOWNSHIFT_SRC)

from core.metrics import balanced_accuracy as _core_balanced_accuracy  # noqa: E402
from core.data import load_registry, parse_task_spec  # noqa: E402
from core.verdict import load_published_scores  # noqa: E402
from phase2.legalbench_eval import normalize as _lb_normalize  # noqa: E402
from downshift.stats import (  # noqa: E402
    holm_bonferroni,
    mcnemar_test,
    non_inferiority_test,
    wilson_ci,
)

logger = logging.getLogger("legalbench_map.phase2.analyze")

FILE_A_COLS = ["task", "candidate", "repeat", "fold", "item_idx", "text_sha256",
               "gold", "pred", "p_pred", "kept", "set_size"]
FILE_B_COLS = ["task", "model_key", "item_idx", "text_sha256", "gold", "raw_output",
               "pred", "parse_ok", "in_tokens", "out_tokens", "cached_tokens",
               "cost_usd", "latency_s", "ts"]

DEFAULT_DELTA = 0.01
DEFAULT_ALPHA = 0.05
HOLM_FAMILY = ["paired_all", "sent", "cascade_vs_cheap", "cascade_vs_incumbent"]

ASYMMETRY_NOTE = (
    "ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; "
    "the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). "
    "Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same "
    "single incumbent prediction, and is reported per repeat and as the mean across repeats."
)

VERDICT_RULES_TEXT = """VERDICT RULES (applied verbatim by phase2/analyze.py)

Definitions. NI(x vs y) = downshift.stats.non_inferiority_test(correct_x, correct_y,
margin=delta) with delta = {delta} (95% one-sided, paired item bootstrap); "passes" means
the lower bound of the one-sided CI on acc_x - acc_y lies above -delta. McNemar = mid-p
McNemar on paired per-item correctness. Holm = Holm-Bonferroni at alpha = {alpha} over the
family of 4 McNemar p-values computed for one (task, model, repeat): (3) cheap vs incumbent
on all items, (4) cheap vs incumbent on the SENT subset, (5a) cascade vs cheap-alone,
(5b) cascade vs incumbent-alone. "cheaper" = cascade $/1k < incumbent-alone $/1k, i.e.
LLM share < 1. All numbers are exact-match accuracy on the joined items.

Per-repeat verdict, first rule that fires wins:
  cheap-alone : NI(cheap-alone vs incumbent-alone) passes.  No LLM needed.
  earns       : NI(cascade vs incumbent-alone) passes AND cheaper AND cascade beats
                cheap-alone (cascade acc - cheap acc > 0 AND McNemar (5a) p < {alpha}
                after Holm).
  incumbent   : neither NI(cascade vs incumbent-alone) nor NI(cheap-alone vs
                incumbent-alone) passes.
  unclear     : anything else.

(task, model) verdict: the per-repeat verdict if EVERY cheap repeat agrees; otherwise
'unclear'. Repeat-to-repeat disagreement is itself evidence that the call is not settled."""


class JoinError(ValueError):
    """Raised when FILE A and FILE B do not describe the same items."""


# ---------------------------------------------------------------------------
# loading + join
# ---------------------------------------------------------------------------

def _read_csv_exact(path, expected_cols, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = list(df.columns)
    if cols != expected_cols:
        raise ValueError(
            f"{label} {path} has columns {cols}; the Phase 2 contract requires exactly {expected_cols}"
        )
    return df


def load_file_a(path) -> pd.DataFrame:
    """FILE A: per-item cheap-model OOF dump. Validates the contract:
    one row per (repeat, item_idx); every item_idx appears exactly once per
    repeat; every repeat covers the same item set."""
    df = _read_csv_exact(path, FILE_A_COLS, "FILE A")
    if len(df) == 0:
        raise ValueError(f"FILE A {path} has zero rows")
    for c in ("repeat", "fold", "item_idx", "kept", "set_size"):
        df[c] = df[c].astype(int)
    df["p_pred"] = df["p_pred"].astype(float)
    df["gold"] = df["gold"].str.strip().str.lower()
    df["pred"] = df["pred"].str.strip().str.lower()
    tasks = df["task"].unique()
    if len(tasks) != 1:
        raise ValueError(f"FILE A {path} mixes tasks {list(tasks)}")
    cands = df["candidate"].unique()
    if len(cands) != 1:
        raise ValueError(f"FILE A {path} mixes candidates {list(cands)}")
    item_sets = {}
    for r, g in df.groupby("repeat"):
        if g["item_idx"].duplicated().any():
            dups = sorted(g.loc[g["item_idx"].duplicated(), "item_idx"].unique().tolist())
            raise ValueError(f"FILE A {path} repeat {r}: item_idx appears more than once: {dups[:20]}")
        item_sets[r] = set(g["item_idx"].tolist())
    ref = next(iter(item_sets.values()))
    for r, s in item_sets.items():
        if s != ref:
            raise ValueError(f"FILE A {path}: repeat {r} covers a different item set than the others")
    if not (df["kept"].isin([0, 1])).all():
        raise ValueError(f"FILE A {path}: kept must be 0/1")
    return df


def load_file_b(path) -> pd.DataFrame:
    """FILE B: incumbent per-item predictions. One row per item_idx."""
    df = _read_csv_exact(path, FILE_B_COLS, "FILE B")
    if len(df) == 0:
        raise ValueError(f"FILE B {path} has zero rows")
    for c in ("item_idx", "parse_ok", "in_tokens", "out_tokens", "cached_tokens"):
        df[c] = df[c].astype(int)
    for c in ("cost_usd", "latency_s"):
        df[c] = df[c].astype(float)
    df["gold"] = df["gold"].str.strip().str.lower()
    # pred is already LegalBench-normalized by the runner; we do NOT coerce it.
    tasks = df["task"].unique()
    if len(tasks) != 1:
        raise ValueError(f"FILE B {path} mixes tasks {list(tasks)}")
    models = df["model_key"].unique()
    if len(models) != 1:
        raise ValueError(f"FILE B {path} mixes model_keys {list(models)}")
    if df["item_idx"].duplicated().any():
        dups = sorted(df.loc[df["item_idx"].duplicated(), "item_idx"].unique().tolist())
        raise ValueError(f"FILE B {path}: item_idx appears more than once: {dups[:20]}")
    return df


SCORING_NOTE = {
    "exact": ("exact: LegalBench's verbatim exact-match rule -- the whole normalized output must equal a "
              "label. Benchmark-faithful; a correct answer wrapped in any other words counts as wrong."),
    "lenient": ("lenient: the earliest whole-word label occurrence in the normalized output (ties -> longest "
                "label); no label anywhere -> unparseable, counts as wrong. Reported alongside exact, never "
                "instead of it. Measures whether the model KNOWS the answer rather than whether it obeys a "
                "2023 completion-style format; the cascade decision uses this scoring."),
}


def lenient_pred(raw_output, classes) -> tuple[str, int]:
    """Earliest whole-word occurrence of any class label in the LegalBench-
    normalized output. Returns (pred, parse_ok). On a miss, pred is the
    normalized raw string (never coerced), parse_ok=0 -- same semantics as
    the runner's exact parse, just a looser match."""
    import re
    raw = raw_output if raw_output is not None else ""
    text = _lb_normalize(raw, stem=False)
    if str(raw).startswith("<error:"):   # runner's API-error marker: never a model answer
        return text, 0
    best = None  # (position, -len, label)
    for c in classes:
        cn = _lb_normalize(str(c), stem=False)
        if not cn:
            continue
        m = re.search(r"(?<!\w)" + re.escape(cn) + r"(?!\w)", text)
        if m:
            key = (m.start(), -len(cn), cn)
            if best is None or key < best:
                best = key
    if best is None:
        return text, 0
    return best[2], 1


def add_lenient_columns(b: pd.DataFrame, classes) -> pd.DataFrame:
    """Derive pred_lenient / parse_ok_lenient from the stored raw_output.
    Computed at analysis time so the runner stays a pure recorder and FILE B's
    contract is unchanged."""
    b = b.copy()
    parsed = [lenient_pred(r, classes) for r in b["raw_output"].astype(str).tolist()]
    b["pred_lenient"] = [pr for pr, _ in parsed]
    b["parse_ok_lenient"] = [ok for _, ok in parsed]
    return b


def join_files(a: pd.DataFrame, b: pd.DataFrame, task: str | None = None):
    """Join FILE A (all repeats) with FILE B on (task, item_idx).

    Returns (merged DataFrame, info dict). Rules:
      * task columns must agree (and match `task` if given).
      * FILE B's item_idx set must be the contiguous prefix {0..n_b-1}. Any
        gap is an error ("missing item_idx in FILE B").
      * Every FILE B item must exist in FILE A.
      * If FILE B covers strictly fewer items than FILE A (the runner's
        --limit N semantics, item_idx < N), the join restricts to FILE B's
        items and info['pilot'] is True. The caller stamps outputs.
      * text_sha256 and gold must agree on every joined row.
    """
    a_task = a["task"].iloc[0]
    b_task = b["task"].iloc[0]
    if a_task != b_task:
        raise JoinError(f"FILE A task {a_task!r} != FILE B task {b_task!r}")
    if task is not None and a_task != task:
        raise JoinError(f"files are for task {a_task!r}, expected {task!r}")

    b_idx = set(b["item_idx"].tolist())
    n_b = len(b_idx)
    expected = set(range(n_b))
    if b_idx != expected:
        missing = sorted(expected - b_idx)
        raise JoinError(
            f"FILE B ({b_task}__{b['model_key'].iloc[0]}) has {n_b} rows but its item_idx set is not "
            f"the contiguous prefix 0..{n_b - 1}: missing item_idx {missing[:20]}"
            f"{' ...' if len(missing) > 20 else ''}. FILE B must cover every item (or, for a pilot, "
            f"exactly item_idx < N for some N)."
        )
    a_idx = set(a["item_idx"].tolist())
    extra = sorted(b_idx - a_idx)
    if extra:
        raise JoinError(
            f"FILE B has item_idx not present in FILE A: {extra[:20]}{' ...' if len(extra) > 20 else ''}"
        )
    missing_in_b = sorted(a_idx - b_idx)
    pilot = len(missing_in_b) > 0
    if pilot:
        logger.warning(
            "FILE B covers %d of FILE A's %d items (item_idx < %d): treating as PILOT and restricting the join",
            n_b, len(a_idx), n_b,
        )
        a = a[a["item_idx"].isin(b_idx)]

    merged = a.merge(b, on=["task", "item_idx"], how="inner", suffixes=("_cheap", "_inc"), validate="m:1")
    per_repeat_counts = merged.groupby("repeat").size()
    if not (per_repeat_counts == n_b).all():
        raise JoinError(f"join produced uneven per-repeat row counts: {per_repeat_counts.to_dict()}")

    sha_bad = merged["text_sha256_cheap"] != merged["text_sha256_inc"]
    if sha_bad.any():
        bad_rows = merged.loc[sha_bad, ["item_idx", "text_sha256_cheap", "text_sha256_inc"]].drop_duplicates("item_idx")
        raise JoinError(
            f"text_sha256 mismatch on {bad_rows['item_idx'].nunique()} item(s) between FILE A and FILE B "
            f"(task {a_task}); first: item_idx={int(bad_rows.iloc[0]['item_idx'])} "
            f"A={bad_rows.iloc[0]['text_sha256_cheap'][:12]}... B={bad_rows.iloc[0]['text_sha256_inc'][:12]}... "
            f"-- the two files do not describe the same test split."
        )
    gold_bad = merged["gold_cheap"] != merged["gold_inc"]
    if gold_bad.any():
        bad = sorted(merged.loc[gold_bad, "item_idx"].unique().tolist())
        raise JoinError(f"gold label mismatch between FILE A and FILE B on item_idx {bad[:20]}")

    merged = merged.sort_values(["repeat", "item_idx"]).reset_index(drop=True)
    info = {
        "task": a_task,
        "candidate": a["candidate"].iloc[0],
        "model_key": b["model_key"].iloc[0],
        "n_joined": n_b,
        "n_file_a_items": len(a_idx),
        "pilot": pilot,
        "repeats": sorted(merged["repeat"].unique().tolist()),
    }
    return merged, info


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def balanced_accuracy(gold, pred, classes) -> float:
    """Phase 1's balanced accuracy (core/metrics.py), extended so that a
    prediction outside the class list (an incumbent parse failure, kept
    verbatim per the contract) counts as wrong for its true class instead
    of raising. Extra labels have zero support in gold, so core's
    'drop unseen classes' step ignores them -- identical to sklearn's
    balanced_accuracy_score on the same inputs, and identical to core's
    when every pred is a class label."""
    gold = list(gold)
    pred = list(pred)
    classes = list(classes)
    extra = sorted(set(pred) - set(classes))
    return _core_balanced_accuracy(gold, pred, classes + extra)


def _acc(correct) -> float:
    c = np.asarray(correct, dtype=bool)
    return float(c.mean()) if c.size else float("nan")


def _paired(correct_a, correct_b, n_boot, seed) -> dict:
    r = mcnemar_test(np.asarray(correct_a, bool), np.asarray(correct_b, bool), n_boot=n_boot, seed=seed)
    return {
        "acc_a": r.acc_a, "acc_b": r.acc_b, "diff": r.diff,
        "n_both_right": r.n_both_right, "n_both_wrong": r.n_both_wrong,
        "n_a_only": r.n_a_only, "n_b_only": r.n_b_only, "n_discordant": r.n_discordant,
        "diff_ci_lo": r.diff_ci_lo, "diff_ci_hi": r.diff_ci_hi,
        "p_value": r.p_value, "test_used": r.test_used,
    }


def _ni(correct_cand, correct_base, delta, n_boot, seed) -> dict:
    r = non_inferiority_test(np.asarray(correct_cand, bool), np.asarray(correct_base, bool),
                             margin=delta, n_boot=n_boot, seed=seed)
    return {
        "candidate_acc": r.candidate_acc, "baseline_acc": r.baseline_acc, "diff": r.diff,
        "margin": r.margin, "diff_ci_lo": r.diff_ci_lo, "diff_ci_hi": r.diff_ci_hi,
        "p_value": r.p_value, "passes": bool(r.passes), "verdict": r.verdict,
    }


def incumbent_alone_stats(b: pd.DataFrame, classes, pred_col: str = "pred", ok_col: str = "parse_ok") -> dict:
    """(1) Incumbent alone, measured. `b` = FILE B rows restricted to the joined items.
    pred_col/ok_col select the scoring: ("pred","parse_ok") = exact, ("pred_lenient","parse_ok_lenient") = lenient."""
    gold = b["gold"].to_numpy()
    pred = b[pred_col].to_numpy()
    correct = gold == pred
    n = int(len(b))
    ci = wilson_ci(int(correct.sum()), n)
    total_cost = float(b["cost_usd"].sum())
    # Rows the runner wrote for a permanently failed API call ('<error: Name>').
    # These are NOT model answers; counting them as misses would fabricate a
    # 'measured' accuracy. analyze_pair stamps the pair invalid when n > 0.
    n_api_errors = int(b["raw_output"].astype(str).str.startswith("<error:").sum())
    return {
        "label": "measured",
        "n": n,
        "n_api_errors": n_api_errors,
        "accuracy": _acc(correct),
        "accuracy_wilson95_lo": ci.lo,
        "accuracy_wilson95_hi": ci.hi,
        "balanced_accuracy": balanced_accuracy(gold, pred, classes),
        "parse_fail_rate": float(1.0 - b[ok_col].mean()),
        "n_parse_fail": int((b[ok_col] == 0).sum()),
        "mean_in_tokens": float(b["in_tokens"].mean()),
        "mean_out_tokens": float(b["out_tokens"].mean()),
        "mean_cached_tokens": float(b["cached_tokens"].mean()),
        "mean_latency_s": float(b["latency_s"].mean()),
        "total_cost_usd": total_cost,
        "cost_per_1k_items_usd": total_cost / n * 1000.0 if n else float("nan"),
    }


def oracle_gate(correct_cheap, correct_inc) -> dict:
    """(6) Route exactly the cheap model's errors to the incumbent."""
    cc = np.asarray(correct_cheap, bool)
    ci = np.asarray(correct_inc, bool)
    correct = np.where(cc, True, ci)
    return {
        "accuracy": _acc(correct),
        "llm_share": float(1.0 - cc.mean()) if cc.size else float("nan"),
        "n_sent": int((~cc).sum()),
    }


def sent_subset_stats(correct_cheap, correct_inc, kept, n_boot, seed) -> dict:
    """(4) Statistics over kept == 0 rows only."""
    kept = np.asarray(kept, int)
    cc = np.asarray(correct_cheap, bool)
    ci = np.asarray(correct_inc, bool)
    mask = kept == 0
    n_sent = int(mask.sum())
    out = {
        "n_sent": n_sent,
        "sent_share": float(mask.mean()) if mask.size else float("nan"),
        "incumbent_accuracy": _acc(ci[mask]) if n_sent else float("nan"),
        "cheap_accuracy": _acc(cc[mask]) if n_sent else float("nan"),
        "incumbent_accuracy_on_kept": _acc(ci[~mask]) if (~mask).sum() else float("nan"),
        "cheap_accuracy_on_kept": _acc(cc[~mask]) if (~mask).sum() else float("nan"),
        "mcnemar_cheap_vs_incumbent": None,
    }
    if n_sent:
        out["mcnemar_cheap_vs_incumbent"] = _paired(cc[mask], ci[mask], n_boot, seed)
    return out


def cascade_stats(gold, pred_cheap, pred_inc, kept, classes, inc_cost_per_1k, delta, n_boot, seed) -> dict:
    """(5) Cascade: cheap answer where kept == 1, incumbent answer otherwise."""
    gold = np.asarray(gold)
    pred_cheap = np.asarray(pred_cheap)
    pred_inc = np.asarray(pred_inc)
    kept = np.asarray(kept, int)
    correct_cheap = gold == pred_cheap
    correct_inc = gold == pred_inc
    correct_casc = np.where(kept == 1, correct_cheap, correct_inc)
    pred_casc = np.where(kept == 1, pred_cheap, pred_inc)
    share = float(1.0 - kept.mean()) if kept.size else float("nan")
    return {
        "accuracy": _acc(correct_casc),
        "balanced_accuracy": balanced_accuracy(gold, pred_casc, classes),
        "llm_share": share,
        "n_sent": int((kept == 0).sum()),
        "cost_per_1k_items_usd": share * inc_cost_per_1k,
        "mcnemar_vs_cheap": _paired(correct_casc, correct_cheap, n_boot, seed),
        "ni_vs_cheap": _ni(correct_casc, correct_cheap, delta, n_boot, seed),
        "mcnemar_vs_incumbent": _paired(correct_casc, correct_inc, n_boot, seed),
        "ni_vs_incumbent": _ni(correct_casc, correct_inc, delta, n_boot, seed),
    }


def per_repeat_verdict(cheap_ni_passes, casc_ni_passes, cheaper, casc_beats_cheap) -> str:
    if cheap_ni_passes:
        return "cheap-alone"
    if casc_ni_passes and cheaper and casc_beats_cheap:
        return "earns"
    if not casc_ni_passes and not cheap_ni_passes:
        return "incumbent"
    return "unclear"


def repeat_stats(gold, pred_cheap, pred_inc, kept, classes, inc_cost_per_1k,
                 delta=DEFAULT_DELTA, alpha=DEFAULT_ALPHA, n_boot=10_000, seed=0) -> dict:
    """Everything for one (task, model, repeat): steps (2)-(7) of the spec."""
    gold = np.asarray(gold)
    pred_cheap = np.asarray(pred_cheap)
    pred_inc = np.asarray(pred_inc)
    kept = np.asarray(kept, int)
    correct_cheap = gold == pred_cheap
    correct_inc = gold == pred_inc

    cheap = {
        "label": "CV-estimated",
        "accuracy": _acc(correct_cheap),
        "balanced_accuracy": balanced_accuracy(gold, pred_cheap, classes),
        "keep_rate": float(kept.mean()) if kept.size else float("nan"),
    }
    paired_all = {
        "mcnemar_cheap_vs_incumbent": _paired(correct_cheap, correct_inc, n_boot, seed),
        "ni_cheap_vs_incumbent": _ni(correct_cheap, correct_inc, delta, n_boot, seed),
    }
    sent = sent_subset_stats(correct_cheap, correct_inc, kept, n_boot, seed)
    casc = cascade_stats(gold, pred_cheap, pred_inc, kept, classes, inc_cost_per_1k, delta, n_boot, seed)
    oracle = oracle_gate(correct_cheap, correct_inc)

    # (7) Holm across the four McNemar p-values.
    raw = {
        "paired_all": paired_all["mcnemar_cheap_vs_incumbent"]["p_value"],
        "sent": sent["mcnemar_cheap_vs_incumbent"]["p_value"] if sent["mcnemar_cheap_vs_incumbent"] else None,
        "cascade_vs_cheap": casc["mcnemar_vs_cheap"]["p_value"],
        "cascade_vs_incumbent": casc["mcnemar_vs_incumbent"]["p_value"],
    }
    names = [k for k in HOLM_FAMILY if raw[k] is not None]
    reject = holm_bonferroni([raw[k] for k in names], alpha=alpha)
    holm = {
        "alpha": alpha,
        "family_size": len(names),
        "raw_p": raw,
        "survives": {k: (bool(reject[names.index(k)]) if k in names else None) for k in HOLM_FAMILY},
    }

    cheaper = casc["llm_share"] < 1.0
    beats_cheap = (casc["mcnemar_vs_cheap"]["diff"] > 0) and bool(holm["survives"]["cascade_vs_cheap"])
    verdict = per_repeat_verdict(
        paired_all["ni_cheap_vs_incumbent"]["passes"],
        casc["ni_vs_incumbent"]["passes"],
        cheaper,
        beats_cheap,
    )
    return {
        "n": int(gold.size),
        "cheap_alone": cheap,
        "paired_all": paired_all,
        "sent_subset": sent,
        "cascade": casc,
        "oracle_gate": oracle,
        "holm": holm,
        "verdict_inputs": {
            "cheap_ni_passes": bool(paired_all["ni_cheap_vs_incumbent"]["passes"]),
            "cascade_ni_passes": bool(casc["ni_vs_incumbent"]["passes"]),
            "cheaper": bool(cheaper),
            "cascade_beats_cheap_after_holm": bool(beats_cheap),
        },
        "verdict": verdict,
    }


def _is_num(v):
    return isinstance(v, (bool, int, float, np.integer, np.floating)) and not isinstance(v, str)


def mean_over_repeats(dicts: list[dict]) -> dict:
    """Recursive mean of every numeric leaf (bools -> fraction true).
    Strings are kept only if identical across repeats; None-bearing leaves
    are averaged over the non-None values (or dropped if all None)."""
    if not dicts:
        return {}
    out = {}
    keys = []
    for d in dicts:
        for k in d:
            if k not in keys:
                keys.append(k)
    for k in keys:
        vals = [d.get(k) for d in dicts]
        present = [v for v in vals if v is not None]
        if not present:
            out[k] = None
        elif all(isinstance(v, dict) for v in present):
            out[k] = mean_over_repeats(present)
        elif all(_is_num(v) for v in present):
            arr = np.asarray([float(v) for v in present], dtype=float)
            out[k] = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
        elif all(isinstance(v, str) for v in present) and len(set(present)) == 1:
            out[k] = present[0]
        else:
            out[k] = None
    return out


# ---------------------------------------------------------------------------
# per-(task, model) analysis
# ---------------------------------------------------------------------------

def analyze_pair(a: pd.DataFrame, b: pd.DataFrame, classes, *, n_test_expected=None,
                 published=None, delta=DEFAULT_DELTA, alpha=DEFAULT_ALPHA,
                 n_boot=10_000, seed=0, extra_meta=None, scoring: str = "exact") -> dict:
    if scoring not in SCORING_NOTE:
        raise ValueError(f"scoring must be one of {list(SCORING_NOTE)}, got {scoring!r}")
    if scoring == "lenient" and "pred_lenient" not in b.columns:
        b = add_lenient_columns(b, classes)
    pred_col, ok_col = ("pred", "parse_ok") if scoring == "exact" else ("pred_lenient", "parse_ok_lenient")
    merged, info = join_files(a, b)
    n = info["n_joined"]
    n_test = int(n_test_expected) if n_test_expected is not None else info["n_file_a_items"]
    pilot = info["pilot"] or n < n_test

    b_joined = b[b["item_idx"] < n].sort_values("item_idx").reset_index(drop=True)
    inc = incumbent_alone_stats(b_joined, classes, pred_col, ok_col)
    inc_pred_key = "pred_inc" if scoring == "exact" else "pred_lenient"

    per_repeat = {}
    for r in info["repeats"]:
        g = merged[merged["repeat"] == r]
        per_repeat[str(r)] = repeat_stats(
            g["gold_cheap"].to_numpy(), g["pred_cheap"].to_numpy(), g[inc_pred_key].to_numpy(),
            g["kept"].to_numpy(), classes, inc["cost_per_1k_items_usd"],
            delta=delta, alpha=alpha, n_boot=n_boot, seed=seed + int(r),
        )
    reps = list(per_repeat.values())
    mean = mean_over_repeats(reps)
    verdicts = [x["verdict"] for x in reps]
    verdict = verdicts[0] if len(set(verdicts)) == 1 else "unclear"
    api_error_stamp = None
    if inc["n_api_errors"] > 0:
        api_error_stamp = (f"INVALID: {inc['n_api_errors']} of {n} incumbent rows are API errors "
                           f"('<error: ...>' from the runner), not model answers -- re-run "
                           f"phase2/run_incumbents.py (it retries error rows on resume)")
        verdict = "invalid-api-errors"

    pub = None
    if published and info["task"] in published:
        p = published[info["task"]]
        pub = {"label": "published-2023", "best_score": p["best_score"], "best_model": p["best_model"],
               "metric": p["metric"], "source_url": p["best_source_url"]}

    result = {
        "task": info["task"],
        "model_key": info["model_key"],
        "scoring": scoring,
        "scoring_note": SCORING_NOTE[scoring],
        "candidate": info["candidate"],
        "classes": list(classes),
        "n_items": n,
        "n_test_expected": n_test,
        "pilot": bool(pilot),
        "pilot_stamp": f"PILOT (n={n}) -- not evidence" if pilot else None,
        "api_error_stamp": api_error_stamp,
        "repeats": info["repeats"],
        "delta": delta,
        "holm_alpha": alpha,
        "n_boot": n_boot,
        "seed": seed,
        "asymmetry_note": ASYMMETRY_NOTE,
        "incumbent_alone": inc,
        "published_2023": pub,
        "per_repeat": per_repeat,
        "mean_over_repeats": mean,
        "per_repeat_verdicts": verdicts,
        "verdict": verdict,
    }
    if extra_meta:
        result["meta"] = extra_meta
    return result


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "nan"
    return f"{xf:.{nd}f}"


def _pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * float(x):.2f}%"


def _paired_line(p: dict | None) -> str:
    if p is None:
        return "n/a (empty subset)"
    return (f"diff={_f(p['diff'])} CI95=[{_f(p['diff_ci_lo'])}, {_f(p['diff_ci_hi'])}], "
            f"a-only={_f(p['n_a_only'])} b-only={_f(p['n_b_only'])} discordant={_f(p['n_discordant'])}, "
            f"mid-p={_f(p['p_value'])}")


def _ni_line(q: dict) -> str:
    return (f"diff={_f(q['diff'])} one-sided-lo={_f(q['diff_ci_lo'])} margin={_f(q['margin'])} "
            f"passes={_f(q['passes'])} ({q.get('verdict', '')})")


def _repeat_section(title: str, s: dict, is_mean: bool) -> list[str]:
    L = [f"### {title}", ""]
    c = s["cheap_alone"]
    L.append(f"- (2) Cheap alone [CV-estimated]: acc={_pct(c['accuracy'])}, "
             f"bal_acc={_pct(c['balanced_accuracy'])}, keep_rate={_pct(c['keep_rate'])}")
    pa = s["paired_all"]
    L.append(f"- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): "
             f"{_paired_line(pa['mcnemar_cheap_vs_incumbent'])}")
    L.append(f"  - non-inferiority cheap vs incumbent: {_ni_line(pa['ni_cheap_vs_incumbent'])}")
    ss = s["sent_subset"]
    L.append(f"- (4) SENT subset (kept==0): n_sent={_f(ss['n_sent'], 1)} ({_pct(ss['sent_share'])}); "
             f"incumbent acc on sent={_pct(ss['incumbent_accuracy'])}, cheap acc on sent={_pct(ss['cheap_accuracy'])}; "
             f"(on kept: incumbent={_pct(ss['incumbent_accuracy_on_kept'])}, cheap={_pct(ss['cheap_accuracy_on_kept'])})")
    L.append(f"  - McNemar on sent, cheap (a) vs incumbent (b): {_paired_line(ss['mcnemar_cheap_vs_incumbent'])}")
    ca = s["cascade"]
    L.append(f"- (5) Cascade: acc={_pct(ca['accuracy'])}, bal_acc={_pct(ca['balanced_accuracy'])}, "
             f"LLM share={_pct(ca['llm_share'])}, $/1k items={_f(ca['cost_per_1k_items_usd'])} (measured incumbent $/1k x share)")
    L.append(f"  - (5a) vs cheap-alone: McNemar {_paired_line(ca['mcnemar_vs_cheap'])}; NI {_ni_line(ca['ni_vs_cheap'])}")
    L.append(f"  - (5b) vs incumbent-alone: McNemar {_paired_line(ca['mcnemar_vs_incumbent'])}; NI {_ni_line(ca['ni_vs_incumbent'])}")
    o = s["oracle_gate"]
    L.append(f"- (6) Oracle gate (route exactly the cheap errors): acc={_pct(o['accuracy'])}, "
             f"LLM share={_pct(o['llm_share'])} -- the ceiling for ANY gate on this cheap model")
    h = s["holm"]
    surv = h["survives"]
    if is_mean:
        L.append("- (7) Holm-Bonferroni survival (fraction of repeats): " +
                 ", ".join(f"{k}={_f(surv[k], 2)}" for k in HOLM_FAMILY))
        L.append(f"- verdict inputs (fraction of repeats): " +
                 ", ".join(f"{k}={_f(v, 2)}" for k, v in s["verdict_inputs"].items()))
    else:
        L.append("- (7) Holm-Bonferroni (alpha=%s, family=%d): " % (_f(h['alpha'], 2), h['family_size']) +
                 ", ".join(f"{k}: p={_f(h['raw_p'][k])} survives={_f(surv[k])}" for k in HOLM_FAMILY))
        L.append(f"- verdict inputs: " + ", ".join(f"{k}={_f(v)}" for k, v in s["verdict_inputs"].items()))
        L.append(f"- **per-repeat verdict: {s['verdict']}**")
    L.append("")
    return L


def render_pair_md(res: dict) -> str:
    L = []
    if res.get("api_error_stamp"):
        L += [f"# {res['api_error_stamp']}", ""]
    if res["pilot"]:
        L += [f"# {res['pilot_stamp']}", ""]
    L += [f"# Phase 2 paired cascade analysis: {res['task']} x {res['model_key']} "
          f"[scoring: {res.get('scoring', 'exact')}]", "",
          f"- scoring: {res.get('scoring_note', SCORING_NOTE['exact'])}", ""]
    if res.get("api_error_stamp"):
        L += [f"**{res['api_error_stamp']}** -- every incumbent number below is meaningless.", ""]
    if res["pilot"]:
        L += [f"**{res['pilot_stamp']}** -- joined items {res['n_items']} < n_test {res['n_test_expected']}.", ""]
    L += [f"- task: `{res['task']}`  incumbent: `{res['model_key']}`  cheap candidate: `{res['candidate']}` "
          f"(best_candidate from configs)",
          f"- items joined on (task, item_idx): {res['n_items']} (n_test expected {res['n_test_expected']}); "
          f"text_sha256 and gold asserted equal on every joined row",
          f"- cheap repeats: {res['repeats']}; delta={res['delta']}; Holm alpha={res['holm_alpha']}; "
          f"bootstrap n_boot={res['n_boot']}, seed={res['seed']}",
          f"- {ASYMMETRY_NOTE}", ""]
    inc = res["incumbent_alone"]
    L += ["## (1) Incumbent alone [measured, this run]", "",
          f"- accuracy={_pct(inc['accuracy'])} Wilson95=[{_pct(inc['accuracy_wilson95_lo'])}, {_pct(inc['accuracy_wilson95_hi'])}] "
          f"(n={inc['n']})",
          f"- balanced accuracy={_pct(inc['balanced_accuracy'])}",
          f"- parse-fail rate={_pct(inc['parse_fail_rate'])} ({inc['n_parse_fail']} items; unparseable output counts as wrong, never coerced)",
          f"- API-error rows={inc.get('n_api_errors', 0)} (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)",
          f"- mean tokens: in={_f(inc['mean_in_tokens'], 1)} out={_f(inc['mean_out_tokens'], 1)} cached={_f(inc['mean_cached_tokens'], 1)}; "
          f"mean latency={_f(inc['mean_latency_s'], 2)}s",
          f"- total cost={_f(inc['total_cost_usd'], 4)} USD; measured $/1k items={_f(inc['cost_per_1k_items_usd'], 4)}"]
    pub = res["published_2023"]
    if pub:
        L.append(f"- published-2023 best for this task (contrast only, not this run): {pub['metric']}={_pct(pub['best_score'])} "
                 f"by {pub['best_model']} ({pub['source_url']})")
    else:
        L.append("- published-2023: no published score found for this task")
    L.append("")
    L += ["## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)", ""]
    for r, s in res["per_repeat"].items():
        L += _repeat_section(f"repeat {r}", s, is_mean=False)
    L += _repeat_section("MEAN across repeats", res["mean_over_repeats"], is_mean=True)
    L += ["## Verdict", "",
          f"- per-repeat verdicts: {res['per_repeat_verdicts']}",
          f"- **(task, model) verdict: {res['verdict']}**", "",
          "```", VERDICT_RULES_TEXT.format(delta=res["delta"], alpha=res["holm_alpha"]), "```", ""]
    return "\n".join(L)


def render_summary_md(results: list[dict], delta, alpha) -> str:
    pilots = [r for r in results if r["pilot"]]
    invalid = [r for r in results if r.get("api_error_stamp")]
    L = []
    if invalid:
        L += ["# INVALID: " + ", ".join(f"{r['task']}__{r['model_key']} ({r['incumbent_alone']['n_api_errors']} API-error rows)"
                                        for r in invalid), ""]
    if pilots:
        L += ["# " + "; ".join(sorted({r["pilot_stamp"] for r in pilots})), ""]
    L += ["# Phase 2 summary: paired cascade analysis", ""]
    if invalid:
        L += [f"**INVALID -- API errors, not model answers**: {len(invalid)} of {len(results)} (task, model) pairs contain "
              f"runner '<error: ...>' rows; their incumbent/cascade numbers are meaningless and their verdict is "
              f"'invalid-api-errors'. Re-run phase2/run_incumbents.py (it retries error rows on resume).", ""]
    if pilots:
        L += [f"**PILOT -- not evidence**: {len(pilots)} of {len(results)} (task, model) pairs were run with fewer "
              f"items than n_test (see the n column).", ""]
    L += [f"- {ASYMMETRY_NOTE}",
          "- Labels: incumbent = measured (this run); cheap = CV-estimated (Phase 1 OOF); published-2023 = contrast only.",
          "- Cheap/cascade/sent columns are MEANS across cheap repeats; per-repeat numbers are in each pair's .md/.json.",
          "- $/1k: incumbent-alone = measured sum(cost_usd)/n*1000; cascade = LLM share x incumbent $/1k (cheap side costs $0).",
          ""]
    lenient = [r for r in results if r.get("scoring") == "lenient"]
    results = [r for r in results if r.get("scoring", "exact") == "exact"]
    if lenient:
        L += ["- Two scorings are reported. EXACT (first table) is benchmark-faithful; LENIENT (second table) "
              "is what the cascade decision uses. Definitions:",
              f"  - {SCORING_NOTE['exact']}", f"  - {SCORING_NOTE['lenient']}", ""]
    hdr = ("| task | model | n | pilot | incumbent acc (measured) | incumbent bal_acc | cheap acc (CV-est) | cheap bal_acc "
           "| cascade acc | cascade bal_acc | LLM share | $/1k incumbent | $/1k cascade | sent n | sent: incumbent acc "
           "| sent: cheap acc | oracle acc | oracle share | published-2023 best | verdict |")
    sep = "|" + "---|" * 20
    L += ["## Exact-match scoring (benchmark-faithful)", "", hdr, sep] if lenient else [hdr, sep]
    for r in sorted(results, key=lambda x: (x["task"], x["model_key"])):
        inc = r["incumbent_alone"]
        m = r["mean_over_repeats"]
        pub = r["published_2023"]
        pub_s = f"{_pct(pub['best_score'])} ({pub['best_model']})" if pub else "n/a"
        L.append(
            f"| {r['task']} | {r['model_key']} | {r['n_items']} | {'PILOT' if r['pilot'] else 'no'} "
            f"| {_pct(inc['accuracy'])} [{_pct(inc['accuracy_wilson95_lo'])}, {_pct(inc['accuracy_wilson95_hi'])}] "
            f"| {_pct(inc['balanced_accuracy'])} "
            f"| {_pct(m['cheap_alone']['accuracy'])} | {_pct(m['cheap_alone']['balanced_accuracy'])} "
            f"| {_pct(m['cascade']['accuracy'])} | {_pct(m['cascade']['balanced_accuracy'])} "
            f"| {_pct(m['cascade']['llm_share'])} "
            f"| {_f(inc['cost_per_1k_items_usd'], 4)} | {_f(m['cascade']['cost_per_1k_items_usd'], 4)} "
            f"| {_f(m['sent_subset']['n_sent'], 1)} | {_pct(m['sent_subset']['incumbent_accuracy'])} "
            f"| {_pct(m['sent_subset']['cheap_accuracy'])} "
            f"| {_pct(m['oracle_gate']['accuracy'])} | {_pct(m['oracle_gate']['llm_share'])} "
            f"| {pub_s} | **{r['verdict']}** ({', '.join(r['per_repeat_verdicts'])}) |"
        )
    if lenient:
        L += ["", "## Lenient scoring (used for the cascade decision)", "", hdr, sep]
        for r in sorted(lenient, key=lambda x: (x["task"], x["model_key"])):
            inc = r["incumbent_alone"]
            m = r["mean_over_repeats"]
            pub = r["published_2023"]
            pub_s = f"{_pct(pub['best_score'])} ({pub['best_model']})" if pub else "n/a"
            L.append(
                f"| {r['task']} | {r['model_key']} | {r['n_items']} | {'PILOT' if r['pilot'] else 'no'} "
                f"| {_pct(inc['accuracy'])} [{_pct(inc['accuracy_wilson95_lo'])}, {_pct(inc['accuracy_wilson95_hi'])}] "
                f"| {_pct(inc['balanced_accuracy'])} "
                f"| {_pct(m['cheap_alone']['accuracy'])} | {_pct(m['cheap_alone']['balanced_accuracy'])} "
                f"| {_pct(m['cascade']['accuracy'])} | {_pct(m['cascade']['balanced_accuracy'])} "
                f"| {_pct(m['cascade']['llm_share'])} "
                f"| {_f(inc['cost_per_1k_items_usd'], 4)} | {_f(m['cascade']['cost_per_1k_items_usd'], 4)} "
                f"| {_f(m['sent_subset']['n_sent'], 1)} | {_pct(m['sent_subset']['incumbent_accuracy'])} "
                f"| {_pct(m['sent_subset']['cheap_accuracy'])} "
                f"| {_pct(m['oracle_gate']['accuracy'])} | {_pct(m['oracle_gate']['llm_share'])} "
                f"| {pub_s} | **{r['verdict']}** ({', '.join(r['per_repeat_verdicts'])}) |"
            )
        L += ["", "Parse-fail rates per scoring are in each pair's .md/.json (exact vs lenient)."]
    L += ["", "```", VERDICT_RULES_TEXT.format(delta=delta, alpha=alpha), "```", ""]
    return "\n".join(L)


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return o


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def discover_incumbent_files(incumbent_dir: Path, tasks_filter=None, models_filter=None):
    """<incumbent-dir>/<task>__<model_key>.csv -> list of (task, model_key, path)."""
    out = []
    for p in sorted(Path(incumbent_dir).glob("*.csv")):
        stem = p.stem
        if "__" not in stem:
            logger.warning("skipping %s: no '__' separator in filename", p)
            continue
        task, model_key = stem.split("__", 1)
        if tasks_filter and task not in tasks_filter:
            continue
        if models_filter and model_key not in models_filter:
            continue
        out.append((task, model_key, p))
    return out


def read_best_candidate(configs_dir: Path, task: str) -> str:
    cfg = Path(configs_dir) / f"{task}.json"
    if not cfg.exists():
        raise FileNotFoundError(f"config for task {task!r} not found at {cfg} (needed for best_candidate)")
    with open(cfg) as f:
        d = json.load(f)
    bc = d.get("best_candidate")
    if not bc:
        raise ValueError(f"{cfg} has no 'best_candidate'")
    return bc


def task_specs_from_registry(registry_path) -> dict:
    p = Path(registry_path)
    if not p.exists():
        logger.warning("registry %s not found; classes will fall back to FILE A gold labels", p)
        return {}
    return {e["task"]: parse_task_spec(e) for e in load_registry(p)}


def run(per_item_dir, incumbent_dir, configs_dir, out_dir, registry_path, published_path,
        tasks_filter=None, models_filter=None, delta=DEFAULT_DELTA, alpha=DEFAULT_ALPHA,
        n_boot=10_000, seed=0) -> list[dict]:
    per_item_dir = Path(per_item_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = task_specs_from_registry(registry_path)
    published = load_published_scores(published_path) if published_path else {}

    pairs = discover_incumbent_files(incumbent_dir, tasks_filter, models_filter)
    if not pairs:
        raise SystemExit(f"no incumbent files found under {incumbent_dir}")
    results = []
    lenient_results = []
    for task, model_key, b_path in pairs:
        candidate = read_best_candidate(configs_dir, task)
        a_path = per_item_dir / f"{task}__{candidate}.csv"
        a = load_file_a(a_path)
        b = load_file_b(b_path)
        if b["model_key"].iloc[0] != model_key:
            raise ValueError(f"{b_path}: model_key column {b['model_key'].iloc[0]!r} != filename {model_key!r}")
        spec = specs.get(task)
        if spec is not None:
            classes = list(spec.classes)
            n_test = spec.n_test_expected
        else:
            classes = sorted(a["gold"].unique().tolist())
            n_test = None
            logger.warning("task %s not in registry; classes=%s from FILE A gold, n_test from FILE A", task, classes)
        b = add_lenient_columns(b, classes)
        res = analyze_pair(a, b, classes, n_test_expected=n_test, published=published, delta=delta,
                           alpha=alpha, n_boot=n_boot, seed=seed, scoring="exact",
                           extra_meta={"file_a": str(a_path), "file_b": str(b_path)})
        (out_dir / f"{task}__{model_key}.md").write_text(render_pair_md(res))
        with open(out_dir / f"{task}__{model_key}.json", "w") as f:
            json.dump(_json_safe(res), f, indent=2)
        logger.info("wrote %s__%s [exact] (verdict=%s)", task, model_key, res["verdict"])
        results.append(res)
        res_l = analyze_pair(a, b, classes, n_test_expected=n_test, published=published, delta=delta,
                             alpha=alpha, n_boot=n_boot, seed=seed, scoring="lenient",
                             extra_meta={"file_a": str(a_path), "file_b": str(b_path)})
        (out_dir / f"{task}__{model_key}__lenient.md").write_text(render_pair_md(res_l))
        with open(out_dir / f"{task}__{model_key}__lenient.json", "w") as f:
            json.dump(_json_safe(res_l), f, indent=2)
        logger.info("wrote %s__%s [lenient] (verdict=%s)", task, model_key, res_l["verdict"])
        lenient_results.append(res_l)
    (out_dir / "summary.md").write_text(render_summary_md(results + lenient_results, delta, alpha))
    return results


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-item-dir", default=str(_PROJECT / "results" / "per_item"), help="FILE A directory")
    ap.add_argument("--incumbent-dir", default=str(_PROJECT / "results" / "incumbent"), help="FILE B directory")
    ap.add_argument("--configs-dir", default=str(_PROJECT / "results" / "configs"), help="<task>.json with best_candidate")
    ap.add_argument("--out-dir", default=str(_PROJECT / "results" / "phase2"))
    ap.add_argument("--registry", default=str(_PROJECT / "data" / "task_registry.json"))
    ap.add_argument("--published", default=str(_PROJECT / "data" / "published_scores.csv"))
    ap.add_argument("--tasks", nargs="*", default=None, help="restrict to these tasks")
    ap.add_argument("--models", nargs="*", default=None, help="restrict to these model keys")
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="non-inferiority margin (accuracy points, 0-1)")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Holm family-wise alpha")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    results = run(args.per_item_dir, args.incumbent_dir, args.configs_dir, args.out_dir, args.registry,
                  args.published, tasks_filter=args.tasks, models_filter=args.models, delta=args.delta,
                  alpha=args.alpha, n_boot=args.n_boot, seed=args.seed)
    print((Path(args.out_dir) / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
