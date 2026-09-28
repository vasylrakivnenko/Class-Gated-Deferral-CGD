"""Rebuild the four Online Cascade Learning evaluation streams, and prove they
are the ones the paper measured.

  Nie, Ding, Hu, Jermaine, Chaudhuri. "Online Cascade Learning for Efficient
  Inference over Streams." ICML 2024, PMLR 235:38071-38090.

Why this file exists. We want to run our class-aware deferral gate against
OCL's policies on OCL's own benchmark, at their budgets, under their metric.
That is only worth doing if we are demonstrably running on their data, and
their data is awkward to establish:

  * the repo ships the LLM annotation streams but NOT the `*_preprocessed.csv`
    files its own configs read. Those are in a Google Drive archive linked from
    the README -- a mutable URL, no revision, no checksum published.
  * the annotation streams are bare label-per-line text files. They carry no
    key, so they are only meaningful if the row order of the CSV they were
    generated against is exactly preserved.

So alignment cannot be assumed; it has to be demonstrated. The demonstration is
that every cell of the paper's Table 1 LLM row comes back, for both teacher
models, under the repo's own `postprocess()` logic:

    IMDB  94.15 / 93.33     HateSpeech  83.34|83.28 / 77.81|82.19
    ISEAR 70.34 / 68.23     FEVER       79.98 / 77.15

Eight cells across four datasets and two unrelated teachers. A misaligned
stream scores at chance, so this is not a weak test -- an earlier attempt to
rebuild ISEAR from public sources scored 0.1542 against a 0.1556 shuffled
control, which is how we knew to keep looking for the authors' own file.

Two things here were measured rather than assumed:

  * DISPLAY CONVENTION. Three of the four tasks round to two decimals; ISEAR
    truncates. ISEAR/GPT-3.5 is 70.3496 -- printed 70.34, where rounding gives
    70.35 -- and ISEAR/Llama-2 is 68.2364, printed 68.23 where rounding gives
    68.24. Both ISEAR cells are off by exactly +0.01 under rounding and both
    are exact under truncation, so this is a real per-task difference in how
    the table was produced, not a fudge to make one number fit. The regression
    test asserts on the integer correct-counts, which are convention-free; the
    printed strings are recorded alongside them for the reader.
  * MALFORMED ANNOTATIONS. Two ISEAR GPT-3.5 rows read 'sadnessjoy' and
    'sadnessfear'. OCL's postprocess resolves these by first-substring-match
    over an insertion-ordered dict, so they become joy and sadness rather than
    being dropped. Replicated exactly; dropping them changes the count.

Run:  python ocl_compare/reconstruct.py
Exits non-zero if any Table 1 cell fails to reproduce.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _provenance as P  # noqa: E402

# The OCL repo has no tags and no releases, so the commit is the only pin.
OCL_REPO = "Flitternie/online_cascade_learning"
OCL_COMMIT = "285e03fc98afabdf7768558d33ee7b5f662b5f19"
RAW = f"https://raw.githubusercontent.com/{OCL_REPO}/{OCL_COMMIT}"
# Linked from the repo README as "Preprocessed data for reproducing paper results".
DRIVE_ID = "1vQR7pqVqmYndHkfpcEpwCSWYDXON0gkl"
DRIVE_URL = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download"

TASKS = ("imdb", "hatespeech", "isear", "fever")
TEACHERS = {"gpt3.5": "gpt3.5_turbo_1106", "llama2": "llama2_70b_chat"}

# ---------------------------------------------------------------------------
# Faithful replicas of the repo's data modules. Deliberately transcribed rather
# than imported: importing would drag in torch and transformers, and the point
# is to show exactly which decision rule produced each label.
# ---------------------------------------------------------------------------
ISEAR_TO_ID = {"joy": 0, "sadness": 1, "anger": 2, "guilt": 3,
               "shame": 4, "fear": 5, "disgust": 6}
WORDS = {                       # insertion order is load-bearing: first match wins
    "imdb": {"positive": 1, "negative": 0},
    "hatespeech": {"no": 0, "yes": 1},
    "isear": ISEAR_TO_ID,
    "fever": {"true": 1, "false": 0},
}


def postprocess(task: str, output: str) -> int:
    """data/<task>.py::postprocess, including the int() fast path the Llama-2
    annotation files rely on (they are already label ids, not words)."""
    low = output.lower().strip()
    try:
        return int(low)
    except ValueError:
        for k, v in WORDS[task].items():
            if k in low:
                return v
        return -1


def gold(task: str, col: pd.Series) -> np.ndarray:
    """data/<task>.py::preprocess, label column only."""
    if task == "isear":
        return col.astype(str).str.strip().map(ISEAR_TO_ID).to_numpy()
    if task == "fever":
        return np.where(col.astype(str).str.strip() == "REFUTES", 0, 1)
    return col.astype(int).to_numpy()


# Rounding for three tasks, truncation for ISEAR -- see the module docstring.
TRUNCATES = {"isear"}


def display(task: str, x: float) -> str:
    return f"{math.floor(x * 10000) / 100:.2f}" if task in TRUNCATES else f"{x * 100:.2f}"


# ---------------------------------------------------------------------------
# The published claim, re-asserted on every run.
# `correct` values are exact integer counts -- convention-free, so they are the
# thing that actually gates. `paper` values are the strings printed in Table 1.
# ---------------------------------------------------------------------------
EXPECTED = {
    "imdb": {"n_rows": 25000, "n_positive": 12500,
             "gpt3.5": {"n_correct": 23538, "n_unparsed": 0, "paper_accuracy": "94.15"},
             "llama2": {"n_correct": 23333, "n_unparsed": 862, "paper_accuracy": "93.33"}},
    "hatespeech": {"n_rows": 10703, "n_positive": 1196,
                   "gpt3.5": {"n_correct": 8920, "n_unparsed": 0, "paper_accuracy": "83.34",
                              "n_true_positive": 996, "paper_recall": "83.28"},
                   "llama2": {"n_correct": 8328, "n_unparsed": 0, "paper_accuracy": "77.81",
                              "n_true_positive": 983, "paper_recall": "82.19"}},
    "isear": {"n_rows": 7666, "n_classes": 7,
              "gpt3.5": {"n_correct": 5393, "n_unparsed": 0, "paper_accuracy": "70.34"},
              "llama2": {"n_correct": 5231, "n_unparsed": 241, "paper_accuracy": "68.23"}},
    "fever": {"n_rows": 6512,
              "gpt3.5": {"n_correct": 5208, "n_unparsed": 0, "paper_accuracy": "79.98"},
              "llama2": {"n_correct": 5024, "n_unparsed": 0, "paper_accuracy": "77.15"}},
}


def load_streams() -> tuple[dict, dict]:
    """Fetch + hash every upstream file, then rebuild the four streams."""
    pins = {}
    zip_path = P.fetch(DRIVE_URL, "ocl_preprocessed_data.zip")
    pins["preprocessed_archive"] = (DRIVE_URL, zip_path)

    csvs = {}
    with zipfile.ZipFile(zip_path) as z:
        members = {os.path.basename(n): n for n in z.namelist() if n.endswith(".csv")}
        for task in TASKS:
            name = f"{task}_preprocessed.csv"
            assert name in members, f"{name} missing from the archive: {sorted(members)}"
            with z.open(members[name]) as fh:
                raw = fh.read()
            # header=0 + explicit names: the repo discards the first row. Not a
            # detail -- keeping it shifts every annotation by one and the whole
            # alignment check fails.
            csvs[task] = pd.read_csv(io.BytesIO(raw), sep=",", header=0,
                                     names=["label", "text"])

    streams = {}
    for task in TASKS:
        df = csvs[task]
        g = gold(task, df["label"])
        assert not pd.isna(g).any(), f"{task}: unmapped gold label"
        entry = {"n_rows": int(len(df)), "gold": g,
                 "text": df["text"].astype(str).to_numpy()}
        if task in ("hatespeech", "imdb"):
            entry["n_positive"] = int((g == 1).sum())
        if task == "isear":
            entry["n_classes"] = int(len(set(g.tolist())))
        for short, slug in TEACHERS.items():
            url = f"{RAW}/data/llm_anntations/{task}_{slug}.txt"
            path = P.fetch(url, f"{task}_{slug}.txt")
            pins[f"{task}_{short}"] = (url, path)
            with open(path) as fh:
                pred = np.array([postprocess(task, l.rstrip("\n")) for l in fh])
            assert len(pred) == len(g), (
                f"{task}/{short}: {len(pred)} annotations vs {len(g)} rows -- "
                "the annotation stream does not describe this CSV")
            entry[short] = {"pred": pred}
        streams[task] = entry
    return streams, P.source_pin(pins)


def measure(streams: dict) -> dict:
    out = {}
    for task in TASKS:
        s = streams[task]
        g = s["gold"]
        rec = {"n_rows": s["n_rows"]}
        for key in ("n_positive", "n_classes"):
            if key in s:
                rec[key] = s[key]
        for short in TEACHERS:
            pred = s[short]["pred"]
            n_correct = int((pred == g).sum())
            cell = {"n_correct": n_correct,
                    "n_unparsed": int((pred == -1).sum()),
                    "paper_accuracy": display(task, n_correct / len(g))}
            if task == "hatespeech":
                tp = int(((pred == 1) & (g == 1)).sum())
                cell["n_true_positive"] = tp
                cell["paper_recall"] = display(task, tp / int((g == 1).sum()))
            rec[short] = cell
        # Loader-independent identity for the stream itself, so a later run can
        # show it read the same rows even if the archive is repackaged.
        rec["content_fingerprint"] = P.content_fingerprint({
            "text": list(s["text"]), "gold": [int(v) for v in g],
            "gpt3.5": [int(v) for v in s["gpt3.5"]["pred"]],
            "llama2": [int(v) for v in s["llama2"]["pred"]]})
        out[task] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true",
                    help="report the measurement without asserting Table 1")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(P.RESULTS, exist_ok=True)
    tee = P.Tee(os.path.join(P.RESULTS, "reconstruct.log"))
    with tee:
        print(__doc__.split("Run:")[0].rstrip())
        print(f"\nOCL repo pinned at {OCL_COMMIT}")
        print(f"preprocessed archive: {DRIVE_URL}\n")

        streams, pins = load_streams()
        measured = measure(streams)

        print(f"\n{'task':<12}{'rows':>7}  {'teacher':<8}{'correct':>8}{'unparsed':>9}"
              f"{'accuracy':>10}{'paper':>8}")
        for task in TASKS:
            m = measured[task]
            for short in TEACHERS:
                c = m[short]
                want = EXPECTED[task][short]["paper_accuracy"]
                ok = "ok" if c["paper_accuracy"] == want else "DIFF"
                print(f"{task:<12}{m['n_rows']:>7}  {short:<8}{c['n_correct']:>8}"
                      f"{c['n_unparsed']:>9}{c['paper_accuracy']:>10}{want:>8}  {ok}")
                if "paper_recall" in c:
                    w = EXPECTED[task][short]["paper_recall"]
                    print(f"{'':<12}{'':>7}  {'':<8}{c['n_true_positive']:>8}"
                          f"{'(tp)':>9}{c['paper_recall']:>10}{w:>8}  "
                          f"{'ok' if c['paper_recall'] == w else 'DIFF'}  recall")

        mismatches = [] if args.no_verify else P.verify(measured, EXPECTED)
        payload = {
            "what": "Reconstruction of the four Online Cascade Learning (Nie et al., "
                    "ICML 2024) evaluation streams, verified against Table 1.",
            "paper": "https://proceedings.mlr.press/v235/nie24a.html",
            "code": f"https://github.com/{OCL_REPO}",
            "ocl_commit": OCL_COMMIT,
            "verified_against_published_claim": not args.no_verify,
            "sources": pins,
            "environment": P.environment(),
            "expected": EXPECTED,
            "measured": measured,
        }
        return P.finish("reconstruct", payload, mismatches, t0)


if __name__ == "__main__":
    raise SystemExit(main())
