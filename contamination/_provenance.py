"""Provenance capture shared by the contamination audits in this directory.

Why this file exists. The original BeaverTails and FinBen audits ran in a /tmp
scratchpad that a machine OOM wiped on 2026-09-16, leaving their headline
numbers (99.8% and 71.5%) quoted in SCOPE.md with no artifact behind them.
Restoring the scripts is only half the repair. A reviewer also has to be able
to tell whether they are looking at the same data we were, so every audit here
records, on every run:

  1. the PINNED HuggingFace dataset revision, plus the sha256 of every file in
     that snapshot -- so "same data" is checkable rather than asserted;
  2. a loader-independent content fingerprint over the columns actually used,
     which survives a change of file format or `datasets` version;
  3. the interpreter, library versions, platform and repo commit behind the run;
  4. the published claim each number backs, re-asserted every run. If the
     upstream dataset moves, the script exits non-zero instead of quietly
     rewriting the number the docs quote.

Point 4 is the one that matters for publication. These scripts are not run
once and filed; they are a regression test on a claim.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REPO_ROOT = os.path.dirname(HERE)


# --------------------------------------------------------------------------
# dataset identity
# --------------------------------------------------------------------------
def snapshot(repo: str, revision: str) -> str:
    """Local path to the pinned snapshot. Uses the cache; downloads if absent."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, repo_type="dataset", revision=revision)


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dataset_pin(repo: str, revision: str) -> dict:
    """Revision + per-file sha256 of the snapshot actually read.

    Symlinks are resolved, so this hashes the real blob bytes. A reviewer who
    re-downloads the same revision gets byte-identical files and therefore the
    same hashes; if they do not, the data moved and no number below is
    comparable.
    """
    root = snapshot(repo, revision)
    files = {}
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            p = os.path.join(dirpath, n)
            rel = os.path.relpath(p, root)
            files[rel] = {"sha256": _sha256(os.path.realpath(p)),
                          "bytes": os.path.getsize(os.path.realpath(p))}
    return {"repo": repo, "revision": revision, "repo_type": "dataset",
            "snapshot_files": dict(sorted(files.items()))}


def content_fingerprint(columns: dict) -> str:
    """sha256 over the rows actually used, independent of file format.

    `columns` maps a column name to its list of values; all lists must be the
    same length. Values are joined with unit/record separators that cannot
    occur in the text fields being hashed.
    """
    names = sorted(columns)
    n = len(columns[names[0]])
    for k in names:
        if len(columns[k]) != n:
            raise ValueError(f"column {k} has {len(columns[k])} rows, expected {n}")
    h = hashlib.sha256()
    h.update(("\x1d".join(names) + "\x1e").encode())
    for i in range(n):
        h.update(("\x1f".join(str(columns[k][i]) for k in names) + "\x1e").encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# run identity
# --------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", *args), cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def environment() -> dict:
    pkgs = {}
    for mod in ("datasets", "huggingface_hub", "numpy", "pyarrow"):
        try:
            pkgs[mod] = __import__(mod).__version__
        except Exception:
            pkgs[mod] = None
    # Two separate questions. `git_dirty` is whether the repo as a whole had
    # uncommitted work -- usually true in this tree and mostly irrelevant. The
    # one a reviewer actually needs is whether the AUDIT CODE differed from the
    # committed version when it ran; if it did, the commit hash below does not
    # describe what was executed. The audit's own output is excluded from both,
    # or a run could never report clean: writing results/ dirties the tree.
    dirty = [l for l in _git("status", "--porcelain").splitlines()
             if "contamination/results/" not in l]
    code_dirty = [l for l in dirty if "contamination/" in l]
    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": pkgs,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty),
        "audit_code_dirty": bool(code_dirty),
        "audit_code_uncommitted": code_dirty,
    }


# --------------------------------------------------------------------------
# claim verification
# --------------------------------------------------------------------------
def verify(measured: dict, expected: dict, path: str = "") -> list:
    """Recursively compare measured values against the published claim.

    Returns a list of human-readable mismatches; empty means the artifact in
    the docs still reproduces. Integer counts are compared exactly. Floats are
    compared at the precision they were published to (one decimal place on a
    percentage), because that is the figure a reader can check.
    """
    bad = []
    for k, want in expected.items():
        here = f"{path}.{k}" if path else str(k)
        if k not in measured:
            bad.append(f"{here}: MISSING from this run")
            continue
        got = measured[k]
        if isinstance(want, dict) and isinstance(got, dict):
            bad += verify(got, want, here)
        elif isinstance(want, float):
            if round(float(got), 1) != round(want, 1):
                bad.append(f"{here}: expected {want}, measured {got}")
        elif isinstance(want, (list, tuple)):
            if list(got) != list(want):
                bad.append(f"{here}: expected {list(want)}, measured {list(got)}")
        elif got != want:
            bad.append(f"{here}: expected {want!r}, measured {got!r}")
    return bad


class Tee:
    """Write stdout to the terminal and to a log artifact at the same time."""

    def __init__(self, path: str):
        self.path = path
        self._fh = None
        self._stdout = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "w")
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self._stdout
        self._fh.close()

    def write(self, s):
        self._stdout.write(s)
        self._fh.write(s)

    def flush(self):
        self._stdout.flush()
        self._fh.flush()


def emit(name: str, payload: dict) -> str:
    """Write the machine-readable artifact. Returns its path."""
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"{name}.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
        fh.write("\n")
    return out


def finish(name: str, payload: dict, mismatches: list, t0: float) -> int:
    """Common tail: write the artifact, report drift, pick an exit code."""
    payload["reproduces_published_claim"] = not mismatches
    payload["mismatches"] = mismatches
    payload["runtime_seconds"] = round(time.time() - t0, 1)
    out = emit(name, payload)
    print(f"\nwrote {out} ({payload['runtime_seconds']:,.1f}s)")
    if mismatches:
        print(f"\n!! {len(mismatches)} value(s) DIFFER from the published claim:")
        for m in mismatches:
            print(f"   {m}")
        print("\nThe docs quote numbers this run did not reproduce. Either the\n"
              "upstream dataset moved (check the revision pin and file hashes\n"
              "above) or the audit changed. Do not publish until this is\n"
              "explained; update EXPECTED only with the reason written down.")
        return 1
    print("\nOK: every published number reproduced exactly.")
    return 0
