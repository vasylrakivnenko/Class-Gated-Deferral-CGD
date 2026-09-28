"""Provenance capture for the Online Cascade Learning comparison.

The generic machinery -- content fingerprints, claim verification, tee'd logs --
is shared with `contamination/` rather than copied; only the three pieces that
are bound to a directory (`environment`, `emit`, `finish`) are defined here, so
that an artifact written by this area reports THIS area's code as clean or
dirty, not the contamination audits'.

What differs from `contamination/_provenance.py` is the provenance model. There
the unit of identity is a pinned HuggingFace revision. Here the inputs are not
content-addressed at all:

  * the four `*_preprocessed.csv` files live in a Google Drive archive linked
    from the OCL README -- a mutable URL with no revision concept;
  * the LLM annotation streams live in a GitHub repo with no tags or releases.

So identity has to be established by hashing. Every source file this area reads
is recorded with its sha256 and the exact URL it came from, and the GitHub side
is additionally pinned to a commit sha. If any of it is replaced upstream, the
hashes move and the run says so instead of quietly measuring something else.
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REPO_ROOT = os.path.dirname(HERE)
AREA = "ocl_compare/"

# Cache for the upstream files. Deliberately outside the repo: this data carries
# no licence (see README) and imdb_preprocessed.csv alone is 33 MB.
CACHE = os.environ.get("OCL_CACHE", os.path.expanduser("~/.cache/zadumai-ocl"))

_spec = importlib.util.spec_from_file_location(
    "_contam_provenance", os.path.join(REPO_ROOT, "contamination", "_provenance.py"))
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

_sha256 = _shared._sha256
content_fingerprint = _shared.content_fingerprint
verify = _shared.verify
Tee = _shared.Tee
_git = _shared._git


def _ssl_context():
    """The python.org interpreters on this machine ship no CA bundle, so a bare
    urlopen fails CERTIFICATE_VERIFY_FAILED. certifi is already present via
    requests; use its bundle rather than disabling verification."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(url: str, name: str) -> str:
    """Download `url` into the cache once; return the local path."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print(f"  fetching {name} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        tmp = path + ".part"
        with urllib.request.urlopen(req, timeout=300, context=_ssl_context()) as r, \
                open(tmp, "wb") as fh:
            fh.write(r.read())
        os.replace(tmp, path)       # never leave a truncated file looking cached
    return path


def source_pin(entries: dict) -> dict:
    """{label: (url, local_path)} -> {label: {url, sha256, bytes}}.

    The sha256 is the whole point: neither upstream location is versioned, so
    this is the only way a later reader can tell whether they hold what we held.
    """
    out = {}
    for label, (url, path) in entries.items():
        out[label] = {"url": url, "sha256": _sha256(path),
                      "bytes": os.path.getsize(path)}
    return out


def environment() -> dict:
    pkgs = {}
    for mod in ("numpy", "pandas", "scikit-learn"):
        try:
            pkgs[mod] = __import__(mod.replace("-", "_")).__version__
        except Exception:
            pkgs[mod] = None
    # Same two questions as the contamination audits: whether the tree as a
    # whole was dirty (usually true, usually irrelevant) and whether THIS
    # area's code differed from its committed version when it ran. The second
    # is the one that decides whether git_commit describes what executed.
    dirty = [l for l in _git("status", "--porcelain").splitlines()
             if f"{AREA}results/" not in l]
    code_dirty = [l for l in dirty if AREA in l]
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


def emit(name: str, payload: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"{name}.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
        fh.write("\n")
    return out


def finish(name: str, payload: dict, mismatches: list, t0: float) -> int:
    payload["reproduces_published_claim"] = not mismatches
    payload["mismatches"] = mismatches
    payload["runtime_seconds"] = round(time.time() - t0, 1)
    out = emit(name, payload)
    print(f"\nwrote {out} ({payload['runtime_seconds']:,.1f}s)")
    if mismatches:
        print(f"\n!! {len(mismatches)} value(s) DIFFER from the published paper:")
        for m in mismatches:
            print(f"   {m}")
        print("\nThe reconstruction no longer reproduces Nie et al.'s Table 1.\n"
              "Either an upstream file was replaced (check the sha256 pins\n"
              "above) or this code changed. Do not run the comparison on top\n"
              "of a stream that failed this check.")
        return 1
    print("\nAll Table 1 cells reproduced. The streams are the paper's.")
    return 0
