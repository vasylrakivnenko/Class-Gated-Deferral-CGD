"""Monkeypatches the socket layer to record every hostname contacted
during a real tiny CV run (majority + tfidf_logreg on the small, real
'abercrombie' LegalBench task) and asserts every recorded hostname ends
in 'huggingface.co' (or is a known HF-Hub backing-storage host observed
in practice, e.g. s3.amazonaws.com for LFS/parquet-backed files) and
that NONE is 'github.com' or any other unexpected host.

Passes on both a warm and a cold Hugging Face cache: on a warm cache,
load_dataset may make zero network calls at all, which vacuously
satisfies "every recorded hostname is allowed"."""
from __future__ import annotations

import socket

import importlib.util
from pathlib import Path

import candidates.baselines as baselines
from core.cv import run_task_candidate_cv
from core.data import load_task_data
from tests.helpers import CallableModule

_TFIDF_PATH = Path(__file__).resolve().parent.parent / "candidates" / "tfidf_logreg.py"


def _load_tfidf_module():
    spec = importlib.util.spec_from_file_location("legalbench_map_candidates.tfidf_logreg", _TFIDF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ALLOWED_EXACT_HOSTS = {"s3.amazonaws.com"}


def _host_allowed(host: str) -> bool:
    return host.endswith("huggingface.co") or host.endswith(".hf.co") or host in ALLOWED_EXACT_HOSTS


def test_only_huggingface_hosts_contacted(monkeypatch):
    contacted = []
    orig_getaddrinfo = socket.getaddrinfo

    def spy_getaddrinfo(host, *args, **kwargs):
        contacted.append(host)
        return orig_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", spy_getaddrinfo)

    entry = {
        "task": "abercrombie",
        "hf_config": "abercrombie",
        "family": "exception",
        "is_reasoning_exception": True,
        "n_train": 5,
        "n_test": 95,
        "input_columns": ["text"],
        "n_classes": 5,
        "class_distribution": {
            "generic": 19, "descriptive": 19, "suggestive": 19,
            "arbitrary": 19, "fanciful": 19,
        },
        "metric": "balanced_accuracy",
        "notes": "",
    }
    task_data = load_task_data(entry)

    run_task_candidate_cv(
        task_data,
        CallableModule(baselines.fit_predict_proba),
        "majority",
        folds=5,
        repeats=1,
        alpha=0.1,
        seed=0,
        cache_dir="/private/tmp/legalbench_map_test_cache",
    )

    tfidf_module = _load_tfidf_module()
    run_task_candidate_cv(
        task_data,
        tfidf_module,
        "tfidf_logreg",
        folds=5,
        repeats=1,
        alpha=0.1,
        seed=0,
        cache_dir="/private/tmp/legalbench_map_test_cache",
    )

    bad_hosts = [h for h in contacted if not _host_allowed(h)]
    assert "github.com" not in contacted
    assert bad_hosts == [], f"contacted disallowed host(s): {sorted(set(bad_hosts))}"
