"""The published_scores.csv loader must reject/error on any row missing a
non-empty source_url."""
from __future__ import annotations

import csv

import pytest

from core.verdict import load_published_scores


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "model", "metric", "score", "source_url"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_rejects_missing_source_url(tmp_path):
    path = tmp_path / "published_scores.csv"
    _write_csv(path, [
        {"task": "abercrombie", "model": "gpt-4", "metric": "accuracy", "score": "0.90", "source_url": ""},
    ])
    with pytest.raises(ValueError):
        load_published_scores(path)


def test_rejects_whitespace_only_source_url(tmp_path):
    path = tmp_path / "published_scores.csv"
    _write_csv(path, [
        {"task": "abercrombie", "model": "gpt-4", "metric": "accuracy", "score": "0.90", "source_url": "   "},
    ])
    with pytest.raises(ValueError):
        load_published_scores(path)


def test_accepts_valid_rows_and_picks_best_score(tmp_path):
    path = tmp_path / "published_scores.csv"
    _write_csv(path, [
        {"task": "abercrombie", "model": "gpt-4", "metric": "accuracy", "score": "0.90",
         "source_url": "https://example.com/paper"},
        {"task": "abercrombie", "model": "gpt-3.5", "metric": "accuracy", "score": "0.80",
         "source_url": "https://example.com/paper"},
    ])
    result = load_published_scores(path)
    assert result["abercrombie"]["best_score"] == pytest.approx(0.90)
    assert result["abercrombie"]["best_model"] == "gpt-4"


def test_missing_file_returns_empty_dict(tmp_path):
    result = load_published_scores(tmp_path / "does_not_exist.csv")
    assert result == {}


def test_rejects_non_numeric_score(tmp_path):
    path = tmp_path / "published_scores.csv"
    _write_csv(path, [
        {"task": "abercrombie", "model": "gpt-4", "metric": "accuracy", "score": "not-a-number",
         "source_url": "https://example.com/paper"},
    ])
    with pytest.raises(ValueError):
        load_published_scores(path)
