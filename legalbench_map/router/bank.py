"""
The free classifiers the harness answers with: one per menu task, fitted
once and saved under router/bank/ (gitignored; rebuild with
build_router_bank.py).

Each task uses the candidate that won its CV run (the `best` row of
results/tasks.csv), refit on ALL of the task's labeled rows (train + test
splits) with the same features and the same inner-CV choice of C as in that
run. Its CV score is the accuracy estimate for the refit model.

A task gets policy "llm" instead of "free" when its best free model trails
the best published LLM score by more than LLM_GAP_POINTS. The two CUAD tasks
where we measured Jev directly put it 7-14 points under the best published
LLM (82% vs 90-96%), so a free model within 10 points of that published
score is assumed to beat Jev, and one further behind is assumed not to.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from candidates import _embed_common, tfidf_logreg
from core.data import load_registry, load_task_data
from router.menu import family

ROOT = Path(__file__).resolve().parents[1]
BANK_DIR = ROOT / "router" / "bank"
REGISTRY = ROOT / "data" / "task_registry.json"
RESULTS_CSV = ROOT / "results" / "tasks.csv"
CACHE_DIR = ROOT / ".cache"  # the CV runs' embedding cache, so refitting re-embeds nothing
LLM_GAP_POINTS = 10.0

EMBED_MODELS = {"embed_small_logreg": "BAAI/bge-small-en-v1.5", "embed_base_logreg": "BAAI/bge-base-en-v1.5"}


@dataclass
class FreeModel:
    task: str
    candidate: str
    classes: list  # sorted labels; predict_proba columns follow this order
    estimator: object  # tfidf: the fitted Pipeline; embed: the fitted LogisticRegression

    def predict_proba(self, texts: list) -> np.ndarray:
        if self.candidate == "tfidf_logreg":
            proba = self.estimator.predict_proba([t[: tfidf_logreg.MAX_CHARS] for t in texts])
        else:
            # Encode directly rather than through embed_texts, so user text
            # never lands in the on-disk embedding cache.
            encoder = _embed_common.get_model(EMBED_MODELS[self.candidate])
            X = encoder.encode(
                texts, batch_size=_embed_common.ENCODE_BATCH_SIZE, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
            proba = self.estimator.predict_proba(X)
        col = {c: j for j, c in enumerate(self.estimator.classes_)}
        return np.stack([proba[:, col[c]] if c in col else np.zeros(len(texts)) for c in self.classes], axis=1)


def _grid_fit(make_estimator, param, X, y, seed):
    """The candidates' hyperparameter search: C from C_GRID by inner
    stratified k-fold (k=3, shrunk for rare classes), maximizing balanced
    accuracy, refit on everything."""
    k = min(tfidf_logreg.INNER_K, min(Counter(y).values()))
    if k < 2:
        return make_estimator(1.0).fit(X, y)
    gs = GridSearchCV(
        make_estimator(1.0), {param: tfidf_logreg.C_GRID}, scoring="balanced_accuracy",
        cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=seed), n_jobs=1, refit=True,
    )
    return gs.fit(X, y).best_estimator_


def fit_model(task: str, candidate: str, X: list, y: list, classes: list, seed: int = 0) -> FreeModel:
    if candidate == "tfidf_logreg":
        X_t = [t[: tfidf_logreg.MAX_CHARS] for t in X]
        est = _grid_fit(lambda C: tfidf_logreg._build_pipeline(C, seed), "clf__C", X_t, y, seed)
    elif candidate in EMBED_MODELS:
        emb, _ = _embed_common.embed_texts(EMBED_MODELS[candidate], task, X, str(CACHE_DIR / candidate))
        est = _grid_fit(lambda C: _embed_common._build_classifier(C, seed), "C", emb, y, seed)
    else:
        raise ValueError(f"{task}: no refit recipe for candidate {candidate!r}")
    return FreeModel(task=task, candidate=candidate, classes=list(classes), estimator=est)


def build(tasks: list | None = None, bank_dir: Path = BANK_DIR, seed: int = 0, log=print) -> dict:
    """Fit and save every "free" task's model; write manifest.json with every
    task's policy and scores. Returns the manifest."""
    bank_dir.mkdir(parents=True, exist_ok=True)
    best = pd.read_csv(RESULTS_CSV).query("best == 1").set_index("task")
    manifest = {}
    for entry in load_registry(REGISTRY):
        task = entry["task"]
        if tasks and task not in tasks:
            continue
        row = best.loc[task]
        policy = "llm" if row.gap_points > LLM_GAP_POINTS else "free"
        data = load_task_data(entry)
        manifest[task] = {
            "family": family(task),
            "policy": policy,
            "candidate": row.candidate,
            "classes": data.spec.classes,
            "cv_metric": row.metric_name,
            "cv_mean": round(float(row.cv_mean_CV_estimated), 4),
            "published_best": float(row.published_best_score),
            "published_best_model": row.published_best_model,
            "gap_points": round(float(row.gap_points), 2),
            "n_fit": len(data.X_train) + len(data.X_test),
        }
        if policy == "free":
            model = fit_model(task, row.candidate, data.X_train + data.X_test, data.y_train + data.y_test,
                              data.spec.classes, seed)
            joblib.dump(model, bank_dir / f"{task}.joblib", compress=3)
        log(f"{task}: {policy} ({row.candidate}, CV {row.cv_mean_CV_estimated:.3f}, gap {row.gap_points:+.1f} pts)")
    (bank_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


class Bank:
    """Reads manifest.json; loads each task's model on first use."""

    def __init__(self, bank_dir: Path = BANK_DIR):
        self.bank_dir = Path(bank_dir)
        path = self.bank_dir / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; build the bank first: python legalbench_map/build_router_bank.py")
        self.manifest = json.loads(path.read_text())
        self._models = {}

    def model(self, task: str) -> FreeModel:
        if task not in self._models:
            self._models[task] = joblib.load(self.bank_dir / f"{task}.joblib")
        return self._models[task]
