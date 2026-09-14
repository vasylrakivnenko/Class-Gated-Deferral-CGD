"""Candidate model implementations for the legalbench_map harness.

Every module in this package exposes:

    NAME: str
    def fit_predict_proba(X_train, y_train, X_test, classes, *, seed,
                           task_key, cache_dir) -> np.ndarray  # (n_test, n_classes)

Modules are imported dynamically by cli.py; nothing in this package
imports the others.
"""
