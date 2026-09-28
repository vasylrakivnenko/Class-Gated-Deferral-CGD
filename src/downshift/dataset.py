"""The one thing the pipeline ingests.

Why this exists. The pipeline grew around the four Online Cascade Learning
streams, so the OCL loader, OCL's label vocabulary and OCL's provenance pins
were reachable from inside the analysis. Adding banking77 then needed a runner
that reached back in and mutated the pipeline's own label table
(`RP.LABEL_NAMES["banking77"] = names`) before calling it. That is the shape of
a pipeline that is not fixed: every new dataset edits the thing that is supposed
to be constant.

So the direction is inverted. A dataset is adapted into `Dataset`; the pipeline
only ever sees `Dataset`; and the pipeline is edited for a bug or an improvement
to the method, never to accommodate a source.

Everything a run needs is here, and nothing else is allowed in:

    name        identifier used for caches, artifacts and tabs
    texts       one string per item
    gold        integer label per item, 0..k-1, contiguous
    label_names one short human phrase per class id -- used verbatim as the NLI
                hypothesis, so it must describe the class, not abbreviate it
    expert      optional: one integer prediction per item from the paid model.
                Absent means the cascade half of the pipeline cannot run, which
                is reported rather than silently skipped.
    source      free text recording where the data came from, carried into the
                artifact so a result is traceable to a loader

`validate()` runs at the boundary, because every one of the checks below is a
mistake that has already reached a result somewhere in this project: labels that
were not contiguous, a label-name map that silently missed a class, and an
expert array that had been filtered while the texts had not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class DatasetError(ValueError):
    """Raised at the boundary, never deeper. An adapter is wrong, not the run."""


@dataclass
class Dataset:
    name: str
    texts: np.ndarray
    gold: np.ndarray
    label_names: dict[int, str]
    expert: np.ndarray | None = None
    source: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.texts = np.asarray(self.texts, dtype=object)
        self.gold = np.asarray(self.gold, dtype=int)
        if self.expert is not None:
            self.expert = np.asarray(self.expert, dtype=int)
        self.label_names = {int(k): str(v) for k, v in self.label_names.items()}
        validate(self)

    @property
    def n(self) -> int:
        return len(self.gold)

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    @property
    def has_expert(self) -> bool:
        return self.expert is not None

    def describe(self) -> dict:
        return {"name": self.name, "n": self.n, "n_classes": self.n_classes,
                "has_expert": self.has_expert, "source": self.source,
                "notes": list(self.notes)}


def validate(d: Dataset) -> None:
    if not d.name or not d.name.strip():
        raise DatasetError("dataset needs a name; it keys the cache and the artifact")
    if len(d.texts) != len(d.gold):
        raise DatasetError(
            f"{d.name}: {len(d.texts)} texts against {len(d.gold)} labels. "
            "A filter applied to one array and not the other silently shifts every "
            "label by the number of dropped rows.")
    if d.n < 2:
        raise DatasetError(f"{d.name}: {d.n} items is not a dataset")
    present = sorted({int(c) for c in d.gold.tolist()})
    if present != list(range(len(present))):
        raise DatasetError(
            f"{d.name}: labels are {present}, which is not 0..{len(present)-1}. "
            "Non-contiguous ids break every per-class array in the pipeline, which "
            "indexes by position. Remap in the adapter.")
    missing = [c for c in present if c not in d.label_names]
    if missing:
        raise DatasetError(
            f"{d.name}: no label name for class(es) {missing}. Names are used "
            "verbatim as NLI hypotheses, so a missing one is a silently weaker "
            "candidate rather than an error at run time.")
    empty = [c for c, v in d.label_names.items() if not str(v).strip()]
    if empty:
        raise DatasetError(f"{d.name}: empty label name for class(es) {empty}")
    if d.expert is not None:
        if len(d.expert) != d.n:
            raise DatasetError(
                f"{d.name}: {len(d.expert)} expert predictions against {d.n} items")
        unseen = sorted({int(c) for c in d.expert.tolist()} - set(present))
        if unseen:
            raise DatasetError(
                f"{d.name}: expert predicts class(es) {unseen} that do not appear in "
                "gold. Unparsed LLM answers must be resolved or dropped in the "
                "adapter, not carried in as a phantom class.")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
_REGISTRY: dict[str, callable] = {}


def register(name: str):
    """Attach a loader to a name. The loader takes no arguments and returns a
    Dataset, so the pipeline never learns how any source is shaped."""
    def wrap(fn):
        if name in _REGISTRY:
            raise DatasetError(f"{name} is already registered")
        _REGISTRY[name] = fn
        return fn
    return wrap


def available() -> list[str]:
    return sorted(_REGISTRY)


def load(name: str) -> Dataset:
    if name not in _REGISTRY:
        raise DatasetError(
            f"unknown dataset {name!r}; registered: {', '.join(available()) or 'none'}")
    d = _REGISTRY[name]()
    if not isinstance(d, Dataset):
        raise DatasetError(f"adapter for {name!r} returned {type(d).__name__}, not Dataset")
    return d
