"""Spec 4.4 -- Calibrator: computing gate thresholds on dev and building the FAISS novelty
index. No training.

WHAT THIS MODULE DOES (spec 6.8)
--------------------------------
Split conformal prediction on **dev, SEEN subflows only** (``Partitions.dev`` is
already novel-filtered by :func:`reflex.data.build_partitions`; nothing is
re-added here):

1. Score every dev agent turn with the frozen checkpoint (``select.score_turn``).
2. Nonconformity per applicable head: ``s = 1 - softmax(gold)``.
3. ``q_h = the ceil((n+1)(1-alpha))/n quantile of s`` -- the finite-sample
   correction, implemented literally in :func:`conformal_quantile`.
4. ``novelty_threshold`` = the ``gate.novelty_percentile`` percentile of
   ``novelty_distance`` over the same dev turns.
5. Repeat step 3 for every alpha in ``gate.alpha_sweep`` (plus ``gate.alpha``
   itself) so the E5 coverage-error curve needs no recalibration.

HEADS CALIBRATED
----------------
``nextstep`` (H1) and ``intent`` (H2) on every dev agent turn; ``action`` (H3) on
take_action turns; ``skeleton`` (H5) on retrieve_utterance turns; ``template``
(H7) once per act position of retrieve_utterance turns, pooled into ONE shared
``q``; and ``value`` (H4) once per value slot of a take_action turn, pooled into
ONE shared ``q`` (:class:`reflex.schemas.Calibration` fixes those six keys). A
value slot with a single candidate column is counted but not scored -- its
prediction set is that candidate at every ``q``, see the single-candidate rule
in :func:`_collect_dev_scores`.

H4 used to be skipped here, on the stated grounds that the gate never consulted
it. That stopped being true when :func:`reflex.gate.evaluate_gate` grew its
value-confidence block: it demands ``calibration.quantiles["value"]`` for every
take_action turn that has value distributions and refuses to invent one, so a
calibration written without that key aborted the FIRST such turn of every Arm B
run with a ContractViolation.

WHICH BRANCH A HEAD IS CALIBRATED ON
------------------------------------
Applicability here matches :func:`reflex.gate.evaluate_gate` exactly: H3/H4 are
collected on turns whose PREDICTED nextstep is take_action and H5/H7 on turns
whose PREDICTED nextstep is retrieve_utterance -- never the gold one. That is
the population the gate actually applies ``q_h`` to, so it is the population
split conformal has to be exchangeable with; fitting on the gold branch instead
would calibrate on turns the gate never routes through that head. A row still
needs a GOLD label to be scored at all, so a turn predicted into a branch its
gold does not share carries a ``-1`` gold index and is skipped by
:func:`nonconformity_scores` (and by the ECE diagnostic) rather than counted.
How many gold-branch turns were dropped that way is reported as ``branch_drops``
in the diagnostics sidecar instead of being left invisible.

A CALIBRATION GAP IS NOT A GATE
-------------------------------
Expected calibration error is computed here for **REPORTING ONLY** and lands in
a clearly-labelled sidecar (``paths.calibration_diagnostics_path``). It is never
read back, never compared to a threshold and never allowed to influence a route.
Ranking correctly and being calibrated are different properties: a head can rank
the gold first every time while being badly miscalibrated, and the reverse.
Routing uses the conformal set size, the novelty distance and slot availability
-- nothing else (spec 6.6). The spec 8.5 ECE that the report quotes is computed
on ``test_seen`` by :func:`reflex.evaluate.calibration_metrics`, which owns every
Section 8 number (boundary ruling 3); the sidecar here is a dev diagnostic with a
different domain and is not a substitute for it.

Read ``cfg``, never a literal: spec 10 forbids any numeric threshold, model id,
path or price in code.

THE TEMPLATE QUANTILE COLLAPSE (found and fixed by external review)
---------------------------------------------------------------------
``h7_absent_gold: max_nonconformity`` used to inject an explicit ``s=1.0`` row
into the SAME nonconformity distribution ``conformal_quantile`` reads to set
the template head's accept/reject threshold, whenever a position's gold
template was absent from its candidate pool. At the real absent rate (~50% of
dev template positions -- DECISIONS D23), that forces ``q`` to 1.0 and the
threshold to 0.0: every candidate enters the prediction set, and the singleton
gate then rejects every turn, regardless of how confident the model actually
was. Reproduced with synthetic data at an absent rate as low as 3%: a
99.9%-confident response was rejected. Fixed in ``_collect_dev_scores``:
absent rows are now EXCLUDED from the quantile-determining distribution in
BOTH ``h7_absent_gold`` modes -- the coverage gap they represent is a
COVERAGE-ABSENCE fact ("is the true answer even a candidate here"), not a
CONFIDENCE fact ("how sure is the model among the candidates it has"), and
conflating the two is what broke the threshold. The coverage gap itself is
NOT deleted -- it is reported as ``template_coverage`` in the diagnostics
sidecar (``calibration_diagnostics_path``), visible and separate, never
folded back into the routing threshold. What is still MISSING, and is a
genuinely open item, not fixed here: there is no per-turn RUNTIME mechanism
that can detect "this specific position's true answer is absent from the
bank" the way dev calibration can (dev has gold labels; a live turn does not)
-- per-candidate softmax alone cannot signal "the right answer isn't even a
candidate." A proper fix needs a Learn-Then-Test-style two-stage design
(https://arxiv.org/abs/2110.01052): calibrate "is this situation supported"
separately from "is this specific response correct," each against its own
objective. That redesign is out of scope for this pass.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from fractions import Fraction
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from reflex.config import get_dotted, resolve_path
from reflex.contracts import ContractViolation, NoveltyIndex
from reflex.schemas import NEXT_STEPS, Bank, Calibration, ContextWindow, NormalizedTurn

__all__ = [
    "nonconformity_scores",
    "conformal_quantile",
    "build_novelty_index",
    "save_novelty_index",
    "load_novelty_index",
    "calibrate",
    "write_calibration",
    "load_calibration",
]


# ---------------------------------------------------------------------------
# Module constants (names, not tunables -- spec 10 bans thresholds/ids/paths/
# prices from code, not the vocabulary the frozen schema already fixes).
# ---------------------------------------------------------------------------

#: The six :attr:`reflex.schemas.Calibration.quantiles` keys, in report order.
#: Frozen by the ``Calibration`` docstring, so they are NOT a config knob:
#: a config that disagreed with the schema would silently produce a calibration
#: the gate cannot read. ``"value"`` is exactly that failure: the schema has
#: named it since gate.py grew its H4 value-confidence check, this tuple did
#: not, and :func:`reflex.gate._quantile` raises rather than defaulting -- so
#: every Arm B run died on its first slot-bearing take_action turn.
_HEADS: tuple[str, ...] = ("nextstep", "intent", "action", "skeleton", "template", "value")

#: Sidecar banner. Repeated in the file so nobody wires it into a decision.
_DIAGNOSTIC_BANNER = (
    "DIAGNOSTIC ONLY -- NOT A GATE, NOT A REPORTED METRIC. A calibration gap is "
    "not a gate: ranking correctness and being calibrated are different "
    "properties, so nothing in reflex.gate may read these numbers. The spec 8.5 "
    "ECE quoted in the report is computed on test_seen by "
    "reflex.evaluate.calibration_metrics. Coverage below is IN-SAMPLE (measured "
    "on the same dev turns the quantiles were fitted on) and is therefore "
    "optimistic by construction."
)


# ---------------------------------------------------------------------------
# 1. Nonconformity and the finite-sample conformal quantile
# ---------------------------------------------------------------------------


def nonconformity_scores(
    probs: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
) -> list[float]:
    """Compute split-conformal nonconformity ``s = 1 - p(gold)`` (spec 6.8 step 1).

    Args:
        probs: One probability vector per turn where the head applies.
        gold_indices: Gold class index per turn. Rows with a negative index are
            SKIPPED, not scored -- the head did not apply there.

    Returns:
        One score per applicable row, in input order.
    """
    if len(probs) != len(gold_indices):
        raise ValueError(
            f"probs and gold_indices must be the same length: "
            f"{len(probs)} != {len(gold_indices)}"
        )
    out: list[float] = []
    for row, gold in enumerate(gold_indices):
        gold = int(gold)
        if gold < 0:
            continue
        vector = probs[row]
        if gold >= len(vector):
            raise ValueError(
                f"gold index {gold} is out of range for row {row} of width {len(vector)}"
            )
        out.append(1.0 - float(vector[gold]))
    return out


def _ceil_k(n: int, alpha: float) -> int:
    """Return ``ceil((n+1)(1-alpha))`` computed EXACTLY, not in binary floats.

    ``math.ceil((n + 1) * (1.0 - alpha))`` is off by one whenever the true
    product is an integer that the double rounds up: e.g. ``n=999,
    alpha=0.059`` gives 942 in floats and 941 exactly. An off-by-one in the
    order statistic is a silently wrong quantile, so the arithmetic is done over
    ``Fraction(str(alpha))`` -- the exact decimal the config author wrote.
    """
    target = (n + 1) * (1 - Fraction(str(float(alpha))))
    return -((-target.numerator) // target.denominator)


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Return the finite-sample conformal quantile (spec 6.8 step 1).

    ``q = the ceil((n+1)(1-alpha))/n quantile of s``. Implement the
    finite-sample correction literally: sort ``s`` ascending and take element
    ``ceil((n+1)*(1-alpha)) - 1`` (0-based), clamping the index to ``n-1``. Using
    ``numpy.quantile(s, 1-alpha)`` instead drops the ``(n+1)`` correction and
    quietly under-covers.

    Args:
        scores: Nonconformity scores.
        alpha: Miscoverage rate, e.g. ``gate.alpha``.

    Returns:
        ``q_h``. The prediction set is then
        ``{classes with softmax >= 1 - q_h}`` (spec 6.6 signal 1).

    Raises:
        ValueError: if ``scores`` is empty or ``alpha`` is outside ``(0, 1)``.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("conformal_quantile needs at least one nonconformity score")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha!r}")
    ordered = sorted(float(s) for s in scores)
    index = _ceil_k(n, alpha) - 1
    # n < 1/alpha - 1 makes the requested order statistic fall past the end: the
    # calibration set is too small for this alpha and the only honest answer is
    # the largest observed score (q = max => the widest set this data supports).
    index = max(0, min(index, n - 1))
    return ordered[index]


# ---------------------------------------------------------------------------
# 2. The novelty index (spec 6.6 signal 2)
# ---------------------------------------------------------------------------


def _as_matrix(vectors: Any, *, name: str = "vectors") -> np.ndarray:
    """Coerce ``vectors`` to a contiguous ``(n, d)`` float32 numpy array."""
    if hasattr(vectors, "detach"):  # torch.Tensor without importing torch
        vectors = vectors.detach().cpu().numpy()
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, d), got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or inf; refusing to index a broken encoder output")
    return np.ascontiguousarray(array, dtype=np.float32)


def _l2_normalize(array: np.ndarray, eps: float) -> np.ndarray:
    """Row-wise L2 normalization. A zero row stays zero (cosine 0 => reads novel)."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms < eps, np.float32(1.0), norms.astype(np.float32))
    return np.ascontiguousarray((array / safe).astype(np.float32))


class _ExactNoveltyIndex:
    """Exact brute-force cosine index -- the :class:`NoveltyIndex` protocol.

    Mathematically identical to ``faiss.IndexFlatIP`` over the same L2-normalized
    vectors (both are exhaustive inner product), so ``novelty_distance`` does not
    depend on which backend is present. Used when ``faiss`` is unavailable or
    ``calibrate.novelty_index_backend`` is ``numpy``.
    """

    backend = "numpy"

    def __init__(self, vectors: np.ndarray, eps: float) -> None:
        self._vectors = vectors
        self._eps = float(eps)

    @property
    def dim(self) -> int:
        return int(self._vectors.shape[1])

    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    @property
    def vectors(self) -> np.ndarray:
        """The normalized indexed vectors (read-only by contract; do not mutate)."""
        return self._vectors

    def max_cosine(self, vectors: Any) -> list[float]:
        """Return the max cosine to any indexed vector, per query row."""
        queries = _l2_normalize(_as_matrix(vectors, name="query vectors"), self._eps)
        if queries.shape[1] != self.dim:
            raise ValueError(f"query dim {queries.shape[1]} != index dim {self.dim}")
        sims = queries @ self._vectors.T
        return [float(v) for v in sims.max(axis=1)]


class _FaissNoveltyIndex:
    """``faiss.IndexFlatIP`` over L2-normalized vectors -- inner product IS cosine.

    Flat (exact) is mandatory: an approximate index would make
    ``novelty_distance`` depend on build order, breaking spec 10 determinism.
    """

    backend = "faiss"

    def __init__(self, vectors: np.ndarray, eps: float) -> None:
        import faiss  # local import: optional dependency

        self._vectors = vectors
        self._eps = float(eps)
        self._index = faiss.IndexFlatIP(int(vectors.shape[1]))
        self._index.add(vectors)

    @property
    def dim(self) -> int:
        return int(self._index.d)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @property
    def vectors(self) -> np.ndarray:
        """The normalized indexed vectors (read-only by contract; do not mutate)."""
        return self._vectors

    def max_cosine(self, vectors: Any) -> list[float]:
        """Return the max cosine to any indexed vector, per query row."""
        queries = _l2_normalize(_as_matrix(vectors, name="query vectors"), self._eps)
        if queries.shape[1] != self.dim:
            raise ValueError(f"query dim {queries.shape[1]} != index dim {self.dim}")
        sims, _ = self._index.search(queries, 1)
        return [float(v) for v in sims[:, 0]]


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except Exception:
        return False
    return True


def build_novelty_index(context_vectors: Any, cfg: dict[str, Any]) -> NoveltyIndex:
    """Build the FAISS index over ALL train context vectors (spec 6.6 signal 2).

    ``faiss.IndexFlatIP`` over L2-normalized vectors, so inner product is
    cosine. Flat (exact) is required: an approximate index would make
    ``novelty_distance`` depend on index build order, and spec 10 demands
    determinism.

    Args:
        context_vectors: ``(n_train_turns, d)`` float32 array.
        cfg: Resolved config.

    Returns:
        A :class:`NoveltyIndex`.
    """
    eps = float(get_dotted(cfg, "calibrate.norm_epsilon"))
    backend = str(get_dotted(cfg, "calibrate.novelty_index_backend")).lower()
    if backend not in ("auto", "faiss", "numpy"):
        raise ValueError(
            f"calibrate.novelty_index_backend must be auto|faiss|numpy, got {backend!r}"
        )
    array = _as_matrix(context_vectors, name="context_vectors")
    if array.shape[0] == 0:
        raise ValueError("cannot build a novelty index from zero context vectors")
    normalized = _l2_normalize(array, eps)
    if backend == "faiss" and not _faiss_available():
        raise ContractViolation(
            "calibrate.novelty_index_backend is 'faiss' but faiss is not importable. "
            "pip install faiss-cpu==1.12.0, or set calibrate.novelty_index_backend=numpy "
            "(the numpy backend is exact and gives identical cosines)."
        )
    if backend == "faiss" or (backend == "auto" and _faiss_available()):
        return _FaissNoveltyIndex(normalized, eps)
    return _ExactNoveltyIndex(normalized, eps)


def save_novelty_index(index: NoveltyIndex, cfg: dict[str, Any], seed: int) -> str:
    """Persist the novelty index; the path goes into :attr:`Calibration.novelty_index_path`.

    Stored as the L2-NORMALIZED float32 matrix in a ``.npz``, not as a
    ``faiss.write_index`` blob: the index is flat/exact, so rebuilding it on load
    is an O(n) copy and produces bit-identical search results, while the ``.npz``
    stays readable on a machine where faiss is not installed (see README
    deviation 3).

    Args:
        index: From :func:`build_novelty_index`.
        cfg: Resolved config.
        seed: The training seed whose encoder produced the vectors.

    Returns:
        The absolute path written.
    """
    vectors = getattr(index, "vectors", None)
    if vectors is None:
        raise ContractViolation(
            "save_novelty_index needs the indexed vectors; pass an index built by "
            "build_novelty_index (it exposes .vectors)"
        )
    path = resolve_path(cfg, "paths.novelty_index_template").format(seed=int(seed))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    matrix = _as_matrix(vectors, name="index vectors")
    with open(path, "wb") as handle:
        np.savez(
            handle,
            vectors=matrix,
            dim=np.int64(matrix.shape[1]),
            size=np.int64(matrix.shape[0]),
            seed=np.int64(seed),
            backend=np.str_(getattr(index, "backend", "unknown")),
            normalized=np.bool_(True),
        )
    return path


def load_novelty_index(path: str, cfg: dict[str, Any]) -> NoveltyIndex:
    """Load a persisted novelty index.

    Args:
        path: From :attr:`Calibration.novelty_index_path`.
        cfg: Resolved config.

    Returns:
        A :class:`NoveltyIndex`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"novelty index not found: {path}. Run `python -m reflex calibrate "
            f"--checkpoint ... --seed ...` first."
        )
    with np.load(path, allow_pickle=False) as payload:
        vectors = np.ascontiguousarray(payload["vectors"], dtype=np.float32)
        stored_dim = int(payload["dim"])
        stored_size = int(payload["size"])
    if vectors.shape != (stored_size, stored_dim):
        raise ContractViolation(
            f"novelty index {path} is inconsistent: vectors {vectors.shape} vs "
            f"recorded (size={stored_size}, dim={stored_dim})"
        )
    # Vectors were normalized before writing; build_novelty_index normalizes
    # again (idempotent) so a hand-written file cannot smuggle in un-normalized
    # rows and silently inflate every cosine.
    return build_novelty_index(vectors, cfg)


# ---------------------------------------------------------------------------
# 3. Reporting-only diagnostics -- NEVER a routing input
# ---------------------------------------------------------------------------


def _expected_calibration_error(
    probs: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
    n_bins: int,
) -> tuple[float, list[dict[str, float]]]:
    """Equal-width ECE over the top-1 probability. REPORTING ONLY (see module docstring).

    Returns ``(ece, reliability_bins)``. Deliberately private: spec 8.5's ECE is
    owned by :func:`reflex.evaluate.calibration_metrics`, and no threshold in
    this module or in :mod:`reflex.gate` may read this value.
    """
    confidences: list[float] = []
    hits: list[float] = []
    for row, gold in enumerate(gold_indices):
        gold = int(gold)
        if gold < 0:
            continue
        vector = np.asarray(probs[row], dtype=np.float64)
        if vector.size == 0:
            continue
        top = int(np.argmax(vector))
        confidences.append(float(vector[top]))
        hits.append(1.0 if top == gold else 0.0)
    total = len(confidences)
    bins: list[dict[str, float]] = []
    if total == 0:
        return 0.0, bins
    conf = np.asarray(confidences)
    acc = np.asarray(hits)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(int(n_bins)):
        low, high = float(edges[i]), float(edges[i + 1])
        mask = (conf > low) & (conf <= high) if i > 0 else (conf >= low) & (conf <= high)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin_lo": low, "bin_hi": high, "n": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(acc[mask].mean())
        ece += (count / total) * abs(bin_acc - bin_conf)
        bins.append(
            {"bin_lo": low, "bin_hi": high, "n": count, "confidence": bin_conf, "accuracy": bin_acc}
        )
    return float(ece), bins


def _in_sample_coverage(scores: Sequence[float], q: float) -> float:
    """Fraction of calibration scores at or below ``q``. IN-SAMPLE, diagnostic only."""
    if not scores:
        return 0.0
    return float(sum(1 for s in scores if float(s) <= float(q)) / len(scores))


# ---------------------------------------------------------------------------
# 4. calibrate(): the spec 6.8 pipeline
# ---------------------------------------------------------------------------

#: Functions in other modules that ``calibrate`` cannot run without. Checked by
#: identity against the frozen stubs so the failure message names the module to
#: implement instead of surfacing a bare NotImplementedError from three frames down.
_REQUIRED: tuple[tuple[str, str], ...] = (
    ("data", "load_ontology"),
    ("data", "load_raw_abcd"),
    ("data", "build_partitions"),
    ("data", "build_context"),
    ("data", "iter_agent_turns"),
    ("data", "subflow_list"),
    ("data", "action_list"),
    ("data", "turn_key"),
    ("compile", "load_bank"),
    ("compile", "load_turn_labels"),
    ("models", "load_checkpoint"),
    ("select", "build_selector"),
    ("select", "score_turn"),
)


def _require_dependencies() -> None:
    """Raise a NotImplementedError naming every dependency still a contract stub."""
    import importlib

    from reflex import contracts

    missing: list[str] = []
    for module_name, func_name in _REQUIRED:
        module = importlib.import_module(f"reflex.{module_name}")
        impl = getattr(module, func_name, None)
        if impl is None or impl is getattr(contracts, func_name):
            missing.append(f"reflex.{module_name}.{func_name}")
    if missing:
        raise NotImplementedError(
            "reflex.calibrate needs these to be implemented first (still contract stubs): "
            + ", ".join(missing)
            + ". calibrate() reads a trained checkpoint and dev scores; it cannot "
            "manufacture either."
        )


def _alpha_key(alpha: float) -> str:
    """JSON key for an alpha in ``alpha_sweep_quantiles``: ``repr(float(alpha))``.

    ``0.10`` therefore keys as ``"0.1"``. Consumers must use this helper's rule
    rather than formatting the alpha themselves.
    """
    return repr(float(alpha))


def _stride_subsample(items: list[Any], cap: int, *, what: str) -> list[Any]:
    """Deterministically keep at most ``cap`` of ``items``, spread across the whole list.

    A cap of 0 (the default everywhere) means "no cap" and returns ``items``
    unchanged, so a full run is byte-identical to one with these keys absent.

    An EVEN STRIDE, not a head slice. The novelty index and the conformal
    quantiles are both distributional objects; taking the first N rows would
    take the first N conversations, and ABCD conversations are grouped by
    nothing in particular but are certainly not a random sample of subflows in
    file order. A stride keeps the sampled rows spread over the split at zero
    cost and stays deterministic, which spec 10 requires.
    """
    if cap <= 0 or len(items) <= cap:
        return items
    step = len(items) / float(cap)
    kept = [items[min(len(items) - 1, int(i * step))] for i in range(cap)]
    warnings.warn(
        f"calibrate: {what} SUBSAMPLED to {len(kept)} of {len(items)} by an even stride. "
        "This calibration is a smoke/development artifact: its quantiles come from fewer "
        "points than the split holds and its novelty index covers less of train, so "
        "novelty_distance reads HIGH and the gate escalates more than a full calibration "
        "would. Set the calibrate.max_* keys to 0 for a reportable calibration.",
        RuntimeWarning,
        stacklevel=2,
    )
    return kept


def _query_vectors(encoded: Any) -> Any:
    """Pull the vector the novelty index is defined on out of an encoder result.

    D11, second half. ``models.ModernBERTReflex.encode_context`` returns a
    ``dict`` -- ``{"pooled", "query"}`` -- not a bare tensor, and its own
    docstring names ``query`` as "the vector H1-H7 read and the one
    :mod:`reflex.calibrate` should index for novelty". ``select.score_turn``
    reads exactly that key off the forward pass before calling
    ``novelty_index.max_cosine``, so indexing anything else here would build the
    index in one space and query it in another: every ``novelty_distance`` would
    be meaningless and the ``novel`` gate reason would fire at random.

    ``pooled`` is deliberately NOT accepted as a fallback. It is a real vector of
    the right rank, so a fallback would silently produce that mismatch instead of
    raising. A stand-in model that returns a bare tensor or array is passed
    through unchanged, which keeps test doubles simple.
    """
    if isinstance(encoded, Mapping):
        if "query" not in encoded:
            raise ContractViolation(
                "encode_context returned a mapping without a 'query' key "
                f"(got {sorted(encoded)!r}). The novelty index must be built on the "
                "same vector select.score_turn queries it with -- models.py calls "
                "that 'query' -- so there is no safe fallback (D11)."
            )
        return encoded["query"]
    return encoded


def _encode_contexts(model: Any, contexts: Sequence[ContextWindow], cfg: dict[str, Any]) -> np.ndarray:
    """Encode context strings to an ``(n, d)`` float32 matrix via the checkpoint's encoder.

    ``models.build_model`` is contracted to expose ``encode_context`` but its
    argument shape is not frozen, so the accepted shapes are tried in a fixed
    order and anything else fails loudly rather than being guessed at.
    """
    texts = [c.text for c in contexts]
    if not texts:
        raise ValueError("no train contexts to index; check Partitions.train")
    fn = getattr(model, "encode_context", None) or getattr(model, "encode_contexts", None)
    if not callable(fn):
        raise ContractViolation(
            "the checkpoint's model exposes no callable encode_context; "
            "reflex.models.build_model must provide it (spec 6.4)"
        )
    batch_size = max(1, int(get_dotted(cfg, "calibrate.encode_batch_size")))

    # D11: reflex.models.encode_context takes TENSORS (input_ids, attention_mask),
    # not a list[str] -- the same shape encode_templates takes. The caller adapts to
    # the frozen contract. The previous three-shape fallback caught only TypeError,
    # and a list[str] passed as `input_ids` does NOT raise TypeError: it sails past
    # the signature and dies deep inside the encoder with an unrelated traceback,
    # so the fallback made a legible error illegible.
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ContractViolation(
            "the checkpoint's model exposes encode_context but no .tokenizer; "
            "reflex.calibrate must tokenize before encoding (spec 6.4)"
        )
    max_len = int(get_dotted(cfg, "data.max_len"))
    # D13: the state line is appended LAST, so overflow must drop the OLDEST turns.
    # Set unconditionally and not restored: left truncation is correct for every
    # use of this tokenizer in this build, so there is no caller to restore it for.
    try:
        tokenizer.truncation_side = str(get_dotted(cfg, "train.truncation_side"))
    except Exception:  # pragma: no cover - tokenizer without the attribute
        pass

    def _call(chunk: list[str]) -> Any:
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        try:
            device = next(model.parameters()).device
        except Exception:  # pragma: no cover - non-torch stand-ins in tests
            device = None
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", None)
        if device is not None:
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        return _query_vectors(fn(input_ids, attention_mask))

    chunks: list[np.ndarray] = []
    try:
        import torch

        context_manager: Any = torch.no_grad()
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        import contextlib

        context_manager = contextlib.nullcontext()
    with context_manager:
        for start in range(0, len(texts), batch_size):
            chunks.append(_as_matrix(_call(texts[start : start + batch_size]), name="context vectors"))
    matrix = np.vstack(chunks)
    if matrix.shape[0] != len(texts):
        raise ContractViolation(
            f"encode_context returned {matrix.shape[0]} vectors for {len(texts)} contexts"
        )
    return matrix


def _iter_contexts(
    partition: dict[int, list[NormalizedTurn]],
    scenarios: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
) -> list[tuple[int, int, NormalizedTurn, ContextWindow]]:
    """Build one :class:`ContextWindow` per agent turn, in deterministic order."""
    from reflex.data import build_context, iter_agent_turns

    rows: list[tuple[int, int, NormalizedTurn, ContextWindow]] = []
    for convo_id, turn_index, turn in iter_agent_turns(partition):
        turns = partition[convo_id]
        context = build_context(turns, turn_index, scenarios.get(int(convo_id), {}), cfg)
        rows.append((int(convo_id), int(turn_index), turn, context))
    return rows


def _scenarios_by_convo(raw: dict[str, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    """``convo_id -> scenario``. Needed because ``Partitions`` carries only turns."""
    out: dict[int, dict[str, Any]] = {}
    for convos in raw.values():
        for convo in convos:
            out[int(convo["convo_id"])] = convo.get("scenario", {}) or {}
    return out


def _gold_index(class_list: Sequence[str], gold: Optional[str], head: str) -> int:
    """Index of ``gold`` in a head's canonical class order; ``-1`` when not applicable."""
    if gold is None:
        return -1
    try:
        return list(class_list).index(gold)
    except ValueError as exc:
        raise ContractViolation(
            f"gold {head} label {gold!r} is not in the head's canonical class list "
            f"({len(class_list)} classes). The checkpoint and the ontology/bank disagree."
        ) from exc


def _value_gold_index(candidates: Sequence[str], gold_values: Sequence[str]) -> int:
    """Index of a gold value inside ONE H4 slot's candidate list; ``-1`` when absent.

    Mirrors the H7 template rule (the gold has to BE one of the candidates the
    model scored) and :mod:`reflex.evaluate`'s value comparison (case-folded
    string equality).

    It deliberately does NOT read ``turn.values[i]`` as the gold for slot ``i``:
    ABCD's gold value list is not positionally aligned with an action's slot
    list. ``verify-identity`` carries three golds that are
    name/account_id/order_id on one turn and name/zip_code/phone on the next,
    and ``select._score_values`` drops slots with no candidate columns, so both
    sides move independently. Membership is the only alignment that holds.

    WHAT A COPY-TIER CANDIDATE IS CALLED. ``select._slot_columns`` names a copy
    column by the value the customer has DISCLOSED for that slot
    (``select._disclosed_lexicon``) and falls back to the bare ``<marker>`` only
    when nothing has been disclosed. So a NON-ENUMERABLE gold (a name, an email,
    an order id) DOES match here once the customer has said it, and matches
    nothing before that -- the undisclosed case stays a coverage miss instead of
    being guessed at. (This docstring used to say copy candidates were always
    markers and therefore never matched; that stopped being true when select
    began naming copy columns by their disclosed value.)

    This function only answers "is the gold among the candidates". Whether a
    matched slot is allowed to SET the threshold is the caller's decision: see
    the single-candidate rule in :func:`_collect_dev_scores`.
    """
    wanted = {str(value).strip().lower() for value in gold_values}
    if not wanted:
        return -1
    for index, name in enumerate(candidates):
        if str(name).strip().lower() in wanted:
            return index
    return -1


def _collect_dev_scores(
    cfg: dict[str, Any],
    selector: Any,
    rows: Sequence[tuple[int, int, NormalizedTurn, ContextWindow]],
    labels: dict[str, Any],
    class_orders: dict[str, Sequence[str]],
    diagnostics: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, list[list[float]]], dict[str, list[int]], list[float], dict[str, int]]:
    """Score dev and bucket ``(probs, gold_index)`` per head. No decisions, no routing.

    Returns ``(probs, golds, novelty, template_coverage)``. ``template_coverage``
    is ``{"n_positions": ..., "n_gold_present": ...}`` -- see the note on
    ``h7_absent_gold`` below for why it exists as a SEPARATE return value
    rather than folded into ``golds["template"]``.

    ``diagnostics``, when given, is filled in place with the two exclusion
    counts that are otherwise invisible in the outputs -- ``branch_drops`` (gold
    take_action / retrieve_utterance turns the model routed the other way, so
    the head the gate would have consulted is not the head their gold belongs
    to) and ``value_coverage`` (how many H4 slots had their gold among the
    candidate columns, and how many of those were single-candidate slots kept
    out of the quantile). Both are REPORTING ONLY; nothing reads them back.
    """
    from reflex.data import turn_key
    # select's own argmax, imported rather than re-implemented: this has to
    # break ties the same way score_turn does or calibrate would disagree with
    # the selector about which branch a turn was predicted into.
    from reflex.select import _argmax, score_turn

    absent_mode = str(get_dotted(cfg, "calibrate.h7_absent_gold")).lower()
    if absent_mode not in ("max_nonconformity", "skip"):
        raise ValueError(
            f"calibrate.h7_absent_gold must be max_nonconformity|skip, got {absent_mode!r}"
        )

    probs: dict[str, list[list[float]]] = {head: [] for head in _HEADS}
    golds: dict[str, list[int]] = {head: [] for head in _HEADS}
    novelty: list[float] = []
    template_coverage = {"n_positions": 0, "n_gold_present": 0}
    value_coverage = {
        "n_slots": 0,
        "n_gold_present": 0,
        "n_no_candidates": 0,
        "n_single_candidate": 0,
        "n_single_candidate_gold_present": 0,
        "n_scored": 0,
    }
    branch_drops = {"action": 0, "skeleton": 0}

    for convo_id, turn_index, turn, context in rows:
        scores = score_turn(selector, context, turn, cfg)
        novelty.append(float(scores.novelty_distance))
        label = labels.get(turn_key("dev", convo_id, turn_index))

        probs["nextstep"].append(list(scores.nextstep_probs))
        golds["nextstep"].append(_gold_index(NEXT_STEPS, turn.nextstep, "nextstep"))
        probs["intent"].append(list(scores.intent_probs))
        golds["intent"].append(_gold_index(class_orders["intent"], turn.intent, "intent"))

        # Applicability follows the PREDICTED nextstep, because that is what
        # gate.evaluate_gate routes on ("never the gold one", its docstring) and
        # therefore the population q_h is applied to. Filtering on turn.nextstep
        # instead made the population depend on select.score_inapplicable_heads:
        # with it off the two filters coincide, but switching it on -- which
        # select.score_turn's docstring recommends as a bias fix -- would quietly
        # start fitting H3/H5/H7 on turns the gate never sends through those
        # heads. Recomputing the argmax is arithmetic, not routing; the gate
        # remains the only decision point.
        predicted_nextstep = NEXT_STEPS[_argmax(scores.nextstep_probs)]
        if turn.nextstep != predicted_nextstep:
            if turn.nextstep == "take_action":
                branch_drops["action"] += 1
            elif turn.nextstep == "retrieve_utterance":
                branch_drops["skeleton"] += 1

        if predicted_nextstep == "take_action" and scores.action_probs:
            probs["action"].append(list(scores.action_probs))
            golds["action"].append(_gold_index(class_orders["action"], turn.action, "action"))

            # H4, one row per value slot of the PREDICTED action, pooled into the
            # single shared quantiles["value"] gate.py asks for. Only a gold
            # take_action turn can supply a gold value, so a turn predicted here
            # whose gold is retrieve_utterance contributes no slot at all --
            # counting its slots would inflate the value_coverage denominator
            # with rows that could never have matched.
            #
            # THE SINGLE-CANDIDATE RULE. A slot with exactly ONE candidate column
            # has the distribution [1.0] (select._softmax of one logit), and
            # select.prediction_set([1.0], q) is {0} for EVERY q in [0, 1]:
            # q_value cannot change what the gate does with that slot. Its
            # nonconformity is exactly 0, so scoring it informs nothing about the
            # slots q_value DOES decide and only dilutes their distribution --
            # with a share f of such rows the alpha-quantile is taken at
            # alpha / (1 - f) of the multi-candidate rows, i.e. those slots are
            # covered below nominal. These are overwhelmingly copy-tier slots
            # whose one column select names by the customer's disclosed value, so
            # they DO match a gold (see _value_gold_index). They are therefore
            # kept out of the quantile (gold index -1, exactly like an absent
            # gold) and reported in value_coverage instead. The stratum is
            # defined by the candidate list alone -- context and predicted
            # action, never the label -- so restricting to it keeps split
            # conformal's exchangeability argument intact.
            gold_values = list(turn.values or []) if turn.nextstep == "take_action" else []
            value_slots = list(scores.value_probs) if gold_values else []
            for slot_index, slot_probs in enumerate(value_slots):
                candidates = (
                    list(scores.value_candidates[slot_index])
                    if slot_index < len(scores.value_candidates)
                    else []
                )
                value_coverage["n_slots"] += 1
                slot_gold = _value_gold_index(candidates, gold_values)
                if slot_gold >= 0:
                    value_coverage["n_gold_present"] += 1
                if len(slot_probs) == 0:
                    # select's "slot with no candidate column" -- the gate's
                    # size-0 escalation. Nothing to score, nothing to match.
                    value_coverage["n_no_candidates"] += 1
                elif len(slot_probs) == 1:
                    value_coverage["n_single_candidate"] += 1
                    if slot_gold >= 0:
                        value_coverage["n_single_candidate_gold_present"] += 1
                    slot_gold = -1
                if slot_gold >= 0:
                    value_coverage["n_scored"] += 1
                # Index-aligned exactly like the template rows below: an absent
                # gold is kept as -1 and skipped by nonconformity_scores, never
                # folded into the distribution that sets the threshold.
                probs["value"].append(list(slot_probs))
                golds["value"].append(slot_gold)

        if predicted_nextstep == "retrieve_utterance" and scores.skeleton_probs:
            gold_skeleton = getattr(label, "skeleton_id", None) if label is not None else None
            probs["skeleton"].append(list(scores.skeleton_probs))
            golds["skeleton"].append(
                _gold_index(class_orders["skeleton"], gold_skeleton, "skeleton")
            )

            # Same denominator rule as the value slots above: H7 positions exist
            # only where the turn HAS gold agent text, so a turn predicted into
            # this branch whose gold is take_action contributes no position.
            gold_templates = list(getattr(label, "template_ids", None) or [])
            template_positions = (
                list(scores.template_probs) if turn.nextstep == "retrieve_utterance" else []
            )
            for position, position_probs in enumerate(template_positions):
                candidates = (
                    list(scores.template_candidates[position])
                    if position < len(scores.template_candidates)
                    else []
                )
                gold_template = gold_templates[position] if position < len(gold_templates) else None
                template_coverage["n_positions"] += 1
                if gold_template is not None and gold_template in candidates:
                    template_coverage["n_gold_present"] += 1
                    probs["template"].append(list(position_probs))
                    golds["template"].append(candidates.index(gold_template))
                elif absent_mode == "max_nonconformity":
                    # BUG FOUND AND FIXED (see module docstring "THE TEMPLATE
                    # QUANTILE COLLAPSE" below): this branch used to append an
                    # explicit s=1 row into the SAME nonconformity distribution
                    # `conformal_quantile` reads to set the accept/reject
                    # threshold. At the real absent rate (~50% of dev template
                    # positions -- the predicted skeleton's candidate pool
                    # simply does not contain the gold template for that
                    # position; see DECISIONS D23), that pushes q to 1.0 and the
                    # threshold to 0.0, so EVERY candidate enters the prediction
                    # set and the singleton gate (gate.py) then rejects every
                    # turn -- reproduced with synthetic data at absent rates as
                    # low as 3%. The original intent (comment removed above:
                    # "dropping it would hide the skeleton head's errors and
                    # inflate coverage") was sound in spirit -- an honest q
                    # should reflect true end-to-end miscoverage -- but mixing
                    # a COVERAGE-ABSENCE fact (is the true answer even a
                    # candidate here) into the same distribution as a
                    # CONFIDENCE fact (how sure is the model among the
                    # candidates it has) makes the quantile answer a different,
                    # much harder question than the one the gate can act on
                    # per-turn, and is mathematically unstable at the coverage
                    # rates this bank actually has. The fix: absent rows are
                    # EXCLUDED from the quantile-determining distribution in
                    # BOTH modes (below), and "max_nonconformity" now means
                    # "count this position in the `template_coverage`
                    # diagnostic as a miss" rather than "poison the threshold
                    # with it" -- the coverage gap is still recorded and
                    # visible, just not folded into a per-turn accept/reject
                    # threshold it cannot honestly answer. `golds["template"]`
                    # already skips negative indices (see `nonconformity_scores`
                    # -- "Rows with a negative index are SKIPPED, not scored"),
                    # so this is what "skip" mode already did for the quantile;
                    # only the coverage BOOKKEEPING differs between the modes.
                    # `probs["template"]` still gets a row appended (the real,
                    # unmodified position_probs -- no extra column) so the two
                    # lists stay index-aligned; nonconformity_scores skips it
                    # via the gold<0 check before ever reading the row.
                    probs["template"].append(list(position_probs))
                    golds["template"].append(-1)
                else:
                    probs["template"].append(list(position_probs))
                    golds["template"].append(-1)

    if diagnostics is not None:
        diagnostics["branch_drops"] = dict(branch_drops)
        diagnostics["value_coverage"] = dict(value_coverage)
    return probs, golds, novelty, template_coverage


def _dev_labels(
    dev_partition: dict[int, list[NormalizedTurn]],
    bank: Bank,
    cfg: dict[str, Any],
) -> list[Any]:
    """The dev turn labels, ALWAYS derived fresh against the compiled bank.

    INTERFACE FIX, found by running the pipeline. ``compile.compile_bank`` writes
    ``labels/train.jsonl`` and NOTHING ELSE -- ``split = "train"`` is hard-set at
    the top of it, correctly, because spec 6.2 compiles from train only. But
    ``calibrate`` needs dev skeleton and template golds to score H5 and H7, and
    called ``load_turn_labels("dev", cfg)`` directly, so a clean tree raised
    ``FileNotFoundError`` before a single dev turn was scored.

    ``train._derive_turn_labels`` already solved exactly this for early stopping
    -- it derives the labels through ``compile``'s own ``split_sentences`` /
    ``label_acts`` and matches against EXISTING bank templates, never extending
    the bank. Reusing it is deliberate: two derivations of the same gold that
    drift apart would make the calibration's H5/H7 quantiles incomparable with
    the dev selection score training stopped on.

    ALWAYS DERIVED, NEVER READ OFF DISK -- this used to try
    ``load_turn_labels("dev", cfg)`` first and only derive on FileNotFoundError,
    which is the trade this function already refused in the other direction (it
    passed ``persist=False`` precisely because ``train --smoke`` labels a
    10-conversation dev subset, and a partial ``labels/dev.jsonl`` left on disk
    would then be silently reused by a later full calibration -- but reading
    whatever is already there has exactly that failure mode, and a worse one).
    Template and skeleton ids are POSITIONAL (``compile`` hands out ``T%06d`` /
    ``S%04d`` by descending count and re-sorts on every recompile), so a labels
    file written against an EARLIER bank still resolves every id -- to a
    different template. ``_gold_index`` cannot catch it: the id exists in the new
    bank, so it returns a valid index for the wrong class instead of raising, and
    ``quantiles['skeleton']``, ``quantiles['template']`` and the whole alpha
    sweep come out fitted against wrong golds with no warning anywhere. Nothing
    guards it on the file either: ``compile.write_turn_labels`` stamps no bank or
    dataset hash and ``compile.load_turn_labels`` checks none, and
    ``paths.labels_dir`` is ONE directory shared across bank fractions, so
    ``compile --fraction`` / ``train --fraction`` runs overwrite each other's dev
    labels. The real fix is provenance on the file itself (a ``__meta__`` header
    carrying bank_hash/dataset_hash, checked on load) and belongs in
    :mod:`reflex.compile`; until that exists, deriving is the only read of these
    golds that cannot be silently stale.

    Note it calls ``train._derive_turn_labels`` and NOT ``train._labels_for``:
    ``_labels_for`` loads ``labels/<split>.jsonl`` first and only derives on
    FileNotFoundError, so routing through it would put the same stale file back
    one frame down. ``_derive_turn_labels`` is the derivation ``_labels_for``
    itself calls, so the "one derivation, no drift" argument above still holds.
    """
    from reflex.train import _derive_turn_labels

    return list(_derive_turn_labels(dev_partition, bank, cfg, "dev"))


def calibrate(cfg: dict[str, Any], checkpoint_path: str, seed: int) -> Calibration:
    """Compute all gate thresholds on dev (spec 6.8). No training.

    **Dev, SEEN SUBFLOWS ONLY** -- spec 6.8's first line. ``Partitions.dev`` is
    already novel-filtered, so use it as given and do not re-add anything.

    Computes ``q_h`` for every head at ``gate.alpha``; the
    ``gate.novelty_percentile`` percentile of dev ``novelty_distance``; and the
    whole ``gate.alpha_sweep`` set of quantiles (spec 6.8 step 3) so E5 needs no
    recalibration.

    Args:
        cfg: Resolved config.
        checkpoint_path: The checkpoint to calibrate. Quantiles are NOT
            transferable across checkpoints.
        seed: That checkpoint's seed.

    Returns:
        A :class:`Calibration`.

    Raises:
        ContractViolation: before any work is done, if the file
            :func:`write_calibration` would write already holds another seed's
            or another checkpoint's calibration (:func:`_refuse_foreign_overwrite`).
    """
    _require_dependencies()

    from reflex.compile import load_bank, load_turn_labels
    from reflex.data import action_list, build_partitions, load_ontology, load_raw_abcd, subflow_list
    from reflex.models import load_checkpoint
    from reflex.select import build_selector

    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    alpha = float(get_dotted(cfg, "gate.alpha"))
    percentile = float(get_dotted(cfg, "gate.novelty_percentile"))
    percentile_method = str(get_dotted(cfg, "calibrate.novelty_percentile_method"))
    min_scores = int(get_dotted(cfg, "calibrate.min_head_scores"))

    # PRE-FLIGHT, before anything expensive or destructive. write_calibration
    # refuses to replace another seed's / checkpoint's file, but it only runs
    # after this function returns (reflex.__main__) -- i.e. after every train
    # context was encoded, every dev turn scored, and the diagnostics sidecar of
    # the calibration being protected had already been overwritten. Asking the
    # same question here costs one small JSON read and fails in the first second.
    _refuse_foreign_overwrite(
        _calibration_write_path(cfg, seed),
        Calibration(alpha=alpha, checkpoint_path=checkpoint_path, seed=int(seed)),
    )

    ontology = load_ontology(cfg)
    bank = load_bank(cfg)
    partitions = build_partitions(cfg)
    scenarios = _scenarios_by_convo(load_raw_abcd(cfg))

    class_orders: dict[str, Sequence[str]] = {
        "nextstep": list(NEXT_STEPS),
        "intent": list(subflow_list(ontology)),
        "action": list(action_list(ontology)),
        "skeleton": [s.skeleton_id for s in bank.skeletons],
    }

    # --- step 1: the novelty index, over TRAIN contexts, BEFORE the selector,
    #     because select.build_selector needs Calibration.novelty_index_path.
    model, _ = load_checkpoint(checkpoint_path, cfg, bank, ontology)
    train_rows = _stride_subsample(
        _iter_contexts(partitions.train, scenarios, cfg),
        int(get_dotted(cfg, "calibrate.max_train_contexts")),
        what="the train contexts behind the novelty index",
    )
    index = build_novelty_index(_encode_contexts(model, [r[3] for r in train_rows], cfg), cfg)
    index_path = save_novelty_index(index, cfg, seed)

    provisional = Calibration(
        alpha=alpha,
        novelty_index_path=index_path,
        checkpoint_path=checkpoint_path,
        seed=int(seed),
    )
    selector = build_selector(cfg, checkpoint_path, bank, ontology, provisional)

    # --- step 2: nonconformity on dev (seen subflows only -- already filtered)
    dev_rows = _stride_subsample(
        _iter_contexts(partitions.dev, scenarios, cfg),
        int(get_dotted(cfg, "calibrate.max_dev_turns")),
        what="the dev turns behind the conformal quantiles",
    )
    labels = {label.turn_id: label for label in _dev_labels(partitions.dev, bank, cfg)}
    collection_diagnostics: dict[str, Any] = {}
    probs, golds, novelty, template_coverage = _collect_dev_scores(
        cfg, selector, dev_rows, labels, class_orders, collection_diagnostics
    )
    scores_by_head = {
        head: nonconformity_scores(probs[head], golds[head]) for head in _HEADS
    }

    for head, head_scores in scores_by_head.items():
        if len(head_scores) < min_scores:
            raise ContractViolation(
                f"head {head!r} produced {len(head_scores)} dev nonconformity scores, "
                f"below calibrate.min_head_scores={min_scores}. A quantile from that few "
                f"points is not a calibration."
            )

    # --- step 3: q_h at gate.alpha and across the whole sweep (spec 6.8 step 3)
    sweep_alphas = [alpha] + [float(a) for a in get_dotted(cfg, "gate.alpha_sweep")]
    alpha_sweep_quantiles: dict[str, dict[str, float]] = {}
    for sweep_alpha in sweep_alphas:
        key = _alpha_key(sweep_alpha)
        if key in alpha_sweep_quantiles:
            continue
        alpha_sweep_quantiles[key] = {
            head: conformal_quantile(scores_by_head[head], sweep_alpha) for head in _HEADS
        }
    quantiles = dict(alpha_sweep_quantiles[_alpha_key(alpha)])

    # --- step 4: the novelty threshold
    if not novelty:
        raise ContractViolation("no dev turns produced a novelty_distance")
    novelty_threshold = float(
        np.percentile(np.asarray(novelty, dtype=np.float64), percentile, method=percentile_method)
    )

    calibration = Calibration(
        alpha=alpha,
        quantiles=quantiles,
        novelty_threshold=novelty_threshold,
        alpha_sweep_quantiles=alpha_sweep_quantiles,
        n_dev_turns={head: len(scores_by_head[head]) for head in _HEADS},
        novelty_index_path=index_path,
        checkpoint_path=checkpoint_path,
        seed=int(seed),
    )

    # --- step 5: diagnostics. REPORTING ONLY; nothing above reads this back.
    if bool(get_dotted(cfg, "calibrate.write_diagnostics")):
        _write_diagnostics(
            cfg, calibration, probs, golds, scores_by_head, novelty, len(dev_rows),
            template_coverage, collection_diagnostics,
        )
    return calibration


def _write_diagnostics(
    cfg: dict[str, Any],
    calibration: Calibration,
    probs: dict[str, list[list[float]]],
    golds: dict[str, list[int]],
    scores_by_head: dict[str, list[float]],
    novelty: Sequence[float],
    n_dev_rows: int,
    template_coverage: dict[str, int],
    collection_diagnostics: Optional[dict[str, Any]] = None,
) -> str:
    """Write the dev ECE / in-sample-coverage sidecar. REPORTING ONLY, never a gate."""
    n_bins = int(get_dotted(cfg, "eval.ece_bins"))
    ece: dict[str, float] = {}
    reliability: dict[str, list[dict[str, float]]] = {}
    for head in _HEADS:
        value, bins = _expected_calibration_error(probs[head], golds[head], n_bins)
        ece[head] = value
        reliability[head] = bins
    coverage = {
        key: {
            head: _in_sample_coverage(scores_by_head[head], quantile)
            for head, quantile in head_quantiles.items()
        }
        for key, head_quantiles in calibration.alpha_sweep_quantiles.items()
    }
    array = np.asarray(novelty, dtype=np.float64)
    extra = dict(collection_diagnostics or {})
    value_coverage = dict(extra.get("value_coverage") or {})
    branch_drops = dict(extra.get("branch_drops") or {})
    payload = {
        "WARNING": _DIAGNOSTIC_BANNER,
        "not_a_gate": True,
        "split": "dev",
        "seed": calibration.seed,
        "checkpoint_path": calibration.checkpoint_path,
        "alpha": calibration.alpha,
        "n_dev_agent_turns": int(n_dev_rows),
        "n_scores_per_head": {head: len(scores_by_head[head]) for head in _HEADS},
        "ece_bins": n_bins,
        "dev_ece": ece,
        "dev_reliability": reliability,
        "dev_in_sample_coverage": coverage,
        "template_coverage": {
            **template_coverage,
            "gold_present_rate": (
                template_coverage["n_gold_present"] / template_coverage["n_positions"]
            ) if template_coverage["n_positions"] else None,
            "note": (
                "Positions where the predicted skeleton's candidate pool did not "
                "contain the gold template -- a coverage-absence fact, distinct from "
                "the template head's CONFIDENCE among the candidates it does have. "
                "This is reported here, separately, precisely so it is never folded "
                "back into `quantiles['template']` -- see the h7_absent_gold handling "
                "in _collect_dev_scores for why that used to collapse the gate."
            ),
        },
        "value_coverage": {
            **value_coverage,
            "gold_present_rate": (
                value_coverage["n_gold_present"] / value_coverage["n_slots"]
            ) if value_coverage.get("n_slots") else None,
            "scored_rate": (
                value_coverage["n_scored"] / value_coverage["n_slots"]
            ) if value_coverage.get("n_slots") and "n_scored" in value_coverage else None,
            "note": (
                "Which H4 slots set `quantiles['value']`. n_slots counts slots of "
                "turns that HAVE at least one gold value: a gold button taking no "
                "arguments (11,327 of ABCD's 36,482 action turns) poses no H4 "
                "question, so it is not a miss and is not in the denominator. "
                "n_gold_present is how many of those slots had the gold among their "
                "candidates; a copy-tier candidate is named by the value the customer "
                "DISCLOSED (select._slot_columns), so it matches once the customer "
                "has said it and is a miss before that. n_scored is the subset that "
                "actually carries a nonconformity score: gold present AND at least "
                "two candidates. n_single_candidate slots are excluded on purpose -- "
                "their distribution is [1.0], the gate's set for them is that one "
                "candidate at every q, and their nonconformity of exactly 0 would only "
                "dilute the distribution that decides the multi-candidate slots "
                "(at a share f of such rows, those slots would be covered at "
                "1 - alpha/(1-f), not 1 - alpha). n_single_candidate_gold_present "
                "reports how many rows that rule removed; n_no_candidates is the "
                "gate's size-0 escalation case."
            ),
        },
        "branch_drops": {
            **branch_drops,
            "note": (
                "Gold take_action / retrieve_utterance dev turns whose PREDICTED "
                "nextstep was the other branch. They carry no H3/H4 (resp. H5/H7) "
                "score, because those heads are calibrated on the population the "
                "gate applies them to -- the predicted branch, per "
                "gate.evaluate_gate -- and a turn the gate would route through the "
                "other head cannot inform this one's threshold. Recorded here so "
                "the exclusion is visible rather than implicit in n_scores_per_head."
            ),
        },
        "calibration_set_used_for_model_selection": True,
        "calibration_set_independence_note": (
            "NOT AN INDEPENDENT CALIBRATION SET. The quantiles above were fitted on "
            "the same partitions.dev turns reflex.train early-stops on (it keeps the "
            "epoch that maximizes dev_selection_score), so the calibration scores are "
            "not independent of the fitted predictor and the nonconformity "
            "distribution is optimistically low. Split conformal's 1-alpha coverage "
            "is therefore NOMINAL here, not guaranteed, and no coverage claim quoted "
            "from these quantiles may be called exact. Making it real needs "
            "partitions.dev split by convo_id into a fit half (early stopping) and a "
            "calibration half (these quantiles), which is a change to reflex.data and "
            "reflex.train, not to this module. The independent measurement is the "
            "test-side empirical coverage from reflex.evaluate.calibration_metrics, "
            "which report.py checks as criterion 4."
        ),
        "novelty_distance": {
            "percentile": float(get_dotted(cfg, "gate.novelty_percentile")),
            "threshold": calibration.novelty_threshold,
            "n": int(array.size),
            "min": float(array.min()) if array.size else 0.0,
            "mean": float(array.mean()) if array.size else 0.0,
            "max": float(array.max()) if array.size else 0.0,
        },
    }
    path = resolve_path(cfg, "paths.calibration_diagnostics_path")
    if "{seed}" in path:
        # Same placeholder rule as paths.calibration_path: a per-seed calibration
        # whose sidecar is one shared file would keep only the last seed's
        # value_coverage / branch_drops / ECE.
        path = path.format(seed=int(calibration.seed))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------


def _calibration_write_path(cfg: dict[str, Any], seed: int) -> str:
    """Where ``seed``'s calibration is written: ``paths.calibration_path``, ``{seed}`` formatted.

    The ONE formatting rule, shared by :func:`write_calibration` and the
    pre-flight check in :func:`calibrate`. ``reflex.run._resolve_calibration``
    and ``reflex.evaluate._load_calibration`` format the read side the same way
    (``"{seed}" in path`` -> ``path.format(seed=int(seed))``) and hand the
    concrete file to :func:`load_calibration`.
    """
    path = resolve_path(cfg, "paths.calibration_path")
    if "{seed}" in path:
        path = path.format(seed=int(seed))
    return path


def _calibration_identity(calibration: Calibration) -> tuple[int, str]:
    """What a calibration file belongs to: ``(seed, absolute checkpoint path)``."""
    checkpoint = str(calibration.checkpoint_path or "")
    return int(calibration.seed), os.path.abspath(checkpoint) if checkpoint else ""


def _refuse_foreign_overwrite(path: str, calibration: Calibration) -> None:
    """Refuse to replace a calibration fitted for a DIFFERENT seed or checkpoint.

    ``paths.calibration_path`` is one un-templated file while quantiles belong to
    exactly one checkpoint and one seed (this module's own contract, repeated in
    the ``Calibration`` docstring). Spec 6.4 asks for every ``train.seeds`` entry
    to be run, so the second ``reflex calibrate`` silently destroyed the first
    seed's quantiles, its entire ``alpha_sweep_quantiles`` table and the
    ``novelty_index_path`` binding them to a seed-scoped ``.npz`` -- none of it
    re-derivable without re-running the calibration. Refusing costs one manual
    step; overwriting costs a run.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            existing = Calibration.from_dict(json.load(handle))
    except (OSError, ValueError, TypeError, KeyError):
        # Unreadable or not a Calibration: there is no identity to protect.
        return
    if _calibration_identity(existing) == _calibration_identity(calibration):
        return
    raise ContractViolation(
        f"{path} already holds the calibration for seed {int(existing.seed)} / "
        f"{existing.checkpoint_path or '<unrecorded>'!r}, but this one is for seed "
        f"{int(calibration.seed)} / {calibration.checkpoint_path!r}. Overwriting it "
        f"would destroy that seed's quantiles, its whole alpha sweep and the novelty "
        f"index they are bound to, and nothing re-derives them. Point "
        f"paths.calibration_path at a per-seed path -- it accepts a {{seed}} "
        f"placeholder, e.g. outputs/calibration/gate_seed{{seed}}.json, formatted here "
        f"exactly as paths.novelty_index_template already is -- or move the existing "
        f"file aside first."
    )


def write_calibration(calibration: Calibration, cfg: dict[str, Any]) -> str:
    """Persist to ``paths.calibration_path`` (``outputs/calibration/gate.json``).

    ``paths.calibration_path`` may carry a ``{seed}`` placeholder, formatted from
    ``calibration.seed`` exactly as :func:`save_novelty_index` formats
    ``paths.novelty_index_template``. With the un-templated default every seed
    writes the same file, so a calibration belonging to another seed or
    checkpoint is refused rather than clobbered -- see
    :func:`_refuse_foreign_overwrite`.

    Args:
        calibration: From :func:`calibrate`.
        cfg: Resolved config.

    Returns:
        The absolute path written.

    Raises:
        ContractViolation: if the target file holds another seed's or another
            checkpoint's calibration.
    """
    path = _calibration_write_path(cfg, calibration.seed)
    _refuse_foreign_overwrite(path, calibration)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(calibration.to_dict(), handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def load_calibration(cfg: dict[str, Any], path: Optional[str] = None) -> Calibration:
    """Load a persisted calibration.

    Args:
        cfg: Resolved config. Uses ``paths.calibration_path`` when ``path`` is None.
        path: Explicit override.

    Returns:
        A :class:`Calibration`.

    Raises:
        FileNotFoundError: if it has not been produced yet.
    """
    resolved = path if path is not None else resolve_path(cfg, "paths.calibration_path")
    if not os.path.isabs(resolved):
        resolved = os.path.join(os.path.dirname(resolve_path(cfg, "paths.calibration_path")), resolved)
    if "{seed}" in resolved:
        # This function's signature is frozen (reflex.contracts) and carries no
        # seed, so a per-seed template can only be resolved by the caller, which
        # is the one that knows the run's seed. Checked on the FINAL path: a
        # relative `path` is joined onto the configured directory, which may be
        # the templated part.
        raise ContractViolation(
            f"calibration path {resolved!r} still carries the {{seed}} placeholder of "
            f"a per-seed paths.calibration_path; load_calibration takes no seed, so "
            f"the caller must format it (path.format(seed=...), exactly as "
            f"write_calibration does) and pass the concrete file as `path`."
        )
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"calibration not found: {resolved}. Run `python -m reflex calibrate "
            f"--checkpoint <path> --seed <n>` first."
        )
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Calibration.from_dict(payload)
