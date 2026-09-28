"""The interface this harness needs from the recency-tagging featurizer.

OWNERSHIP
---------
The featurizer (windowing + recency tagging + TF-IDF) is built by a DIFFERENT
agent. This module contains no implementation of it -- deliberately. It contains
only the contract, the loader, and a conformance check that can be run with no
corpus on disk, so the two halves can be wired together and verified before the
coordinator spends a full-corpus parse on them.

If ``load_featurizer`` raises, the featurizer is not wired yet. That is the
correct failure: this harness must not silently fall back to a private rendering,
because a probe result measured on a rendering that ``reflex.select`` cannot
reproduce is not transferable to the ship path.

THE TWO JOBS, KEPT SEPARATE
---------------------------
1. RENDER: ``ContextWindow -> str``, applying a window K, optional recency
   tagging, and optionally an order-destroying permutation (the D7 control).
2. VECTORIZE: ``list[str] -> sparse matrix``, word 1-2gram TF-IDF, fit on train
   rows only and applied to dev rows.

They are separate because the D7 controls act on (1) while (2) must be held
byte-identical across the control arms, or the control measures two things at
once (DECISIONS D19: "an ablation is only as good as its claim to isolate one
variable").

THE CONFORMANCE REQUIREMENT, WHICH IS THE POINT OF THIS FILE
------------------------------------------------------------
``reflex.select._render_variant`` is the SHIP path's renderer: it is what a
trained checkpoint will actually be fed if ``select.head_context`` is ever
activated. A probe that renders differently measures a representation the system
cannot deploy.

So the contract is not prose. It is: for every ``RenderSpec`` in
:data:`CONFORMANCE_SPECS` and every synthetic context in
:func:`synthetic_contexts`, with ``shuffle=None``::

    featurizer.render_context(ctx, spec) == reflex.select._render_variant(
        ctx, reflex.select._HeadContext(k=spec.k,
                                        tag_recency=spec.tag_recency,
                                        recency_buckets=spec.recency_buckets,
                                        recency_mode=spec.recency_mode),
        None)

:func:`conformance_report` checks exactly that and needs no ABCD data. Run it
first; a failure here is a wiring bug, not a measurement.

SEMANTICS THE FEATURIZER MUST HONOUR (each one is a control that would otherwise
silently measure the wrong thing)
--------------------------------------------------------------------------------
* ``k``: ``int`` = last k TURN LINES, ``"full"`` = all of them, ``"inherit"`` =
  return ``ContextWindow.text`` unchanged. The STATE LINE is not a turn and is
  never windowed away -- ``reflex.data.build_context`` puts it last on purpose
  (D13: left truncation must keep it).
* ``recency_buckets`` is MAXB, and the bucket of the i-th of n chosen lines is
  ``min(n - 1 - i, MAXB)``. So ``MAXB=3`` yields r0..r3, FOUR buckets. D7's
  table is indexed by MAXB, and its default is MAXB=3.
* ``recency_mode``: ``"word"`` tags every token (what D7 measured on TF-IDF);
  ``"turn"`` tags the line once. The speaker prefix is emitted once, outside the
  tagging, in both modes.
* ``shuffle="within_window"``: permute the chosen turn lines AFTER windowing and
  BEFORE tagging, holding the state line fixed. This isolates ORDER. Permuting
  before windowing would change WHICH turns are visible as well as their order,
  which is a confounded control and must not be reported as one.
* ``shuffle`` must be deterministic given ``(shuffle_seed, context.convo_id,
  context.turn_index)``, so the same row shuffles identically in the fit pass and
  the score pass of the ``shuf->shuf`` arm.
* Rendering must be a pure function of ``(ContextWindow, RenderSpec)``. No global
  state, no fitted state, no ordering dependence.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, asdict
from typing import Any, Optional, Protocol, Sequence, runtime_checkable


# --------------------------------------------------------------------------- #
# The spec object passed to the featurizer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RenderSpec:
    """One rendering configuration. Mirrors ``reflex.select._HeadContext`` plus
    the two fields the D7 order control needs.

    Attributes:
        k: ``int`` | ``"full"`` | ``"inherit"``.
        tag_recency: Whether to prefix recency buckets.
        recency_buckets: MAXB. Bucket of the i-th of n lines is
            ``min(n - 1 - i, MAXB)``, so MAXB=3 means r0..r3.
        recency_mode: ``"word"`` | ``"turn"``.
        shuffle: ``None`` | ``"within_window"`` | ``"pre_window"``.
            ``"pre_window"`` exists only so the confound can be MEASURED; it is
            never a headline control.
        shuffle_seed: Seed for the permutation. Ignored when ``shuffle`` is None.
    """

    k: Any = "inherit"
    tag_recency: bool = False
    recency_buckets: int = 0
    recency_mode: str = "word"
    shuffle: Optional[str] = None
    shuffle_seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        """Short human label, e.g. ``k6/tag3/word/shuf``."""
        window = "full" if self.k == "full" else ("inh" if self.k == "inherit" else f"k{self.k}")
        tag = f"tag{self.recency_buckets}/{self.recency_mode}" if self.tag_recency else "plain"
        order = "" if self.shuffle is None else f"/{self.shuffle}"
        return f"{window}/{tag}{order}"


@dataclass(frozen=True)
class VectorizerSpec:
    """TF-IDF configuration, held IDENTICAL across the arms of one control.

    Defaults name the representation D7 describes ("the same word 1-2gram TF-IDF
    + logreg, same C"). The exact ``min_df`` / ``sublinear_tf`` / ``C`` of the
    original run are NOT recorded anywhere in this repo -- the harness that
    produced digest ``08ad053a37a67cc36aeec16e0a02306c`` is not in the tree. They
    are therefore knobs, and ``run_response_probe validate`` sweeps them to find
    the setting that reproduces D7's cell. Do not treat these defaults as the
    original configuration; they are a starting point.
    """

    analyzer: str = "word"
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 1.0
    sublinear_tf: bool = True
    lowercase: bool = True
    max_features: Optional[int] = None
    # classifier, carried here so one object fingerprints the whole representation
    #
    # ``classifier`` is part of the representation's identity, not a detail: at
    # H7's label sizes the SOLVER decides whether the fit is possible at all
    # (lbfgs keeps a 21-vector dense history; one-vs-rest kinds keep none), so a
    # number is only quotable with the kind that produced it (D6). Kinds:
    # ``logreg`` (lbfgs, multinomial -- the certified nextstep path),
    # ``logreg_ovr`` (exact, one-vs-rest), ``sgd_log`` (SGD, one-vs-rest).
    classifier: str = "logreg"
    C: float = 1.0
    max_iter: int = 1000
    class_weight: Optional[str] = None
    #: SGD's regularization strength. ``None`` derives it from ``C`` and the fit
    #: size as ``1 / (C * n_fit)``, the standard correspondence. Ignored by the
    #: exact solvers, which take ``C`` directly.
    alpha: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ngram_range"] = list(self.ngram_range)
        return d


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class Featurizer(Protocol):
    """What ``probes`` calls. Implemented by the featurizer agent."""

    def render_context(self, context: Any, spec: RenderSpec) -> str:
        """Render one :class:`reflex.schemas.ContextWindow` under ``spec``.

        Must equal ``reflex.select._render_variant`` for every spec with
        ``shuffle=None``; see :func:`conformance_report`.
        """
        ...

    def make_vectorizer(self, spec: VectorizerSpec) -> Any:
        """Return an UNFITTED sklearn-compatible vectorizer.

        Must expose ``fit_transform(texts)`` / ``transform(texts)`` and be
        deterministic. The harness fits it on TRAIN rows only and transforms dev
        rows with the fitted object; it never refits on dev.
        """
        ...

    def fingerprint(self) -> dict[str, Any]:
        """Version/configuration identity of the featurizer.

        Goes into the run digest. Two probe results are only comparable when
        their fingerprints match -- D6's rule that a baseline is only a baseline
        with its configuration attached, made mechanical.
        """
        ...


#: The specs :func:`conformance_report` checks. They cover: the inherit path, the
#: two windows D2b measured, both tagging modes, and the MAXB boundary where
#: bucketing saturates (D7: "k6 at MAXB 6/8/12 is byte-identical, because a
#: 6-turn window only has 6 positions" -- that identity is itself a check).
CONFORMANCE_SPECS: tuple[RenderSpec, ...] = (
    RenderSpec(k="inherit"),
    RenderSpec(k="full"),
    RenderSpec(k=6),
    RenderSpec(k=4),
    RenderSpec(k=0),
    RenderSpec(k=6, tag_recency=True, recency_buckets=3, recency_mode="word"),
    RenderSpec(k=6, tag_recency=True, recency_buckets=1, recency_mode="word"),
    RenderSpec(k=6, tag_recency=True, recency_buckets=8, recency_mode="word"),
    RenderSpec(k=6, tag_recency=True, recency_buckets=12, recency_mode="word"),
    RenderSpec(k=6, tag_recency=True, recency_buckets=3, recency_mode="turn"),
    RenderSpec(k="full", tag_recency=True, recency_buckets=3, recency_mode="word"),
    RenderSpec(k="full", tag_recency=True, recency_buckets=0, recency_mode="word"),
)


def load_featurizer(dotted: str, **kwargs: Any) -> Featurizer:
    """Import and construct the featurizer named by ``probe.featurizer_factory``.

    ``dotted`` is ``"package.module:callable"``. The callable is invoked with
    ``**kwargs`` and must return something satisfying :class:`Featurizer`.

    Raises:
        RuntimeError: with an explicit message when the featurizer is absent or
            does not satisfy the protocol. This harness deliberately has NO
            fallback renderer: see the module docstring.
    """
    if ":" not in dotted:
        raise RuntimeError(
            f"probe.featurizer_factory must be 'module.path:callable', got {dotted!r}"
        )
    module_name, _, attr = dotted.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"the recency-tagging featurizer is not wired: cannot import {module_name!r} "
            f"({exc}). probes/ deliberately ships no fallback renderer -- a probe measured "
            f"on a rendering reflex.select cannot reproduce is not transferable to the ship "
            f"path. Point probe.featurizer_factory at the featurizer agent's module."
        ) from exc
    try:
        factory = getattr(module, attr)
    except AttributeError as exc:
        raise RuntimeError(
            f"{module_name!r} has no attribute {attr!r}; probe.featurizer_factory is stale."
        ) from exc
    obj = factory(**kwargs)
    missing = [
        name for name in ("render_context", "make_vectorizer", "fingerprint")
        if not callable(getattr(obj, name, None))
    ]
    if missing:
        raise RuntimeError(
            f"{dotted} returned {type(obj).__name__}, which is missing {missing}. "
            f"See probes.featurizer_api.Featurizer for the required interface."
        )
    return obj  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Conformance: runs with NO corpus on disk
# --------------------------------------------------------------------------- #


def synthetic_contexts() -> list[Any]:
    """A handful of :class:`ContextWindow` objects covering the awkward shapes.

    Built by hand rather than loaded, so conformance can be checked while the
    corpus is being re-downloaded. Covers: a conversation opener (no prior
    turns), a thread shorter than k, a thread longer than k, a three-way thread
    with an ``action`` turn (22.5% of predicted turns follow one -- D2), and a
    turn line containing the ``|`` separator in its body.
    """
    from reflex.schemas import ContextWindow

    def _make(convo_id: int, turn_index: int, turns: list[str], state: str) -> Any:
        text = "\n".join(turns + [state])
        return ContextWindow(
            convo_id=convo_id,
            turn_index=turn_index,
            text=text,
            turns=list(turns),
            disclosed={},
            actions_so_far=[],
            context_hash="",
        )

    state = "state|disclosed: order.order_id=3348917502 || actions: pull-up-account(crystal minh)"
    return [
        _make(1, 0, [], "state|disclosed:  || actions: "),
        _make(2, 1, ["agent|hello, how can i help you today?"], state),
        _make(
            3, 4,
            [
                "agent|hello, how can i help you today?",
                "customer|i need to return an item",
                "agent|sure, can i have your account id?",
                "customer|my account id is <account_id>",
            ],
            state,
        ),
        _make(
            4, 9,
            [
                "agent|hi! how may i help you?",
                "customer|i want a refund",
                "agent|what is your username?",
                "customer|it is <username>",
                "action|pull-up-account",
                "agent|thank you.",
                "customer|ok",
                "action|validate-purchase",
                "agent|your refund is processing.",
            ],
            state,
        ),
        _make(5, 2, ["customer|the price is 50|60 dollars", "agent|let me check."], state),
    ]


def conformance_report(featurizer: Featurizer) -> dict[str, Any]:
    """Check the featurizer against ``reflex.select._render_variant``.

    Returns a dict with ``ok`` and, on failure, the first few disagreements with
    both renderings verbatim. Needs no ABCD data.

    Also checks the two invariants the D7 record states about the bucketing,
    because they are cheap and they catch an off-by-one that would otherwise
    only show up as an unreproducible accuracy:

    * k6 at MAXB 6 / 8 / 12 renders IDENTICALLY (a 6-turn window has only 6
      positions), and
    * ``tag_recency=True, recency_buckets=0`` collapses every line to bucket r0,
      which is the "one bit of recency is most of the effect" arm only if it is
      NOT identical to the untagged rendering.
    """
    from reflex import select as _select

    contexts = synthetic_contexts()
    failures: list[dict[str, Any]] = []
    checked = 0
    for spec in CONFORMANCE_SPECS:
        ship_spec = _select._HeadContext(
            k=spec.k,
            tag_recency=spec.tag_recency,
            recency_buckets=spec.recency_buckets,
            recency_mode=spec.recency_mode,
        )
        for ctx in contexts:
            want = _select._render_variant(ctx, ship_spec, None)
            got = featurizer.render_context(ctx, spec)
            checked += 1
            if got != want:
                if len(failures) < 5:
                    failures.append(
                        {"spec": spec.label(), "convo_id": ctx.convo_id,
                         "select": want, "featurizer": got}
                    )

    notes: list[str] = []
    long_ctx = contexts[3]
    saturate = {
        b: featurizer.render_context(
            long_ctx, RenderSpec(k=6, tag_recency=True, recency_buckets=b)
        )
        for b in (6, 8, 12)
    }
    if len({v for v in saturate.values()}) != 1:
        notes.append(
            "MAXB 6/8/12 at k6 do NOT render identically. A 6-turn window has only 6 "
            "positions, so they must. The bucket formula is min(n-1-i, MAXB); check for "
            "an off-by-one."
        )
    plain = featurizer.render_context(long_ctx, RenderSpec(k=6))
    zero = featurizer.render_context(long_ctx, RenderSpec(k=6, tag_recency=True, recency_buckets=0))
    if plain == zero:
        notes.append(
            "recency_buckets=0 renders identically to untagged. It must not: every line "
            "gets bucket r0, which changes the vocabulary even though it carries no order "
            "information. That arm is the 'tagging with zero order' null and it is needed "
            "to separate the vocabulary effect from the order effect."
        )

    # Determinism of the shuffle, which the shuf->shuf arm depends on.
    a = featurizer.render_context(
        long_ctx, RenderSpec(k=6, shuffle="within_window", shuffle_seed=17)
    )
    b = featurizer.render_context(
        long_ctx, RenderSpec(k=6, shuffle="within_window", shuffle_seed=17)
    )
    if a != b:
        notes.append(
            "shuffle is not deterministic for a fixed (seed, convo_id, turn_index). The "
            "shuf->shuf control refits on the shuffled train rows and scores the shuffled "
            "dev rows; if a row shuffles differently between the two passes the control "
            "measures noise, not order."
        )
    c = featurizer.render_context(
        long_ctx, RenderSpec(k=6, shuffle="within_window", shuffle_seed=18)
    )
    if a == c:
        notes.append("shuffle ignores shuffle_seed; two seeds produced the same rendering.")

    return {
        "ok": not failures and not notes,
        "checks_run": checked,
        "n_failures": len(failures),
        "failures": failures,
        "notes": notes,
        "featurizer_fingerprint": featurizer.fingerprint(),
    }
