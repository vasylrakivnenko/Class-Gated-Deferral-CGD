"""Arm B0 -- the cheap baseline the encoder must beat (DECISIONS D4 / D6 / D7).

Three independent per-head classifiers, each a recency-tagged (or deliberately
plain) word 1-2gram TF-IDF from :mod:`reflex.featurize` plus a logistic
regression. No encoder, no pretraining, no attention, ~0.3 ms/turn.

WHY THIS IS AN ARM AND NOT A FOOTNOTE
-------------------------------------
D4: TF-IDF + logreg runs BEFORE the encoder, and the 149M-parameter encoder must
beat it by a margin worth having. D6 then found the intent bar briefed to the
architecture probe (0.6141) had been measured at K=6 only and was 0.8074 at full
thread -- an arm reporting 0.72 had been "beating a baseline crippled by its
context window". D7 finished the job: recency-tagged TF-IDF beats every neural
arm tried, including a 33.7M-param frozen-encoder MLP, at ~1/50th the latency.

So this module exists to make the bar impossible to misquote. Every number it
produces comes out of :meth:`ArmB0.evaluate` bolted to the full configuration
that produced it, beside the label-blind constant for the same rows (D5's
constant-predictor guard) and, when a config is supplied, beside the external
probe bars already recorded in ``report.probe_cheap_bars`` /
``report.probe_constant_bars``.

THE PER-HEAD PLAN, AND THE ONE PLACE IT DISAGREES WITH select.py
----------------------------------------------------------------
Each head runs at its own measured winning config, and two of the three axes
point in opposite directions:

* ``nextstep`` -- k6 window, recency-tagged MAXB=3. 0.8345-0.8351, verified
  twice. Local and ordered.
* ``action``   -- FULL window, recency-tagged MAXB=3. 0.8136. That is the
  highest action number in the record; **tagged + k6 on action was never
  measured**. select.py's ``head_context.plan`` gives action ``K: 6`` on the
  strength of the D2b plain-TF-IDF row (k6 0.8086 vs full 0.8013) and of D7's
  prose ("nextstep, action: k6 window, recency-tagged"), but the only MEASURED
  tagged-action cell is at the full window. This module ships the measured cell
  and flags the gap; closing it is one probe run (see the report).
* ``intent``   -- full thread, PLAIN/untagged. 0.8074. Tagging HURTS it
  (0.7981), and its label-blind constant is 0.0226, not 0.7227. Global and
  order-invariant.

Structured features are deliberately absent: prev-speaker + n_prior_turns add
+0.26 / +0.05 on top of tagging, both inside the 0.68-point CI. D7: "tagging
subsumes prev-speaker."

THE CAVEAT SHIPS HERE TOO
-------------------------
D7, at the same prominence as the result: this does NOT show that encoders are
useless on ABCD. The encoder arms lost because a frozen mean-pooled whole-context
vector DESTROYS recency while also diluting the global signal -- a representation
failure, not evidence against pretraining. A sequence model given the same
recency bias and trained to convergence has not been fairly tested. Anyone
reading "cheap wins" out of Arm B0 is over-reading it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from reflex.config import get_dotted
from reflex.contracts import ContractViolation
from reflex.featurize import (
    ContextLike,
    FeaturizeConfig,
    RecencySpec,
    RecencyTfidfFeaturizer,
    WindowSpec,
)
from reflex.schemas import NormalizedTurn

__all__ = [
    "B0_HEADS",
    "NO_ACTION",
    "CONFIG_KEYS",
    "ClassifierSpec",
    "HeadSpec",
    "RowSpec",
    "B0Head",
    "ArmB0",
    "default_plan",
    "head_label",
    "agent_rows",
]

#: Every dotted config key :mod:`reflex.arm_b0` and :mod:`reflex.featurize` read
#: out of the ``baseline_b0`` block, relative to that block.
#:
#: D18 records three silent YAML failures in one hour and asks for "a test that
#: every key the code READS actually EXISTS -- twenty such keys were missing
#: here, each a latent runtime KeyError in a module that had been called
#: 'finished' because it imported cleanly". This list is what makes that test
#: possible in BOTH directions: no key the code reads is missing from
#: ``PROPOSED_CONFIG_B0.yaml``, and no key in that file is dead weight.
CONFIG_KEYS: tuple[str, ...] = (
    "plan.nextstep.K",
    "plan.nextstep.tag_recency",
    "plan.nextstep.recency_buckets",
    "plan.intent.K",
    "plan.intent.tag_recency",
    "plan.intent.recency_buckets",
    "plan.action.K",
    "plan.action.tag_recency",
    "plan.action.recency_buckets",
    "defaults.recency_mode",
    "defaults.include_state",
    "defaults.include_speaker",
    "defaults.tag_speaker",
    "defaults.lowercase",
    "tfidf.ngram_range",
    "tfidf.min_df",
    "tfidf.max_df",
    "tfidf.max_features",
    "tfidf.sublinear_tf",
    "tfidf.use_idf",
    "tfidf.norm",
    "classifier.C",
    "classifier.solver",
    "classifier.max_iter",
    "classifier.tol",
    "classifier.class_weight",
    "classifier.random_state",
    "classifier.ovr",
    "rows.include_synthetic_end",
    "rows.action_null_class",
)

#: The three heads with a measured winning config. skeleton (H5) and template
#: (H7) are UNMEASURED on both the window axis and the order axis; D2b and D7
#: both say measure before choosing and do not inherit a default, so they are
#: absent here rather than quietly given nextstep's plan.
B0_HEADS: tuple[str, ...] = ("nextstep", "intent", "action")

#: Label for "this turn takes no action". The action head is scored over the
#: SAME rows as nextstep with this null class, not only over take_action turns
#: -- see :func:`head_label` for the evidence.
NO_ACTION = "__none__"


# --------------------------------------------------------------------------- #
# Rows and labels
# --------------------------------------------------------------------------- #


def agent_rows(
    partition: Mapping[int, Sequence[NormalizedTurn]],
    *,
    include_synthetic_end: bool = False,
) -> list[tuple[int, int, NormalizedTurn]]:
    """Agent-side rows for Arm B0, in ``iter_agent_turns`` order.

    ``include_synthetic_end`` defaults to FALSE, and that default is DERIVED,
    not transcribed. The record reports the same label-blind constant, 0.7227,
    for both nextstep and action. ABCD's nextstep is 1:1 with the speaker, so
    "retrieve_utterance" and "action is None" are the same set of turns and the
    two constants coincide exactly -- but only while the synthetic
    end_conversation turn is absent. Including it adds one row per conversation
    that is end_conversation (lowering the nextstep constant) AND action-None
    (raising the action constant), so the two could no longer be equal. Their
    equality is therefore evidence that the probe's rows exclude it.

    That is an inference from two numbers, not something this agent verified
    against the harness, and it is the kind of thing D6 says must travel with
    its reasoning. Flip the flag to test it.
    """
    from reflex.data import iter_agent_turns  # local import: data.py pulls the corpus loaders

    rows = []
    for convo_id, turn_index, turn in iter_agent_turns(dict(partition)):
        if not include_synthetic_end and getattr(turn, "is_synthetic_end", False):
            continue
        rows.append((convo_id, turn_index, turn))
    return rows


@dataclass(frozen=True)
class RowSpec:
    """Which rows Arm B0 is fitted and scored on, and how a null action is named.

    Both defaults are DERIVED from the record rather than transcribed from it;
    :func:`agent_rows` carries the derivation.
    """

    include_synthetic_end: bool = False
    action_null_class: str = "__none__"

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any], *, block: str = "baseline_b0") -> "RowSpec":
        base = cls()
        rows: Any = {}
        block_cfg = cfg.get(block) if isinstance(cfg, Mapping) else None
        if isinstance(block_cfg, Mapping) and isinstance(block_cfg.get("rows"), Mapping):
            rows = block_cfg["rows"]
        return cls(
            include_synthetic_end=bool(rows.get("include_synthetic_end", base.include_synthetic_end)),
            action_null_class=str(rows.get("action_null_class", base.action_null_class)),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "include_synthetic_end": bool(self.include_synthetic_end),
            "action_null_class": self.action_null_class,
        }


def head_label(turn: NormalizedTurn, head: str, *, null_action: str = NO_ACTION) -> Optional[str]:
    """The gold label one head predicts for one turn.

    ``nextstep`` and ``intent`` read the field of the same name. ``action``
    reads :attr:`NormalizedTurn.action`, mapping ``None`` to :data:`NO_ACTION`
    rather than dropping the row -- see :func:`agent_rows` for why the action
    head is scored over every agent-side row.

    Returns ``None`` only when the field itself is missing, and
    :meth:`ArmB0.fit` drops those rows for that head alone.
    """
    if head == "nextstep":
        return turn.nextstep
    if head == "intent":
        return turn.intent
    if head == "action":
        return turn.action if turn.action is not None else null_action
    raise ContractViolation(f"Arm B0 covers {B0_HEADS}; no head {head!r}")


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClassifierSpec:
    """Logistic regression knobs.

    D7 says the tagged and plain arms were run with the "same C" and never names
    it, so ``C`` is sklearn's default and is a config key, not a literal. The
    solver is likewise unpinned, so it too is sklearn's default (``lbfgs``,
    multinomial and deterministic).

    TWO THINGS MEASURED HERE, BOTH ABOUT THE 55-WAY INTENT HEAD
    -----------------------------------------------------------
    1. ``liblinear`` -- the obvious choice for sparse text, and what a 2020-era
       "TF-IDF + logreg" almost always meant -- **can no longer fit a multiclass
       problem at all** on the sklearn installed here (1.9.0):
       ``ValueError: The 'liblinear' solver does not support multiclass
       classification (n_classes >= 3)``. Verified by running it. If the frozen
       probe harness used liblinear on an older sklearn, reproducing its numbers
       needs ``ovr: true`` below, not the same solver string.
    2. ``lbfgs`` is multinomial and holds DENSE ``n_classes x n_features``
       state. At D7's tagged vocabulary of 148k features and 55 intents that is
       ~65 MB per vector, and L-BFGS keeps roughly twenty of them: order 1 GB
       for the intent head alone, before the data matrix. On a machine that OOM'd
       tonight that is worth knowing before the first full-corpus fit.

    ``ovr`` is the lean answer to both: one binary liblinear fit at a time,
    ~1.2 MB of coefficients each, deterministic, and the historically standard
    TF-IDF text baseline. It is OFF by default only because the record does not
    pin it and sklearn's default is multinomial.

    ``max_iter`` is raised above sklearn's default on purpose. D5: a fine-tuned
    encoder once scored BELOW its own frozen version purely because it failed to
    converge, so "always check the training loss before attributing a result to
    architecture". :meth:`ArmB0.fit` records convergence per head and
    :meth:`ArmB0.describe` reports it.
    """

    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[str] = None
    tol: float = 1e-4
    random_state: int = 0
    ovr: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "C": float(self.C),
            "solver": self.solver,
            "max_iter": int(self.max_iter),
            "class_weight": self.class_weight,
            "tol": float(self.tol),
            "random_state": int(self.random_state),
            "ovr": bool(self.ovr),
        }


@dataclass(frozen=True)
class HeadSpec:
    """One head's complete plan: how to featurize and how to classify."""

    head: str
    featurize: FeaturizeConfig
    classifier: ClassifierSpec = field(default_factory=ClassifierSpec)
    #: Free-text provenance -- which DECISIONS row this plan implements.
    provenance: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "featurize": self.featurize.describe(),
            "classifier": self.classifier.describe(),
            "provenance": self.provenance,
        }


def default_plan() -> dict[str, HeadSpec]:
    """The measured winning config per head, as library defaults.

    Spec 10 forbids thresholds in Python source, and the RUNNABLE path is
    :meth:`ArmB0.from_cfg`, which reads ``baseline_b0.plan`` from config. This
    function exists so the module is usable and testable standalone and so the
    defaults are visible next to their provenance; the two must agree, and
    ``tests/test_arm_b0.py`` asserts they do against ``PROPOSED_CONFIG_B0.yaml``.
    """
    tagged = RecencySpec(enabled=True, max_bucket=3, mode="word")
    plain = RecencySpec(enabled=False, max_bucket=3, mode="word")
    return {
        "nextstep": HeadSpec(
            head="nextstep",
            featurize=FeaturizeConfig(window=WindowSpec.last_k(6), recency=tagged),
            provenance="D7 'the ship configuration, verified twice': tagged k6 r0..r3, 0.8345-0.8351",
        ),
        "intent": HeadSpec(
            head="intent",
            featurize=FeaturizeConfig(window=WindowSpec.full(), recency=plain),
            provenance="D2b/D6/D7: full thread, UNTAGGED (tagging hurts: 0.7981 vs 0.8078), 0.8074",
        ),
        "action": HeadSpec(
            head="action",
            featurize=FeaturizeConfig(window=WindowSpec.full(), recency=tagged),
            provenance=(
                "D7 continued: tagged FULL window, 0.8136 -- the only measured tagged-action "
                "cell. Tagged+k6 on action is UNMEASURED; select.head_context.plan assumes K=6"
            ),
        ),
    }


# --------------------------------------------------------------------------- #
# One head
# --------------------------------------------------------------------------- #


class B0Head:
    """One head: featurizer + logreg + the label-blind constant for its rows."""

    def __init__(self, spec: HeadSpec) -> None:
        self.spec = spec
        self.featurizer = RecencyTfidfFeaturizer(spec.featurize)
        self._clf: Any = None
        self._classes: list[str] = []
        self._constant_label: Optional[str] = None
        self._converged: Optional[bool] = None
        self._n_train: int = 0

    # -- fit ---------------------------------------------------------------- #

    def fit(self, contexts: Sequence[ContextLike], labels: Sequence[Optional[str]]) -> "B0Head":
        if len(contexts) != len(labels):
            raise ContractViolation(
                f"{self.spec.head}: {len(contexts)} contexts but {len(labels)} labels"
            )
        keep = [i for i, y in enumerate(labels) if y is not None]
        if not keep:
            raise ContractViolation(f"{self.spec.head}: no rows carry a label")
        docs = self.featurizer.build_documents([contexts[i] for i in keep])
        y = [str(labels[i]) for i in keep]
        self._n_train = len(y)
        # Deterministic under ties: Counter.most_common resolves a tie by
        # insertion order, which would make the constant bar depend on row order.
        counts = Counter(y)
        self._constant_label = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        x = self.featurizer.fit_transform(docs)
        if len(set(y)) < 2:
            # A single-class head cannot be fitted by logreg; the constant IS the
            # model. Recorded rather than crashed so a tiny fixture still runs.
            self._clf = None
            self._classes = sorted(set(y))
            self._converged = True
            return self

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression

        import warnings as _warnings

        spec = self.spec.classifier
        n_classes = len(set(y))
        base = LogisticRegression(
            C=spec.C,
            solver=spec.solver,
            max_iter=spec.max_iter,
            class_weight=spec.class_weight,
            tol=spec.tol,
            random_state=spec.random_state,
        )
        if spec.ovr:
            from sklearn.multiclass import OneVsRestClassifier

            self._clf = OneVsRestClassifier(base)
        else:
            if spec.solver == "liblinear" and n_classes > 2:
                # sklearn >= 1.9 refuses this outright. Fail with the fix named.
                raise ContractViolation(
                    f"{self.spec.head}: solver 'liblinear' cannot fit {n_classes} classes on "
                    "this sklearn (>= 1.9 dropped its built-in one-vs-rest). Set "
                    "baseline_b0.classifier.ovr=true to keep liblinear one-vs-rest, or use "
                    "the lbfgs/saga solver."
                )
            self._clf = base
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            self._clf.fit(x, y)
        self._converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        self._classes = [str(c) for c in self._clf.classes_]
        return self

    # -- predict ------------------------------------------------------------ #

    def _require_fitted(self) -> None:
        if self._constant_label is None:
            raise ContractViolation(f"{self.spec.head}: not fitted")

    def predict(self, contexts: Sequence[ContextLike]) -> list[str]:
        self._require_fitted()
        if self._clf is None:
            return [str(self._constant_label)] * len(contexts)
        docs = self.featurizer.build_documents(contexts)
        return [str(p) for p in self._clf.predict(self.featurizer.transform(docs))]

    def predict_proba(self, contexts: Sequence[ContextLike]) -> list[list[float]]:
        """Probabilities over :attr:`classes`, one row per context."""
        self._require_fitted()
        if self._clf is None:
            return [[1.0] for _ in contexts]
        docs = self.featurizer.build_documents(contexts)
        return [[float(v) for v in row] for row in self._clf.predict_proba(self.featurizer.transform(docs))]

    def probabilities_in_order(
        self, contexts: Sequence[ContextLike], class_order: Sequence[str]
    ) -> list[list[float]]:
        """Probabilities re-indexed onto a CANONICAL class order.

        :class:`reflex.schemas.SelectorScores` requires each vector to sum to 1
        over the head's full class list in its canonical order
        (``NEXT_STEPS`` for H1, ``data.subflow_list(ontology)`` for H2,
        ``data.action_list(ontology)`` for H3). A class the training split never
        saw gets 0.0, which keeps the sum at 1. A class the model HAS but the
        canonical order lacks is a contract violation, not a silent drop.
        """
        order = list(class_order)
        index = {name: i for i, name in enumerate(order)}
        missing = [c for c in self.classes if c not in index and c != NO_ACTION]
        if missing:
            raise ContractViolation(
                f"{self.spec.head}: fitted classes {missing} are absent from the canonical "
                f"class order ({len(order)} entries); the score vector could not sum to 1"
            )
        out: list[list[float]] = []
        for row in self.predict_proba(contexts):
            dense = [0.0] * len(order)
            for prob, name in zip(row, self.classes):
                if name in index:
                    dense[index[name]] = float(prob)
            out.append(dense)
        return out

    # -- introspection ------------------------------------------------------ #

    @property
    def classes(self) -> list[str]:
        self._require_fitted()
        return list(self._classes) if self._classes else [str(self._constant_label)]

    @property
    def constant_label(self) -> str:
        """The label-blind constant for this head: the training majority class."""
        self._require_fitted()
        return str(self._constant_label)

    @property
    def converged(self) -> Optional[bool]:
        return self._converged

    def describe(self) -> dict[str, Any]:
        out = self.spec.describe()
        out["fitted"] = self._constant_label is not None
        if self._constant_label is not None:
            out["n_train_rows"] = self._n_train
            out["n_classes"] = len(self.classes)
            out["constant_label"] = self.constant_label
            out["converged"] = self._converged
            out["degenerate_single_class"] = self._clf is None
        if self.featurizer.is_fitted:
            out["n_features"] = self.featurizer.n_features
        return out


# --------------------------------------------------------------------------- #
# The arm
# --------------------------------------------------------------------------- #


def _accuracy(pred: Sequence[str], gold: Sequence[Optional[str]]) -> tuple[float, int]:
    keep = [(p, g) for p, g in zip(pred, gold) if g is not None]
    if not keep:
        return float("nan"), 0
    hits = sum(1 for p, g in keep if p == g)
    return hits / len(keep), len(keep)


class ArmB0:
    """The three-head cheap baseline.

    Fit and evaluate::

        arm = ArmB0.from_cfg(cfg)                 # or ArmB0() for library defaults
        arm.fit(train_contexts, train_labels)     # labels: {head: [label per context]}
        result = arm.evaluate(dev_contexts, dev_labels, cfg=cfg)

    :meth:`evaluate` never returns a bare number: every head's accuracy comes
    back with the featurizer and classifier configuration that produced it, the
    label-blind constant for the same rows, and -- when ``cfg`` is given -- the
    external probe bars from ``report.probe_cheap_bars`` and
    ``report.probe_constant_bars``, labelled as external and NOT as a result of
    this run.
    """

    def __init__(
        self,
        plan: Optional[Mapping[str, HeadSpec]] = None,
        rows: Optional[RowSpec] = None,
    ) -> None:
        self.rows: RowSpec = rows or RowSpec()
        self.plan: dict[str, HeadSpec] = dict(plan or default_plan())
        unknown = set(self.plan) - set(B0_HEADS)
        if unknown:
            raise ContractViolation(
                f"Arm B0 covers {B0_HEADS}; skeleton and template are UNMEASURED on both the "
                f"window and the order axis and must be measured before being given a plan. Got {sorted(unknown)}"
            )
        self.heads: dict[str, B0Head] = {name: B0Head(spec) for name, spec in self.plan.items()}

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any], *, block: str = "baseline_b0") -> "ArmB0":
        """Build from config. See :meth:`FeaturizeConfig.from_cfg` for the block shape."""
        try:
            plan_raw = get_dotted(dict(cfg), f"{block}.plan")
        except KeyError as exc:
            raise ContractViolation(
                f"config block {block!r} is missing. Merge PROPOSED_CONFIG_B0.yaml into "
                "configs/default.yaml (coordinator-owned) before running Arm B0."
            ) from exc
        clf_raw: Mapping[str, Any] = {}
        block_cfg = cfg.get(block) if isinstance(cfg, Mapping) else None
        if isinstance(block_cfg, Mapping) and isinstance(block_cfg.get("classifier"), Mapping):
            clf_raw = block_cfg["classifier"]
        base = ClassifierSpec()
        shared_clf = ClassifierSpec(
            C=float(clf_raw.get("C", base.C)),
            solver=str(clf_raw.get("solver", base.solver)),
            max_iter=int(clf_raw.get("max_iter", base.max_iter)),
            class_weight=clf_raw.get("class_weight", base.class_weight),
            tol=float(clf_raw.get("tol", base.tol)),
            random_state=int(clf_raw.get("random_state", base.random_state)),
            ovr=bool(clf_raw.get("ovr", base.ovr)),
        )
        defaults = default_plan()
        plan: dict[str, HeadSpec] = {}
        for head in plan_raw:
            plan[head] = HeadSpec(
                head=head,
                featurize=FeaturizeConfig.from_cfg(cfg, head, block=block),
                classifier=shared_clf,
                provenance=defaults[head].provenance if head in defaults else "",
            )
        return cls(plan, rows=RowSpec.from_cfg(cfg, block=block))

    # -- fit / predict ------------------------------------------------------ #

    def fit(
        self,
        contexts: Sequence[ContextLike],
        labels: Mapping[str, Sequence[Optional[str]]],
    ) -> "ArmB0":
        missing = set(self.plan) - set(labels)
        if missing:
            raise ContractViolation(f"no training labels for head(s) {sorted(missing)}")
        for head, model in self.heads.items():
            model.fit(contexts, labels[head])
        return self

    def fit_rows(
        self,
        contexts: Sequence[ContextLike],
        turns: Sequence[NormalizedTurn],
    ) -> "ArmB0":
        """Convenience: derive every head's labels from the turns themselves."""
        if len(contexts) != len(turns):
            raise ContractViolation(f"{len(contexts)} contexts but {len(turns)} turns")
        return self.fit(contexts, self.labels_from_turns(turns))

    def labels_from_turns(self, turns: Sequence[NormalizedTurn]) -> dict[str, list[Optional[str]]]:
        return {
            head: [head_label(t, head, null_action=self.rows.action_null_class) for t in turns]
            for head in self.plan
        }

    def rows_for(
        self, partition: Mapping[int, Sequence[NormalizedTurn]]
    ) -> list[tuple[int, int, NormalizedTurn]]:
        """The agent-side rows this arm is fitted and scored on, per :class:`RowSpec`."""
        return agent_rows(partition, include_synthetic_end=self.rows.include_synthetic_end)

    def predict(self, contexts: Sequence[ContextLike]) -> dict[str, list[str]]:
        return {head: model.predict(contexts) for head, model in self.heads.items()}

    def predict_proba(self, contexts: Sequence[ContextLike]) -> dict[str, list[list[float]]]:
        return {head: model.predict_proba(contexts) for head, model in self.heads.items()}

    # -- evaluation --------------------------------------------------------- #

    def evaluate(
        self,
        contexts: Sequence[ContextLike],
        labels: Mapping[str, Sequence[Optional[str]]],
        *,
        cfg: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Accuracy per head, each bolted to its configuration and its constant.

        The shape is deliberate. D5's constant-predictor guard says every
        headline metric is reported beside what a label-blind constant scores,
        and a metric a constant wins is dropped, not caveated -- ABCD's nextstep
        is 1:1 with the speaker, so a constant scores 0.7227 and "looks
        deceptively strong". D6 says a baseline is only a baseline WITH its
        configuration attached. Neither is optional, so neither is separable
        from the number here.

        ``external_probe_bar`` / ``external_probe_constant``, when ``cfg`` is
        given, are read from the coordinator-owned ``report.probe_*_bars`` keys.
        They belong to the frozen probe harness (dev n=13,284, CI half-width
        0.68 points, digest 08ad053a37a67cc36aeec16e0a02306c), NOT to this run,
        and are labelled as external for exactly the D6 reason.
        """
        out: dict[str, Any] = {"heads": {}}
        cheap_bars: Mapping[str, Any] = {}
        const_bars: Mapping[str, Any] = {}
        if cfg is not None:
            for key, sink in (("report.probe_cheap_bars", "cheap"), ("report.probe_constant_bars", "const")):
                try:
                    value = get_dotted(dict(cfg), key)
                except KeyError:
                    value = {}
                if sink == "cheap":
                    cheap_bars = value or {}
                else:
                    const_bars = value or {}
            out["external_bars_source"] = (
                "report.probe_* -- frozen probe harness, dev n=13,284, CI half-width 0.68 pts, "
                "digest 08ad053a37a67cc36aeec16e0a02306c. EXTERNAL to this run."
            )
        for head, model in self.heads.items():
            gold = list(labels[head])
            acc, n = _accuracy(model.predict(contexts), gold)
            const_pred = [model.constant_label] * len(contexts)
            const_acc, _ = _accuracy(const_pred, gold)
            row: dict[str, Any] = {
                "accuracy": acc,
                "n_scored": n,
                "constant_accuracy": const_acc,
                "constant_label": model.constant_label,
                "beats_constant": (acc > const_acc) if n else None,
                "config": model.describe(),
            }
            if cfg is not None:
                row["external_probe_bar"] = cheap_bars.get(head)
                row["external_probe_constant"] = const_bars.get(head)
            out["heads"][head] = row
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "arm": "B0",
            "rows": self.rows.describe(),
            "heads": {head: model.describe() for head, model in self.heads.items()},
            "structured_features": (
                "none, deliberately: prev-speaker + n_prior_turns add +0.26 / +0.05 on top of "
                "recency tagging, both inside the 0.68-point CI (D7)"
            ),
            "caveat": (
                "D7: this does NOT show encoders are useless on ABCD. The encoder arms lost to a "
                "representation failure (frozen mean-pooled context destroys recency), not to "
                "pretraining. A sequence model with the same recency bias, trained to "
                "convergence, has not been fairly tested."
            ),
        }
