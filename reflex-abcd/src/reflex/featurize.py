"""Shared recency-tagging TF-IDF featurizer (DECISIONS D2b / D6 / D7).

WHAT THIS IS
------------
The cheap representation that beat every neural arm measured on ABCD: window the
prior turns, prefix every word token with the RECENCY BUCKET of the turn it came
from, then ordinary word 1-2gram TF-IDF. Same words, same model, ~0.3 ms/turn.
The only thing tagging adds is that the bag can distinguish "refund in the last
turn" from "refund six turns ago" -- so, unlike plain TF-IDF, it is NOT
permutation-invariant and a turn-order shuffle can hurt it. That is the whole
point: D7 records that two earlier order controls were incapable of failing.

This module is a LIBRARY, deliberately: :mod:`reflex.arm_b0` consumes it for the
three measured heads, and a separate probe harness consumes it for the heads
that are still unmeasured (skeleton H5, template H7 -- D2b and D7 both say
measure before choosing, do not inherit a default). Nothing here trains, scores,
reads a corpus or decides anything.

THE MEASURED RECORD THIS IMPLEMENTS (dev n=13,284, CI half-width 0.68 points,
probe digest 08ad053a37a67cc36aeec16e0a02306c)::

    head      winning config                          score    label-blind const
    nextstep  recency-tagged, k6 window, MAXB=3       0.8345-0.8351      0.7227
    action    recency-tagged, FULL window, MAXB=3     0.8136             0.7227
    intent    PLAIN (untagged), full thread           0.8074             0.0226

Granularity saturates at 3-4 buckets. MAXB=1 -- one bit, "is this the most
recent turn, yes or no" -- already buys +3.1 (full) / +2.5 (k6). Past 4 buckets
it is flat and costs vocabulary (56k -> 148k features). Hence
``RecencySpec.max_bucket`` defaults to 3 and D7's instruction: expose it as a
tunable, do not search it hard.

Structured features (prev-speaker, n_prior_turns) buy +0.26 / +0.05 on top of
tagging, both inside CI. D7: "tagging subsumes prev-speaker." This module
therefore emits NO structured features and offers no hook to add them.

WHAT IS PINNED BY MEASUREMENT AND WHAT IS NOT -- read before quoting a number
----------------------------------------------------------------------------
D6 is the local law: "a baseline is only a baseline WITH its configuration
attached." The probe harness that produced the table above is external to this
repo, so these sub-choices are RECONSTRUCTIONS, not transcriptions. Each is a
config key with a documented default, and :meth:`RecencyTfidfFeaturizer.describe`
emits all of them so a score can never travel without them:

* ``include_state`` -- default FALSE. :func:`reflex.data.build_context` appends a
  ``state|`` line summarizing the WHOLE prefix (disclosed scenario fields plus
  every action taken so far), and K truncates the turn list only. Including it
  would leak whole-thread information into every "k6" condition and so would
  dissolve the very contrast D2b measures (that the full thread COSTS nextstep
  2.1 points). The D2b/D6/D7 windows are all expressed purely as turn counts, so
  the probe was near-certainly turns-only. NOT directly verified.
* ``min_df`` / ``max_features`` / ``sublinear_tf`` -- default to sklearn's own
  defaults. The only handle the record gives on the probe's pruning is its
  vocabulary sizes (56k plain-full, 148k at MAXB=8); at ``min_df=1`` this
  featurizer will very likely exceed 56k, so reconciling ``min_df`` against that
  56k figure is the first thing to do when the harness next runs on the corpus.
  :attr:`RecencyTfidfFeaturizer.n_features` exists for exactly that check.
* ``tag_speaker`` -- default FALSE, matching the in-repo precedent in
  ``reflex.select._tag_turn`` (the speaker prefix stays untagged; only the body
  is tagged). Note the speaker is still recoverable per bucket through bigrams:
  ``"agent r0|hi"`` says the most recent turn is an agent turn.

CROSS-MODULE WARNING
--------------------
``reflex.select._tag_turn`` renders word-mode tagging as a STRING for a
transformer tokenizer: it splits the body on whitespace and joins with ``|``, so
``"hi!"`` becomes ``"r0|hi!"``. Feeding that string to a TF-IDF vectorizer with
sklearn's default ``token_pattern`` (``\\b\\w\\w+\\b``) SILENTLY SPLITS THE TAG
OFF THE TOKEN -- ``r0|refund`` becomes the two features ``r0`` and ``refund``,
and every bit of recency information is destroyed while the run still completes
and reports a plausible number.

So this module owns its tokenization end to end, on BOTH of its paths:

* :func:`build_document` (Arm B0) builds the document itself, as whitespace-
  separated pre-tagged tokens, and the vectorizer merely splits it;
* :class:`ProbeFeaturizer` (the ``probes/`` harness, which requires the ship
  rendering byte-for-byte) renders exactly as select.py does and pairs it with
  :func:`tokenize_rendered`, which parses that layout back into the same tokens.

``tests/test_featurize.py`` asserts the two paths tokenize identically over
every spec in ``probes.featurizer_api.CONFORMANCE_SPECS``, and separately
demonstrates that a stock ``TfidfVectorizer()`` over the ship rendering does
sever the tags. Never pair select.py's rendering with a default token_pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from reflex.config import get_dotted
from reflex.contracts import ContractViolation
from reflex.schemas import ContextWindow

__all__ = [
    "WORD_PATTERN",
    "WindowSpec",
    "RecencySpec",
    "TfidfSpec",
    "FeaturizeConfig",
    "RecencyTfidfFeaturizer",
    "split_context",
    "window_turns",
    "bucket_sequence",
    "line_tokens",
    "build_document",
    "build_documents",
    "tokenize_rendered",
    "ProbeFeaturizer",
    "build_featurizer",
]

#: sklearn's default ``token_pattern``. Kept identical so that a run with
#: recency OFF produces exactly the token multiset sklearn's default analyzer
#: would extract from :attr:`ContextWindow.text` -- the property
#: ``test_featurize.py::test_plain_mode_matches_sklearn_default_analyzer``
#: asserts, so "plain TF-IDF" here really is the plain TF-IDF of the record.
WORD_PATTERN = r"(?u)\b\w\w+\b"

_FULL = "full"
_RECENCY_MODES = ("word", "turn")


# --------------------------------------------------------------------------- #
# Window
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowSpec:
    """Which prior turns enter the bag. Covers every window D2b measured.

    ``first`` and ``last`` are counts of TURNS, oldest-first ordering preserved,
    never duplicated: when ``first + last`` would cover the whole thread the
    whole thread is returned contiguously.

    =============================  ``first``  ``last``  D2b row
    full thread                    ``None``   ``None``  "full thread"
    last 6 turns                   ``None``   ``6``     "k6 (last 6 turns)"
    first 2 turns only             ``2``      ``None``  "first 2 turns only"
    first 2 + last 6               ``2``      ``6``     "first2 + last6"
    =============================  =========  ========  ==================

    D2b's caveat travels with this class: first2+last6 is the least-bad single
    global window (within CI of k6 on nextstep, ties on action) but gives up 7.8
    points of intent to buy that simplicity. Intent evidence is DISTRIBUTED
    across the thread, not concentrated at either end -- first 2 turns alone
    score 0.5531 on intent, below even k6.
    """

    first: Optional[int] = None
    last: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("first", "last"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractViolation(f"WindowSpec.{name} must be an int or None, got {value!r}")
            if value < 0:
                raise ContractViolation(f"WindowSpec.{name} must be >= 0, got {value}")

    # -- constructors ------------------------------------------------------- #

    @classmethod
    def full(cls) -> "WindowSpec":
        """The whole thread so far."""
        return cls(first=None, last=None)

    @classmethod
    def last_k(cls, k: int) -> "WindowSpec":
        """The last ``k`` turns. ``WindowSpec.last_k(6)`` is D2b's "k6"."""
        return cls(first=None, last=int(k))

    @classmethod
    def first_n(cls, n: int) -> "WindowSpec":
        """The first ``n`` turns only."""
        return cls(first=int(n), last=None)

    @classmethod
    def first_last(cls, n: int, m: int) -> "WindowSpec":
        """The first ``n`` and the last ``m`` turns, no duplication."""
        return cls(first=int(n), last=int(m))

    @classmethod
    def parse(cls, raw: Any) -> "WindowSpec":
        """Read a window out of config.

        Accepts, in order of preference:

        * ``"full"`` / ``"all"`` / ``None`` -> the whole thread;
        * an int ``K`` or the string ``"k6"`` -> the last K turns (so a
          ``data.context_turns_K``-shaped value means the same thing here);
        * ``"first2+last6"`` -> both ends;
        * a mapping ``{"first": 2, "last": 6}``.

        ``True``/``False`` are rejected outright: ``bool`` is an ``int``
        subclass, and DECISIONS D18 records an hour in which a bare YAML scalar
        parsed as the wrong type and nothing raised.
        """
        if raw is None:
            return cls.full()
        if isinstance(raw, WindowSpec):
            return raw
        if isinstance(raw, bool):
            raise ContractViolation(f"window must be an int, a string or a mapping, got {raw!r}")
        if isinstance(raw, int):
            return cls.last_k(raw)
        if isinstance(raw, Mapping):
            unknown = set(raw) - {"first", "last"}
            if unknown:
                raise ContractViolation(f"unknown window keys {sorted(unknown)}; expected 'first'/'last'")
            return cls(first=raw.get("first"), last=raw.get("last"))
        text = str(raw).strip().lower()
        if text in {_FULL, "all", "none", ""}:
            return cls.full()
        parts = [p.strip() for p in text.split("+") if p.strip()]
        first: Optional[int] = None
        last: Optional[int] = None
        for part in parts:
            match = re.fullmatch(r"(first|last|k)(\d+)", part)
            if match:
                which, count = match.group(1), int(match.group(2))
                if which == "first":
                    first = count
                else:  # "last" and the shorthand "k6"
                    last = count
                continue
            if re.fullmatch(r"\d+", part):
                last = int(part)
                continue
            raise ContractViolation(
                f"could not parse window {raw!r}: expected 'full', an int, 'k6', "
                "'first2+last6' or {'first': 2, 'last': 6}"
            )
        return cls(first=first, last=last)

    # -- behaviour ---------------------------------------------------------- #

    def is_full(self) -> bool:
        return self.first is None and self.last is None

    def max_turns_needed(self) -> Optional[int]:
        """Upper bound on prior turns this window can ever read, ``None`` if unbounded."""
        if self.is_full():
            return None
        return (self.first or 0) + (self.last or 0)

    def apply(self, lines: Sequence[str]) -> list[str]:
        """Select turns from an oldest-first sequence. See :func:`window_turns`."""
        return window_turns(lines, self)

    def describe(self) -> dict[str, Any]:
        return {"first": self.first, "last": self.last, "label": str(self)}

    def __str__(self) -> str:
        if self.is_full():
            return "full"
        if self.first is None:
            return f"last{self.last}"
        if self.last is None:
            return f"first{self.first}"
        return f"first{self.first}+last{self.last}"


def window_turns(lines: Sequence[str], window: WindowSpec) -> list[str]:
    """Apply ``window`` to an OLDEST-FIRST sequence of rendered turn lines.

    No turn is ever duplicated and the oldest-first order is preserved. When
    ``first + last >= len(lines)`` the whole sequence is returned, which keeps a
    head+tail window contiguous on short threads instead of emitting the middle
    twice.
    """
    lines = list(lines)
    if window.is_full():
        return lines
    n = len(lines)
    if window.first is None:
        return lines[max(0, n - int(window.last or 0)) :] if window.last else []
    if window.last is None:
        return lines[: int(window.first)]
    if int(window.first) + int(window.last) >= n:
        return lines
    return lines[: int(window.first)] + lines[n - int(window.last) :]


# --------------------------------------------------------------------------- #
# Recency
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecencySpec:
    """How -- and whether -- tokens carry the recency of their turn.

    Attributes:
        enabled: ``False`` is plain TF-IDF, the intent head's winning config
            (0.8074 at full thread; tagging HURTS intent, 0.7981). Not a
            degraded mode: it is the measured best for a global, order-invariant
            target.
        max_bucket: MAXB. Buckets run ``r0`` (the most recent turn in the
            window) through ``r{max_bucket}``, which absorbs everything at or
            beyond that distance -- so ``max_bucket=3`` means FOUR buckets,
            ``r0..r3``, matching D7's "MAXB=3 (r0..r3)". Degenerate and
            informative endpoints, both tested:

            * ``0`` -- one bucket. Every token gets ``r0``, so the document is a
              bijective relabelling of the plain document and carries no recency
              information at all.
            * ``1`` -- two buckets, ONE BIT ("is this the most recent turn").
              D7: that single bit buys +3.1 full / +2.5 k6, most of a 12-point
              order effect.
            * ``3`` -- the default and the ship configuration. Saturation is at
              3-4; past 4 it is flat and pure vocabulary cost.
        mode: ``word`` tags every token (what D7 measured on TF-IDF); ``turn``
            emits the bucket once per line, which is what an encoder input can
            afford. Mirrors ``select.head_context.recency_mode``.
        tag_speaker: whether the speaker token is tagged too. Default ``False``,
            matching ``reflex.select._tag_turn``. Ignored in ``turn`` mode,
            where the single bucket token already precedes the body.
        sep: separator between bucket and token. Never appears inside a token
            (:data:`WORD_PATTERN` cannot match it) and the document is split on
            whitespace, so the tag cannot be severed from its token.

    A note on gapped windows: buckets are numbered WITHIN the selected window
    (``min(n - 1 - i, max_bucket)``), matching ``reflex.select._render_variant``.
    For a head+tail window the head turns are further away than their index
    suggests -- but any such window carries at least 6 tail turns, so at the
    default ``max_bucket=3`` the head block saturates at ``r3`` under either
    numbering and the distinction cannot bite. It can bite at ``max_bucket > 6``
    on a gapped window; that combination is unmeasured and undefended.
    """

    enabled: bool = True
    max_bucket: int = 3
    mode: str = "word"
    tag_speaker: bool = False
    sep: str = "|"

    def __post_init__(self) -> None:
        if isinstance(self.max_bucket, bool) or not isinstance(self.max_bucket, int):
            raise ContractViolation(f"RecencySpec.max_bucket must be an int, got {self.max_bucket!r}")
        if self.max_bucket < 0:
            raise ContractViolation(f"RecencySpec.max_bucket must be >= 0, got {self.max_bucket}")
        if self.mode not in _RECENCY_MODES:
            raise ContractViolation(
                f"RecencySpec.mode must be one of {_RECENCY_MODES}, got {self.mode!r}"
            )
        if not self.sep or any(ch.isspace() for ch in self.sep):
            raise ContractViolation(
                f"RecencySpec.sep must be non-empty and whitespace-free, got {self.sep!r}; "
                "whitespace would let the vectorizer split the bucket off its token"
            )

    def tag(self, bucket: int) -> str:
        return f"r{int(bucket)}"

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "max_bucket": int(self.max_bucket),
            "mode": self.mode,
            "tag_speaker": bool(self.tag_speaker),
            "sep": self.sep,
        }


def bucket_sequence(n_turns: int, max_bucket: int) -> list[int]:
    """Recency bucket per turn for an oldest-first window of ``n_turns``.

    The LAST element is always ``0`` (``r0`` = most recent). Distance from the
    most recent turn is capped at ``max_bucket``::

        bucket_sequence(5, 3) -> [3, 3, 2, 1, 0]
        bucket_sequence(5, 1) -> [1, 1, 1, 1, 0]      # one bit
        bucket_sequence(5, 0) -> [0, 0, 0, 0, 0]      # no recency at all

    This is also D7's own sanity check: a 6-turn window has only 6 positions, so
    MAXB 6, 8 and 12 give byte-identical output on k6.
    """
    if max_bucket < 0:
        raise ContractViolation(f"max_bucket must be >= 0, got {max_bucket}")
    return [min(n_turns - 1 - i, int(max_bucket)) for i in range(int(n_turns))]


# --------------------------------------------------------------------------- #
# TF-IDF / document assembly
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TfidfSpec:
    """Vectorizer knobs. Every default that the measured record does not pin is
    sklearn's own default, and is flagged as unpinned in the module docstring.

    ``ngram_range`` is ``(1, 2)`` because D7's whole argument rests on word
    1-2grams: its plain-TF-IDF shuffle control is quantified as "only 5.5% of
    uni+bigram mass can respond at all (the bigrams straddling turn junctions)".
    Those straddling bigrams are preserved here -- no barrier token is inserted
    between turns -- because removing them would change the representation the
    record measured.
    """

    ngram_range: tuple[int, int] = (1, 2)
    min_df: Union[int, float] = 1
    max_df: Union[int, float] = 1.0
    max_features: Optional[int] = None
    sublinear_tf: bool = False
    use_idf: bool = True
    norm: Optional[str] = "l2"

    def describe(self) -> dict[str, Any]:
        return {
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "sublinear_tf": bool(self.sublinear_tf),
            "use_idf": bool(self.use_idf),
            "norm": self.norm,
        }


@dataclass(frozen=True)
class FeaturizeConfig:
    """One head's complete, self-describing featurization configuration.

    D6 in one object: nothing here is read from a global default at call time,
    so a score computed with this config can always be reported beside it.
    """

    window: WindowSpec = field(default_factory=WindowSpec.full)
    recency: RecencySpec = field(default_factory=RecencySpec)
    tfidf: TfidfSpec = field(default_factory=TfidfSpec)
    include_state: bool = False
    include_speaker: bool = True
    lowercase: bool = True
    token_pattern: str = WORD_PATTERN

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FeaturizeConfig":
        """Build from a plain config mapping (a ``baseline_b0.plan.<head>`` block).

        Recognized keys: ``window`` (or ``K``), ``tag_recency``,
        ``recency_buckets``, ``recency_mode``, ``tag_speaker``,
        ``include_state``, ``include_speaker``, ``lowercase``, ``token_pattern``
        and a nested ``tfidf`` mapping. ``K``/``tag_recency``/``recency_buckets``
        are spelled the way ``select.head_context.plan`` spells them so the two
        plans can be diffed by eye.
        """
        known = {
            "window", "K", "tag_recency", "recency_buckets", "recency_mode", "tag_speaker",
            "include_state", "include_speaker", "lowercase", "token_pattern", "tfidf",
        }
        unknown = set(raw) - known
        if unknown:
            raise ContractViolation(
                f"unknown featurizer config keys {sorted(unknown)}; expected a subset of {sorted(known)}"
            )
        if "window" in raw and "K" in raw:
            raise ContractViolation("give either 'window' or 'K', not both")
        window = WindowSpec.parse(raw.get("window", raw.get("K")))
        defaults = RecencySpec()
        recency = RecencySpec(
            enabled=bool(raw.get("tag_recency", defaults.enabled)),
            max_bucket=int(raw.get("recency_buckets", defaults.max_bucket)),
            mode=str(raw.get("recency_mode", defaults.mode)),
            tag_speaker=bool(raw.get("tag_speaker", defaults.tag_speaker)),
        )
        tfidf_raw = raw.get("tfidf") or {}
        if not isinstance(tfidf_raw, Mapping):
            raise ContractViolation(f"'tfidf' must be a mapping, got {tfidf_raw!r}")
        tfidf_defaults = TfidfSpec()
        ngram = tfidf_raw.get("ngram_range", tfidf_defaults.ngram_range)
        tfidf = TfidfSpec(
            ngram_range=(int(ngram[0]), int(ngram[1])),
            min_df=tfidf_raw.get("min_df", tfidf_defaults.min_df),
            max_df=tfidf_raw.get("max_df", tfidf_defaults.max_df),
            max_features=tfidf_raw.get("max_features", tfidf_defaults.max_features),
            sublinear_tf=bool(tfidf_raw.get("sublinear_tf", tfidf_defaults.sublinear_tf)),
            use_idf=bool(tfidf_raw.get("use_idf", tfidf_defaults.use_idf)),
            norm=tfidf_raw.get("norm", tfidf_defaults.norm),
        )
        base = cls()
        return cls(
            window=window,
            recency=recency,
            tfidf=tfidf,
            include_state=bool(raw.get("include_state", base.include_state)),
            include_speaker=bool(raw.get("include_speaker", base.include_speaker)),
            lowercase=bool(raw.get("lowercase", base.lowercase)),
            token_pattern=str(raw.get("token_pattern", base.token_pattern)),
        )

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any], head: str, *, block: str = "baseline_b0") -> "FeaturizeConfig":
        """Build one head's config from the resolved project config.

        Reads ``<block>.plan.<head>``, with ``<block>.tfidf`` and
        ``<block>.defaults`` merged underneath it so shared knobs are written
        once. Raises a clear :class:`ContractViolation` when the block is
        missing, because ``PROPOSED_CONFIG_B0.yaml`` has not been merged into
        ``configs/default.yaml`` (which this agent must not edit -- the
        coordinator owns that file).
        """
        try:
            plan = get_dotted(dict(cfg), f"{block}.plan")
        except KeyError as exc:
            raise ContractViolation(
                f"config block {block!r} is missing. Merge PROPOSED_CONFIG_B0.yaml into "
                "configs/default.yaml (coordinator-owned) before running Arm B0."
            ) from exc
        if not isinstance(plan, Mapping) or head not in plan:
            raise ContractViolation(f"{block}.plan has no entry for head {head!r}")
        entry = plan[head]
        if not isinstance(entry, Mapping):
            raise ContractViolation(f"{block}.plan.{head} must be a mapping, got {entry!r}")
        merged: dict[str, Any] = {}
        block_cfg = cfg.get(block) if isinstance(cfg, Mapping) else None
        if isinstance(block_cfg, Mapping):
            defaults = block_cfg.get("defaults")
            if isinstance(defaults, Mapping):
                merged.update(defaults)
            shared_tfidf = block_cfg.get("tfidf")
            if isinstance(shared_tfidf, Mapping):
                merged["tfidf"] = dict(shared_tfidf)
        for key, value in entry.items():
            if key == "tfidf" and isinstance(value, Mapping):
                merged_tfidf = dict(merged.get("tfidf") or {})
                merged_tfidf.update(value)
                merged["tfidf"] = merged_tfidf
            else:
                merged[key] = value
        return cls.from_mapping(merged)

    def with_(self, **changes: Any) -> "FeaturizeConfig":
        """Return a copy with fields replaced. Ablations stay one call wide."""
        return replace(self, **changes)

    def describe(self) -> dict[str, Any]:
        """The whole configuration, JSON-ready. Attach this to every number."""
        return {
            "window": self.window.describe(),
            "recency": self.recency.describe(),
            "tfidf": self.tfidf.describe(),
            "include_state": bool(self.include_state),
            "include_speaker": bool(self.include_speaker),
            "lowercase": bool(self.lowercase),
            "token_pattern": self.token_pattern,
        }


def split_context(context: ContextWindow) -> tuple[list[str], Optional[str]]:
    """Split a :class:`ContextWindow` into (turn lines, state line).

    :attr:`ContextWindow.turns` already holds the rendered ``"speaker|text"``
    lines, oldest first; the state line is whatever :attr:`ContextWindow.text`
    carries after them. Mirrors ``reflex.select._split_context`` so both
    consumers read the frozen layout the same way, and returns ``None`` for the
    state rather than guessing if the layout ever changes.

    **The cap that this cannot undo:** ``ContextWindow.turns`` has ALREADY been
    truncated to ``data.context_turns_K``. A featurizer window can only narrow
    that, never widen it. Build contexts with ``data.context_turns_K=full`` and
    let this module own windowing -- see
    :meth:`RecencyTfidfFeaturizer.assert_window_reachable`.
    """
    turns = list(context.turns)
    text = context.text or ""
    if not turns:
        return [], (text or None)
    prefix = "\n".join(turns) + "\n"
    if text.startswith(prefix):
        remainder = text[len(prefix) :]
        return turns, (remainder or None)
    return turns, None


def _compile_pattern(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern)


def line_tokens(
    line: str,
    bucket: Optional[int],
    config: FeaturizeConfig,
    *,
    pattern: Optional["re.Pattern[str]"] = None,
) -> list[str]:
    """Tokenize one rendered ``"speaker|text"`` line into (optionally) tagged tokens.

    ``bucket is None`` means "do not tag" -- used for the state line and for the
    whole document when recency is disabled.
    """
    rx = pattern or _compile_pattern(config.token_pattern)
    raw = line.lower() if config.lowercase else line
    speaker, sep, body = raw.partition("|")
    if not sep:
        speaker, body = "", raw
    speaker_tokens = rx.findall(speaker) if config.include_speaker else []
    body_tokens = rx.findall(body)
    if bucket is None or not config.recency.enabled:
        return speaker_tokens + body_tokens
    tag = config.recency.tag(bucket)
    glue = config.recency.sep
    if config.recency.mode == "turn":
        # select.py renders turn mode as "speaker|r0|body words here", so the
        # bucket is glued to the first WHITESPACE token of the body and the rest
        # of the line is plain. Tokenizing per whitespace chunk (rather than over
        # the whole body) is what keeps this path byte-identical to
        # :func:`tokenize_rendered` over the ship rendering.
        #
        # A consequence worth knowing, and a property of the SHIP rendering
        # rather than of this code: if that first whitespace token yields no word
        # token -- "customer|i need to return an item", where "i" is one
        # character and the word pattern needs two -- the line's bucket is LOST
        # entirely. `turn` mode is unmeasured on TF-IDF anyway (D7 measured
        # `word`), but this is the kind of thing that would quietly weaken a
        # turn-mode arm rather than break it.
        chunks = body.split()
        if not chunks:
            return speaker_tokens
        out = speaker_tokens + [f"{tag}{glue}{tok}" for tok in rx.findall(chunks[0])]
        for chunk in chunks[1:]:
            out.extend(rx.findall(chunk))
        return out
    if config.recency.tag_speaker:
        speaker_tokens = [f"{tag}{glue}{tok}" for tok in speaker_tokens]
    return speaker_tokens + [f"{tag}{glue}{tok}" for tok in body_tokens]


def build_document(
    turn_lines: Sequence[str],
    state_line: Optional[str],
    config: FeaturizeConfig,
) -> str:
    """Render one context into the whitespace-separated token string to vectorize.

    Windowing happens first, then bucketing over the SELECTED lines, then
    tokenization. The state line, when included, is appended last and untagged
    (it summarizes the whole prefix, so it has no single recency).

    The result is a plain string on purpose: the probe harness can cache it,
    hash it, and eyeball it, and the vectorizer's tokenizer is a bare
    ``str.split``, so nothing can re-tokenize a tag off its token.
    """
    rx = _compile_pattern(config.token_pattern)
    chosen = window_turns(turn_lines, config.window)
    if config.recency.enabled:
        buckets: list[Optional[int]] = list(bucket_sequence(len(chosen), config.recency.max_bucket))
    else:
        buckets = [None] * len(chosen)
    tokens: list[str] = []
    for line, bucket in zip(chosen, buckets):
        tokens.extend(line_tokens(line, bucket, config, pattern=rx))
    if config.include_state and state_line:
        tokens.extend(line_tokens(state_line, None, config, pattern=rx))
    return " ".join(tokens)


ContextLike = Union[ContextWindow, Sequence[str]]


def _as_lines_and_state(item: ContextLike) -> tuple[Sequence[str], Optional[str]]:
    """Normalize one input to (turn lines, state line).

    Deliberately only two accepted forms. A ``(lines, state)`` pair was
    considered and rejected: ``("agent|hi", "customer|yo")`` is both a pair and
    a two-line thread, and a dispatcher that guesses between them would
    misfeaturize two-turn contexts in silence. Callers with a state line of
    their own call :func:`build_document` directly.
    """
    if isinstance(item, ContextWindow):
        return split_context(item)
    if isinstance(item, (list, tuple)) and all(isinstance(x, str) for x in item):
        return list(item), None
    raise ContractViolation(
        "expected a ContextWindow or a sequence of 'speaker|text' lines; got "
        f"{type(item).__name__}"
    )


def build_documents(items: Iterable[ContextLike], config: FeaturizeConfig) -> list[str]:
    """Render many contexts. Accepts :class:`ContextWindow` objects or bare
    sequences of rendered ``"speaker|text"`` lines."""
    out: list[str] = []
    for item in items:
        lines, state = _as_lines_and_state(item)
        out.append(build_document(lines, state, config))
    return out


# --------------------------------------------------------------------------- #
# The featurizer
# --------------------------------------------------------------------------- #


def _split_tokens(document: str) -> list[str]:
    """The vectorizer's tokenizer: documents are pre-tokenized, so just split.

    Module-level (not a lambda) so a fitted featurizer stays picklable.
    """
    return document.split()


class RecencyTfidfFeaturizer:
    """Recency-tagged word 1-2gram TF-IDF over a windowed dialogue context.

    The importable component. It does not train, score, or read a corpus; it
    turns contexts into a sparse matrix and tells you exactly how.

    Typical use::

        from reflex.featurize import FeaturizeConfig, RecencyTfidfFeaturizer, WindowSpec, RecencySpec

        cfg = FeaturizeConfig(window=WindowSpec.last_k(6), recency=RecencySpec(max_bucket=3))
        fx = RecencyTfidfFeaturizer(cfg)
        X_train = fx.fit_transform_contexts(train_contexts)
        X_dev = fx.transform_contexts(dev_contexts)
        fx.describe()          # attach to every number you report (D6)
        fx.n_features          # reconcile against D7's 56k / 148k vocabulary sizes

    Fitting is deterministic: sklearn sorts the vocabulary, and the document
    strings are a pure function of the inputs and the config.
    """

    def __init__(self, config: Optional[FeaturizeConfig] = None) -> None:
        self.config = config or FeaturizeConfig()
        self._vectorizer: Any = None

    # -- documents ---------------------------------------------------------- #

    def build_documents(self, items: Iterable[ContextLike]) -> list[str]:
        """Render contexts to token strings without vectorizing them."""
        return build_documents(items, self.config)

    def assert_window_reachable(self, cfg: Mapping[str, Any]) -> None:
        """Refuse to run when ``data.context_turns_K`` is narrower than this window.

        A :class:`ContextWindow` is already truncated to the global
        ``data.context_turns_K``; a wider featurizer window silently gets fewer
        turns than it asked for and reports a number for a configuration that
        never ran. That is precisely the D6 failure mode, so this raises rather
        than warns.
        """
        try:
            raw = get_dotted(dict(cfg), "data.context_turns_K")
        except KeyError:
            return
        if raw is None or (isinstance(raw, str) and str(raw).strip().lower() in {"full", "all", "none", ""}):
            return
        try:
            global_k = int(raw)
        except (TypeError, ValueError):
            return
        needed = self.config.window.max_turns_needed()
        if needed is None or needed > global_k:
            raise ContractViolation(
                f"featurizer window {self.config.window} needs "
                f"{'the full thread' if needed is None else needed} prior turns but "
                f"data.context_turns_K={global_k} already truncated every ContextWindow to "
                f"{global_k}. Build contexts with data.context_turns_K=full and let "
                "reflex.featurize own the window (that is the point of a per-head window)."
            )

    # -- fit / transform ---------------------------------------------------- #

    def _make_vectorizer(self) -> Any:
        from sklearn.feature_extraction.text import TfidfVectorizer  # local: keep import cost off module load

        spec = self.config.tfidf
        return TfidfVectorizer(
            analyzer="word",
            tokenizer=_split_tokens,
            token_pattern=None,
            lowercase=False,  # already lowercased while building the document
            preprocessor=None,
            ngram_range=tuple(spec.ngram_range),
            min_df=spec.min_df,
            max_df=spec.max_df,
            max_features=spec.max_features,
            sublinear_tf=spec.sublinear_tf,
            use_idf=spec.use_idf,
            norm=spec.norm,
        )

    def fit(self, documents: Sequence[str]) -> "RecencyTfidfFeaturizer":
        self._vectorizer = self._make_vectorizer()
        self._vectorizer.fit(list(documents))
        return self

    def fit_transform(self, documents: Sequence[str]) -> Any:
        self._vectorizer = self._make_vectorizer()
        return self._vectorizer.fit_transform(list(documents))

    def transform(self, documents: Sequence[str]) -> Any:
        self._require_fitted()
        return self._vectorizer.transform(list(documents))

    def fit_contexts(self, items: Iterable[ContextLike]) -> "RecencyTfidfFeaturizer":
        return self.fit(self.build_documents(items))

    def fit_transform_contexts(self, items: Iterable[ContextLike]) -> Any:
        return self.fit_transform(self.build_documents(items))

    def transform_contexts(self, items: Iterable[ContextLike]) -> Any:
        return self.transform(self.build_documents(items))

    # -- introspection ------------------------------------------------------ #

    def _require_fitted(self) -> None:
        if self._vectorizer is None:
            raise ContractViolation("RecencyTfidfFeaturizer is not fitted; call fit() or fit_transform() first")

    @property
    def is_fitted(self) -> bool:
        return self._vectorizer is not None

    @property
    def vectorizer(self) -> Any:
        """The underlying fitted ``TfidfVectorizer``."""
        self._require_fitted()
        return self._vectorizer

    @property
    def n_features(self) -> int:
        """Vocabulary size. D7 reports 56k (plain, full window) growing to 148k at
        MAXB=8 -- the one handle the record gives on the probe's pruning."""
        self._require_fitted()
        return len(self._vectorizer.vocabulary_)

    def feature_names(self) -> list[str]:
        self._require_fitted()
        return list(self._vectorizer.get_feature_names_out())

    def describe(self) -> dict[str, Any]:
        """Configuration plus fitted vocabulary size. Report this with the score."""
        out = self.config.describe()
        out["fitted"] = self.is_fitted
        if self.is_fitted:
            out["n_features"] = self.n_features
        return out


# --------------------------------------------------------------------------- #
# Adapter for the probes/ harness (probes.featurizer_api.Featurizer)
# --------------------------------------------------------------------------- #


def tokenize_rendered(
    text: str,
    *,
    token_pattern: str = WORD_PATTERN,
) -> list[str]:
    """Tokenize a RENDERED context string (select.py layout) into tagged tokens.

    THIS IS THE FUNCTION THAT STOPS THE SILENT FAILURE. The ship renderer,
    ``reflex.select._render_variant``, emits lines like::

        agent|r0|hello, r0|how r0|can r0|help
        state|disclosed: ... || actions: ...

    A stock ``TfidfVectorizer()`` over that string applies ``\\b\\w\\w+\\b`` and
    yields ``agent``, ``r0``, ``hello``, ``r0``, ``how``, ... -- the bucket
    becomes its own feature, every content word is untagged, and the run still
    completes and reports a plausible accuracy. Every recency-tagged arm would
    silently measure plain TF-IDF plus a bucket-count feature.

    This tokenizer parses the layout instead: the speaker prefix is emitted once
    as its own token, and each ``r{n}|word`` chunk keeps its bucket glued to
    every word token it contains. The output is exactly what
    :func:`build_document` produces for the same window, tagging and state
    setting, so the Arm B0 path and the probe path cannot diverge --
    ``tests/test_featurize.py`` asserts that equality over every spec in
    ``probes.featurizer_api.CONFORMANCE_SPECS``.
    """
    rx = _compile_pattern(token_pattern)
    tokens: list[str] = []
    for line in text.split("\n"):
        if not line:
            continue
        speaker, sep, rest = line.partition("|")
        if not sep:
            speaker, rest = "", line
        tokens.extend(rx.findall(speaker))
        for chunk in rest.split():
            head, glue, tail = chunk.partition("|")
            if glue and re.fullmatch(r"r\d+", head):
                tokens.extend(f"{head}|{tok}" for tok in rx.findall(tail))
            else:
                tokens.extend(rx.findall(chunk))
    return tokens


def _shuffle_seed(seed: int, convo_id: Any, turn_index: Any) -> int:
    """A stable integer seed for one row's permutation.

    ``hash()`` is salted per process for strings, so a row would shuffle
    differently in the fit pass and the score pass of the ``shuf->shuf`` arm and
    the control would measure noise instead of order. sha256 is not salted.
    """
    import hashlib

    digest = hashlib.sha256(f"{seed}:{convo_id}:{turn_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class ProbeFeaturizer:
    """Implements ``probes.featurizer_api.Featurizer``.

    Two jobs, kept separate exactly as that harness requires:

    1. :meth:`render_context` -- ``ContextWindow -> str``, byte-identical to
       ``reflex.select._render_variant`` for every spec with ``shuffle=None``.
       That identity is the harness's whole conformance requirement: a probe
       measured on a rendering the ship path cannot reproduce is not
       transferable. It is re-implemented here rather than delegated because the
       order control needs to permute the chosen lines BETWEEN windowing and
       tagging, which ``_render_variant`` has no hook for; the harness's
       :func:`probes.featurizer_api.conformance_report` is what proves the two
       still agree, and it needs no corpus.
    2. :meth:`make_vectorizer` -- an unfitted TF-IDF whose tokenizer understands
       the rendering (see :func:`tokenize_rendered`).

    The shuffle is a pure function of ``(shuffle_seed, convo_id, turn_index)``,
    so a row permutes identically in the fit pass and the score pass.
    """

    #: Rendering rules this adapter guarantees, for the run digest.
    RENDER_CONTRACT = "reflex.select._render_variant (shuffle=None), bucket=min(n-1-i, MAXB)"

    def __init__(self, token_pattern: str = WORD_PATTERN) -> None:
        self.token_pattern = token_pattern

    # -- render ------------------------------------------------------------- #

    def render_context(self, context: ContextWindow, spec: Any) -> str:
        k = getattr(spec, "k", "inherit")
        tag_recency = bool(getattr(spec, "tag_recency", False))
        buckets = max(0, int(getattr(spec, "recency_buckets", 0)))
        mode = str(getattr(spec, "recency_mode", "word"))
        shuffle = getattr(spec, "shuffle", None)
        seed = int(getattr(spec, "shuffle_seed", 0))

        # _HeadContext.is_inherited(): k == "inherit" AND not tag_recency.
        if k == "inherit" and not tag_recency and shuffle is None:
            return context.text

        turns, state = split_context(context)
        if state is None and turns:
            # _render_variant's fallback when build_context's layout changed.
            return context.text

        if shuffle == "pre_window":
            turns = self._permute(list(turns), seed, context)

        if k in ("full", "inherit"):
            chosen = list(turns)
        else:
            k_int = int(k)
            chosen = list(turns[max(0, len(turns) - k_int) :])

        if shuffle == "within_window":
            chosen = self._permute(chosen, seed, context)
        elif shuffle not in (None, "pre_window"):
            raise ContractViolation(
                f"unknown shuffle {shuffle!r}; expected None, 'within_window' or 'pre_window'"
            )

        if tag_recency:
            n = len(chosen)
            chosen = [
                _render_tag_turn(line, min(n - 1 - i, buckets), mode)
                for i, line in enumerate(chosen)
            ]

        lines = list(chosen)
        if state:
            lines.append(state)
        return "\n".join(lines)

    @staticmethod
    def _permute(lines: list[str], seed: int, context: ContextWindow) -> list[str]:
        import random

        rng = random.Random(_shuffle_seed(seed, context.convo_id, context.turn_index))
        out = list(lines)
        rng.shuffle(out)
        return out

    # -- vectorize ---------------------------------------------------------- #

    def make_vectorizer(self, spec: Any) -> Any:
        """Unfitted sklearn TF-IDF over rendered context strings."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        analyzer = str(getattr(spec, "analyzer", "word"))
        if analyzer != "word":
            raise ContractViolation(
                f"analyzer must be 'word' -- D7 measured word 1-2grams; got {analyzer!r}"
            )
        ngram = tuple(getattr(spec, "ngram_range", (1, 2)))
        return TfidfVectorizer(
            analyzer="word",
            tokenizer=tokenize_rendered,
            token_pattern=None,
            lowercase=bool(getattr(spec, "lowercase", True)),
            ngram_range=(int(ngram[0]), int(ngram[1])),
            min_df=getattr(spec, "min_df", 1),
            max_df=getattr(spec, "max_df", 1.0),
            max_features=getattr(spec, "max_features", None),
            sublinear_tf=bool(getattr(spec, "sublinear_tf", False)),
        )

    # -- identity ----------------------------------------------------------- #

    def fingerprint(self) -> dict[str, Any]:
        from reflex import __version__

        return {
            "featurizer": "reflex.featurize.ProbeFeaturizer",
            "reflex_version": __version__,
            "render_contract": self.RENDER_CONTRACT,
            "token_pattern": self.token_pattern,
            "tokenizer": "reflex.featurize.tokenize_rendered",
            "tag_separator": "|",
            "speaker_token": "emitted once per line, untagged",
            "shuffle_seed_rule": "sha256(seed:convo_id:turn_index) -- unsalted, stable across processes",
        }


def _render_tag_turn(line: str, bucket: int, mode: str) -> str:
    """Byte-identical to ``reflex.select._tag_turn``.

    Duplicated rather than imported: ``_tag_turn`` is private to a frozen module
    this agent may not edit, and importing a private name would make this module
    break silently if select.py's internals move. The harness's conformance
    report compares the two renderings directly, so a divergence is caught
    mechanically rather than trusted.
    """
    tag = f"r{bucket}"
    speaker, sep, body = line.partition("|")
    if not sep:
        speaker, body = "", line
    if mode == "turn":
        tagged = f"{tag}|{body}"
    else:
        tagged = " ".join(f"{tag}|{token}" for token in body.split())
    return f"{speaker}|{tagged}" if sep else tagged


def build_featurizer(**kwargs: Any) -> ProbeFeaturizer:
    """Factory for ``probes/probe.yaml``'s ``probe.featurizer_factory``.

    Set that key to ``"reflex.featurize:build_featurizer"``.
    """
    return ProbeFeaturizer(**kwargs)
