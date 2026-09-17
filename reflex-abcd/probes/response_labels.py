"""Where the gold label for H5 / H7 / H4 comes from, and what fraction of turns
have one.

EVERYTHING BELOW WAS READ OUT OF THE CODE, NOT ASSUMED. The code paths are named
inline so the derivation can be re-checked against the compiler rather than
against this docstring.

H5 -- SKELETON (fixed-width classifier)
---------------------------------------
Code path: ``compile.compile_bank`` -> ``compile.split_sentences`` ->
``compile.delexicalize_sentence`` -> ``compile.label_acts`` ->
``compile.extract_skeletons`` -> ``TurnLabel.skeleton_id``.

Per agent turn with ``nextstep == "retrieve_utterance"``:

1. ``split_sentences(turn.text)`` (pysbd) splits the agent utterance.
2. Each sentence is delexicalized; a sentence whose delexicalized form is empty
   is DROPPED, so it never reaches the act labeller.
3. ``label_acts`` tags each surviving sentence with one of the 9 acts in
   ``compile.act_inventory``.
4. The gold skeleton is the ORDERED TUPLE of those acts, looked up in the
   train-built skeleton table: ``skeleton_ids.get(tuple(turn_acts))``.

Consequences that the metric has to respect:

* The label space is closed over TRAIN. ``extract_skeletons`` keeps every
  observed tuple with no minimum count, so on train the lookup never misses and
  ``skeleton_id`` is non-null for every retrieve turn. On DEV/TEST an unseen act
  tuple yields ``None``, and ``train._gold_index`` turns that into ``-1``, which
  ``models.compute_losses`` masks off SILENTLY. That out-of-vocabulary rate is a
  hard ceiling on H5 and this module measures it (:func:`audit`).
* The gold is a function of the ACT LABELLER, which is ``rules_plus_embed`` and
  whose accuracy is UNMEASURED (DECISIONS D15: both check files carry a blank
  human tally). So H5's ceiling is the act labeller's accuracy, and no H5 number
  can be read as "how well the model predicts what the agent said" -- only as
  "how well it predicts what the act labeller said the agent said".
* ``end_conversation`` and ``take_action`` turns have no skeleton at all.

H7 -- TEMPLATE (per-act scoring head, NOT a flat 4,489-way classifier)
----------------------------------------------------------------------
Code path: ``compile.extract_templates`` -> ``compile.dedup_templates`` ->
the ``(act, _normalize_for_dedup(surface_form)) -> template_id`` lookup in
``compile.compile_bank`` -> ``TurnLabel.template_ids``; scored by
``select._score_templates``.

Per sentence position of a retrieve turn, the gold is the template whose merged
SURFACE FORMS contain that sentence's normalized delexicalized text, within the
same act. ``template_ids`` is one entry per act position, and an entry is the
EMPTY STRING when the lookup misses -- which happens when the sentence's template
was dropped by ``compile.min_template_count: 2`` (a phrasing no human used
twice).

The single most important fact for choosing a metric: **``select._score_templates``
ranks each position only against ``templates_by_act[act]``**, not against the
whole bank. The label space for one position is that act's pool, which on the
current bank ranges from ~208 (INSTRUCT) to ~1,066 (ASK) candidates -- not 4,489.
Any metric that treats H7 as a single ~4,500-way choice is measuring a head that
does not exist.

H4 -- VALUES (flat index over [enumerable values | copy positions])
-------------------------------------------------------------------
Code path: ``TurnLabel.values`` (raw ABCD gold, copied verbatim from the turn) ->
``train.build_train_examples`` (``value_gold = str(values[0])``) ->
``models.ModernBERT.value_target_index`` -> a single integer column.

The label is NOT a string. It is a column in a space of width
``len(model.value_list) + model.value_context_len`` (126 + 100 = 226 on ABCD
v1.1): an enumerable value occupies its ``value_list`` column, while a
non-enumerable value occupies the COPY column of its ``<marker>`` inside
``model.value_candidate_tokens(context_texts, action)``. ``value_target_index``
returns ``-1`` when the gold is neither -- the official processor DROPS those
rows for AST and stores ``-1`` for CDS, so they are unscorable, not wrong.

Two ceilings travel with H4 and both are reported by :func:`audit`:

* ``train.value_position: first`` supervises ``values[0]`` ONLY, because H4 has
  no position input (``train.build_train_examples`` raises on any other setting).
  For ``verify-identity`` and ``validate-purchase`` the gold has three values;
  positions b and c are unmeasurable by H4 AS DEFINED, not merely unmeasured.
* The copy tier depends on the WINDOW: ``value_candidate_tokens`` is built from
  ``context.turns``, so narrowing K can push the gold marker out of the candidate
  list and turn a scorable row into a ``-1``. That makes H4's scorable
  DENOMINATOR a function of the window being probed. Comparing H4 across windows
  without holding the denominator fixed compares two different questions; this
  module therefore records the per-window resolvable set and
  :mod:`probes.response_metrics` intersects them.

A WARNING ABOUT DEV GOLDS, WHICH IS A REAL ASYMMETRY AND NOT A NICETY
----------------------------------------------------------------------
``compile.compile_bank`` writes ``labels/train.jsonl`` and nothing else.
``train._derive_turn_labels`` is the only existing path to a dev/test gold, and
it is NOT the same procedure: it deliberately SKIPS delexicalization (its own
docstring says so), matching a RAW sentence against templates whose text is
DELEXICALIZED. Train golds are delex-vs-delex; dev golds are raw-vs-delex. Only
71 of 4,489 templates bear a slot, so the bound is small, but it is a
train/dev label asymmetry that biases dev H7 coverage DOWNWARD, and it must be
reported next to any dev H7 number rather than discovered later.

``train._normalize_for_match`` is also a hand-copy of
``compile._normalize_for_dedup``. They are byte-identical today (verified by
:func:`check_normalizer_drift`); a future edit to one and not the other would
silently cost dev recall, so the harness asserts it every run.
"""

from __future__ import annotations

import inspect
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence


# --------------------------------------------------------------------------- #
# Row types -- one per scorable unit, which differs per head
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class H5Row:
    """One retrieve turn. The scorable unit for skeleton is the TURN."""

    turn_id: str
    convo_id: int
    turn_index: int
    context: Any                      # reflex.schemas.ContextWindow
    gold_skeleton_id: Optional[str]   # None => act tuple unseen in the train bank
    gold_acts: tuple                  # the act tuple itself, for the near-miss tiers


@dataclass(frozen=True)
class H7Row:
    """One SENTENCE POSITION of a retrieve turn. The scorable unit for template.

    ``gold_template_id == ""`` means the sentence had no surviving template
    (``compile.min_template_count``). Such a row is scorable only under the
    UNCONDITIONAL denominator, where it counts as a miss.
    """

    turn_id: str
    convo_id: int
    turn_index: int
    position: int
    n_positions: int
    context: Any
    act: str
    gold_template_id: str


@dataclass(frozen=True)
class H4Row:
    """One take_action turn. The scorable unit for values."""

    turn_id: str
    convo_id: int
    turn_index: int
    context: Any
    action: str
    action_with_position: str      # the " a" suffix the official processor appends
    gold_value: str                # values[0] -- the only position H4 is trained on
    n_gold_values: int
    gold_column: int = -1          # filled by H4TargetIndexer; -1 == unscorable
    candidate_tokens: tuple = ()


# --------------------------------------------------------------------------- #
# Label spaces, read out of the bank and the ontology
# --------------------------------------------------------------------------- #


@dataclass
class LabelSpaces:
    """The canonical class order for each response-selection head.

    Read from the same places ``select.build_selector`` reads them, so a probe
    and the shipped selector cannot disagree about what class 7 means.
    """

    skeleton_ids: list = field(default_factory=list)
    skeleton_acts: dict = field(default_factory=dict)
    template_ids: list = field(default_factory=list)
    template_act: dict = field(default_factory=dict)
    template_text: dict = field(default_factory=dict)
    templates_by_act: dict = field(default_factory=dict)
    value_list: list = field(default_factory=list)
    value_context_len: int = 0
    bank_hash: str = ""

    @property
    def h4_width(self) -> int:
        return len(self.value_list) + int(self.value_context_len)

    def summary(self) -> dict:
        return {
            "bank_hash": self.bank_hash,
            "h5_n_classes": len(self.skeleton_ids),
            "h7_n_templates_total": len(self.template_ids),
            "h7_n_candidates_by_act": {a: len(v) for a, v in sorted(self.templates_by_act.items())},
            "h4_width": self.h4_width,
            "h4_n_enumerable": len(self.value_list),
            "h4_value_context_len": int(self.value_context_len),
        }


def label_spaces(bank: Any, cfg: dict) -> LabelSpaces:
    """Build :class:`LabelSpaces` from a compiled bank plus the ontology."""
    from reflex.config import get_dotted
    from reflex.data import load_ontology
    from reflex.models import _value_list

    ontology = load_ontology(cfg)
    by_act: dict = defaultdict(list)
    for template in bank.templates:
        by_act[template.act].append(template.template_id)
    return LabelSpaces(
        skeleton_ids=[s.skeleton_id for s in bank.skeletons],
        skeleton_acts={s.skeleton_id: tuple(s.acts) for s in bank.skeletons},
        template_ids=[t.template_id for t in bank.templates],
        template_act={t.template_id: t.act for t in bank.templates},
        template_text={t.template_id: t.text_delex for t in bank.templates},
        templates_by_act={a: list(v) for a, v in sorted(by_act.items())},
        value_list=list(_value_list(ontology)),
        value_context_len=int(get_dotted(cfg, "model.value_context_len")),
        bank_hash=str(getattr(bank, "bank_hash", "") or ""),
    )


# --------------------------------------------------------------------------- #
# Guards on the label derivation itself
# --------------------------------------------------------------------------- #


def check_normalizer_drift() -> dict:
    """Assert ``train._normalize_for_match`` still mirrors ``compile._normalize_for_dedup``.

    They are two hand-copies of one function (train.py says so in its docstring).
    Drift between them costs recall on every DERIVED dev label while leaving
    train labels untouched, so it would show up as "H7 is worse on dev" rather
    than as an error. Compared on behaviour over awkward inputs, not on source
    text, because either may be reformatted.
    """
    from reflex.compile import _normalize_for_dedup
    from reflex.train import _normalize_for_match

    probes = [
        "Thank you.", "thank you", "Thank you!!", "Can I have your order id?",
        "  Multiple   spaces ,here . ", "Sure;no space after semicolon",
        "ends with dots...", "!", "...", "Your order {order_id} ships <name> today.",
        "A,B,C", "UPPER CASE TEXT", "", "   ",
    ]
    disagreements = [
        {"input": p, "compile": _normalize_for_dedup(p), "train": _normalize_for_match(p)}
        for p in probes
        if _normalize_for_dedup(p) != _normalize_for_match(p)
    ]
    return {
        "ok": not disagreements,
        "n_probes": len(probes),
        "disagreements": disagreements,
        "note": (
            "compile._normalize_for_dedup builds TRAIN golds; train._normalize_for_match "
            "builds DERIVED dev/test golds. Drift silently lowers dev H7 coverage only."
        ),
    }


def dev_gold_provenance() -> dict:
    """Record HOW the dev golds were derived, so a number cannot be quoted without it."""
    from reflex import train as _train

    source = inspect.getsource(_train._derive_turn_labels)
    return {
        "function": "reflex.train._derive_turn_labels",
        "delexicalizes": "delexicalize" in source and "does NOT delexicalize" not in source,
        "documented_asymmetry": (
            "train golds are delex-vs-delex; derived dev golds match a RAW sentence "
            "against DELEXICALIZED template text. Only slot-bearing templates can be "
            "affected; the harness reports how many of the bank's templates bear a slot "
            "so the bound is explicit rather than quoted from a stale comment."
        ),
        "extends_bank": False,
        "unmatched_sentence_gold": "\"\" (empty string), i.e. no gold, not a wrong gold",
    }


# --------------------------------------------------------------------------- #
# H4's index-space target
# --------------------------------------------------------------------------- #


class H4TargetIndexer:
    """Map ``(context, action, gold value string)`` to H4's integer column.

    ``models.ModernBERT.value_target_index`` is the SINGLE SOURCE OF TRUTH for
    this index (models.py says so about ``value_list``), so by default this class
    calls it. Building the model loads the encoder weights, which is the only
    expensive thing in this module; it does NOT parse the corpus.

    ``mirror_only=True`` reproduces the same arithmetic from a bare tokenizer,
    for the case where the coordinator does not want to pay for the encoder. It
    is OFF by default and, when on, :meth:`verify_against_model` must be run at
    least once against the real model before any H4 number is reported -- an
    index-space label derived by a second implementation is exactly the kind of
    unattributed number this project refuses.
    """

    def __init__(self, model: Any = None, tokenizer: Any = None, ontology: dict = None,
                 value_context_len: int = 0, mirror_only: bool = False) -> None:
        from reflex.models import _enumerable_by_category, _value_list

        self.model = model
        self.mirror_only = bool(mirror_only) or model is None
        self.tokenizer = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
        if ontology is None and model is not None:
            ontology = getattr(model, "ontology", None)
        self.value_list = list(model.value_list) if model is not None else list(_value_list(ontology))
        self.enumerable = (
            dict(model.enumerable) if model is not None and hasattr(model, "enumerable")
            else _enumerable_by_category(ontology)
        )
        self.value_by_action: dict = {}
        if model is not None and getattr(model, "value_by_action", None):
            self.value_by_action = dict(model.value_by_action)
        elif ontology is not None:
            for _section, buttons in ontology.get("actions", {}).items():
                for button, categories in buttons.items():
                    self.value_by_action[button] = list(categories)
        self.value_context_len = int(
            getattr(model, "value_context_len", 0) or value_context_len
        )

    # -- the mirror; kept to the shape of models.value_candidate_tokens ------- #
    def _candidate_tokens(self, context_texts: Sequence[str], action: str) -> list:
        filtered: list = []
        for text in context_texts:
            for token in self.tokenizer.tokenize(text):
                if token in filtered:
                    continue
                if len(token) > 2:
                    filtered.append(token)
        effective_max = self.value_context_len - (len(self.tokenizer.tokenize(action)) + 3)
        return filtered[-effective_max:]

    def _mirror_index(self, context_texts: Sequence[str], action: str, value: str) -> tuple:
        potential = self.value_by_action.get(action.split(" ")[0], [])
        tokens = self._candidate_tokens(context_texts, action)
        target = -1
        for option in potential:
            if option in self.enumerable:
                if value in self.enumerable[option]:
                    target = self.value_list.index(value)
            else:
                marker = f"<{option}>"
                if marker in tokens:
                    target = len(self.value_list) + tokens.index(marker)
            if target >= 0:
                break
        return target, tokens

    def index(self, context_texts: Sequence[str], action: str, value: str) -> tuple:
        """Return ``(column, candidate_tokens)``; column ``-1`` means unscorable."""
        if not self.mirror_only:
            return self.model.value_target_index(list(context_texts), action, value)
        return self._mirror_index(context_texts, action, value)

    def verify_against_model(self, model: Any, rows: Sequence["H4Row"], limit: int = 500) -> dict:
        """Check the mirror against ``models.ModernBERT.value_target_index``."""
        bad = []
        n = 0
        for row in rows[:limit]:
            texts = _context_texts(row.context)
            want, _ = model.value_target_index(texts, row.action_with_position, row.gold_value)
            got, _ = self._mirror_index(texts, row.action_with_position, row.gold_value)
            n += 1
            if want != got and len(bad) < 10:
                bad.append({"turn_id": row.turn_id, "model": want, "mirror": got})
        return {"ok": not bad, "n_checked": n, "disagreements": bad}


def _context_texts(context: Any) -> list:
    """``context.turns`` with the ``speaker|`` prefix stripped.

    Exactly what ``train.build_train_examples`` and ``select._score_values``
    feed to ``value_candidate_tokens``; duplicated nowhere else in this package.
    """
    return [piece.split("|", 1)[1] if "|" in piece else piece for piece in context.turns]


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #


def build_rows(
    partitions: Any,
    split: str,
    bank: Any,
    labels: Sequence[Any],
    cfg: dict,
    h4_indexer: Optional[H4TargetIndexer] = None,
) -> dict:
    """Turn one split into ``{"h5": [...], "h7": [...], "h4": [...]}``.

    Uses ``reflex.data.build_context`` so every row carries the SAME
    :class:`~reflex.schemas.ContextWindow` the trained model would see; the
    featurizer re-windows from that object per arm. ``data.context_turns_K``
    must therefore be at least as wide as the widest window probed -- a
    ContextWindow cannot recover turns the loader already dropped, and
    ``select._render_variant`` only warns about it. :func:`audit` re-checks it.

    Cost note (DEFECTS_OPEN D-10): the expensive part is ``build_partitions``
    and ``load_raw_abcd``, done ONCE by the caller. This function is linear in
    turns and allocates one ContextWindow per agent-side turn.
    """
    from reflex.data import build_context, iter_agent_turns, load_raw_abcd, turn_key

    spaces = label_spaces(bank, cfg)
    partition = getattr(partitions, split)
    raw = load_raw_abcd(cfg)
    scenarios: dict = {}
    for convos in raw.values():
        for convo in convos:
            scenarios[int(convo["convo_id"])] = convo.get("scenario", {}) or {}

    by_turn_id = {label.turn_id: label for label in labels}
    multi_value = set(__import__("reflex.config", fromlist=["get_dotted"]).get_dotted(
        cfg, "train.multi_value_actions"))

    h5: list = []
    h7: list = []
    h4: list = []
    for convo_id, turn_index, turn in iter_agent_turns(partition):
        turns = partition[convo_id]
        turn_id = turn_key(split, convo_id, turn_index)
        label = by_turn_id.get(turn_id)
        context = build_context(turns, turn_index, scenarios.get(int(convo_id), {}), cfg)

        if turn.nextstep == "retrieve_utterance":
            if label is None:
                continue
            template_ids = list(label.template_ids or [])
            acts = tuple(
                spaces.template_act.get(t, "") if t else ""
                for t in template_ids
            )
            h5.append(H5Row(
                turn_id=turn_id, convo_id=int(convo_id), turn_index=int(turn_index),
                context=context, gold_skeleton_id=label.skeleton_id,
                gold_acts=tuple(spaces.skeleton_acts.get(label.skeleton_id, ()) or acts),
            ))
            gold_acts = spaces.skeleton_acts.get(label.skeleton_id) or acts
            for position, template_id in enumerate(template_ids):
                act = (
                    spaces.template_act.get(template_id, "")
                    if template_id
                    else (gold_acts[position] if position < len(gold_acts) else "")
                )
                h7.append(H7Row(
                    turn_id=turn_id, convo_id=int(convo_id), turn_index=int(turn_index),
                    position=position, n_positions=len(template_ids), context=context,
                    act=act, gold_template_id=template_id or "",
                ))

        elif turn.nextstep == "take_action":
            values = list((label.values if label is not None else turn.values) or [])
            if not values:
                continue
            action = str(turn.action or "")
            suffixed = f"{action} a" if action in multi_value else action
            row = H4Row(
                turn_id=turn_id, convo_id=int(convo_id), turn_index=int(turn_index),
                context=context, action=action, action_with_position=suffixed,
                gold_value=str(values[0]), n_gold_values=len(values),
            )
            if h4_indexer is not None:
                column, tokens = h4_indexer.index(
                    _context_texts(context), suffixed, str(values[0])
                )
                row = H4Row(**{**row.__dict__, "gold_column": int(column),
                               "candidate_tokens": tuple(tokens)})
            h4.append(row)

    return {"h5": h5, "h7": h7, "h4": h4, "spaces": spaces}


# --------------------------------------------------------------------------- #
# The audit: what fraction of turns even have a gold, and every ceiling
# --------------------------------------------------------------------------- #


def audit(rows: dict, cfg: dict, split: str) -> dict:
    """Label-space and coverage facts. NO model, NO accuracy -- ceilings only.

    Every number here is a property of the DATA and the BANK. They are the
    denominators every metric in :mod:`probes.response_metrics` is quoted
    against, and several of them are hard ceilings no head can exceed.
    """
    from reflex.config import get_dotted

    spaces: LabelSpaces = rows["spaces"]
    h5: list = rows["h5"]
    h7: list = rows["h7"]
    h4: list = rows["h4"]

    known = set(spaces.skeleton_ids)
    h5_oov = sum(1 for r in h5 if r.gold_skeleton_id is None or r.gold_skeleton_id not in known)
    sk_counts = Counter(r.gold_skeleton_id for r in h5 if r.gold_skeleton_id in known)

    h7_with_gold = [r for r in h7 if r.gold_template_id]
    by_act = defaultdict(list)
    for r in h7_with_gold:
        by_act[r.act].append(r)

    turns = defaultdict(list)
    for r in h7:
        turns[r.turn_id].append(r)
    fully_covered = sum(1 for rs in turns.values() if rs and all(r.gold_template_id for r in rs))
    partly = sum(
        1 for rs in turns.values()
        if any(r.gold_template_id for r in rs) and not all(r.gold_template_id for r in rs)
    )

    h4_scorable = [r for r in h4 if r.gold_column >= 0]
    h4_copy = sum(1 for r in h4_scorable if r.gold_column >= len(spaces.value_list))
    arity3 = sum(1 for r in h4 if r.n_gold_values > 1)

    k = get_dotted(cfg, "data.context_turns_K")
    turn_lengths = [len(r.context.turns) for r in h5[:20000]]

    return {
        "split": split,
        "bank_hash": spaces.bank_hash,
        "loader_context_turns_K": k,
        "context_turns_observed_max": max(turn_lengths) if turn_lengths else 0,
        "WARNING_window_capped": (
            "data.context_turns_K is an int: any probe window wider than it is silently "
            "truncated and its arm is NOT the window it is labelled as. Re-run with "
            "data.context_turns_K=full."
        ) if not isinstance(k, str) else None,

        "h5": {
            "n_turns": len(h5),
            "n_classes": len(spaces.skeleton_ids),
            "gold_present": len(h5) - h5_oov,
            "gold_oov_rate": (h5_oov / len(h5)) if h5 else None,
            "oov_note": (
                "an act tuple unseen in train. train._gold_index maps it to -1 and "
                "models.compute_losses masks it off SILENTLY: it is a ceiling, not an error."
            ),
            "majority_class": sk_counts.most_common(1)[0][0] if sk_counts else None,
            "majority_share": (
                sk_counts.most_common(1)[0][1] / sum(sk_counts.values())
            ) if sk_counts else None,
            "ceiling_note": (
                "H5's gold is produced by compile.act_labeler (rules_plus_embed), whose "
                "accuracy is UNMEASURED -- DECISIONS D15: delex_check.md and act_check.md "
                "both carry a blank human tally. Any H5 number is conditional on it."
            ),
        },

        "h7": {
            "n_positions": len(h7),
            "n_positions_with_gold": len(h7_with_gold),
            "gold_present_rate": (len(h7_with_gold) / len(h7)) if h7 else None,
            "missing_gold_cause": (
                "compile.min_template_count=%s dropped the sentence's template; "
                "TurnLabel.template_ids stores \"\" for it." % get_dotted(cfg, "compile.min_template_count")
            ),
            "n_turns": len(turns),
            "turns_fully_covered": fully_covered,
            "turns_fully_covered_rate": (fully_covered / len(turns)) if turns else None,
            "turns_partly_covered": partly,
            "candidates_by_act": {a: len(v) for a, v in sorted(spaces.templates_by_act.items())},
            "positions_by_act": {a: len(v) for a, v in sorted(by_act.items())},
            "slot_bearing_templates": sum(
                1 for tid, text in spaces.template_text.items() if "{" in text
            ),
            "label_space_note": (
                "select._score_templates ranks a position only within templates_by_act[act]. "
                "The per-position label space is that act's pool, not the whole bank."
            ),
        },

        "h4": {
            "n_take_action_turns_with_values": len(h4),
            "index_width": spaces.h4_width,
            "n_enumerable_columns": len(spaces.value_list),
            "n_copy_columns": int(spaces.value_context_len),
            "n_scorable": len(h4_scorable),
            "scorable_rate": (len(h4_scorable) / len(h4)) if h4 else None,
            "unscorable_note": (
                "value_target_index returned -1: the gold is neither an enumerable value of "
                "one of the action's categories nor a <marker> present in the copy window. "
                "The official processor DROPS these for AST; they are unscorable, not wrong."
            ),
            "scorable_via_copy": h4_copy,
            "scorable_via_enumerable": len(h4_scorable) - h4_copy,
            "multi_value_turns": arity3,
            "multi_value_share": (arity3 / len(h4)) if h4 else None,
            "position_ceiling_note": (
                "train.value_position='first' supervises values[0] only (H4 has no position "
                "input; build_train_examples raises on any other setting). Positions b and c "
                "of verify-identity / validate-purchase are UNMEASURABLE by H4 as defined."
            ),
            "denominator_note": (
                "the copy tier is built from context.turns, so the scorable set is a "
                "FUNCTION OF THE WINDOW. Cross-window H4 comparisons must be taken on the "
                "intersection of the scorable sets; response_metrics does that."
            ),
        },
    }


def write_audit(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    return path
