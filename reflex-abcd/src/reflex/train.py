"""Spec 4.3 -- Trainer: training the encoder and all heads. No thresholding.

WHAT THIS MODULE OWNS
---------------------
The optimizer, the epoch loop, the batch assembly, and the early-stopping
signal. It owns NO threshold (spec 4.4 ``calibrate``), NO inference policy
(spec 4.5 ``select``), NO decision (spec 4.6 ``gate``) and NO reported metric
(spec 4.9 ``evaluate``). :func:`dev_selection_score` is explicitly not a
reported number -- it is loop control, and the contract says so.

THE TRAINING LOOP (spec 6.4)
----------------------------
::

    for epoch in range(epochs):
        model.sync_frozen_encoder()          # H7: the frozen twin catches up ...
        model.refresh_template_cache()       # ... and every template is re-embedded
        hard_negative_table = nearest same-act templates in that fresh cache
        for batch in shuffled(train_examples):
            losses = model.compute_losses(batch)   # 7 masked heads, one encoder pass
            losses["total"].backward(); clip; step; schedule
        score = dev_selection_score(model, dev_examples, cfg)   # H7 R@1 + H3 acc
        early stopping on `score`, patience = train.early_stopping_patience

``sync_frozen_encoder`` + ``refresh_template_cache`` happen at the TOP of each
epoch, so epoch 0 trains against features from the pretrained encoder and every
later epoch against the encoder as it stood when that epoch began. Between
refreshes the features are deliberately stale -- that is what spec 6.4's
"frozen copy, refreshed once per epoch" means. The trainable half of the
template embedding (``h7_template_projection``) still receives gradient on every
step, so H7 is never frozen, only its expensive input is.

FOUR PLACES THE FROZEN CONTRACT FORCED A DESIGN CHOICE
------------------------------------------------------
1. **H4's gold index is tokenizer-dependent, and**
   :func:`build_train_examples` **is handed no tokenizer.** The official index
   space (``utils/process.py::value_to_id``) is ``len(value_list) +
   tokens.index('<slot>')`` over a *tokenized* context. So
   :func:`build_train_examples` stores the raw ingredients
   (``value_action``/``value_gold``/``value_context_texts``) and
   :func:`train` resolves them exactly once, through
   ``model.value_target_index`` -- the single source of truth for that index
   space. An example that never passes through :func:`train` carries
   ``value_label == -1`` and H4 is simply masked off for it. Re-deriving the
   filter here instead would have forked the official logic into two copies.
2. **``compute_losses`` takes ONE H7 position per example** (``act_ids`` is
   ``(B,)``), but a multi-sentence agent turn has one gold template per act
   position. :func:`train` therefore cycles the trained position by epoch
   (``train.h7_position_strategy: cycle``), a pure function of
   ``(epoch, n_positions)``, so every position of every turn is trained within
   ``max(n_positions)`` epochs. :func:`dev_selection_score` scores ALL positions
   regardless, because evaluating one position would under-report H7.
3. **Hard negatives want "nearest", and** :func:`collate_batch` **is handed no
   model.** So the loop precomputes a nearest-same-act table from the freshly
   refreshed cache each epoch and attaches it to the examples;
   :func:`collate_batch` uses it when present and otherwise falls back to a
   deterministic same-act draw seeded from the example index, so the function
   remains usable standalone exactly as its contract describes.
4. **``labels/dev.jsonl`` does not exist.** ``compile.compile_bank`` labels the
   TRAIN split only, but early stopping is defined on dev H7/H3. Rather than
   silently training with a dead H7 dev signal, :func:`train` derives the
   missing split's labels from the bank with compile's own public helpers
   (``split_sentences`` + ``label_acts``) and persists them through
   ``compile.write_turn_labels`` so the cost is paid once. Gated on
   ``train.derive_missing_labels``; see the module report for the deviation.

WHAT LIVES IN CONFIG (spec 10)
------------------------------
``train.lr``, ``train.epochs``, ``train.batch_size``, ``train.weight_decay``,
``train.loss_weights``, ``train.early_stopping_patience``, ``train.seeds``,
``train.checkpoint_dir``, plus the REFLEX additions this module owns:
``train.warmup_ratio``, ``train.max_grad_norm``, ``train.log_every``,
``train.truncation_side``, ``train.value_position``,
``train.multi_value_actions``, ``train.h7_position_strategy``,
``train.hard_negatives_from_cache``, ``train.derive_missing_labels``,
``train.smoke``, ``train.smoke_conversations``,
``train.smoke_dev_conversations``, ``train.smoke_epochs``,
``train.smoke_max_steps``, ``train.dev_max_conversations``. Device and
determinism come from ``runtime.train_device`` / ``runtime.deterministic`` /
``runtime.torch_threads``. There is no tunable number in this file.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import time
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn

from reflex.config import get_dotted, repo_root, resolve_path
from reflex.contracts import ContractViolation
from reflex.schemas import ACT_INVENTORY, NEXT_STEPS, Bank, Partitions, TurnLabel

__all__ = [
    "build_train_examples",
    "collate_batch",
    "dev_selection_score",
    "train",
]

#: Placeholder act id for rows H7 does not apply to. Any index is fine -- those
#: rows are masked out of the InfoNCE term -- but it must be in range.
_NO_ACT = 0

#: Sentinel for "this head does not apply to this row", the same ``-1``
#: ``utils/process.py`` writes and ``models.compute_losses`` reads.
_NA = -1

#: Mixed into the per-example negative-sampling seed so that two different
#: derivations of a seed from the same example index cannot collide.
_NEGATIVE_SEED_SALT = 0x5EED

#: Splits :class:`Partitions` exposes, in the order the manifest lists them.
_SPLITS: tuple[str, ...] = ("train", "dev", "test_seen", "test_novel")


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #


def _normalize_for_match(text: str) -> str:
    """Normalize a sentence for template lookup.

    Mirrors ``compile._normalize_for_dedup`` -- lowercase, collapse whitespace,
    tighten punctuation spacing, drop a trailing ``.``/``!`` run but keep ``?``.
    It is duplicated rather than imported because it is private to
    :mod:`reflex.compile`; :func:`_derive_turn_labels` is the only caller and a
    drift here costs recall on a derived dev label, never a wrong train label.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
    return re.sub(r"[.!\s]+$", "", text) or text.strip()


def _class_index(names: Sequence[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def _gold_index(index: dict[str, int], gold: Optional[str], head: str) -> int:
    """Class id of ``gold`` in a head's canonical order; ``-1`` when absent."""
    if gold is None or gold == "":
        return _NA
    try:
        return index[gold]
    except KeyError as exc:
        raise ContractViolation(
            f"gold {head} label {gold!r} is not in the head's canonical class list "
            f"({len(index)} classes). The bank/ontology used here is not the one the "
            f"labels were produced against."
        ) from exc


def _resolve_device(cfg: dict[str, Any]) -> torch.device:
    """``runtime.train_device``: ``auto`` picks MPS, then CUDA, then CPU."""
    requested = str(get_dotted(cfg, "runtime.train_device")).lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _seed_everything(cfg: dict[str, Any], seed: int) -> None:
    """Seed python / numpy / torch and honour ``runtime.deterministic``."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on this machine
        torch.cuda.manual_seed_all(seed)
    threads = int(get_dotted(cfg, "runtime.torch_threads"))
    if threads > 0:
        torch.set_num_threads(threads)
    if bool(get_dotted(cfg, "runtime.deterministic")):
        try:
            # warn_only: MPS has no deterministic kernel for several ops and a
            # hard failure here would make the documented default device
            # unusable. The seeds above still make a run reproducible on CPU.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch builds
            pass


def _git_commit() -> str:
    """HEAD sha of the repo, or ``"unknown"`` outside a git work tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - git absent
        return "unknown"
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else "unknown"


def _config_hash(cfg: dict[str, Any]) -> str:
    """Stable 16-char sha256 over the resolved config, provenance keys excluded."""
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _machine() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def _subset(partition: dict[int, list[Any]], limit: int) -> dict[int, list[Any]]:
    """First ``limit`` conversations in partition order; ``limit <= 0`` keeps all."""
    if limit <= 0 or limit >= len(partition):
        return partition
    return {cid: turns for i, (cid, turns) in enumerate(partition.items()) if i < limit}


# --------------------------------------------------------------------------- #
# 4.3  examples
# --------------------------------------------------------------------------- #


def build_train_examples(
    partitions: Partitions,
    bank: Bank,
    labels: Sequence[TurnLabel],
    cfg: dict[str, Any],
    split: str = "train",
) -> list[dict[str, Any]]:
    """Assemble training examples: one per agent-side turn.

    See :func:`reflex.contracts.build_train_examples` for the frozen contract.
    One example per :func:`reflex.data.iter_agent_turns` turn -- the same
    denominator spec 8.1 uses -- in conversation order then turn order.

    Head masking follows spec 6.4 exactly: ``action_label`` and ``value_*`` are
    set only on ``take_action`` turns, ``skeleton_label`` / ``template_indices``
    / ``act_labels`` only on ``retrieve_utterance`` turns, everything else is
    ``-1`` so ``models.compute_losses`` masks it off. A turn whose gold label
    exists but whose template was dropped from the bank (``min_template_count``)
    keeps its skeleton label and loses only its H7 row.

    ``value_label`` is left at ``-1`` here and resolved in :func:`train`; see
    design note 1 at the top of this module.

    Args:
        partitions: From :func:`reflex.data.build_partitions`.
        bank: The compiled bank.
        labels: From :func:`reflex.compile.load_turn_labels` for the same
            ``split``. Turns with no label row still produce an example (H1/H2
            are always supervised); only the bank-derived heads go silent.
        cfg: Resolved config.
        split: Which partition to build from; one of ``Partitions``' four.

    Returns:
        Examples in deterministic order (conversation, then turn).

    Raises:
        ContractViolation: on an unknown ``split`` or a gold label that is not
            in the canonical class order for its head.
    """
    from reflex.data import build_context, iter_agent_turns, load_ontology, load_raw_abcd, turn_key
    from reflex.data import action_list, subflow_list

    if split not in _SPLITS:
        raise ContractViolation(f"unknown split {split!r}; expected one of {list(_SPLITS)}")
    partition: dict[int, list[Any]] = getattr(partitions, split)

    ontology = load_ontology(cfg)
    nextstep_index = _class_index(list(NEXT_STEPS))
    intent_index = _class_index(subflow_list(ontology))
    action_index = _class_index(action_list(ontology))
    act_index = _class_index(list(ACT_INVENTORY))
    skeleton_index = _class_index([s.skeleton_id for s in bank.skeletons])
    template_index = _class_index([t.template_id for t in bank.templates])
    template_act = {t.template_id: t.act for t in bank.templates}

    value_by_action: dict[str, list[str]] = {}
    for _section, buttons in ontology.get("actions", {}).items():
        for button, categories in buttons.items():
            value_by_action[button] = list(categories)
    multi_value_actions = set(get_dotted(cfg, "train.multi_value_actions"))
    value_position = str(get_dotted(cfg, "train.value_position"))
    if value_position != "first":
        raise ContractViolation(
            f"train.value_position={value_position!r} is not supported: H4 as defined in spec "
            "6.4 has no position input, so the a/b/c positions of verify-identity and "
            "validate-purchase are indistinguishable to it. 'first' trains position 'a'."
        )

    # Scenarios live on the raw conversations, not on Partitions, and
    # build_context needs them for the disclosure guard.
    raw = load_raw_abcd(cfg)
    scenarios: dict[int, dict[str, Any]] = {}
    for convos in raw.values():
        for convo in convos:
            scenarios[int(convo["convo_id"])] = convo.get("scenario", {}) or {}

    by_turn_id = {label.turn_id: label for label in labels}

    take_action = nextstep_index["take_action"]
    retrieve = nextstep_index["retrieve_utterance"]

    examples: list[dict[str, Any]] = []
    for convo_id, turn_index, turn in iter_agent_turns(partition):
        turns = partition[convo_id]
        context = build_context(turns, turn_index, scenarios.get(int(convo_id), {}), cfg)
        turn_id = turn_key(split, convo_id, turn_index)
        label = by_turn_id.get(turn_id)

        nextstep_label = _gold_index(nextstep_index, turn.nextstep, "nextstep")
        example: dict[str, Any] = {
            "turn_id": turn_id,
            "split": split,
            "convo_id": int(convo_id),
            "turn_index": int(turn_index),
            "context_text": context.text,
            "nextstep_label": nextstep_label,
            "intent_label": _gold_index(intent_index, turn.intent, "intent"),
            "action_label": _NA,
            "value_label": _NA,
            "value_action": None,
            "value_gold": None,
            "value_context_texts": [],
            "value_context_tokens": [],
            "skeleton_label": _NA,
            "template_indices": [],
            "act_labels": [],
            "h7_position": 0,
            "hard_negatives": None,
        }

        if nextstep_label == take_action:
            example["action_label"] = _gold_index(action_index, turn.action, "action")
            potential = value_by_action.get(turn.action or "", [])
            values = list(turn.values or [])
            if potential and values:
                # utils/process.py::CDSProcessor.collect_examples: the two
                # three-input buttons are expanded into positions a/b/c; every
                # other button takes values[0]. H4 has no position input, so we
                # train position 'a' -- see the ContractViolation above.
                action_name = (
                    f"{turn.action} a" if turn.action in multi_value_actions else str(turn.action)
                )
                example["value_action"] = action_name
                example["value_gold"] = str(values[0])
                example["value_context_texts"] = [
                    piece.split("|", 1)[1] if "|" in piece else piece for piece in context.turns
                ]
        elif nextstep_label == retrieve and label is not None:
            example["skeleton_label"] = _gold_index(
                skeleton_index, label.skeleton_id, "skeleton"
            )
            positions: list[int] = []
            acts: list[int] = []
            for template_id in label.template_ids or []:
                if not template_id:
                    continue
                if template_id not in template_index:
                    raise ContractViolation(
                        f"{turn_id}: gold template {template_id!r} is not in this bank. The "
                        f"labels were produced against a different bank_hash."
                    )
                positions.append(template_index[template_id])
                acts.append(act_index[template_act[template_id]])
            example["template_indices"] = positions
            example["act_labels"] = acts

        examples.append(example)
    return examples


# --------------------------------------------------------------------------- #
# 4.3  collation
# --------------------------------------------------------------------------- #


def _same_act_negatives(
    example: dict[str, Any],
    position: int,
    gold_template: int,
    act_id: int,
    pool: Sequence[int],
    k: int,
) -> list[int]:
    """``k`` negative template indices for one H7 row.

    Prefers ``example["hard_negatives"]`` -- the nearest same-act neighbours the
    epoch loop precomputed from the freshly refreshed template cache. Falls back
    to a deterministic draw from the act's pool, seeded from the example's own
    turn id and position rather than from global RNG state, so a batch is
    reproducible however the loader shuffles.
    """
    precomputed = example.get("hard_negatives")
    if precomputed is not None and position < len(precomputed):
        chosen = list(precomputed[position])[:k]
        if len(chosen) == k:
            return chosen
    else:
        chosen = []
    candidates = [t for t in pool if t != gold_template]
    if not candidates:
        return (chosen + [gold_template] * k)[:k]
    rng = random.Random(
        hash((example["turn_id"], position, act_id, _NEGATIVE_SEED_SALT)) & 0xFFFFFFFF
    )
    while len(chosen) < k:
        chosen.append(candidates[rng.randrange(len(candidates))])
    return chosen[:k]


def collate_batch(examples: Sequence[dict[str, Any]], cfg: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Collate examples into padded tensors, including H7 hard negatives.

    See :func:`reflex.contracts.collate_batch` for the frozen contract.

    Context strings are tokenized to ``data.max_len`` with
    ``truncation_side = train.truncation_side`` (``left`` by default): the
    ``state|`` line is the LAST line of the context, so left truncation keeps
    both the state and the most recent turns on the few percent of turns that
    overflow.

    The H7 candidate block is emitted as ``template_indices`` ``(B, 1+K)`` --
    bank indices, gold at column 0 -- not as features. Features are a property
    of the model's once-per-epoch cache; :func:`train` gathers them. The
    negatives are ``model.hard_negatives_per_positive`` templates OF THE SAME
    ACT, nearest-first when the loop attached a table and otherwise a
    per-example seeded draw.

    Args:
        examples: From :func:`build_train_examples`, optionally annotated by
            :func:`train` with ``h7_position``, ``hard_negatives``,
            ``value_label`` and ``value_context_tokens``.
        cfg: Resolved config. Uses ``data.max_len``, ``train.truncation_side``,
            ``model.hard_negatives_per_positive`` and ``model.value_context_len``.
        tokenizer: ``model.tokenizer`` -- the one carrying the ABCD ``<slot>``
            markers. A plain ``build_encoder`` tokenizer would shatter
            ``<order_id>`` and H4 could never hit a copy target.

    Returns:
        A batch dict of tensors plus a ``masks`` sub-dict of per-head booleans.
    """
    if not examples:
        raise ContractViolation("collate_batch received an empty example list")

    max_len = int(get_dotted(cfg, "data.max_len"))
    n_negatives = int(get_dotted(cfg, "model.hard_negatives_per_positive"))
    value_context_len = int(get_dotted(cfg, "model.value_context_len"))
    tokenizer.truncation_side = str(get_dotted(cfg, "train.truncation_side"))

    encoded = tokenizer(
        [example["context_text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    batch: dict[str, Any] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
    }

    def _column(key: str) -> torch.Tensor:
        return torch.tensor([int(example[key]) for example in examples], dtype=torch.long)

    batch["nextstep_labels"] = _column("nextstep_label")
    batch["intent_labels"] = _column("intent_label")
    batch["action_labels"] = _column("action_label")
    batch["value_labels"] = _column("value_label")
    batch["skeleton_labels"] = _column("skeleton_label")

    # ---- H4: the official AST second input (the copy source) ---------------- #
    token_rows = [list(example.get("value_context_tokens") or []) for example in examples]
    width = min(max(len(row) for row in token_rows), value_context_len)
    if width > 0:
        pad_id = tokenizer.pad_token_id or 0
        copy_ids = torch.full((len(examples), width), int(pad_id), dtype=torch.long)
        copy_mask = torch.zeros((len(examples), width), dtype=torch.long)
        for i, row in enumerate(token_rows):
            row = row[:width]
            if not row:
                continue
            ids = tokenizer.convert_tokens_to_ids(row)
            copy_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            copy_mask[i, : len(ids)] = 1
        batch["context_input_ids"] = copy_ids
        batch["context_attention_mask"] = copy_mask

    # ---- H6 / H7: one act position per example ------------------------------ #
    act_ids: list[int] = []
    act_labels: list[int] = []
    gold_templates: list[int] = []
    candidate_rows: list[list[int]] = []
    applicable: list[bool] = []
    templates_by_act_index = cfg.get("_templates_by_act_index") or {}

    for example in examples:
        positions = example["template_indices"]
        acts = example["act_labels"]
        if positions and acts:
            position = int(example.get("h7_position", 0)) % len(positions)
            gold = int(positions[position])
            act_id = int(acts[position])
            pool = templates_by_act_index.get(act_id) or [gold]
            negatives = _same_act_negatives(example, position, gold, act_id, pool, n_negatives)
            act_ids.append(act_id)
            act_labels.append(act_id)
            gold_templates.append(gold)
            candidate_rows.append([gold] + negatives)
            applicable.append(True)
        else:
            act_ids.append(_NO_ACT)
            act_labels.append(_NA)
            gold_templates.append(_NA)
            candidate_rows.append([0] * (1 + n_negatives))
            applicable.append(False)

    batch["act_ids"] = torch.tensor(act_ids, dtype=torch.long)
    batch["act_labels"] = torch.tensor(act_labels, dtype=torch.long)
    batch["template_indices"] = torch.tensor(candidate_rows, dtype=torch.long)
    batch["template_gold_index"] = torch.tensor(gold_templates, dtype=torch.long)
    batch["template_mask"] = torch.tensor(applicable, dtype=torch.bool)
    batch["masks"] = {
        "nextstep": batch["nextstep_labels"] >= 0,
        "intent": batch["intent_labels"] >= 0,
        "action": batch["action_labels"] >= 0,
        "values": batch["value_labels"] >= 0,
        "skeleton": batch["skeleton_labels"] >= 0,
        "act": batch["act_labels"] >= 0,
        "template": batch["template_mask"],
    }
    return batch


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# 4.3  the early-stopping signal (NOT a reported metric -- see contracts.py)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _forward_dev(
    model: Any,
    dev_examples: Sequence[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """One eval pass over dev: returns ``(query (N, q), action_pred (N,))``."""
    batch_size = int(get_dotted(cfg, "train.batch_size"))
    device = next(model.parameters()).device
    queries: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    for start in range(0, len(dev_examples), batch_size):
        chunk = list(dev_examples[start : start + batch_size])
        batch = _to_device(collate_batch(chunk, cfg, model.tokenizer), device)
        outputs = model(batch)
        queries.append(outputs["query"].detach().float().cpu())
        actions.append(outputs["action_logits"].detach().float().argmax(dim=-1).cpu())
    return torch.cat(queries), torch.cat(actions)


def dev_selection_score(model: Any, dev_examples: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, float]:
    """Compute the EARLY-STOPPING signal only (spec 6.4: dev H7 recall@1 + H3 accuracy).

    See :func:`reflex.contracts.dev_selection_score` for the frozen contract.
    **This is not a reported metric** -- spec 4.9 makes :mod:`reflex.evaluate`
    the only place metrics are computed, and nothing here may be quoted.

    ``h7_recall_at_1`` is measured over every ACT POSITION that has a gold
    template, not just the one position the current epoch trains, and each
    position is ranked against the templates OF ITS OWN ACT -- which is the
    candidate set :mod:`reflex.select` will face. ``h3_accuracy`` is over
    ``take_action`` turns only. Both are ``0.0`` when their denominator is
    empty, which is honest rather than undefined: an epoch with no dev action
    turns genuinely provides no H3 evidence.

    Args:
        model: The model being trained. Its template cache is refreshed here if
            it is empty, so the function is safe to call standalone.
        dev_examples: Dev examples from :func:`build_train_examples`.
        cfg: Resolved config.

    Returns:
        ``{"h7_recall_at_1", "h3_accuracy", "score"}``; ``score`` is the sum
        early stopping maximizes.
    """
    if not dev_examples:
        return {"h7_recall_at_1": 0.0, "h3_accuracy": 0.0, "score": 0.0}

    was_training = model.training
    model.eval()
    try:
        if model.template_features.numel() == 0:
            model.refresh_template_cache()
        query, action_pred = _forward_dev(model, dev_examples, cfg)

        # ---- H3 ---------------------------------------------------------- #
        gold_actions = torch.tensor(
            [int(example["action_label"]) for example in dev_examples], dtype=torch.long
        )
        action_rows = gold_actions >= 0
        h3_total = int(action_rows.sum())
        h3_hits = int((action_pred[action_rows] == gold_actions[action_rows]).sum()) if h3_total else 0

        # ---- H7: every position, ranked inside its own act ----------------- #
        by_act: dict[int, list[tuple[int, int]]] = {}
        for row, example in enumerate(dev_examples):
            for position, template in enumerate(example["template_indices"]):
                act_id = int(example["act_labels"][position])
                by_act.setdefault(act_id, []).append((row, int(template)))

        device = next(model.parameters()).device
        features = model.template_features
        act_pool: dict[int, list[int]] = cfg.get("_templates_by_act_index") or {}
        h7_total = 0
        h7_hits = 0
        for act_id, rows in by_act.items():
            pool = act_pool.get(act_id)
            if not pool:
                continue
            pool_tensor = torch.tensor(pool, dtype=torch.long, device=features.device)
            position_of = {template: i for i, template in enumerate(pool)}
            queries = query[[row for row, _ in rows]].to(device)
            scores = model.score_templates(
                queries,
                torch.full((queries.shape[0],), act_id, dtype=torch.long, device=device),
                features.index_select(0, pool_tensor).to(device),
            )
            predicted = scores.argmax(dim=-1).cpu().tolist()
            for (_, gold), chosen in zip(rows, predicted):
                if gold not in position_of:
                    continue
                h7_total += 1
                h7_hits += int(chosen == position_of[gold])
    finally:
        if was_training:
            model.train()

    h7_recall = h7_hits / h7_total if h7_total else 0.0
    h3_accuracy = h3_hits / h3_total if h3_total else 0.0
    return {
        "h7_recall_at_1": float(h7_recall),
        "h3_accuracy": float(h3_accuracy),
        "score": float(h7_recall + h3_accuracy),
    }


# --------------------------------------------------------------------------- #
# Dev labels: derived, because compile writes labels/train.jsonl only
# --------------------------------------------------------------------------- #


def _derive_turn_labels(
    partition: dict[int, list[Any]],
    bank: Bank,
    cfg: dict[str, Any],
    split: str,
) -> list[TurnLabel]:
    """Label a non-train split against an ALREADY-COMPILED bank.

    ``compile.compile_bank`` writes ``labels/train.jsonl`` and nothing else, but
    early stopping is defined on dev H7/H3, so the dev split needs skeleton and
    template golds too. This reproduces only the two steps of spec 6.2 that a
    non-train split can legitimately take -- sentence splitting and act
    labelling, both through compile's own public functions -- and then matches
    each sentence to an EXISTING bank template. It never extends the bank:
    train-only compilation (spec 6.2) is preserved, an unmatched sentence simply
    has no gold.

    It deliberately does NOT delexicalize, because delexicalization needs
    ``fill.collect_slot_sources`` (still a stub) and ``compile``'s private
    per-conversation source builder. The cost is bounded: only 49 of the bank's
    4,417 templates bear a slot at all (DEFECTS_OPEN D-2), so at most ~1% of dev
    sentences can be lost to a literal that a template has masked.
    """
    from reflex.compile import label_acts, split_sentences
    from reflex.data import iter_agent_turns, turn_key

    lookup: dict[tuple[str, str], str] = {}
    for template in bank.templates:
        for form in template.surface_forms or [template.text_delex]:
            lookup.setdefault((template.act, _normalize_for_match(form)), template.template_id)
    skeleton_of = {tuple(s.acts): s.skeleton_id for s in bank.skeletons}

    stubs: list[tuple[str, Any, list[str]]] = []
    sentences: list[str] = []
    for convo_id, turn_index, turn in iter_agent_turns(partition):
        turn_id = turn_key(split, convo_id, turn_index)
        pieces = split_sentences(turn.text, cfg) if turn.nextstep == "retrieve_utterance" else []
        stubs.append((turn_id, turn, pieces))
        sentences.extend(pieces)

    acts = label_acts(sentences, cfg)
    cursor = 0
    labels: list[TurnLabel] = []
    for turn_id, turn, pieces in stubs:
        turn_acts = acts[cursor : cursor + len(pieces)]
        cursor += len(pieces)
        if turn.nextstep == "retrieve_utterance":
            template_ids = [
                lookup.get((act, _normalize_for_match(text)), "")
                for act, text in zip(turn_acts, pieces)
            ]
            labels.append(
                TurnLabel(
                    turn_id=turn_id,
                    nextstep=str(turn.nextstep),
                    intent=str(turn.intent),
                    skeleton_id=skeleton_of.get(tuple(turn_acts)),
                    template_ids=template_ids,
                )
            )
        else:
            labels.append(
                TurnLabel(
                    turn_id=turn_id,
                    nextstep=str(turn.nextstep),
                    intent=str(turn.intent),
                    action=turn.action,
                    values=list(turn.values) if turn.values else None,
                )
            )
    return labels


def _labels_for(
    partition: dict[int, list[Any]],
    bank: Bank,
    cfg: dict[str, Any],
    split: str,
    persist: bool,
) -> tuple[list[TurnLabel], str]:
    """Load ``labels/<split>.jsonl``, deriving and persisting it when absent."""
    from reflex.compile import load_turn_labels, write_turn_labels

    try:
        return list(load_turn_labels(split, cfg)), "loaded"
    except FileNotFoundError:
        if not bool(get_dotted(cfg, "train.derive_missing_labels")):
            raise
    labels = _derive_turn_labels(partition, bank, cfg, split)
    if persist:
        write_turn_labels(labels, split, cfg)
        return labels, "derived+persisted"
    return labels, "derived"


# --------------------------------------------------------------------------- #
# H7 hard negatives: nearest same-act neighbours in the fresh template cache
# --------------------------------------------------------------------------- #


def _templates_by_act_index(bank: Bank) -> dict[int, list[int]]:
    """``act id -> [bank template index]``, the H7 candidate pool per act."""
    act_index = _class_index(list(ACT_INVENTORY))
    pools: dict[int, list[int]] = {}
    for position, template in enumerate(bank.templates):
        pools.setdefault(act_index[template.act], []).append(position)
    return pools


@torch.no_grad()
def _nearest_same_act(
    features: torch.Tensor,
    pools: dict[int, list[int]],
    k: int,
) -> dict[int, list[int]]:
    """``template index -> its k nearest same-act neighbours`` (cosine, gold excluded).

    Computed from the FROZEN feature cache -- the semantically meaningful half
    of the template embedding, and the half that is stable within an epoch.
    Using the trainable projection instead would make epoch 0's negatives pure
    noise, since that projection starts random.
    """
    table: dict[int, list[int]] = {}
    if features.numel() == 0:
        return table
    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    for pool in pools.values():
        if len(pool) < 2:
            for template in pool:
                table[template] = []
            continue
        index = torch.tensor(pool, dtype=torch.long, device=normalized.device)
        block = normalized.index_select(0, index)
        similarity = block @ block.t()
        similarity.fill_diagonal_(-2.0)
        top = similarity.topk(min(k, len(pool) - 1), dim=-1).indices.cpu().tolist()
        for row, template in enumerate(pool):
            table[template] = [pool[column] for column in top[row]]
    return table


# --------------------------------------------------------------------------- #
# 4.3  the training loop
# --------------------------------------------------------------------------- #


def _build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW with ``train.weight_decay``, excluding biases and norm weights."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower() or "layernorm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(get_dotted(cfg, "train.weight_decay"))},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(get_dotted(cfg, "train.lr")),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any], total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then linear decay; ``train.warmup_ratio == 0`` gives decay only."""
    warmup = int(round(float(get_dotted(cfg, "train.warmup_ratio")) * max(total_steps, 1)))

    def _factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        remaining = max(total_steps - warmup, 1)
        return max(0.0, (total_steps - step) / remaining)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _factor)


def _resolve_value_labels(model: Any, examples: Sequence[dict[str, Any]]) -> int:
    """Fill ``value_label`` / ``value_context_tokens`` via ``model.value_target_index``.

    The official index space (``utils/process.py::value_to_id``) is defined over
    tokenized context, so it can only be computed once a tokenizer exists. Doing
    it here -- once per :func:`train` call, through the model's own reproduction
    of that function -- keeps a single source of truth for it.

    Returns:
        How many examples got a usable (``>= 0``) H4 target.
    """
    resolved = 0
    for example in examples:
        if not example.get("value_action") or example.get("value_gold") is None:
            continue
        target, tokens = model.value_target_index(
            example["value_context_texts"], example["value_action"], example["value_gold"]
        )
        example["value_label"] = int(target)
        example["value_context_tokens"] = list(tokens)
        resolved += int(target >= 0)
    return resolved


def _head_counts(examples: Sequence[dict[str, Any]]) -> dict[str, int]:
    """How many rows each head can actually learn from. Reported in the manifest."""
    return {
        "nextstep": sum(1 for e in examples if e["nextstep_label"] >= 0),
        "intent": sum(1 for e in examples if e["intent_label"] >= 0),
        "action": sum(1 for e in examples if e["action_label"] >= 0),
        "values": sum(1 for e in examples if e["value_label"] >= 0),
        "skeleton": sum(1 for e in examples if e["skeleton_label"] >= 0),
        "act": sum(1 for e in examples if e["act_labels"]),
        "template": sum(1 for e in examples if e["template_indices"]),
    }


def train(cfg: dict[str, Any], seed: int, fraction: float = 1.0, size: str = "base") -> dict[str, Any]:
    """Train the encoder and all heads jointly (spec 6.4). No thresholding here.

    See :func:`reflex.contracts.train` for the frozen contract.

    AdamW with ``train.lr`` / ``train.weight_decay``, linear warmup+decay,
    gradient clipping at ``train.max_grad_norm``, ``train.epochs`` epochs of the
    seven masked losses summed with ``train.loss_weights``, early stopping on
    :func:`dev_selection_score` with ``train.early_stopping_patience``. Every
    RNG is seeded from ``seed`` and ``runtime.deterministic`` is honoured.

    ``train.smoke`` shrinks the run to ``train.smoke_conversations`` train and
    ``train.smoke_dev_conversations`` dev conversations for
    ``train.smoke_epochs`` epochs. It exercises every head on real ABCD data and
    is the only mode that is cheap enough to run on every change; it is NOT a
    result.

    Args:
        cfg: Resolved config.
        seed: One of ``train.seeds``. Spec 6.4 requires all three.
        fraction: Learning-curve fraction the bank was compiled from (E4).
        size: ``"base"`` or ``"small"`` (E3c).

    Returns:
        ``{"checkpoint_path", "seed", "epochs_run", "best_dev", "resolved_encoder",
        "bank_hash"}`` plus ``manifest_path``, ``history``, ``device`` and
        ``head_counts`` for the caller's log.

    Raises:
        ContractViolation: if ``fraction`` disagrees with ``bank.source_fraction``.
    """
    from reflex.compile import load_bank
    from reflex.data import build_partitions, load_ontology
    from reflex.models import build_model, save_checkpoint

    started = time.time()
    _seed_everything(cfg, seed)

    smoke = bool(get_dotted(cfg, "train.smoke"))
    epochs = int(
        get_dotted(cfg, "train.smoke_epochs") if smoke else get_dotted(cfg, "train.epochs")
    )
    batch_size = int(get_dotted(cfg, "train.batch_size"))
    patience = int(get_dotted(cfg, "train.early_stopping_patience"))
    log_every = int(get_dotted(cfg, "train.log_every"))
    max_grad_norm = float(get_dotted(cfg, "train.max_grad_norm"))
    max_steps = int(get_dotted(cfg, "train.smoke_max_steps")) if smoke else 0
    strategy = str(get_dotted(cfg, "train.h7_position_strategy"))
    use_cached_negatives = bool(get_dotted(cfg, "train.hard_negatives_from_cache"))

    bank = load_bank(cfg)
    if abs(float(bank.source_fraction) - float(fraction)) > 1e-9:
        raise ContractViolation(
            f"--fraction {fraction} disagrees with the compiled bank's source_fraction "
            f"{bank.source_fraction}. Recompile with `reflex compile --fraction {fraction}` or "
            f"train with --fraction {bank.source_fraction}: a bank compiled from a different "
            f"slice of train has a different skeleton/template inventory, so H5 and H7 would "
            f"be scored against class ids the checkpoint never saw."
        )
    ontology = load_ontology(cfg)
    partitions = build_partitions(cfg)

    train_partition = partitions.train
    dev_partition = partitions.dev
    if smoke:
        train_partition = _subset(train_partition, int(get_dotted(cfg, "train.smoke_conversations")))
        dev_partition = _subset(dev_partition, int(get_dotted(cfg, "train.smoke_dev_conversations")))
    else:
        dev_partition = _subset(dev_partition, int(get_dotted(cfg, "train.dev_max_conversations")))
    full_dev = len(dev_partition) == len(partitions.dev)

    train_labels, train_label_origin = _labels_for(train_partition, bank, cfg, "train", persist=False)
    dev_labels, dev_label_origin = _labels_for(dev_partition, bank, cfg, "dev", persist=full_dev)

    subset = Partitions(
        train=train_partition,
        dev=dev_partition,
        test_seen=partitions.test_seen,
        test_novel=partitions.test_novel,
        novel_subflows=partitions.novel_subflows,
        dataset_hash=partitions.dataset_hash,
    )
    train_examples = build_train_examples(subset, bank, train_labels, cfg, split="train")
    dev_examples = build_train_examples(subset, bank, dev_labels, cfg, split="dev")
    if not train_examples:
        raise ContractViolation("no train examples; check Partitions.train and the smoke limits")

    # The act pools are a pure function of the bank; every collate_batch call
    # needs them and the frozen signature passes only cfg, so they ride along on
    # a private cfg key rather than being rebuilt 3,000 times per epoch.
    pools = _templates_by_act_index(bank)
    cfg = dict(cfg)
    cfg["_templates_by_act_index"] = pools

    device = _resolve_device(cfg)
    model = build_model(cfg, bank, ontology, size=size)
    model.to(device)

    value_rows = _resolve_value_labels(model, train_examples)
    _resolve_value_labels(model, dev_examples)
    head_counts = _head_counts(train_examples)

    optimizer = _build_optimizer(model, cfg)
    steps_per_epoch = max(1, (len(train_examples) + batch_size - 1) // batch_size)
    if max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch * epochs)

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_dev: dict[str, float] = {}
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_epoch = 0
    stale = 0
    order = list(range(len(train_examples)))

    for epoch in range(epochs):
        # Spec 6.4 H7: the frozen twin catches up, then every template is
        # re-embedded. Both happen exactly once, at the top of the epoch.
        model.eval()
        model.sync_frozen_encoder()
        model.refresh_template_cache()
        negatives = (
            _nearest_same_act(
                model.template_features,
                pools,
                int(get_dotted(cfg, "model.hard_negatives_per_positive")),
            )
            if use_cached_negatives
            else {}
        )
        for example in train_examples:
            positions = example["template_indices"]
            if not positions:
                continue
            example["h7_position"] = epoch % len(positions) if strategy == "cycle" else 0
            example["hard_negatives"] = (
                [negatives.get(int(t), []) for t in positions] if negatives else None
            )

        model.train()
        epoch_rng = random.Random(seed * 1_000_003 + epoch)
        epoch_rng.shuffle(order)
        totals: dict[str, float] = {}
        counted = 0
        for step in range(steps_per_epoch):
            chunk = [train_examples[i] for i in order[step * batch_size : (step + 1) * batch_size]]
            if not chunk:
                break
            batch = _to_device(collate_batch(chunk, cfg, model.tokenizer), device)
            batch["template_features"] = model.template_features.index_select(
                0, batch["template_indices"].reshape(-1).clamp(min=0)
            ).reshape(batch["template_indices"].shape[0], batch["template_indices"].shape[1], -1)
            losses = model.compute_losses(batch)
            losses["total"].backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            counted += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            if log_every > 0 and (step + 1) % log_every == 0:
                print(
                    f"  epoch {epoch} step {step + 1}/{steps_per_epoch} "
                    f"total={totals['total'] / counted:.4f}",
                    flush=True,
                )

        mean_losses = {name: value / max(counted, 1) for name, value in totals.items()}
        dev_score = dev_selection_score(model, dev_examples, cfg)
        history.append(
            {
                "epoch": epoch,
                "steps": counted,
                "losses": mean_losses,
                "dev": dev_score,
                "lr": scheduler.get_last_lr()[0],
                "seconds": round(time.time() - started, 1),
            }
        )
        print(
            f"epoch {epoch}: "
            + " ".join(f"{k}={v:.4f}" for k, v in sorted(mean_losses.items()))
            + f" | dev h7@1={dev_score['h7_recall_at_1']:.4f} "
            f"h3={dev_score['h3_accuracy']:.4f} score={dev_score['score']:.4f}",
            flush=True,
        )

        if dev_score["score"] > best_score:
            best_score = dev_score["score"]
            best_dev = dict(dev_score)
            best_epoch = epoch
            best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if patience >= 0 and stale > patience:
                print(f"early stop at epoch {epoch} (no dev gain for {stale} epochs)", flush=True)
                break

    if best_state is not None:
        model.to("cpu")
        model.load_state_dict(best_state)

    run_id = f"train-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-seed{seed}"
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "kind": "train",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "seeds": list(get_dotted(cfg, "train.seeds")),
        "fraction": float(fraction),
        "size": size,
        "smoke": smoke,
        "config_path": cfg.get("_config_path", ""),
        "config_hash": _config_hash(cfg),
        "overrides": list(cfg.get("_overrides", [])),
        "dataset_hash": partitions.dataset_hash,
        "bank_hash": bank.bank_hash,
        "git_commit": _git_commit(),
        "model_ids": {
            "requested_encoder": get_dotted(cfg, "model.encoder"),
            "resolved_encoder": model.resolved_encoder,
            "act_labeler_embed_model": get_dotted(cfg, "compile.act_labeler_embed_model"),
        },
        "machine": _machine(),
        "device": str(device),
        "novel_subflows": list(partitions.novel_subflows),
        "label_origin": {"train": train_label_origin, "dev": dev_label_origin},
        "counts": {
            "train_conversations": len(train_partition),
            "dev_conversations": len(dev_partition),
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "supervised_rows_per_head": head_counts,
            "h4_resolved_targets": value_rows,
            "templates": len(bank.templates),
            "skeletons": len(bank.skeletons),
        },
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_dev": best_dev,
        "history": history,
        "seconds": round(time.time() - started, 1),
        "llm_enabled": bool(get_dotted(cfg, "llm.enabled")),
    }

    checkpoint_path = save_checkpoint(model, cfg, seed, extra=manifest)
    manifest["checkpoint_path"] = checkpoint_path
    manifest_path = os.path.join(
        resolve_path(cfg, "train.checkpoint_dir"),
        os.path.basename(checkpoint_path).rsplit(".", 1)[0] + ".manifest.json",
    )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)

    return {
        "checkpoint_path": checkpoint_path,
        "seed": int(seed),
        "epochs_run": len(history),
        "best_dev": best_dev,
        "resolved_encoder": model.resolved_encoder,
        "bank_hash": bank.bank_hash,
        "manifest_path": manifest_path,
        "history": history,
        "device": str(device),
        "head_counts": head_counts,
    }
