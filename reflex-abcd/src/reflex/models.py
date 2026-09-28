"""Spec 6.4 -- Encoder + heads H1-H7: definitions and checkpoint I/O only.

No training loop, no thresholds, no inference policy. :mod:`reflex.train` owns
the optimizer and the epoch loop; :mod:`reflex.select` owns turning scores into a
:class:`~reflex.schemas.Selection`; :mod:`reflex.gate` owns every decision.

ARCHITECTURE (spec 6.4)
-----------------------
::

    context string --> encoder --------> pooled (d)
                                            |
                                     query MLP d->512->256 (GELU)
                                            |
                                          q (256)
            +-------------+-------------+---+-----+-------------+-------------+
            |             |             |         |             |             |
           H1            H2            H3        H4            H5            H6/H7

* **H1 nextstep** -- 3-way over :data:`~reflex.schemas.NEXT_STEPS`. That ORDER IS
  NORMATIVE: the official ``utils/evaluate.py::cds_report`` branches on
  ``nextstep_label == 0/1/2`` meaning retrieve_utterance / take_action /
  end_conversation. :class:`_ReflexModel` re-derives the order from the ontology
  and raises :class:`~reflex.contracts.ContractViolation` if it disagrees.
* **H2 intent** -- 55-way over ``data.subflow_list(ontology)``.
* **H3 action** -- 30-way over ``data.action_list(ontology)`` (the flattened
  buttons).
* **H4 values** -- a pointer/classifier head over the SAME index space the
  official AST baseline uses, so that our value predictions can be handed to
  ``ast_report`` / ``cds_report`` unmodified. Concretely, reading
  ``utils/process.py``::

      prepare_value_labels(ontology)  ->  126 enumerable values (lowercased,
                                          de-duplicated, ontology file order)
      value_to_id(...)                ->  start_idx = len(value_list) = 126, and a
                                          copy target ``start_idx + tokens.index('<slot>')``
                                          into the filtered context-token list, which
                                          ``convert_context_tokens`` pads to 100.

  So the head is ``len(value_list) + model.value_context_len`` wide: columns
  ``[0, 126)`` are enumerable values and columns ``[126, 226)`` are copy
  positions in the context-token input. :meth:`_ReflexModel.value_target_index`
  reproduces ``value_to_id`` so that train/select/evaluate all agree on that
  index space.
* **H5 skeleton** -- softmax over ``[s.skeleton_id for s in bank.skeletons]``.
* **H6 act** -- 9-way over :data:`~reflex.schemas.ACT_INVENTORY`.
  **REPORTING ONLY (spec 6.4). Never feed ``act_logits`` into selection** -- the
  act of a turn is implied by the chosen skeleton position, and using H6 to pick
  it would make the skeleton head decorative.
* **H7 template** -- act-conditioned retrieval: ``q_i = W_act[act] @ q``,
  ``score = cosine(q_i, e_t)``. ``e_t`` is a trainable linear projection of a
  template feature that comes from a FROZEN COPY of the encoder, refreshed once
  per epoch (:meth:`_ReflexModel.sync_frozen_encoder` +
  :meth:`_ReflexModel.refresh_template_cache`). Freezing the expensive half and
  keeping the cheap projection trainable is what makes "refresh once per epoch"
  sound: the cached part is the part that is allowed to be stale.

MASKED LOSSES (spec 6.4)
------------------------
H3/H4 apply only on ``take_action`` turns, H5/H7 only on ``retrieve_utterance``
turns. :meth:`_ReflexModel.compute_losses` implements the masking from the gold
labels themselves -- a label ``< 0`` means "this head does not apply here", the
same convention ``utils/process.py`` uses (``action_id, value_id = -1, -1``) and
the same one ``cds_report`` uses (``num_turns_include_action = sum(bslot_label >=
0)``). It also cross-checks the masks against the gold nextstep so a mislabelled
example fails loudly instead of quietly training H3 on an agent turn.

WHAT LIVES IN CONFIG (spec 10)
------------------------------
Every model id, width and numeric knob is read from ``cfg``: ``model.encoder``,
``model.fallback_encoder``, ``model.small_encoder``, ``model.fallback_small_encoder``,
``model.query_mlp``, ``model.dropout``, ``model.pooling``, ``model.value_context_len``,
``model.template_max_len``, ``model.infonce_temperature``,
``model.add_slot_marker_tokens``, ``model.local_files_only``,
``train.checkpoint_dir``, ``train.loss_weights``, ``train.batch_size``. The only
bare numbers in this file are ``_NEG_INF_LOGIT`` (a masking sentinel) and
``_CHECKPOINT_FORMAT`` (a file-format version) -- neither is tunable.

DEVICE
------
Nothing here moves the model to a device. ``build_model`` and ``load_checkpoint``
return a CPU model; :mod:`reflex.train` (``runtime.train_device``) and
:mod:`reflex.select` (``runtime.device``) call ``.to(...)`` themselves.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import math
import os
import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from reflex import contracts
from reflex.config import get_dotted, require_filled, resolve_path
from reflex.contracts import ContractViolation, ReflexError
from reflex.schemas import ACT_INVENTORY, NEXT_STEPS, Bank

__all__ = [
    "build_encoder",
    "build_model",
    "save_checkpoint",
    "load_checkpoint",
]

#: Additive sentinel for masked-out logits. Finite (not ``-inf``) so that a
#: softmax over an entirely-masked row still yields numbers instead of NaN.
#: This is numerical plumbing, not a tunable threshold, so spec 10 does not
#: apply to it.
_NEG_INF_LOGIT = -1.0e9

#: Bumped whenever the checkpoint payload layout changes.
_CHECKPOINT_FORMAT = 1

#: Config keys tried in order for each ``size``. The first that loads wins and
#: the resolved id is reported so the manifest can record a silent swap.
_ENCODER_KEYS: dict[str, tuple[str, ...]] = {
    "base": ("model.encoder", "model.fallback_encoder"),
    "small": ("model.small_encoder", "model.fallback_small_encoder"),
}


# --------------------------------------------------------------------------- #
# Class orders derived from the data (never from a literal)
# --------------------------------------------------------------------------- #


def _data_or_fallback(name: str, ontology: dict[str, Any], fallback: Any) -> list[str]:
    """Return ``reflex.data.<name>(ontology)``, or ``fallback`` while data is a stub.

    Spec 4.1 owns the canonical class orders. :mod:`reflex.models` must not
    fork them -- a different flattening order silently renumbers every head. So
    we call the owner when it is implemented and only fall back while
    ``reflex.data`` is still re-exporting the frozen contract stub (detected
    exactly the way ``tests/test_contracts.py`` detects it: identity against
    :mod:`reflex.contracts`). The fallbacks below are
    ``utils/process.py::prepare_*_labels`` verbatim, which is what spec 4.1 must
    implement anyway.
    """
    from reflex import data as data_module

    impl = getattr(data_module, name, None)
    if impl is not None and impl is not getattr(contracts, name):
        return list(impl(ontology))
    return list(fallback(ontology))


def _fallback_subflows(ontology: dict[str, Any]) -> list[str]:
    """``prepare_intent_labels`` from the official ``utils/process.py``."""
    out: list[str] = []
    for _flow, subflows in ontology["intents"]["subflows"].items():
        out.extend(subflows)
    return out


def _fallback_actions(ontology: dict[str, Any]) -> list[str]:
    """``prepare_action_labels`` from the official ``utils/process.py``."""
    out: list[str] = []
    for _section, buttons in ontology["actions"].items():
        out.extend(buttons.keys())
    return out


def _fallback_nextsteps(ontology: dict[str, Any]) -> list[str]:
    """``prepare_nextstep_labels`` from the official ``utils/process.py``."""
    return list(ontology["next_steps"])


def _value_list(ontology: dict[str, Any]) -> list[str]:
    """``prepare_value_labels`` from the official ``utils/process.py``, verbatim.

    Including its quirk: the membership test is against the RAW value while the
    list stores the LOWERCASED one (the "remove exactly one instance of
    credit_card" comment). On ABCD v1.1 both readings give the same 126 entries,
    but the official form is reproduced so the index space cannot drift from the
    baseline's if the ontology ever changes.

    No spec Section 4 module owns this list, so :mod:`reflex.models` defines it
    once and republishes it as ``model.value_list`` and as checkpoint metadata
    ``value_list``. **train / select / evaluate must read it from there** rather
    than re-deriving it.
    """
    value_list: list[str] = []
    for _category, values in ontology["values"]["enumerable"].items():
        for val in values:
            if val not in value_list:
                value_list.append(val.lower())
    return value_list


def _enumerable_by_category(ontology: dict[str, Any]) -> dict[str, list[str]]:
    """Lowercased enumerable values per category (``BaseProcessor.prepare_labels``)."""
    return {
        category: [val.lower() for val in values]
        for category, values in ontology["values"]["enumerable"].items()
    }


def _value_by_action(ontology: dict[str, Any]) -> dict[str, list[str]]:
    """``action -> required value slots`` (``BaseProcessor.prepare_labels``)."""
    out: dict[str, list[str]] = {}
    for _section, actions in ontology["actions"].items():
        for action, targets in actions.items():
            out[action] = list(targets)
    return out


def _slot_marker_tokens(ontology: dict[str, Any]) -> list[str]:
    """The 11 ABCD ``<slot>`` markers, exactly as ``utils/load.py`` builds them.

    ``load_tokenizer`` does ``tokenizer.add_tokens([f'<{slot}>' ...])`` over
    ``ontology['values']['non_enumerable']``. Without this, ``<order_id>``
    shatters into ``<``, ``order``, ``_``, ``id``, ``>`` and H4's copy targets
    (``tokens.index('<order_id>')``) can never be hit.
    """
    non_enumerable = ontology["values"]["non_enumerable"]
    return [f"<{slot}>" for _category, slots in non_enumerable.items() for slot in slots]


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("\x00".join(ids).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 6.4  encoder
# --------------------------------------------------------------------------- #


def build_encoder(cfg: dict[str, Any], size: str = "base") -> Any:
    """Instantiate the context encoder (spec 6.4).

    Tries ``model.encoder``, falls back to ``model.fallback_encoder`` if the
    first will not load, and uses ``model.small_encoder`` (then
    ``model.fallback_small_encoder``) when ``size == "small"`` (ablation E3c).
    The fallback that actually fired must be recorded in the manifest -- a
    silent encoder swap would make two runs incomparable, so the resolved id is
    returned rather than hidden.

    The tokenizer returned here is the BASE tokenizer. The 11 ABCD ``<slot>``
    markers are added by :func:`build_model`, which is the function that has the
    ontology; it mutates this same tokenizer object and resizes the embedding
    matrix to match. **Callers that build a model must tokenize with
    ``model.tokenizer``, not with a separately built one.**

    Args:
        cfg: Resolved config.
        size: ``"base"`` or ``"small"``.

    Returns:
        ``(encoder, tokenizer, resolved_model_id)``.

    Raises:
        ValueError: on an unknown ``size``.
        ReflexError: if no candidate encoder for ``size`` could be loaded; the
            message lists every id tried and why it failed.
    """
    if size not in _ENCODER_KEYS:
        raise ValueError(f"unknown encoder size {size!r}; expected one of {sorted(_ENCODER_KEYS)}")

    from transformers import AutoModel, AutoTokenizer  # heavy; imported lazily

    local_only = bool(get_dotted(cfg, "model.local_files_only"))
    attempts: list[str] = []
    for key in _ENCODER_KEYS[size]:
        model_id = require_filled(cfg, key)
        if not isinstance(model_id, str) or not model_id:
            attempts.append(f"{key}={model_id!r}: not a model id")
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
            encoder = AutoModel.from_pretrained(model_id, local_files_only=local_only)
        except Exception as exc:  # noqa: BLE001 - any load failure means "try the fallback"
            attempts.append(f"{key}={model_id!r}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
            continue
        return encoder, tokenizer, model_id

    raise ReflexError(
        f"could not load any encoder for size={size!r}. Tried, in order:\n  "
        + "\n  ".join(attempts)
        + "\nSet model.encoder / model.fallback_encoder (or the small_* pair) in "
        "configs/default.yaml to something this machine can load."
    )


# --------------------------------------------------------------------------- #
# 6.4  the multi-task model
# --------------------------------------------------------------------------- #


class _ReflexModel(nn.Module):
    """Shared encoder + query MLP + heads H1-H7. Built by :func:`build_model`.

    Private on purpose: spec Section 4 gives :mod:`reflex.models` four public
    names and this is not one of them. Other modules receive instances from
    :func:`build_model` / :func:`load_checkpoint` and use the documented
    methods.

    Public surface (methods and attributes other modules may rely on):

    ``encode_context``, ``encode_templates``, ``score_templates``, ``forward``
        The four the contract promises.
    ``compute_losses``
        Masked multi-task loss (spec 6.4). Used by :mod:`reflex.train`.
    ``sync_frozen_encoder`` / ``refresh_template_cache``
        The once-per-epoch H7 refresh.
    ``value_candidate_tokens`` / ``value_target_index``
        ``utils/process.py::value_to_id`` reproduced, so H4's index space is the
        official one. Its copy tier does not read the gold value -- see the
        warning on the method.
    ``value_target_index_value_aware``
        The same index space, but the copy column resolved BY the gold value,
        plus the reason it resolved that way. NOT YET WIRED IN: nothing calls
        it, and every reported H4 number still comes from
        ``value_target_index`` via ``probes/response_labels.py``.
    ``tokenizer``
        The tokenizer WITH the ABCD ``<slot>`` markers. Always use this one.
    ``class_orders``
        ``{"next_steps", "subflows", "actions", "skeleton_ids", "acts",
        "value_list", "template_ids"}`` -- the canonical order of every head's
        classes, also embedded in the checkpoint.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        encoder: Any,
        tokenizer: Any,
        resolved_encoder: str,
        bank: Bank,
        ontology: dict[str, Any],
        size: str,
    ) -> None:
        super().__init__()

        # -- class orders, all derived from the data ------------------------- #
        next_steps = _data_or_fallback("nextstep_list", ontology, _fallback_nextsteps)
        if tuple(next_steps) != NEXT_STEPS:
            raise ContractViolation(
                "ontology['next_steps'] is not the normative order. "
                f"expected {list(NEXT_STEPS)}, got {next_steps}. "
                "utils/evaluate.py::cds_report branches on nextstep_label == 0/1/2 with "
                "exactly that meaning; re-ordering silently corrupts every cascading score."
            )
        self.next_steps: list[str] = list(next_steps)
        self.subflows: list[str] = _data_or_fallback("subflow_list", ontology, _fallback_subflows)
        self.actions: list[str] = _data_or_fallback("action_list", ontology, _fallback_actions)
        self.acts: list[str] = list(ACT_INVENTORY)
        self.value_list: list[str] = _value_list(ontology)
        self.enumerable: dict[str, list[str]] = _enumerable_by_category(ontology)
        self.value_by_action: dict[str, list[str]] = _value_by_action(ontology)

        self.skeleton_ids: list[str] = [s.skeleton_id for s in bank.skeletons]
        self.template_ids: list[str] = [t.template_id for t in bank.templates]
        self.template_texts: list[str] = [t.text_delex for t in bank.templates]
        act_index = {a: i for i, a in enumerate(self.acts)}
        unknown_acts = sorted({t.act for t in bank.templates} - set(act_index))
        if unknown_acts:
            raise ContractViolation(
                f"bank templates use acts outside ACT_INVENTORY: {unknown_acts}"
            )
        self.template_acts: list[int] = [act_index[t.act] for t in bank.templates]
        if not self.skeleton_ids:
            raise ContractViolation(
                "bank.skeletons is empty: H5 would have zero classes. Run `reflex compile` first."
            )
        if not self.template_ids:
            raise ContractViolation(
                "bank.templates is empty: H7 has nothing to retrieve. Run `reflex compile` first."
            )

        # -- config ----------------------------------------------------------- #
        query_dims = [int(x) for x in get_dotted(cfg, "model.query_mlp")]
        if not query_dims:
            raise ContractViolation("model.query_mlp must list at least one layer width")
        dropout_p = float(get_dotted(cfg, "model.dropout"))
        self.pooling: str = str(get_dotted(cfg, "model.pooling"))
        if self.pooling not in ("cls", "mean"):
            raise ContractViolation(f"model.pooling must be 'cls' or 'mean', got {self.pooling!r}")
        self.value_context_len: int = int(get_dotted(cfg, "model.value_context_len"))
        self.template_max_len: int = int(get_dotted(cfg, "model.template_max_len"))
        self.infonce_temperature: float = float(get_dotted(cfg, "model.infonce_temperature"))
        if self.infonce_temperature <= 0:
            raise ContractViolation("model.infonce_temperature must be > 0")
        self.loss_weights: dict[str, float] = {
            k: float(v) for k, v in dict(get_dotted(cfg, "train.loss_weights")).items()
        }
        self._default_batch_size = int(get_dotted(cfg, "train.batch_size"))

        # -- encoder + tokenizer ---------------------------------------------- #
        self.tokenizer = tokenizer
        self.resolved_encoder = resolved_encoder
        self.size = size
        self.source_fraction = float(bank.source_fraction)
        self.bank_hash = str(bank.bank_hash)

        self.slot_marker_tokens: list[str] = []
        if bool(get_dotted(cfg, "model.add_slot_marker_tokens")):
            markers = _slot_marker_tokens(ontology)
            added = tokenizer.add_tokens(markers)
            self.slot_marker_tokens = list(markers)
            if added:
                # Newly created embedding rows are randomly initialized -> seed
                # the RNG before build_model if you need bit-identical builds.
                encoder.resize_token_embeddings(len(tokenizer))

        self.encoder = encoder
        # The frozen twin used for template features (spec 6.4: "template
        # embeddings come from a FROZEN copy of the encoder, refreshed once per
        # epoch"). Excluded from the checkpoint and re-synced on load.
        self.frozen_encoder = copy.deepcopy(encoder)
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad_(False)

        hidden = int(encoder.config.hidden_size)
        self.hidden_size = hidden
        self._accepts_token_type_ids = (
            "token_type_ids" in inspect.signature(encoder.forward).parameters
            and bool(getattr(encoder.config, "type_vocab_size", 0))
        )
        self._warned_token_type_ids = False

        # -- query MLP: d -> 512 -> 256, GELU between layers -------------------- #
        layers: list[nn.Module] = []
        dims = [hidden] + query_dims
        for i in range(len(query_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(query_dims) - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout_p))
        self.query_mlp = nn.Sequential(*layers)
        self.query_dim = query_dims[-1]
        self.dropout = nn.Dropout(dropout_p)

        # -- heads -------------------------------------------------------------- #
        self.h1_nextstep = nn.Linear(self.query_dim, len(self.next_steps))
        self.h2_intent = nn.Linear(self.query_dim, len(self.subflows))
        self.h3_action = nn.Linear(self.query_dim, len(self.actions))
        self.h4_value_enumerable = nn.Linear(self.query_dim, len(self.value_list))
        self.h4_value_copy = nn.Linear(self.query_dim, hidden)
        self.h5_skeleton = nn.Linear(self.query_dim, len(self.skeleton_ids))
        self.h6_act = nn.Linear(self.query_dim, len(self.acts))
        # H7: one act-conditioned projection per act. Identity init means the
        # model starts act-agnostic (q_i == q) and learns the conditioning.
        self.h7_act_projection = nn.Parameter(
            torch.eye(self.query_dim).unsqueeze(0).repeat(len(self.acts), 1, 1)
        )
        self.h7_template_projection = nn.Linear(hidden, self.query_dim)

        #: Frozen-encoder template features, (n_templates, d). Non-persistent:
        #: rebuilt by refresh_template_cache, never written to the checkpoint.
        self.register_buffer(
            "template_features", torch.zeros(0, hidden), persistent=False
        )

    # ---------------------------------------------------------------- orders #

    @property
    def class_orders(self) -> dict[str, list[str]]:
        """Canonical class order of every head. Mirrored into the checkpoint."""
        return {
            "next_steps": list(self.next_steps),
            "subflows": list(self.subflows),
            "actions": list(self.actions),
            "skeleton_ids": list(self.skeleton_ids),
            "acts": list(self.acts),
            "value_list": list(self.value_list),
            "template_ids": list(self.template_ids),
        }

    @property
    def value_head_size(self) -> int:
        """``len(value_list) + model.value_context_len`` -- the official AST width."""
        return len(self.value_list) + self.value_context_len

    # --------------------------------------------------------------- encoding #

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0]
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def _run_encoder(
        self,
        encoder: Any,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"attention_mask": attention_mask}
        if token_type_ids is not None:
            if self._accepts_token_type_ids:
                kwargs["token_type_ids"] = token_type_ids
            elif not self._warned_token_type_ids:
                self._warned_token_type_ids = True
                warnings.warn(
                    f"encoder {self.resolved_encoder!r} takes no token_type_ids; the speaker "
                    "segment ids are being ignored. Encode the speaker in the context string "
                    "instead (reflex.data.build_context).",
                    RuntimeWarning,
                    stacklevel=3,
                )
        return encoder(input_ids=input_ids, **kwargs).last_hidden_state

    def encode_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of context strings.

        Args:
            input_ids: ``(B, L)`` from ``model.tokenizer``.
            attention_mask: ``(B, L)``.
            token_type_ids: ``(B, L)`` speaker segments. Ignored (with one
                warning) by encoders that have no segment embeddings, which
                includes ModernBERT.
            return_hidden: also return the ``(B, L, d)`` token states.

        Returns:
            ``{"pooled": (B, d), "query": (B, query_dim)}``, plus ``"hidden"``
            when asked. ``query`` is the vector H1-H7 read and the one
            :mod:`reflex.calibrate` should index for novelty.
        """
        hidden_states = self._run_encoder(self.encoder, input_ids, attention_mask, token_type_ids)
        pooled = self._pool(hidden_states, attention_mask)
        query = self.query_mlp(self.dropout(pooled))
        out = {"pooled": pooled, "query": query}
        if return_hidden:
            out["hidden"] = hidden_states
        return out

    @torch.no_grad()
    def encode_templates(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed templates with the FROZEN encoder copy (spec 6.4).

        Returns ``(B, d)`` pooled features with no grad. These are the features
        the once-per-epoch cache stores; the trainable part of the template
        embedding is :attr:`h7_template_projection`, applied later by
        :meth:`score_templates`, so gradients still reach H7 between refreshes.
        """
        was_training = self.frozen_encoder.training
        self.frozen_encoder.eval()
        hidden_states = self._run_encoder(self.frozen_encoder, input_ids, attention_mask, None)
        pooled = self._pool(hidden_states, attention_mask)
        if was_training:  # pragma: no cover - the frozen twin is kept in eval
            self.frozen_encoder.train()
        return pooled

    def sync_frozen_encoder(self) -> None:
        """Copy the live encoder's weights into the frozen twin.

        Call once per epoch, immediately before :meth:`refresh_template_cache`.
        Between calls the template features are deliberately stale: that is what
        "frozen copy, refreshed once per epoch" means.
        """
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad_(False)

    def refresh_template_cache(
        self,
        texts: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Re-embed every bank template with the frozen encoder; cache the result.

        Args:
            texts: Defaults to the bank's ``text_delex`` in ``template_ids``
                order. Pass explicitly only to embed a different surface form.
            batch_size: Defaults to ``train.batch_size``.

        Returns:
            ``(n_templates, d)`` features, also stored on
            :attr:`template_features`.
        """
        texts = list(self.template_texts if texts is None else texts)
        if len(texts) != len(self.template_ids):
            raise ContractViolation(
                f"refresh_template_cache got {len(texts)} texts for {len(self.template_ids)} templates"
            )
        step = int(batch_size or self._default_batch_size)
        device = next(self.parameters()).device
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), step):
            batch = texts[start : start + step]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.template_max_len,
                return_tensors="pt",
            )
            chunks.append(
                self.encode_templates(
                    encoded["input_ids"].to(device),
                    encoded.get("attention_mask", None).to(device)
                    if encoded.get("attention_mask", None) is not None
                    else None,
                )
            )
        features = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, self.hidden_size, device=device)
        self.template_features = features
        return features

    # ------------------------------------------------------------------- H7 #

    def score_templates(
        self,
        query: torch.Tensor,
        act_ids: torch.Tensor,
        template_features: torch.Tensor,
    ) -> torch.Tensor:
        """``cosine(W_act[act] @ q, e_t)`` -- the H7 score (spec 6.4).

        Args:
            query: ``(B, query_dim)`` from :meth:`encode_context`.
            act_ids: ``(B,)`` or ``(B, P)`` act indices into
                :data:`~reflex.schemas.ACT_INVENTORY` (``P`` = act positions of a
                skeleton).
            template_features: FROZEN-encoder features, one of
                ``(T, d)`` (score every template, e.g. the whole bank or the
                in-batch golds), ``(B, C, d)`` (per-example candidates) or
                ``(B, P, C, d)`` (per-position candidates).

        Returns:
            Cosine similarities in ``[-1, 1]``: ``(B, T)`` / ``(B, P, T)`` /
            ``(B, C)`` / ``(B, P, C)`` respectively. Divide by
            ``model.infonce_temperature`` to get logits; :meth:`forward` does.
        """
        if query.dim() != 2:
            raise ContractViolation(f"query must be (B, query_dim), got {tuple(query.shape)}")
        projections = self.h7_act_projection[act_ids]  # (B, q, q) or (B, P, q, q)
        if act_ids.dim() == 1:
            conditioned = torch.einsum("bij,bj->bi", projections, query)
        elif act_ids.dim() == 2:
            conditioned = torch.einsum("bpij,bj->bpi", projections, query)
        else:
            raise ContractViolation(f"act_ids must be (B,) or (B, P), got {tuple(act_ids.shape)}")
        conditioned = F.normalize(conditioned, dim=-1)

        embedded = F.normalize(self.h7_template_projection(template_features), dim=-1)
        if embedded.dim() == 2:  # (T, d) -> score against a shared pool
            return conditioned @ embedded.t()
        if embedded.dim() == 3 and conditioned.dim() == 2:  # (B, C, d)
            return torch.einsum("bq,bcq->bc", conditioned, embedded)
        if embedded.dim() == 4 and conditioned.dim() == 3:  # (B, P, C, d)
            return torch.einsum("bpq,bpcq->bpc", conditioned, embedded)
        raise ContractViolation(
            f"template_features shape {tuple(template_features.shape)} does not match "
            f"act_ids shape {tuple(act_ids.shape)}"
        )

    # -------------------------------------------------------------- forward #

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run the encoder and every head.

        Args:
            batch: keys, all optional except the first two --

                ``input_ids`` ``(B, L)``, ``attention_mask`` ``(B, L)``
                    The context string.
                ``token_type_ids`` ``(B, L)``
                    Speaker segments, if the encoder supports them.
                ``context_input_ids`` / ``context_attention_mask`` ``(B, Lc)``
                    The official AST second input: the filtered context tokens
                    H4 copies values out of. ``Lc <= model.value_context_len``.
                ``act_ids`` ``(B,)`` or ``(B, P)``
                    Which act each H7 query position is conditioned on.
                ``template_features`` ``(B, C, d)`` / ``(B, P, C, d)`` / ``(T, d)``
                    Candidate features, gold at index 0 by the training
                    convention :meth:`compute_losses` expects. Omit to skip H7.

        Returns:
            ``{"pooled", "query", "nextstep_logits" (B,3), "intent_logits" (B,55),
            "action_logits" (B,30), "value_logits" (B, 126+100),
            "skeleton_logits" (B,S), "act_logits" (B,9)}`` plus
            ``"template_cosine"`` / ``"template_logits"`` when H7 inputs are
            present.

            ``act_logits`` is H6: **reporting only** (spec 6.4). Selection must
            not read it.
        """
        encoded = self.encode_context(
            batch["input_ids"],
            batch.get("attention_mask"),
            batch.get("token_type_ids"),
        )
        query = encoded["query"]
        out: dict[str, torch.Tensor] = {
            "pooled": encoded["pooled"],
            "query": query,
            "nextstep_logits": self.h1_nextstep(query),
            "intent_logits": self.h2_intent(query),
            "action_logits": self.h3_action(query),
            "skeleton_logits": self.h5_skeleton(query),
            "act_logits": self.h6_act(query),
        }
        out["value_logits"] = self._value_logits(
            query,
            batch.get("context_input_ids"),
            batch.get("context_attention_mask"),
        )

        template_features = batch.get("template_features")
        act_ids = batch.get("act_ids")
        if template_features is not None and act_ids is not None:
            cosine = self.score_templates(query, act_ids, template_features)
            out["template_cosine"] = cosine
            out["template_logits"] = cosine / self.infonce_temperature
        return out

    def _value_logits(
        self,
        query: torch.Tensor,
        context_input_ids: Optional[torch.Tensor],
        context_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """H4 over ``[enumerable values | copy positions]`` (official AST layout)."""
        batch_size = query.shape[0]
        enumerable_logits = self.h4_value_enumerable(query)
        copy_logits = query.new_full((batch_size, self.value_context_len), _NEG_INF_LOGIT)

        if context_input_ids is not None:
            length = int(context_input_ids.shape[1])
            if length > self.value_context_len:
                raise ContractViolation(
                    f"context_input_ids has {length} positions but model.value_context_len is "
                    f"{self.value_context_len}; the official utils/process.py pads this input to "
                    "exactly that width and H4's copy indices are positions in it."
                )
            hidden_states = self._run_encoder(
                self.encoder, context_input_ids, context_attention_mask, None
            )
            pointer = self.h4_value_copy(query)  # (B, d)
            scores = torch.einsum("bd,bld->bl", pointer, hidden_states) / math.sqrt(self.hidden_size)
            if context_attention_mask is not None:
                scores = scores.masked_fill(context_attention_mask == 0, _NEG_INF_LOGIT)
            copy_logits = torch.cat(
                [scores, copy_logits[:, length:]], dim=1
            ) if length < self.value_context_len else scores
        return torch.cat([enumerable_logits, copy_logits], dim=1)

    # --------------------------------------------------------------- losses #

    def compute_losses(
        self,
        batch: dict[str, Any],
        outputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Masked multi-task loss (spec 6.4). :mod:`reflex.train` owns the loop.

        Masking rule, straight from the spec: **H3/H4 only on take_action turns,
        H5/H7 only on retrieve_utterance turns.** A gold label ``< 0`` marks a
        row the head does not apply to -- the same ``-1`` convention
        ``utils/process.py`` writes and ``cds_report`` reads. The masks are
        cross-checked against the gold nextstep, so an example builder that
        labels an agent turn with an action fails loudly here.

        Args:
            batch: ``nextstep_labels`` and ``intent_labels`` ``(B,)`` are
                required; ``action_labels``, ``value_labels``,
                ``skeleton_labels``, ``act_labels`` ``(B,)`` and the H7 inputs
                are optional. H7 additionally reads
                ``template_features`` ``(B, C, d)`` with the GOLD AT INDEX 0,
                ``act_ids`` ``(B,)``, and optionally
                ``template_gold_index`` ``(B,)`` (bank indices, used to mask
                in-batch duplicates of the gold) and
                ``template_mask`` ``(B,)`` (bool, defaults to
                ``skeleton_labels >= 0``).
            outputs: Reuse a previous :meth:`forward` result instead of
                recomputing it. Must carry ``"query"``; every head including H7
                reads that one query, so all seven share a single encoder pass
                and (in training) a single dropout sample.

        Returns:
            ``{"nextstep", "intent", "action", "values", "skeleton", "template",
            "act", "total"}``. Heads with no applicable row contribute an exact
            zero that still carries a gradient path. ``total`` applies
            ``train.loss_weights``.
        """
        outputs = self.forward(batch) if outputs is None else outputs
        nextstep_labels = batch["nextstep_labels"]
        take_action = self.next_steps.index("take_action")
        retrieve = self.next_steps.index("retrieve_utterance")

        losses: dict[str, torch.Tensor] = {}
        losses["nextstep"] = _masked_cross_entropy(outputs["nextstep_logits"], nextstep_labels)
        losses["intent"] = _masked_cross_entropy(outputs["intent_logits"], batch["intent_labels"])

        action_labels = batch.get("action_labels")
        _check_mask_alignment("action", action_labels, nextstep_labels, take_action)
        losses["action"] = _masked_cross_entropy(outputs["action_logits"], action_labels)

        value_labels = batch.get("value_labels")
        _check_mask_alignment("values", value_labels, nextstep_labels, take_action)
        losses["values"] = _masked_cross_entropy(outputs["value_logits"], value_labels)

        skeleton_labels = batch.get("skeleton_labels")
        _check_mask_alignment("skeleton", skeleton_labels, nextstep_labels, retrieve)
        losses["skeleton"] = _masked_cross_entropy(outputs["skeleton_logits"], skeleton_labels)

        # H6 is reporting-only but still trained: it is what makes the act
        # classifier worth reporting in the first place (spec 6.2 check).
        losses["act"] = _masked_cross_entropy(outputs["act_logits"], batch.get("act_labels"))

        losses["template"] = self._template_loss(
            batch, outputs, skeleton_labels, nextstep_labels, retrieve
        )

        total = None
        for name, value in losses.items():
            weight = self.loss_weights.get(name)
            if weight is None:
                raise ContractViolation(f"train.loss_weights has no weight for head {name!r}")
            term = value * weight
            total = term if total is None else total + term
        losses["total"] = total if total is not None else torch.zeros((), device=nextstep_labels.device)
        return losses

    def _template_loss(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        skeleton_labels: Optional[torch.Tensor],
        nextstep_labels: torch.Tensor,
        retrieve: int,
    ) -> torch.Tensor:
        """InfoNCE over in-batch negatives + ``model.hard_negatives_per_positive``.

        Candidate layout produced by :func:`reflex.train.collate_batch`:
        ``template_features[b, 0]`` is b's gold and ``template_features[b, 1:]``
        are its hard negatives OF THE SAME ACT. The negative pool for row ``b``
        is therefore every other row's gold (in-batch) plus its own K hard
        negatives, exactly as spec 6.4 asks.

        The query is taken from ``outputs``, never re-encoded. Re-encoding would
        cost a third full encoder pass per step AND -- because dropout is live in
        training -- would train H7 against a query sample that H1-H6 never saw.
        """
        features = batch.get("template_features")
        act_ids = batch.get("act_ids")
        if features is None or act_ids is None:
            return torch.zeros((), device=nextstep_labels.device)
        if features.dim() != 3:
            raise ContractViolation(
                "compute_losses expects template_features of shape (B, 1+K, d) with the gold "
                f"at index 0, got {tuple(features.shape)}"
            )

        applicable = batch.get("template_mask")
        if applicable is None:
            applicable = (
                skeleton_labels >= 0
                if skeleton_labels is not None
                else nextstep_labels == retrieve
            )
        applicable = applicable.bool()
        if not bool(applicable.any()):
            return features.sum() * 0.0

        query = outputs["query"]

        gold_features = features[:, 0, :]  # (B, d)
        in_batch = self.score_templates(query, act_ids, gold_features)  # (B, B)
        hard = self.score_templates(query, act_ids, features[:, 1:, :])  # (B, K)

        batch_size = in_batch.shape[0]
        eye = torch.eye(batch_size, dtype=torch.bool, device=in_batch.device)
        # Columns belonging to rows the head does not apply to are not valid
        # negatives (their "gold" is a placeholder).
        dead_columns = (~applicable).unsqueeze(0).expand(batch_size, batch_size) & ~eye
        in_batch = in_batch.masked_fill(dead_columns, _NEG_INF_LOGIT * self.infonce_temperature)

        gold_index = batch.get("template_gold_index")
        if gold_index is not None:
            # A different row whose gold IS this row's gold is a false negative.
            duplicate = (gold_index.unsqueeze(0) == gold_index.unsqueeze(1)) & ~eye
            in_batch = in_batch.masked_fill(duplicate, _NEG_INF_LOGIT * self.infonce_temperature)

        logits = torch.cat([in_batch, hard], dim=1) / self.infonce_temperature
        targets = torch.arange(batch_size, device=logits.device)
        return F.cross_entropy(logits[applicable], targets[applicable])

    # ------------------------------------------- official AST value helpers #

    def value_candidate_tokens(self, context_texts: list[str], action: str) -> list[str]:
        """The context tokens H4 may copy from -- ``value_to_id``'s filter, verbatim.

        From ``utils/process.py::BaseProcessor.value_to_id``: tokenize each
        context utterance, keep tokens longer than 2 characters, de-duplicate
        preserving first-occurrence order, then keep the LAST
        ``value_context_len - (len(tokenize(action)) + 3)`` of them.

        Args:
            context_texts: Utterance texts WITHOUT the ``speaker|`` prefix, in
                chronological order.
            action: The button name (``"verify-identity a"`` style position
                suffixes included, as the official code passes them).

        Returns:
            The token list. Its indices are H4's copy targets: global column
            ``len(value_list) + i``.

        Note:
            The official code then prepends ``[CLS]`` when embedding this list
            (``convert_context_tokens``) WITHOUT shifting the copy index, so the
            baseline's own decoding in ``evaluate.qualify`` is off by one. We
            keep the index space (so labels are comparable) but score position
            ``i`` of THIS list, i.e. without the off-by-one.
        """
        filtered: list[str] = []
        for text in context_texts:
            for token in self.tokenizer.tokenize(text):
                if token in filtered:
                    continue
                if len(token) > 2:
                    filtered.append(token)
        effective_max = self.value_context_len - (len(self.tokenizer.tokenize(action)) + 3)
        return filtered[-effective_max:]

    def value_target_index(
        self,
        context_texts: list[str],
        action: str,
        value: str,
        potential_vals: Optional[list[str]] = None,
    ) -> tuple[int, list[str]]:
        """``utils/process.py::value_to_id`` reproduced. Returns ``(target_id, tokens)``.

        ``target_id`` is ``-1`` when the value is neither an enumerable value of
        one of the action's slots nor a ``<slot>`` marker present in the
        context tokens -- the official processor DROPS those examples (AST) or
        stores ``-1`` (CDS).

        Args:
            context_texts: As for :meth:`value_candidate_tokens`.
            action: Button name, possibly with the ``" a"/" b"/" c"`` position
                suffix the official code appends for ``verify-identity`` and
                ``validate-purchase``.
            value: The gold value string.
            potential_vals: ``value_by_action[action]``; looked up when omitted,
                stripping any position suffix first.

        Warning:
            THE COPY TIER OF THIS FUNCTION IS NOT A FUNCTION OF ``value``. The
            ``else`` branch accepts the FIRST ``potential_vals`` entry whose
            marker is present, whatever the gold value is, so for any action
            with two or more non-enumerable slots (verify-identity,
            validate-purchase, record-reason, enter-details, update-order,
            update-account) the label is decided by ontology order and the
            context alone. That is faithful -- ``utils/process.py::value_to_id``
            does exactly this, and :mod:`reflex.train` needs the official index
            space for its labels -- but it is NOT a correct gold label. Measured
            on ``test_seen`` with the real ModernBERT-base tokenizer and the
            shipped config (2,372 valued take_action turns, 2,193 resolvable --
            the denominators ``outputs/probes/response/select.json`` publishes):
            489 rows resolve through the copy tier and 284 of those have two or
            more candidate markers present, i.e. the label is decided by
            ontology order, and some are provably wrong (convo 777
            ``validate-purchase``, gold ``rodriguezdomingo525@email.com``,
            labelled at ``<username>``'s column; convo 3259 ``verify-identity``,
            gold ``376-285-0809``, a phone, labelled at ``<zip_code>``'s).

            THIS IS STILL THE FUNCTION EVERY REPORTED H4 NUMBER COMES FROM.
            :meth:`value_target_index_value_aware` resolves by the value and
            says how, but nothing calls it yet: the reporting path is
            ``probes/response_labels.py`` (``H4TargetIndexer.index`` calls this
            method, and ``_mirror_index`` / ``verify_against_model`` reproduce
            and check this same ordering). Until that call site changes, H4's
            published gold column carries the label error described here.
        """
        if potential_vals is None:
            base_action = action.split(" ")[0]
            potential_vals = self.value_by_action.get(base_action, [])
        tokens = self.value_candidate_tokens(context_texts, action)
        target_id = -1
        for option in potential_vals:
            if option in self.enumerable:
                if value in self.enumerable[option]:
                    target_id = self.value_list.index(value)
            else:
                marker = f"<{option}>"
                if marker in tokens:
                    target_id = len(self.value_list) + tokens.index(marker)
            if target_id >= 0:
                break
        return target_id, tokens

    #: How :meth:`value_target_index_value_aware` resolved a row. The first two
    #: are decided BY THE GOLD VALUE; ``copy_unambiguous`` is decided by the
    #: context but could not have been decided otherwise (one candidate); the
    #: rest resolve to ``-1`` and are unscorable, each for a different reason.
    VALUE_RESOLUTIONS: tuple[str, ...] = (
        "enumerable",
        "copy_typed",
        "copy_unambiguous",
        "copy_ambiguous",
        "copy_typed_marker_absent",
        "copy_typed_not_an_action_slot",
        "miss",
    )

    def value_target_index_value_aware(
        self,
        context_texts: list[str],
        action: str,
        value: str,
        potential_vals: Optional[list[str]] = None,
    ) -> tuple[int, list[str], str]:
        """H4's gold column resolved BY THE VALUE. ``(target_id, tokens, resolution)``.

        Same index space as :meth:`value_target_index` -- an enumerable hit is
        ``value_list.index(value)``, a copy hit is
        ``len(value_list) + tokens.index(marker)`` -- so the two are directly
        comparable. What differs is only WHICH column the copy tier picks, and
        it is picked by typing the gold value rather than by ontology order:

        * the value is an enumerable value of one of the action's slots ->
          ``enumerable`` (unchanged; that tier already read the value);
        * the value's SHAPE types it to one of the action's non-enumerable slots
          (``compile._type_literal``, the same evidence the compiler uses) and
          that slot's marker is present -> ``copy_typed``;
        * it types to a slot of this action whose marker is absent ->
          ``copy_typed_marker_absent``, ``-1``: the value is not copyable here;
        * it types to a slot this action does not even have ->
          ``copy_typed_not_an_action_slot``, ``-1`` (convo 3259
          ``verify-identity`` with a phone gold). ``_type_literal`` never
          answers outside the action's own slots, so this takes a second,
          candidate-free typing pass, with two carve-outs: a ``street_address``
          shape counts as ``full_address`` when that is the slot the action has
          (a full address begins with its street address), and a bare
          alphanumeric token is NOT refused when the action has a ``username``
          slot, because a username has no shape of its own;
        * it has no shape at all (``username``, ``security_answer``,
          ``details_slotval``) and exactly ONE candidate marker is present ->
          ``copy_unambiguous``: still context-decided, but ontology order did
          not decide it, so the label stands;
        * shapeless with two or more markers present -> ``copy_ambiguous``,
          ``-1``. This is the population :meth:`value_target_index` silently
          labels by list order; refusing it is what keeps a label that is not a
          function of the value out of a reported numerator. Report it as its
          own count rather than folding it into ``miss``. It is CONSERVATIVE,
          not a claim the old label was wrong: most of these rows are
          validate-purchase usernames, which ontology order happens to label at
          ``<username>``.

        STATUS: NOT WIRED IN. No caller uses this. It is meant as the REPORTING
        derivation -- deliberately not what :mod:`reflex.train` uses, since
        training labels must stay in the official AST index space, which is
        what :meth:`value_target_index` is for -- but the reporting path,
        ``probes/response_labels.py``, still calls :meth:`value_target_index`
        (``H4TargetIndexer.index``; its ``_mirror_index`` and
        ``verify_against_model`` would have to follow, asserting parity only
        where the resolution is ``enumerable``, ``copy_typed`` or
        ``copy_unambiguous``). Wiring it in CHANGES PUBLISHED NUMBERS and needs
        a probe re-run: measured on ``test_seen`` (real tokenizer, shipped
        config) it agrees with :meth:`value_target_index` on 2,068 of 2,372
        rows, moves 7 to a different column, and refuses 297 the other resolves
        (220 ``copy_ambiguous``, 48 ``copy_typed_marker_absent``, 29
        ``copy_typed_not_an_action_slot``), so H4's ``n_resolvable`` would go
        from 2,193 to 1,896.
        """
        from reflex.compile import _type_literal  # local: compile pulls in data

        if potential_vals is None:
            base_action = action.split(" ")[0]
            potential_vals = self.value_by_action.get(base_action, [])
        tokens = self.value_candidate_tokens(context_texts, action)
        n_values = len(self.value_list)

        for option in potential_vals:
            if option in self.enumerable and value in self.enumerable[option]:
                return self.value_list.index(value), tokens, "enumerable"

        copyable = [option for option in potential_vals if option not in self.enumerable]
        typed = _type_literal(str(value), list(potential_vals), self.enumerable)
        if not typed and copyable:
            # _type_literal only ever answers with one of the action's OWN
            # non-enumerable slots, so a value shaped like a slot this action
            # does not have comes back None and would fall through to the
            # shapeless branch below -- which is how a phone number entered
            # into verify-identity got labelled at <zip_code>. Ask again with no
            # candidate list, i.e. against every shape.
            foreign = _type_literal(str(value), [], self.enumerable)
            if foreign == "street_address" and "full_address" in copyable:
                # A full address BEGINS with its street address; the shape test
                # cannot tell them apart, and this action only has the one.
                typed = "full_address"
            elif foreign and foreign not in copyable:
                # A username has no shape of its own and a bare alphanumeric
                # token can be one ("sanyaa1253" has the account_id shape), so
                # that case stays shapeless rather than being refused.
                if not ("username" in copyable and str(value).strip().isalnum()):
                    return -1, tokens, "copy_typed_not_an_action_slot"
        if typed:
            if typed not in copyable:
                return -1, tokens, "copy_typed_not_an_action_slot"
            marker = f"<{typed}>"
            if marker not in tokens:
                return -1, tokens, "copy_typed_marker_absent"
            return n_values + tokens.index(marker), tokens, "copy_typed"

        present = [option for option in copyable if f"<{option}>" in tokens]
        if len(present) == 1:
            return n_values + tokens.index(f"<{present[0]}>"), tokens, "copy_unambiguous"
        if len(present) > 1:
            return -1, tokens, "copy_ambiguous"
        return -1, tokens, "miss"


def _masked_cross_entropy(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
    """Cross entropy over rows whose label is ``>= 0``; exact zero when none are.

    ``F.cross_entropy(..., ignore_index=-1)`` returns NaN for an all-ignored
    batch, which then poisons every other head through ``total``. Returning
    ``logits.sum() * 0`` keeps the value at zero and the graph connected.
    """
    if labels is None:
        return logits.sum() * 0.0
    valid = labels >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid])


def _check_mask_alignment(
    head: str,
    labels: Optional[torch.Tensor],
    nextstep_labels: torch.Tensor,
    expected_nextstep: int,
) -> None:
    """Fail loudly when a head is asked to train on a turn it does not apply to."""
    if labels is None:
        return
    wrong = (labels >= 0) & (nextstep_labels != expected_nextstep)
    if bool(wrong.any()):
        raise ContractViolation(
            f"{int(wrong.sum())} example(s) carry a gold {head} label on a turn whose gold "
            f"nextstep is not index {expected_nextstep}. Spec 6.4 masks H3/H4 to take_action and "
            "H5/H7 to retrieve_utterance; label the others -1."
        )


def build_model(cfg: dict[str, Any], bank: Bank, ontology: dict[str, Any], size: str = "base") -> Any:
    """Build the multi-task model: shared encoder, query MLP, heads H1-H7 (spec 6.4).

    Head output widths come from the DATA, never from a literal: H1 =
    ``len(NEXT_STEPS)``, H2 = ``len(subflow_list(ontology))``, H3 =
    ``len(action_list(ontology))``, H5 = ``len(bank.skeletons)``, H6 =
    ``len(ACT_INVENTORY)``. H4 is a scoring head over the official AST index
    space (``len(value_list) + model.value_context_len``) and H7 scores
    ``cosine(W_act[act] @ q, e_t)`` over per-turn candidate sets; template
    embeddings come from a FROZEN copy of the encoder, refreshed once per epoch.

    Also adds the 11 ABCD ``<slot>`` markers to the tokenizer and resizes the
    embedding matrix (``utils/load.py`` does the same), because H4's copy
    targets are token positions of those markers. Seed the RNG before calling if
    you need bit-identical new embedding rows.

    Args:
        cfg: Resolved config. Uses ``model.query_mlp`` and the ``model.*``
            additions documented at the top of this module.
        bank: Needed for the skeleton head width and the template inventory.
        ontology: Needed for the intent and action head widths.
        size: ``"base"`` or ``"small"``.

    Returns:
        An ``nn.Module`` exposing ``encode_context``, ``encode_templates``,
        ``score_templates`` and ``forward`` (plus ``compute_losses``,
        ``sync_frozen_encoder``, ``refresh_template_cache`` and the
        official-parity value helpers).

    Raises:
        ContractViolation: if the ontology's nextstep order is not the
            normative one, or the bank has no skeletons/templates.
    """
    encoder, tokenizer, resolved = build_encoder(cfg, size)
    return _ReflexModel(cfg, encoder, tokenizer, resolved, bank, ontology, size)


# --------------------------------------------------------------------------- #
# 6.4  checkpoint I/O
# --------------------------------------------------------------------------- #


def _checkpoint_name(seed: int, size: str, fraction: float) -> str:
    """``seed1.pt`` for the default build; suffixed when size/fraction differ."""
    name = f"seed{seed}"
    if size != "base":
        name += f"_{size}"
    if abs(fraction - 1.0) > 1e-9:
        name += f"_frac{fraction:g}"
    return name + ".pt"


def save_checkpoint(model: Any, cfg: dict[str, Any], seed: int, extra: Optional[dict[str, Any]] = None) -> str:
    """Save a training checkpoint under ``train.checkpoint_dir``.

    The checkpoint embeds everything needed to reproduce inference class orders
    -- resolved encoder id, subflow list, action list, skeleton ids, template
    ids, act inventory, the official value list, the seed -- so that a
    calibration produced against it cannot be silently paired with a different
    head ordering. :func:`load_checkpoint` re-derives those orders from the
    bank and ontology it is handed and refuses to load on a mismatch.

    The frozen encoder twin is NOT saved (it is a copy of the live encoder and
    doubles the file); :func:`load_checkpoint` re-syncs it. The template feature
    cache is not saved either -- it is stale by construction.

    Args:
        model: The trained model from :func:`build_model`.
        cfg: Resolved config.
        seed: The training seed.
        extra: Extra provenance to store alongside, under ``metadata["extra"]``.

    Returns:
        The absolute checkpoint path.
    """
    directory = resolve_path(cfg, "train.checkpoint_dir")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, _checkpoint_name(seed, model.size, model.source_fraction))

    state_dict = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("frozen_encoder.")
    }
    orders = model.class_orders
    metadata: dict[str, Any] = {
        "format_version": _CHECKPOINT_FORMAT,
        "seed": int(seed),
        "size": model.size,
        "resolved_encoder": model.resolved_encoder,
        "hidden_size": model.hidden_size,
        "query_dim": model.query_dim,
        "pooling": model.pooling,
        "value_context_len": model.value_context_len,
        "value_head_size": model.value_head_size,
        "infonce_temperature": model.infonce_temperature,
        "slot_marker_tokens": list(model.slot_marker_tokens),
        "tokenizer_len": len(model.tokenizer),
        "source_fraction": model.source_fraction,
        "bank_hash": model.bank_hash,
        "config_path": cfg.get("_config_path", ""),
        "overrides": list(cfg.get("_overrides", [])),
        "extra": dict(extra or {}),
    }
    metadata.update(orders)
    metadata["class_order_hashes"] = {name: _hash_ids(ids) for name, ids in orders.items()}

    torch.save({"state_dict": state_dict, "metadata": metadata}, path)
    return os.path.abspath(path)


def load_checkpoint(path: str, cfg: dict[str, Any], bank: Bank, ontology: dict[str, Any]) -> Any:
    """Load a checkpoint into a freshly built model, in eval mode.

    Args:
        path: Checkpoint path from :func:`save_checkpoint`.
        cfg: Resolved config.
        bank: Must be the same bank the checkpoint was trained against.
        ontology: The ontology.

    Returns:
        ``(model, checkpoint_metadata)``. The model is on CPU and in eval mode;
        its frozen encoder twin has been re-synced from the loaded weights, and
        its template feature cache is EMPTY -- call ``refresh_template_cache()``
        before scoring templates.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ContractViolation: if the checkpoint's stored class orders disagree with
            the ones derived from ``bank`` and ``ontology``, or if its weights
            do not fit the rebuilt model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if metadata.get("format_version") != _CHECKPOINT_FORMAT:
        raise ContractViolation(
            f"checkpoint format {metadata.get('format_version')!r} != {_CHECKPOINT_FORMAT} ({path})"
        )

    model = build_model(cfg, bank, ontology, size=metadata.get("size", "base"))

    mismatches: list[str] = []
    for name, ids in model.class_orders.items():
        stored = metadata.get(name)
        if stored is None:
            mismatches.append(f"{name}: absent from the checkpoint")
        elif list(stored) != list(ids):
            mismatches.append(
                f"{name}: checkpoint has {len(stored)} entries (hash "
                f"{_hash_ids(list(stored))}), this bank/ontology gives {len(ids)} "
                f"(hash {_hash_ids(ids)})"
            )
    if metadata.get("resolved_encoder") != model.resolved_encoder:
        mismatches.append(
            f"resolved_encoder: checkpoint {metadata.get('resolved_encoder')!r} != "
            f"loaded {model.resolved_encoder!r}"
        )
    if int(metadata.get("value_context_len", -1)) != model.value_context_len:
        mismatches.append(
            f"value_context_len: checkpoint {metadata.get('value_context_len')} != "
            f"{model.value_context_len}"
        )
    if mismatches:
        raise ContractViolation(
            f"checkpoint {path} does not match this bank/ontology/config:\n  "
            + "\n  ".join(mismatches)
            + "\nHead class ids would be renumbered; calibration and metrics would be garbage."
        )

    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    stray_missing = [key for key in missing if not key.startswith("frozen_encoder.")]
    if stray_missing or unexpected:
        raise ContractViolation(
            f"checkpoint {path} weights do not fit the rebuilt model. "
            f"missing={stray_missing[:5]} unexpected={list(unexpected)[:5]}"
        )
    model.sync_frozen_encoder()
    model.eval()
    return model, metadata
