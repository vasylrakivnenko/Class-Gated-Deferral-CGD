"""Spec 4.8 -- LLM agent: THE ONLY PLACE that calls an LLM at inference time.

PAID-CALL KILL SWITCH: fully implemented, but DISABLED whenever ``llm.enabled``
is false, in which case every entry point raises
:class:`~reflex.contracts.LLMDisabledError`. Zero paid API calls is a hard
project rule, including test calls.

STATUS: IMPLEMENTED. No call has ever been made from this build -- ``llm.enabled``
ships false, and :func:`build_llm_agent` / :func:`llm_decide` both refuse before
any prompt is rendered.

HOW THE KILL SWITCH IS LAYERED
------------------------------
1. ``llm.enabled: false`` (config default) -> :class:`LLMDisabledError` from BOTH
   :func:`build_llm_agent` (construction) and :func:`llm_decide` (call), so no
   code path can hold a stale handle and bypass the switch.
2. Even with ``llm.enabled: true``, ``llm.strong`` / ``llm.cheap`` /
   ``llm.provider`` / ``llm.wire_format`` / ``llm.api_base`` / ``llm.api_key_env``
   / ``llm.prices_usd_per_million`` / ``llm.price_list_date`` all ship as
   ``<fill: ...>``, so construction raises
   :class:`~reflex.contracts.PlaceholderConfigError` naming the key.
3. Even fully configured, ``llm.dry_run: true`` (config default) estimates tokens
   and cost and returns WITHOUT reaching a backend.
4. Only then does a backend exist, and it is a stdlib ``urllib`` client -- this
   module imports no vendor SDK, so there is nothing to accidentally
   auto-configure from an ambient API key.

The three token/cost entry points -- :func:`render_agent_prompt`,
:func:`estimate_prompt_tokens`, :func:`llm_cost_usd` -- are PURE and never touch
the network, which is how ``outputs/compile/armA_cost_estimate.md`` is produced
with the kill switch on and every model id unfilled.

Read ``cfg``, never a literal: spec 10 forbids any numeric threshold, model id,
path or price in code.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from reflex.config import get_dotted, require_filled, resolve_path
from reflex.contracts import (
    BudgetExceededError,
    LLMDisabledError,
    PlaceholderConfigError,
    ReflexError,
)
from reflex.schemas import ContextWindow, LLMDecision, NormalizedTurn

__all__ = [
    "load_prompt",
    "render_agent_prompt",
    "parse_llm_json",
    "build_llm_agent",
    "llm_decide",
    "estimate_prompt_tokens",
    "llm_cost_usd",
]

_LOG = logging.getLogger(__name__)

#: Dotted config key per ``which`` accepted by :func:`load_prompt`.
_PROMPT_KEYS: dict[str, str] = {
    "agent_A": "paths.prompt_agent_a",
    "act_labeling": "paths.prompt_act_labeling",
}

#: Section markers inside a prompt file. Everything after ``[SYSTEM]`` up to
#: ``[USER]`` is the system message; the rest is the user message.
_SYSTEM_MARKER = "[SYSTEM]"
_USER_MARKER = "[USER]"

#: Config keys whose ``<fill: ...>`` placeholders must be filled before a paid
#: call is possible. Checked together so the operator sees the whole list at once.
_REQUIRED_LLM_KEYS: tuple[str, ...] = (
    "llm.provider",
    "llm.wire_format",
    "llm.api_base",
    "llm.api_key_env",
    "llm.price_list_date",
)

_MODEL_KEYS: tuple[str, ...] = ("strong", "cheap")

#: Upper bound (exclusive) on a valid ``candidate_index``.
#:
#: SPEC 10 EXCEPTION, DELIBERATE AND NARROW. Every other number in this module
#: comes from ``cfg``, but :func:`parse_llm_json`'s signature is FROZEN as
#: ``(raw, ontology)`` -- it receives no ``cfg``, and the ontology does not carry
#: a candidate count. The bound is not a tunable threshold: it is the frozen
#: contract (``candidate_index``/``utt_rank`` in ``[-1, 100)``) and a verified
#: dataset invariant (exactly 100 candidates on every one of the 95,129 agent
#: turns, 0 on every other turn). Reported as a deviation in the module 4.8
#: hand-off notes.
_MAX_CANDIDATE_INDEX = 100

#: Hex characters kept from a sha256 digest. Matches the contract's
#: "sha256_hex_16" for prompt/context hashes.
_HASH_LEN = 16

_TOKENIZER_HEURISTIC = "heuristic"


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #


def _sha16(text: str) -> str:
    """Stable 16-hex-char sha256 of ``text`` (utf-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_LEN]


def _strip_comment_lines(text: str) -> str:
    """Drop ``#``-prefixed bookkeeping lines from a prompt file.

    Prompt files carry a provenance header (spec 6.9 step 3: revisions used,
    freeze status). Those lines are documentation, not instructions, and must not
    be billed or sent.
    """
    kept = [line for line in text.splitlines() if not line.startswith("#")]
    return "\n".join(kept).strip("\n")


def _require_no_placeholders(cfg: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """Resolve every key, collecting ALL unfilled ones into one error message."""
    values: dict[str, Any] = {}
    unfilled: list[str] = []
    for key in keys:
        try:
            values[key] = require_filled(cfg, key)
        except PlaceholderConfigError:
            unfilled.append(key)
    if unfilled:
        raise PlaceholderConfigError(
            "cannot build the LLM agent: these config keys are still "
            f"`<fill: ...>` placeholders: {', '.join(unfilled)}. "
            "Fill them in configs/default.yaml (or pass --set key=value). "
            "Do not guess a value -- a wrong model id or price silently "
            "corrupts every cost number in Section 8."
        )
    return values


# --------------------------------------------------------------------------- #
# load_prompt
# --------------------------------------------------------------------------- #


def load_prompt(cfg: dict[str, Any], which: str) -> tuple[str, str]:
    """Load a frozen prompt file and its hash.

    See :func:`reflex.contracts.load_prompt` for the frozen contract.

    The hash covers the COMMENT-STRIPPED text, i.e. exactly the bytes that can
    reach a model. Editing the ``#`` provenance header (bumping the revision
    count, recording the freeze) therefore does not change the recorded hash,
    while changing one word of an instruction does.
    """
    if which not in _PROMPT_KEYS:
        raise ValueError(
            f"unknown prompt {which!r}; expected one of {sorted(_PROMPT_KEYS)}"
        )
    path = resolve_path(cfg, _PROMPT_KEYS[which])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prompt file for {which!r} not found: {path} "
            f"(config key {_PROMPT_KEYS[which]!r})"
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    text = _strip_comment_lines(raw)
    if not text:
        raise ValueError(f"prompt file {path} has no content outside '#' comments")
    return text, _sha16(text)


# --------------------------------------------------------------------------- #
# render_agent_prompt and its renderers
# --------------------------------------------------------------------------- #


def _split_sections(prompt_template: str) -> tuple[str, str]:
    """Split a prompt file into ``(system_template, user_template)``."""
    if _SYSTEM_MARKER not in prompt_template or _USER_MARKER not in prompt_template:
        raise ValueError(
            f"prompt template must contain {_SYSTEM_MARKER} and {_USER_MARKER} markers"
        )
    _, _, rest = prompt_template.partition(_SYSTEM_MARKER)
    system, _, user = rest.partition(_USER_MARKER)
    if prompt_template.count(_USER_MARKER) != 1 or prompt_template.count(_SYSTEM_MARKER) != 1:
        raise ValueError("prompt template must contain exactly one of each section marker")
    return system.strip("\n"), user.strip("\n")


def _flatten_subflows(ontology: dict[str, Any]) -> list[str]:
    """``ontology['intents']['subflows']`` (flow -> [subflow]) flattened, file order."""
    subflows = ontology.get("intents", {}).get("subflows", {})
    out: list[str] = []
    if isinstance(subflows, dict):
        for values in subflows.values():
            out.extend(values if isinstance(values, (list, tuple)) else [values])
    else:
        out.extend(subflows)
    return out


def _flatten_actions(ontology: dict[str, Any]) -> dict[str, tuple[str, list[str]]]:
    """``action -> (category, required_slots)`` over all ontology action categories."""
    out: dict[str, tuple[str, list[str]]] = {}
    for category, actions in ontology.get("actions", {}).items():
        if isinstance(actions, dict):
            for action, slots in actions.items():
                out[action] = (category, list(slots) if slots else [])
        else:
            for action in actions:
                out[action] = (category, [])
    return out


def _render_guidelines(guidelines: dict[str, Any]) -> str:
    """Render ``guidelines.json`` (10 flows / 55 subflows) to deterministic text.

    Spec 6.9 step 1 puts the company guidelines in the system block. The JSON is
    ~90 KB; rendering it as prose rather than dumping JSON is both cheaper in
    tokens and easier for a model to follow. File order is preserved so the
    rendered string is byte-identical on every call, which is what makes
    provider-side prompt caching possible (see
    ``outputs/compile/armA_cost_estimate.md``).
    """
    lines: list[str] = []
    for flow, body in guidelines.items():
        description = body.get("description", "") if isinstance(body, dict) else ""
        lines.append(f"## FLOW: {flow}" + (f" -- {description}" if description else ""))
        subflows = body.get("subflows", {}) if isinstance(body, dict) else {}
        for subflow, detail in subflows.items():
            lines.append(f"### SUBFLOW: {subflow}")
            for instruction in detail.get("instructions", []) or []:
                lines.append(f"- {instruction}")
            for action in detail.get("actions", []) or []:
                button = action.get("button", "")
                kind = action.get("type", "")
                text = action.get("text", "")
                lines.append(f"- [{button}] ({kind}) {text}")
                for sub in action.get("subtext", []) or []:
                    lines.append(f"    * {sub}")
        lines.append("")
    return "\n".join(lines).strip("\n")


def _render_action_list(ontology: dict[str, Any]) -> str:
    """``action_name (category): slot, slot`` for every button, ontology order."""
    lines = []
    for action, (category, slots) in _flatten_actions(ontology).items():
        suffix = ": " + ", ".join(slots) if slots else ": (no values)"
        lines.append(f"- {action} ({category}){suffix}")
    return "\n".join(lines)


def _render_kb(kb: dict[str, Any]) -> str:
    """``subflow -> expected action sequence`` from ``kb.json``, file order."""
    lines = []
    for subflow, actions in kb.items():
        rendered = ", ".join(actions) if isinstance(actions, (list, tuple)) else str(actions)
        lines.append(f"- {subflow}: {rendered}")
    return "\n".join(lines)


def _render_candidate_block(candidate_texts: Sequence[str]) -> str:
    """Number the candidates 0..N-1 so ``candidate_index`` is a RANK.

    The numbering must line up with :attr:`NormalizedTurn.utt_rank`, which is a
    rank into this turn's own ``candidates`` list -- not a global utterance id.
    Getting this wrong would make every Arm A retrieval score meaningless.
    """
    if not candidate_texts:
        return ""
    lines = ["CANDIDATE UTTERANCES (choose exactly one by its number)"]
    lines.extend(f"{i}: {text}" for i, text in enumerate(candidate_texts))
    return "\n".join(lines)


def render_agent_prompt(
    context: ContextWindow,
    turn: NormalizedTurn,
    candidate_texts: Sequence[str],
    guidelines: dict[str, Any],
    ontology: dict[str, Any],
    kb: dict[str, Any],
    prompt_template: str,
    cfg: dict[str, Any],
) -> tuple[str, str]:
    """Render the Arm A prompt (spec 6.9 step 1).

    See :func:`reflex.contracts.render_agent_prompt` for the frozen contract.

    TWO PROPERTIES THIS FUNCTION GUARANTEES
    ---------------------------------------
    1. The system string is IDENTICAL for every turn of every conversation: it
       is built only from ``guidelines`` / ``ontology`` / ``kb`` / the template,
       never from ``context`` or ``turn``. That is what lets a provider cache the
       ~24k-token prefix instead of re-billing it 30k times (spec 8.6 cost).
    2. The user string contains ``context.text`` VERBATIM -- the same string the
       encoder sees -- plus the numbered candidates and nothing else. No gold
       field of ``turn`` is ever rendered; the candidate block is keyed off
       ``turn.candidates`` (an INPUT that ABCD provides), never off
       ``turn.nextstep`` (the LABEL). Rendering the label would leak the answer
       and invalidate spec 8.7's paired comparison.
    """
    del cfg  # rendering has no tunables; every knob here would be a spec-13 risk
    system_template, user_template = _split_sections(prompt_template)
    fields = {
        "guidelines_text": _render_guidelines(guidelines),
        "action_list": _render_action_list(ontology),
        "kb_text": _render_kb(kb),
        "subflow_list": "\n".join(f"- {s}" for s in _flatten_subflows(ontology)),
        "context_text": context.text,
        # Candidates exist iff ABCD supplied them for this turn. `turn.candidates`
        # is an input; `turn.nextstep` is the label we are asking the model for.
        "candidate_block": _render_candidate_block(
            candidate_texts if turn.candidates else []
        ),
    }
    return system_template.format(**fields), user_template.format(**fields)


# --------------------------------------------------------------------------- #
# parse_llm_json
# --------------------------------------------------------------------------- #


def _unwrap_code_fence(raw: str) -> str:
    """Remove a single surrounding ```/```json fence, if present.

    This is NOT the "repair truncated JSON" the contract forbids: the payload
    must still be exact, complete JSON after unwrapping. A fence is the one
    well-formed deviation frontier models make constantly, and paying for a
    retry to get the same object without three backticks is pure waste.
    """
    text = raw.strip()
    if not text.startswith("```"):
        return text
    newline = text.find("\n")
    if newline == -1:
        return text
    first_line = text[3:newline].strip()
    if first_line and not first_line.isalpha():  # ```{"nextstep": ...} -- not a fence
        return text
    body = text[newline + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()


def parse_llm_json(raw: str, ontology: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Strictly parse the LLM's JSON output (spec 6.9 step 2).

    See :func:`reflex.contracts.parse_llm_json` for the frozen contract.

    Returns ``None`` -- never a fabricated decision -- on any of:
      * not a single complete JSON object (no truncation repair);
      * a missing required key;
      * a ``nextstep`` outside ``ontology['next_steps']``;
      * an ``intent`` outside the 55 subflows;
      * an ``action`` that is not a known button, or absent on a
        ``take_action`` turn;
      * ``values`` that is not a list of strings;
      * a ``candidate_index`` that is not an int in ``[-1, 100)``.

    Extra keys are IGNORED rather than rejected. The contract enumerates the
    rejections it wants and extra keys are not among them; a model that adds a
    ``"reasoning"`` field has still answered, and burning a retry on it costs
    money for nothing.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        obj = json.loads(_unwrap_code_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None

    required = ("nextstep", "intent", "action", "values", "candidate_index")
    if any(key not in obj for key in required):
        return None

    next_steps = list(ontology.get("next_steps", ()))
    nextstep = obj["nextstep"]
    if not isinstance(nextstep, str) or nextstep not in next_steps:
        return None

    intent = obj["intent"]
    if not isinstance(intent, str) or intent not in _flatten_subflows(ontology):
        return None

    known_actions = _flatten_actions(ontology)
    action = obj["action"]
    if isinstance(action, str) and not action.strip():
        action = None
    if action is not None:
        if not isinstance(action, str) or action not in known_actions:
            return None
    if nextstep == "take_action" and action is None:
        return None
    if nextstep != "take_action" and action is not None:
        return None

    values = obj["values"]
    if values is None:
        values = []
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        return None

    index = obj["candidate_index"]
    # bool is an int subclass; `True` is not a rank.
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    if not -1 <= index < _MAX_CANDIDATE_INDEX:
        return None
    if nextstep != "retrieve_utterance" and index != -1:
        return None

    return {
        "nextstep": nextstep,
        "intent": intent,
        "action": action,
        "values": values,
        "candidate_index": index,
    }


# --------------------------------------------------------------------------- #
# Token estimation and cost
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=8)
def _token_counter(spec: str) -> Callable[[str], int]:
    """Build a LOCAL token counter from an ``llm.tokenizer`` spec.

    Accepted specs: ``"tiktoken:<encoding>"`` and ``"hf:<model_id>"``. The
    ``"heuristic"`` spec never reaches here. Never a hosted token-counting
    endpoint -- that is still a billed call.
    """
    kind, _, name = spec.partition(":")
    if kind == "tiktoken" and name:
        import tiktoken  # local BPE tables; no inference endpoint

        encoding = tiktoken.get_encoding(name)
        return lambda text: len(encoding.encode(text, disallowed_special=()))
    if kind == "hf" and name:
        from transformers import AutoTokenizer  # local files / HF cache

        tokenizer = AutoTokenizer.from_pretrained(name)
        return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
    raise ReflexError(
        f"unsupported llm.tokenizer {spec!r}. Use {_TOKENIZER_HEURISTIC!r}, "
        "'tiktoken:<encoding>' or 'hf:<model_id>'."
    )


def estimate_prompt_tokens(system: str, user: str, cfg: dict[str, Any]) -> tuple[int, int]:
    """Estimate ``(tokens_in, tokens_out)`` for a prompt WITHOUT calling any API.

    See :func:`reflex.contracts.estimate_prompt_tokens` for the frozen contract.

    ``llm.tokenizer`` selects the counter. The default ``heuristic`` divides
    characters by ``llm.chars_per_token`` and needs no model files at all, so the
    estimate works on a machine with no network and no HF cache. Set
    ``llm.tokenizer: tiktoken:<encoding>`` once the provider is known, for a
    tighter number. ``tokens_out`` is ``llm.est_tokens_out`` -- the reply is a
    fixed-shape JSON object, so its length barely varies.
    """
    spec = str(get_dotted(cfg, "llm.tokenizer"))
    if spec == _TOKENIZER_HEURISTIC:
        chars_per_token = float(get_dotted(cfg, "llm.chars_per_token"))
        if chars_per_token <= 0:
            raise ReflexError(f"llm.chars_per_token must be > 0, got {chars_per_token!r}")
        count = lambda text: math.ceil(len(text) / chars_per_token)  # noqa: E731
    else:
        count = _token_counter(spec)
    tokens_in = count(system) + count(user)
    return int(tokens_in), int(get_dotted(cfg, "llm.est_tokens_out"))


def llm_cost_usd(tokens_in: int, tokens_out: int, model_key: str, cfg: dict[str, Any]) -> float:
    """Compute LLM cost per spec 8.6 from the config price table.

    See :func:`reflex.contracts.llm_cost_usd` for the frozen contract. Raises
    :class:`PlaceholderConfigError` while the table is unfilled: a silent 0.0
    would print a free LLM in the report.

    The ``/ 1e6`` is not a hidden constant -- it is the unit declared by the key
    name ``llm.prices_usd_per_million``.
    """
    if model_key not in _MODEL_KEYS:
        raise ValueError(f"unknown model_key {model_key!r}; expected one of {_MODEL_KEYS}")
    prices = require_filled(cfg, f"llm.prices_usd_per_million.{model_key}")
    try:
        price_in = float(prices["in"])
        price_out = float(prices["out"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReflexError(
            f"llm.prices_usd_per_million.{model_key} must be a mapping with numeric "
            f"'in' and 'out' USD-per-million-token prices, got {prices!r}"
        ) from exc
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


# --------------------------------------------------------------------------- #
# Backends -- implemented, never reached in this build
# --------------------------------------------------------------------------- #


@dataclass
class _HTTPBackend:
    """An :class:`~reflex.contracts.LLMBackend` over stdlib ``urllib``.

    Deliberately NOT a vendor SDK: an SDK picks up an ambient ``*_API_KEY`` from
    the environment and can be constructed (and billed) by accident. Here the key
    must be named explicitly by ``llm.api_key_env``, the URL by ``llm.api_base``,
    and the request shape by ``llm.wire_format``.

    ``wire_format`` supports the two shapes the market actually uses:
      * ``openai_chat``       -- POST {api_base}/chat/completions
      * ``anthropic_messages`` -- POST {api_base}/messages
    """

    model_id: str
    wire_format: str
    api_base: str
    api_key_env: str
    timeout_s: float

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ReflexError(
                f"environment variable {self.api_key_env!r} (llm.api_key_env) is unset; "
                "cannot authenticate. Refusing to call without an explicit key."
            )
        return key

    def _request(self, system: str, user: str, temperature: float, max_tokens: int):
        base = self.api_base.rstrip("/")
        if self.wire_format == "openai_chat":
            url = f"{base}/chat/completions"
            payload = {
                "model": self.model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            headers = {
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            }
        elif self.wire_format == "anthropic_messages":
            url = f"{base}/messages"
            payload = {
                "model": self.model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            headers = {
                "x-api-key": self._api_key(),
                "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
                "Content-Type": "application/json",
            }
        else:
            raise ReflexError(
                f"unsupported llm.wire_format {self.wire_format!r}; "
                "expected 'openai_chat' or 'anthropic_messages'"
            )
        return urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )

    def _parse_response(self, body: dict[str, Any]) -> tuple[str, int, int]:
        if self.wire_format == "openai_chat":
            text = body["choices"][0]["message"]["content"] or ""
            usage = body.get("usage", {}) or {}
            return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        blocks = body.get("content", []) or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        usage = body.get("usage", {}) or {}
        return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
        """Return ``(text, tokens_in, tokens_out)`` for one completion."""
        request = self._request(system, user, temperature, max_tokens)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - never called in this build
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ReflexError(f"LLM HTTP {exc.code} from {self.api_base}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - never called in this build
            raise ReflexError(f"LLM request to {self.api_base} failed: {exc.reason}") from exc
        return self._parse_response(body)


# --------------------------------------------------------------------------- #
# The agent handle
# --------------------------------------------------------------------------- #


@dataclass
class _Agent:
    """Opaque handle returned by :func:`build_llm_agent`.

    Carries the resources :func:`llm_decide` needs but cannot receive as
    arguments (its signature is frozen): the rendered-prompt inputs, the frozen
    prompt and its hash, the response cache, and the mutable budget counter.
    """

    model_key: str
    model_id: str
    prompt_template: str
    prompt_hash: str
    guidelines: dict[str, Any]
    ontology: dict[str, Any]
    kb: dict[str, Any]
    cache_dir: str
    budget_usd: float
    backend: Optional[_HTTPBackend] = None
    spend_usd: float = 0.0
    calls: int = 0
    cache_hits: int = 0
    cache_key_collisions: int = 0
    parse_failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def build_llm_agent(cfg: dict[str, Any], model_key: str) -> Any:
    """Construct the LLM agent handle, honouring the kill switch.

    See :func:`reflex.contracts.build_llm_agent` for the frozen contract.

    Fails at CONSTRUCTION, in this order: kill switch, then ``model_key``, then
    every unfilled placeholder at once. A misconfigured Arm A run dies in a
    second instead of after an hour of setup -- and a run that was never
    authorized to spend money dies before it renders a single prompt.
    """
    if not get_dotted(cfg, "llm.enabled"):
        raise LLMDisabledError(
            "refusing to build the LLM agent: llm.enabled is false (the paid-call "
            "kill switch). ZERO PAID API CALLS is a hard project rule. To spend "
            "money you must deliberately set llm.enabled: true AND fill llm.strong, "
            "llm.cheap, llm.provider, llm.wire_format, llm.api_base, llm.api_key_env, "
            "llm.prices_usd_per_million and llm.price_list_date in "
            "configs/default.yaml. Until then Arm A can only be estimated, not run: "
            "see outputs/compile/armA_cost_estimate.md."
        )
    if model_key not in _MODEL_KEYS:
        raise ValueError(f"unknown model_key {model_key!r}; expected one of {_MODEL_KEYS}")

    resolved = _require_no_placeholders(
        cfg, (f"llm.{model_key}", f"llm.prices_usd_per_million.{model_key}", *_REQUIRED_LLM_KEYS)
    )
    model_id = str(resolved[f"llm.{model_key}"])

    # Import lazily: reflex.data owns the loaders (spec 4.1) and pulling it in at
    # module import time would make the kill-switch test depend on module 4.1.
    from reflex import data as data_mod

    prompt_template, prompt_hash = load_prompt(cfg, "agent_A")
    cache_dir = resolve_path(cfg, "llm.cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    backend = _HTTPBackend(
        model_id=model_id,
        wire_format=str(resolved["llm.wire_format"]),
        api_base=str(resolved["llm.api_base"]),
        api_key_env=str(resolved["llm.api_key_env"]),
        timeout_s=float(get_dotted(cfg, "llm.request_timeout_s")),
    )
    agent = _Agent(
        model_key=model_key,
        model_id=model_id,
        prompt_template=prompt_template,
        prompt_hash=prompt_hash,
        guidelines=data_mod.load_guidelines(cfg),
        ontology=data_mod.load_ontology(cfg),
        kb=data_mod.load_kb(cfg),
        cache_dir=cache_dir,
        budget_usd=float(get_dotted(cfg, "llm.budget_usd_per_run")),
        backend=backend,
    )
    _LOG.warning(
        "LLM agent ARMED: model_key=%s model_id=%s provider=%s dry_run=%s budget_usd=%.2f",
        model_key,
        model_id,
        resolved["llm.provider"],
        get_dotted(cfg, "llm.dry_run"),
        agent.budget_usd,
    )
    return agent


# --------------------------------------------------------------------------- #
# Response cache (spec 6.9 step 5)
# --------------------------------------------------------------------------- #


def _cache_key(agent: _Agent, context: ContextWindow) -> str:
    """Spec 6.9 step 5 key: ``(model, prompt_hash, context_hash)``.

    ``context_hash`` may be empty on a hand-built :class:`ContextWindow`; fall
    back to hashing ``context.text`` so a missing hash cannot collapse every
    turn onto one cache entry.
    """
    context_hash = context.context_hash or _sha16(context.text)
    return _sha16(f"{agent.model_id}|{agent.prompt_hash}|{context_hash}")


def _cache_path(agent: _Agent, key: str) -> str:
    return os.path.join(agent.cache_dir, _sha16(agent.model_id), key[:2], f"{key}.json")


def _cache_read(agent: _Agent, key: str, user_hash: str) -> Optional[dict[str, Any]]:
    """Read a cache entry, treating a user-block mismatch as a MISS.

    The spec's key does not cover the candidate block, so two turns with the same
    K-turn context could in principle map to one key. Storing the user-block hash
    in the entry and re-checking it turns that silent wrong answer into an
    ordinary miss, and counts it.
    """
    path = _cache_path(agent, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            entry = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if entry.get("user_hash") != user_hash:
        agent.cache_key_collisions += 1
        _LOG.warning("LLM cache key collision at %s: user block differs; re-calling", path)
        return None
    return entry


def _cache_write(agent: _Agent, key: str, user_hash: str, payload: dict[str, Any]) -> None:
    path = _cache_path(agent, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = dict(payload)
    entry["user_hash"] = user_hash
    entry["model_id"] = agent.model_id
    entry["prompt_hash"] = agent.prompt_hash
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, sort_keys=True)
    os.replace(tmp, path)


def _log_parse_failure(agent: _Agent, context: ContextWindow, raw_attempts: Sequence[str]) -> None:
    """Spec 6.9 step 2: a parse failure is LOGGED and counted incorrect, not dropped."""
    agent.parse_failures += 1
    record = {
        "model_id": agent.model_id,
        "prompt_hash": agent.prompt_hash,
        "convo_id": context.convo_id,
        "turn_index": context.turn_index,
        "context_hash": context.context_hash or _sha16(context.text),
        "attempts": [attempt[:2000] for attempt in raw_attempts],
    }
    _LOG.warning(
        "LLM parse failure after %d attempt(s) on convo %s turn %s -- counted INCORRECT",
        len(raw_attempts),
        context.convo_id,
        context.turn_index,
    )
    try:
        with open(os.path.join(agent.cache_dir, "parse_failures.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:  # pragma: no cover - logging must never break a run
        pass


# --------------------------------------------------------------------------- #
# llm_decide
# --------------------------------------------------------------------------- #


def _utterance_text(candidate_texts: Sequence[str], index: int) -> Optional[str]:
    return candidate_texts[index] if 0 <= index < len(candidate_texts) else None


def _decision_from_parsed(
    parsed: dict[str, Any],
    agent: _Agent,
    candidate_texts: Sequence[str],
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
    cache_hit: bool,
) -> LLMDecision:
    index = int(parsed["candidate_index"])
    return LLMDecision(
        nextstep=parsed["nextstep"],
        intent=parsed["intent"],
        action=parsed["action"],
        values=list(parsed["values"]),
        candidate_index=index,
        utterance_text=_utterance_text(candidate_texts, index),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        parse_failed=False,
        cache_hit=cache_hit,
        model_id=agent.model_id,
        prompt_hash=agent.prompt_hash,
    )


def llm_decide(
    agent: Any,
    context: ContextWindow,
    turn: NormalizedTurn,
    candidate_texts: Sequence[str],
    cfg: dict[str, Any],
) -> LLMDecision:
    """Get one decision from the LLM (spec 6.9). The only inference-time LLM call.

    See :func:`reflex.contracts.llm_decide` for the frozen contract.

    Order of operations, and why:
      1. kill switch  -- re-checked here, not just at construction, so no stale
         handle can outlive a config change;
      2. cache        -- a hit is not re-billed and reports ``latency_ms=0.0``;
      3. budget       -- checked BEFORE the call, on the ESTIMATE, so the cap
         cannot be breached by the very call that discovers it;
      4. call         -- temperature ``llm.temperature`` (0);
      5. parse        -- strict, with ``llm.max_retries_on_parse_failure`` (1)
         retry, then ``parse_failed=True``: counted INCORRECT and logged, never
         dropped and never fabricated.
    """
    if not get_dotted(cfg, "llm.enabled"):
        raise LLMDisabledError(
            "refusing to call the LLM: llm.enabled is false (the paid-call kill "
            "switch). ZERO PAID API CALLS is a hard project rule. No prompt was "
            "rendered and no request was made."
        )
    if not isinstance(agent, _Agent):
        raise ReflexError(
            f"llm_decide expects the handle from build_llm_agent, got {type(agent).__name__}"
        )

    system, user = render_agent_prompt(
        context,
        turn,
        candidate_texts,
        agent.guidelines,
        agent.ontology,
        agent.kb,
        agent.prompt_template,
        cfg,
    )
    user_hash = _sha16(user)
    key = _cache_key(agent, context)

    entry = _cache_read(agent, key, user_hash)
    if entry is not None:
        agent.cache_hits += 1
        if entry.get("parse_failed"):
            return LLMDecision(
                nextstep="",
                intent="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0.0,
                parse_failed=True,
                cache_hit=True,
                model_id=agent.model_id,
                prompt_hash=agent.prompt_hash,
            )
        return _decision_from_parsed(
            entry["parsed"], agent, candidate_texts, 0, 0, 0.0, cache_hit=True
        )

    est_in, est_out = estimate_prompt_tokens(system, user, cfg)

    if get_dotted(cfg, "llm.dry_run"):
        # Spec 10's dry run: estimate token volume and cost, call nothing, and
        # return parse_failed=True so no fake decision can be scored as correct.
        cost = llm_cost_usd(est_in, est_out, agent.model_key, cfg)
        with agent.lock:
            agent.spend_usd += cost
            agent.calls += 1
        return LLMDecision(
            nextstep="",
            intent="",
            tokens_in=est_in,
            tokens_out=est_out,
            latency_ms=0.0,
            parse_failed=True,
            cache_hit=False,
            model_id=agent.model_id,
            prompt_hash=agent.prompt_hash,
        )

    max_attempts = 1 + int(get_dotted(cfg, "llm.max_retries_on_parse_failure"))
    est_cost = llm_cost_usd(est_in, est_out * max_attempts, agent.model_key, cfg)
    with agent.lock:
        projected = agent.spend_usd + est_cost
        if projected > agent.budget_usd:
            raise BudgetExceededError(
                f"aborting run: this call would take LLM spend to ${projected:.4f}, "
                f"over the llm.budget_usd_per_run cap of ${agent.budget_usd:.2f} "
                f"(spent ${agent.spend_usd:.4f} over {agent.calls} calls). "
                "Raise the cap deliberately or shrink the run."
            )

    temperature = float(get_dotted(cfg, "llm.temperature"))
    max_tokens = int(get_dotted(cfg, "llm.max_tokens_out"))
    tokens_in = tokens_out = 0
    latency_ms = 0.0
    attempts: list[str] = []
    parsed: Optional[dict[str, Any]] = None

    for _ in range(max_attempts):
        started = time.perf_counter()
        raw, used_in, used_out = agent.backend.complete(system, user, temperature, max_tokens)
        latency_ms += (time.perf_counter() - started) * 1000.0
        # Bill what the provider says it billed; fall back to the estimate only
        # if usage is missing, so cost is never silently understated.
        tokens_in += used_in or est_in
        tokens_out += used_out or est_out
        attempts.append(raw)
        with agent.lock:
            agent.calls += 1
            agent.spend_usd += llm_cost_usd(used_in or est_in, used_out or est_out, agent.model_key, cfg)
        parsed = parse_llm_json(raw, agent.ontology)
        if parsed is not None:
            break

    if parsed is None:
        _log_parse_failure(agent, context, attempts)
        _cache_write(agent, key, user_hash, {"parse_failed": True, "attempts": attempts})
        return LLMDecision(
            nextstep="",
            intent="",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            parse_failed=True,
            cache_hit=False,
            model_id=agent.model_id,
            prompt_hash=agent.prompt_hash,
        )

    _cache_write(agent, key, user_hash, {"parse_failed": False, "parsed": parsed})
    return _decision_from_parsed(
        parsed, agent, candidate_texts, tokens_in, tokens_out, latency_ms, cache_hit=False
    )
