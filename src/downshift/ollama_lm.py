"""A DSPy LM that speaks ollama's native /api/chat.

Why not just use `dspy.LM("ollama_chat/...")`? Because of one parameter.

Qwen3 and other hybrid-reasoning models emit a `<think>` block before every
answer. For a three-way sentiment label that is ~300 wasted tokens per call, and
those are *billed output tokens* on any hosted endpoint. Measured locally, a
single classification takes 1.98s with reasoning and 0.04s without -- a 50x gap
that decides whether the demo runs in ten minutes or two hours, and a ~300x gap
in output-token cost that decides whether the "cheap model" is actually cheap.

Ollama exposes exactly the switch we need as a top-level `think` field on
/api/chat. As of litellm 1.100.0 there is no route to it: the `ollama_chat`
provider raises on `think=`, and neither `extra_body` nor `chat_template_kwargs`
survive the trip through the OpenAI-compatible endpoint (verified -- all three
still returned full reasoning traces). So we bypass litellm for local models.

The class is ~100 lines and buys three things litellm was not giving us anyway:
the `think` switch, exact prompt/completion token counts straight from ollama's
counters (the cost axis is only as honest as these), and reasoning tokens
reported separately so the chart can show what "thinking" actually costs.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import dspy
from litellm.types.utils import Choices, Message, ModelResponse, Usage

DEFAULT_API_BASE = "http://localhost:11434"


class OllamaLM(dspy.BaseLM):
    """Local ollama chat model with explicit control over reasoning.

    Args:
        model: ollama tag, e.g. "qwen3:1.7b". No provider prefix.
        think: None leaves the model's default alone; False forces reasoning off
            (the fast, cheap path); True forces it on. Non-reasoning models
            reject the field, so we retry once without it -- see `forward`.
        num_ctx: ollama silently truncates to 4096 by default, which quietly
            corrupts few-shot prompts once GEPA grows the instruction. Set it.
    """

    def __init__(self, model: str, think: bool | None = None,
                 api_base: str = DEFAULT_API_BASE, temperature: float = 0.0,
                 max_tokens: int = 1024, num_ctx: int = 8192,
                 timeout: int = 300, **kwargs):
        super().__init__(model=model, temperature=temperature,
                         max_tokens=max_tokens, **kwargs)
        self.think = think
        self.api_base = api_base.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self._think_unsupported = False

    # ── transport ───────────────────────────────────────────────────────────

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.api_base}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise dspy.LMError(f"ollama HTTP {e.code} for {self.model}: {detail}") from e
        except urllib.error.URLError as e:
            raise dspy.LMError(
                f"cannot reach ollama at {self.api_base} ({e.reason}). Is `ollama serve` running?"
            ) from e

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        merged = {**self.kwargs, **kwargs}

        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": merged.get("temperature", 0.0),
                "num_predict": merged.get("max_tokens", 1024),
                "num_ctx": self.num_ctx,
            },
        }
        # Ollama rejects `think` outright on models with no reasoning mode, so we
        # probe once and remember. Cheaper than maintaining a capability table
        # that goes stale every time a new tag lands.
        if self.think is not None and not self._think_unsupported:
            body["think"] = self.think

        try:
            data = self._post(body)
        except dspy.LMError as e:
            if "think" in str(e).lower() and "think" in body:
                self._think_unsupported = True
                body.pop("think")
                data = self._post(body)
            else:
                raise

        return self._to_openai_shape(data)

    # ── response shaping ────────────────────────────────────────────────────

    def _to_openai_shape(self, data: dict) -> ModelResponse:
        msg = data.get("message", {}) or {}
        content = msg.get("content", "") or ""
        reasoning = msg.get("thinking") or ""

        # A reasoning model that hits num_predict inside its <think> block returns
        # empty content with a full token count. Downstream that looks like a
        # model too dumb to answer, when it is really max_tokens set too low --
        # a misdiagnosis that would put a wrong bar on the chart. Name it here.
        if not content and reasoning and data.get("done_reason") == "length":
            raise dspy.LMError(
                f"{self.model}: reasoning consumed the whole {self.kwargs.get('max_tokens')}-token "
                f"budget and no answer was emitted ({len(reasoning)} chars of reasoning). "
                f"Raise max_tokens or set think=False."
            )

        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)

        message = Message(role="assistant", content=content)
        # DSPy surfaces this as `reasoning_content`; keeping it visible (rather
        # than discarding it) is what lets the report show that a "cheap" model
        # spent 300 output tokens thinking before saying one word.
        if reasoning:
            message.reasoning_content = reasoning

        response = ModelResponse(
            id=data.get("created_at", "ollama"),
            choices=[Choices(finish_reason=data.get("done_reason", "stop"),
                             index=0, message=message)],
            model=self.model,
            usage=Usage(prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens),
        )
        # Local inference has no per-token bill; the chart prices these rows off
        # the published rate card for the equivalent hosted model instead.
        response._hidden_params = {
            "response_cost": 0.0,
            "ollama_total_duration_ns": data.get("total_duration"),
            "ollama_reasoning_chars": len(reasoning),
        }
        return response

    def copy(self, **kwargs):
        new = super().copy(**kwargs)
        for attr in ("think", "api_base", "num_ctx", "timeout", "_think_unsupported"):
            if attr not in kwargs:
                setattr(new, attr, getattr(self, attr))
        return new
