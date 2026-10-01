"""Our own GPU behind one internal API, "zadum-gpu/1", so the router never depends on what
serves the model there.

Whatever runs on the GPU (llama.cpp today; vLLM, SGLang, TGI... tomorrow), a small gateway
next to it (gpu/gpu_gateway.py) speaks this API, and the router talks only to the gateway, with
`OwnGPU` below. A new deployment needs a new engine adapter in gpu/gpu_gateway.py and nothing
else: this file, router/reader.py and the admin switch stay as they are.

The API: JSON over HTTP, every call with `Authorization: Bearer <ZADUM_GPU_TOKEN>`.

    GET /v1/health
      200 {"api": "zadum-gpu/1", "ok": true, "model": "gemma-4-26b-a4b-it-q4_0", "engine": "llama.cpp",
           "context": 32768}
      503 the same with "ok": false and "error": "..." (the gateway is up, the engine isn't)

    POST /v1/generate
      {"messages": [{"role": "user", "content": "..."}],  chat messages, as in OpenAI's API
       "max_tokens": 200, "temperature": 0,
       "json_schema": {...} | null,                        constrain the reply to this JSON schema
       "thinking": false}                                  the model's reasoning mode, where it has one
      200 {"text": "...", "model": "...", "engine": "...",
           "usage": {"prompt_tokens": 495, "completion_tokens": 21},
           "timing": {"server_ms": 250.1, "prefill_ms": 78.0, "decode_ms": 161.3}}
           (server_ms: the engine call as the gateway saw it; prefill/decode: as the engine
           reports them, or null when it doesn't)

    Errors, any endpoint: 4xx/5xx {"error": {"message": "...", "retryable": true | false}}
      401 bad or missing token; 400 a request the engine refused (not retryable);
      502/503 the engine failed or is down (retryable).

Where it is: ZADUM_GPU_URL (default http://127.0.0.1:18000, the SSH tunnel to the pod, see
router/STATUS.md) and ZADUM_GPU_TOKEN, from the environment, the repo's .env or ~/.env.
"""
from __future__ import annotations

import os
import time

API = "zadum-gpu/1"
DEFAULT_URL = "http://127.0.0.1:18000"


class GPUError(RuntimeError):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def setting(name: str, default: str | None = None) -> str | None:
    """From the environment, the repo's .env, or ~/.env."""
    from router.reader import ReaderError, api_key
    try:
        return api_key(name)
    except ReaderError:
        return default


class OwnGPU:
    """A client for zadum-gpu/1. Fails fast (the router waits on it): 2 s to connect, one
    retry when the gateway says the error is retryable."""

    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = 15, connect_timeout: float = 2):
        import requests
        self.url = (url or setting("ZADUM_GPU_URL", DEFAULT_URL)).rstrip("/")
        self.token = token or setting("ZADUM_GPU_TOKEN")
        if not self.token:
            raise GPUError("ZADUM_GPU_TOKEN not found in the environment, the repo's .env or ~/.env", retryable=False)
        self.session, self.timeout = requests.Session(), (connect_timeout, timeout)

    def _call(self, method: str, path: str, body: dict | None = None, timeout=None) -> dict:
        import requests
        try:
            r = self.session.request(method, self.url + path, json=body, timeout=timeout or self.timeout,
                                     headers={"Authorization": f"Bearer {self.token}"})
        except requests.RequestException as e:
            raise GPUError(f"can't reach the GPU gateway at {self.url}: {type(e).__name__}") from e
        try:
            d = r.json()
        except ValueError:
            raise GPUError(f"HTTP {r.status_code} from the GPU gateway, not JSON: {r.text[:120]}") from None
        if r.status_code != 200:
            err = d.get("error") if isinstance(d.get("error"), dict) else {"message": d.get("error") or r.reason}
            raise GPUError(f"HTTP {r.status_code}: {err.get('message')}", retryable=bool(err.get("retryable", r.status_code >= 500)))
        return d

    def health(self) -> dict:
        """The gateway's /v1/health; raises GPUError unless it and the engine are up."""
        d = self._call("GET", "/v1/health", timeout=(self.timeout[0], 3))
        if d.get("api") != API:
            raise GPUError(f"the gateway speaks {d.get('api')!r}, not {API!r}", retryable=False)
        return d

    def generate(self, messages: list, max_tokens: int = 200, temperature: float = 0, json_schema: dict | None = None,
                 thinking: bool = False) -> dict:
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "json_schema": json_schema, "thinking": thinking}
        for attempt in range(2):
            try:
                d = self._call("POST", "/v1/generate", body)
                return {**d, "attempts": attempt + 1}
            except GPUError as e:
                if not e.retryable or attempt:
                    raise
                time.sleep(0.2)
        raise AssertionError("unreachable")
