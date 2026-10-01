"""
The gateway in front of our own GPU's inference engine. It speaks our internal API,
zadum-gpu/1 (spec: router/gpu.py), and translates it for whatever serves the model, so the
router never changes when the deployment does.

Runs on the GPU machine next to the engine, with Python 3.9+ alone (no packages):
    ZADUM_GPU_TOKEN=... python3 gpu_gateway.py --engine llama.cpp --upstream http://127.0.0.1:8090 \\
        --model gemma-4-26b-a4b-it-q4_0 [--host 127.0.0.1] [--port 8000]
It listens on 127.0.0.1 by default; the router reaches it through an SSH tunnel (router/STATUS.md).
Every request must carry `Authorization: Bearer $ZADUM_GPU_TOKEN`.

Engines: `llama.cpp` (llama-server) and `openai` (anything with OpenAI's chat completions and
JSON-schema response_format: vLLM, SGLang, TGI...). A new engine is a subclass of Engine with
health() and generate(), added to ENGINES.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import socket
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API = "zadum-gpu/1"


class EngineError(Exception):
    def __init__(self, message: str, status: int = 502, retryable: bool = True):
        super().__init__(message)
        self.status, self.retryable = status, retryable


class Engine:
    """One inference server. generate() gets the API's request and returns its reply's
    text, usage and timing; the gateway adds the rest."""
    name = ""

    def __init__(self, upstream: str, model: str, timeout: float = 60):
        self.upstream, self.model, self.timeout = upstream.rstrip("/"), model, timeout

    def health(self) -> dict:
        raise NotImplementedError

    def generate(self, req: dict) -> dict:
        raise NotImplementedError

    def _http(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.upstream + path, data=data, method=method,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=timeout or self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")[:300]
            raise EngineError(f"{self.name} said HTTP {e.code}: {text}", 400 if 400 <= e.code < 500 else 502,
                              retryable=e.code >= 500) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise EngineError(f"{self.name} unreachable at {self.upstream}: {e}", 503) from None


class OpenAICompatible(Engine):
    """OpenAI's /v1/chat/completions: vLLM, SGLang, TGI, LM Studio... Timing is the call as
    seen from here; prefill and decode aren't reported."""
    name = "openai"

    def health(self) -> dict:
        models = [m.get("id") for m in self._http("GET", "/v1/models", timeout=3).get("data", [])]
        return {"served": models}

    def body(self, req: dict) -> dict:
        body = {"model": self.model, "messages": req["messages"], "max_tokens": req.get("max_tokens", 200),
                "temperature": req.get("temperature", 0),
                "chat_template_kwargs": {"enable_thinking": bool(req.get("thinking"))}}
        if req.get("json_schema"):
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": req["json_schema"]}}
        return body

    def generate(self, req: dict) -> dict:
        t = time.perf_counter()
        d = self._http("POST", "/v1/chat/completions", self.body(req))
        server_ms = (time.perf_counter() - t) * 1000
        u = d.get("usage") or {}
        return {"text": d["choices"][0]["message"].get("content") or "",
                "usage": {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")},
                "timing": {"server_ms": round(server_ms, 1), **self.timing(d)}}

    def timing(self, d: dict) -> dict:
        return {"prefill_ms": None, "decode_ms": None}


class LlamaCpp(OpenAICompatible):
    """llama.cpp's llama-server. Start it without the host-RAM prompt cache, which stalls
    every request (--cache-ram 0 --ctx-checkpoints 0 --kv-unified; router/STATUS.md)."""
    name = "llama.cpp"

    def health(self) -> dict:
        if self._http("GET", "/health", timeout=3).get("status") != "ok":
            raise EngineError("llama-server isn't ready", 503)
        props = self._http("GET", "/props", timeout=3)
        return {"context": (props.get("default_generation_settings") or {}).get("n_ctx")}

    def body(self, req: dict) -> dict:
        return {**super().body(req), "cache_prompt": False}  # each read is a new document

    def timing(self, d: dict) -> dict:
        t = d.get("timings") or {}
        return {"prefill_ms": t.get("prompt_ms"), "decode_ms": t.get("predicted_ms")}


ENGINES = {e.name: e for e in (LlamaCpp, OpenAICompatible)}


def make_handler(engine: Engine, token: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: through the SSH tunnel a new connection costs a round trip

        def setup(self):
            super().setup()
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # no 40 ms Nagle/delayed-ACK stalls

        def _authorized(self) -> bool:
            given = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
            if given and hmac.compare_digest(given.encode(), token.encode()):
                return True
            self._error(401, "bad or missing token", retryable=False)
            return False

        def do_GET(self):
            if self.path != "/v1/health":
                return self._error(404, "no such endpoint", retryable=False)
            if not self._authorized():
                return
            base = {"api": API, "model": engine.model, "engine": engine.name}
            try:
                self._json(200, {**base, "ok": True, **engine.health()})
            except EngineError as e:
                self._json(503, {**base, "ok": False, "error": str(e)})

        def do_POST(self):
            if self.path != "/v1/generate":
                return self._error(404, "no such endpoint", retryable=False)
            if not self._authorized():
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if not isinstance(req.get("messages"), list) or not req["messages"]:
                    raise ValueError("messages must be a non-empty list")
            except ValueError as e:
                return self._error(400, f"bad request: {e}", retryable=False)
            try:
                out = engine.generate(req)
            except EngineError as e:
                return self._error(e.status, str(e), e.retryable)
            except (KeyError, IndexError, TypeError) as e:
                return self._error(502, f"{engine.name} replied in an unexpected shape: {e!r}", retryable=True)
            self._json(200, {**out, "model": engine.model, "engine": engine.name})

        def _error(self, code: int, message: str, retryable: bool):
            self._json(code, {"error": {"message": message, "retryable": retryable}})

        def _json(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._headers_buffer.append(b"\r\n" + body)  # headers and body in one write
            self.flush_headers()

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=list(ENGINES), required=True)
    p.add_argument("--upstream", required=True, help="the engine's base URL, e.g. http://127.0.0.1:8090")
    p.add_argument("--model", required=True, help="the model's name as the router logs it (and as an openai engine serves it)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    token = os.environ.get("ZADUM_GPU_TOKEN", "")
    if len(token) < 16:
        raise SystemExit("Set ZADUM_GPU_TOKEN (16+ characters) in the environment.")
    engine = ENGINES[args.engine](args.upstream, args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, token))
    print(f"{API} gateway for {engine.name} at {args.upstream} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
