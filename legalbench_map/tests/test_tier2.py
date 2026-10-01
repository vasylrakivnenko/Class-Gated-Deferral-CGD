"""Tier 2's switchable readers (router/tier2.py) and our own GPU's API: the router's client
(router/gpu.py) against the gateway (gpu/gpu_gateway.py) in front of a fake llama-server."""
from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from router import reader, tier2
from router.gpu import GPUError, OwnGPU
from router.usage import Usage

spec = importlib.util.spec_from_file_location("gpu_gateway", Path(__file__).resolve().parents[1] / "gpu" / "gpu_gateway.py")
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)

TOKEN = "t" * 32
REPLY = '{"answer": "$500 per animal", "clause": 1}'


def serve(handler) -> tuple[str, ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


class FakeLlamaServer(BaseHTTPRequestHandler):
    bodies, status = [], 200

    def do_GET(self):
        self._json(200, {"status": "ok"} if self.path == "/health" else {"default_generation_settings": {"n_ctx": 32768}})

    def do_POST(self):
        FakeLlamaServer.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        if FakeLlamaServer.status != 200:
            return self._json(FakeLlamaServer.status, {"error": "busy"})
        self._json(200, {"choices": [{"message": {"content": REPLY}}], "usage": {"prompt_tokens": 120, "completion_tokens": 18},
                         "timings": {"prompt_ms": 21.5, "predicted_ms": 140.25}})

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def gpu():
    """The gateway in front of a fake llama-server; yields the gateway's URL."""
    FakeLlamaServer.bodies, FakeLlamaServer.status = [], 200
    upstream, s1 = serve(FakeLlamaServer)
    engine = gateway.LlamaCpp(upstream, "gemma-test")
    url, s2 = serve(gateway.make_handler(engine, TOKEN))
    yield url
    s1.shutdown(), s2.shutdown()


def test_gateway_speaks_zadum_gpu_1_for_llama_cpp(gpu):
    client = OwnGPU(gpu, TOKEN)
    h = client.health()
    assert (h["api"], h["ok"], h["engine"], h["model"], h["context"]) == ("zadum-gpu/1", True, "llama.cpp", "gemma-test", 32768)
    d = client.generate([{"role": "user", "content": "Q?"}], max_tokens=50, json_schema=reader.SCHEMA)
    assert d["text"] == REPLY and d["usage"] == {"prompt_tokens": 120, "completion_tokens": 18}
    assert (d["timing"]["prefill_ms"], d["timing"]["decode_ms"]) == (21.5, 140.25) and d["timing"]["server_ms"] > 0
    sent = FakeLlamaServer.bodies[-1]  # what llama-server got: the API's request in its own terms
    assert sent["response_format"]["json_schema"]["schema"] == reader.SCHEMA and sent["max_tokens"] == 50
    assert sent["chat_template_kwargs"] == {"enable_thinking": False} and sent["cache_prompt"] is False


def test_gateway_refuses_a_wrong_token(gpu):
    with pytest.raises(GPUError, match="401") as e:
        OwnGPU(gpu, "wrong-token-wrong-token").health()
    assert not e.value.retryable


def test_engine_errors_come_back_retryable_and_are_retried_once(gpu):
    FakeLlamaServer.status = 503
    with pytest.raises(GPUError, match="502") as e:
        OwnGPU(gpu, TOKEN).generate([{"role": "user", "content": "Q?"}])
    assert e.value.retryable and len(FakeLlamaServer.bodies) == 2


def test_own_gpu_reader_reads_through_the_gateway(gpu):
    llm = reader.OwnGPULLM(OwnGPU(gpu, TOKEN))
    r = reader.read(tier2.PROBE_QUESTION, tier2.PROBE_CLAUSES, llm)
    assert r.fired and r.answer == "$500 per animal"
    assert r.usage["server_s"] > 0 and r.usage["prefill_ms"] == 21.5 and r.usage["engine"] == "llama.cpp"


def test_a_gateway_that_is_down_is_a_reader_error():
    llm = reader.OwnGPULLM(OwnGPU("http://127.0.0.1:9", TOKEN))  # nothing listens on port 9
    with pytest.raises(reader.ReaderError, match="can't reach"):
        llm("Q?")


# ---- the switch

class FakeLLM:
    def __init__(self, reply=REPLY, error=None):
        self.reply, self.error = reply, error

    def __call__(self, prompt):
        if self.error:
            raise self.error
        return self.reply, {"prompt_tokens": 10}


def options(**replies):
    return {i: tier2.Option(i, f"reader {i}", "m", "w", 0.8 if i == "b" else 0.7, "", "", lambda r=r: r, "PATH")
            for i, r in replies.items()}


def test_switching_tests_first_and_keeps_the_old_reader_if_the_test_fails(tmp_path):
    store = Usage(tmp_path / "usage.db", daily_limit=5)
    t = tier2.Tier2("a", store, options(a=FakeLLM(), b=FakeLLM('{"answer": null, "clause": null}'),
                                        c=FakeLLM(error=reader.ReaderError("HTTP 503"))))
    assert t.reader.option.id == "a" and t.chosen["by"] == "--tier2"
    for bad, why in (("b", "didn't answer"), ("c", "HTTP 503")):
        with pytest.raises(reader.ReaderError, match=why):
            t.select(bad, by="admin@x.org")
        assert t.reader.option.id == "a"
    t.select("off", by="admin@x.org")
    assert t.reader is None and t.status()["active"] == "off"


def test_the_choice_survives_a_restart_and_beats_the_flag(tmp_path):
    store = Usage(tmp_path / "usage.db", daily_limit=5)
    opts = options(a=FakeLLM(), b=FakeLLM())
    tier2.Tier2("a", store, opts).select("b", by="admin@x.org")
    t = tier2.Tier2("a", Usage(tmp_path / "usage.db", daily_limit=5), opts)
    assert t.reader.option.id == "b" and t.chosen["by"] == "admin@x.org" and t.chosen["test"]["answer"] == "$500 per animal"


def test_old_flag_values_still_work():
    t = tier2.Tier2("priority", None, {"fireworks-priority": options(x=FakeLLM())["x"]})
    assert t.chosen["option"] == "fireworks-priority" and t.reader is not None


def test_a_reader_that_cannot_start_leaves_tier2_off_and_says_why():
    def broken():
        raise reader.ReaderError("OPENROUTER_API_KEY not found")
    t = tier2.Tier2("x", None, {"x": tier2.Option("x", "x", "m", "w", 0.7, "", "", broken, "PATH")})
    assert t.reader is None and "OPENROUTER_API_KEY" in t.error and t.status()["active"] == "off"


def test_each_reader_logs_its_option_and_uses_its_own_check():
    from tests.test_qtree import PETS, _tier2
    gemma = tier2.Connected(options(b=None)["b"], FakeLLM('{"answer": "$500 per animal", "clause": null}'))
    h, _ = _tier2(gemma, noul=0.75)  # enough for gpt-oss-120b (0.7), not for Gemma (0.8)
    a = h.answer("How much is the pet deposit?", PETS)
    assert a.path == "deferred" and "0.75 < 0.8" in a.reason and a.reader["usage"]["option"] == "b"


def test_tier2_stats_per_reader(tmp_path):
    u = Usage(tmp_path / "usage.db", daily_limit=5)
    base = {"answer": None, "path": "deferred", "ms": 1, "llm_calls": 1}
    for option, ms, reason in (("priority", 300, "copied word for word ..."), ("fireworks-priority", 500, "the reader found it isn't stated"),
                               ("gpu-gemma", 250, "copied word for word ..."), ("gpu-gemma", None, "the reader was unavailable (down)")):
        read = {"fired": reason.startswith("copied"), "ms": ms, "reason": reason, "usage": {"option": option}}
        u.log("a@x.org", "ok", "jev", {**base, "reader": read}, request_id=f"req_{option}_{ms}", question="Q?", document="D")
    s = u.tier2_stats()
    assert s["fireworks-priority"] == {"reads": 2, "answered": 1, "unavailable": 0, "p50_ms": 500, "p90_ms": 500}
    assert s["gpu-gemma"] == {"reads": 2, "answered": 1, "unavailable": 1, "p50_ms": 250, "p90_ms": 250}
    assert {r["tier2"] for r in u.stats()["recent"]} == {"fireworks-priority", "gpu-gemma"}
