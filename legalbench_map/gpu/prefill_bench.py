"""Prefill throughput against llama-server, with our real Tier 2 reader prompts.
usage: prefill_bench.py PROMPTS.jsonl [--url http://127.0.0.1:8090] [--conc 1,4,8,16] [--n 286] [--answer [--no-schema]]
--answer: the reader's real request (JSON schema, up to 200 tokens) instead of max_tokens=1.
Runs on the GPU box against llama-server itself (gpu/setup_pod.sh); gpu/check_gpu.py times the whole path from the router."""
import argparse, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("prompts")
ap.add_argument("--url", default="http://127.0.0.1:8090")
ap.add_argument("--conc", default="1,4,8,16")
ap.add_argument("--n", type=int, default=286)
ap.add_argument("--answer", action="store_true")
ap.add_argument("--no-schema", action="store_true")
a = ap.parse_args()
NOSCHEMA = a.no_schema
prompts = [json.loads(l)["prompt"] for l in open(a.prompts)][: a.n]
SCHEMA = {"type": "object", "properties": {"answer": {"type": ["string", "null"]}, "clause": {"type": ["integer", "null"]}},
          "required": ["answer", "clause"]}


def call(prompt):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 200 if a.answer else 1}
    if a.answer and not NOSCHEMA:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}}
    req = urllib.request.Request(f"{a.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d["timings"], (time.perf_counter() - t) * 1000


pct = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(len(xs) * p))]
call(prompts[0])  # warm-up
for c in [int(x) for x in a.conc.split(",")]:
    t = time.perf_counter()
    with ThreadPoolExecutor(c) as ex:
        res = list(ex.map(call, prompts))
    wall = time.perf_counter() - t
    toks = sum(r[0]["prompt_n"] for r in res)
    pps = [r[0]["prompt_per_second"] for r in res]
    ms = [r[1] for r in res]
    pms = [r[0]["prompt_ms"] for r in res]
    print(f"concurrency {c:2}: {len(res)} requests, {toks:,} prompt tokens (p50 {pct([r[0]['prompt_n'] for r in res], .5)}) | "
          f"AGGREGATE {toks / wall:,.0f} prompt tok/s over {wall:.1f}s | per request: prefill {pct(pps, .5):,.0f} tok/s p50, "
          f"prefill {pct(pms, .5):.0f} ms p50 | request {pct(ms, .5):.0f} ms p50, {pct(ms, .9):.0f} ms p90", flush=True)
