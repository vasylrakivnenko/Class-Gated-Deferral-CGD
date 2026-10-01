"""
Checks our own GPU from the router, the way Tier 2 calls it: the gateway's health, one test
read, and optionally N real reader prompts timed end to end (router -> tunnel -> gateway ->
engine). Exits 1 if the GPU can't serve.

    cd legalbench_map && ../.venv/bin/python gpu/check_gpu.py [--bench N]

RTX 4090 + llama.cpp, 2026-10-01, 286 reads: 0.29 s p50, 0.36 s p90, 0.63 s p99; GPU 0.25 s p50.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import reader, tier2  # noqa: E402
from router.gpu import GPUError, OwnGPU  # noqa: E402

PROMPTS = Path(__file__).resolve().parent / "bench_prompts.jsonl"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", type=int, default=0, help="time this many real reader prompts (up to 286)")
    args = p.parse_args()
    try:
        gpu = OwnGPU()
        print("health:", json.dumps(gpu.health()))
        llm = reader.OwnGPULLM(gpu)
        test = tier2.probe(llm)
    except (GPUError, reader.ReaderError) as e:
        print(f"FAILED: {e}")
        return 1
    print(f"test read: {test['answer']!r} in {test['ms']} ms")
    if args.bench:
        prompts = [json.loads(line)["prompt"] for line in PROMPTS.open()][:args.bench]
        wall, gpu_ms, bad = [], [], 0
        for prompt in prompts:
            t = time.perf_counter()
            text, usage = llm(prompt)
            wall.append((time.perf_counter() - t) * 1000)
            gpu_ms.append(usage.get("server_s", 0) * 1000)
            bad += reader._parse(text) is None
        pct = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(len(xs) * q))]
        print(f"{len(prompts)} reads: round trip p50 {pct(wall, .5):.0f} ms, p90 {pct(wall, .9):.0f}, "
              f"p99 {pct(wall, .99):.0f} | on the GPU p50 {pct(gpu_ms, .5):.0f} ms | replies that weren't JSON: {bad}")
        print("expected (RTX 4090, 2026-10-01): p50 ~290 ms, p90 ~360, p99 ~630; GPU ~250; 0 not JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
