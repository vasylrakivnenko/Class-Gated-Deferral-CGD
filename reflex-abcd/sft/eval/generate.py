"""Generate against a deployed fine-tuned model for every prompt in the eval
set, and write {turn_id, generation} JSONL that score_generations.py consumes.

Concurrent (asyncio + AsyncFireworks): a serial run measured ~0.37 req/s on
this single-GPU dedicated deployment (~2.7s/request), which would put 8,889
prompts at ~80 minutes. Concurrency is bounded by --concurrency, not
unlimited, because this is a single-replica deployment (min=max=1
replica_count) and an unbounded burst would just queue at the server, not
speed anything up, while making failures harder to diagnose.

Model addressing for a LIVE-MERGE dedicated deployment (per Fireworks docs):
the bare fine-tuned model id 404s on a fresh deployment; the full form is
"<model-path>#<deployment-resource-path>".

USAGE
-----
    python sft/eval/generate.py \
        --model accounts/vasyl-r/models/abcd-reflex-qwen3-0p6b-v1 \
        --deployment accounts/vasyl-r/deployments/abcd-unconstrained-eval \
        --system "You are a customer service agent..." \
        --out sft/eval/data/gen_unconstrained.jsonl \
        --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time


async def _generate_one(client, route, system, prompt, max_tokens, sem):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=route,
                messages=[{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content or "", None
        except Exception as e:
            return "", f"{type(e).__name__}: {str(e)[:200]}"


async def _run(args):
    from fireworks import AsyncFireworks
    client = AsyncFireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id="vasyl-r")
    route = f"{args.model}#{args.deployment}"

    prompts = []
    with open(args.prompts, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            prompts.append((d["turn_id"], d["prompt"]))
    if args.limit:
        prompts = prompts[:args.limit]
    if args.smoke:
        prompts = prompts[:args.smoke]

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_done = 0
    n_err = 0
    results: list[tuple[str, str, str | None]] = [None] * len(prompts)  # type: ignore

    async def _worker(i, turn_id, prompt):
        nonlocal n_done, n_err
        text, err = await _generate_one(client, route, args.system, prompt, args.max_tokens, sem)
        results[i] = (turn_id, text, err)
        n_done += 1
        if err:
            n_err += 1
        if n_done % 200 == 0 or n_done == len(prompts):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(prompts) - n_done) / rate / 60 if rate > 0 else float("inf")
            print(f"  {n_done}/{len(prompts)}  err={n_err}  {rate:.1f}/s  eta={eta:.1f}min",
                  file=sys.stderr)

    await asyncio.gather(*[_worker(i, tid, p) for i, (tid, p) in enumerate(prompts)])

    if args.smoke:
        for turn_id, text, err in results:
            print(f"--- {turn_id} ---\nGEN: {text[:300]}{' ERR:'+err if err else ''}\n")
    else:
        with open(args.out, "w", encoding="utf-8") as fh:
            for turn_id, text, err in results:
                fh.write(json.dumps({"turn_id": turn_id, "generation": text,
                                     "error": err}, ensure_ascii=False) + "\n")

    print(json.dumps({"n_total": len(prompts), "n_err": n_err,
                      "elapsed_s": round(time.time() - t0, 1),
                      "out": None if args.smoke else args.out}, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--prompts", default="sft/eval/data/eval_prompts_test_seen.jsonl")
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.smoke and not args.out:
        raise SystemExit("--out is required unless --smoke is set")
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
