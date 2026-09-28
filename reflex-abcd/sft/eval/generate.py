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

SUBSETTING (--limit): eval_prompts_test_seen.jsonl is written one record per
turn in conversation order, so a head slice of N rows is a contiguous block of
whole conversations (N=500 -> 56 conversations, N=1000 -> 108), not a random
sample of test_seen. --limit therefore draws WHOLE CONVERSATIONS at random
under --seed until N turns are covered, and the seed / conversation count /
exact row count are written to <out>.sample.json and printed in the run
summary. --limit-mode head restores the old contiguous-block behaviour, only
for reproducing an already-published subset artifact.

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


def _sample_by_convo(rows, limit, seed):
    """Draw WHOLE conversations at random (seeded) until `limit` turns are
    covered, instead of taking the head of the prompt file.

    Rows are clustered by conversation, so a head slice's effective sample
    size is its conversation count, and which conversations it gets is
    decided by file order rather than by chance. Conversations are kept
    intact here so a conversation-clustered CI (probes/response_metrics.py
    :: bootstrap_ci) stays computable over the subset downstream.
    """
    import random

    by_convo = {}
    for r in rows:
        by_convo.setdefault(r["convo_id"], []).append(r)
    convos = sorted(by_convo)
    random.Random(seed).shuffle(convos)
    keep, n_rows = set(), 0
    for c in convos:
        if n_rows >= limit:
            break
        keep.add(c)
        n_rows += len(by_convo[c])
    kept = [r for r in rows if r["convo_id"] in keep]  # file order preserved
    meta = {
        "mode": "convo", "seed": seed, "limit_requested": limit,
        "n_rows": len(kept), "n_convos": len(keep),
        "n_convos_in_prompt_file": len(convos),
        "note": "whole conversations sampled at random under `seed` until "
                "`limit_requested` turns were covered; n_rows overshoots "
                "limit_requested by the last conversation's remaining turns. "
                "The sampling unit is the conversation, so any interval over "
                "this subset must be clustered by convo_id.",
    }
    return kept, meta


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

    rows = []
    with open(args.prompts, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))

    sample_meta = None
    if args.limit:
        if args.limit_mode == "convo":
            rows, sample_meta = _sample_by_convo(rows, args.limit, args.seed)
        else:
            rows = rows[:args.limit]
            sample_meta = {
                "mode": "head", "seed": None, "limit_requested": args.limit,
                "n_rows": len(rows),
                "n_convos": len({r["convo_id"] for r in rows}),
                "note": "CONTIGUOUS prompt-file block, NOT a random sample of "
                        "test_seen -- only for reproducing an already-published "
                        "subset artifact.",
            }
        print(f"subset: {sample_meta['n_rows']} turns in "
              f"{sample_meta['n_convos']} conversations (mode={sample_meta['mode']}, "
              f"seed={sample_meta['seed']})", file=sys.stderr)
    prompts = [(d["turn_id"], d["prompt"]) for d in rows]
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
        if sample_meta:
            # The subset is part of the result: without the seed and the
            # conversation count, the rate this arm gets cannot be read
            # correctly or reproduced.
            with open(args.out + ".sample.json", "w", encoding="utf-8") as fh:
                json.dump({"prompts": args.prompts, **sample_meta}, fh, indent=2)

    summary = {"n_total": len(prompts), "n_err": n_err,
               "elapsed_s": round(time.time() - t0, 1),
               "out": None if args.smoke else args.out}
    if sample_meta:
        summary["sample"] = sample_meta
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--prompts", default="sft/eval/data/eval_prompts_test_seen.jsonl")
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="generate for about this many turns instead of the "
                         "whole prompt file -- see --limit-mode")
    ap.add_argument("--limit-mode", choices=["convo", "head"], default="convo",
                    help="convo (default): sample WHOLE conversations at random "
                         "under --seed until --limit turns are covered. head: "
                         "the first --limit rows of the prompt file, which is a "
                         "contiguous block of conversations and NOT a random "
                         "sample of test_seen (kept only to reproduce already-"
                         "published subset artifacts).")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --limit-mode convo conversation sampling")
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
