"""LLM-as-judge pass over sft/eval/data/llm_judge_sample.jsonl (built by
build_llm_judge_sample.py): for each compose@1 MISS from the certified
selector, show a judge model the conversation context, the gold (reference)
answer, and the cache's actual predicted answer, and ask it to rate the
predicted answer's adequacy GIVEN that reference.

This does not change or re-derive compose@1. It answers a different question:
of the turns the mechanical metric calls wrong, how many are wrong in a way a
person would actually care about, versus a reasonable alternate phrasing/order
that the strict template-id match cannot credit? The exact-match number stays
the reported headline; this is a qualitative read on its failure mode.

JUDGE MODEL: accounts/fireworks/models/glm-5p3-flash (serverless, no
fine-tuning -- pure inference). Confirmed present via client.models.list().
It is a REASONING model: `reasoning_content` is generated before `content`
and both count against `max_tokens`, so this must be set high enough that
reasoning does not crowd out the JSON answer (a 150-token budget left content
empty or truncated on ~half the sample; 700 is used here).

CAVEAT (report this alongside every number this script produces): LLM judges
have a documented leniency bias -- they tend to call things "fine" more often
than a careful human would. Treat this as a directional signal, not a
validated accuracy, the same status this project gave the act labeller (D24)
before a human audit existed.

USAGE
-----
    python sft/eval/run_llm_judge.py \
        --sample sft/eval/data/llm_judge_sample.jsonl \
        --out sft/eval/data/llm_judge_results.jsonl \
        --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

JUDGE_SYSTEM = (
    "You are an impartial quality reviewer for a customer-service chat system. "
    "You will see a conversation so far, a REFERENCE reply that is known to be "
    "correct and appropriate, and a CANDIDATE reply that a different (automated) "
    "system produced instead. Using the reference as your guide to what a good "
    "reply looks like in this situation, judge whether the CANDIDATE reply is "
    "ALSO an adequate, appropriate thing to say to the customer at this point -- "
    "not whether it is worded identically to the reference. "
    "Respond with ONLY a JSON object, no other text, of the form: "
    '{"verdict": "appropriate"|"borderline"|"wrong", "reason": "<one sentence>"}. '
    "Use \"appropriate\" if the candidate is a valid, helpful reply even if "
    "phrased differently or taking a slightly different but still correct path. "
    "Use \"borderline\" if it is partially adequate but has a real gap, "
    "ambiguity, or minor mismatch with the situation. "
    "Use \"wrong\" if it is a materially incorrect action, misleading, "
    "unhelpful, or not responsive to the conversation."
)


def _build_prompt(row: dict) -> str:
    return (
        f"CONVERSATION SO FAR:\n{row['context']}\n\n"
        f"REFERENCE (known-correct) REPLY:\n{row['gold_text']}\n\n"
        f"CANDIDATE REPLY (produced by the automated system instead):\n"
        f"{row['predicted_text']}\n\n"
        "Judge the CANDIDATE reply as instructed."
    )


def _parse_verdict(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"verdict": "parse_error", "reason": text[:200]}
    try:
        obj = json.loads(m.group(0))
        v = str(obj.get("verdict", "")).strip().lower()
        if v not in ("appropriate", "borderline", "wrong"):
            v = "parse_error"
        return {"verdict": v, "reason": obj.get("reason", "")}
    except Exception:
        return {"verdict": "parse_error", "reason": text[:200]}


async def _judge_one(client, model, row, sem, max_tokens):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": _build_prompt(row)}],
                max_tokens=max_tokens,
                temperature=0,
            )
            text = resp.choices[0].message.content or ""
            parsed = _parse_verdict(text)
            return {**row, **parsed, "raw_judge_output": text, "error": None}
        except Exception as e:
            return {**row, "verdict": "call_error", "reason": "",
                    "raw_judge_output": "", "error": f"{type(e).__name__}: {str(e)[:200]}"}


async def _run(args):
    from fireworks import AsyncFireworks
    client = AsyncFireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id="fireworks")

    rows = []
    with open(args.sample, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    results = await asyncio.gather(*[
        _judge_one(client, args.model, row, sem, args.max_tokens) for row in rows
    ])

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    n_scored = len(results) - counts.get("call_error", 0) - counts.get("parse_error", 0)

    summary = {
        "model": args.model,
        "n_total": len(results),
        "counts": counts,
        "n_scored": n_scored,
        "share_appropriate_of_scored": (counts.get("appropriate", 0) / n_scored) if n_scored else None,
        "share_borderline_of_scored": (counts.get("borderline", 0) / n_scored) if n_scored else None,
        "share_wrong_of_scored": (counts.get("wrong", 0) / n_scored) if n_scored else None,
        "elapsed_s": round(time.time() - t0, 1),
        "caveat": ("LLM-judge leniency bias is a known phenomenon -- treat these shares as a "
                  "directional read on compose@1's misses, not a validated accuracy. Not "
                  "human-audited (same open-caveat status as the act labeller, DECISIONS D24)."),
        "note": ("These 'wrong'/'borderline'/'appropriate' turns are ALL compose@1 MISSES by "
                "the mechanical metric -- this only asks how bad the misses are, it does not "
                "change the headline 27.7%/12.4% compose@1 numbers."),
    }
    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="sft/eval/data/llm_judge_sample.jsonl")
    ap.add_argument("--out", default="sft/eval/data/llm_judge_results.jsonl")
    ap.add_argument("--model", default="accounts/fireworks/models/glm-5p3-flash")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
