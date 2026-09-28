"""Run the SAME judge protocol (run_llm_judge.py's prompt/parsing) across
MULTIPLE judge models on the SAME 100-row sample, then report inter-judge
agreement -- not just each model's own verdict shares.

WHY: one judge's 75.5% appropriate / 15.3% borderline / 9.2% wrong (D26) is
one model's opinion, and LLM-judges are known to be lenient in ways that
don't generalize across model families. Comparing GLM-5.3-Flash against its
larger sibling GLM-5.3 and against a DIFFERENT model family (gpt-oss-120b)
tells us whether the "most misses are fine, just non-gold phrasing" reading
is a judge-model artifact or something three different models converge on.

METRIC: exact 3-way verdict agreement (appropriate/borderline/wrong) between
every judge pair, PLUS a coarser "adequate" (appropriate+borderline) vs
"wrong" agreement, since the appropriate/borderline line is fuzzier than the
wrong/not-wrong line and conflating them would understate real agreement.

USAGE
-----
    python sft/eval/run_llm_judge_multi.py \
        --sample sft/eval/data/llm_judge_sample.jsonl \
        --out-dir sft/eval/data/judge_multi \
        --models accounts/fireworks/models/glm-5p3-flash,accounts/fireworks/models/glm-5p3,accounts/fireworks/models/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_llm_judge import _judge_one, JUDGE_SYSTEM  # noqa: E402


def _short_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


async def _run_one_model(client, model, rows, concurrency, max_tokens, out_dir):
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    results = await asyncio.gather(*[
        _judge_one(client, model, row, sem, max_tokens) for row in rows
    ])
    name = _short_name(model)
    path = os.path.join(out_dir, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    n_scored = sum(v for k, v in counts.items() if k in ("appropriate", "borderline", "wrong"))
    print(f"  {name}: {counts}  n_scored={n_scored}  {time.time()-t0:.1f}s", file=sys.stderr)
    return name, results


def _agreement(models_results: dict[str, list[dict]], rows: list[dict]) -> dict:
    by_turn = {name: {r["turn_id"]: r["verdict"] for r in res} for name, res in models_results.items()}
    names = list(models_results.keys())

    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            n_both = n_exact = n_coarse = 0
            for row in rows:
                tid = row["turn_id"]
                va, vb = by_turn[a].get(tid), by_turn[b].get(tid)
                if va not in ("appropriate", "borderline", "wrong") or vb not in ("appropriate", "borderline", "wrong"):
                    continue
                n_both += 1
                if va == vb:
                    n_exact += 1
                ca = "adequate" if va in ("appropriate", "borderline") else "wrong"
                cb = "adequate" if vb in ("appropriate", "borderline") else "wrong"
                if ca == cb:
                    n_coarse += 1
            pairwise[f"{a} vs {b}"] = {
                "n_compared": n_both,
                "exact_3way_agreement": (n_exact / n_both) if n_both else None,
                "coarse_adequate_vs_wrong_agreement": (n_coarse / n_both) if n_both else None,
            }

    n_all = all_exact = all_coarse = 0
    for row in rows:
        tid = row["turn_id"]
        verdicts = [by_turn[n].get(tid) for n in names]
        if any(v not in ("appropriate", "borderline", "wrong") for v in verdicts):
            continue
        n_all += 1
        if len(set(verdicts)) == 1:
            all_exact += 1
        coarse = {("adequate" if v in ("appropriate", "borderline") else "wrong") for v in verdicts}
        if len(coarse) == 1:
            all_coarse += 1

    return {
        "pairwise": pairwise,
        "all_models_unanimous": {
            "n_compared": n_all,
            "exact_3way": (all_exact / n_all) if n_all else None,
            "coarse_adequate_vs_wrong": (all_coarse / n_all) if n_all else None,
        },
    }


async def _run(args):
    from fireworks import AsyncFireworks
    client = AsyncFireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id="fireworks")

    rows = []
    with open(args.sample, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    models_results: dict[str, list[dict]] = {}
    print(f"judging {len(rows)} rows x {len(models)} models...", file=sys.stderr)
    for model in models:
        name, results = await _run_one_model(client, model, rows, args.concurrency, args.max_tokens, args.out_dir)
        models_results[name] = results

    per_model_summary = {}
    for name, results in models_results.items():
        counts: dict[str, int] = {}
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        n_scored = sum(v for k, v in counts.items() if k in ("appropriate", "borderline", "wrong"))
        per_model_summary[name] = {
            "counts": counts, "n_scored": n_scored,
            "share_appropriate": (counts.get("appropriate", 0) / n_scored) if n_scored else None,
            "share_borderline": (counts.get("borderline", 0) / n_scored) if n_scored else None,
            "share_wrong": (counts.get("wrong", 0) / n_scored) if n_scored else None,
        }

    agreement = _agreement(models_results, rows)

    summary = {
        "models": models,
        "n_rows": len(rows),
        "per_model": per_model_summary,
        "agreement": agreement,
        "caveat": ("All judges are LLMs, and LLM-judge leniency bias is well documented -- "
                  "cross-model agreement shows whether models CONVERGE, it does not mean "
                  "they are CORRECT. No human audit of this sample exists yet."),
    }
    out_path = os.path.join(args.out_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="sft/eval/data/llm_judge_sample.jsonl")
    ap.add_argument("--out-dir", default="sft/eval/data/judge_multi")
    ap.add_argument("--models", default=(
        "accounts/fireworks/models/glm-5p3-flash,"
        "accounts/fireworks/models/glm-5p3,"
        "accounts/fireworks/models/gpt-oss-120b"
    ))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
