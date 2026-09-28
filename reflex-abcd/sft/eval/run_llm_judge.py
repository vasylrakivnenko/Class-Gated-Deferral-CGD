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

ATTRITION IS NEVER SILENT. A row whose response cannot be parsed is almost
always budget exhaustion (reasoning consumed all of max_tokens, leaving
`content` empty or a JSON object truncated mid-`reason`), not a malformed
answer. Such a row used to be dropped from the denominator with no retry, so
every reported share was computed over a non-random subset -- observed at up
to 12/100 rows for glm-5p3. Now: each row is retried with a DOUBLED token
budget before it is given up on, `finish_reason` is recorded so truncation is
distinguishable from genuinely malformed JSON, and the summary reports BOTH
denominators (`*_of_scored` and `*_of_all`) plus `n_unscored`. If more than
5% of rows end up unscored the run exits non-zero with a loud banner --
a share computed on 88 of 100 rows is not a share of the sample.

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

#: Above this share of rows ending with no usable verdict, the run's shares are
#: not a read on the sample any more and the script exits non-zero.
MAX_UNSCORED_SHARE = 0.05

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


def _weighted_shares(results: list[dict]) -> dict | None:
    """Stratum-weighted verdict shares, or None if the sample is unweighted.

    Denominator is the summed weight of the rows that produced a verdict, not
    the row count -- a Hajek estimator of the population share, so this arm can
    be compared against the simple-random arms.
    """
    has_w = [r for r in results if isinstance(r.get("stratum_weight"), (int, float))]
    if not has_w:
        return None
    scored = [r for r in has_w
              if r["verdict"] in ("appropriate", "borderline", "wrong")]
    total_w = sum(r["stratum_weight"] for r in scored)
    if not total_w:
        return None

    def share(v: str) -> float:
        return sum(r["stratum_weight"] for r in scored if r["verdict"] == v) / total_w

    return {
        "design": "stratum-weighted (equal-allocation strata reweighted to population)",
        "n_weighted_rows": len(has_w),
        "n_weighted_rows_scored": len(scored),
        "sum_weight_all": sum(r["stratum_weight"] for r in has_w),
        "sum_weight_scored": total_w,
        # Kish effective sample size of the scored rows: how many equally
        # weighted rows these shares are really worth. Equal allocation over
        # very unequal strata makes this far smaller than n_weighted_rows_scored.
        "kish_effective_n": total_w ** 2 / sum(r["stratum_weight"] ** 2 for r in scored),
        "share_appropriate_weighted": share("appropriate"),
        "share_borderline_weighted": share("borderline"),
        "share_wrong_weighted": share("wrong"),
        "note": ("Quote these when placing this arm beside simple-random arms; the "
                "unweighted share_*_of_scored above averages over act-sequence strata, "
                "not over turns. sum_weight_scored < sum_weight_all means unscored "
                "rows carried weight that is missing from these shares. Read the "
                "weighted shares against kish_effective_n, not n: they are only as "
                "precise as a simple random sample of that many rows."),
    }


async def _judge_one(client, model, row, sem, max_tokens, max_retries=3):
    """Judge one row, retrying an unparseable response with a DOUBLED token
    budget (and a raised call after a backoff, same budget) before giving up.

    An empty or truncated `content` from a reasoning model means the budget
    was spent on `reasoning_content`, so the correct response is more budget,
    not a silent drop -- dropping it biases the denominator toward whichever
    rows happened to be cheap to reason about.
    """
    async with sem:
        budget = max_tokens
        attempt = 0
        last: dict = {}
        while attempt <= max_retries:
            attempt += 1
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                             {"role": "user", "content": _build_prompt(row)}],
                    max_tokens=budget,
                    temperature=0,
                )
                choice = resp.choices[0]
                text = choice.message.content or ""
                parsed = _parse_verdict(text)
                last = {**row, **parsed, "raw_judge_output": text, "error": None,
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "attempts": attempt, "max_tokens_used": budget}
                if parsed["verdict"] != "parse_error":
                    return last
                budget *= 2  # reasoning ate the budget -- give it more, don't drop the row
            except Exception as e:
                last = {**row, "verdict": "call_error", "reason": "",
                        "raw_judge_output": "", "finish_reason": None,
                        "attempts": attempt, "max_tokens_used": budget,
                        "error": f"{type(e).__name__}: {str(e)[:200]}"}
                # A raised call (rate limit, transient 5xx) is not a budget
                # problem: keep the budget, back off, then retry.
                if attempt <= max_retries:
                    await asyncio.sleep(min(2 ** attempt, 30))
        return last


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
        _judge_one(client, args.model, row, sem, args.max_tokens,
                   max_retries=args.max_retries) for row in rows
    ])

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    n_all = len(results)
    n_dropped = counts.get("call_error", 0) + counts.get("parse_error", 0)
    # `n_scored` is the SUBSET that produced a usable verdict; `n_all` is the
    # sample. Both denominators are reported because they differ whenever the
    # judge failed to answer, and a share of the subset is not a share of the
    # sample.
    n_scored = n_all - n_dropped
    dropped_share = (n_dropped / n_all) if n_all else 0.0
    finish_reasons: dict[str, int] = {}
    for r in results:
        if r["verdict"] in ("parse_error", "call_error"):
            fr = str(r.get("finish_reason"))
            finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

    # Stratified samples (build_llm_judge_sample.py) carry a per-row design
    # weight, because their allocation is equal-per-stratum rather than
    # proportional -- an unweighted share there averages over failure MODES,
    # not over turns, and is not comparable to a simple-random arm's share.
    # Simple-random samples carry no weight and get no weighted block.
    weighted = _weighted_shares(results)

    summary = {
        "model": args.model,
        "n_total": n_all,
        "counts": counts,
        "n_scored": n_scored,
        "n_dropped": n_dropped,
        "n_unscored": n_dropped,
        "dropped_share_of_all": dropped_share,
        "dropped_finish_reasons": finish_reasons,
        "max_retries": args.max_retries,
        "share_appropriate_of_scored": (counts.get("appropriate", 0) / n_scored) if n_scored else None,
        "share_borderline_of_scored": (counts.get("borderline", 0) / n_scored) if n_scored else None,
        "share_wrong_of_scored": (counts.get("wrong", 0) / n_scored) if n_scored else None,
        # Denominator is every row in the sample, unscored rows included -- the
        # conservative reading, and the one comparable across judge models with
        # different drop rates.
        "share_appropriate_of_all": (counts.get("appropriate", 0) / n_all) if n_all else None,
        "share_borderline_of_all": (counts.get("borderline", 0) / n_all) if n_all else None,
        "share_wrong_of_all": (counts.get("wrong", 0) / n_all) if n_all else None,
        "attrition_ok": dropped_share <= MAX_UNSCORED_SHARE,
        "weighted": weighted,
        "elapsed_s": round(time.time() - t0, 1),
        "caveat": ("LLM-judge leniency bias is a known phenomenon -- treat these shares as a "
                  "directional read on compose@1's misses, not a validated accuracy. Not "
                  "human-audited (same open-caveat status as the act labeller, DECISIONS D24)."),
        "note": ("These 'wrong'/'borderline'/'appropriate' turns are ALL compose@1 MISSES by "
                "the mechanical metric -- this only asks how bad the misses are, it does not "
                "change the headline 27.7%/12.4% compose@1 numbers."),
        "denominator_note": ("*_of_scored divides by the rows that produced a usable verdict; "
                            "*_of_all divides by every row in the sample. They differ by "
                            f"n_unscored={n_dropped}. Quote *_of_all, or quote *_of_scored "
                            "WITH n_unscored beside it -- never *_of_scored alone."),
    }
    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

    if dropped_share > MAX_UNSCORED_SHARE:
        print(
            "\n" + "!" * 72 +
            f"\nATTRITION: {n_dropped}/{n_all} rows ({dropped_share:.1%}) produced no usable "
            f"verdict after {args.max_retries} retries."
            f"\nfinish_reason of dropped rows: {finish_reasons}"
            "\nThe *_of_scored shares in the summary are computed on a NON-RANDOM subset and must not "
            "be published as this sample's shares.\n" + "!" * 72,
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="sft/eval/data/llm_judge_sample.jsonl")
    ap.add_argument("--out", default="sft/eval/data/llm_judge_results.jsonl")
    ap.add_argument("--model", default="accounts/fireworks/models/glm-5p3-flash")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--max-retries", type=int, default=3,
                    help="retries for a row whose response could not be parsed; the "
                         "token budget DOUBLES on each retry (empty content means the "
                         "reasoning model ran out of budget, not that it answered badly)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
