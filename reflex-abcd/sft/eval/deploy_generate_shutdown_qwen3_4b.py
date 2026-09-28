"""End-to-end, cost-bounded pipeline for the Qwen3-4B LoRA SFT model on
Fireworks: create a LIVE-MERGE dedicated deployment of the fine-tuned model,
generate against the SAME 500-prompt eval slice used for the SmolLM2
checkpoints, score compose@1, THEN DELETE THE DEPLOYMENT -- so this never
runs up hourly GPU billing beyond the single generation pass. Teardown runs
in a `finally` block so it fires even if generation fails partway through.

The PEFT addon is deployed directly as the deployment's base_model, which
makes Fireworks live-merge it, matching sft/eval/generate.py's working
qwen3-0.6B path. This exact config was confirmed with a zero-cost
deployments.create(validate_only=True) before any billed resource existed.

USAGE
-----
    python sft/eval/deploy_generate_shutdown_qwen3_4b.py
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import sys
import time

ACCOUNT_ID = "vasyl-r"
BASE_MODEL = "accounts/fireworks/models/qwen3-4b-instruct-2507"
FT_MODEL = "accounts/vasyl-r/models/abcd-unconstrained-qwen3-4b-v1"
DEPLOYMENT_ID = os.environ.get("EVAL_DEPLOYMENT_ID", "abcd-qwen3-4b-eval2")
EVAL_START = int(os.environ.get("EVAL_START", "0"))
EVAL_END = int(os.environ.get("EVAL_END", "500"))
OUT_PATH = os.environ.get("EVAL_OUT", "sft/eval/data/gen_qwen3_4b.jsonl")
SYSTEM_PROMPT = (
    "You are a customer service agent. Given the conversation so far and "
    "the facts the customer has already disclosed, write the agent's next "
    "message. Reply with that message only -- no preamble, no explanation, no quotes."
)


def _poll_tolerant(fn, ok_state, max_tries, sleep_s, label):
    """Poll fn() -> obj with a .state attribute, tolerating transient network
    errors (a prior run crashed on one SSL handshake timeout mid-poll, which
    left a READY billing deployment orphaned since the crash skipped
    teardown too)."""
    for i in range(max_tries):
        try:
            obj = fn()
            state = getattr(obj, "state", None)
            print(f"  {label} state={state}", file=sys.stderr)
            if state == ok_state:
                return obj
        except Exception as e:
            print(f"  {label} poll error (tolerated): {e}", file=sys.stderr)
        time.sleep(sleep_s)
    raise RuntimeError(f"{label} did not reach {ok_state} in time")


DELETE_CONFIRM_TRIES = 3
DELETE_CONFIRM_SLEEP_S = 5


def _looks_not_found(exc) -> bool:
    """True only for an unambiguous 'this deployment does not exist' error.
    A network/auth error must NOT be read as confirmation of deletion."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return "notfound" in text or "not found" in text or "404" in text


def _confirm_deleted(client):
    """Return None once the deployment is provably gone or provably DELETING;
    otherwise return the last observed state (or an 'unknown ...' string)."""
    state = None
    for _ in range(DELETE_CONFIRM_TRIES):
        try:
            obj = client.deployments.get(DEPLOYMENT_ID, account_id=ACCOUNT_ID)
        except Exception as e:
            if _looks_not_found(e):
                return None
            state = f"unknown (state read failed: {type(e).__name__}: {str(e)[:120]})"
            time.sleep(DELETE_CONFIRM_SLEEP_S)
            continue
        state = getattr(obj, "state", None)
        if state is not None and str(state).upper().startswith("DELET"):
            return None
        time.sleep(DELETE_CONFIRM_SLEEP_S)
    return state if state is not None else "unknown (state never read)"


def _teardown(client, deployment_name):
    """Delete the deployment and VERIFY it is gone.

    A failed teardown is a HARD failure, never a warning: an undeleted
    deployment is an hourly-billed GPU that outlives this process, so this
    raises RuntimeError rather than returning, and the caller must let that
    make the process exit NON-ZERO.
    """
    print(f"tearing down deployment {deployment_name} ...", file=sys.stderr)
    last = None
    for attempt in range(10):
        try:
            # ignore_checks is required: Fireworks refuses to delete a
            # deployment that served inference in the last hour, which is
            # always true right after a generation run.
            client.deployments.delete(DEPLOYMENT_ID, account_id=ACCOUNT_ID,
                                      ignore_checks=True)
            print("deployment delete requested.", file=sys.stderr)
            still = _confirm_deleted(client)
            if still is None:
                print("teardown CONFIRMED: deployment is gone or deleting.", file=sys.stderr)
                return
            last = f"delete request accepted but deployment is still state={still}"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  teardown attempt {attempt+1}/10 failed: {last} -- retrying in 8s",
              file=sys.stderr)
        time.sleep(8)
    try:
        obj = client.deployments.get(DEPLOYMENT_ID, account_id=ACCOUNT_ID)
        final_state = getattr(obj, "state", "unknown")
    except Exception as e:
        final_state = ("gone (404 on final read)" if _looks_not_found(e)
                       else f"unknown (final read failed: {type(e).__name__})")
    raise RuntimeError(
        f"TEARDOWN FAILED after 10 attempts; deployment {deployment_name} "
        f"state={final_state}; last error: {last} -- CHECK THE FIREWORKS CONSOLE "
        f"AND DELETE THIS DEPLOYMENT MANUALLY, IT IS PROBABLY STILL BILLING.")


REQUEST_TIMEOUT_S = 20


async def _generate_one(client, route, prompt, sem):
    async with sem:
        err = None
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=route,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": prompt}],
                        max_tokens=80, temperature=0,
                    ),
                    timeout=REQUEST_TIMEOUT_S,
                )
                return resp.choices[0].message.content or "", None
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(2 * (attempt + 1))
        return "", err


async def _generate_all(route, prompts, concurrency=8, out_path=None):
    """Writes each result to out_path AS IT COMES IN (not just at the end) so
    a stall/kill partway through does not lose already-completed generations
    -- the prior run lost 200/500 completed generations this way.
    """
    from fireworks import AsyncFireworks
    client = AsyncFireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT_ID)
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(prompts)
    out_fh = open(out_path, "w", encoding="utf-8") if out_path else None
    n_done = 0

    async def _worker(i, turn_id, prompt):
        nonlocal n_done
        text, err = await _generate_one(client, route, prompt, sem)
        results[i] = (turn_id, text, err)
        n_done += 1
        if out_fh:
            out_fh.write(json.dumps({"turn_id": turn_id, "generation": text,
                                     "error": err}, ensure_ascii=False) + "\n")
            out_fh.flush()
        if n_done % 25 == 0 or n_done == len(prompts):
            print(f"  generated {n_done}/{len(prompts)}", file=sys.stderr)

    t0 = time.time()
    await asyncio.gather(*[_worker(i, tid, p) for i, (tid, p) in enumerate(prompts)])
    print(f"generation done in {time.time()-t0:.0f}s", file=sys.stderr)
    if out_fh:
        out_fh.close()
    await client.close()
    return results


def main() -> int:
    from fireworks import Fireworks
    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT_ID)

    print(f"creating live-merge dedicated deployment of FT model {FT_MODEL} ...", file=sys.stderr)
    dep = client.deployments.create(
        account_id=ACCOUNT_ID,
        base_model=FT_MODEL,
        deployment_id=DEPLOYMENT_ID,
        display_name="abcd-qwen3-4b-eval (temporary, torn down after scoring)",
        min_replica_count=1,
        max_replica_count=1,
        accelerator_type="NVIDIA_H100_80GB",
        accelerator_count=1,
        expire_time=datetime.datetime.now(datetime.timezone.utc)
                    + datetime.timedelta(minutes=45),
    )
    deployment_name = dep.name
    print(f"deployment: {deployment_name}", file=sys.stderr)

    try:
        print("waiting for deployment to become READY ...", file=sys.stderr)
        _poll_tolerant(
            lambda: client.deployments.get(DEPLOYMENT_ID, account_id=ACCOUNT_ID),
            "READY", 90, 10, "deployment")

        # Live-merge deployments are addressed as
        #   <model-resource-path>#<deployment-resource-path>
        # exactly as sft/eval/generate.py does for the qwen3-0.6B addon.
        route = f"{FT_MODEL}#{deployment_name}"
        print(f"route: {route}", file=sys.stderr)

        prompts = []
        with open("sft/eval/data/eval_prompts_test_seen.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                prompts.append((r["turn_id"], r["prompt"]))
        prompts = prompts[EVAL_START:EVAL_END]
        print(f"prompt slice [{EVAL_START}:{EVAL_END}] -> {len(prompts)} prompts", file=sys.stderr)

        print("smoke-testing route with 1 single-shot request before committing "
              "(retrying -- 'DEPLOYED' state may lag actual routing readiness) ...",
              file=sys.stderr)
        smoke_text = None
        smoke_err = None
        for attempt in range(6):
            try:
                smoke_resp = client.chat.completions.create(
                    model=route,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": "customer: hi, i need help with my order."}],
                    max_tokens=40, temperature=0,
                )
                smoke_text = smoke_resp.choices[0].message.content
                break
            except Exception as e:
                smoke_err = e
                print(f"  smoke test attempt {attempt+1}/6 failed: {e} -- retrying in 10s",
                      file=sys.stderr)
                time.sleep(10)
        if smoke_text is None:
            raise RuntimeError(f"single-shot smoke test failed after 6 attempts: {smoke_err}")
        print(f"single-shot smoke test OK, generation: {smoke_text!r}", file=sys.stderr)

        print("smoke-testing 10 REAL eval prompts (concurrency + real content sanity) "
              "before committing to the full 500 ...", file=sys.stderr)
        smoke10 = asyncio.run(_generate_all(route, prompts[:10], concurrency=4, out_path=None))
        n_smoke_err = sum(1 for _, _, e in smoke10 if e)
        for tid, text, err in smoke10[:3]:
            print(f"  smoke[{tid}] -> {text[:120]!r}  err={err}", file=sys.stderr)
        if n_smoke_err > 0:
            raise RuntimeError(f"10-prompt smoke test had {n_smoke_err}/10 errors, "
                               f"aborting before the full 500-prompt run. first error: "
                               f"{next(e for _, _, e in smoke10 if e)}")
        print(f"10-prompt smoke test OK ({10 - n_smoke_err}/10 succeeded), "
              f"proceeding to full 500-prompt run ...", file=sys.stderr)

        out_path = OUT_PATH
        results = asyncio.run(_generate_all(route, prompts, concurrency=8, out_path=out_path))
        n_err = sum(1 for _, _, e in results if e)
        print(json.dumps({"wrote": out_path, "n": len(results), "n_err": n_err}, indent=2))

    finally:
        # Capture whether an exception is ALREADY propagating before calling
        # _teardown: inside the except below sys.exc_info() would report the
        # teardown error, not the original one.
        already_failing = sys.exc_info()[0] is not None
        try:
            _teardown(client, deployment_name)
        except Exception as teardown_exc:
            print(f"CRITICAL: {teardown_exc}", file=sys.stderr)
            if not already_failing:
                raise
            print("CRITICAL: teardown failed while another error was already "
                  "propagating -- that error is re-raised, so this process "
                  "exits NON-ZERO. DELETE THE DEPLOYMENT MANUALLY.",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
