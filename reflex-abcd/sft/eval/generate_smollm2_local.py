"""Generate against a locally fine-tuned SmolLM2-360M checkpoint for every
prompt in the eval set, writing {turn_id, generation, error} JSONL in the
SAME shape gen_unconstrained.jsonl uses, so score_generations.py scores this
arm identically to every other arm this session (compose@1, conditional +
unconditional, against the certified bank).

USAGE
-----
    PYTHONPATH=. python sft/eval/generate_smollm2_local.py \
        --checkpoint sft/eval/checkpoints/smollm2_360m_dryrun_15t \
        --out sft/eval/data/gen_smollm2_dryrun_15t.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time


SYSTEM_PROMPT = (
    "You are a customer service agent. Given the conversation so far and "
    "the facts the customer has already disclosed, write the agent's next "
    "message. Reply with that message only -- no preamble, no explanation, no quotes."
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", default="sft/eval/data/eval_prompts_test_seen.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--cpu-threads", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.cpu_threads)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    sys.path.insert(0, "sft/eval")
    from train_smollm2_local import render_prompt

    print(f"loading checkpoint: {args.checkpoint}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32)
    model.eval()
    pad_id = tok.pad_token_id or tok.eos_token_id

    rows = []
    with open(args.prompts, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f"generating for {len(rows)} prompts", file=sys.stderr)

    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as out_fh:
        for i, r in enumerate(rows):
            prompt_text = render_prompt(SYSTEM_PROMPT, r["prompt"])
            inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
            error = None
            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                        pad_token_id=pad_id)
                gen_ids = out[0][inputs["input_ids"].shape[1]:]
                generation = tok.decode(gen_ids, skip_special_tokens=True).strip()
            except Exception as e:  # pragma: no cover
                generation = ""
                error = f"{type(e).__name__}: {str(e)[:200]}"
            out_fh.write(json.dumps({"turn_id": r["turn_id"], "generation": generation,
                                     "error": error}, ensure_ascii=False) + "\n")
            if (i + 1) % args.log_every == 0 or (i + 1) == len(rows):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(rows) - i - 1) / rate if rate > 0 else 0
                print(f"progress: {i+1}/{len(rows)} rate={rate:.2f}/s eta={eta:.0f}s",
                      file=sys.stderr)

    print(f"done. wrote {args.out} in {time.time()-t0:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
