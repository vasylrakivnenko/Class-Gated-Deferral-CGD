"""Quick, interactive/spot-check generation against a locally fine-tuned
SmolLM2-360M checkpoint (train_smollm2_local.py's output). Two modes:

  --n-samples K   generate on K random held-out test_seen prompts and print
                  context / gold / generation side by side (default mode)
  --prompt "..."  generate on one free-form prompt (must already be in this
                  arm's "speaker|text ... state|..." context format to match
                  what the model was trained on -- see render_prompt)

Not a scored eval -- sft/eval/score_generations.py is the real metric. This
is just for a quick look/"does this feel coherent" check.

USAGE
-----
    python sft/eval/try_smollm2_local.py --checkpoint sft/eval/checkpoints/smollm2_360m_dryrun --n-samples 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from train_smollm2_local import render_prompt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="sft/eval/checkpoints/smollm2_360m_dryrun")
    ap.add_argument("--prompts", default="sft/eval/data/eval_prompts_test_seen.jsonl")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--prompt", default=None,
                    help="free-form user-turn context; overrides --n-samples")
    ap.add_argument("--system",
                    default="You are a customer service agent. Given the conversation "
                            "so far and the facts the customer has already disclosed, "
                            "write the agent's next message. Reply with that message "
                            "only -- no preamble, no explanation, no quotes.")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cpu"
    print(f"loading checkpoint: {args.checkpoint}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32)
    model.to(device)
    model.eval()

    def generate(user_text: str) -> str:
        prompt_text = render_prompt(args.system, user_text)
        inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen_ids, skip_special_tokens=True).strip()

    if args.prompt is not None:
        print(f"PROMPT: {args.prompt}\n")
        print(f"GENERATION: {generate(args.prompt)}")
        return 0

    rows = []
    with open(args.prompts, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n_samples, len(rows)))

    for r in sample:
        gold_text = " ".join(g["text"] for g in r.get("gold", []) if g.get("text"))
        gen = generate(r["prompt"])
        print("=" * 70)
        print(f"CONTEXT:\n{r['prompt']}\n")
        print(f"GOLD:  {gold_text or '(uncovered / no gold text)'}")
        print(f"MODEL: {gen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
