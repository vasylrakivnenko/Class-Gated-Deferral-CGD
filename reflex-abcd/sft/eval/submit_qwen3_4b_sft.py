"""Submit a Fireworks SFT job for Qwen3-4B-Instruct-2507 on the SAME
unconstrained-arm training data and hyperparameters used for the
Ministral-3-3B run, with held-out eval enabled.

WHY QWEN3-4B RATHER THAN MINISTRAL-3-3B: the ministral3 family is currently
unservable on this account. Both serving paths hit the same known Fireworks
engine regression (text-completion build inside the blocked range
[4.349.3, 4.427.1)): live-merge is refused at deployment validation, and
multi-LoRA validates but then crashes at inference ("fault filter abort",
then 404). Qwen3 is proven servable here -- the qwen3-0.6B addon from this
project still validates for live-merge today.

The dataset object is REUSED verbatim from the Ministral run, so both arms
train on a provably identical dataset (hence its "-mistral3b" name).

Hyperparameters match the Ministral run exactly (LoRA r=8, 2 epochs,
lr 1e-4). The only addition is eval_auto_carveout=True, which holds out a
slice to produce a dev-loss curve comparable to the local SmolLM2 runs.

Training is billed per token as a one-off. This creates NO deployment and
therefore incurs NO hourly GPU billing.

USAGE
-----
    python sft/eval/submit_qwen3_4b_sft.py
"""

from __future__ import annotations

import os
import sys

ACCOUNT_ID = "vasyl-r"
DATASET = "accounts/vasyl-r/datasets/abcd-unconstrained-train-mistral3b"
BASE_MODEL = "accounts/fireworks/models/qwen3-4b-instruct-2507"
JOB_ID = "abcd-unconstrained-qwen3-4b-v1"


def main() -> int:
    from fireworks import Fireworks

    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT_ID)

    d = client.datasets.get(DATASET.rsplit("/", 1)[-1], account_id=ACCOUNT_ID)
    if getattr(d, "state", None) != "READY":
        print(f"dataset not READY (state={getattr(d, 'state', None)}), aborting", file=sys.stderr)
        return 1
    print(f"reusing dataset {DATASET} (state=READY)", file=sys.stderr)

    print(f"submitting SFT job {JOB_ID} (base={BASE_MODEL.rsplit('/', 1)[-1]}, "
          f"LoRA r=8, 2 epochs, lr=1e-4, eval_auto_carveout=True) ...", file=sys.stderr)
    job = client.supervised_fine_tuning_jobs.create(
        account_id=ACCOUNT_ID,
        supervised_fine_tuning_job_id=JOB_ID,
        dataset=DATASET,
        base_model=BASE_MODEL,
        epochs=2,
        lora_rank=8,
        learning_rate=1e-4,
        eval_auto_carveout=True,
        output_model=f"accounts/{ACCOUNT_ID}/models/{JOB_ID}",
        display_name="ABCD unconstrained arm, Qwen3-4B-Instruct LoRA SFT",
    )
    print(f"submitted: {getattr(job, 'name', job)}", file=sys.stderr)
    cost = getattr(job, "estimated_cost", None)
    if cost is not None:
        print(f"estimated cost: {cost}", file=sys.stderr)
    print("NOTE: training only -- no deployment created, no hourly GPU billing.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
