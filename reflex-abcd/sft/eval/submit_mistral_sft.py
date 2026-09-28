"""Submit a Fireworks SFT job for Ministral-3-3B-Instruct on the SAME
unconstrained-arm training data used for qwen3-0.6B this session
(sft/data/abcd_train_lexical.jsonl) -- LoRA, 2 epochs (matches this
project's own local SmolLM2 methodology), NO deployment created.

IMPORTANT: this script only creates the dataset + submits the training
job. It deliberately does NOT create a Fireworks deployment afterward --
that's what incurs hourly dedicated-GPU billing. Deploying/generating/
scoring is a separate, explicit step to run later.

USAGE
-----
    python sft/eval/submit_mistral_sft.py
"""

from __future__ import annotations

import os
import sys
import time


def main() -> int:
    from fireworks import Fireworks

    account_id = "vasyl-r"
    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=account_id)

    dataset_id = "abcd-unconstrained-train-mistral3b"
    print(f"creating dataset {dataset_id}...", file=sys.stderr)
    try:
        client.datasets.create(
            account_id=account_id,
            dataset_id=dataset_id,
            dataset={"display_name": "ABCD unconstrained train (Mistral-3B SFT)",
                    "format": "CHAT", "example_count": "71133"},
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
        print("dataset already exists, reusing", file=sys.stderr)

    print("uploading training file (82MB, may take a bit)...", file=sys.stderr)
    with open("sft/data/abcd_train_lexical.jsonl", "rb") as fh:
        client.datasets.upload(dataset_id, account_id=account_id, file=fh)

    print("waiting for dataset to become READY...", file=sys.stderr)
    for _ in range(60):
        d = client.datasets.get(dataset_id, account_id=account_id)
        state = getattr(d, "state", None)
        print(f"  dataset state={state}", file=sys.stderr)
        if state == "READY":
            break
        time.sleep(10)
    else:
        print("dataset did not become READY in time, aborting job submission", file=sys.stderr)
        return 1

    job_id = "abcd-unconstrained-mistral3b-v1"
    print(f"submitting SFT job {job_id} (base=ministral-3-3b-instruct-2512, LoRA, 2 epochs)...",
          file=sys.stderr)
    job = client.supervised_fine_tuning_jobs.create(
        account_id=account_id,
        supervised_fine_tuning_job_id=job_id,
        dataset=f"accounts/{account_id}/datasets/{dataset_id}",
        base_model="accounts/fireworks/models/ministral-3-3b-instruct-2512",
        epochs=2,
        lora_rank=8,
        learning_rate=1e-4,
        output_model=f"accounts/vasyl-r/models/{job_id}",
        display_name="ABCD unconstrained arm, Ministral-3-3B LoRA SFT",
    )
    print(f"submitted: {job.name if hasattr(job, 'name') else job}", file=sys.stderr)
    print("NOTE: training only -- no deployment created. Check job status with "
          "check_mistral_sft.py before deploying.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
