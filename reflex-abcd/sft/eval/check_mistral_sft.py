"""Check status of the Mistral-3B SFT job. Does NOT create a deployment --
that's a separate, explicit step (deploying incurs hourly dedicated-GPU
billing on Fireworks).

USAGE
-----
    python sft/eval/check_mistral_sft.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    from fireworks import Fireworks

    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id="vasyl-r")
    j = client.supervised_fine_tuning_jobs.get("abcd-unconstrained-mistral3b-v1", account_id="vasyl-r")
    print(f"state={j.state}")
    print(f"progress: epoch={j.job_progress.epoch} percent={j.job_progress.percent} "
          f"processed={j.job_progress.total_processed_requests}")
    print(f"output_model={j.output_model}")
    if j.state == "JOB_STATE_COMPLETED":
        print("\nDONE. To generate/score, you'll need to create a deployment "
              "(hourly billed) -- not done automatically by this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
