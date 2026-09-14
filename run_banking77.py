"""Banking77 run — the harder second task, alongside (not replacing) the
Financial PhraseBank one. Results land in runs/banking77/.

77 intent classes over short customer-banking queries. Why this is a
different question from the sentiment task:
  - majority floor is 1.3% instead of 61.2%, so there is real headroom
  - the 77-label list makes every prompt ~1,930 input tokens (vs ~245), which
    both raises cost and is ~92% cacheable -- the run measures both
  - TF-IDF scores 91.6% given all 10k training rows but 21.2% given 2k, so the
    encoder-vs-prompting comparison finally has a data-scale dimension
"""
import sys
sys.path.insert(0, "src")
from downshift.experiment import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    task="banking77",
    candidates=[
        "qwen3-1.7b-direct",        # local, free
        "gpt-5.4-nano",
        "gpt-5-nano",
        "gpt-4.1-mini",
        "deepseek-v4-flash",
        "nemotron-lightning-30b",
        "claude-sonnet-4-6",
        "claude-opus-4-8",          # reference
    ],
    optimize=["gpt-5.4-nano", "deepseek-v4-flash"],
    reflection_model="gpt-5.2",
    reference_model="claude-opus-4-8",
    accuracy_bar=0.85,
    non_inferiority_margin=0.03,
    max_metric_calls=250,
    n_train=200, n_val=120, n_test=250,
    max_tokens=2048, reasoning_max_tokens=16000,
    num_threads=12, seed=0,
)
run_experiment(cfg)
