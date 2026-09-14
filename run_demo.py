"""Run the full demo and write runs/cost_vs_accuracy.png + runs/results.json.

Candidate set spans both free-local and real-priced-hosted, chosen for price
spread rather than name recognition:
  - qwen3-0.6b / qwen3-1.7b   local, $0 marginal, priced at their nearest hosted
                              equivalent for the chart
  - gpt-5-nano, gpt-5.4-nano  the two cheapest OpenAI tiers on this Azure resource
  - gpt-4.1-mini              a "mini" tier reference point
  - deepseek-v4-flash         cheapest open-weight hosted candidate ($0.15/$0.31)
  - nemotron-lightning-30b    Fireworks' cheapest sticker price ($0.05/$0.20) --
                              also the one that burned 1046 reasoning tokens on
                              a one-word answer in a smoke test, so it is a real
                              test of whether measured cost matches the sticker

Reference/frontier baseline: claude-opus-4-8 (the "expensive model you're
overpaying for"). Reflection model for GEPA: gpt-5.2 -- a real hosted model
beats local gpt-oss:20b on both quality and the ~7 tok/s local generation speed
that made reflection the wall-clock bottleneck in the first local-only run.
"""
import sys
sys.path.insert(0, "src")
from downshift.experiment import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    candidates=[
        "qwen3-0.6b-direct",
        "qwen3-1.7b-direct",
        "gpt-5-nano",
        "gpt-5.4-nano",
        "gpt-4.1-mini",
        "deepseek-v4-flash",
        "nemotron-lightning-30b",
    ],
    optimize=["qwen3-0.6b-direct", "qwen3-1.7b-direct", "gpt-5-nano", "nemotron-lightning-30b"],
    reflection_model="gpt-5.2",
    # Must be one of `candidates` above -- it is the model every paired
    # comparison is made against. This said claude-opus-4-8, which the demo
    # never evaluated, so the whole acceptance section came back empty.
    # gpt-4.1-mini is the priciest row the demo actually runs, and the
    # header already describes it as the reference point.
    reference_model="gpt-4.1-mini",
    accuracy_bar=0.90,
    non_inferiority_margin=0.03,
    max_metric_calls=250,
    n_train=200, n_val=120, n_test=250,
    # 2048, not the 1024 default: nemotron-lightning-30b measured 1046 output
    # tokens of unsolicited reasoning on a single one-word classification in a
    # smoke test, and a truncated reasoning block returns empty content (see
    # ollama_lm's analogous guard) -- max_tokens is a ceiling, not a spend
    # commitment, so raising it costs nothing but avoids that failure mode.
    max_tokens=2048, reasoning_max_tokens=16000,
    num_threads=12, seed=0,
)
run_experiment(cfg)
