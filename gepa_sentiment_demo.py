"""
GEPA -> cheapest-model demo: optimize a 3-class sentiment classifier on a LOCAL ollama model.

Verified 2026-09-09 against dspy==3.3.1 / gepa==0.1.4, Python 3.14.3, ollama 0.33.3, Apple M4 Pro.

  uv venv --python 3.14 && uv pip install "dspy==3.3.1"
  ollama pull qwen3:1.7b && ollama pull gemma3:latest
  uv run python gepa_sentiment_demo.py
"""
import random
import time

import litellm  # noqa: F401  eager import: avoids a circular-import race when num_threads > 1
import dspy

OLLAMA = "http://localhost:11434"
STUDENT_MODEL = "ollama_chat/qwen3:1.7b"
REFLECTION_MODEL = "ollama_chat/gemma3:latest"
LABELS = {"positive", "negative", "neutral"}

# think=False is REQUIRED for Qwen3: with thinking on, ChatAdapter mis-parses and
# sentiment comes back as 'positive[[ ## completed ## ]]'.
student_lm = dspy.LM(STUDENT_MODEL, api_base=OLLAMA, api_key="",
                     temperature=0.0, max_tokens=4000, think=False, cache=False)
# Reflection LM must be the STRONGER model and needs a big max_tokens + temperature=1.0.
reflection_lm = dspy.LM(REFLECTION_MODEL, api_base=OLLAMA, api_key="",
                        temperature=1.0, max_tokens=8000)

dspy.configure(lm=student_lm)
student_lm("ping")  # warm up ollama + litellm before any parallel evaluation


class Sentiment(dspy.Signature):
    """Label the text."""  # deliberately vague: leaves GEPA room to improve

    sentence: str = dspy.InputField()
    sentiment: str = dspy.OutputField()


program = dspy.Predict(Sentiment)

RAW = [
    ("Well, that's just great. My flight got cancelled again.", "negative"),
    ("Shipment tracking number is 4471X.", "neutral"),
    ("Couldn't be happier with how this turned out!", "positive"),
    ("Sure, waiting 3 hours was exactly what I wanted.", "negative"),
    ("The battery lasts all day, which surprised me.", "positive"),
    ("Product dimensions: 12 x 8 x 3 inches.", "neutral"),
    ("It's fine. Does the job.", "neutral"),
    ("Blown away by the quality here.", "positive"),
    ("Yeah, no. Returning it immediately.", "negative"),
    ("Arrived Tuesday as scheduled.", "neutral"),
    ("This exceeded every expectation I had.", "positive"),
    ("Absolute garbage, save your money.", "negative"),
    ("Oh fantastic, another software update.", "negative"),
    ("I cannot recommend this highly enough.", "positive"),
    ("Does what it says on the tin.", "neutral"),
    ("Somehow it's even worse than the last one.", "negative"),
    ("Not bad at all, actually impressed.", "positive"),
    ("Warranty covers 12 months from purchase.", "neutral"),
    ("The service was, shall we say, memorable.", "negative"),
    ("Took forever but got here eventually.", "neutral"),
    ("I've had better, I've had worse.", "neutral"),
    ("Ships from our Ohio warehouse.", "neutral"),
    ("Ten out of ten would not repeat.", "negative"),
    ("Genuinely made my week better.", "positive"),
    ("The film was not without its merits, though few.", "negative"),
    ("I mean, it works I guess. Nothing special.", "neutral"),
    ("Fantastic service, will come again.", "positive"),
    ("Order #8812 shipped via ground.", "neutral"),
    ("Wow, just wow. And not in a good way.", "negative"),
    ("Better than I dared hope.", "positive"),
    ("Contains 200mg per serving.", "neutral"),
    ("If only everything worked this well.", "positive"),
    ("Let's just say I won't be back.", "negative"),
    ("Received the replacement unit today.", "neutral"),
    ("Honestly a steal at this price.", "positive"),
    ("Great, now it doesn't turn on either.", "negative"),
]
data = [dspy.Example(sentence=s, sentiment=l).with_inputs("sentence") for s, l in RAW]
random.Random(0).shuffle(data)
trainset, valset, testset = data[:12], data[12:24], data[24:]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA feedback metric. MUST return dspy.Prediction(score=float, feedback=str).

    GEPA calls this with pred_name/pred_trace when it wants predictor-level feedback.
    A bare float works but wastes GEPA: with no feedback it only sees
    "This trajectory got a score of {score}."
    """
    got = (getattr(pred, "sentiment", "") or "").strip().lower()
    want = gold.sentiment.strip().lower()
    if got == want:
        return dspy.Prediction(score=1.0, feedback=f"Correct: '{gold.sentence}' -> {want}.")
    if got not in LABELS:
        fb = (f"FORMAT ERROR on '{gold.sentence}': emitted '{got}', not a valid label. "
              f"Emit exactly one lowercase token from {sorted(LABELS)}. Gold was '{want}'.")
    else:
        fb = (f"WRONG on '{gold.sentence}': said '{got}', gold is '{want}'. "
              f"Sarcasm and understatement invert surface polarity; purely factual "
              f"statements are neutral even when the event described is bad.")
    return dspy.Prediction(score=0.0, feedback=fb)


# Chart numbers come from dspy.Evaluate on a HELD-OUT testset, never from
# detailed_results.val_aggregate_scores (its first entry is not comparable - see notes).
evaluate = dspy.Evaluate(devset=testset, metric=metric, num_threads=4, display_progress=False)
baseline = evaluate(program).score
print(f"BASELINE accuracy: {baseline:.1f}%")

optimizer = dspy.GEPA(
    metric=metric,
    max_metric_calls=80,           # exactly one of max_metric_calls / max_full_evals / auto
                                   # 80 ~= 1-2 min here; 200 took 17 min with a 4b reflection LM
    reflection_lm=reflection_lm,   # required unless you pass instruction_proposer
    reflection_minibatch_size=3,
    candidate_selection_strategy="pareto",
    num_threads=4,
    failure_score=0.0,
    perfect_score=1.0,
    track_stats=True,              # required for .detailed_results
    seed=0,
    log_dir=f"./gepa_log/{int(time.time())}",  # NEVER reuse a log_dir: GEPA resumes from it
)

t0 = time.time()
optimized = optimizer.compile(program, trainset=trainset, valset=valset)
elapsed = time.time() - t0

final = evaluate(optimized).score
print(f"OPTIMIZED accuracy: {final:.1f}%  (baseline {baseline:.1f}%)  in {elapsed:.0f}s")

print("\n=== OPTIMIZED INSTRUCTION (this is the artifact to show in the demo) ===")
for name, pred in optimized.named_predictors():
    print(f"--- {name} ---\n{pred.signature.instructions}")

r = optimized.detailed_results
print(f"\nmetric_calls={r.total_metric_calls} candidates={len(r.candidates)} "
      f"val_curve={r.val_aggregate_scores} best_idx={r.best_idx}")

optimized.save("./optimized_sentiment.json", save_program=False)  # state only (JSON, safe)
optimized.save("./optimized_sentiment_prog/", save_program=True)  # whole program (pickle)
# reload: dspy.Predict(Sentiment).load("./optimized_sentiment.json")
#     or: dspy.load("./optimized_sentiment_prog/", allow_pickle=True)
print("saved.")
