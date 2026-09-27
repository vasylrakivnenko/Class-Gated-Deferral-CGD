import os
import json
import urllib.request
import time

api_key = (os.environ.get("GCP_API_2") or os.environ.get("GEMINI_API_KEY") or "").strip()

# Fetch test samples from Hugging Face LegalBench
url = "https://huggingface.co/datasets/nguha/legalbench/raw/main/data/cuad_covenant_not_to_sue/test.tsv"
req = urllib.request.urlopen(url)
content = req.read().decode("utf-8")
lines = [l for l in content.split("\n") if l.strip()]

yes_items = []
no_items = []

for line in lines[1:]:
    parts = line.split("\t")
    if len(parts) >= 3:
        idx, text, label = parts[0], parts[1].strip(), parts[2].strip().lower()
        if label == "yes" and len(yes_items) < 5:
            yes_items.append({"id": f"cns-{len(yes_items)+1}", "text": text, "label": "yes"})
        elif label == "no" and len(no_items) < 5:
            no_items.append({"id": f"cns-{len(no_items)+6}", "text": text, "label": "no"})

all_samples = yes_items + no_items

# Save sample file
with open("cuad_covenant_samples.json", "w") as f:
    json.dump(all_samples, f, indent=2)

print(f"Saved {len(all_samples)} samples to cuad_covenant_samples.json")

# Benchmark with Gemini 3.1 Flash-Lite
model_id = "gemini-3.1-flash-lite"
model_name = "Gemini 3.1 Flash-Lite"
in_rate = 0.075
out_rate = 0.30

prompt_preamble = (
    "You are a specialized legal AI contract analyzer. Determine whether the contractual clause contains "
    "a \"Covenant Not To Sue\" / no-contest obligation (i.e. restricts a party from contesting the validity of the "
    "counterparty's intellectual property ownership or bringing claims against the counterparty).\n"
    "Respond with ONLY \"yes\" or \"no\"."
)

print("=" * 88, flush=True)
print(f"BENCHMARK: LegalBench CUAD Covenant Not To Sue | Model: {model_name} ({model_id})", flush=True)
print("=" * 88, flush=True)
print(f"{'Item ID':<8} | {'Gold':<5} | {'Pred':<5} | {'Match':<6} | {'Latency':<8} | {'Clause Snippet'}", flush=True)
print("-" * 88, flush=True)

correct = 0
total_time = 0
total_in = 0
total_out = 0
item_runs = []

for it in all_samples:
    prompt = f"{prompt_preamble}\n\nClause: \"{it['text']}\"\n\nAnswer:"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 10,
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }

    t0 = time.time()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8")
    )

    pred = "no"
    in_tok = 150
    out_tok = 1
    elapsed = 0

    try:
        res = urllib.request.urlopen(req, timeout=12)
        elapsed = time.time() - t0
        data = json.loads(res.read().decode("utf-8"))
        candidate = data.get("candidates", [{}])[0]
        text_part = candidate.get("content", {}).get("parts", [{}])[0].get("text", "").strip().lower()
        pred = "yes" if "yes" in text_part else "no"
        usage = data.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", 150)
        out_tok = usage.get("candidatesTokenCount", 1)
    except Exception as e:
        print(f"Error on {it['id']}: {e}")
        elapsed = time.time() - t0

    is_match = (pred == it["label"])
    if is_match:
        correct += 1

    total_time += elapsed
    total_in += in_tok
    total_out += out_tok

    item_runs.append({
        "id": it["id"],
        "gold": it["label"],
        "pred": pred,
        "correct": is_match,
        "latency": elapsed,
        "in_tok": in_tok,
        "out_tok": out_tok
    })

    snippet = it["text"][:55].replace("\n", " ") + "..."
    match_str = "YES" if is_match else "NO"
    print(f"{it['id']:<8} | {it['label']:<5} | {pred:<5} | {match_str:<6} | {elapsed:<8.2f} | {snippet}", flush=True)
    time.sleep(0.5)

n = len(all_samples)
acc = correct / n
avg_lat = total_time / n
avg_in = total_in / n
avg_out = total_out / n
cost_per_1k = 1000 * (avg_in * in_rate + avg_out * out_rate) / 1_000_000

print("-" * 88, flush=True)
print(f"SUMMARY FOR {model_name}:", flush=True)
print(f"  Accuracy:      {correct}/{n} ({acc*100:.1f}%)", flush=True)
print(f"  Avg Latency:   {avg_lat:.2f}s", flush=True)
print(f"  Tokens / Call: in={avg_in:.1f}, out={avg_out:.1f}", flush=True)
print(f"  Cost / 1k:     ${cost_per_1k:.6f}", flush=True)
print("=" * 88, flush=True)

results = {
    model_id: {
        "model_id": model_id,
        "model_name": model_name,
        "accuracy": acc,
        "correct": correct,
        "total": n,
        "avg_latency": avg_lat,
        "avg_in_tokens": avg_in,
        "avg_out_tokens": avg_out,
        "cost_per_1k": cost_per_1k,
        "items": item_runs
    }
}

with open("cuad_covenant_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved benchmark results to cuad_covenant_results.json")
