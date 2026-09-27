import os
import json
import urllib.request
import time

api_key = (os.environ.get("GCP_API_2") or os.environ.get("gcp_api_2")).strip()

with open("cuad_audit_samples.json") as f:
    items = json.load(f)

# Select 4 representative test items: 2 positive (audit/inspection rights), 2 negative (general clauses)
test_items = [
    {"id": "AR-1", "label": "yes", "text": items[0]["text"]},
    {"id": "AR-2", "label": "yes", "text": items[1]["text"]},
    {"id": "AR-5", "label": "no",  "text": items[4]["text"]},
    {"id": "AR-6", "label": "no",  "text": items[5]["text"]},
]

models = [
    {"id": "gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash-Lite", "in_rate": 0.075, "out_rate": 0.30},
    {"id": "gemini-2.5-flash",      "name": "Gemini 2.5 Flash",      "in_rate": 0.075, "out_rate": 0.30},
]

overall_results = {}

for m in models:
    model_id = m["id"]
    model_name = m["name"]
    print("=" * 88, flush=True)
    print(f"BENCHMARK: LegalBench CUAD Audit Rights | Model: {model_name} ({model_id})", flush=True)
    print("=" * 88, flush=True)
    print(f"{'Item ID':<8} | {'Gold':<5} | {'Pred':<5} | {'Match':<6} | {'Latency':<8} | {'Clause Snippet'}", flush=True)
    print("-" * 88, flush=True)
    
    correct = 0
    total_time = 0
    total_in = 0
    total_out = 0
    item_runs = []

    for it in test_items:
        prompt = (
            "You are a legal contract analyzer. Determine whether the following contract clause contains "
            "an audit rights provision (i.e. gives a party the right to audit, review, or inspect books, records, accounts, or facilities of the counterparty).\n"
            "Respond with ONLY \"yes\" or \"no\".\n\n"
            f"Clause: \"{it['text']}\"\n\n"
            "Answer:"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 20,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }

        res = None
        elapsed = 0
        for attempt in range(5):
            try:
                t0 = time.time()
                req = urllib.request.Request(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    res = json.loads(resp.read().decode())
                elapsed = time.time() - t0
                total_time += elapsed
                break
            except Exception as e:
                time.sleep((attempt + 1) * 3)

        if not res:
            print(f"{it['id']:<8} | {it['label']:<5} | ERROR | FAIL   | 0.00s    | Call failed", flush=True)
            continue

        parts = res["candidates"][0]["content"].get("parts", [])
        raw = parts[0]["text"].strip().lower() if parts else "empty"
        pred = "yes" if "yes" in raw else ("no" if "no" in raw else raw)
        is_correct = (pred == it["label"])
        if is_correct:
            correct += 1

        usage = res.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", 0)
        out_tok = usage.get("candidatesTokenCount", 0)
        total_in += in_tok
        total_out += out_tok

        clean_text = it["text"].replace("\n", " ")
        snip = (clean_text[:40] + "...") if len(clean_text) > 40 else clean_text
        status = "PASS" if is_correct else "FAIL"
        print(f"{it['id']:<8} | {it['label']:^5} | {pred:^5} | {status:^6} | {elapsed:.2f}s    | {snip}", flush=True)
        item_runs.append({
            "id": it["id"],
            "gold": it["label"],
            "pred": pred,
            "correct": is_correct,
            "latency": elapsed,
            "in_tok": in_tok,
            "out_tok": out_tok
        })
        time.sleep(1.5)

    n = len(item_runs)
    acc = correct / n if n > 0 else 0
    avg_in = total_in / n if n > 0 else 0
    avg_out = total_out / n if n > 0 else 0
    avg_lat = total_time / n if n > 0 else 0
    cost_per_1k = 1000 * (avg_in * m["in_rate"] + avg_out * m["out_rate"]) / 1e6

    print("-" * 88, flush=True)
    print(f"RESULTS FOR {model_name}:", flush=True)
    print(f"  Accuracy: {correct}/{n} ({acc*100:.1f}%)", flush=True)
    print(f"  Average Latency: {avg_lat:.2f}s", flush=True)
    print(f"  Average Tokens: {avg_in:.1f} input / {avg_out:.1f} output", flush=True)
    print(f"  Measured Cost per 1,000 queries: ${cost_per_1k:.6f}", flush=True)
    print("=" * 88, flush=True)
    print("", flush=True)

    overall_results[model_id] = {
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
    time.sleep(3.0)

with open("cuad_audit_results.json", "w") as f:
    json.dump(overall_results, f, indent=2)

print("ALL RUNS COMPLETE. Saved to cuad_audit_results.json", flush=True)
