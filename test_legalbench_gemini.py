import os
import subprocess
import json
import urllib.request
import time

api_key = (os.environ.get('GCP_API_2') or os.environ.get('gcp_api_2')).strip()
model = 'gemini-2.5-flash'

raw_blob = subprocess.check_output(
    ['git', 'show', '44edf85:legalbench_map/cache/hf/hub/datasets--nguha--legalbench/blobs/f560f7cc2d7d6c53975f2077b4fa81670488f7d5'],
    text=True, errors='replace'
)

lines = raw_blob.strip().split('\n')
yes_items = []
no_items = []

for line in lines[1:]:
    parts = line.split('\t')
    if len(parts) >= 3:
        idx, text, label = parts[0], parts[1], parts[2].strip().lower()
        if label == 'yes' and len(yes_items) < 5:
            yes_items.append({'id': idx, 'text': text, 'gold': 'yes'})
        elif label == 'no' and len(no_items) < 5:
            no_items.append({'id': idx, 'text': text, 'gold': 'no'})

items = yes_items + no_items
print(f"Running tiny benchmark on LegalBench (CUAD Anti-Assignment): {len(items)} items...")
print(f"Model: {model} using GCP_API_2")

correct = 0
total_in_tokens = 0
total_out_tokens = 0
total_time = 0

print("-" * 86)
print(f"{'ID':<6} | {'Gold':<6} | {'Predicted':<10} | {'Match':<6} | {'Latency':<8} | {'Clause Sample'}")
print("-" * 86)

for it in items:
    prompt = (
        'You are a legal contract analyzer. Determine whether the following contract clause contains '
        'an anti-assignment provision (i.e. restricts or conditions assignment or transfer of the agreement or rights).\n'
        'Respond with ONLY "yes" or "no".\n\n'
        f'Clause: "{it["text"]}"\n\n'
        'Answer:'
    )

    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {
            'temperature': 0.0,
            'maxOutputTokens': 10
        }
    }

    t0 = time.time()
    req = urllib.request.Request(
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
    elapsed = time.time() - t0
    total_time += elapsed

    raw_ans = res['candidates'][0]['content']['parts'][0]['text'].strip().lower()
    pred = 'yes' if 'yes' in raw_ans else ('no' if 'no' in raw_ans else raw_ans)
    is_correct = (pred == it['gold'])
    if is_correct:
        correct += 1

    usage = res.get('usageMetadata', {})
    in_tok = usage.get('promptTokenCount', 0)
    out_tok = usage.get('candidatesTokenCount', 0)
    total_in_tokens += in_tok
    total_out_tokens += out_tok

    clean_text = it['text'].replace('\n', ' ')
    snippet = (clean_text[:40] + '...') if len(clean_text) > 40 else clean_text
    status = 'PASS' if is_correct else 'FAIL'
    print(f"{it['id']:<6} | {it['gold']:<6} | {pred:<10} | {status:<6} | {elapsed:.2f}s   | {snippet}")

acc = correct / len(items)
avg_in = total_in_tokens / len(items)
avg_out = total_out_tokens / len(items)
cost_per_1k = 1000 * (avg_in * 0.075 + avg_out * 0.30) / 1e6

print("=" * 86)
print(f"Accuracy: {correct}/{len(items)} ({acc * 100:.1f}%)")
print(f"Average Latency: {total_time / len(items):.2f}s per call")
print(f"Average Tokens: {avg_in:.1f} input / {avg_out:.1f} output")
print(f"Measured Cost per 1,000 queries: ${cost_per_1k:.6f}")
print("=" * 86)
