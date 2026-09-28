"""
Run the FULL 416-row test split for cuad_change_of_control through Jev.

Reads results_jev_coc/requests.jsonl (all 416 test rows, 208 yes /
208 no, each with the correct original LegalBench question as a
Choice-primitive payload), calls the TypeSafe/Jev API for each row,
writes results_jev_coc/raw_responses.jsonl, then scores against
gold labels and prints + saves a summary.

Resumable: rerunning skips indices already present in raw_responses.jsonl.

Usage:
    /Users/vasyl/zadumai/.venv/bin/python run_jev_coc.py
"""
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE.parent.parent / ".env"
REQUESTS_PATH = HERE / "requests.jsonl"
RESPONSES_PATH = HERE / "raw_responses.jsonl"
SUMMARY_PATH = HERE / "summary.json"
FAILURES_PATH = HERE / "failures.jsonl"

API_URL = "https://api.typesafe.ai/v1/systemone"
MAX_RETRIES = 5
BASE_BACKOFF = 2.0


def load_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("JEV_API="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"JEV_API not found in {ENV_PATH}")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def call_jev(session: requests.Session, api_key: str, payload: dict) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(BASE_BACKOFF * (2 ** attempt))
            continue

        if resp.status_code == 200:
            return {"status": 200, "body": resp.json()}
        if resp.status_code in (429, 529) or resp.status_code >= 500:
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            time.sleep(BASE_BACKOFF * (2 ** attempt))
            continue
        return {"status": resp.status_code, "body": resp.text[:1000]}

    return {"status": "error", "body": f"exhausted retries: {last_err}"}


def main() -> None:
    api_key = load_api_key()
    requests_rows = load_jsonl(REQUESTS_PATH)
    if not requests_rows:
        print(f"no requests found at {REQUESTS_PATH}", file=sys.stderr)
        sys.exit(1)

    done = {r["index"] for r in load_jsonl(RESPONSES_PATH)}
    todo = [r for r in requests_rows if r["index"] not in done]
    print(f"{len(requests_rows)} total, {len(done)} already done, {len(todo)} to run")

    session = requests.Session()
    with open(RESPONSES_PATH, "a") as out_f, open(FAILURES_PATH, "a") as fail_f:
        for n, row in enumerate(todo, 1):
            result = call_jev(session, api_key, row["payload"])
            record = {"index": row["index"], "response": result}
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if result["status"] != 200:
                fail_f.write(json.dumps(record) + "\n")
                fail_f.flush()
                print(f"[{n}/{len(todo)}] index={row['index']} FAILED: {result}")
            elif n % 20 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] done")

    score()


def score() -> None:
    requests_rows = {r["index"]: r for r in load_jsonl(REQUESTS_PATH)}
    responses = load_jsonl(RESPONSES_PATH)

    tp = tn = fp = fn = 0
    n_scored = 0
    n_failed = 0
    confidences = []
    total_input_tok = 0
    total_output_tok = 0

    for resp_row in responses:
        idx = resp_row["index"]
        req_row = requests_rows.get(idx)
        if req_row is None:
            continue
        result = resp_row["response"]
        if result.get("status") != 200:
            n_failed += 1
            continue
        body = result["body"]
        try:
            answer = body["answers"]["label"]
            pred = answer["choice"].strip().lower()
            confidences.append(answer.get("confidence"))
        except (KeyError, TypeError):
            n_failed += 1
            continue

        gold = req_row["gold"]
        usage = body.get("usage", {})
        total_input_tok += usage.get("input_tokens", 0)
        total_output_tok += usage.get("output_tokens", 0)

        n_scored += 1
        if gold == "yes" and pred == "yes":
            tp += 1
        elif gold == "no" and pred == "no":
            tn += 1
        elif gold == "no" and pred == "yes":
            fp += 1
        elif gold == "yes" and pred == "no":
            fn += 1

    accuracy = (tp + tn) / n_scored if n_scored else float("nan")
    recall_yes = tp / (tp + fn) if (tp + fn) else float("nan")
    recall_no = tn / (tn + fp) if (tn + fp) else float("nan")
    balanced_accuracy = (recall_yes + recall_no) / 2

    summary = {
        "model": "jev (API)",
        "n_requested": len(requests_rows),
        "n_scored": n_scored,
        "n_failed": n_failed,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "recall_yes": recall_yes,
        "recall_no": recall_no,
        "mean_confidence": sum(c for c in confidences if c is not None) / len(confidences) if confidences else None,
        "total_input_tokens": total_input_tok,
        "total_output_tokens": total_output_tok,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print("\n=== Jev FULL 416-row summary (cuad_change_of_control) ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFor comparison: tfidf_logreg CV result on the same 416-row test split: "
          f"balanced_accuracy=0.921, 95% CI [0.906, 0.936] (published best LLM: 0.897).")


if __name__ == "__main__":
    main()
