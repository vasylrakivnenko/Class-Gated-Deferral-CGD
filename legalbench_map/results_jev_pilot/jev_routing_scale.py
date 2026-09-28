"""
34-question routing scale test against the real Jev API. Sends JEV_API
from .env externally, so this must be run by hand, not by the assistant.

Usage:
    /Users/vasyl/zadumai/.venv/bin/python jev_routing_scale.py
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routing_menu import CRITERIA, TEST_CASES, INSTRUCTIONS, is_correct

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE.parent.parent / ".env"
API_URL = "https://api.typesafe.ai/v1/systemone"


def load_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("JEV_API="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"JEV_API not found in {ENV_PATH}")


def main() -> None:
    api_key = load_api_key()
    n_correct = 0
    rows = []
    for expected, phrase in TEST_CASES:
        payload = {
            "state": phrase,
            "model": "jev-latest",
            "questions": {
                "route": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}
            },
        }
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"[ERR ] HTTP {resp.status_code} {resp.text[:300]}")
            continue
        body = resp.json()
        answer = body["answers"]["route"]
        choice = answer["choice"]
        probs = answer.get("probabilities", {})
        top2 = sorted(probs.items(), key=lambda kv: -kv[1])[:2]
        margin = (top2[0][1] - top2[1][1]) if len(top2) == 2 else None
        correct = is_correct(expected, choice)
        n_correct += correct
        rows.append((correct, expected, phrase, choice, margin))
        mark = "OK " if correct else "MISS"
        exp_str = "/".join(sorted(expected)) if isinstance(expected, set) else expected
        print(f"[{mark}] expected={exp_str}  got={choice}  margin={margin}")

    print(f"\n{n_correct}/{len(TEST_CASES)} correct")

    misses = [r for r in rows if not r[0]]
    print(f"\n=== {len(misses)} misses ===")
    for correct, expected, phrase, choice, margin in misses:
        exp_str = "/".join(sorted(expected)) if isinstance(expected, set) else expected
        print(f"  expected={exp_str}  got={choice}  margin={margin}")
        print(f"    phrase: {phrase}")


if __name__ == "__main__":
    main()
