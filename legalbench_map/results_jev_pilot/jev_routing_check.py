"""
Same routing test as kev_routing_check.py, but against the real Jev API
instead of local Kev. Sends the JEV_API key from .env externally, so this
must be run by hand, not by the assistant.

Usage:
    /Users/vasyl/zadumai/.venv/bin/python jev_routing_check.py
"""
import glob
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE.parent.parent / ".env"
API_URL = "https://api.typesafe.ai/v1/systemone"


def load_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("JEV_API="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"JEV_API not found in {ENV_PATH}")


paths = sorted(glob.glob("/Users/vasyl/zadumai/legalbench_map/data/task_instructions/*.json"))
criteria = {}
for p in paths:
    d = json.load(open(p))
    q = d.get("base_prompt_preamble", "").strip()
    if not q:
        continue
    q = q.split("Question:")[-1].split("Answer:")[0].strip() if "Question:" in q else q
    criteria[d["task"]] = q

criteria["none_of_these"] = "The request does not match any of the other listed questions."

INSTRUCTIONS = (
    "A user submitted a free-text request. Which ONE of the following predefined "
    "legal classification questions does the user's request match? If none of them "
    "match, choose none_of_these."
)

test_cases = {
    "cuad_cap_on_liability (paraphrase)": "Is there a maximum dollar amount either side can be forced to pay if they breach this deal?",
    "cuad_audit_rights (paraphrase)": "Can one company inspect the other's financial records to check they're following the agreement?",
    "learned_hands_housing (paraphrase)": "My landlord is trying to kick me out and won't fix anything, what do I do?",
    "opp115_data_security (paraphrase)": "How does this app keep my personal info safe from hackers?",
    "supply_chain_disclosure_disclosed_accountability (faithful)": "Does this company's statement say whether, and to what extent, it has internal accountability procedures for staff who violate anti-trafficking standards?",
    "supply_chain_disclosure_best_practice_accountability (faithful)": "Does this company's statement describe an actual internal compliance mechanism for human trafficking standards, not just a requirement that suppliers comply with the law?",
    "out-of-scope (should be none_of_these)": "What's the weather going to be like tomorrow in Austin?",
}

expected = {
    "cuad_cap_on_liability (paraphrase)": "cuad_cap_on_liability",
    "cuad_audit_rights (paraphrase)": "cuad_audit_rights",
    "learned_hands_housing (paraphrase)": "learned_hands_housing",
    "opp115_data_security (paraphrase)": "opp115_data_security",
    "supply_chain_disclosure_disclosed_accountability (faithful)": "supply_chain_disclosure_disclosed_accountability",
    "supply_chain_disclosure_best_practice_accountability (faithful)": "supply_chain_disclosure_best_practice_accountability",
    "out-of-scope (should be none_of_these)": "none_of_these",
}


def main() -> None:
    api_key = load_api_key()
    n_correct = 0
    for label, phrase in test_cases.items():
        payload = {
            "state": phrase,
            "model": "jev-latest",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": criteria,
                }
            },
        }
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"[ERR ] {label}: HTTP {resp.status_code} {resp.text[:300]}")
            continue
        body = resp.json()
        answer = body["answers"]["route"]
        choice = answer["choice"]
        conf = answer.get("confidence")
        correct = choice == expected[label]
        n_correct += correct
        mark = "OK " if correct else "MISS"
        print(f"[{mark}] {label}")
        print(f"       -> {choice}  (confidence={conf})")
        probs = answer.get("probabilities", {})
        top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        print(f"       top3: {top3}")
        print()

    print(f"{n_correct}/{len(test_cases)} correct")


if __name__ == "__main__":
    main()
