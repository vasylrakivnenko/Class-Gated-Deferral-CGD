"""
Minimal client for the systemone judgment API, served by both Jev (hosted,
api.typesafe.ai, key JEV_API in the repo's .env) and Kev (local server,
`python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008`, no key).

Two question types are used: `choice` (probabilities over named options)
and `noul` (probability that a yes/no question is true). Every call is
counted, with its token usage, so the harness can report what an answer
cost.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

JEV_URL = "https://api.typesafe.ai/v1/systemone"
KEV_URL = "http://127.0.0.1:8008/v1/systemone"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

MAX_RETRIES = 5
BASE_BACKOFF = 2.0


class SystemOneError(RuntimeError):
    pass


def _jev_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("JEV_API="):
            return line.split("=", 1)[1].strip()
    raise SystemOneError(f"JEV_API not found in {ENV_PATH}")


class SystemOne:
    def __init__(self, url: str, model: str, api_key: str | None = None, timeout: float = 60):
        self.url = url
        self.model = model
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._session = requests.Session()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_version = None  # e.g. "jev-1.13.0", as reported by the server

    @classmethod
    def jev(cls) -> "SystemOne":
        return cls(JEV_URL, "jev-latest", api_key=_jev_api_key())

    @classmethod
    def kev(cls, url: str = KEV_URL) -> "SystemOne":
        return cls(url, "kev-latest")

    def ask(self, state: str, questions: dict) -> dict:
        """POST one request; return its `answers` dict. Retries on 429/529/5xx
        and connection errors; any other non-200 status raises."""
        payload = {"state": state, "model": self.model, "questions": questions}
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.post(self.url, headers=self._headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = str(e)
            else:
                if resp.status_code == 200:
                    body = resp.json()
                    self.calls += 1
                    usage = body.get("usage") or {}
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
                    self.model_version = body.get("model", self.model_version)
                    return body["answers"]
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if not (resp.status_code in (429, 529) or resp.status_code >= 500):
                    break
            time.sleep(BASE_BACKOFF * (2 ** attempt))
        raise SystemOneError(f"{self.url} failed: {last_err}")

    def choice(self, state: str, instructions: str, criteria: dict) -> dict:
        """Returns {"choice": option, "probabilities": {option: p, ...}}."""
        answer = self.ask(state, {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}})["q"]
        return {"choice": answer["choice"], "probabilities": dict(answer.get("probabilities", {}))}

    def noul(self, state: str, instructions: str) -> float:
        """Probability that the yes/no question `instructions` is true of `state`."""
        answer = self.ask(state, {"q": {"type": "noul", "instructions": instructions, "criteria": {"true": "", "false": ""}}})["q"]
        return float(answer["noul"])
