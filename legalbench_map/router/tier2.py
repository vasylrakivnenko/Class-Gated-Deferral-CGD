"""The Tier 2 readers the admin page switches between (router/STATUS.md, TIER 2 LIVE).

Each option is one way to serve router/reader.py's contract: a model, where it runs, and how
sure Jev's check must be. The check differs by model, from the bake-off (2026-09-30, 320
questions): gpt-oss-120b >= 0.7, Gemma 4 26B-A4B >= 0.8. Our own GPU runs the same Gemma
(Google's Q4_0 QAT file), which gave NextBit's answer on 151 of the bake-off's first 160
questions (most of the other 9 differ by a "the"), so it keeps Gemma's 0.8.

The choice is saved in usage.db (`settings`), so it survives restarts; until an admin picks
one, ask_ui.py's --tier2 flag decides. A switch runs one test read first and keeps the old
reader if that fails.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass
from typing import Callable

from router import reader
from router.usage import LEGACY_TIER2 as LEGACY  # the old --tier2 values, as logged
SETTING = "tier2"
# The test read before a switch: one clause that states the answer.
PROBE_QUESTION = "How much is the pet deposit?"
PROBE_CLAUSES = ["9. PETS. If Tenant keeps a pet, Tenant shall pay a pet deposit of $500 per animal."]


@dataclass(frozen=True)
class Option:
    id: str
    name: str  # as the answer's "answered by" shows it
    model: str
    where: str  # where the selected clauses go
    check_min: float
    price: str
    measured: str
    make: Callable[[], Callable]
    needs: str  # the key it reads


OPTIONS = {o.id: o for o in (
    Option("fireworks-standard", "gpt-oss-120b (Fireworks)", "gpt-oss-120b", "Fireworks serverless, standard queue",
           0.7, "$0.15 / $0.60 per 1M tokens in / out · ≈ $1.18 per 10k questions",
           "bake-off: 99.0% precise, 91% coverage; 0.35 s p50, 0.82 s p99",
           lambda: reader.FireworksLLM(tier=None), "FIREWORKS_API_KEY"),
    Option("fireworks-priority", "gpt-oss-120b (Fireworks, priority)", "gpt-oss-120b",
           "Fireworks serverless, priority queue", 0.7,
           "$0.18 / $0.72 per 1M tokens in / out · ≈ $1.42 per 10k questions",
           "bake-off: same replies as standard; 0.33 s p50, 0.74 s p99",
           lambda: reader.FireworksLLM(tier="priority"), "FIREWORKS_API_KEY"),
    Option("gpu-gemma", "Gemma 4 26B-A4B (our GPU)", "Gemma 4 26B-A4B, Q4_0 QAT", "our own GPU (zadum-gpu/1 gateway)",
           0.8, "the GPU's hourly rate, used or not; no per-token cost",
           "same answers as NextBit on 151/160; from this server 0.29 s p50, 0.63 s p99 (286 real reads)",
           lambda: reader.OwnGPULLM(), "ZADUM_GPU_TOKEN"),
    Option("openrouter-gemma", "Gemma 4 26B-A4B (OpenRouter, NextBit)", "Gemma 4 26B-A4B",
           "OpenRouter, pinned to NextBit (no fallbacks)", 0.8,
           "$0.0765 / $0.255 per 1M tokens in / out (+5.5% fee) · ≈ $0.46 per 10k questions",
           "bake-off: 98.5% precise, 90% coverage; 0.52 s p50, 1.15 s p99",
           lambda: reader.OpenRouterLLM(), "OPENROUTER_API_KEY"),
)}


def probe(llm) -> dict:
    """One read of a clause that states the answer: it must answer, copied word for word."""
    r = reader.read(PROBE_QUESTION, PROBE_CLAUSES, llm)
    if not r.fired or "500" not in (r.answer or ""):
        raise reader.ReaderError(f"the test read didn't answer ({r.reason}; reply {r.raw[:120]!r})")
    return {"ms": round(r.ms), "answer": r.answer}


class Connected:
    """An option's live client, as the harness calls it."""

    def __init__(self, option: Option, llm):
        self.option, self.llm = option, llm
        self.name, self.check_min = option.name, option.check_min

    def __call__(self, prompt: str):
        out = self.llm(prompt)
        text, usage = out if isinstance(out, tuple) else (out, {})
        return text, {**usage, "option": self.option.id}


class Tier2:
    """The live Tier 2 reader (`reader`, None when off), switchable at runtime. `store` is
    usage.Usage (or anything with get_setting/set_setting), where the choice is saved."""

    def __init__(self, default: str | None = None, store=None, options: dict | None = None):
        self.options, self.store = options or OPTIONS, store
        self._lock = threading.Lock()
        saved = store.get_setting(SETTING) if store else None
        self.chosen = saved or {"option": LEGACY.get(default, default) if default else "off", "by": "--tier2", "at": None}
        self.reader, self.error = None, None
        if self.chosen["option"] != "off":
            try:  # no test read at startup: an outage defers questions, it mustn't stop the service
                self.reader = self._connect(self.chosen["option"])
            except Exception as e:  # a missing key: Tier 2 stays off, and the admin page says why
                self.error = f"{self.chosen['option']} didn't start: {e}"

    def _connect(self, option_id: str) -> Connected:
        if option_id not in self.options:
            raise reader.ReaderError(f"no Tier 2 option {option_id!r}")
        opt = self.options[option_id]
        return Connected(opt, opt.make())

    def select(self, option_id: str, by: str) -> dict:
        """Switch after a test read; raises ReaderError (the old reader stays) if it fails."""
        option_id = LEGACY.get(option_id, option_id)
        with self._lock:
            test = None
            if option_id == "off":
                new = None
            else:
                try:
                    new = self._connect(option_id)
                    test = probe(new)
                except reader.ReaderError:
                    raise
                except Exception as e:
                    raise reader.ReaderError(f"{type(e).__name__}: {e}") from e
            self.chosen = {"option": option_id, "by": by, "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                           "test": test}
            self.reader, self.error = new, None
            if self.store:
                self.store.set_setting(SETTING, self.chosen, by)
            return self.chosen

    @property
    def active(self) -> str:
        """The live option's id; "off" when no reader is up."""
        return self.reader.option.id if self.reader else "off"

    def status(self) -> dict:
        """For the admin page: what's live, and whether each option could run now."""
        options = []
        for opt in self.options.values():
            ready, note = True, ""
            try:
                reader.api_key(opt.needs)
            except reader.ReaderError:
                ready, note = False, f"{opt.needs} isn't set"
            if ready and opt.id == "gpu-gemma":
                live = self.reader.llm if self.reader and self.reader.option.id == opt.id else None
                try:
                    t = time.perf_counter()
                    h = (live or opt.make()).health()
                    note = f"up: {h.get('model')} on {h.get('engine')} ({(time.perf_counter() - t) * 1000:.0f} ms)"
                except Exception as e:
                    ready, note = False, f"down: {e}"
            options.append({"id": opt.id, "name": opt.name, "model": opt.model, "where": opt.where,
                            "check_min": opt.check_min, "price": opt.price, "measured": opt.measured,
                            "ready": ready, "note": note})
        return {"active": self.active, "chosen": self.chosen, "error": self.error, "options": options}
