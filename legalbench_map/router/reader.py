"""Tier 2 (router/STATUS.md): an LLM as a grounded reader, for fact questions
the span rules and Jev's pick defer (router/spans.py).

It reads the few clauses the matcher selects for a question and copies the
answer from them, or says the clauses don't state it. Code checks the copy:
an answer that isn't word for word in the clause it cites (or, failing that,
in one of the clauses shown) is dropped and the question defers. Then Jev
checks that the clause states that answer to the question (noul >= CHECK_MIN).
The model proposes; the text and Jev decide.

Contract
    in:  a question and at most MAX_CLAUSES numbered clauses (MAX_CHARS in all)
    out: JSON {"answer": "<words copied exactly from one clause>" | null,
               "clause": <that clause's number> | null}

`complete(prompt) -> str | (str, usage)` is any LLM call: FireworksLLM (gpt-oss-120b),
OpenRouterLLM (Gemma 4 26B-A4B, pinned to one provider), OwnGPULLM (whatever our own GPU
serves, through its gateway: router/gpu.py) or LocalLLM (llama.cpp's server, for evals).
Which one is live is switched on /admin (router/tier2.py).

Bake-off (2026-09-30, 320 questions, half real CUAD contracts; qtree/bakeoff.py,
qtree/verify.py): with today's system first and this reader on its deferrals,
gpt-oss-120b + Jev >= 0.7 answered 202/204 right (99.0%) and covered 91% of the
answerable questions (74% before); ~0.35 s per reader call.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from router import spans

MAX_CLAUSES = 8
MAX_CHARS = 6000
MAX_ANSWER_WORDS = 40
# Jev's check on the reader's answer. On the bake-off's first 160 questions >= 0.7 removed every false
# fire and kept 96/97 right; on the next 160 (unseen) it added 18 right answers and no wrong ones.
CHECK_MIN = 0.7
CHECK = ('Question about this text: "{q}"\nProposed answer, quoted from the text: "{a}"\n'
         'Does the text state that "{a}" is the answer to this question? Say false if the text is about '
         'something else, answers a different question, or doesn\'t settle it.')

PROMPT = """You answer questions about a document, using only the numbered clauses from it below.
Copy the answer word for word from ONE clause: the shortest exact phrase that answers the question (for example a
date, an amount, a duration, a party, a place, or a short phrase). Do not rephrase, calculate, or combine clauses.
If the clauses do not state the answer, the answer is null.

CLAUSES:
{clauses}

QUESTION: {question}

Reply with JSON only: {{"answer": "<exact words from the clause>" or null, "clause": <clause number> or null}}"""

SCHEMA = {"type": "object", "properties": {"answer": {"type": ["string", "null"]},
                                           "clause": {"type": ["integer", "null"]}},
          "required": ["answer", "clause"]}


class ReaderError(RuntimeError):
    pass


@dataclass
class ReaderResult:
    fired: bool
    answer: str | None = None
    clause: str | None = None  # the clause the answer was found in
    reason: str = ""
    raw: str = ""  # the model's reply
    ms: float = 0.0
    usage: dict = field(default_factory=dict)
    check: float | None = None  # Jev's probability that the clause states the answer, when checked
    self_p: float | None = None  # the model's own confidence on a yes/no `decide` call: uncalibrated, not a check

    def to_dict(self) -> dict:
        return asdict(self)


def select_clauses(question: str, document: str, max_clauses: int = MAX_CLAUSES, max_chars: int = MAX_CHARS) -> list:
    """The clauses the reader sees: those holding the most of the question's key
    terms (loose synonyms), best first up to the budget, then in document order."""
    us = spans.units(document)
    terms = spans.key_terms(question)
    if not terms:
        return []
    scored = []
    for i, u in enumerate(us):
        found = spans._unit_terms(u)
        s = sum(spans._covers(t, found, loose=True) for t in terms)
        if s:
            scored.append((-s, i))
    picked, size = [], 0
    for _, i in sorted(scored):
        if len(picked) >= max_clauses or size + len(us[i]) > max_chars:
            continue
        picked.append(i)
        size += len(us[i])
    return [us[i] for i in sorted(picked)]


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.lower().split()).strip(" .,;:\"'")


def grounded(answer: str, clauses: list, cited: int | None) -> str | None:
    """The clause the answer is copied from, or None: the cited clause first,
    then any clause shown."""
    a = _norm(answer)
    if not a or len(a.split()) > MAX_ANSWER_WORDS:
        return None
    order = ([cited - 1] if isinstance(cited, int) and 1 <= cited <= len(clauses) else []) + list(range(len(clauses)))
    for i in order:
        if a in _norm(clauses[i]):
            return clauses[i]
    return None


# "[* * *] days", "[ ] day of [ ], 2020", "___________, 2015": a redacted value is still a value.
_REDACTED = re.compile(r"\[[\s*xX]*\]|_{3,}")
_ALSO = {"MONEY": ("PERCENT", "CARDINAL"), "PERCENT": ("MONEY", "CARDINAL"), "CARDINAL": ("MONEY", "PERCENT"),
         "PARTY": ("JURISDICTION",)}  # "By how much can the fee rise?" -> "no more than 5%"; "Whose courts?"
# The sentence names the document itself ("This Agreement shall become effective on the Closing Date").
_THIS_DOC = re.compile(r"\b(?:this|the)\s+(?:[A-Za-z-]+\s+){0,4}?(?:agreement|contract|lease|licen[cs]e|amendment|"
                       r"addendum|term)\b|\bEffective\s+Date\b", re.I)
_FILLER = frozenset("a an the as of on at in from upon to its this that date day time".split())


def _own_date(answer: str, clause: str, question: str) -> bool:
    """For the document's own date: the sentence the answer is in names the document, no other
    event ("assign and transfer ..., effective upon creation") comes before the answer in it,
    and the answer says more than the question ("as of the Effective Date")."""
    words = set(re.findall(r"[a-z0-9]+", answer.lower())) - _FILLER
    if not words - set(re.findall(r"[a-z0-9]+", question.lower())):
        return False
    flat = " ".join(clause.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').split())
    at = flat.lower().find(_norm(answer))
    if at < 0:
        return False
    before = re.split(r"[.;]\s", flat[:at])[-1]
    after = re.split(r"[.;](?:\s|$)", flat[at:])[0]
    return bool(_THIS_DOC.search(before + after)) and not spans.other_event(before[-150:])


def fits(answer: str, clause: str, question: str, answer_type: str | None, document: str = "") -> bool:
    """The answer is the kind of thing asked for: a value of the type ("How long does the warranty
    last?" isn't answered by "for the duration of this Agreement"), and for the document's own
    date, the date of the document and not of some other event (see _own_date). Definitions are
    phrases, so any answer fits. On cuad_blind2 (seen once, 2026-09-30) Jev passed 10 answers
    that were wrong or unlabeled, 9 of them misfits like these."""
    if not answer_type or answer_type == "DEFINITION" or answer_type not in spans.ANSWERED_TYPES:
        return True  # a phrase, or a type with no value pattern (ENTITY, ACTION...): Jev's check alone decides
    if answer_type == "DATE" and spans.document_date_kind(question) not in (None, "expir"):
        return _own_date(answer, clause, question)
    text = _REDACTED.sub("30", answer)
    parties = spans.find_parties(document[:spans.PARTY_SCAN_CHARS]) if answer_type == "PARTY" else []
    return any(spans.candidates(t, [s], parties) for t in (answer_type, *_ALSO.get(answer_type, ()))
               for s in (text, text.title()))


def _parse(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict) and "answer" in d:
                return d
        except json.JSONDecodeError:
            continue
    return None


def read(question: str, clauses: list, complete, check=None, answer_type: str | None = None,
         document: str = "", check_min: float = CHECK_MIN) -> ReaderResult:
    """`check(state, instructions) -> p` is Jev's noul; None skips the check (the bake-off
    scores it separately, and /admin's Tier 1 switch turns it off). With `answer_type`, an
    answer that doesn't fit the question is dropped before the check (`fits`). `check_min`
    depends on the model (router/tier2.py)."""
    if not clauses:
        return ReaderResult(False, reason="no clause mentions what the question asks about")
    prompt = PROMPT.format(clauses="\n".join(f"[{i + 1}] {c}" for i, c in enumerate(clauses)), question=question)
    t = time.perf_counter()
    out = complete(prompt)
    ms = (time.perf_counter() - t) * 1000
    text, usage = (out if isinstance(out, tuple) else (out, {}))
    d = _parse(text)
    if d is None:
        return ReaderResult(False, reason="the reply wasn't the JSON asked for", raw=text[:500], ms=ms, usage=usage)
    if d.get("answer") in (None, "", "null"):
        return ReaderResult(False, reason="the reader found it isn't stated", raw=text[:500], ms=ms, usage=usage)
    clause = grounded(str(d["answer"]), clauses, d.get("clause"))
    if clause is None:
        return ReaderResult(False, reason="the answer isn't copied word for word from the clauses (dropped)",
                            answer=None, raw=text[:500], ms=ms, usage=usage)
    answer = str(d["answer"]).strip(" .,;:")
    if not fits(answer, clause, question, answer_type, document):
        return ReaderResult(False, reason=f'"{answer}" isn\'t the kind of answer asked for ({answer_type.lower()}'
                                          + (" of the document itself" if answer_type == "DATE" else "") + "); dropped",
                            clause=clause, raw=text[:500], ms=ms, usage=usage)
    if check is None:
        return ReaderResult(True, answer, clause, reason="copied from the clause it cites, unchecked (Tier 1 off)",
                            raw=text[:500], ms=ms, usage=usage)
    p = round(float(check(clause, CHECK.format(q=question, a=answer))), 3)
    if p < check_min:
        return ReaderResult(False, reason=f'Jev doubts the clause states "{answer}" as the answer ({p:.2f} < '
                                          f'{check_min}); dropped', clause=clause, raw=text[:500], ms=ms,
                            usage=usage, check=p)
    return ReaderResult(True, answer, clause, reason=f"copied word for word from a clause; Jev checked the clause "
                                                     f"states it ({p:.2f})", raw=text[:500], ms=ms, usage=usage,
                        check=p)


DECIDE = """You answer a yes/no question about a document, using only the numbered clauses from it below.
Decide only from what the clauses say, not from general knowledge or what is usual in such documents.
If the clauses do not settle the question, the answer is null.

CLAUSES:
{clauses}

QUESTION: {question}

Reply with JSON only: {{"answer": "yes" or "no" or null, "clause": <the clause number that settles it> or null, \
"p": <how sure you are, 0 to 1>}}"""


def decide(question: str, clauses: list, complete) -> ReaderResult:
    """Tier 2 on a yes/no question: only used when /admin turns Tier 1 (Jev) off, because
    Jev normally answers these (harness `_fallback`). Nothing checks the answer here --
    `read`'s verbatim-copy check cannot apply to yes/no -- so `self_p` is the model's own
    uncalibrated number and the answer is reported as unchecked. Not measured by the
    bake-off, which only scored fact reading."""
    if not clauses:
        return ReaderResult(False, reason="no clause mentions what the question asks about")
    prompt = DECIDE.format(clauses="\n".join(f"[{i + 1}] {c}" for i, c in enumerate(clauses)), question=question)
    t = time.perf_counter()
    out = complete(prompt)
    ms = (time.perf_counter() - t) * 1000
    text, usage = (out if isinstance(out, tuple) else (out, {}))
    d = _parse(text)
    if d is None:
        return ReaderResult(False, reason="the reply wasn't the JSON asked for", raw=text[:500], ms=ms, usage=usage)
    answer = str(d.get("answer") or "").strip().lower()
    if answer not in ("yes", "no"):
        return ReaderResult(False, reason="the reader found the clauses don't settle it", raw=text[:500], ms=ms,
                            usage=usage)
    n = d.get("clause")
    clause = clauses[n - 1] if isinstance(n, int) and 1 <= n <= len(clauses) else None
    try:
        self_p = min(1.0, max(0.0, float(d.get("p"))))
    except (TypeError, ValueError):
        self_p = None
    return ReaderResult(True, answer, clause, reason="the reader decided it from the clauses; nothing checked it "
                                                     "(Tier 1 off, so Jev didn't answer or verify)",
                        raw=text[:500], ms=ms, usage=usage, self_p=self_p)


def api_key(name: str) -> str:
    """From the environment, the repo's .env, or ~/.env."""
    if os.environ.get(name):
        return os.environ[name]
    for path in (Path(__file__).resolve().parents[2] / ".env", Path.home() / ".env"):
        try:
            for line in path.read_text().splitlines():
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            continue
    raise ReaderError(f"{name} not found in the environment, the repo's .env or ~/.env")


class FireworksLLM:
    """gpt-oss-120b on Fireworks' serverless API (the document's selected clauses leave the
    machine, as they do for Jev). `tier="priority"` pays 20% more for a shorter queue; on the
    bake-off both tiers gave the same replies, 0.35 vs 0.33 s p50."""
    MODEL = "accounts/fireworks/models/gpt-oss-120b"

    def __init__(self, tier: str | None = None, timeout: float = 20, max_tokens: int = 1200):
        import requests
        self.session, self.tier, self.timeout, self.max_tokens = requests.Session(), tier, timeout, max_tokens
        self.key = api_key("FIREWORKS_API_KEY")
        self.service_tier = "priority" if tier == "priority" else "standard"  # as requested; the reply doesn't say
        self.name = "gpt-oss-120b (Fireworks" + (", priority)" if tier == "priority" else ")")

    def __call__(self, prompt: str):
        import requests
        body = {"model": self.MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": self.max_tokens, "reasoning_effort": "low"}
        if self.tier:
            body["service_tier"] = self.tier
        last = ""
        for attempt in range(2):
            try:
                r = self.session.post("https://api.fireworks.ai/inference/v1/chat/completions", json=body,
                                      timeout=self.timeout, headers={"Authorization": f"Bearer {self.key}"})
            except requests.RequestException as e:
                last = str(e)
            else:
                if r.status_code == 200:
                    d = r.json()
                    # Fireworks' own time (queue + compute), apart from the network: how the tiers differ
                    timing = {k: float(r.headers[h]) for k, h in (("server_s", "Fireworks-Server-Processing-Time"),
                                                                  ("ttft_s", "Fireworks-Server-Time-To-First-Token"))
                              if r.headers.get(h)}
                    return d["choices"][0]["message"].get("content") or "", {**d.get("usage", {}), **timing,
                                                                             "service_tier": self.service_tier,
                                                                             "attempts": attempt + 1}
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    break
            time.sleep(1 + attempt)
        raise ReaderError(f"Fireworks failed: {last}")


class OpenRouterLLM:
    """A model on OpenRouter, pinned to one provider with no fallbacks (the bake-off measured
    each provider apart: Gemma 4 26B-A4B on NextBit, 0.52 s p50, 1.15 s p99)."""

    def __init__(self, model: str = "google/gemma-4-26b-a4b-it", provider: str | None = "NextBit",
                 timeout: float = 20, max_tokens: int = 300):
        import requests
        self.session, self.model, self.provider = requests.Session(), model, provider
        self.timeout, self.max_tokens = timeout, max_tokens
        self.key = api_key("OPENROUTER_API_KEY")

    def __call__(self, prompt: str):
        import requests
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": self.max_tokens}
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        last = ""
        for attempt in range(2):
            try:
                r = self.session.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=self.timeout,
                                      headers={"Authorization": f"Bearer {self.key}"})
            except requests.RequestException as e:
                last = str(e)
            else:
                if r.status_code == 200:
                    d = r.json()
                    if "choices" not in d:  # OpenRouter reports a provider's failure inside a 200
                        last = f"no choices: {str(d.get('error'))[:200]}"
                    else:
                        return d["choices"][0]["message"].get("content") or "", {**d.get("usage", {}),
                                                                                 "served_by": d.get("provider"),
                                                                                 "attempts": attempt + 1}
                else:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    if r.status_code not in (408, 429, 500, 502, 503, 504):
                        break
            time.sleep(1 + attempt)
        raise ReaderError(f"OpenRouter failed: {last}")


class OwnGPULLM:
    """Whatever our own GPU serves, through its gateway (router/gpu.py: the reader never
    knows which engine runs there). JSON-constrained, reasoning off, deterministic."""

    def __init__(self, gpu=None, max_tokens: int = 200):
        from router.gpu import GPUError, OwnGPU
        try:
            self.gpu = gpu or OwnGPU()
        except GPUError as e:
            raise ReaderError(f"own GPU: {e}") from e
        self.max_tokens = max_tokens

    def health(self) -> dict:
        from router.gpu import GPUError
        try:
            return self.gpu.health()
        except GPUError as e:
            raise ReaderError(f"own GPU: {e}") from e

    def __call__(self, prompt: str):
        from router.gpu import GPUError
        try:
            d = self.gpu.generate([{"role": "user", "content": prompt}], max_tokens=self.max_tokens, json_schema=SCHEMA)
        except GPUError as e:
            raise ReaderError(f"own GPU failed: {e}") from e
        timing = d.get("timing") or {}
        usage = {**(d.get("usage") or {}), "model": d.get("model"), "engine": d.get("engine"), "attempts": d["attempts"],
                 **({"server_s": timing["server_ms"] / 1000} if timing.get("server_ms") is not None else {}),
                 **{k: timing[k] for k in ("prefill_ms", "decode_ms") if timing.get(k) is not None}}
        return d.get("text") or "", usage


class LocalLLM:
    """llama.cpp's OpenAI-compatible server, deterministic, JSON-constrained."""

    def __init__(self, url: str = "http://127.0.0.1:8090", template_kwargs: dict | None = None, max_tokens: int = 256,
                 schema: bool = True):
        import requests
        self.session, self.url = requests.Session(), url.rstrip("/")
        self.template_kwargs, self.max_tokens, self.schema = template_kwargs or {}, max_tokens, schema

    def __call__(self, prompt: str):
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": self.max_tokens,
                "cache_prompt": False}
        if self.template_kwargs:
            body["chat_template_kwargs"] = self.template_kwargs
        if self.schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}}
        r = self.session.post(f"{self.url}/v1/chat/completions", json=body, timeout=900)
        r.raise_for_status()
        d = r.json()
        msg = d["choices"][0]["message"]
        return (msg.get("content") or ""), {**d.get("usage", {}), **d.get("timings", {})}
