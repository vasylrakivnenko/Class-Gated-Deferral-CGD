"""Choice questions ("Does the tenant or the landlord pay for water?", "Is the
notice period 30 days or 60 days?"): the one alternative the text states, or
defer (router/STATUS.md, M3). Built on router/spans.py's clause matching.

1. The question's key terms, minus the words of its options, find the clauses
   about it (every key term in the clause).
2. Rule: in those clauses exactly one option is stated, and not negated
   ("The bonus is not guaranteed" doesn't state "guaranteed"); for parties, the
   one that is the subject of the clause's verb. No LLM call.
3. Otherwise the LLM picks among the options plus "neither / the text doesn't
   say", from those clauses (or from a short document), at MIN_P or more.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from router import frames, spans
from router.pretier0 import PARTY_SCAN_CHARS, find_parties

MIN_P = spans.MIN_P
NONE = "neither / the text doesn't say"
_NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                            "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
                      "hundred": 100})
_NEGATION = re.compile(r"\b(?:not|no|never|without|neither|nor)\b|n't\b", re.I)


@dataclass
class ChoiceResult:
    fired: bool
    answer: str | None = None
    confidence: float = 0.0
    how: str = ""  # "rule" | "llm" | ""
    reason: str = ""
    evidence: list = field(default_factory=list)
    options: list = field(default_factory=list)
    probabilities: dict = field(default_factory=dict)
    llm_calls: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _numbers(text: str) -> set:
    out = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    out |= {float(_NUMBER_WORDS[w]) for w in re.findall(r"[a-z]+", text.lower()) if w in _NUMBER_WORDS}
    return out


def _words(text: str) -> set:
    return {w for w in frames.tokens(text) if w not in frames.STOPWORDS and len(w) > 2 and not w.isdigit()}


def _states(option: str, unit: str) -> bool:
    """Whether the clause states the option: its numbers and its content words, not negated."""
    nums = _numbers(option)
    if nums and not nums <= _numbers(unit):
        return False
    words = _words(option) - {"per"}
    unit_words = set(frames.tokens(unit))
    if words and not all(any(w == u or (len(min(w, u, key=len)) >= 4 and (u.startswith(w) or w.startswith(u)))
                             for u in unit_words) for w in words):
        return False
    if not nums and not words:
        return False
    # not negated: no negation in the few words before the option's first word
    first = next(iter(sorted(words, key=len, reverse=True)), None) if words else None
    if first:
        m = re.search(rf"\b{re.escape(first[:4])}\w*", unit, re.I)
        if m and _NEGATION.search(unit[max(0, m.start() - 40):m.start()]):
            return False
    return True


def answer(frame, document: str, llm=None) -> ChoiceResult:
    options = [o.strip(" ,?.") for o in frame.options if o.strip(" ,?.")]
    keys = {re.sub(r"[^a-z0-9]", "", o.lower()) for o in options}
    if len(options) < 2 or len(keys) < len(options) or not all(_words(o) or _numbers(o) for o in options):
        return ChoiceResult(False, reason="couldn't tell the alternatives apart", options=options)
    question = frame.lookup or frame.question
    us = spans.units(document)
    option_words = set().union(*(_words(o) for o in options))
    terms = [t for t in spans.key_terms(question)
             if ":" in t or not any(t == w or t[:5] == w[:5] for w in option_words)]
    required = [t for t in terms if t not in spans.SOFT_TERMS]
    matching = [i for i, u in enumerate(us) if required and all(spans._covers(t, spans._unit_terms(u)) for t in required)]
    parties = find_parties(document[:PARTY_SCAN_CHARS])
    party_options = [o for o in options if any(re.fullmatch(rf"(?:the\s+)?{re.escape(p)}", o, re.I) for p in parties)]
    stated = {}
    if matching and len(party_options) == len(options):
        tw = {t for t in terms if ":" not in t} | {t.split(":")[1].lower() for t in terms if ":" in t}
        for i in matching:
            agent = spans._agent(us[i], parties, tw)
            for o in options:
                if agent and re.fullmatch(rf"(?:the\s+)?{re.escape(agent)}", o, re.I):
                    stated.setdefault(o, []).append(i)
    elif matching:
        for i in matching:
            for o in options:
                if _states(o, us[i]):
                    stated.setdefault(o, []).append(i)
    if len(stated) == 1 and len(matching) <= 3 and not spans.qualified(question):
        o, units_ = next(iter(stated.items()))
        return ChoiceResult(True, o, 1.0, "rule", reason=f'the clauses about it state "{o}", not the other option',
                            evidence=[{"text": us[i], "label": o, "p": 1.0} for i in units_[:2]], options=options)
    if llm is None or (not matching and len(document) > spans.LOOSE_MAX_CHARS):
        why = "no clause names everything the question asks about" if not matching else \
            "the clauses about it state more than one option" if len(stated) > 1 else \
            "the clauses about it state none of the options"
        return ChoiceResult(False, reason=why, options=options)
    state_units = matching or list(range(len(us)))
    state, size = [], 0
    for i in state_units:
        if size + len(us[i]) > spans.MAX_STATE_CHARS:
            break
        state.append(us[i])
        size += len(us[i])
    criteria = {o: "" for o in options}
    criteria[NONE] = "the text states neither alternative, or doesn't say"
    out = llm.choice("\n\n".join(state), question, criteria)
    pick, probs = out["choice"], out["probabilities"]
    p = float(probs.get(pick, 0.0))
    top = dict(sorted(probs.items(), key=lambda kv: -kv[1])[:4])
    if pick == NONE or pick not in options:
        return ChoiceResult(False, reason="the LLM found that the text states neither", options=options,
                            probabilities=top, llm_calls=1)
    if p < MIN_P:
        return ChoiceResult(False, reason=f"the LLM's pick is unsure ({p:.2f} < {MIN_P})", options=options,
                            probabilities=top, llm_calls=1)
    ev = next((i for i in state_units if _states(pick, us[i])), state_units[0] if state_units else None)
    return ChoiceResult(True, pick, round(p, 3), "llm", reason="the LLM picked it among the alternatives",
                        evidence=[{"text": us[ev], "label": pick, "p": round(p, 3)}] if ev is not None else [],
                        options=options, probabilities=top, llm_calls=1)
