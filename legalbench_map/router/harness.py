"""
Answer a question about a user's document with one of our free classifiers
when the question is one we cover, and with an LLM (Jev, or Kev locally)
otherwise.

    1. Route: Jev reads the QUESTION only and picks a menu task or
       none_of_these (router/menu.py).
    2. Gate: trust the pick only if p(none_of_these) < P_NONE_MAX and the
       pick leads the runner-up by at least MARGIN_MIN.
    3. Policy: tasks where our free model is weak (router/bank.py) are
       answered by the LLM with the task's own question.
    4. Text check: the LLM confirms the document is the kind of text the
       task's classifier was trained on ("Is this text a commercial
       contract...?").
    5. Classify: clause/sentence-level families split long documents into
       units, classify each one, and answer "yes" if any unit says yes.
    Any failed step falls back to the LLM, asked the user's own question as
    a yes/no (noul) question about the document.

Only the question goes to the router. With `llm` = Kev, the document never
leaves the machine.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from router.bank import Bank
from router.menu import CRITERIA, DIVERSITY_FAMILY, INSTRUCTIONS, SPLIT_UNIT, TEXT_TYPES, family
from router.systemone import SystemOne

# Calibrated on Jev's answers to the results_jev_pilot routing sets: correct
# routes had p(none_of_these) <= 0.04 (65 of 65), wrong routes 0.17-0.47
# (5 of 5). No wrong route has been seen near a tie between two tasks; the
# lowest correct margin was 0.48, so MARGIN_MIN is only a backstop.
# Phrasings unlike that set land in between ("Is this person's problem about
# housing?" scored 0.13-0.15), so the gate errs toward the LLM.
P_NONE_MAX = 0.10
MARGIN_MIN = 0.30
# Noul probability the document is the task's text type. On 42 LegalBench
# texts Jev scored its own family >= 0.83 (bar one terse sentence) and other
# families <= 0.08; Kev bunches near 0.5 (0.57-0.92 vs 0.06-0.54), so with Kev
# as the reader more documents fall back.
TEXT_TYPE_MIN = 0.5
SNIPPET_CHARS = 2000  # document prefix the text check reads
MAX_UNIT_CHARS = 1500  # longer paragraphs are cut into sentence groups
MIN_UNIT_CHARS = 40  # shorter pieces (headings, numbering) join the next unit
N_EVIDENCE = 3

# A sentence ends at . ! ? or ; followed by a capital, except after
# abbreviations common in legal text ("Smith v. Jones", "Acme Inc. The").
_ABBREVIATIONS = ("v", "vs", "Inc", "Co", "Corp", "Ltd", "No", "Sec", "Art", "Mr", "Ms", "Dr", "St", "e.g", "i.e", "U.S")
_SENTENCE_END = re.compile(
    "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVIATIONS) + r"(?<=[.!?;])\s+(?=[A-Z(\"'])"
)


@dataclass
class Route:
    choice: str
    p_none: float
    margin: float
    accepted: bool


@dataclass
class Answer:
    question: str
    answer: str  # "yes" / "no", or a category label
    confidence: float  # probability of `answer` from whoever answered
    path: str  # "classifier" | "llm_task" | "llm_fallback"
    reason: str
    asked: str = ""  # the question actually answered: the task's criteria, or the user's own question
    task: str | None = None
    answered_by: str = ""
    route: Route | None = None
    n_units: int = 1
    evidence: list = field(default_factory=list)  # [{"text", "label", "p"}], most confident first
    probabilities: dict = field(default_factory=dict)  # every option's probability, `answer` among them
    llm_calls: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def route_scores(probabilities: dict) -> tuple[float, float]:
    """(p_none, margin) from the router's probabilities, counting the six
    diversity options as one: they share a criteria text, so Jev splits its
    probability among them."""
    p = dict(probabilities)
    p["diversity"] = sum(p.pop(k, 0.0) for k in DIVERSITY_FAMILY)
    top = sorted(p.values(), reverse=True) + [0.0]
    return p.get("none_of_these", 0.0), top[0] - top[1]


def split_units(text: str, unit: str) -> list:
    """Split a document into roughly clause-sized ("paragraph") or
    sentence-sized ("sentence") units."""
    text = text.strip()
    if unit == "sentence":
        pieces = _SENTENCE_END.split(text)
    else:
        sep = r"\n\s*\n" if re.search(r"\n\s*\n", text) else r"\n"
        pieces = []
        for para in re.split(sep, text):
            para = " ".join(para.split())
            if len(para) <= MAX_UNIT_CHARS:
                pieces.append(para)
                continue
            chunk = ""
            for sent in _SENTENCE_END.split(para):
                if chunk and len(chunk) + 1 + len(sent) > MAX_UNIT_CHARS:
                    pieces.append(chunk)
                    chunk = ""
                chunk = f"{chunk} {sent}".strip()
            pieces.append(chunk)
    units, carry = [], ""
    for piece in (" ".join(p.split()) for p in pieces):
        if not piece:
            continue
        piece = f"{carry} {piece}".strip()
        if len(piece) < MIN_UNIT_CHARS:
            carry = piece
            continue
        units.append(piece)
        carry = ""
    if carry:
        if units:
            units[-1] = f"{units[-1]} {carry}"
        else:
            units.append(carry)
    return units or [text]


def _yes_no(p_yes: float) -> tuple[str, float, dict]:
    probs = {"yes": round(p_yes, 3), "no": round(1.0 - p_yes, 3)}
    return ("yes", p_yes, probs) if p_yes >= 0.5 else ("no", 1.0 - p_yes, probs)


def aggregate(units: list, proba, classes: list) -> tuple[str, float, list, dict]:
    """Binary: "yes" if any unit's p(yes) >= 0.5, scored by the most
    yes-like unit. Multi-class: the most confident label other than "other"
    (unfair_tos's catch-all), if any unit has one. Returns (answer,
    confidence, evidence, probabilities)."""
    if classes == ["no", "yes"]:
        p_yes = proba[:, 1]
        order = p_yes.argsort()[::-1]
        evidence = [{"text": units[i], "label": "yes", "p": round(float(p_yes[i]), 3)} for i in order[:N_EVIDENCE]]
        answer, conf, probs = _yes_no(float(p_yes[order[0]]))
        return answer, conf, evidence, probs
    labels = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    hits = [i for i in range(len(units)) if classes[labels[i]] != "other"] or list(range(len(units)))
    hits.sort(key=lambda i: -conf[i])
    evidence = [{"text": units[i], "label": classes[labels[i]], "p": round(float(conf[i]), 3)} for i in hits[:N_EVIDENCE]]
    best = hits[0]
    probs = {c: round(float(proba[best, j]), 3) for j, c in enumerate(classes)}
    return classes[labels[best]], float(conf[best]), evidence, probs


class Harness:
    def __init__(self, router: SystemOne, llm: SystemOne, bank: Bank):
        self.router = router
        self.llm = llm
        self.bank = bank

    def route(self, question: str) -> Route:
        out = self.router.choice(question, INSTRUCTIONS, CRITERIA)
        p_none, margin = route_scores(out["probabilities"])
        accepted = out["choice"] != "none_of_these" and p_none < P_NONE_MAX and margin >= MARGIN_MIN
        return Route(out["choice"], round(p_none, 3), round(margin, 3), accepted)

    def answer(self, question: str, document: str) -> Answer:
        calls_before = self.router.calls + (self.llm.calls if self.llm is not self.router else 0)
        result = self._answer(question, document)
        result.llm_calls = self.router.calls + (self.llm.calls if self.llm is not self.router else 0) - calls_before
        return result

    def _answer(self, question: str, document: str) -> Answer:
        route = self.route(question)
        if route.choice == "none_of_these":
            return self._fallback(question, document, route, "the question matches none of our tasks")
        if not route.accepted:
            return self._fallback(question, document, route,
                                  f"router unsure it matches {route.choice} (p(none)={route.p_none}, margin={route.margin})")
        task = route.choice
        info = self.bank.manifest.get(task)
        if info is None:
            return self._fallback(question, document, route, f"matched {task}, which is not in the bank")
        fam = family(task)
        if info["policy"] == "llm":
            return self._llm_task(question, task, info, document, route)
        p_type = self.llm.noul(document[:SNIPPET_CHARS], f"Is this text {TEXT_TYPES[fam]}?")
        if p_type < TEXT_TYPE_MIN:
            return self._fallback(question, document, route,
                                  f"matched {task}, but the document doesn't look like {TEXT_TYPES[fam]} (p={p_type:.2f})")
        units = split_units(document, SPLIT_UNIT[fam]) if fam in SPLIT_UNIT else [document]
        model = self.bank.model(task)
        answer, conf, evidence, probs = aggregate(units, model.predict_proba(units), model.classes)
        return Answer(
            question=question, answer=answer, confidence=round(conf, 3), path="classifier",
            reason=f"matched {task}; document looks like {TEXT_TYPES[fam]} (p={p_type:.2f})",
            asked=CRITERIA[task], task=task,
            answered_by=f"free classifier ({info['candidate']}, CV {info['cv_metric']} {info['cv_mean']:.3f})",
            route=route, n_units=len(units), evidence=evidence, probabilities=probs,
        )

    def _llm_task(self, question: str, task: str, info: dict, document: str, route: Route) -> Answer:
        reason = (f"matched {task}, where our free model is weak "
                  f"(CV {info['cv_mean']:.2f} vs {info['published_best']:.2f} for {info['published_best_model']})")
        if info["classes"] == ["no", "yes"]:
            answer, conf, probs = _yes_no(self.llm.noul(document, CRITERIA[task]))
        else:
            out = self.llm.choice(document, CRITERIA[task], {c: c for c in info["classes"]})
            answer, probs = out["choice"], out["probabilities"]
            conf = probs.get(answer, 0.0)
        return Answer(question=question, answer=answer, confidence=round(conf, 3), path="llm_task",
                      reason=reason, asked=CRITERIA[task], task=task,
                      answered_by=self.llm.model_version or self.llm.model, route=route, probabilities=probs)

    def _fallback(self, question: str, document: str, route: Route, reason: str) -> Answer:
        answer, conf, probs = _yes_no(self.llm.noul(document, question))
        return Answer(question=question, answer=answer, confidence=round(conf, 3), path="llm_fallback",
                      reason=reason, asked=question, answered_by=self.llm.model_version or self.llm.model,
                      route=route, probabilities=probs)
