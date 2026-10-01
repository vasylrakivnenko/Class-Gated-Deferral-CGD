"""
Answer a question about a user's document with one of our free classifiers
when the question is one we cover, and with an LLM (Jev, or Kev locally)
otherwise.

    0a. Pre-Tier 0 (optional, router/pretier0.py): regex preparation, ~1 ms;
       answers only when certain: "Is X discussed?" when one sentence holds
       every word of X, and questions one clause's frame settles. It rewords the question for Tier 0 in "we/you" documents.
    0b. Tier 0 (optional, router/tier0.py): a local NLI model answers
       yes/no when one sentence of the document settles the question; no
       LLM call. Otherwise it defers and the steps below run.
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
from dataclasses import asdict, dataclass, field, replace

from router import pretier0 as pre
from router import choice, qtree, reader, spans
from router.bank import Bank
from router.menu import CRITERIA, DIVERSITY_FAMILY, INSTRUCTIONS, SPLIT_UNIT, TEXT_TYPES, family
from router.systemone import SystemOne, SystemOneError

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
# Why each kind of question the tiers don't (yet) answer is deferred (router/qtree.py, router/STATUS.md).
NOT_ANSWERED = {
    "span": "asks for a fact (who / when / how much...); fact answers are not built yet",
    "choice": "asks which of the named alternatives holds; choice answers are not built yet",
    "backward": "asks why; the text states rules, not their reasons, so lookup doesn't answer this",
    "forward": "asks what happens if something occurs; consequences are not answered by lookup yet",
    "process": "asks how to do something; procedures are not answered by lookup yet",
    "unclassified": "couldn't tell what kind of question this is (two questions in one, or not a question); "
                    "ask one question at a time",
}
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
    path: str  # "pretier0" | "tier0" | "tier0net" | "classifier" | "llm_task" | "llm_fallback" | "llm_judgment" | "span" | "llm_span" | "reader" | "llm_decide" | "choice" | "llm_choice" | "deferred" | "declined"
    reason: str
    asked: str = ""  # the question actually answered: the task's criteria, or the user's own question
    task: str | None = None
    answered_by: str = ""
    route: Route | None = None
    n_units: int = 1
    evidence: list = field(default_factory=list)  # [{"text", "label", "p"}], most confident first
    probabilities: dict = field(default_factory=dict)  # every option's probability, `answer` among them
    llm_calls: int = 0
    tier0: dict | None = None  # Tier 0's attempt (router/tier0.py), None if it wasn't run
    pretier0: dict | None = None  # Pre-Tier 0's check (router/pretier0.py), None if it wasn't run
    qtree: dict | None = None  # the question's kind (router/qtree.py), decided before anything else
    span: dict | None = None  # a fact question's search (router/spans.py), None for other kinds
    reader: dict | None = None  # Tier 2's read of a deferred fact (router/reader.py), None if it didn't run
    netreader: dict | None = None  # the reader network's attempt (router/netreader.py), None if it wasn't run

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
    def __init__(self, router: SystemOne, llm: SystemOne, bank: Bank, tier0=None, pretier0: bool = False,
                 classifiers: bool = True, reader=None, jev: bool = True, netreader=None):
        self.router = router
        self.llm = llm
        self.bank = bank
        self.tier0 = tier0  # a router.tier0.Tier0, tried before routing; None to skip
        self.pretier0 = pretier0  # run router/pretier0.py's checks first
        self.classifiers = classifiers  # False: free classifiers on standby, their tasks go to the fallback
        self.reader = reader  # Tier 2 for deferred facts: an LLM call (router/reader.py), checked by `llm`; None to skip
        # Tier 1 in /admin (router/stages.py). False: Jev neither answers nor checks anything --
        # the span and choice rules run without its pick, Tier 2 reads unchecked, and the yes/no
        # questions it would have answered go to Tier 2's `decide`. Routing needs it, so it's off too.
        self.jev = jev
        self.netreader = netreader  # a router.netreader.NetReader, tried after Tier 0; None to skip

    @property
    def _pick(self) -> SystemOne | None:
        """Jev where it picks a fact or a choice the rules missed; None when Tier 1 is off,
        which leaves `spans` and `choice` to answer by rule or defer."""
        return self.llm if self.jev else None

    def route(self, question: str) -> Route:
        out = self.router.choice(question, INSTRUCTIONS, CRITERIA)
        p_none, margin = route_scores(out["probabilities"])
        accepted = out["choice"] != "none_of_these" and p_none < P_NONE_MAX and margin >= MARGIN_MIN
        return Route(out["choice"], round(p_none, 3), round(margin, 3), accepted)

    def answer(self, question: str, document: str) -> Answer:
        frame = qtree.classify(question)
        result = self._answer_kind(question, document, frame)
        if result.path == "choice" and frame.leaf == "boolean":  # the document decided: it asked which one
            frame = replace(frame, leaf="choice", lehnert="disjunctive", options=frame.maybe_options,
                            trace=[*frame.trace, "the text states exactly one alternative: it asks which"])
        result.qtree = frame.to_dict()
        return result

    def _calls(self) -> int:
        return self.router.calls + (self.llm.calls if self.llm is not self.router else 0)

    def _answer_kind(self, question: str, document: str, frame) -> Answer:
        if frame.leaf == "judgmental":
            # Advice or a legal conclusion: a contract calling itself enforceable doesn't
            # make it so, so the lookup tiers never answer; Jev gives its reading.
            before = self._calls()
            result = self._fallback(question, document, None,
                                    f'asks for advice or a legal conclusion ("{frame.cue}"), which the text alone '
                                    f"can't settle, so the lookup tiers don't answer it")
            if result.path == "llm_fallback":  # Jev gave the reading; with Tier 1 off it keeps
                result.path = "llm_judgment"   # whatever `_decide` returned (llm_decide or deferred)
            result.llm_calls = self._calls() - before
            return result
        if frame.leaf == "request":
            return self._not_answered(question, frame, "declined",
                                      f'asks for a task ("{frame.cue}"), not a question about the text; '
                                      f"the router answers questions")
        if frame.leaf == "span":
            return self._span(question, document, frame)
        if frame.leaf == "choice":
            return self._choice(question, document, frame)
        if frame.leaf != "boolean":
            if self.reader is not None:
                # why / what if / how / unclassified: no local tier answers these, so the Tier 2 reader
                # reads the clauses about it (Jev's check still guards). Unmeasured, the user's call (2026-10-01).
                return self._read(question, document, frame, spans.SpanResult(False, reason=NOT_ANSWERED[frame.leaf]),
                                  self._calls())
            return self._not_answered(question, frame, "deferred", NOT_ANSWERED[frame.leaf])
        if frame.maybe_options and self.pretier0:
            # "Does the photographer deliver on a USB drive or through an online gallery?": if the
            # text states exactly one of the two, the question was which one. By rule only.
            r = choice.answer(replace(frame, options=frame.maybe_options), document, llm=None)
            if r.fired:
                return Answer(question=question, answer=r.answer, confidence=r.confidence, path="choice",
                              reason=f"the text states only one of the alternatives ({' / '.join(frame.maybe_options)}), "
                                     f"so the question asks which: {r.reason}",
                              asked=frame.lookup or question, answered_by="Pre-Tier 0 choice rules (router/choice.py)",
                              evidence=r.evidence, probabilities={r.answer: r.confidence}, span=r.to_dict())
        result = self._lookup(frame.lookup or question, document)
        result.question = question  # as asked; `asked` holds what the tiers answered
        return result

    def _span(self, question: str, document: str, frame) -> Answer:
        """A fact question: a span copied from the document (router/spans.py), by rule or
        picked by the LLM among the document's candidates; otherwise deferred."""
        before = self._calls()
        r = spans.answer(frame, document, llm=self._pick, rules=self.pretier0)
        calls = self._calls() - before
        if not r.fired:
            if self.reader is not None:
                return self._read(question, document, frame, r, before)
            a = self._not_answered(question, frame, "deferred",
                                   f"asks for a fact ({(frame.answer_type or '').lower()}); {r.reason}")
            a.span, a.llm_calls, a.evidence = r.to_dict(), calls, r.evidence
            return a
        rule = r.how == "rule"
        return Answer(question=question, answer=r.answer, confidence=r.confidence, path="span" if rule else "llm_span",
                      reason=r.reason, asked=frame.lookup or question,
                      answered_by="Pre-Tier 0 fact rules (router/spans.py)" if rule else
                      (self.llm.model_version or self.llm.model),
                      evidence=r.evidence, probabilities={r.answer: r.confidence} if rule else r.probabilities,
                      llm_calls=calls, span=r.to_dict())

    def _read(self, question: str, document: str, frame, r, before: int) -> Answer:
        """Tier 2: the reader copies the fact from the clauses about it; Jev checks the clause
        states it. The bake-off measured the types in spans.ANSWERED_TYPES; since 2026-10-01 it reads
        every fact the local tiers leave (ACTION, ENTITY... unmeasured), with Jev's check as the guard."""
        asked = frame.lookup or question
        try:
            t2 = reader.read(asked, reader.select_clauses(asked, document), self.reader,
                             check=self.llm.noul if self.jev else None, answer_type=frame.answer_type,
                             document=document, check_min=getattr(self.reader, "check_min", reader.CHECK_MIN))
        except (reader.ReaderError, SystemOneError) as e:
            option = getattr(self.reader, "option", None)  # router/tier2.py's Connected
            t2 = reader.ReaderResult(False, reason=f"the reader was unavailable ({str(e)[:120]})",
                                     usage={"option": option.id} if option else
                                     {"service_tier": getattr(self.reader, "service_tier", None)})
        calls = self._calls() - before
        name = getattr(self.reader, "name", "LLM reader")
        if not t2.fired:
            fact = f"asks for a fact ({(frame.answer_type or '').lower()}); " if frame.leaf == "span" else ""
            a = self._not_answered(question, frame, "deferred", f"{fact}{r.reason}; Tier 2: {t2.reason}")
            a.span, a.reader, a.llm_calls, a.evidence = r.to_dict(), t2.to_dict(), calls, r.evidence
            a.answered_by = f"Tier 2: {name}, no answer it could stand behind"  # it read the clauses, not the tree
            return a
        return Answer(question=question, answer=t2.answer, confidence=t2.check, path="reader",
                      reason=f"{r.reason}; Tier 2: {t2.reason}", asked=asked,
                      answered_by=f"Tier 2: {name}, " + (f"checked by {self.llm.model_version or self.llm.model}"
                                                         if self.jev else "unchecked (Tier 1 off)"),
                      evidence=[{"text": t2.clause, "label": t2.answer, "p": t2.check}],
                      probabilities={t2.answer: t2.check}, llm_calls=calls, span=r.to_dict(), reader=t2.to_dict())

    def _choice(self, question: str, document: str, frame) -> Answer:
        """A choice question: the one alternative the text states (router/choice.py), by rule
        or picked by the LLM; otherwise deferred."""
        before = self._calls()
        r = choice.answer(frame, document, llm=self._pick, rules=self.pretier0)
        calls = self._calls() - before
        if not r.fired:
            a = self._not_answered(question, frame, "deferred",
                                   f"asks which of {' / '.join(frame.options)} holds; {r.reason}")
            a.span, a.llm_calls = r.to_dict(), calls
            return a
        rule = r.how == "rule"
        return Answer(question=question, answer=r.answer, confidence=r.confidence,
                      path="choice" if rule else "llm_choice", reason=r.reason, asked=frame.lookup or question,
                      answered_by="Pre-Tier 0 choice rules (router/choice.py)" if rule else
                      (self.llm.model_version or self.llm.model),
                      evidence=r.evidence, probabilities={r.answer: r.confidence} if rule else r.probabilities,
                      llm_calls=calls, span=r.to_dict())

    def _not_answered(self, question: str, frame, path: str, reason: str) -> Answer:
        """No answer and no LLM call: the question isn't one these tiers answer."""
        return Answer(question=question, answer="not answered", confidence=0.0, path=path, reason=reason,
                      asked=frame.lookup or question, answered_by="question tree (router/qtree.py)")

    def _lookup(self, question: str, document: str) -> Answer:
        calls_before = self._calls()
        p0 = pre.check(question, document) if self.pretier0 else None
        if p0 is not None and p0.fired:
            return Answer(question=question, answer=p0.answer, confidence=1.0, path="pretier0", reason=p0.reason,
                          asked=p0.rewritten or question, answered_by="Pre-Tier 0 (regex)", evidence=p0.evidence,
                          probabilities={p0.answer: 1.0}, pretier0=p0.to_dict())
        t0_question = p0.rewritten if p0 is not None and p0.rewritten else question  # in the document's voice
        t0 = self.tier0.answer(t0_question, document) if self.tier0 is not None else None
        if t0 is not None and t0.fired:
            return Answer(
                question=question, answer=t0.answer, confidence=t0.confidence, path="tier0", reason=t0.reason,
                asked=t0.hypothesis, answered_by="Tier 0 (local NLI, nli-deberta-v3-xsmall)",
                n_units=t0.n_units, evidence=t0.evidence,
                probabilities={t0.answer: t0.confidence}, tier0=t0.to_dict(),
                pretier0=p0.to_dict() if p0 is not None else None,
            )
        nr = self.netreader.answer(t0_question, document) if self.netreader is not None else None
        if nr is not None and nr.fired:
            return Answer(
                question=question, answer=nr.answer, confidence=nr.confidence, path="tier0net", reason=nr.reason,
                asked=t0_question, answered_by=self.netreader.name, n_units=nr.n_units, evidence=nr.evidence,
                probabilities={nr.answer: nr.confidence}, tier0=t0.to_dict() if t0 is not None else None,
                pretier0=p0.to_dict() if p0 is not None else None, netreader=nr.to_dict(),
            )
        result = self._answer(question, document)
        result.tier0 = t0.to_dict() if t0 is not None else None
        result.netreader = nr.to_dict() if nr is not None else None
        result.pretier0 = p0.to_dict() if p0 is not None else None
        result.llm_calls = self._calls() - calls_before
        return result

    def _answer(self, question: str, document: str) -> Answer:
        if not self.jev:
            # Routing is a Jev call, and so is the document-type check before a classifier runs.
            return self._fallback(question, document, None, "Tier 1 (Jev) is off, so no routing")
        if not self.classifiers:
            # Routing only picks a classifier or the task's own question; with the
            # classifiers on standby it would cost an LLM call to reach another one.
            return self._fallback(question, document, None, "the free classifiers are on standby, so no routing")
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
        if not self.classifiers:
            return self._fallback(question, document, route, f"matched {task}; the free classifier is on standby")
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

    def _fallback(self, question: str, document: str, route: Route | None, reason: str) -> Answer:
        if not self.jev:
            return self._decide(question, document, route, reason)
        answer, conf, probs = _yes_no(self.llm.noul(document, question))
        return Answer(question=question, answer=answer, confidence=round(conf, 3), path="llm_fallback",
                      reason=reason, asked=question, answered_by=self.llm.model_version or self.llm.model,
                      route=route, probabilities=probs)

    def _decide(self, question: str, document: str, route: Route | None, reason: str) -> Answer:
        """Tier 1 off: the hosted reader decides the yes/no from the clauses it is shown
        (router/reader.py `decide`). Nothing checks it -- the verbatim-copy check that guards
        Tier 2's fact reading can't apply to a yes/no, and Jev is the thing that was turned
        off -- so the confidence is the model's own and the answer says it is unchecked. The
        bake-off never measured this path; it only scored fact reading."""
        if self.reader is None:
            return Answer(question=question, answer="not answered", confidence=0.0, path="deferred",
                          reason=f"{reason}, and Tier 2 is off too, so nothing is left to answer it",
                          asked=question, answered_by="the tiers /admin has on", route=route)
        try:
            d = reader.decide(question, reader.select_clauses(question, document), self.reader)
        except (reader.ReaderError, SystemOneError) as e:
            d = reader.ReaderResult(False, reason=f"the reader was unavailable ({str(e)[:120]})")
        name = getattr(self.reader, "name", "LLM reader")
        if not d.fired:
            return Answer(question=question, answer="not answered", confidence=0.0, path="deferred",
                          reason=f"{reason}; Tier 2: {d.reason}", asked=question,
                          answered_by=f"Tier 2: {name}", route=route, reader=d.to_dict())
        conf = round(d.self_p, 3) if d.self_p is not None else 0.0
        return Answer(question=question, answer=d.answer, confidence=conf, path="llm_decide",
                      reason=f"{reason}; {d.reason}", asked=question,
                      answered_by=f"Tier 2: {name}, unchecked (Tier 1 off)",
                      evidence=[{"text": d.clause, "label": d.answer, "p": conf}] if d.clause else [],
                      probabilities={d.answer: conf} if d.answer else {}, route=route, reader=d.to_dict())
