"""Fact questions ("When is rent due?", "How long is the notice period?"):
answer with a span copied from the document, or defer (router/STATUS.md, M2).

1. Candidates: legal-aware patterns find every value of the question's answer
   type in the document ("thirty (30) days", "$2,400", "the first day of each
   calendar month", "the laws of the State of Delaware", the parties...). They
   are tuned to over-find: selection happens next.
2. The question's frame minus its wh-word finds the clauses that talk about
   the thing asked about (QA-SRL, He et al. 2015): every key term of the
   question must appear in the clause, itself or as a lexicon synonym.
3. Rule: one such clause, one candidate of the right type in it, no condition
   in it -> that candidate, no LLM call. Otherwise, when an LLM is given, it
   picks among the candidates with "none of these" as an option (TypeSafe's
   select-instead-of-generate); a pick below MIN_P, or none, defers.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from router import frames
from router.pretier0 import PARTY_SCAN_CHARS, find_parties

# The LLM's pick must be at least this likely, and it only picks among clauses that name
# every key term: on held-out CUAD contracts (2026-09-30) its picks from such clauses were
# 14/14 right, from looser pools 20/26, from unanchored document dates 17/27, and picks
# below 0.9 were 0/7 right.
MIN_P = 0.90
# In a document this short the LLM sees most of it, so it may also pick when no clause names
# every key term (paraphrased questions); in long contracts such picks were 77% right.
import os as _os
LOOSE_MAX_CHARS = int(_os.environ.get("SPANS_LOOSE_MAX_CHARS", "8000"))  # 96/96 right under 8k chars; 42/53 above
MAX_OPTIONS = 40  # candidates offered to the LLM
MAX_STATE_CHARS = 12_000  # clauses shown to the LLM
ANSWERED_TYPES = frozenset("DATE DURATION MONEY PERCENT FREQUENCY PARTY JURISDICTION DEFINITION CARDINAL AGE".split())

_NUM_WORD = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
             r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
             r"thousand|a|an)")
_NUM = rf"(?:\d[\d,]*(?:\.\d+)?|{_NUM_WORD}(?:[\s-]+{_NUM_WORD})*)"
_PAREN = r"(?:\s*\(\s*\d[\d,]*(?:\.\d+)?\s*\))?"
_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|" \
          r"Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
_ORDINAL = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|fifteenth|twentieth|last|" \
           r"\d{1,2}(?:st|nd|rd|th))"
_UNIT = r"(?:second|minute|hour|day|week|month|year)s?"

PATTERNS = {
    "DURATION": [
        rf"\b{_NUM}{_PAREN}\s*(?:full |calendar |business |working |banking |consecutive )?{_UNIT}\b",
        rf"\b{_NUM}{_PAREN}-(?:second|minute|hour|day|week|month|year)\b",  # "1-year", "thirty-day"
        r"\bmonth[- ]to[- ]month\b|\byear[- ]to[- ]year\b",
    ],
    "DATE": [
        rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?{_MONTHS},?\s+\d{{4}}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
        rf"\bthe\s+{_ORDINAL}\s+(?:business\s+|calendar\s+)?day\s+of\s+(?:each|every|the|any)\s+(?:calendar\s+)?"
        r"(?:month|year|quarter|week)\b",
        rf"\bthe\s+{_ORDINAL}\s+of\s+(?:each|every|the)\s+(?:calendar\s+)?month\b",
        rf"\b(?:this\s+)?{_ORDINAL}\s+day\s+of\s+{_MONTHS},?\s+\d{{4}}\b",
        rf"\b{_MONTHS},?\s+\d{{4}}\b",
    ],
    "MONEY": [
        r"(?:US\$|USD\s?|\$|€|£|EUR\s?|GBP\s?)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand|[mMkK]\b))?",
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|USD|euros|EUR)\b",
    ],
    "PERCENT": [
        rf"\b\d+(?:\.\d+)?\s?%|\b{_NUM}\s+percent(?:\s*\(\s*\d+(?:\.\d+)?\s*%\s*\))?",
    ],
    "FREQUENCY": [
        rf"\b(?:once|twice|thrice|{_NUM}{_PAREN}\s+times)\s+(?:per|a|each|every|in\s+any|during\s+any)\s+"
        r"(?:calendar\s+|fiscal\s+|contract\s+)?(?:day|week|month|quarter|year|twelve[- ]month\s+period|12[- ]month\s+period)\b",
        r"\b(?:daily|weekly|bi-?weekly|monthly|quarterly|semi-?annually|annually|yearly)\b",
        r"\bevery\s+(?:\w+\s+(?:\(\d+\)\s+)?)?(?:day|week|month|quarter|year)s?\b",
        r"\b(?:each|every|per)\s+(?:calendar\s+)?(?:month|year|quarter|week)\b",
    ],
    "JURISDICTION": [],  # built from PLACES below
    "AGE": [  # "sixteen (16) years of age", "18 years old", "the age of 13"
        rf"\b{_NUM}{_PAREN}\s+years?\s+(?:of\s+age|old)\b",
        rf"\bthe\s+age\s+of\s+{_NUM}{_PAREN}",
        rf"\b(?:aged|age)\s+{_NUM}{_PAREN}\b",
    ],
    "CARDINAL": [
        rf"\b{_NUM}{_PAREN}\s+(?!{_UNIT}\b|percent\b|times\b)[a-z]+(?:\s+[a-z]+)?\b",
    ],
}
# Governing-law places: a gazetteer, so a company name before "law" isn't taken for one.
PLACES = """Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|
Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|
Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|
Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|
Wisconsin|Wyoming|District of Columbia|Puerto Rico|Ontario|Quebec|British Columbia|Alberta|Manitoba|Saskatchewan|
Nova Scotia|New Brunswick|England and Wales|England|Wales|Scotland|Northern Ireland|United States of America|
United States|Canada|Mexico|Brazil|Argentina|Chile|United Kingdom|Ireland|France|Germany|Netherlands|Belgium|
Luxembourg|Switzerland|Austria|Italy|Spain|Portugal|Sweden|Norway|Denmark|Finland|Poland|Israel|India|People's Republic
of China|PRC|China|Hong Kong|Singapore|Japan|Republic of Korea|South Korea|Korea|Taiwan|Australia|New Zealand|
South Africa|United Arab Emirates|Cayman Islands|Bermuda|British Virgin Islands|Russia|Russian Federation|Turkey|
Thailand|Malaysia|Indonesia|Philippines|Vietnam|Hungary|Czech Republic|Greece|Cyprus|Malta""".replace("\n", "")
PATTERNS["JURISDICTION"] = [rf"\b(?:the\s+)?(?:(?:State|Commonwealth|Province|Republic|Kingdom)\s+of\s+)?(?:{PLACES})\b"]
_COMPILED = {t: [re.compile(p) for p in ps] for t, ps in PATTERNS.items()}
_DEFINITION = re.compile(r"[\"“]([A-Z][^\"”]{1,60})[\"”]\s*(?:\)\s*)?(?:shall\s+)?(?:means?|refers?\s+to|shall\s+have\s+the\s+"
                         r"meaning|is\s+defined\s+as|includes?)\s+([^;]{3,400}?)(?=[.;]\s|[.;]$|$)")
# Questions about one of several values in time: the rule tier leaves those to the LLM.
_QUALIFIED = re.compile(r"\b(?:old|older|previous|prior|original|former|initial|new|newer|current|updated|revised|"
                        r"increased|reduced|reduction|increase|before the|after the)\b", re.I)
def qualified(question: str) -> bool:
    """ "What was the old rent?" (not "the law of New York": names don't count)."""
    q = re.sub(r"(?<!^)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", " ", question)
    q = re.sub(r"\bhow old\b|\byears? old\b", " ", q, flags=re.I)  # an age question, not "the old rent"
    return bool(_QUALIFIED.search(q))


# Conditions in the clause: the rule tier leaves those to the LLM or defers.
_CONDITION = re.compile(r"\b(?:if|unless|except|provided that|provided, however|subject to|notwithstanding|in the event|"
                        r"otherwise)\b", re.I)
# Words of a fact question that describe the answer, not what it's about.
_ANSWER_WORDS = frozenset(frames.normalize(w) for w in """
who whom whose what when where which why how much many long often soon frequently quickly early late far old is are was
were be been do does did can could will would shall should may might must has have had the a an of for to in on by at
with from as this that these those it its there date day time amount sum number period length duration deadline
frequency percentage percent rate price cost value figure kind type sort state country jurisdiction place location
address party parties who's what's specify specified say says state states agreement contract lease policy document
""".split())
# Question words -> the words clauses use for them: "When is rent due?" / "Tenant shall pay rent on..."
# Safe synonyms: the rule tier may answer through them.
SPAN_SYNONYMS = {
    "due": {"pay", "payable", "paid", "payment", "due"},
    "expir": {"expir", "expiration", "expiry"},
    "start": {"start", "commenc", "begin"},
    "begin": {"begin", "commenc", "start"},
    "commenc": {"commenc", "begin", "start"},
    "govern": {"govern", "governed", "construed"},
    "law": {"law", "laws", "govern", "governed"},
    "renew": {"renew", "renewal", "automatically"},
    "notic": {"notic", "notify", "notification"},
    "late": {"late", "overdue", "delinquent"},
    "fee": {"fee", "charge"},
    "cancel": {"cancel", "terminat"},
}  # not minimum ≈ "at least": it took "at least 25 images" for "a minimum of 400" (spans_test2)
# Loose ones only pick the clauses shown to the LLM: "When does the lease expire?" may be
# answered by a clause about the term's end, but the rule tier mustn't take the term's start.
LOOSE_SYNONYMS = {
    "expir": {"end", "terminat", "term"},
    "effect": {"commenc", "begin", "start", "date"},
    "sign": {"execut", "dated", "date", "made", "entered"},
    "cost": {"fee", "price", "charge"},
    "fee": {"cost", "penalty", "price"},
    "late": {"past", "received"},
    "renew": {"extend", "extension"},
    "terminat": {"cancel", "end"},
}
# Filler nouns: they rank clauses but aren't required ("the water and sewer bills").
SOFT_TERMS = frozenset(frames.normalize(w) for w in "bill bills charge charges service services cost costs thing things "
                                                     "amount amounts expense expenses".split())
# A concept's own head words ("term" in "renewal term") aren't extra terms.
_CONCEPT_HEADS = frozenset(frames.normalize(w) for w in "term terms period agreement contract".split())
# Verbs that carry no content: "How many parking spaces does the tenant get?"
LIGHT_VERBS = frozenset(frames.normalize(w) for w in "get gets got have has had receive receives give gives take takes "
                                                    "make makes do does allow allows let lets need needs keep keeps "
                                                    "offer offers provide provides become becomes became each every "
                                                    "any all per go goes went come comes".split())


@dataclass
class Candidate:
    type: str
    text: str
    unit: int  # index into the document's units
    start: int  # offset in the unit
    term: str = ""  # DEFINITION: the defined term


@dataclass
class SpanResult:
    fired: bool
    answer: str | None = None
    confidence: float = 0.0
    how: str = ""  # "rule" | "llm" | "" (deferred)
    reason: str = ""
    evidence: list = field(default_factory=list)  # [{"text", "label", "p"}]
    options: list = field(default_factory=list)  # what the LLM chose among
    probabilities: dict = field(default_factory=dict)
    llm_calls: int = 0
    pool: str = ""  # what the LLM chose from: "strict" (clauses naming every key term), "loose", "document"

    def to_dict(self) -> dict:
        return asdict(self)


def units(document: str) -> list:
    """Clauses: sentences, cut again at semicolons ("Landlord shall pay for
    water; Tenant shall pay for electricity")."""
    from router.tier0 import sentences
    out, heading = [], ""
    for s in sentences(document):
        if _HEADING_ONLY.match(s):
            heading = f"{heading} {s}".strip()  # "9. GOVERNING LAW." heads the next clause
            continue
        parts = [p.strip() for p in re.split(r";\s+(?=\w)", s) if len(p.strip()) > 3]
        if heading and parts:
            parts[0] = f"{heading} {parts[0]}"
            heading = ""
        out += parts
    if heading:
        out.append(heading)
    return out


_HEADING_ONLY = re.compile(r"^\s*(?:(?:Section|Article|§)\s*)?(?:\d+(?:\.\d+)*\.?|[IVX]+\.)?\s*[A-Z][A-Z0-9 &/,\-]{2,60}[.:]?\s*$")


def candidates(answer_type: str, unit_texts: list, parties: list) -> list:
    out = []
    for i, u in enumerate(unit_texts):
        if answer_type == "PARTY":
            for p in parties:
                for m in re.finditer(rf"\b{re.escape(p)}\b", u):
                    out.append(Candidate("PARTY", p, i, m.start()))
            continue
        if answer_type == "DEFINITION":
            for m in _DEFINITION.finditer(u):
                out.append(Candidate("DEFINITION", m.group(2).strip(), i, m.start(2), term=m.group(1)))
            continue
        spans = []
        for rx in _COMPILED.get(answer_type, []):
            for m in rx.finditer(u):
                spans.append((m.start(), m.end()))
        spans.sort(key=lambda s: (s[0], -s[1]))
        last_end = -1
        for s, e in spans:  # longest first at each start, no overlaps
            if s >= last_end:
                last_end = e
                if answer_type == "DURATION" and _is_age(u[:s], u[e:], u):
                    continue  # "users under eighteen years of age": an age, not a duration
                out.append(Candidate(answer_type, u[s:e].strip(" ,.;:"), i, s))
    return out


def _is_age(before: str, after: str, unit: str) -> bool:
    if re.match(r"\s*(?:\(\s*\d+\s*\)\s*)?(?:of\s+age|old|or\s+older|or\s+younger|of\s+age\s+or)\b", after, re.I):
        return True
    if re.search(r"\b(?:age|aged|ages)\s+(?:of\s+)?$", before, re.I):
        return True
    return bool(re.search(r"\b(?:under|over|below|above|younger\s+than|older\s+than)\s+$", before, re.I)
                and re.search(r"\b(?:minor|minors|child|children|age|aged|old|teen|teens|adult|adults)\b", unit, re.I))


def key_terms(question: str) -> list:
    """What the question is about, as normalized words and lexicon concepts:
    "When is rent due?" -> ["rent", "due"]."""
    toks = frames.tokens(question)
    terms = []
    for t in frames.tag(toks, frames._BASE_TRIE):
        if t.kind in ("MODAL", "BLOCK", "BREAK"):
            continue
        if t.kind == "WORD":
            if t.name in _ANSWER_WORDS or len(t.name) < 2 or t.name in frames.STOPWORDS or t.name in LIGHT_VERBS:
                continue
            terms.append(t.name)
        elif (t.kind, t.name) not in frames._GENERIC_CONCEPTS:
            terms.append(f"{t.kind}:{t.name}")
            # "renewal term" is the TERM concept, but "renewal" is what the question is about
            end = t.end if t.end > t.start else t.start + 1
            for w in toks[t.start:end]:
                if w not in _ANSWER_WORDS and w not in frames.STOPWORDS and w not in LIGHT_VERBS and len(w) > 2 \
                        and w not in _CONCEPT_HEADS:
                    terms.append(w)
    return list(dict.fromkeys(terms))


def _unit_terms(text: str) -> set:
    toks = frames.tokens(text)
    found = set(toks)
    for t in frames.tag(toks, frames._BASE_TRIE):
        if t.kind != "WORD":
            found.add(f"{t.kind}:{t.name}")
    return found


def _covers(term: str, found: set, loose: bool = False) -> bool:
    if term in found:
        return True
    if ":" not in term:
        variants = {term} | SPAN_SYNONYMS.get(term, set()) | SPAN_SYNONYMS.get(term[:5], set())
        if loose:
            variants |= LOOSE_SYNONYMS.get(term, set()) | LOOSE_SYNONYMS.get(term[:5], set())
        # One word a prefix of the other ("renew"/"renewal"), not merely sharing five letters
        # ("electricity"/"electronic").
        return any(v in found or any(len(min(v, w, key=len)) >= 4 and (w.startswith(v) or v.startswith(w))
                                     for w in found if ":" not in w) for v in variants)
    return False


def _heading(unit: str) -> set:
    """The words of a clause's heading: "2. RENT. Tenant shall..." -> {"rent"}."""
    m = re.match(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?((?:[A-Z][A-Z&/\- ]{2,40}))[.:]", unit)
    return set(frames.tokens(m.group(1))) if m else set()


def _agent(unit: str, parties: list, term_words: set) -> str | None:
    """For "who" questions: the party that is the subject of the clause's verb
    the question is about ("Landlord shall repair the heating system")."""
    from router.qtree import _nlp
    doc = _nlp()(unit)
    for t in doc:
        if t.pos_ != "VERB":
            continue
        stem = frames.normalize(t.lower_)
        if not (stem in term_words or stem[:5] in {w[:5] for w in term_words} or t.lemma_.lower() in term_words):
            continue
        subj = next((c for c in t.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if subj is None and t.dep_ == "conj":
            subj = next((c for c in t.head.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if subj is not None:
            text = unit[subj.left_edge.idx:subj.right_edge.idx + len(subj.right_edge)]
            hit = [p for p in parties if re.search(rf"\b{re.escape(p)}\b", text)]
            if len(hit) == 1:
                return hit[0]
    return None


_DOC_DATE_Q = re.compile(r"\bdate of (?:this|the) (?:agreement|contract|lease|document)\b|\bwhen (?:was|is|were) (?:this|the) "
                         r"(?:agreement|contract|lease) (?:signed|dated|made|executed|entered into|concluded)\b|"
                         r"\b(?:agreement|contract|lease) date\b", re.I)
_PREAMBLE_DATE = re.compile(r"\b(?:dated|as of|made|entered into|executed|effective)\b|\bday of\b", re.I)


def document_date_kind(question: str) -> str | None:
    """Which of the document's own dates a date question asks for ("agreement", "effectiv",
    "commenc", "start", "begin", "expir"), or None when it asks about some other date."""
    terms = key_terms(question)
    about_doc = re.search(r"\b(?:this|the)\s+(?:agreement|contract|lease|license|nda|plan)\b", question, re.I)
    kind = "agreement" if _DOC_DATE_Q.search(question) else next(
        (k for k in ("effectiv", "commenc", "start", "begin", "expir") if any(t[:5] == k[:5] for t in terms)), None) \
        or ("effectiv" if re.search(r"\bkick(?:s|ed)? in\b|\b(?:take|takes|go|goes) (?:into )?effect\b", question, re.I)
            else None)
    return kind if kind and (about_doc or kind == "agreement") else None


def _term_words(term: str) -> set:
    """The words a key term can appear as in a clause."""
    if ":" in term:
        kind, name = term.split(":", 1)
        words = {w for p in frames.LEXICON.get((kind, name), []) for w in frames.tokens(p)}
        return {w for w in words if w not in frames.STOPWORDS and len(w) > 2} or {name.lower()}
    return {term} | SPAN_SYNONYMS.get(term, set()) | SPAN_SYNONYMS.get(term[:5], set())


def _attached(unit: str, unit_cands: list, terms: list) -> list:
    """The candidates inside the parse subtree of a clause word that matches a key
    term (its head, for a noun): "effective as of [Nov 1, 2002]", not "remain in
    effect until [Nov 1, 2007]" (QA-SRL: the answer is an argument of the predicate)."""
    from router.qtree import _nlp
    words = set().union(*(_term_words(t) for t in terms))
    doc = _nlp()(unit)
    spans_ = []
    for t in doc:
        norm = frames.normalize(t.lower_)
        if norm in words or any(len(min(norm, w, key=len)) >= 4 and (norm.startswith(w) or w.startswith(norm))
                                for w in words):
            anchor = t if t.pos_ in ("VERB", "ADJ", "AUX") else t.head
            if anchor.dep_ in ("xcomp", "advcl") and anchor.head.lemma_ == "be":
                anchor = anchor.head  # the purpose hangs off "be"
            elif anchor.dep_ in ("acl", "relcl") and anchor.head.pos_ in ("NOUN", "PROPN"):
                anchor = anchor.head  # "[sixteen (16) years of age] to buy Premium": the noun it describes
            spans_.append((anchor.left_edge.idx, anchor.right_edge.idx + len(anchor.right_edge)))
    return [c for c in unit_cands if any(s <= c.start < e for s, e in spans_)]


def answer(frame, document: str, llm=None) -> SpanResult:
    typ = frame.answer_type
    if typ not in ANSWERED_TYPES:
        return SpanResult(False, reason=f"answers of type {typ} are not built yet")
    us = units(document)
    parties = find_parties(document[:PARTY_SCAN_CHARS]) if typ == "PARTY" else []
    cands = candidates(typ, us, parties)
    if not cands:
        return SpanResult(False, reason=f"the text has no {typ.lower()} at all")
    question = frame.lookup or frame.question
    if typ == "DEFINITION":
        return _definition(question, us, cands, frame, llm)
    terms = key_terms(question)
    if typ == "DATE" and (kind := document_date_kind(question)):
        # Dates of the document itself ("When does this agreement become effective?"): only their
        # anchors answer by rule; amendments and exhibits carry other "effective" dates.
        return _document_date(us, cands, frame, llm, kind)
    if not terms:
        return SpanResult(False, reason="couldn't tell what the question is about")
    unit_terms = {i: _unit_terms(us[i]) for i in {c.unit for c in cands}}
    required = [t for t in terms if t not in SOFT_TERMS] or terms
    matching = sorted(i for i, found in unit_terms.items() if all(_covers(t, found) for t in required))
    if len(matching) > 1:
        headed = [i for i in matching if any(t in _heading(us[i]) or t[:5] in {w[:5] for w in _heading(us[i])}
                                             for t in terms if ":" not in t)]
        if len(headed) == 1:
            matching = headed  # "How much is the rent?": the clause headed RENT, not the late-fee clause
    conditional = typ != "JURISDICTION" and any(_CONDITION.search(us[i]) for i in matching)
    # "What was the old rent before the reduction?": a value that changed; which one is meant is
    # for the LLM to read, not the rule.
    conditional = conditional or qualified(question)
    if typ == "PARTY":
        tw = {t for t in terms if ":" not in t} | {t.split(":")[1].lower() for t in terms if ":" in t}
        agents = {a for i in matching if (a := _agent(us[i], parties, tw))}
        if matching and len(agents) == 1 and not conditional and len(matching) <= 2:
            a = agents.pop()
            return SpanResult(True, a, 1.0, "rule", reason="one party is the subject of the clause's verb",
                              evidence=[{"text": us[i], "label": a, "p": 1.0} for i in matching])
    else:
        # Rule: the clauses about it hold one value attached to the words the question is about.
        attached = [c for i in matching for c in _attached(us[i], [c for c in cands if c.unit == i], terms)]
        values = {_value_key(typ, c.text) for c in attached}
        if matching and len(values) == 1 and not conditional and len(matching) <= 2:
            c = attached[0]
            return SpanResult(True, c.text, 1.0, "rule",
                              reason=f"one clause names the {' '.join(t.split(':')[-1].lower() for t in terms)} "
                                     f"and one {typ.lower()} attached to it",
                              evidence=[{"text": us[c.unit], "label": c.text, "p": 1.0}])
    if llm is None or (not matching and len(document) > LOOSE_MAX_CHARS):
        why = "no clause names everything the question asks about" if not matching else \
            "the clause about it has a condition" if conditional else \
            f"no single {typ.lower()} is attached to what the question asks about"
        return SpanResult(False, reason=why)
    return _choose(frame, us, cands, matching, terms, llm)


_QUOTED = re.compile(r"[\"“‘']([^\"”’']{2,60})[\"”’']")


def _definition(question: str, us: list, cands: list, frame, llm) -> SpanResult:
    """ "What does 'Personal Data' mean?": the definition of that very term, not one
    that mentions it."""
    m = _QUOTED.search(question)
    if m:
        term = m.group(1)
    else:
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question[1:])
        term = max(caps, key=len) if caps else ""
    key = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    hits = [c for c in cands if key(c.term) == key(term)] if term else []
    if len({key(c.text) for c in hits}) == 1:
        c = hits[0]
        return SpanResult(True, c.text, 1.0, "rule", reason=f'the text defines "{c.term}"',
                          evidence=[{"text": us[c.unit], "label": c.text, "p": 1.0}])
    return SpanResult(False, reason=f'the text doesn\'t define "{term}"' if term and not hits else
                      "couldn't tell which term the question asks about" if not term else
                      f'the text defines "{term}" more than once')


def _value_key(typ: str, text: str) -> str:
    """Values that name the same thing count once: "the State of New York" / "New York"."""
    t = text.lower()
    if typ == "JURISDICTION":
        t = re.sub(r"^(?:the\s+)?(?:(?:state|commonwealth|province|republic|kingdom)\s+of\s+)?", "", t)
    return re.sub(r"[^a-z0-9]", "", t)


# The document's own dates, by their strongest anchors: a defined term right after the date
# ('as of March 29, 1999 (the "Effective Date")') or the preamble ("This Agreement is entered
# into as of..."). A date after "that certain ... dated" belongs to another agreement.
_THIS_DOC = re.compile(r"\bthis\b[^.;]{0,60}?\b(?:agreement|contract|lease|amendment|addendum|license|plan)\b", re.I)
_MADE = re.compile(r"\b(?:is|was|are)?\s*(?:made|entered\s+into|dated|executed|effective)\b[^.;]{0,80}$", re.I)
_OTHER_DOC = re.compile(r"\bthat\s+certain\b|\b(?:agreement|contract|lease|amendment|letter)\s+dated\b[^.;]{0,20}$", re.I)
_DEFINED_DATE = r"\s*,?\s*\(\s*(?:the|this|such)?\s*[\"“](?:{})\s+Date[\"”]"
_DOC_DATE_KINDS = {"effectiv": "Effective|Commencement|Start", "commenc": "Commencement|Effective|Start",
                   "start": "Start|Commencement|Effective", "begin": "Commencement|Start|Effective",
                   "expir": "Expiration|Termination|End", "agreement": "Agreement|Execution|Signing"}


def _anchored_dates(us: list, cands: list, kinds: str) -> list:
    rx = re.compile(_DEFINED_DATE.format(kinds))
    return [c for c in cands if rx.match(us[c.unit][c.start + len(c.text):c.start + len(c.text) + 60])]


# "This Agreement shall commence on", "The Term will begin on", "effective as of": right before the date.
_STARTS = re.compile(r"\b(?:commenc\w*|begin\w*|take\s+effect|become\s+effective|be\s+effective|effective)\s*"
                     r"(?:on|as\s+of|from|upon)?\s*(?:the\s+date\s+of\s*)?$", re.I)
_THIS_TERM = re.compile(r"\bthis\b[^.;]{0,60}?\b(?:agreement|contract|lease|license)\b|\bthe\s+(?:initial\s+)?term\b", re.I)
# "Constellation may terminate this Agreement, effective as of December 31, 2023": the date some other
# event takes effect (an ending, renewal or change), not the agreement's start. Lowercase only, so the
# words in prose count and headings ("TERM AND TERMINATION") and document names ("This Amended and
# Restated Agreement", "This SECOND AMENDMENT ... effective as of") don't.
_OTHER_EVENT = re.compile(r"\b(?:terminat|expir|cancel|rescind|rescission|withdr[ae]w|suspen[ds]|amend|modif|renew|"
                          r"extend|extension|replac|supersed|assign|transfer|appoint)[a-z]*\b|\belect(?:s|ed|ion)?\b")


def other_event(text: str):
    """The first other-event word in `text`, skipping "this amendment" (the document itself) and
    "unless ... terminated," (a condition, not the event the date belongs to); None if none."""
    text = re.sub(r"\bunless\b[^,]*,?", " ", text)
    return next((m for m in _OTHER_EVENT.finditer(text) if not re.search(r"\bthis\s+$", text[:m.start()], re.I)),
                None)


_DATE_KIND_NAMES = {"effectiv": "effective", "commenc": "commencement", "start": "start", "begin": "start",
                    "expir": "expiration"}


def _preamble_dates(us: list, cands: list, starts: bool = False) -> list:
    """The document's own date: signed ("This Agreement is entered into as of <date>") or,
    with `starts`, when it takes effect ("This Agreement shall commence on <date>")."""
    out = []
    for c in cands:
        before = us[c.unit][:c.start]
        if _OTHER_DOC.search(before):
            continue
        if starts:
            sentence = re.split(r"[.;]\s", before)[-1][-150:]
            if _THIS_TERM.search(before) and _STARTS.search(before[-40:]) and not other_event(sentence):
                out.append(c)
        elif c.unit < 6 and _THIS_DOC.search(before) and _MADE.search(before):
            out.append(c)
    return out


def _document_date(us: list, cands: list, frame, llm, kind: str = "agreement") -> SpanResult | None:
    """The document's own date (signed / effective / expiring), by its anchors;
    None when the question isn't about the document itself."""
    found = _anchored_dates(us, cands, _DOC_DATE_KINDS[kind])
    if not found and kind == "agreement":
        found = _preamble_dates(us, cands)
    elif not found and kind in ("effectiv", "commenc", "start", "begin"):
        found = _preamble_dates(us, cands, starts=True)
    values = {_value_key("DATE", c.text) for c in found}
    if len(values) == 1:
        c = found[0]
        return SpanResult(True, c.text, 1.0, "rule",
                          reason="the date the document defines as its " + ("date" if kind == "agreement" else
                                                                            f"{_DATE_KIND_NAMES[kind]} date"),
                          evidence=[{"text": us[c.unit], "label": c.text, "p": 1.0}])
    if llm is None or not found:
        return SpanResult(False, reason="the document names no single date of its own for this" if not found else
                          f"the document gives {len(values)} such dates")
    return _choose(frame, us, found, sorted({c.unit for c in found}), [], llm, kind="document")


def _choose(frame, us: list, cands: list, matching: list, terms: list, llm, kind: str = "") -> SpanResult:
    """The LLM picks the answer among the candidates, or "none of these"."""
    kind = kind or ("strict" if matching else "loose")
    if matching:
        pool = [c for c in cands if c.unit in set(matching)]
    else:
        # No clause names every key term: offer the clauses that name the most.
        score = lambda i: sum(_covers(t, _unit_terms(us[i]), loose=True) for t in terms)
        scored = sorted({c.unit for c in cands}, key=lambda i: -score(i))
        best = [i for i in scored if score(i) > 0][:8]
        if not best:
            return SpanResult(False, reason="no clause mentions what the question asks about")
        pool = [c for c in cands if c.unit in set(best)]
    # Every clause in the pool goes into the state, most relevant first; each option is
    # described by its occurrence in the most relevant clause.
    relevance = {i: sum(_covers(t, _unit_terms(us[i]), loose=True) for t in terms) for i in {c.unit for c in pool}}
    pool = sorted(pool, key=lambda c: (-relevance[c.unit], c.unit, c.start))
    options, seen = [], set()
    for c in pool:
        if c.text.lower() not in seen:
            seen.add(c.text.lower())
            options.append(c)
    options = options[:MAX_OPTIONS]
    state, size = [], 0
    for i in sorted(relevance, key=lambda i: (-relevance[i], i)):
        if size + len(us[i]) > MAX_STATE_CHARS:
            break
        state.append(us[i])
        size += len(us[i])
    none = "none of these"
    criteria = {c.text: "" if c.type == "PARTY" else f'as written in: "{us[c.unit][:300]}"' for c in options}
    criteria[none] = "the text does not answer the question with any of these"
    out = llm.choice("\n\n".join(state), f"Which of these answers the question: {frame.lookup or frame.question}",
                     criteria)
    pick, probs = out["choice"], out["probabilities"]
    p = float(probs.get(pick, 0.0))
    top = dict(sorted(probs.items(), key=lambda kv: -kv[1])[:4])
    if pick == none or pick not in criteria:
        return SpanResult(False, reason="the LLM found no candidate that answers it", options=[c.text for c in options],
                          probabilities=top, llm_calls=1, pool=kind)
    if p < MIN_P:
        return SpanResult(False, reason=f"the LLM's pick is unsure ({p:.2f} < {MIN_P})",
                          options=[c.text for c in options], probabilities=top, llm_calls=1, pool=kind)
    unit = next(c.unit for c in options if c.text == pick)
    return SpanResult(True, pick, round(p, 3), "llm", reason="the LLM picked it among the text's candidates",
                      evidence=[{"text": us[unit], "label": pick, "p": round(p, 3)}],
                      options=[c.text for c in options], probabilities=top, llm_calls=1, pool=kind)
