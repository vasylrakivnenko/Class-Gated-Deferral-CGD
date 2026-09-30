"""
Pre-Tier 0: regex-only preparation before any model, in about a
millisecond. No parser, no NLI, no LLM.

It answers only what it is certain of. First,
"Is X discussed here?" is "yes" when one sentence contains every content
word of X (auditing/audit, books/book count as the same word). A missing word
proves nothing (the text may use a synonym), so then it defers. Every other
question goes on to Tier 0; on the way Pre-Tier 0
  - answers from clause frames (Pre-Tier 0 v2, router/frames.py) when one
    clause settles the question; on LegalBench held-out items it was 98.5%
    precise (264/268, 2026-09-30) at 2.2% coverage,
  - finds the document's parties (Tier 0 checks a deciding sentence is about
    the party the question asks about), and
  - in a contract naming its parties, reads the reader's "we" as the one
    party that does the question's action there ("Can we audit their
    books?" -> "Can the Licensee audit the Licensor's books?"), as a model
    would guess anyway, and
  - in a document that speaks as "we" to "you" (privacy policies, terms of
    service), turns the reader's question to the document's voice: the
    reader's "they" is the document's "we", their "my" its "your"
    ("Do they sell my data?" -> "Do we sell your data?").
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from router import frames

# Role words a contract capitalizes as defined parties. Two-word roles come
# first so "Receiving Party" wins over a shorter match.
ROLES = [
    "Receiving Party", "Disclosing Party", "Service Provider",
    "Licensee", "Licensor", "Sublicensee", "Tenant", "Landlord", "Lessee", "Lessor", "Buyer", "Seller",
    "Purchaser", "Vendor", "Supplier", "Customer", "Client", "Contractor", "Subcontractor", "Company",
    "Employer", "Employee", "Executive", "Consultant", "Provider", "Recipient", "Discloser", "Borrower",
    "Lender", "Franchisor", "Franchisee", "Distributor", "Manufacturer", "Reseller", "Guarantor", "Assignor",
    "Assignee", "Developer", "Publisher", "Sponsor", "Investor", "Shareholder", "Grantor", "Grantee",
    "Mortgagor", "Mortgagee", "Owner",
]
_ROLE_RE = re.compile(r"\b(" + "|".join(re.escape(r) for r in ROLES) + r")\b")
# ("Acme"), (the "Licensee"), (hereinafter referred to as "Beta") ...
_DEFINED = re.compile(r"""\(\s*(?:the\s+|hereinafter(?:\s+referred\s+to\s+as)?\s+|collectively,?\s+)?["“]([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,2})["”]\s*\)""")
_ENTITY_BEFORE = re.compile(r"\b(Inc|LLC|L\.L\.C|Ltd|Limited|Corp|Corporation|Co|L\.P|LP|LLP|plc|PLC|GmbH|AG|S\.A|N\.V|B\.V)\.?,?\s*$")
_ROLE_SUFFIX = re.compile(r"(ee|or|er|ant|Party)$")
_DOC_PRONOUNS = re.compile(r"\b(we|us|our|you|your)\b", re.I)

_Q_WORD = re.compile(r"[A-Za-z]+")
# "Is X discussed here?" asks whether the text covers X, not what it says about X.
# "covered" and "addressed" also have plain meanings ("Are the parking spaces
# covered?"), so they count only next to a word for the text itself.
_TEXT_REF = r"(?:here|herein|in (?:this|the) (?:clause|agreement|contract|document|text|section|lease|policy|passage))"
TOPIC_QUESTION = re.compile(
    r"\b(discuss(?:ed|es)?|mention(?:ed|s)?|referred to|refers? to|talks? about|say anything|says anything|"
    r"deals? with|dealt with)\b"
    rf"|\b(cover(?:ed|s)?|address(?:ed|es)?)\s+{_TEXT_REF}"
    r"|^\s*(is|are) there (a|an|any)\b.*\b(clause|provision|section|term|paragraph|language)s?\b",
    re.I,
)
_TOPIC_NEGATION = re.compile(r"\b(not|no|never|without)\b|n't\b", re.I)
# Words of a topic question that aren't the topic: function words, the topic
# verbs, and the words for the text itself.
_NOT_TOPIC = frozenset(
    "a an the and or of to in on for by with at from as is are was were be been being do does did has have had "
    "can could will would shall should may might must this that these those it its there here any anything "
    "something some about discuss discussed discusses mention mentioned mentions address addressed addresses "
    "cover covered covers refer refers referred talk talks say says deal deals dealt clause clauses provision "
    "provisions section sections term terms paragraph paragraphs language document text contract agreement "
    "lease policy passage excerpt".split()
)
PARTY_SCAN_CHARS = 20_000  # contracts name their parties up front; keeps long documents ~1 ms


@dataclass
class PreTier0Result:
    fired: bool  # answered with certainty
    answer: str | None = None  # "yes" when fired
    reason: str = ""
    parties: list = field(default_factory=list)
    evidence: list = field(default_factory=list)  # [{"text", "label", "p"}] when fired
    frames: dict | None = None  # Pre-Tier 0 v2 (router/frames.py): its full result
    rewritten: str | None = None  # the question in the document's own voice, for Tier 0
    ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def find_parties(document: str) -> list:
    """The document's parties in order of first mention: capitalized role
    words, plus quoted defined terms that are a role ("the "Consultant"") or
    name a company ("Acme Corp. ("Acme")")."""
    found = {}
    for m in _ROLE_RE.finditer(document):
        found.setdefault(m.group(1), m.start())
    for m in _DEFINED.finditer(document):
        name = m.group(1)
        is_role = _ROLE_SUFFIX.search(name) and name not in ("Other", "Order", "Number", "Letter", "Center", "Quarter")
        if is_role or _ENTITY_BEFORE.search(document[max(0, m.start() - 40):m.start()]):
            found.setdefault(name, m.start())
    return sorted(found, key=found.get)


# Reader's words -> the words of a document that speaks as "we" to "you".
_TO_DOC_VOICE = {"they": "we", "them": "us", "their": "our", "theirs": "ours", "themselves": "ourselves",
                 "i": "you", "me": "you", "my": "your", "mine": "yours", "myself": "yourself"}
_YOU_TO_WE = {"you": "we", "your": "our", "yours": "ours", "yourself": "ourselves"}


def to_document_voice(question: str) -> str | None:
    """The reader's question as the document would put it: they -> we,
    my -> your. "You" becomes "we" only next to I/my ("Do you sell my
    data?"); alone it may be the generic "can one". None if nothing changes."""
    words = {w.lower() for w in _Q_WORD.findall(question)}
    mapping = dict(_TO_DOC_VOICE)
    if words & {"i", "me", "my", "mine", "myself"}:
        mapping.update(_YOU_TO_WE)
    if not words & set(mapping):
        return None

    def sub(m):
        w = m.group(0)
        new = mapping.get(w.lower())
        if new is None:
            return w
        return new.capitalize() if m.start() == 0 or w[0].isupper() and w != "I" else new
    out = _Q_WORD.sub(sub, question)
    # "Am I allowed" -> "Are you allowed", "Was I" -> "Were you"
    out = re.sub(r"^(Am|Was)(\s+)(you)\b", lambda m: {"Am": "Are", "Was": "Were"}[m.group(1)] + m.group(2) + m.group(3), out)
    return out


FIRST_PERSON = {"we", "us", "our", "ours", "i", "me", "my", "mine"}
OTHER_SIDE = {"they", "them", "their", "theirs"}
_AGREE = {"do": "does", "are": "is", "have": "has", "were": "was", "am": "is"}


def _as_noun(party: str) -> str:
    return f"the {party}" if _ROLE_RE.fullmatch(party) or _ROLE_SUFFIX.search(party) else party


def _read_first_person(question: str, document: str, parties: list) -> tuple[str, str] | None:
    """In a contract naming its parties, "we" is read as the one party that
    does the question's action there, and "they" as the other of two:
    "Can we audit their books?" -> "Can the Licensee audit the Licensor's
    books?". None when no, or more than one, party does it."""
    words = {w.lower() for w in _Q_WORD.findall(question)}
    if not words & FIRST_PERSON or len(parties) < 2:
        return None
    acting = frames.acting_parties(question, document, parties)
    if not acting or len(acting) != 1:
        return None
    me = next(p for p in parties if p.lower() in acting)
    others = [p for p in parties if p != me]
    them = others[0] if len(others) == 1 else None
    if words & OTHER_SIDE and them is None:
        return None

    def sub(m):
        w, lw = m.group(0), m.group(0).lower()
        party = me if lw in FIRST_PERSON else them if lw in OTHER_SIDE else None
        if party is None:
            return w
        noun = _as_noun(party)
        if lw in ("our", "ours", "my", "mine", "their", "theirs"):
            noun += "'s"
        return noun[:1].upper() + noun[1:] if m.start() == 0 else noun
    out = _Q_WORD.sub(sub, question)
    if m := re.match(r"^(Do|Are|Have|Were|Am)(?=\s+(?:we|i)\b)", question, re.I):  # "Do we" -> "Does the Licensee"
        aux = _AGREE[m.group(1).lower()]
        out = (aux.capitalize() if m.group(1)[0].isupper() else aux) + out[len(m.group(1)):]
    return me, out


def check(question: str, document: str) -> PreTier0Result:
    start = time.perf_counter()
    result = _check(question, document)
    if not result.fired:
        fr = frames.answer(result.rewritten or question, document, result.parties)
        result.frames = fr.to_dict()
        if fr.answer:
            result.fired, result.answer, result.evidence = True, fr.answer, fr.evidence
            result.reason = "one clause settles it" + (' (the text permits it: "may", not "must")' if fr.qualifier == "may" else "")
    result.ms = round((time.perf_counter() - start) * 1000, 2)
    return result


def _stem(word: str) -> str:
    """auditing/audits/audited -> audit, books -> book, rights -> right."""
    w = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            w = w[: -len(suffix)]
            break
    return w[:6]


def topic_words(question: str) -> list:
    return [w for w in _Q_WORD.findall(question) if w.lower() not in _NOT_TOPIC and len(w) > 1]


def _topic(question: str, document: str) -> PreTier0Result | None:
    """"yes" if one sentence holds every topic word; None to defer."""
    if not TOPIC_QUESTION.search(question) or _TOPIC_NEGATION.search(question):
        return None
    words = topic_words(question)
    if not words:
        return None
    wanted = {_stem(w) for w in words}
    # Every stem must occur somewhere (a few fast scans); then only sentences
    # around the rarest stem are split into words.
    counts = {}
    for stem in wanted:
        counts[stem] = len(re.findall(rf"\b{stem}", document, re.I))
        if not counts[stem]:
            return None
    rarest = min(counts, key=counts.get)
    sentences = frames.Sentences(document)
    for m in re.finditer(rf"\b{rarest}", document, re.I):
        _, sentence = sentences.at(m.start())
        if wanted <= {_stem(w) for w in _Q_WORD.findall(sentence)}:
            return PreTier0Result(True, "yes", reason=f"one sentence has every topic word ({', '.join(words)})",
                                  evidence=[{"text": sentence, "label": "yes", "p": 1.0}])
    return None


def _check(question: str, document: str) -> PreTier0Result:
    if (topic := _topic(question, document)) is not None:
        return topic
    parties = find_parties(document[:PARTY_SCAN_CHARS])
    if not _DOC_PRONOUNS.search(document):
        reading = _read_first_person(question, document, parties)
        if reading:
            party, rewritten = reading
            return PreTier0Result(False, reason=f'read "we" as {_as_noun(party)}, the only party that does this here',
                                  parties=parties, rewritten=rewritten)
        return PreTier0Result(False, reason=f"{len(parties)} part{'y' if len(parties) == 1 else 'ies'} found", parties=parties)
    flipped = to_document_voice(question)
    if flipped is None:
        return PreTier0Result(False, reason="the document speaks as we/you; the question needs no rewording", parties=parties)
    return PreTier0Result(False, reason=f'the document speaks as we/you; Tier 0 reads the question as "{flipped}"',
                          parties=parties, rewritten=flipped)
