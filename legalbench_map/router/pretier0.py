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
When the frames defer, a few hand-checked legal formulas may still settle it
(router/equivalences.py: "does not include information independently
developed" answers "Can the receiving party independently develop similar
information?"), "yes" only.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field

from router import equivalences, frames

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
# "Can the Employee discuss the Agreement with ...?" asks what someone may say, not what the text covers: only the text
# itself (or a passive: "Is X discussed?") makes a topic question.
_SOMEONE_DISCUSSES = re.compile(r"^(?:can|may|must|should|could|will|would|shall|do|does|did|is|are|am)\s+(?!(?:the |this |"
                                r"that |your |our )?(?:agreement|contract|clause|document|text|section|provision|policy|"
                                r"lease|terms|nda|passage|excerpt|paragraph|it|there)\b)(?:[\w'-]+\s+){1,4}?(?:discuss|"
                                r"mention|refer|talk|deal|say|tell)\b", re.I)
# "Is there a non-compete clause restricting the Executive?" asks what the clause does to someone, not only whether
# one is there (2026-10-01: answered from "she is not subject to any non-competition ... restrictions").
# "Is there an accelerated vesting provision ...?" asks whether the contract provides it: a sentence denying it ("are
# not subject to accelerated vesting") doesn't make it a "yes" (2026-10-01, v6). "Is X discussed?" still asks only
# whether the text covers X.
_ASKS_PROVISION = re.compile(r"^\s*(?:is|are) there\b", re.I)
_RELATIONAL_TOPIC = re.compile(r"\b(?:clause|provision|section|term|paragraph|language)s?\s+(?:that\s+\w+|which\s+\w+|"
                               r"\w+ing)\s+(?:the|a|an|any|its|their|his|her|my|our|your|either|each)\b", re.I)
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


# Routing (2026-10-02, STATUS.md "passive and noun shapes"): on unseen user-style questions, Pre-Tier 0's answers
# from a catch-all presence frame (every word of the question in one sentence, its structure unchecked) or from an
# action frame on a question qshapes had to reword (passives, nouns, conditions first ...) were right ~88-92%, and
# on those same questions the reader network was right on every one it answered. So Pre-Tier 0 abstains there and
# the network (then Jev) answers. Measured on v7: rules + network 95.9% -> 98.3%, answered 24.0% -> 21.6%.
ROUTE_RISKY = os.environ.get("PRETIER0_ROUTE_RISKY", "1") == "1"


def _left_to_the_network(fr: dict, asked: str) -> bool:
    if not ROUTE_RISKY:
        return False
    frame = fr.get("frame") or {}
    if frame.get("asks") == "EXISTS":
        return bool(frame.get("strict"))
    if frame.get("asks") == "PROPERTY":
        return False
    return frames.qshapes.canonical(asked) != asked


def check(question: str, document: str) -> PreTier0Result:
    start = time.perf_counter()
    result = _check(question, document)
    if not result.fired:
        fr = frames.answer(result.rewritten or question, document, result.parties)
        result.frames = fr.to_dict()
        if fr.answer and _left_to_the_network(fr.to_dict() if hasattr(fr, "to_dict") else {}, result.rewritten or question):
            result.reason = ("left to the reader network: Pre-Tier 0's answers from this kind of match (a reworded "
                             "question, or the words of a question without its structure) were right ~90% on unseen "
                             "questions, the network's 99-100%")
        elif fr.answer:
            result.fired, result.answer, result.evidence = True, fr.answer, fr.evidence
            qualifier = fr.qualifier or ""
            result.reason = "one clause settles it" + (' (the text permits it: "may", not "must")' if qualifier.startswith("may") else "")
            if frames.CONDITIONAL in qualifier:
                result.reason += f', {frames.CONDITIONAL}' + (f': "{fr.condition}"' if fr.condition else "")
        elif (eq := equivalences.answer(result.rewritten or question, document)) is not None:
            # a fixed legal formula the frames don't read ("does not include information independently developed")
            result.fired, result.answer, result.evidence = True, eq["answer"], eq["evidence"]
            result.reason = f"a legal formula settles it: {eq['reason']} ({eq['rule']}, router/equivalences.py)"
    result.ms = round((time.perf_counter() - start) * 1000, 2)
    return result


# How a topic question's words are found (since 2026-10-01; the 6-letter prefix before it read "employer" as
# "employee" and "assignor" as "assign", and found "change of control" in "change the control panel"):
#   - a word matches its inflections (frames.normalize) and derivations (terminate / termination, assign /
#     assignment), never a party role made from it (employer / employee, assignor / assign);
#   - a term of art (_TERMS) must be there as the term;
#   - otherwise the words only have to share the sentence ("auditing of records" in "audit the books and records").
# Requiring the words a question writes together to stay together ("audit rights" as "right to audit") was tried
# and dropped: on CUAD dev it lost more right answers than it saved ("the laws of New York shall govern").
# On CUAD dev topic questions (extensive/topic/eval_topic.py) it answers 20.8%, up from 18.4%, with 96.8% of answers
# agreeing with CUAD's labels (was 96.4%); nearly all the rest mention the topic under another CUAD category.
# Noun and adverb endings a word's family shares a root across (longest tried first). Not -ee/-or/-er/-ant:
# those make the parties ("assignee", "licensor", "employer"), which aren't the act.
# frames.normalize drops a final "e" from words outside the lexicon, so "-able" can arrive as "-abl".
_DERIVATIONS = sorted("ification ication ability ibility ation ition ment ance ence able ible anc enc abl ibl ity ion ure "
                      "ify ate at al ly".split(), key=len, reverse=True)
_FAMILY = {"payment": "pay"}  # too short for the suffix rule
_NOT_FAMILY = frozenset({"government"})  # looks like govern + ment, isn't governing
_TERMS = [re.compile(p, re.I) for p in (
    r"\bchanges? (?:of|in) (?:the )?(?:\w+ )?control\b", r"\bright of first refusal\b", r"\bright of first offer\b",
    r"\bright of first negotiation\b", r"\bmost[- ]favou?red[- ](?:nation|customer)s?\b", r"\bcovenants? not to sue\b",
    r"\bforce majeure\b", r"\bliquidated damages\b", r"\bthird[- ]part(?:y|ies)[- ](?:\w+ )?beneficiar\w*")]
_TOPIC_WORD = re.compile(r"[a-z0-9]+")


def _key(word: str) -> str:
    if word.endswith("ies") and len(word) > 5:  # warranties -> warranty (the stemmer would leave "warranti")
        word = word[:-3] + "y"
    n = frames.normalize(word)
    return _FAMILY.get(n, n)


def _root(key: str) -> str:
    """terminate / termination -> termin, assign / assignment / assignable -> assign, confidential /
    confidentiality -> confidenti, committed / commitment -> commit; "assignee" stays "assignee".
    Up to two endings come off, and none that would leave under 5 letters."""
    if key in _NOT_FAMILY:
        return key
    for _ in range(2):
        cut = next((sfx for sfx in _DERIVATIONS if key.endswith(sfx) and len(key) - len(sfx) >= 5), None)
        if cut is None:
            break
        key = key[:-len(cut)]
    if key[-1:] in ("e", "y") and not key.endswith("ee") and len(key) > 5:
        key = key[:-1]
    if len(key) > 5 and key[-1] == key[-2] and key[-1] not in "aeiousl":  # committ -> commit, transferr -> transfer
        key = key[:-1]
    return key


def _same(a: str, b: str) -> bool:
    """Whether two keys are one word's family: equal, or the same root of 5 letters or more."""
    return a == b or (len(ra := _root(a)) >= 5 and ra == _root(b))


def _words(text: str) -> list:
    """Lowercased words, "'s" dropped; hyphenated words split ("non-compete" -> non, compete)."""
    return _TOPIC_WORD.findall(re.sub(r"'s\b", "", text.lower().replace("’", "'")))


def topic_words(question: str) -> tuple:
    """(terms of art, topic words as keys)."""
    low = question.lower()
    terms = [t for t in _TERMS if t.search(low)]
    for t in terms:
        low = t.sub(" ", low)  # its words are the term's
    words = [_key(w) for w in _words(low) if w not in _NOT_TOPIC and len(w) > 1]
    return terms, words


def _discusses(sentence: str, terms: list, words: list) -> bool:
    if not all(t.search(sentence) for t in terms):
        return False
    keys = [_key(w) for w in _words(sentence)]
    return all(any(_same(k, w) for k in keys) for w in words)


def _finder(word: str) -> re.Pattern:
    """A fast scan for the word's sentences: its stem as a prefix, and its irregular forms (paid -> pay)."""
    base = _root(word) if len(_root(word)) >= 5 else word
    base = base[:-1] if base[-1] in "ey" and len(base) > 4 else base  # "share" also finds "sharing"
    forms = [re.escape(base)] + [re.escape(f) for f, v in frames.IRREGULAR.items() if v == word and f != base]
    forms += [re.escape(f) for f, v in _FAMILY.items() if v == word]
    return re.compile(r"\b(?:" + "|".join(forms) + ")", re.I)


def _topic(question: str, document: str) -> PreTier0Result | None:
    """"yes" if one sentence holds every topic word (as above); None to defer."""
    if not TOPIC_QUESTION.search(question) or _TOPIC_NEGATION.search(question) or _RELATIONAL_TOPIC.search(question) \
            or _SOMEONE_DISCUSSES.search(question):
        return None
    terms, words = topic_words(question)
    if not terms and not words:
        return None
    # Every word and term must occur somewhere (a few fast scans); then only the
    # sentences around the rarest one are split into words.
    finders = terms + [_finder(w) for w in words]
    counts = []
    for f in finders:
        counts.append(len(f.findall(document)))
        if not counts[-1]:
            return None
    rarest = finders[counts.index(min(counts))]
    sentences, _ = frames.indexed(document)
    seen = set()
    for m in rarest.finditer(document):
        start, sentence = sentences.at(m.start())
        if start in seen:
            continue
        seen.add(start)
        if _discusses(sentence, terms, words) and not (_ASKS_PROVISION.search(question) and frames._negated_sentence(
                frames._raw_tokens(sentence), frames.tag(frames.tokens(sentence), frames._BASE_TRIE))):
            named = [t.search(sentence).group(0) for t in terms] + words
            return PreTier0Result(True, "yes", reason=f"one sentence has every topic word ({', '.join(named)})",
                                  evidence=[{"text": sentence, "label": "yes", "p": 1.0}])
    return None


def _check(question: str, document: str) -> PreTier0Result:
    if (topic := _topic(frames.qshapes.canonical(question), document)) is not None:  # "Is the clause about X?" too
        return topic
    parties = find_parties(document[:PARTY_SCAN_CHARS])
    if not frames.occurs_any(("we", "us", "our", "you", "your"), document):
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
