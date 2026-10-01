"""
Pre-Tier 0 v2: answer a yes/no question by matching concept frames, with no
model. Runs in shadow mode for now (router/pretier0.py records its answer;
it doesn't decide).

    1. Normalize words: lowercase, then a lemma table built from the lexicon
       at import (shares/sharing/shared -> share, paid -> pay); words outside
       the lexicon get a crude suffix stem on both sides.
    2. Tag concepts with a phrase trie, longest match first:
       "information about you" -> USER_DATA, "advertising partners" ->
       ADVERTISERS, "may" -> MAY, "shall not" -> NOT. Boilerplate such as
       "including without limitation" is tagged NEUTRAL and ignored.
    3. The question becomes a frame, one of two kinds:
       - action: who (actor), does what (action; "or" gives alternatives),
         to what (things), and what it asks (does / can / must / is it
         prohibited). "A party" is any party. Words outside the lexicon
         become literal slots that must appear as they are.
       - property: a thing and an attribute ("Is the license
         non-transferable?", "Is a party's liability capped?").
    4. A sentence clause answers only if it has the whole frame and no
       blocker (unless, except, subject to, if, without...). For an action
       the actor must come before the action; its modality decides:
       NOT -> "no" ("yes" to "is it prohibited");  MAY -> "yes" to "can",
       "yes, may" to "does", defer to "must";  MUST/DOES -> "yes". A
       property is only ever answered "yes", and not if negated.
    Anything missing, blocked, or two sentences disagreeing -> defer.
Since 2026-10-01 (STATUS.md, "Pre-Tier 0 next level"): questions are first put in the frame's shape
(router/qshapes.py: "Does the agreement require X to Y?" -> "Must X Y?", a leading condition moves to the end,
passives); the question's named subject must be the one acting; a verb the lexicon lacks is matched as written
(OPEN); conditions must be of the same kind ("after" is not "during" or "except"); presence frames keep the
question's relations (_presence_guards); the document's own names for its confidential information count as it.
"""
from __future__ import annotations

import bisect
import functools
import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import ahocorasick

from router import qshapes

# concept -> phrases. Kinds: ACTOR, MODAL, ACTION, THING, PROP, BLOCK, NEUTRAL.
LEXICON = {
    # actors ("they" in a question has already become "we" for a we/you document)
    ("ACTOR", "WE"): ["we", "the company"],
    ("ACTOR", "USER"): ["you", "the user", "users"],
    ("ACTOR", "ALL"): ["either party", "each party", "both parties", "the parties", "any party", "each of the parties",
                       "either of the parties", "the parties hereto", "each party hereto", "either party hereto"],
    ("ACTOR", "NONE"): ["neither party", "no party", "neither of the parties"],  # also means NOT
    ("ACTOR", "ANY"): ["a party", "one party", "one of the parties"],  # in a question: whichever party
    ("ACTOR", "IT"): ["it", "such party", "each such party"],  # the actor named before it in the sentence
    ("ACTOR", "OTHER"): ["the other party", "other party", "the other parties", "the non-breaching party"],
    ("ACTOR", "RECEIVER"): ["receiving party", "the receiving party", "recipient", "the recipient", "receiving parties",
                            "recipient party", "the recipient party", "recipient parties"],
    ("ACTOR", "DISCLOSER"): ["disclosing party", "the disclosing party", "discloser", "the discloser"],
    # modality; precedence NOT > MAY > MUST > DOES
    ("MODAL", "NOT"): ["not", "never", "no right", "cannot", "can't", "won't", "don't", "doesn't", "shan't",
                       "may in no event", "in no event", "exempt", "excused", "relieved", "released from", "waive",
                       "fail", "under no circumstances", "nor", "disclaim"],
    ("MODAL", "NEITHER"): ["neither"],  # "Neither X nor Y will grant": negates the actor that follows
    ("MODAL", "BAN"): ["prohibited", "forbidden", "restricted from", "barred", "precluded", "restrained from",
                       "not permitted", "not allowed", "not be permitted", "refrain from"],
    ("MODAL", "MAY"): ["may", "can", "could", "permitted", "allowed", "entitled", "free to", "right",
                       "reserves the right", "reserve the right", "option", "discretion"],
    ("MODAL", "MUST"): ["must", "shall", "required", "obligated", "obliged", "agree", "undertake", "have to",
                        "has to", "responsible for", "covenant"],
    ("MODAL", "DOES"): ["will", "do", "does", "did", "is", "are"],
    # actions
    ("ACTION", "SHARE"): ["share", "disclose", "provide", "transfer", "make available", "give access", "pass on",
                          "release", "transmit", "distribute", "divulge", "reveal", "communicate"],
    ("ACTION", "SELL"): ["sell", "rent out", "trade"],
    ("ACTION", "COLLECT"): ["collect", "gather", "obtain"],
    ("ACTION", "USE"): ["use", "process", "utilize", "exploit"],
    ("ACTION", "STORE"): ["store", "retain", "keep", "hold"],
    ("ACTION", "DELETE"): ["delete", "erase", "remove"],
    ("ACTION", "RETURN_DESTROY"): ["return", "destroy", "return or destroy", "destroy or return"],
    ("ACTION", "COPY"): ["copy", "make copies", "make a copy", "reproduce", "duplicate", "create copies", "create a copy",
                         "produce copies", "make any copies"],
    ("ACTION", "DEVELOP_INDEPENDENTLY"): ["independently develop", "develop independently"],
    ("ACTION", "TRACK"): ["track", "monitor"],
    ("ACTION", "AUDIT"): ["audit", "inspect", "examine"],
    ("ACTION", "ASSIGN"): ["assign"],
    ("ACTION", "SUBLET"): ["sublet", "sublease"],
    ("ACTION", "TERMINATE"): ["terminate", "cancel"],
    ("ACTION", "RENEW"): ["renew", "extend"],
    ("ACTION", "PAY"): ["pay", "payable"],
    ("ACTION", "REPAIR"): ["repair", "maintain", "fix"],
    ("ACTION", "INSURE"): ["insure", "carry insurance", "maintain insurance", "procure insurance",
                           "obtain insurance", "purchase insurance", "keep insurance"],
    ("ACTION", "CARRY"): ["carry", "procure"],
    ("ACTION", "SOLICIT"): ["solicit", "induce", "entice"],
    ("ACTION", "HIRE"): ["hire", "employ", "engage", "recruit"],
    ("ACTION", "COMPETE"): ["compete"],
    ("ACTION", "DISPARAGE"): ["disparage", "make disparaging statements", "defame", "make any disparaging"],
    ("ACTION", "CHALLENGE"): ["challenge", "contest", "dispute", "dispute the validity", "deny the validity", "oppose",
                              "attack the validity", "attack"],
    ("ACTION", "PURCHASE"): ["purchase", "buy", "order"],
    ("ACTION", "GRANT"): ["grant"],
    ("ACTION", "INDEMNIFY"): ["indemnify", "hold harmless", "defend and indemnify"],
    ("ACTION", "OPT_OUT"): ["opt out", "unsubscribe"],
    ("ACTION", "NOTIFY"): ["notify", "inform", "give notice", "give written notice", "give prompt notice",
                           "give prompt written notice", "provide notice", "provide written notice",
                           "provide prompt notice", "provide prompt written notice", "send notice", "give notification",
                           "advise"],
    # things
    ("THING", "USER_DATA"): ["your data", "your information", "your personal data", "your personal information",
                             "information about you", "data about you", "personal data", "personal information",
                             "data", "information", "personally identifiable information"],
    ("THING", "CONFIDENTIAL_INFO"): ["confidential information", "proprietary information", "trade secrets",
                                     "evaluation material", "evaluation materials", "confidential material"],
    ("THING", "THIRD_PARTIES"): ["third parties", "third party", "others", "outside companies", "any person"],
    ("THING", "ADVERTISERS"): ["advertisers", "advertising partners", "ad networks", "advertising networks",
                               "marketing partners"],
    ("THING", "ANALYTICS"): ["analytics providers", "analytics partners"],
    ("THING", "SERVICE_PROVIDERS"): ["service providers", "vendors", "processors"],
    ("THING", "ADVISORS"): ["advisors", "advisers", "consultants", "attorneys", "accountants", "legal counsel",
                            "professional advisors", "professional advisers", "auditors"],
    ("THING", "EMPLOYEES"): ["employees", "employee", "staff", "personnel", "officers and employees"],
    ("THING", "CUSTOMERS"): ["customers", "customer", "clients", "client"],
    ("THING", "AFFILIATES"): ["affiliates", "affiliate", "subsidiaries"],
    ("THING", "BOOKS"): ["books", "records", "books and records", "accounts"],
    ("THING", "PREMISES"): ["premises", "property", "apartment", "unit"],
    ("THING", "AGREEMENT"): ["agreement", "contract", "lease"],
    ("THING", "CONSENT"): ["consent", "approval"],
    ("THING", "LICENSE"): ["license", "licence", "licenses", "license grant", "license granted", "sublicense"],
    ("THING", "LIABILITY"): ["liability", "aggregate liability", "total liability", "cumulative liability"],
    ("THING", "IP"): ["intellectual property", "intellectual property rights", "ip rights", "patents",
                      "trademarks", "copyrights"],
    ("THING", "INSURANCE"): ["insurance", "insurance coverage", "insurance policy", "insurance policies"],
    ("THING", "REVENUE"): ["revenue", "revenues", "profits", "profit", "net sales", "gross revenue",
                           "net revenue", "revenue or profits"],
    ("THING", "WITHOUT_CAUSE"): ["without cause", "for convenience", "for any reason", "for any or no reason",
                                 "for no reason", "at will", "with or without cause", "at any time"],
    ("THING", "NOTICE"): ["notice", "written notice", "prompt notice", "prompt written notice", "notification",
                          "immediate notice", "advance notice", "prior notice", "prior written notice", "notice thereof"],
    ("THING", "RIGHTS"): ["rights", "right"],  # only an action's object (frames._action_frame); "right" alone tags as MAY
    ("THING", "LAW_REQUIRED"): ["required by law", "legally required", "required by applicable law", "required by any law",
                                "law requires", "the law requires", "legal requirement", "legal requirements",
                                "compelled by law", "legally compelled", "obligated by law", "required by statute",
                                "by operation of law", "pursuant to law", "applicable law", "the law", "law", "laws",
                                "legal process", "court order", "subpoena", "judicial process", "regulatory authority"],
    ("THING", "ORAL"): ["orally", "oral", "verbally", "verbal", "spoken", "in oral form", "by word of mouth"],
    ("THING", "EXISTENCE"): ["existence of this agreement", "existence of the agreement", "fact that",
                             "that the agreement exists", "that this agreement exists", "the agreement exists",
                             "terms of this agreement", "existence"],
    # properties (property frames: "Is the license non-transferable?")
    ("PROP", "NON_TRANSFERABLE"): ["non transferable", "nontransferable", "not transferable", "non assignable",
                                   "nonassignable", "not assignable"],
    ("PROP", "NON_EXCLUSIVE"): ["non exclusive", "nonexclusive"],
    ("PROP", "EXCLUSIVE"): ["exclusive", "exclusively"],
    ("PROP", "IRREVOCABLE"): ["irrevocable", "irrevocably"],
    ("PROP", "PERPETUAL"): ["perpetual", "perpetually", "in perpetuity"],
    ("PROP", "ROYALTY_FREE"): ["royalty free", "royalty-free", "fully paid up", "fully paid"],
    ("PROP", "WORLDWIDE"): ["worldwide", "world wide", "throughout the world"],
    ("PROP", "SUBLICENSABLE"): ["sublicensable", "sublicenseable", "with the right to sublicense"],
    ("PROP", "UNLIMITED"): ["unlimited", "uncapped", "without limit", "no limit"],
    ("PROP", "CAPPED"): ["capped", "not exceed", "exceed", "limited to", "shall not exceed", "cap", "not to exceed",
                         "maximum aggregate", "in excess of"],
    ("PROP", "JOINTLY_OWNED"): ["jointly owned", "joint ownership", "co owned", "owned jointly", "jointly own",
                                "joint owners", "co owners", "co ownership"],
    # blockers: the answer depends on something else
    ("BLOCK", "CONDITION"): ["unless", "except", "excepting", "provided that", "provided however", "subject to",
                             "only if", "if", "without", "notwithstanding", "conditioned on", "conditioned upon",
                             "in the event", "only", "during", "within", "sole discretion", "save", "other than",
                             "as soon as",
                             "solely", "until", "so long as", "as long as", "to the extent", "where", "except as",
                             "when", "whenever", "at the request", "upon request", "on request", "if requested",
                             "on the occurrence", "in case", "after", "before", "prior to", "following"],
    ("BLOCK", "UPON"): ["upon"],  # a condition, unless it only sets notice ("upon 30 days' written notice")
    # boilerplate that looks like a blocker or a negation but isn't
    ("NEUTRAL", "BOILERPLATE"): ["including without limitation", "without limitation", "without limiting",
                                 "including but not limited to", "but not limited to", "not limited to",
                                 "including, without limitation", "at its own expense", "at its sole cost",
                                 "from time to time", "in accordance with", "without prejudice",
                                 "not less than", "no less than", "not more than", "no more than", "not later than",
                                 "no later than", "not earlier than", "no earlier than", "not fewer than",
                                 # the whole agreement, not a specific condition ("subject to Section 9" still blocks)
                                 "subject to the terms and conditions of this agreement",
                                 "subject to the terms and conditions hereof", "subject to the terms of this agreement",
                                 "subject to the terms and conditions set forth herein",
                                 "subject to the terms and conditions set forth in this agreement",
                                 "subject to the provisions of this agreement", "subject to this agreement",
                                 "in accordance with the terms of this agreement", "under this agreement"],
}
EXTRA_LEXICON = Path(__file__).with_name("lexicon_extra.json")


_EXTRA_PHRASES: set = set()


def _merge_extra(lexicon: dict, path: Path) -> None:
    """Add the LLM-proposed, LLM-verified phrases (lexicon_extra.json) after
    the hand-curated ones. A phrase another concept already has is skipped."""
    if not path.exists():
        return
    taken = {p for phrases in lexicon.values() for p in phrases}
    for key, entry in json.loads(path.read_text())["concepts"].items():
        kind, name = key.split(":", 1)
        phrases = lexicon.setdefault((kind, name), [])
        for p in entry["phrases"]:
            if p not in taken:
                phrases.append(p)
                taken.add(p)
                _EXTRA_PHRASES.add(p)


_merge_extra(LEXICON, EXTRA_LEXICON)
# A verb with an object stands for a composite action: "maintain ... insurance" is INSURE.
COMPOSITE = {"INSURE": ({"REPAIR", "CARRY", "COLLECT", "PURCHASE", "STORE", "SHARE"}, "INSURANCE"),
             "NOTIFY": ({"SHARE"}, "NOTICE")}  # "shall provide the Discloser with prompt written notice"
# A narrower thing answers for a broader one: sharing with advertisers is sharing with third parties.
BROADER = {"ADVERTISERS": "THIRD_PARTIES", "ANALYTICS": "THIRD_PARTIES", "SERVICE_PROVIDERS": "THIRD_PARTIES"}
IRREGULAR = {"paid": "pay", "sold": "sell", "kept": "keep", "held": "hold", "gave": "give", "given": "give",
             "made": "make", "sent": "send", "told": "tell", "shown": "show", "done": "do", "had": "have",
             "has": "have", "is": "is", "are": "are", "was": "is", "were": "are", "can't": "can't",
             "won't": "won't", "don't": "don't", "doesn't": "doesn't", "bought": "buy", "hired": "hire",
             "employed": "employ", "licensed": "license", "granted": "grant"}
QUESTION_AUX = {"do": "DOES", "does": "DOES", "did": "DOES", "will": "DOES", "is": "DOES", "are": "DOES", "am": "DOES",
                "can": "CAN", "may": "CAN", "could": "CAN", "must": "MUST", "shall": "MUST", "should": "MUST"}
STOPWORDS = frozenset("a an the of to in on for by with at from as and or 's this that these those its our their "
                      "your my any all be been being it upon per such no more than hereby herein hereunder "
                      "thereof hereof its own also further additionally "
                      # fillers people type ("just by using the service", "can they actually ...")
                      "just simply merely actually really basically".split())
# Actor names a model can't know: a capitalized name right before a modal ("SpringCo shall").
_NAME_BEFORE_MODAL = re.compile(
    r"\b((?:[A-Z][\w&.-]*\s+){0,2}[A-Z][\w&.-]*)\s*(?:\([^)]{0,60}\)\s*)?,?\s+"
    r"(?:shall|will|may|must|agrees|hereby|can|cannot|covenants|undertakes|represents)\b")
_NOT_NAMES = frozenset("this the such each either neither any all no in if notwithstanding upon except unless subject "
                       "nothing where when during after before prior following section article schedule exhibit "
                       "agreement information term products product services service software license price fees "
                       "fee payment payments notice date territory confidential parties party it he she they "
                       "we you i there which who that these those its".split())
_PASSIVE_AUX = {"be", "been", "being", "is", "are", "was", "were"}
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|;")
_CLAUSE_BREAK = {";", "but", "however", "whereas"}


def _forms(base: str) -> set:
    forms = {base, base + "s", base + "es", base + "ed", base + "ing"}
    if base.endswith("e"):
        forms |= {base + "d", base[:-1] + "ing"}
    if base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
        forms |= {base[:-1] + "ies", base[:-1] + "ied"}
    if re.search(r"[^aeiou][aeiou][bdgklmnprt]$", base):
        forms |= {base + base[-1] + "ed", base + base[-1] + "ing"}
    return forms


def _crude(word: str) -> str:
    """expire/expires/expired/expiring -> expir; words outside the lexicon only."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and not word.endswith("ee") and len(word) > 4:  # licensee stays apart from license
        word = word[:-1]
    return word


_LEMMA = {}
for _phrases in LEXICON.values():
    for _phrase in _phrases:
        if _phrase in _EXTRA_PHRASES:
            continue
        for _word in _phrase.split():
            for _form in _forms(_word):
                # "disclosing" is a form of "disclose" and of itself ("disclosing party"): the shortest base wins
                if _form not in _LEMMA or len(_word) < len(_LEMMA[_form]):
                    _LEMMA[_form] = _word
# Added phrases extend the table but never remap a curated form ("account books"
# must not pull "accounts" onto "account"); their inflected words ("never
# expires") are left to the stemmer, which treats both sides alike.
for _phrase in _EXTRA_PHRASES:
    for _word in _phrase.split():
        if _crude(_word) != _word or _word in _LEMMA:
            continue
        for _form in _forms(_word):
            _LEMMA.setdefault(_form, _word)
_LEMMA.update(IRREGULAR)


def normalize(token: str) -> str:
    return _LEMMA.get(token) or _crude(token)


def tokens(text: str) -> list:
    text = re.sub(r"'s\b", "", text.lower().replace("’", "'"))  # "Licensor's books" -> licensor books
    return [normalize(t) for t in _TOKEN.findall(text)]


def _trie_of(items) -> dict:
    trie = {}
    for concept, phrases in items:
        for phrase in phrases:
            node = trie
            for word in tokens(phrase):
                node = node.setdefault(word, {})
            node.setdefault("$", concept)
    return trie


_BASE_TRIE = _trie_of(LEXICON.items())


def _build_trie(extra: dict):
    """The lexicon's trie, plus a small one for this document's parties: the
    big one is built once; tag() takes the longer match of the two (the
    lexicon's on a tie), as one merged trie would."""
    if not extra:
        return _BASE_TRIE
    return (_BASE_TRIE, _trie_of(extra.items()))


@dataclass
class Tag:
    kind: str  # ACTOR | MODAL | ACTION | THING | PROP | BLOCK | WORD | BREAK
    name: str
    start: int  # token index
    end: int = -1  # token index after the tag (start + 1 when not set)


def tag(toks: list, trie: dict) -> list:
    """Longest-match concept tags; uncovered content words become WORD tags;
    NEUTRAL boilerplate is dropped."""
    out, i = [], 0
    tries = trie if isinstance(trie, tuple) else (trie,)
    while i < len(toks):
        best = None
        for t in tries:
            node, j = t, i
            while j < len(toks) and toks[j] in node:
                node = node[toks[j]]
                j += 1
                if "$" in node and (best is None or j > best[1]):
                    best = (node["$"], j)
        if best:
            (kind, name), end = best
            if kind != "NEUTRAL":
                out.append(Tag(kind, name, i, end))
            i = end
            continue
        t = toks[i]
        if t in _CLAUSE_BREAK:
            out.append(Tag("BREAK", t, i))
        elif t not in STOPWORDS:
            out.append(Tag("WORD", t, i))
        i += 1
    return out


@dataclass
class Frame:
    asks: str  # DOES | CAN | MUST | PROHIBITED | PROPERTY
    actor: str | None  # None for a property frame
    action: str | None  # the first of `actions`
    things: list  # (kind, name) of THING, WORD and ACTOR tags
    actions: list = field(default_factory=list)  # alternatives joined by "or"
    props: list = field(default_factory=list)  # property frame: alternatives joined by "or"
    conds: list = field(default_factory=list)  # the question's condition ("if ... change of control"): stems the clause's must hold
    only: bool = False  # the question says "only"/"solely": the clause must too
    cues: list = field(default_factory=list)  # the question's condition words ("after", "without"), by kind (_CUE_KIND)
    cue_words: list = field(default_factory=list)  # the same words as written
    alts: list = field(default_factory=list)  # presence frame: alternatives, each a list of stem groups
    negative: bool = False  # presence frame: the question asks for a negation ("that no license is granted")
    require: bool = False  # presence frame: "does X require consent" ("not ... without consent" counts)
    when: bool = False  # presence frame: "specify when ...": the sentence must hold a date or duration
    strict: bool = False  # presence frame from a catch-all: any condition in the sentence defers
    dated: bool = False  # presence frame asking for a date or period ("the date on which it becomes effective")
    guards: dict = field(default_factory=dict)  # presence frame: what the question's relations need (_presence_guards)
    values: list = field(default_factory=list)  # numbers, amounts, months the question names: the sentence must have them all
    compare: str = ""  # "up" / "down" when the question puts a value on one side ("more than $175,000,000")
    any_things: list = field(default_factory=list)  # things the question asks about as "any X": the text must say any/all


@dataclass
class FrameResult:
    answer: str | None = None  # "yes" | "no" | None (defer)
    qualifier: str | None = None  # "may" when the text only permits it; CONDITIONAL in it when the "yes" has one
    reason: str = ""
    frame: dict | None = None
    evidence: list = field(default_factory=list)
    ms: float = 0.0
    condition: str = ""  # the condition a "yes, with a condition" quotes ("unless the Licensor consents in writing")

    def to_dict(self) -> dict:
        return asdict(self)


def _party_lexicon(parties) -> dict:
    return {("ACTOR", p.lower()): [p.lower()] for p in parties}


def named_actors(document: str, limit: int = 20_000) -> list:
    """Capitalized names acting in the document ("SpringCo shall ...") that
    aren't lexicon concepts, from its first `limit` characters."""
    names = {}
    for m in _NAME_BEFORE_MODAL.finditer(document[:limit]):
        words = m.group(1).split()
        while words and words[0].lower() in ("the", "and", "or"):
            words = words[1:]
        if not words or any(w.lower().strip(".,") in _NOT_NAMES for w in words):
            continue
        name = " ".join(words)
        toks = tokens(name)
        node = _BASE_TRIE
        for t in toks:
            node = node.get(t, {}) if isinstance(node, dict) else {}
        if "$" in node:  # already a concept ("Recipient", "Licensee")
            continue
        names.setdefault(name, m.start())
    return sorted(names, key=names.get)


def _alternatives(tags: list, first: Tag, kind: str) -> list:
    """`first` and the tags of `kind` joined to it by "or" (the WORD "or" is a
    stopword, so adjacency of same-kind tags with only "or" between counts)."""
    out = [first]
    for t in tags:
        if t.kind == kind and t.start > out[-1].start and t is not first:
            out.append(t)
    return out


def question_frame(question: str, trie: dict) -> Frame | str:
    """The question's frame, or a reason it has none: an action or property
    frame, else a presence frame ("Does the agreement specify ...")."""
    if qshapes.two_questions(question):
        return "two questions in one (\"..., or is it prohibited?\")"
    question = qshapes.canonical(question)
    frame = _action_frame(question, trie)
    if isinstance(frame, str):
        presence = _presence_frame(question, trie)
        if presence is not None:
            frame = presence
    if not isinstance(frame, str):
        frame.values = sorted(values(question))
        frame.compare = _compare(question) if frame.values else ""
    return frame


_UP = re.compile(r"\b(?:more than|greater than|in excess of|exceed\w*|over|above|at least|no less than|not less than|"
                 r"minimum(?: of)?|or more)\b", re.I)
_DOWN = re.compile(r"\b(?:less than|fewer than|under|below|up to|no more than|not more than|not to exceed|at most|"
                   r"maximum(?: of)?|or less|within)\b", re.I)


def _compare(text: str) -> str:
    up, down = _UP.search(text), _DOWN.search(text)
    return "" if bool(up) == bool(down) else "up" if up else "down"


def _same_side(frame: Frame, sentence: str) -> bool:
    """"Can the Borrower use more than $175,000,000 ...?" isn't answered by "up to $175,000,000": a value the question
    puts on one side needs that side in the sentence, next to it."""
    if not frame.compare:
        return True
    for v in frame.values:
        for m in re.finditer(re.escape(v) if v[0].isalpha() else r"[\d,.]*".join(re.escape(c) for c in v), sentence, re.I):
            if _compare(sentence[max(0, m.start() - 45):m.end() + 12]) == frame.compare:
                return True
    return False


def _action_frame(question: str, trie: dict) -> Frame | str:
    raw = question.lower()
    toks = tokens(question)
    if not toks or toks[0] not in QUESTION_AUX:
        return "doesn't start with an auxiliary"
    tags = tag(toks[1:], trie)
    # "the shares", "any copies", "its return": an action word after a determiner is a noun in a question
    tags = [Tag("WORD", toks[1:][t.start], t.start, t.end) if t.kind == "ACTION" and t.start
            and toks[1:][t.start - 1] in _DETERMINERS and t.end in (-1, t.start + 1) else t for t in tags]
    first_block = next((t.start for t in tags if t.kind == "BLOCK"), len(toks))
    if any(t.name in ("NOT", "NONE") and t.start < first_block for t in tags):
        return "negated question"  # (a "not" in the question's condition is the condition's: "if it isn't defined")
    # A condition in the question ("... if the other party undergoes a change
    # of control", "... only for the purposes of the agreement") is what the
    # clause's own condition must match; the frame is the part before it.
    conds, only, cues, cue_words = [], False, [], []
    blocks = [t for t in tags if t.kind == "BLOCK"]
    if blocks:
        b = blocks[0]
        only = toks[1:][b.start] in ONLY_WORDS
        cues = sorted({_cue_kind(toks[1:][t.start:t.end]) for t in blocks})
        cue_words = sorted({" ".join(toks[1:][t.start:t.end]) for t in blocks})
        conds = _content_items([t for t in tags if t.start > b.start and t.kind != "BLOCK"], trie)
        if not conds and not only:
            return "a condition with nothing to match"
        tags = [t for t in tags if t.start < b.start]
    actors = [t for t in tags if t.kind == "ACTOR"]
    actions = [t for t in tags if t.kind == "ACTION"]
    props = [t for t in tags if t.kind == "PROP"]
    if not actions and props and toks[0] in ("is", "are"):
        return _property_frame(raw, tags, props)
    if actors and (not actions or actions[0].start > actors[0].start and any(
            t.kind == "WORD" for t in tags if actors[0].start < t.start < actions[0].start)) and (
            verb := _open_verb(toks[0], tags, actors[0], toks[1:])) is not None:
        # the question's own verb comes first: "Can the Tenant give an Early Termination Notice" asks about giving
        tags = [Tag("ACTION", f"{OPEN}{verb.name}", verb.start, verb.end) if t is verb else
                Tag("WORD", toks[1:][t.start], t.start, t.end) if t.kind == "ACTION" else t for t in tags]
        actions = [t for t in tags if t.kind == "ACTION"]
    if not actors or not actions or actors[0].start > actions[0].start:
        return "no actor before an action"
    # Nothing the question asks about may come before its actor: "Is there a limit on the number of subsidiaries
    # to which the Company can assign ...?" is not "Can the Company assign ...?" (2026-10-01; generated questions,
    # where the frame from the actor on answered what the question didn't ask).
    if any(not (t.kind == "THING" and t.name == "AGREEMENT" or t.kind == "MODAL" or t.kind == "WORD" and t.name in _PREFACE)
           for t in tags if t.start < actors[0].start):
        return "the question names something before its actor"
    toks1 = toks[1:]
    for a, b in zip(actions, actions[1:]):  # only "soliciting or hiring": adjacent, joined by "or"
        if toks1[a.start + 1:b.start] != ["or"]:
            return "more than one action (\"use and distribute\")"
    actor, action = actors[0], actions[0]
    between = [t for t in tags if actor.start < t.start < action.start]
    if any(t.kind == "WORD" and t.name != "have" for t in between):
        return "a word between the actor and the action isn't in the lexicon"
    asks = QUESTION_AUX[toks[0]]
    if len({t.name for t in between if t.kind == "MODAL" and t.name in ("MAY", "MUST", "BAN")}) > 1:
        return "two modalities at once (\"prohibited from having to\")"
    for t in between:  # "Is the tenant allowed to", "Does the tenant have to", "Is a party prohibited from"
        if t.kind == "MODAL":
            asks = {"MAY": "CAN", "MUST": "MUST", "BAN": "PROHIBITED"}.get(t.name, asks)
    last_action = actions[-1]
    things = [(t.kind, t.name) if t.kind != "MODAL" else ("THING", "RIGHTS") for t in tags
              if t.start > last_action.start and (t.kind in ("THING", "WORD", "ACTOR")
                                                  or t.kind == "MODAL" and toks[1:][t.start] == "right")]
    # ("its rights" after the verb is what it acts on, not "may": "Can a party transfer its rights?")
    if any(k == "PROP" for k, _ in ((t.kind, t.name) for t in tags if t.start > actor.start)):
        return "a property inside an action question"
    things = [th for th in things if th != ("ACTOR", "IT")]  # "with its employees": "its" is the actor's own
    any_things = [(t.kind, t.name) for t in tags if t.start > last_action.start and t.kind in ("THING", "WORD")
                  and t.start >= 1 and toks[1:][t.start - 1] == "any" and t.start >= 2
                  and toks[1:][t.start - 2] in ("from", "to", "with", "by", "for")]  # "from any insurer"
    return Frame(asks, actor.name, action.name, things, actions=[a.name for a in actions], conds=conds, only=only,
                 cues=cues, cue_words=cue_words, any_things=any_things)


_DETERMINERS = frozenset("the a an such any all its their his her our your my these those this that each no every "
                         "some".split())
OPEN = "V:"  # an action outside the lexicon: the question's own verb, matched as that word ("V:surrender")
_NOT_VERBS = frozenset("ther which who whom whose what when where why how one each other any all some no".split())


_NOUN_ENDINGS = ("anc", "enc", "ment", "tion", "sion", "ity", "ness", "ship", "ism")
_BEFORE_VERB = frozenset("can may must shall will would could should do does did to from".split())


def _open_verb(aux: str, tags: list, actor: Tag, toks: list) -> Tag | None:
    """The question's verb when the lexicon doesn't know it: the word right after the actor and its modals ("Must
    the Holder surrender the Note", "Is the Corporation required to cooperate"). Only after a modal or "do" (or a
    modal word: "required to"), so "Is the Company solvent" isn't read as a verb. (2026-10-01)"""
    after = [t for t in tags if t.start > actor.start]
    k, modal = 0, aux in ("can", "may", "must", "shall", "should", "will", "would", "could", "do", "does", "did")
    while k < len(after) and (after[k].kind == "MODAL" or after[k].kind == "WORD" and (
            after[k].name.endswith("ly") or after[k].name == "have")):
        modal = modal or after[k].kind == "MODAL"
        k += 1
    if not modal or k >= len(after) or after[k].kind != "WORD":
        return None
    v = after[k]
    if v.name in _GENERIC or v.name in _NOT_VERBS or len(v.name) < 3 or not v.name.isalpha():
        return None
    if v.name.endswith(_NOUN_ENDINGS) or v.start >= 2 and toks[v.start - 2] in ("entitled", "entitl", "subject"):
        return None  # "entitled to severance": a noun
    prev = toks[v.start - 1] if v.start else ""
    actor_end = (actor.end if actor.end > 0 else actor.start + 1) - 1
    if not (prev in _BEFORE_VERB or prev.endswith("ly") or v.start - 1 == actor_end):
        return None  # "Does a party have a right of first refusal": "first" is no verb
    return v


_PREFACE = frozenset(normalize(w) for w in "say says said state states provide provides specify specifies stipulate "
                     "stipulates mean means true actually really still ever also legally contractually expressly "
                     "explicitly clearly generally typically normally".split())
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty".split())}
_NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
                      "ninety": "90", "hundred": "100", "forty five": "45", "forty-five": "45"})
_VALUE = re.compile(r"\d[\d,]*(?:\.\d+)?|\b(?:" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b"
                    r"|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
                    re.I)


def values(text: str) -> set:
    """Numbers, amounts and months in the text, normalized: "$5,000,000" -> 5000000,
    "eighteen (18)" -> 18, "December" -> december. ("may" only as a month next to a digit.)"""
    out = set()
    for m in _VALUE.finditer(text):
        v = m.group(0).lower()
        if v == "may" and not re.match(r"\s*\d", text[m.end():m.end() + 3]):
            continue
        if v[0].isdigit():
            v = v.replace(",", "").rstrip(".")
            v = v[:-3] if v.endswith(".00") else v
        out.add(_NUMBER_WORDS.get(v, v))
    return out


ONLY_WORDS = {"only", "solely", "exclusively"}
_EXCEPT_ONLY = re.compile(r"\b(?:other than|except|save|unless)\b", re.I)
_NEGATION_WORDS = {"no", "not", "nothing", "never", "none", "neither", "nor", "cannot"}
_NOT_NEGATION_AFTER = {"later", "less", "more", "fewer", "earlier", "limitation", "event"}
# Words of a question that say how it asks, not what it asks about.
_GENERIC = frozenset(normalize(w) for w in (
    "specify specified specifies say says state states contain contains include includes address addresses "
    "set out stipulate stipulates describe describes which what when whether how who where there one other party "
    "parties agreement contract clause section provision it its any some certain do does is are be has have that "
    "this these those undergo undergoes occur occurs happen happens experience experiences become becomes "
    "require requires required trigger triggers result results cause causes lead leads apply applies exist exists "
    "specific particular given period compel compels compelled obligate obligates mandate mandates").split())
_GENERIC_CONCEPTS = {("THING", "AGREEMENT"), ("ACTOR", "ALL"), ("ACTOR", "ANY"), ("ACTOR", "OTHER"),
                     ("ACTOR", "NONE"), ("ACTOR", "IT"), ("ACTOR", "WE"), ("ACTOR", "USER")}
# Single-word phrases too common to show a concept is present ("others", "data").
_PRESENCE_SKIP = {"others", "other", "data", "information", "any person", "right", "unit", "property", "records",
                  "accounts", "order", "use", "process", "hold", "keep", "release", "provide", "engage", "carry",
                  "extend", "return", "exceed", "cap", "fix", "trade", "option", "discretion"}
_PRESENCE_SUBJECTS = (r"(?:the |this )?(?:agreement|contract|clause|license|licence|lease|document|nda|section|provision|"
                      r"policy|terms)s?")
_PRESENCE_VERBS = (r"(?:specify|specifies|say|says|state|states|provide|provides|set out|sets out|contain|contains|"
                   r"include|includes|address|addresses|stipulate|stipulates|impose|imposes|establish|establishes|"
                   r"require|requires|grant|grants|give|gives|allow for|provide for|indicate|indicates|mention|mentions|"
                   r"set forth|sets forth|spell out|spells out|identify|identifies|lay out|lays out|describe|describes|"
                   r"list|lists|define|defines)")
_PRESENCE_PATTERNS = [
    re.compile(rf"^(?:does|do|did) {_PRESENCE_SUBJECTS} (?P<verb>{_PRESENCE_VERBS})(?: that| whether| for| to)?\s+(?P<rest>.+)$"),
    re.compile(r"^(?:is|are) there (?:a |an |any )?(?P<rest>.+)$"),
    re.compile(r"^(?:does|do) (?P<rest>\w+ing\b.+)$"),
    re.compile(r"^(?:is|are) (?:a|an|the|any|either|each) [\w' -]{2,40}? (?P<verb>entitled|required|obligated) to "
               r"(?:receive |pay |collect |recover )?(?P<rest>(?!\w+ )?[^?]+)$"),
    re.compile(r"^(?:does|do) (?:a|an|the|any|either|each) [\w' -]{2,40}? (?:have|has) (?P<rest>.+)$"),
    # last: any other auxiliary question, as long as it names two things
    re.compile(r"^(?:does|do|is|are|will|can|must|shall|may) (?P<rest>.+)$"),
]
_WHEN_VERBS = {normalize(w) for w in "expire end terminate begin commence start run lapse cease".split()} | {
    _crude(w) for w in "expire end terminate begin commence start run lapse cease".split()} | {"TERMINATE", "RENEW"}
_DATE_OR_DURATION = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|sixty|ninety)"
    r"\s*(?:\(\d+\)\s*)?(?:full\s+|calendar\s+|consecutive\s+|contract\s+)?(?:years?|months?|days?|weeks?)\b"
    r"|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:19|20)\d\d\b|\banniversary\b|\bin perpetuity\b|\bperpetual", re.I)


@dataclass(frozen=True)
class Item:
    """One thing a presence frame or a condition needs: any stem (a prefix of
    a sentence word) or any phrase (in the sentence's normalized words)."""
    label: str
    stems: frozenset
    phrases: frozenset


def _concept_item(kind: str, name: str, surface: str, lexicon: dict | None = None) -> Item:
    phrases_in = (lexicon or LEXICON).get((kind, name), [])
    stems, phrases = set(), {" ".join(tokens(surface))} if surface else set()
    for p in phrases_in:
        if p in _PRESENCE_SKIP:
            continue
        words = p.split()
        if len(words) == 1:
            st = _crude(words[0])
            if len(st) >= 3:
                stems.add(st)
        else:
            phrases.add(" ".join(tokens(p)))
    return Item(f"{kind}:{name}", frozenset(stems), frozenset(ph for ph in phrases if ph))


_VOICE_ITEMS = {"WE": Item("ACTOR:WE", frozenset(), frozenset({"we", "us", "our", "the company", "company"})),
                "USER": Item("ACTOR:USER", frozenset(), frozenset({"you", "your", "user", "users", "the user"}))}


def _content_items(tags: list, trie: dict, toks: list | None = None) -> list:
    """What a stretch of question must find in a clause: concepts (with their
    synonyms, and the question's own word) and literal words (as stems),
    minus the generic ones."""
    items = []
    for t in tags:
        # the question's own word, when the tag is that one word ("property"), not a phrase ("initial term")
        one_word = t.end in (-1, t.start + 1)
        own = _crude(toks[t.start]) if toks and t.start < len(toks) and one_word else ""
        if t.kind == "WORD" and t.name in ("i", "me"):  # an "I" no party was found for: the text must speak as "I"
            items.append(Item("WORD:i", frozenset(), frozenset({"i", "me", "my"})))
            continue
        if t.kind == "WORD":
            if t.name in _GENERIC or len(t.name) < 3 or t.name in _NEGATION_WORDS:
                continue
            items.append(Item(f"WORD:{t.name}", frozenset([t.name]), frozenset()))
        elif t.kind == "ACTOR" and t.name in _VOICE_ITEMS:
            items.append(_VOICE_ITEMS[t.name])  # "Is the company responsible ...?" isn't "you are responsible"
        elif t.kind in ("THING", "ACTION", "PROP", "ACTOR"):
            if (t.kind, t.name) in _GENERIC_CONCEPTS or t.kind == "ACTOR" and t.name.isupper() and t.name not in (
                    "RECEIVER", "DISCLOSER"):
                continue
            item = _concept_item(t.kind, t.name, "")
            if len(own) >= 4 and own not in _PRESENCE_SKIP:
                item = Item(item.label, item.stems | {own}, item.phrases)
            items.append(item)
    return items


def _presence_frame(question: str, trie: dict) -> Frame | None:
    q = " ".join(question.lower().replace("’", "'").split()).rstrip("?. ")
    q_named = any(t.kind == "ACTOR" and (t.kind, t.name) not in _GENERIC_CONCEPTS for t in tag(tokens(q)[1:], trie)) \
        or qshapes.subject(question) is not None  # "Is the Secured Party required to pay ...?" asks who pays
    if re.match(r"^(?:is|are) there\b.*\bfor\s+(?:me|us|you|them|him|her|the\s+\w+|a\s+party|either\s+party)\s+to\s+\w+", q):
        return None  # "Is there any right for me to own the game code?" asks what someone may do
    for i, pattern in enumerate(_PRESENCE_PATTERNS):
        if i >= 3 and q_named:
            return None
        m = pattern.match(q)
        if not m:
            continue
        rest = m.group("rest")
        verb = m.groupdict().get("verb") or ""
        strict = i >= 3  # "entitled/required to", "does a party have", catch-all
        if i in (3, 4) and not re.match(r"^(?:is|are|does|do) (?:a|an|any|either|each) (?:party|parties|of the parties)\b|"
                                        r"^(?:is|are|does|do) the parties\b", q) and re.search(
                                            r"^(?:is|are|does|do) (?:the|either|each) (?:other|non\W?\w+|\w+ing) part", q):
            return None  # "Is the other party required to file ...?" asks who must, not whether it's mentioned
        if strict and any(t.kind == "ACTOR" and (t.kind, t.name) not in _GENERIC_CONCEPTS
                          for t in tag(tokens(rest), trie)):
            return None  # "Must the employee reimburse the employer?" is about who does what
        if i >= len(_PRESENCE_PATTERNS) - 2:
            # The catch-alls don't take over action questions the action frame
            # turned down ("Is the tenant exempt from paying rent?"), nor negations.
            rtags = tag(tokens(rest), trie)
            if any(t.kind == "MODAL" and t.name in ("NOT", "BAN", "NEITHER") or t.name == "NONE" for t in rtags):
                return None
            # A condition word in the verb's place is the question's verb, which the presence
            # frame would drop: "does it save my health data?" ("save" as in "save as provided")
            # is not "is health data mentioned?". ("...continue after the agreement terminates"
            # keeps its "after".)
            if any(t.kind == "BLOCK" and 1 <= t.start <= 2 for t in rtags):
                return None
            named = [t for t in rtags if t.kind == "ACTOR" and (t.kind, t.name) not in _GENERIC_CONCEPTS]
            if named and any(t.kind == "ACTION" and t.start > named[0].start for t in rtags):
                return None
        # "Does assigning require consent?" asks about a consent; "Is the employee required to accrue vacation?" and
        # "Can the service require me to indemnify them?" don't (no "shall not ... without" reading for them)
        require = (verb.startswith("requir") or bool(re.search(r"\brequire[sd]?\b", rest))) and i != 3 and not re.search(
            r"\brequire[sd]?\s+(?:\w+\s+){0,4}?to\s+\w+", f"{verb} {rest}")
        when = bool(re.match(r"(?:when|how long)\b", rest))
        dated = bool(re.search(r"\b(?:date|dates|how long|duration|period)\b", rest)) and not when
        negative = bool(re.search(r"\b(no|not|nothing|never|none)\b", rest))
        # "entitled to X or Y": noun alternatives; elsewhere "or" stays inside one item list
        parts = [p for p in re.split(r"\s+or\s+", rest)] if i == 3 else [rest]
        alts = []
        for part in parts:
            ptoks = tokens(part)
            items = _content_items(tag(ptoks, trie), trie, ptoks)
            if dated:  # "the date on which it becomes effective": the date is the answer, not a word to find
                items = [it for it in items if it.label not in ("WORD:date", "WORD:dat", "WORD:period", "WORD:duration")]
            if when:  # "when its initial term expires": the head noun, and a date or duration for the value
                items = [it for it in items if not it.label.startswith("ACTION:") and it.label.split(":")[1] not in _WHEN_VERBS]
                items = items[-1:]
            if items:
                alts.append(items)
        if not alts or i == len(_PRESENCE_PATTERNS) - 1 and any(len(a) < 2 for a in alts):
            return None
        if i == len(_PRESENCE_PATTERNS) - 1 and _IDENTITY.match(q):
            return None  # "Is the Covered Person a corporation?": both words in one sentence don't make it so
        return Frame("EXISTS", None, None, [], alts=alts, negative=negative, require=require, when=when, strict=strict,
                     dated=dated, guards=_presence_guards(q))
    return None


# A presence frame only asks that one sentence hold every word; these keep the relations a question adds between
# them (2026-10-01, from generated questions where the words were there and the answer wasn't):
_IDENTITY = re.compile(r"^(?:is|are) (?:the |a |an )?(?!this\b|it\b|there\b)[\w'-]+(?: [\w'-]+){0,4} (?:a|an) [\w-]+(?: [\w-]+)?$")
_SIDE_WORDS = {"before": {"before", "prior", "until", "till", "preceding", "earlier"},
               "after": {"after", "following", "upon", "once", "subsequent", "thereafter", "surviv", "survive", "survives",
                         "survival", "continue", "continues", "continu", "beyond", "post", "later", "remain"}}
_EXCLUDERS = re.compile(r"\b(?:excluding|exclusive of|without regard to|without reference to|without giving effect to|"
                        r"other than|excluded|exclusion of|disregarding)\b(?:\W+\w+){0,2}?\W+$")
_FORBIDS_S = re.compile(r"\b(?:prohibit\w*|forbid\w*|ban|bans|banned|barred|not\s+(?:be\s+)?(?:permitted|allowed)|"
                        r"may\s+not|shall\s+not|must\s+not|restrict\w*)\b", re.I)
_PERMITS_S = re.compile(r"\b(?:permitted|allowed|may|entitled|free\s+to)\b", re.I)
_EXCLUDES_S = re.compile(r"\b(?:exclusive of|excluding|exclude[sd]?|does not include|do not include|shall not include|"
                         r"not including)\b", re.I)
_RESTRICT_Q = re.compile(r"\b(?:limited to|only|solely|exclusively|restricted to)\b")
_RESTRICT_S = re.compile(r"\b(?:only|solely|exclusively|(?<!not )limited to|restricted to|limited solely|confined to)\b", re.I)


def _presence_guards(q: str) -> dict:
    """What the question's relations need from the sentence: a side of a time ("before October 10"), "without",
    an agent or means ("owed by Macquarie", "by assigning"), a restriction ("limited to technical information")."""
    g = {}
    if m := re.search(r"\b(before|prior to|until|earlier than|after|following|later than)\b", q):
        g["side"] = "before" if m.group(1) in ("before", "prior to", "until", "earlier than") else "after"
    if re.search(r"\bwithout\b", q):
        g["without"] = True
    agents = [(w, a) for w, a in re.findall(r"\b([a-z]{3,})\s+by\s+(?:the\s+|a\s+|an\s+|its\s+|their\s+)?([a-z]{3,})", q)
              if w not in ("and", "or", "not", "made", "done", "set", "determined", "provided", "required", "permitted",
                           "governed", "construed", "interpreted", "enforced", "bound", "covered", "secured")]
    if agents:
        g["agents"] = [[_crude(w), _crude(a)] for w, a in agents]
    if _RESTRICT_Q.search(q):
        g["restrict"] = True
    if re.search(r"\binclud", q):
        g["include"] = True
    if re.search(r"\b(?:permitted|allowed|entitled|free to|may|can)\b", q) and not re.search(r"\bnot\b", q):
        g["permit"] = True
    elif re.search(r"\b(?:prohibited|forbidden|banned|barred|not allowed|not permitted)\b", q):
        g["forbid"] = True
    if m := _DETAIL_Q.search(q):
        noun = m.group(1)
        g["detail"] = next((k for k, nouns in _DETAIL_KINDS.items() if noun in nouns), "number")
    if re.match(r"^(?:is|are) (?:a|an|the|any|either|each) [\w' -]{2,40}? (?:required|obligated) to\b", q) or re.search(
            r"\b(?:required|mandatory|obligatory)\b", q) and not re.match(
            r"^(?:does|do) (?:the |this )?(?:agreement|contract|clause|lease|terms)\b", q):
        g["obligation"] = True
    if re.search(r"\b(?:all|every|each)\b", q):
        g["all"] = True
    if re.search(r"\b(?:if|when|whenever|once|unless|after|upon|in the event)\b", q):
        g["cond"] = True  # a concept can come back in the condition ("If the lease is terminated, does ... terminate?")
    return g


# "Is there a specified interest rate ...?", "Does it state the exact amount ...?": the sentence must give one
_DETAIL_Q = re.compile(r"\b(?:specified|specific|exact|particular|fixed|stated|set|maximum|minimum|precise|"
                       r"(?:limit|cap|ceiling|floor)\s+(?:on|of|to|for)(?:\s+the)?)\s+(?:\w+\s+)?"
                       r"(rate|amount|number|percentage|price|fee|fees|sum|date|deadline|cap|limit|figure|dollar|"
                       r"interest|period|time|payment|salary|cost)\b")
_DETAIL_KINDS = {"rate": {"rate", "percentage", "interest"}, "date": {"date", "deadline"},
                 "money": {"amount", "price", "fee", "fees", "sum", "dollar", "payment", "salary", "cost", "figure"},
                 "number": {"number", "cap", "limit"}, "time": {"period", "time"}}
_DETAIL_S = {"rate": re.compile(r"%|\bper\s?cent|\bbasis points\b", re.I),
             "date": re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|"
                                r"december)\s+\d|\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.I),
             "money": re.compile(r"[$€£]\s?\d|\b\d[\d,.]*\s*(?:dollars|usd|eur|euros|pounds)\b", re.I),
             "number": re.compile(r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|"
                                  r"thirty|sixty|ninety|hundred)\b", re.I),
             "time": _DATE_OR_DURATION}
_OBLIGATION_S = re.compile(r"\b(?:shall|must|will|required|agree\w*|obligated|obliged|undertakes?|covenants?|responsible)\b",
                           re.I)


def _guards_fail(frame: Frame, sentence: str, raw: list, norm: str) -> str | None:
    g = frame.guards
    if not g:
        return None
    words = set(raw) | {_crude(w) for w in raw}
    if "side" in g and not words & _SIDE_WORDS[g["side"]]:
        return f"the question asks what holds {g['side']} a time; the sentence doesn't say"
    if g.get("without") and "without" not in raw:
        return "the question asks 'without ...'; the sentence doesn't say without"
    for w, a in g.get("agents", []):
        if not re.search(rf"\b{re.escape(w)}\w*\s+by\s+(?:\w+\s+){{0,3}}{re.escape(a)}", " ".join(raw)):
            return f"the question names who or how ('{w} by {a}'); the sentence doesn't"
    if g.get("permit") and _FORBIDS_S.search(sentence):
        return "the question asks whether it is allowed; the sentence prohibits something"
    if g.get("forbid") and _PERMITS_S.search(sentence) and not _FORBIDS_S.search(sentence):
        return "the question asks whether it is prohibited; the sentence permits it"
    if g.get("include") and _EXCLUDES_S.search(sentence):
        return "the question asks whether it includes something; the sentence excludes things"
    if g.get("restrict") and not _RESTRICT_S.search(sentence):
        return "the question asks whether it is limited to something; the sentence states no limit"
    if "detail" in g and not _DETAIL_S[g["detail"]].search(sentence):
        return f"the question asks for a specific {g['detail']}; the sentence gives none"
    if g.get("obligation") and not _OBLIGATION_S.search(sentence):
        return "the question asks whether it is required; the sentence states no obligation"
    if g.get("all") and re.search(r"\b(?:some|certain|any of)\b", sentence, re.I):
        return "the question asks about all of them; the sentence speaks of some"
    return None


def _excluded(alt: list, raw: list, sentence: str) -> bool:
    """Whether an item of the question is only in the sentence as what it excludes ("governed by the laws of Ohio,
    excluding conflict of law provisions", "other than a natural person")."""
    low = " ".join(raw)
    for it in alt:
        starts = [m.start() for st in it.stems for m in re.finditer(rf"\b{re.escape(st)}", low)]
        starts += [m.start() for ph in it.phrases for m in re.finditer(rf"\b{re.escape(ph.split()[0])}", low)]
        if starts and all(_EXCLUDERS.search(low[:s]) for s in starts):
            return True
    return False


def _property_frame(raw: str, tags: list, props: list) -> Frame | str:
    if " and " in raw:
        return "more than one property"
    subject = [(t.kind, t.name) for t in tags if t.start < props[0].start and t.kind in ("THING", "WORD")]
    if not subject or any(t.kind in ("ACTION", "MODAL") for t in tags):
        return "no subject for the property"
    if any(t.kind == "WORD" for t in tags if t.start > props[0].start):
        return "words after the property"
    return Frame("PROPERTY", None, None, subject, props=[p.name for p in props])


def _has_thing(thing: tuple, clause: list) -> bool:
    kind, name = thing
    for t in clause:
        if t.kind == kind and t.name == name:
            return True
        if kind == "THING" and t.kind == "THING" and BROADER.get(t.name) == name:
            return True
        if kind == "ACTOR" and t.kind == "ACTOR" and (name in ("OTHER", "ANY") or t.name == name):
            return True
    return False


def _clauses(tags: list) -> list:
    out, cur = [], []
    for t in tags:
        if t.kind == "BREAK":
            out.append(cur)
            cur = []
        else:
            cur.append(t)
    return out + [cur]


def _upon_notice(t: Tag, tags: list) -> bool:
    return t.name == "UPON" and any((u.kind == "WORD" and u.name.startswith("notic") or u.kind == "THING" and u.name == "NOTICE")
                                    and 0 < u.start - t.start <= 8 for u in tags)


def _blocked(tags: list) -> bool:
    return any(t.kind == "BLOCK" and not _upon_notice(t, tags) for t in tags)


def condition_text(sentence: str, frame: "Frame | None" = None, max_chars: int = 160) -> str:
    """The sentence's first condition or exception, from its cue to the end of its clause ("unless the Licensor
    consents in writing"), for a "yes, with a condition" to quote. Not "upon ... notice", nor the cue that brings
    in the question's own condition."""
    raw = _raw_tokens(sentence)
    tags = tag([normalize(w) for w in raw], _BASE_TRIE)
    norm = " " + " ".join(normalize(w) for w in raw) + " "
    for t in tags:
        if t.kind != "BLOCK" or _upon_notice(t, tags):
            continue
        if frame is not None and frame.conds and any(_item_present(it, raw, norm, t.start, t.start + 16)
                                                     for it in frame.conds):
            continue
        cue = re.search(r"(?<!\w)" + r"\W+".join(map(re.escape, raw[t.start:t.end])) + r"(?!\w)", sentence, re.I)
        if not cue:
            continue
        # to the first break that leaves the condition a few words ("unless, in its judgment, ..." reads on)
        rest, text = sentence[cue.start():], ""
        for end in re.finditer(r"[;.,]\s|[;.]$", rest):
            text = rest[:end.start()]
            if len(text.split()) >= 3:
                break
        text = text if len(text.split()) >= 3 else rest.rstrip(" .;")
        return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"
    return ""


def _actor_matches(asked: str, actor: str) -> bool:
    if actor in ("ALL", "NONE"):
        return True
    if asked in ("ANY", "ALL"):
        return actor not in ("OTHER",)
    return actor == asked


def _judge(frame: Frame, sentence: str, trie: dict) -> tuple[str, str | None] | str | None:
    """(answer, qualifier) if the sentence settles the frame, a reason string
    if it would but is blocked, None if it doesn't address the frame. A frame
    without things asks only whether the actor may do the action at all."""
    if frame.asks == "EXISTS":
        return _judge_presence(frame, sentence, trie)
    if frame.values and not set(frame.values) <= values(sentence):
        return None
    if not _same_side(frame, sentence):
        return None
    if frame.asks == "PROPERTY":
        return _judge_property(frame, tag(tokens(_CROSS_REFERENCE.sub(" ", sentence)), trie), sentence)
    toks = tokens(sentence)
    tags = tag(toks, trie)
    if ("THING", "RIGHTS") in [tuple(t) for t in frame.things]:
        first = min((t.start for t in tags if t.kind == "ACTION"), default=len(toks))
        tags = [Tag("THING", "RIGHTS", t.start, t.end) if t.kind == "MODAL" and t.start > first and toks[t.start] == "right"
                else t for t in tags]
    literal = {a[len(OPEN):] for a in (frame.actions or [frame.action]) if a and a.startswith(OPEN)}
    if literal:
        tags = [Tag("ACTION", OPEN + t.name, t.start, t.end) if t.kind == "WORD" and t.name in literal else t for t in tags]
    caps = _case_flags(sentence)
    for clause in _clauses(tags):
        wanted = set(frame.actions or [frame.action])
        composite = {v for a in wanted if a in COMPOSITE for v in COMPOSITE[a][0]
                     if any(t.kind == "THING" and t.name == COMPOSITE[a][1] for t in clause)}
        # a composite verb needs its thing close after it ("provide ... with prompt written notice", "maintain ...
        # insurance"), not anywhere in the clause ("not prohibited from disclosing ..., provided that it gives notice")
        actions = [t for t in clause if t.kind == "ACTION" and (t.name in wanted or t.name in composite and (
            "NOTIFY" not in wanted or any(u.kind == "THING" and u.name == "NOTICE" and 0 < u.start - t.start <= 8
                                          for u in clause)))]
        for action in actions:
            before = [t for t in clause if t.start < action.start]
            passive = bool(action.start and toks[action.start - 1] in _PASSIVE_AUX)
            if passive:
                # "This Agreement may be terminated ... by either party": the agent acts.
                actor = _agent(clause, action, toks, caps, frame)
                anyone = False
                if actor is None and frame.actor in ("ANY", "ALL") and not any(
                        toks[j] == "by" for j in range(action.start + 1, min(action.start + 4, len(toks)))):
                    # "The Source Code shall be deposited": by anyone. Only ever a "yes": "Revenue ... shall not be
                    # shared", "there are no consents required to be obtained" defer (no new "no" answers).
                    actor, anyone = Tag("ACTOR", "ANY", action.start), True
                    commas = _comma_positions(sentence)
                    if any(t.kind == "BLOCK" and toks[t.start] in ("if", "when", "where", "unless", "whenever", "once")
                           and 0 < action.start - t.start <= 10 and not any(t.start < c <= action.start for c in commas)
                           for t in clause):
                        continue  # "if the Agreement is terminated for any reason other than ...": a condition, not a fact
                if actor is None:
                    continue
                if not _actor_matches(frame.actor, actor.name):
                    continue
                subject_start = max((t.start for t in clause if t.kind == "ACTION" and t.start < action.start), default=-1)
                # only the modal run right before the verb ("which may be maintained by X"),
                # not "shall not be limited" earlier in the sentence
                modals, words, k = set(), set(), len(before) - 1
                while k >= 0 and (before[k].kind == "MODAL" or before[k].kind == "WORD" and before[k].name.endswith("ly")):
                    if before[k].kind == "MODAL":
                        modals.add(before[k].name)
                        words.add(toks[before[k].start])
                    k -= 1
                if not all(_has_thing(th, [t for t in clause if t is not actor]) for th in frame.things):
                    continue
                if _describes_only(frame, modals, words | {toks[action.start - 1]}):
                    continue
                lo = clause[0].start if clause else 0
                if anyone and any(w in ("no", "none", "nothing", "never") for w in _raw_tokens(sentence)[lo:action.start]):
                    modals = modals | {"NOT"}  # "there are no consents required to be obtained"
                verdict = _verdict(frame, sentence, toks, tags, clause, actor, modals, subject_start)
                if anyone and not (isinstance(verdict, tuple) and verdict[0] == "yes"):
                    continue
                if verdict is not None:
                    return verdict
                continue
            actors = [t for t in before if t.kind == "ACTOR"]
            if not actors and frame.actor in ("ANY", "ALL"):
                actors = [s for s in [_subject_before_modal(before, action, toks, caps)] if s is not None]
            if not actors:
                continue
            actor = actors[-1]
            if actor.name == "IT":  # "Each Party agrees that it shall maintain": the actor named before
                named = [t for t in tags if t.kind == "ACTOR" and t.name != "IT" and t.start < actor.start]
                if not named:
                    continue
                actor = Tag("ACTOR", named[-1].name, actor.start)
            if not _actor_matches(frame.actor, actor.name):
                continue
            # The actor must be the action's subject: right before a modal or the
            # action ("rights of Franchisor ... which may disparage" is not), with
            # no other action in between ("X may terminate ... if Y challenges"),
            # and the action not passive ("of the Recipient shall be returned").
            after_actor = [t for t in clause if actor.start < t.start <= action.start and t.kind != "ACTOR"
                           and not (t.kind == "WORD" and t.name.endswith("ly"))]  # "Licensee expressly agrees"
            if not after_actor or after_actor[0].kind not in ("MODAL", "ACTION"):
                continue
            if any(t.kind == "ACTION" and t is not action and (t.name != "NOTIFY" or "NOTIFY" in wanted)
                   for t in after_actor):
                continue  # (giving notice on the way isn't another act: "may, after giving notice, audit")
            if after_actor[0] is action and action.start and toks[action.start - 1] == "to" and _object_of_preposition(
                    toks, actor.start):
                continue  # "compensation to such Grantor) to use": an object, not the actor ("the right of X to use" is)
            if _after_preposition(toks, actor.start) and not (action.start and toks[action.start - 1] == "to"):
                continue  # "without notice or consent to Guarantor, assign": Guarantor isn't the one assigning
            if _negated_control(toks, clause, actor):
                continue  # "shall not be construed to require CBS to establish": nothing is said CBS must do
            if not all(_has_thing(th, [t for t in clause if t.start > action.start or th[0] == "ACTOR"])
                       for th in frame.things):
                continue
            if frame.any_things and not _any_scope(frame, toks, clause):
                continue  # "from any insurer" isn't "with financially sound and reputable insurers"
            modals = {t.name for t in before if t.kind == "MODAL" and t.start > actor.start}
            if actor.start and toks[actor.start - 1] == "no":
                modals = modals | {"NOT"}  # "no Corporation match of 401(k) contributions can be made"
            if _describes_only(frame, modals, {toks[t.start] for t in before if t.kind == "MODAL" and t.start > actor.start}):
                continue
            prev = max((t.start for t in clause if t.kind == "ACTION" and t.start < actor.start), default=-1)
            verdict = _verdict(frame, sentence, toks, tags, clause, actor, modals, prev)
            if verdict is not None:
                return verdict
    return None


# Who information can go to: a question "can X share it with its employees?" names one of these.
_RECIPIENTS = {"EMPLOYEES", "ADVISORS", "THIRD_PARTIES", "AFFILIATES", "ADVERTISERS", "ANALYTICS", "SERVICE_PROVIDERS",
               "CUSTOMERS"}
_EXCEPTION_WORDS = {"other than", "except", "excepting", "save"}
# After "except", a circumstance ("except with consent", "except as required by law"), not a list of recipients.
_CIRCUMSTANCE = {"with", "as", "in", "where", "if", "pursuant", "upon", "when", "under", "by", "that", "which", "insofar",
                 "so", "otherwise", "after", "before", "during", "within", "until", "unless", "the extent"}


def _permitted_by_exception(frame, toks, clause, actor) -> bool:
    """"Recipient shall not disclose Confidential Information to any person other than to its directors, officers
    and employees": the recipients the question asks about are the exception to a prohibition, so it may share with
    them (2026-10-01; LegalBench dev's ContractNLI sharing misses). Only when the exception lists recipients, not
    a circumstance ("except with the prior written consent ... to its employees" permits nothing)."""
    wanted = [th for th in frame.things if th[0] == "THING" and th[1] in _RECIPIENTS]
    if frame.asks not in ("CAN", "DOES") or not wanted:
        return False
    for i, t in enumerate(clause):
        if t.kind != "BLOCK" or t.start < actor.start or " ".join(toks[t.start:t.end]) not in _EXCEPTION_WORDS:
            continue
        j = t.end
        while j < len(toks) and toks[j] in ("to", "for", "disclosur", "disclosure"):
            j += 1
        if j >= len(toks) or toks[j] in _CIRCUMSTANCE or " ".join(toks[j:j + 2]) in _CIRCUMSTANCE:
            continue
        listed = []  # the exception's own words: up to the next condition or clause break
        for u in clause[i + 1:]:
            if u.kind == "BLOCK":
                break
            listed.append(u)
        if all(_has_thing(th, listed) for th in wanted):
            return True
    return False


_COMPARATIVE = re.compile(r"\b(?:more|fewer|less|greater|longer)\b(?:\s+\w+){0,6}?\s+than\b")


CONDITIONAL = "with a condition"  # the qualifier of a "yes" the sentence attaches a condition to


def _conditional(verdict, condition: str | None):
    """A sentence with a condition or exception ("unless", "subject to", "except"...): its "yes" becomes "yes, with
    a condition", anything else defers with the condition as the reason. On LegalBench dev + held-out + fresh half A,
    the "yes" answers this veto held back were right 540 times out of 541, its "no" answers wrong 31 times out of 42
    ("shall not disclose ... except to its directors" asked about employees). (2026-10-01,
    /root/zadumai_nli_proto/extensive/v3/size_vetoes.py)"""
    if condition is None:
        return verdict
    if isinstance(verdict, tuple) and verdict[0] == "yes":
        return ("yes", f"{verdict[1]}, {CONDITIONAL}" if verdict[1] else CONDITIONAL)
    return condition


def _base_qualifier(qualifier: str | None) -> str | None:
    """The qualifier without the condition: "yes" and "yes, with a condition" agree, "yes" and "yes, may" don't."""
    return (qualifier or "").replace(CONDITIONAL, "").strip(", ") or None


_COPULA = frozenset("is are was were be been being".split())


def _describes_only(frame: Frame, modals: set, words: set) -> bool:
    """A "must" question and a sentence that only says it is so ("information that ... (d) is identified by the
    disclosing party as confidential"): a description, often one alternative of a definition, not a requirement."""
    return frame.asks == "MUST" and modals <= {"DOES"} and bool(words) and words <= _COPULA


def _comma_positions(sentence: str) -> list:
    """Where the commas fall, as indexes into tokens(sentence) (a comma before token i is at i)."""
    out, i = [], 0
    for w in re.findall(r"[a-z]+(?:'[a-z]+)?|;|,", re.sub(r"'s\b", "", sentence.lower().replace("’", "'"))):
        if w == ",":
            out.append(i)
        else:
            i += 1
    return out


_OWED_TO = frozenset(normalize(w) for w in "compensation payment payments royalty royalties fee fees notice consent credit "
                     "charge charges cost costs liability owed due payable paid refund reimbursement".split())


def _object_of_preposition(toks: list, start: int) -> bool:
    """"(... without payment of royalty or other compensation to such Grantor) to use": the party is who gets paid,
    not who uses ("access to Licensor to audit" still reads Licensor as the one auditing)."""
    j = start - 1
    while j >= 0 and toks[j] in ("the", "such", "each", "any", "a", "an", "said"):
        j -= 1
    return j >= 1 and toks[j] in ("to", "for", "from") and toks[j - 1] in _OWED_TO


def _any_scope(frame, toks, clause) -> bool:
    for th in frame.any_things:
        hits = [t for t in clause if _has_thing(th, [t])]
        if not any(set(toks[max(0, t.start - 3):t.start]) & {"any", "all", "each", "every", "whatever", "whichever"}
                   for t in hits):
            return False
    return True


def _after_preposition(toks: list, start: int) -> bool:
    j = start - 1
    while j >= 0 and toks[j] in ("the", "such", "each", "any", "a", "an", "said"):
        j -= 1
    return j >= 0 and toks[j] in ("to", "for", "from", "with", "against", "upon", "toward", "towards", "without")


_CONTROL = frozenset(normalize(w) for w in "require requires requiring permit permits allow allows cause causes "
                     "authorize authorizes enable enables entitle entitles oblige obligate force compel direct".split())


def _negated_control(toks: list, clause: list, actor: Tag) -> bool:
    """The actor as the object of a verb like "require" that a negation governs ("shall not be construed to require
    CBS to establish any plan"): the sentence asserts nothing the actor must or may do."""
    j = actor.start - 1
    while j >= 0 and toks[j] in ("the", "such", "each", "any", "a", "an", "said"):
        j -= 1
    if j < 0 or toks[j] not in _CONTROL:
        return False
    return any(t.kind == "MODAL" and t.name in ("NOT", "BAN") and 0 < j - t.start <= 6 for t in clause) or (
        j >= 2 and "not" in toks[max(0, j - 6):j])


def _restricted_thing(frame, toks, clause) -> bool:
    """"... or to any of Receiving Party's employees who do not have a need to know": the "no" covers only some of
    them, so it doesn't answer whether the employees may be told."""
    for th in frame.things:
        for t in clause:
            if _has_thing(th, [t]) and t.kind == "THING":
                end = t.end if t.end > 0 else t.start + 1
                if any(w in ("who", "that", "which", "whose") for w in toks[end:end + 2]):
                    return True
    return False


def _verdict(frame, sentence, toks, tags, clause, actor, modals, prev):
    """The answer once actor, action and things are in place: conditions, then modality."""
    condition = None
    if frame.conds or frame.only:
        met = _conditions_met(frame, toks, tags, _raw_tokens(sentence), clause)
        if met is False:
            return None
        if isinstance(met, str) and met.startswith(_MISMATCH):
            return met[len(_MISMATCH):]  # the question's own condition isn't the sentence's: defer
        if isinstance(met, str):
            condition = met
    elif _blocked(tags):
        if ("NOT" in modals or "BAN" in modals) and _permitted_by_exception(frame, toks, clause, actor):
            return ("yes", "may" if frame.asks == "DOES" else None)
        condition = "the sentence has a condition or exception"
        if ("THING", "WITHOUT_CAUSE") in frame.things and not _EXPLICIT_WITHOUT_CAUSE.search(sentence):
            # "Can a party terminate without cause?" from "Either party may terminate at any time if the other
            # fails materially to comply": the condition is the cause, so this is no "yes, with a condition"
            # (unless the sentence itself says "without cause", "for convenience", ...)
            return condition
    if ("THING", "WITHOUT_CAUSE") in frame.things and _MUTUAL.search(sentence):
        return "the parties end it together, not one party alone"  # "may terminate at any time by mutual consent"
    verdict = _modality(frame, sentence, toks, clause, actor, modals, prev)
    if isinstance(verdict, tuple) and verdict[0] == "no" and _restricted_thing(frame, toks, clause):
        return "the sentence forbids it only for some of them"
    return _conditional(verdict, condition)


_EXPLICIT_WITHOUT_CAUSE = re.compile(r"\b(?:without cause|for (?:its |their )?convenience|for any (?:or no )?reason|"
                                     r"for no reason|at will)\b", re.I)
_MUTUAL = re.compile(r"\bmutual(?:ly)?\b|\bby (?:the )?(?:written )?agreement of\b|\bby (?:mutual )?written agreement\b"
                     r"|\bby (?:an )?agreement in writing\b|\bexecuted by (?:both|all|the) parties\b|"
                     r"\bby (?:the )?parties' (?:mutual )?(?:written )?agreement\b", re.I)


def _modality(frame, sentence, toks, clause, actor, modals, prev):
    """The answer from the actor's modals and negation."""
    if frame.only and ("NOT" in modals or "BAN" in modals) and any(t.kind == "BLOCK" for t in clause) \
            and _EXCEPT_ONLY.search(sentence):
        modals = modals - {"NOT", "BAN"}  # "agrees not to use ... other than for the purposes": only for them
    # "..., nor shall either party use": a "nor" after the previous action and
    # before this actor negates it. (Only "nor": in "not contributing with
    # coverage the Sponsor may carry", "not" is not the Sponsor's.)
    if "NOT" in modals and "BAN" in modals and any(
            t.kind == "MODAL" and t.name == "BAN" and any(u.kind == "MODAL" and u.name == "NOT" and 0 < t.start - u.start <= 2
                                                          for u in clause) for t in clause):
        modals = (modals - {"NOT", "BAN"}) | {"MAY"}  # "is not prohibited from disclosing": it may
    negated = (actor.name == "NONE" or "NOT" in modals or "BAN" in modals
               or any(t.name == "NEITHER" and t.start < actor.start for t in clause)
               or any(t.kind == "MODAL" and toks[t.start] == "nor" and prev < t.start < actor.start for t in clause))
    if frame.asks == "PROHIBITED":
        if negated:
            return ("yes", None)
        if "MAY" in modals:
            return ("no", None)
        return "the text says it happens, not whether it is prohibited"
    if negated and _COMPARATIVE.search(" ".join(toks[actor.start:])):
        return "the text limits how much, it doesn't forbid it"  # "shall not make more copies than necessary"
    if negated:
        return ("no", None)
    if "MAY" in modals:
        if frame.asks == "MUST":
            return "the text permits it but doesn't require it"
        return ("yes", "may" if frame.asks == "DOES" else None)
    return ("yes", None)


_NOT_SUBJECTS = {"such", "this", "the", "any", "all", "each", "it", "which", "that", "these", "those", "said", "no",
                 "in", "if", "upon", "during", "after", "before", "notwithstanding", "subject", "except", "unless"}


def _case_flags(sentence: str) -> list:
    """Whether each token (aligned with tokens(sentence)) starts with a capital."""
    return [w[0].isupper() for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|;", re.sub(r"'s\b", "", sentence.replace("’", "'")))]


def _subject_before_modal(before: list, action: Tag, toks: list, caps: list) -> Tag | None:
    """For a question about any party: the capitalized noun right before the
    modal run ("Customer specifically agrees to maintain", "Buyer shall obtain")."""
    run = [t for t in before if t.start < action.start]
    i = len(run) - 1
    while i >= 0 and (run[i].kind == "MODAL" or run[i].kind == "WORD" and (run[i].name.endswith("ly") or run[i].name == "have")):
        i -= 1
    j = i + 1
    while j < len(run) and run[j].kind == "WORD" and run[j].name.endswith("ly"):
        j += 1
    if i < 0 or j >= len(run) or run[j].kind != "MODAL":
        return None
    subj = run[i]
    if subj.kind not in ("THING", "WORD") or subj.start >= len(caps) or not caps[subj.start]:
        return None
    if toks[subj.start] in _NOT_SUBJECTS:
        return None
    return Tag("ACTOR", f"SUBJ:{toks[subj.start]}", subj.start)


def _agent(clause: list, action: Tag, toks: list, caps: list, frame: Frame) -> Tag | None:
    """The "by ..." agent of a passive action, within a few words after it."""
    for j in range(action.start + 1, min(action.start + 9, len(toks))):
        if toks[j] != "by":
            continue
        after = [t for t in clause if t.start > j and not (t.kind == "WORD" and t.name in ("either", "both", "one"))][:1]
        if not after or after[0].start > j + 3:
            return None
        t = after[0]
        if t.kind == "ACTOR":
            return t
        if frame.actor in ("ANY", "ALL") and t.kind in ("THING", "WORD") and t.start < len(caps) and caps[t.start] \
                and toks[t.start] not in _NOT_SUBJECTS:
            return Tag("ACTOR", f"SUBJ:{toks[t.start]}", t.start)
        return None
    return None


def _raw_tokens(text: str) -> list:
    """The sentence's words before normalization, aligned with tokens(text)."""
    return _TOKEN.findall(re.sub(r"'s\b", "", text.lower().replace("’", "'")))


def _item_present(item: Item, raw: list, norm: str, lo: int = 0, hi: int | None = None) -> bool:
    """Whether the item is in raw[lo:hi] (norm: the normalized words, space-joined, for phrases)."""
    window = raw[lo:hi]
    if any(w.startswith(st) for st in item.stems for w in window):
        return True
    if item.phrases:
        text = " " + " ".join(normalize(w) for w in window) + " " if (lo or hi is not None) else norm
        return any(f" {ph} " in text for ph in item.phrases)
    return False


def _negated_sentence(raw: list, tags: list) -> bool:
    if any(t.kind == "MODAL" and t.name in ("NOT", "NEITHER", "BAN") or t.kind == "ACTOR" and t.name == "NONE"
           for t in tags):
        return True
    return any(w in ("no", "nothing", "none") and (i + 1 >= len(raw) or raw[i + 1] not in _NOT_NEGATION_AFTER)
               for i, w in enumerate(raw))


# ", unless earlier terminated as provided herein,": a condition on ending early, not on when the term ends.
_UNLESS_EARLIER = re.compile(r",\s*(?:unless|except|subject to)\b[^,.;]{0,140},", re.I)
# "This Agreement shall commence on the Effective Date and shall terminate on December 31, 2022": the agreement
# itself as the subject ("This Development Agreement", "The Agreement"), for questions about its term (2026-10-01;
# 128 of LegalBench dev's misses had every word but this). Only "shall"/"will" (or "and", sharing an earlier one:
# "will take effect ... and remain in effect for one year") with a verb that ends or lasts it, and the date or
# period soon after; no
# "may", renewal or notice on the way ("may be terminated upon thirty (30) days' notice" is a right to end it).
_AGREEMENT_TERM = re.compile(
    r"\b(?:this|the)\s+(?:\w+\s+){0,2}?(?:agreement|contract|lease|license|licence|attachment|addendum|amendment)\b"
    r"(?:(?!\brenew|\bnotice\b|\bmay\b)[^.;]){0,160}?"
    r"\b(?:shall|will|and)\s+(?:automatically\s+)?(?:terminate|expire|end|continue|remain|run|be\s+in\s+(?:full\s+)?"
    r"(?:force|effect))\b(?:(?!\brenew|\bnotice\b)[^.;]){0,70}?" + f"(?:{_DATE_OR_DURATION.pattern})"
    r"(?![^.;]{0,25}\bnotice\b)", re.I)


def _about_the_term(frame: Frame) -> bool:
    return frame.when and any(it.label == "THING:W_TERM" for alt in frame.alts for it in alt)


def _states_when(frame: Frame, sentence: str) -> bool:
    """"The term of this Agreement shall be twelve (12) months", "...shall
    expire on December 31, 2021": the head noun as the subject of a verb that
    sets its date or duration (not "prior to the expiration of the Term", not a renewal term)."""
    stems = sorted({st for alt in frame.alts for it in alt for st in it.stems}, key=len, reverse=True)
    if not stems:
        return False
    sentence = _UNLESS_EARLIER.sub(" ", sentence)
    if _about_the_term(frame) and _AGREEMENT_TERM.search(sentence):
        return True
    head = "|".join(map(re.escape, stems))
    not_after = "".join(f"(?<!{w} )" for w in ("renewal", "current", "applicable", "extension", "successive", "additional", "extend the", "of the",
                                                 "of its", "of this", "during the", "during its", "within the"))
    pattern = re.compile(
        rf"{not_after}\b(?:{head})(?:s|es)?\b(?:[^.;,]{{0,80}}?"
        r"\b(?:shall|will|is|be|continu\w*|expir\w*|end\w*|run\w*|remain\w*|last\w*)\b[^.;]{0,60}?"
        rf"|[^.;,]{{0,30}}?\bof\s+)(?:{_DATE_OR_DURATION.pattern})", re.I)
    return bool(pattern.search(sentence))


def _enough_of_each(alt: list, raw: list) -> bool:
    """A concept the question names twice ("Does the termination of the lease cancel ...?": TERMINATE twice) needs two
    words of it in the sentence, not one ("shall survive the termination")."""
    labels = [it.label for it in alt]
    for it in alt:
        if it.label.startswith("WORD:"):
            continue  # a plain word can come back ("right of first refusal or first offer")
        k = labels.count(it.label)
        if k > 1 and sum(1 for w in raw if any(w.startswith(st) for st in it.stems)) < k:
            return False
    return True


_BOILERPLATE = re.compile(r"\b(?:including,?\s+)?(?:but\s+)?(?:not\s+limited\s+to|without\s+limitation|without\s+limiting"
                          r"(?:\s+the\s+generality\s+of\s+the\s+foregoing)?)\b", re.I)


def _judge_presence(frame: Frame, sentence: str, trie: dict):
    """"yes" when one alternative's items are all in the sentence with the
    question's polarity; a reason when present but negated; None otherwise."""
    plain = _raw_tokens(_BOILERPLATE.sub(" ", sentence))  # "including, but not limited to" isn't a limit
    plain_norm = " " + " ".join(normalize(w) for w in plain) + " "
    raw = _raw_tokens(sentence)
    norm = " " + " ".join(normalize(w) for w in raw) + " "
    if not any(all(_item_present(it, plain, plain_norm) for it in alt) and (frame.guards.get("cond") or _enough_of_each(
            alt, plain)) for alt in frame.alts) and not (
            _about_the_term(frame) and _AGREEMENT_TERM.search(_UNLESS_EARLIER.sub(" ", sentence))):
        return None
    if frame.when and not _states_when(frame, sentence):
        return None
    if frame.dated and not _DATE_OR_DURATION.search(sentence):
        return None
    if frame.values and not set(frame.values) <= values(sentence):
        return None
    if not _same_side(frame, sentence):
        return None
    if (why := _guards_fail(frame, sentence, raw, norm)) is not None:
        return None if frame.strict else why
    if any(all(_item_present(it, raw, norm) for it in alt) and _excluded(alt, raw, sentence) for alt in frame.alts):
        return "the sentence names it only to exclude it"
    stags = tag([normalize(w) for w in raw], trie)
    condition = "the sentence has a condition or exception" if frame.strict and _blocked(stags) else None
    if condition and any(t.kind == "BLOCK" and "discretion" in raw[t.start:t.end] for t in stags):
        # "The Company may, in its sole discretion, pay an annual bonus to the Employee": a presence frame doesn't
        # know whose discretion it is, and asked "Is the employee entitled to a bonus?" the answer is no (adv-38).
        return condition
    return _conditional(_presence_polarity(frame, raw, stags), condition)


def _presence_polarity(frame: Frame, raw: list, stags: list):
    negated = _negated_sentence(raw, stags)
    if frame.negative:
        return ("yes", None) if negated else "the question asks for a negation the sentence doesn't have"
    if frame.require:
        has_without = any(w in ("without", "unless", "except") for w in raw)
        if negated and has_without or not negated and not has_without:
            return ("yes", None)  # "shall not assign without consent" / "subject to the prior consent"
        return "can't tell whether it is required"
    if negated:
        return "the sentence negates it"
    return ("yes", None)


# What kind of condition a cue brings in. A question's "after a change of control" is met by the sentence's "upon a
# change of control" or "if there is a change of control", not by "before" or "during" it; its "without consent" only
# by "without consent"; and a sentence's "except in the case of a change of control" exempts that case from the rule,
# so the rule doesn't answer the question (2026-10-01, generated questions).
_CUE_KINDS = {
    "if": "if", "only if": "if", "in the event": "if", "in case": "if", "where": "if", "when": "if", "whenever": "if",
    "provided that": "if", "provided however": "if", "once": "if", "as soon as": "if", "conditioned on": "if", "conditioned upon": "if",
    "so long as": "if", "as long as": "if", "to the extent": "if", "on the occurrence": "if", "upon": "if",
    "after": "if", "following": "if", "at the request": "if", "upon request": "if", "on request": "if",
    "if requested": "if", "before": "before", "prior to": "before", "until": "before",
    "unless": "except", "except": "except", "excepting": "except", "except as": "except", "other than": "except",
    "save": "except", "notwithstanding": "except", "without": "without", "during": "during", "within": "if",
    "only": "only", "solely": "only", "subject to": "if", "sole discretion": "discretion",
}


_MISMATCH = "mismatch: "
_AFTER_WORDS = {"after", "following", "from", "since", "beginning", "commencing", "starting", "subsequent"}
_BEFORE_WORDS = {"before", "prior", "until", "till", "through", "thru", "by", "ending", "preceding", "up"}
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"}


def _date_side(cue_words: list) -> str | None:
    """"after" or "before", when the question's condition is one side of a date ("after July 31, 2017")."""
    first = {w.split(" ")[0] for w in cue_words}
    return "after" if first & {"after", "following"} else "before" if first & {"before", "prior", "until"} else None


def _date_side_in(raw: list, conds: list) -> str | None:
    """The side of the date the sentence's words put the condition's month on ("through July 31" -> before)."""
    months = {it.label.split(":")[1] for it in conds if it.label.split(":")[1] in _MONTHS | set(_NUMBER_WORDS)}
    for i, w in enumerate(raw):
        if w in months:
            near = raw[max(0, i - 4):i]
            if "later" in near and {"no", "not"} & set(raw[max(0, i - 6):i]) or "within" in near:
                return "before"  # "no later than fifteen days", "within thirty days"
            if set(near) & _AFTER_WORDS or "later" in near or "more" in near:
                return "after"
            if set(near) & _BEFORE_WORDS:
                return "before"
    return None


def _cue_kind(words: list) -> str:
    phrase = " ".join(words)
    for p in (phrase, re.sub(r"\s+(?:of|that)$", "", phrase), phrase.split(" ")[0] if phrase else ""):
        if p in _CUE_KINDS:
            return _CUE_KINDS[p]
    return phrase


def _conditions_met(frame: Frame, toks: list, tags: list, raw: list, clause: list | None = None) -> bool | str:
    """For a question with a condition: True when the clause's conditions are
    the question's ("upon a Change of Control"), False when the clause lacks
    it, a reason when the clause adds another condition. Judged within the
    action's clause (between ";" breaks), not the whole sentence."""
    if clause:
        lo, hi = clause[0].start, max(t.end if t.end > 0 else t.start + 1 for t in clause)
        toks, raw = toks[lo:hi], raw[lo:hi]
        tags = [Tag(t.kind, t.name, t.start - lo, (t.end - lo) if t.end > 0 else -1) for t in clause]
    norm = " " + " ".join(toks) + " "
    if not all(_item_present(it, raw, norm) for it in frame.conds):
        return False
    if "without" in frame.cues and not any(t.kind == "BLOCK" and _cue_kind(raw[t.start:t.end]) == "without"
                                           and any(_item_present(it, raw, norm, t.end, t.end + 7) for it in frame.conds)
                                           for t in tags):
        return _MISMATCH + "the question asks 'without ...'; the sentence doesn't say it happens without it"
    if (side := _date_side(frame.cue_words)) and (other := _date_side_in(raw, frame.conds)) and other != side:
        return _MISMATCH + f"the question asks {side} a date the sentence puts on the other side of it"
    if frame.only and not any(w in ONLY_WORDS for w in raw) and not (
            _EXCEPT_ONLY.search(" ".join(raw)) and any(w in ("not", "no", "nor") for w in raw)):
        return False  # "only X", or "not ... except X"
    # the question's condition stated under a cue of its own kind ("In the event that ... required ... to disclose"):
    # then another cue near one of its words ("notify ... prior to disclosure") is about something else
    covered = {it.label for t in tags if t.kind == "BLOCK" and _cue_kind(raw[t.start:t.end]) in frame.cues
               for it in frame.conds if _item_present(it, raw, norm, t.start, t.start + 16)}
    for t in tags:
        if t.kind != "BLOCK" or _upon_notice(t, tags):
            continue
        if frame.only and raw[t.start] in ONLY_WORDS | {"except", "save", "other", "unless"}:
            continue
        if frame.conds and any(_item_present(it, raw, norm, t.start, t.start + 16) for it in frame.conds):
            kind = _cue_kind(raw[t.start:t.end])
            # only a cue that governs the condition itself ("during the term", not "upon request ... or on termination")
            governed = {it.label for it in frame.conds if _item_present(it, raw, norm, t.end, t.end + 7)}
            direct = bool(governed - covered)
            if direct and frame.cues and set(frame.cues) - {"only"} and kind not in frame.cues and kind != "only":
                return _MISMATCH + f'the sentence puts it under "{" ".join(raw[t.start:t.end])}", the question under "{frame.cues[0]}"'
            continue
        return "the sentence has another condition"
    return True


# "a non-exclusive, non-transferable (except in accordance with Section 14.1) license": an exception that only
# points elsewhere doesn't undo the property the sentence states (2026-10-01; property questions only: "shall not
# assign (except as permitted under Section 14)" stays a condition on the "no").
_CROSS_REFERENCE = re.compile(r"\((?:except|other than|save|subject to|unless|but)\b[^()]{0,80}?\b(?:section|article|"
                              r"paragraph|clause|schedule|exhibit)s?\s*[\w.]+[^()]{0,40}\)", re.I)


_INSURANCE_WORDS = re.compile(r"\b(?:insur\w*|umbrella|general liability|excess liability|coverage|polic(?:y|ies)|"
                              r"underwriter|per occurrence)\b", re.I)


def _judge_property(frame: Frame, tags: list, sentence: str = ""):
    for clause in _clauses(tags):
        props = [t for t in clause if t.kind == "PROP" and t.name in frame.props]
        if not props or not all(_has_thing(th, clause) for th in frame.things):
            continue
        prop = props[0]
        if prop.name in ("CAPPED", "UNLIMITED") and _INSURANCE_WORDS.search(sentence):
            continue  # "Umbrella/Excess Liability with limits of $5,000,000": insurance limits, not a cap on liability
        # A condition after the property and its thing limits what the thing covers ("non-transferable license
        # to reproduce the Software only for installation"), not the property; one before it can undo it
        # ("Upon expiration, the licenses will become perpetual"). (2026-10-01)
        named = max([prop.start] + [next((t.start for t in clause if _has_thing(th, [t])), prop.start)
                                    for th in frame.things])
        if any(t.kind == "BLOCK" and t.start < named and not _upon_notice(t, tags) for t in tags):
            return "the sentence has a condition or exception"
        if any(t.kind == "MODAL" and t.name in ("NOT", "BAN") and 0 < prop.start - t.start <= 3 for t in clause):
            return "the property is negated"
        return ("yes", None)
    return None


# Every period ends a sentence, "Section 14.1" too: not splitting there (2026-10-01) lost more right answers on
# LegalBench dev (23) and held-out (25) than it gained (16, 10), as longer sentences carry more conditions.
_BOUNDARY = re.compile(r"[.!?\n]")
FEW_SENTENCES = 64  # below this, check candidate sentences one by one instead of rescanning
# A question whose words fill more sentences than this isn't one a single clause
# settles (and judging them all would break the ~10 ms budget on long documents).
MAX_CANDIDATES = 64


class Sentences:
    """The sentence around any position, from boundaries found once (one
    O(n) pass; each lookup is a binary search)."""

    def __init__(self, document: str):
        self.document = document
        self.ends = [m.start() for m in _BOUNDARY.finditer(document)]

    def index(self, pos: int) -> int:
        return bisect.bisect_left(self.ends, pos)

    def text(self, i: int) -> str:
        start = self.ends[i - 1] + 1 if i else 0
        end = self.ends[i] + 1 if i < len(self.ends) else len(self.document)
        return self.document[start:end].strip()

    def at(self, pos: int) -> tuple[int, str]:
        i = self.index(pos)
        return i, self.text(i)


def _phrase_forms(phrases) -> frozenset:
    """Prefilter forms: each whole phrase with its first and last word
    inflected ("make available" -> "made available"), hyphens as spaces
    ("non-transferable" -> "non transferable"); the document's lowercased
    copy gets the same treatment (see _prefilter_text)."""
    out = set()
    for phrase in phrases:
        words = re.findall(r"[a-z0-9']+", phrase.lower())
        if not words:
            continue
        # forms of the word and of its normalized base, so the prefilter is never narrower than the tokenizer
        firsts = _forms(words[0]) | _forms(normalize(words[0])) | {k for k, v in IRREGULAR.items() if v == words[0]}
        if len(words) == 1:
            out |= firsts
            continue
        lasts = _forms(words[-1]) | _forms(normalize(words[-1])) | {k for k, v in IRREGULAR.items() if v == words[-1]}
        out |= {" ".join([f, *words[1:-1], l]) for f in firsts for l in lasts}
    return frozenset(out)


_PREFILTER_TABLE = str.maketrans({"-": " ", "\n": " ", "\t": " ", "’": "'", "\u2011": " ", "\u2013": " "})


_LAST_INDEXED: list = [None, None, None]  # (document, Sentences, prefilter text): one question reads the document several times


def indexed(document: str) -> tuple:
    """The document's sentence index and prefilter text, computed once per document."""
    if _LAST_INDEXED[0] is not document:
        _LAST_INDEXED[:] = [document, Sentences(document), _prefilter_text(document)]
    return _LAST_INDEXED[1], _LAST_INDEXED[2]


def occurs_any(words, document: str) -> bool:
    """Whether any of the whole words occurs in the document (one automaton pass)."""
    return next(_occurs(frozenset(words), False, indexed(document)[1]), None) is not None


def _prefilter_text(document: str) -> str:
    """Lowercased, with hyphens and line breaks as spaces: same length, so
    offsets still map to the original's sentences."""
    return document.lower().translate(_PREFILTER_TABLE)


@functools.lru_cache(maxsize=512)
def _automaton(forms: frozenset):
    """An Aho-Corasick automaton (a trie with failure links, in C) over the
    forms: one pass over the document finds every occurrence of all of them."""
    a = ahocorasick.Automaton()
    for f in forms:
        a.add_word(f, len(f))
    a.make_automaton()
    return a


def _occurs(forms: frozenset, prefix: bool, text: str):
    """Start offsets of whole-word (or, with `prefix`, word-start) occurrences in lowercased `text`."""
    for end, n in _automaton(forms).iter(text):
        start = end - n + 1
        if start and text[start - 1].isalnum():
            continue
        if not prefix and end + 1 < len(text) and text[end + 1].isalnum():
            continue
        yield start


def _candidates(slots: list, sentences: Sentences, lowered: str) -> list:
    """Sentences holding every slot. While many sentences are left, a slot is
    found with one automaton pass over the document; once few are left, it is
    checked in those sentences only."""
    candidates = None
    for slot in slots:
        if candidates is not None and len(candidates) <= FEW_SENTENCES:
            candidates = {i for i in candidates if next(_occurs(*slot, _prefilter_text(sentences.text(i))), None) is not None}
        else:
            found = {sentences.index(pos) for pos in _occurs(*slot, lowered)}
            candidates = found if candidates is None else candidates & found
        if not candidates:
            return []
    return sorted(candidates)


def _action_forms(frame: Frame, lexicon: dict) -> frozenset:
    names = set(frame.actions or [frame.action])
    names |= {v for a in list(names) if a in COMPOSITE for v in COMPOSITE[a][0]}
    return _phrase_forms([p for a in names if not a.startswith(OPEN) for p in lexicon[("ACTION", a)]])


def _action_slot(frame: Frame, lexicon: dict):
    """The prefilter slot for the frame's action: its lexicon phrases, or a literal verb's stem as a word start."""
    names = frame.actions or [frame.action]
    if any(a.startswith(OPEN) for a in names):
        return (frozenset(a[len(OPEN):] for a in names if a.startswith(OPEN)), True)
    return (_action_forms(frame, lexicon), False)


def _actor_slot(frame: Frame, lexicon: dict):
    """The actor's surface forms, or None when any party will do."""
    if frame.actor in ("ANY", "ALL"):
        return None
    return (_phrase_forms(lexicon[("ACTOR", frame.actor)] + ["party", "parties"]), False)


def acting_parties(question: str, document: str, parties: list) -> set | None:
    """The parties that do the question's action in the document ("Licensee
    shall have the right ... to audit" -> {"licensee"}), or None when the
    question has no action or a sentence gives it to every party."""
    trie = _build_trie(_party_lexicon(parties))
    frame = question_frame(question, trie)
    if isinstance(frame, str) or frame.asks in ("PROPERTY", "EXISTS") or any(a.startswith(OPEN) for a in frame.actions):
        return None
    sentences, lowered = indexed(document)
    found = set()
    candidates = _candidates([(_action_forms(frame, LEXICON), False)], sentences, lowered)
    if len(candidates) > MAX_CANDIDATES:
        return None
    for i in candidates:
        for clause in _clauses(tag(tokens(sentences.text(i)), trie)):
            for action in (t for t in clause if t.kind == "ACTION" and t.name in frame.actions):
                actors = [t for t in clause if t.kind == "ACTOR" and t.start < action.start]
                if actors and actors[-1].name in ("ALL", "NONE"):
                    return None
                if actors and actors[-1].name not in ("WE", "USER", "ANY", "OTHER"):
                    found.add(actors[-1].name)
    return found


def answer(question: str, document: str, parties: list = ()) -> FrameResult:
    start = time.perf_counter()
    result = _answer(question, document, parties)
    result.ms = round((time.perf_counter() - start) * 1000, 2)
    return result


def _thing_slot(kind: str, name: str, lexicon: dict):
    if kind == "WORD":
        return (frozenset([name]), True)
    if kind == "ACTOR" and name in ("OTHER", "ANY"):
        return None
    narrower = [p for k, n in BROADER.items() if n == name for p in lexicon[("THING", k)]] if kind == "THING" else []
    return (_phrase_forms(lexicon[(kind, name)] + narrower), False)


def _item_slot(item: Item):
    """Prefilter forms for an item: its stems, and the first word of each phrase (as a prefix)."""
    forms = set(item.stems) | {ph.split()[0] for ph in item.phrases if ph}
    return (frozenset(f for f in forms if len(f) >= 3), True)


def _answer_presence(frame: Frame, fd: dict, document: str, trie: dict) -> FrameResult:
    sentences, lowered = indexed(document)
    found = set()
    for alt in frame.alts:
        slots = [_item_slot(it) for it in alt]
        if all(sl[0] for sl in slots):
            found |= set(_candidates(slots, sentences, lowered))
    if len(found) > MAX_CANDIDATES:
        return FrameResult(reason=f"{len(found)} sentences name it: too many for one clause to settle", frame=fd)
    hits, blocked = [], None
    for i in sorted(found):
        sentence = sentences.text(i)
        verdict = _judge_presence(frame, sentence, trie)
        if isinstance(verdict, str):
            blocked = blocked or (verdict, sentence)
        elif verdict:
            hits.append((sentence, verdict))
    if blocked:
        return FrameResult(reason=blocked[0], frame=fd, evidence=[{"text": blocked[1], "label": "blocked", "p": 0.0}])
    if not hits:
        return FrameResult(reason="no sentence has everything the question names", frame=fd)
    sentence, (_, qualifier) = next((h for h in hits if h[1][1]), hits[0])  # a condition is reported
    return FrameResult("yes", qualifier, reason="one sentence has everything the question names", frame=fd,
                       evidence=[{"text": sentence, "label": f"yes, {qualifier}" if qualifier else "yes", "p": 1.0}],
                       condition=condition_text(sentence, frame) if qualifier else "")


def _with_aliases(frame: Frame, aliases: list) -> Frame:
    """A presence frame's confidential-information item, also matched by the document's own names for it."""
    if not frame.alts:
        return frame
    extra = frozenset(" ".join(tokens(a)) for a in aliases)
    alts = [[Item(it.label, it.stems, it.phrases | extra) if it.label == "THING:CONFIDENTIAL_INFO" else it for it in alt]
            for alt in frame.alts]
    return replace(frame, alts=alts)


def _jsonable(o):
    """Sets as sorted lists. `asdict` leaves Item.stems/phrases frozen sets, and
    a frame travels to the page as JSON."""
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


_RECEIVER_WORDS = re.compile(r"\b(?:receiving\s+part|recipient|receiver)", re.I)
_DISCLOSER_WORDS = re.compile(r"\b(?:disclosing\s+part|discloser)", re.I)


_INFO_TERM = r"[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,3}"
_INFO_HEAD = re.compile(r"\b(?:Information|Material|Materials|Data|Know-?How|Documentation|Documents)$")
_SECRET = re.compile(r"confiden|proprietary|non-?public|secret", re.I)
_DEFINES = re.compile(rf"[\"“](?P<t>{_INFO_TERM})[\"”]\s*(?:shall\s+)?(?:means?|include[sd]?|refers?\s+to|is\s+defined\s+as|"
                      r"shall\s+have\s+the\s+meaning)\b")
_NAMES = re.compile(rf"\(\s*(?:(?:collectively|hereinafter|together|herein)[,\s]+(?:referred\s+to\s+as\s+|called\s+)?)?"
                    rf"(?:the\s+)?[\"“](?P<t>{_INFO_TERM})[\"”]\s*\)")


ALIAS_SCAN_CHARS = 20_000  # definitions come early; keeps long documents fast


def info_aliases(document: str, limit: int = ALIAS_SCAN_CHARS) -> list:
    """The document's own names for what it protects ("all such information (collectively, the "Evaluation
    Material")", '"Information" means any non-public information ...'), lowercased: in this document they are
    confidential information (2026-10-01, STATUS.md item 2: "Borrower shall use the Information solely ...")."""
    head = document[:limit]
    out = []
    for m in _DEFINES.finditer(head):
        t = m.group("t")
        if _INFO_HEAD.search(t) and _SECRET.search(head[m.end():m.end() + 300] + t):
            out.append(t)
    for m in _NAMES.finditer(head):
        t = m.group("t")
        if _INFO_HEAD.search(t) and _SECRET.search(head[max(0, m.start() - 300):m.start()] + t):
            out.append(t)
    known = {p for p in LEXICON[("THING", "CONFIDENTIAL_INFO")]}
    # a one-word name ("Information") only when the document keeps using it capitalized
    return sorted({t.lower() for t in out if t.lower() not in known and (" " in t or len(re.findall(rf"\b{t}\b", head)) >= 2)})


def _named_subject(question: str, document: str, parties) -> str | None:
    """The question's subject as a party ("Must the Agent pay ...?" -> "Agent") when the document names it so
    (capitalized, as a whole word) and it isn't a party or a lexicon actor already. Then a sentence answers only if
    the Agent is the one acting, not when it merely appears ("The Agent shall have received evidence of payment")."""
    subj = qshapes.subject(question)
    if not subj or subj.lower() in {p.lower() for p in parties}:
        return None
    node = _BASE_TRIE
    for t in tokens(subj):
        node = node.get(t, {})
    if node.get("$", ("",))[0] == "ACTOR":
        return None
    name = re.escape(subj)
    acts = re.compile(rf"(?<![\w-]){name}\s*(?:\([^)]{{0,60}}\)\s*)?,?\s+(?:\w+ly\s+)?(?:shall|will|may|must|agrees|"
                      r"hereby|can|cannot|covenants|undertakes|represents|warrants|acknowledges|is|are|has|have|does|"
                      r"will|would|should)\b")
    defined = re.compile(rf"[\"“]{name}[\"”]")
    return subj if acts.search(document) or defined.search(document) else None


def _answer(question: str, document: str, parties) -> FrameResult:
    question = qshapes.canonical(question)
    if subj := _named_subject(question, document, parties):
        parties = [*parties, subj]
    q_trie = _build_trie(_party_lexicon(parties)) if parties else _BASE_TRIE
    frame = question_frame(question, q_trie)
    if isinstance(frame, str):
        return FrameResult(reason=f"no frame: {frame}")
    if frame.actor == "RECEIVER" and not _RECEIVER_WORDS.search(document):
        # A one-way NDA can name its receiving party and never call it that ("The Contractor shall not disclose
        # Confidential Information..."): there the question asks what any party does (2026-10-01).
        frame = replace(frame, actor="ANY")
    if ("ACTOR", "DISCLOSER") in [tuple(t) for t in frame.things] and not _DISCLOSER_WORDS.search(document):
        # "notify the disclosing party" in an NDA that calls it "the Company": the other party
        frame = replace(frame, things=[("ACTOR", "OTHER") if tuple(t) == ("ACTOR", "DISCLOSER") else t for t in frame.things])
    fd = _jsonable(asdict(frame))
    # Sentences also name actors a question can't ("SpringCo shall ..."): they
    # count as parties when the question asks about any party.
    names = named_actors(document) if frame.actor in ("ANY", "ALL") else []
    lexicon = {**LEXICON, **_party_lexicon(parties), **_party_lexicon(names)}
    extra = _party_lexicon(list(parties) + names)
    trie = _build_trie(extra) if extra else _BASE_TRIE
    if any(tuple(t) == ("THING", "CONFIDENTIAL_INFO") for t in frame.things) or any(
            it.label == "THING:CONFIDENTIAL_INFO" for alt in frame.alts for it in alt):
        if aliases := info_aliases(document):
            lexicon[("THING", "CONFIDENTIAL_INFO")] = LEXICON[("THING", "CONFIDENTIAL_INFO")] + aliases
            frame = _with_aliases(frame, aliases)
            # first, so the document's own name wins a tie ("the Information" is its confidential information,
            # not just any information)
            trie = (_trie_of([(("THING", "CONFIDENTIAL_INFO"), aliases)]),) + (trie if isinstance(trie, tuple) else (trie,))
    # Only sentences holding some form of every slot are tagged. Literal words
    # and things are usually rarer than actions and actors, so they go first
    # and an empty intersection stops early.
    if frame.asks == "EXISTS":
        return _answer_presence(frame, fd, document, trie)
    ranked = []
    for kind, name in frame.things:
        slot = _thing_slot(kind, name, lexicon)
        if slot:
            ranked.append((0 if kind == "WORD" else 1, slot))
    if frame.asks == "PROPERTY":
        ranked.append((2, (_phrase_forms([p for n in frame.props for p in lexicon[("PROP", n)]]), False)))
    else:
        ranked.append((2, _action_slot(frame, lexicon)))
        if (actor := _actor_slot(frame, lexicon)) is not None:
            ranked.append((3, actor))
    slots = [slot for _, slot in sorted(ranked, key=lambda x: x[0])]
    hits, blocked = [], None
    sentences, lowered = indexed(document)
    candidates = _candidates(slots, sentences, lowered)
    if len(candidates) > MAX_CANDIDATES:
        return FrameResult(reason=f"{len(candidates)} sentences name it: too many for one clause to settle", frame=fd)
    for i in candidates:
        sentence = sentences.text(i)
        verdict = _judge(frame, sentence, trie)
        if isinstance(verdict, str):
            blocked = blocked or (verdict, sentence)
        elif verdict:
            hits.append((sentence, verdict))
    if blocked:
        return FrameResult(reason=blocked[0], frame=fd, evidence=[{"text": blocked[1], "label": "blocked", "p": 0.0}])
    if not hits:
        return FrameResult(reason="no sentence has the whole frame", frame=fd)
    if len({(v[0], _base_qualifier(v[1])) for _, v in hits}) > 1:
        return FrameResult(reason="sentences disagree", frame=fd,
                           evidence=[{"text": s, "label": v[0], "p": 1.0} for s, v in hits[:3]])
    sentence, (ans, qualifier) = next((h for h in hits if CONDITIONAL in (h[1][1] or "")), hits[0])
    if frame.asks != "PROPERTY":
        # A sentence where the same actor does the same action the other way, even
        # to something else, makes it depend: "shall not disclose to any third
        # party" vs "may disclose to its legal counsel".
        bare = Frame(frame.asks, frame.actor, frame.action, [], actions=frame.actions, conds=frame.conds, only=frame.only,
                     cues=frame.cues, cue_words=frame.cue_words)
        bare_slots = [_action_slot(frame, lexicon)]
        if (actor := _actor_slot(frame, lexicon)) is not None:
            bare_slots.append(actor)
        others = _candidates(bare_slots, sentences, lowered)
        if len(others) > MAX_CANDIDATES:
            return FrameResult(reason=f"{len(others)} sentences have this action: too many to rule out a contrary one",
                               frame=fd)
        for i in others:
            other = _judge(bare, sentences.text(i), trie)
            if isinstance(other, tuple) and other[0] != ans:
                return FrameResult(reason="another sentence says otherwise about the same action", frame=fd,
                                   evidence=[{"text": sentence, "label": ans, "p": 1.0},
                                             {"text": sentences.text(i), "label": other[0], "p": 1.0}])
    label = f"{ans}, {qualifier}" if qualifier else ans
    return FrameResult(ans, qualifier, reason="one sentence has the whole frame", frame=fd,
                       evidence=[{"text": sentence, "label": label, "p": 1.0}],
                       condition=condition_text(sentence, frame) if CONDITIONAL in (qualifier or "") else "")
