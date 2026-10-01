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
                       "reserves the right", "option", "discretion"],
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
    ("ACTION", "COPY"): ["copy", "make copies", "make a copy", "reproduce", "duplicate"],
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
    ("THING", "EXISTENCE"): ["existence of this agreement", "existence of the agreement", "fact that",
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
COMPOSITE = {"INSURE": ({"REPAIR", "CARRY", "COLLECT", "PURCHASE", "STORE", "SHARE"}, "INSURANCE")}
# A narrower thing answers for a broader one: sharing with advertisers is sharing with third parties.
BROADER = {"ADVERTISERS": "THIRD_PARTIES", "ANALYTICS": "THIRD_PARTIES", "SERVICE_PROVIDERS": "THIRD_PARTIES"}
IRREGULAR = {"paid": "pay", "sold": "sell", "kept": "keep", "held": "hold", "gave": "give", "given": "give",
             "made": "make", "sent": "send", "told": "tell", "shown": "show", "done": "do", "had": "have",
             "has": "have", "is": "is", "are": "are", "was": "is", "were": "are", "can't": "can't",
             "won't": "won't", "don't": "don't", "doesn't": "doesn't", "bought": "buy", "hired": "hire",
             "employed": "employ", "licensed": "license", "granted": "grant"}
QUESTION_AUX = {"do": "DOES", "does": "DOES", "did": "DOES", "will": "DOES", "is": "DOES", "are": "DOES",
                "can": "CAN", "may": "CAN", "could": "CAN", "must": "MUST", "shall": "MUST", "should": "MUST"}
STOPWORDS = frozenset("a an the of to in on for by with at from as and or 's this that these those its our their "
                      "your my any all be been being it upon per such no more than hereby herein hereunder "
                      "thereof hereof its own also further additionally".split())
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
    alts: list = field(default_factory=list)  # presence frame: alternatives, each a list of stem groups
    negative: bool = False  # presence frame: the question asks for a negation ("that no license is granted")
    require: bool = False  # presence frame: "does X require consent" ("not ... without consent" counts)
    when: bool = False  # presence frame: "specify when ...": the sentence must hold a date or duration
    strict: bool = False  # presence frame from a catch-all: any condition in the sentence defers
    dated: bool = False  # presence frame asking for a date or period ("the date on which it becomes effective")
    values: list = field(default_factory=list)  # numbers, amounts, months the question names: the sentence must have them all


@dataclass
class FrameResult:
    answer: str | None = None  # "yes" | "no" | None (defer)
    qualifier: str | None = None  # "may" when the text only permits it
    reason: str = ""
    frame: dict | None = None
    evidence: list = field(default_factory=list)
    ms: float = 0.0

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
    frame = _action_frame(question, trie)
    if isinstance(frame, str):
        presence = _presence_frame(question, trie)
        if presence is not None:
            frame = presence
    if not isinstance(frame, str):
        frame.values = sorted(values(question))
    return frame


def _action_frame(question: str, trie: dict) -> Frame | str:
    raw = question.lower()
    toks = tokens(question)
    if not toks or toks[0] not in QUESTION_AUX:
        return "doesn't start with an auxiliary"
    tags = tag(toks[1:], trie)
    if any(t.name in ("NOT", "NONE") for t in tags):
        return "negated question"
    # A condition in the question ("... if the other party undergoes a change
    # of control", "... only for the purposes of the agreement") is what the
    # clause's own condition must match; the frame is the part before it.
    conds, only = [], False
    blocks = [t for t in tags if t.kind == "BLOCK"]
    if blocks:
        b = blocks[0]
        only = toks[1:][b.start] in ONLY_WORDS
        conds = _content_items([t for t in tags if t.start > b.start and t.kind != "BLOCK"], trie)
        if not conds and not only:
            return "a condition with nothing to match"
        tags = [t for t in tags if t.start < b.start]
    actors = [t for t in tags if t.kind == "ACTOR"]
    actions = [t for t in tags if t.kind == "ACTION"]
    props = [t for t in tags if t.kind == "PROP"]
    if not actions and props and toks[0] in ("is", "are"):
        return _property_frame(raw, tags, props)
    if not actors or not actions or actors[0].start > actions[0].start:
        return "no actor before an action"
    toks1 = toks[1:]
    for a, b in zip(actions, actions[1:]):  # only "soliciting or hiring": adjacent, joined by "or"
        if toks1[a.start + 1:b.start] != ["or"]:
            return "more than one action (\"use and distribute\")"
    actor, action = actors[0], actions[0]
    between = [t for t in tags if actor.start < t.start < action.start]
    if any(t.kind == "WORD" and t.name != "have" for t in between):
        return "a word between the actor and the action isn't in the lexicon"
    asks = QUESTION_AUX[toks[0]]
    for t in between:  # "Is the tenant allowed to", "Does the tenant have to", "Is a party prohibited from"
        if t.kind == "MODAL":
            asks = {"MAY": "CAN", "MUST": "MUST", "BAN": "PROHIBITED"}.get(t.name, asks)
    last_action = actions[-1]
    things = [(t.kind, t.name) for t in tags
              if t.start > last_action.start and t.kind in ("THING", "WORD", "ACTOR")]
    if any(k == "PROP" for k, _ in ((t.kind, t.name) for t in tags if t.start > actor.start)):
        return "a property inside an action question"
    return Frame(asks, actor.name, action.name, things, actions=[a.name for a in actions], conds=conds, only=only)


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
    "specific particular given period").split())
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
                   r"require|requires|grant|grants|give|gives|allow for|provide for)")
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


def _content_items(tags: list, trie: dict, toks: list | None = None) -> list:
    """What a stretch of question must find in a clause: concepts (with their
    synonyms, and the question's own word) and literal words (as stems),
    minus the generic ones."""
    items = []
    for t in tags:
        # the question's own word, when the tag is that one word ("property"), not a phrase ("initial term")
        one_word = t.end in (-1, t.start + 1)
        own = _crude(toks[t.start]) if toks and t.start < len(toks) and one_word else ""
        if t.kind == "WORD":
            if t.name in _GENERIC or len(t.name) < 3 or t.name in _NEGATION_WORDS:
                continue
            items.append(Item(f"WORD:{t.name}", frozenset([t.name]), frozenset()))
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
    q_named = any(t.kind == "ACTOR" and (t.kind, t.name) not in _GENERIC_CONCEPTS for t in tag(tokens(q)[1:], trie))
    for i, pattern in enumerate(_PRESENCE_PATTERNS):
        if i >= 3 and q_named:
            return None
        m = pattern.match(q)
        if not m:
            continue
        rest = m.group("rest")
        verb = m.groupdict().get("verb") or ""
        strict = i >= 3  # "entitled/required to", "does a party have", catch-all
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
        require = verb.startswith("requir") or bool(re.search(r"\brequire[sd]?\b", rest))
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
        return Frame("EXISTS", None, None, [], alts=alts, negative=negative, require=require, when=when, strict=strict,
                     dated=dated)
    return None


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
    return t.name == "UPON" and any(u.kind == "WORD" and u.name.startswith("notic") and 0 < u.start - t.start <= 8 for u in tags)


def _blocked(tags: list) -> bool:
    return any(t.kind == "BLOCK" and not _upon_notice(t, tags) for t in tags)


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
    if frame.asks == "PROPERTY":
        return _judge_property(frame, tag(tokens(_CROSS_REFERENCE.sub(" ", sentence)), trie), sentence)
    toks = tokens(sentence)
    tags = tag(toks, trie)
    caps = _case_flags(sentence)
    for clause in _clauses(tags):
        wanted = set(frame.actions or [frame.action])
        composite = {v for a in wanted if a in COMPOSITE for v in COMPOSITE[a][0]
                     if any(t.kind == "THING" and t.name == COMPOSITE[a][1] for t in clause)}
        actions = [t for t in clause if t.kind == "ACTION" and (t.name in wanted or t.name in composite)]
        for action in actions:
            before = [t for t in clause if t.start < action.start]
            passive = bool(action.start and toks[action.start - 1] in _PASSIVE_AUX)
            if passive:
                # "This Agreement may be terminated ... by either party": the agent acts.
                actor = _agent(clause, action, toks, caps, frame)
                if actor is None:
                    continue
                if not _actor_matches(frame.actor, actor.name):
                    continue
                subject_start = max((t.start for t in clause if t.kind == "ACTION" and t.start < action.start), default=-1)
                # only the modal run right before the verb ("which may be maintained by X"),
                # not "shall not be limited" earlier in the sentence
                modals, k = set(), len(before) - 1
                while k >= 0 and (before[k].kind == "MODAL" or before[k].kind == "WORD" and before[k].name.endswith("ly")):
                    if before[k].kind == "MODAL":
                        modals.add(before[k].name)
                    k -= 1
                if not all(_has_thing(th, [t for t in clause if t is not actor]) for th in frame.things):
                    continue
                verdict = _verdict(frame, sentence, toks, tags, clause, actor, modals, subject_start)
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
            if any(t.kind == "ACTION" and t is not action for t in after_actor):
                continue
            if not all(_has_thing(th, [t for t in clause if t.start > action.start or th[0] == "ACTOR"])
                       for th in frame.things):
                continue
            modals = {t.name for t in before if t.kind == "MODAL" and t.start > actor.start}
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


def _verdict(frame, sentence, toks, tags, clause, actor, modals, prev):
    """The answer once actor, action and things are in place: conditions, then modality."""
    if frame.conds or frame.only:
        met = _conditions_met(frame, toks, tags, _raw_tokens(sentence), clause)
        if met is False:
            return None
        if isinstance(met, str):
            return met
    elif _blocked(tags):
        if ("NOT" in modals or "BAN" in modals) and _permitted_by_exception(frame, toks, clause, actor):
            return ("yes", "may" if frame.asks == "DOES" else None)
        return "the sentence has a condition or exception"
    if frame.only and ("NOT" in modals or "BAN" in modals) and any(t.kind == "BLOCK" for t in clause) \
            and _EXCEPT_ONLY.search(sentence):
        modals = modals - {"NOT", "BAN"}  # "agrees not to use ... other than for the purposes": only for them
    # "..., nor shall either party use": a "nor" after the previous action and
    # before this actor negates it. (Only "nor": in "not contributing with
    # coverage the Sponsor may carry", "not" is not the Sponsor's.)
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


def _judge_presence(frame: Frame, sentence: str, trie: dict):
    """"yes" when one alternative's items are all in the sentence with the
    question's polarity; a reason when present but negated; None otherwise."""
    raw = _raw_tokens(sentence)
    norm = " " + " ".join(normalize(w) for w in raw) + " "
    if not any(all(_item_present(it, raw, norm) for it in alt) for alt in frame.alts) and not (
            _about_the_term(frame) and _AGREEMENT_TERM.search(_UNLESS_EARLIER.sub(" ", sentence))):
        return None
    if frame.when and not _states_when(frame, sentence):
        return None
    if frame.dated and not _DATE_OR_DURATION.search(sentence):
        return None
    if frame.values and not set(frame.values) <= values(sentence):
        return None
    stags = tag([normalize(w) for w in raw], trie)
    if frame.strict and _blocked(stags):
        return "the sentence has a condition or exception"
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
    if frame.only and not any(w in ONLY_WORDS for w in raw) and not (
            _EXCEPT_ONLY.search(" ".join(raw)) and any(w in ("not", "no", "nor") for w in raw)):
        return False  # "only X", or "not ... except X"
    for t in tags:
        if t.kind != "BLOCK" or _upon_notice(t, tags):
            continue
        if frame.only and raw[t.start] in ONLY_WORDS | {"except", "save", "other", "unless"}:
            continue
        if frame.conds and any(_item_present(it, raw, norm, t.start, t.start + 16) for it in frame.conds):
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
    return _phrase_forms([p for a in names for p in lexicon[("ACTION", a)]])


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
    if isinstance(frame, str) or frame.asks in ("PROPERTY", "EXISTS"):
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
            hits.append(sentence)
    if blocked:
        return FrameResult(reason=blocked[0], frame=fd, evidence=[{"text": blocked[1], "label": "blocked", "p": 0.0}])
    if not hits:
        return FrameResult(reason="no sentence has everything the question names", frame=fd)
    return FrameResult("yes", None, reason="one sentence has everything the question names", frame=fd,
                       evidence=[{"text": hits[0], "label": "yes", "p": 1.0}])


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


def _answer(question: str, document: str, parties) -> FrameResult:
    q_trie = _build_trie(_party_lexicon(parties)) if parties else _BASE_TRIE
    frame = question_frame(question, q_trie)
    if isinstance(frame, str):
        return FrameResult(reason=f"no frame: {frame}")
    if frame.actor == "RECEIVER" and not _RECEIVER_WORDS.search(document):
        # A one-way NDA can name its receiving party and never call it that ("The Contractor shall not disclose
        # Confidential Information..."): there the question asks what any party does (2026-10-01).
        frame = replace(frame, actor="ANY")
    fd = _jsonable(asdict(frame))
    # Sentences also name actors a question can't ("SpringCo shall ..."): they
    # count as parties when the question asks about any party.
    names = named_actors(document) if frame.actor in ("ANY", "ALL") else []
    lexicon = {**LEXICON, **_party_lexicon(parties), **_party_lexicon(names)}
    trie = _build_trie(_party_lexicon(list(parties) + names)) if (parties or names) else _BASE_TRIE
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
        ranked.append((2, (_action_forms(frame, lexicon), False)))
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
    if len({v for _, v in hits}) > 1:
        return FrameResult(reason="sentences disagree", frame=fd,
                           evidence=[{"text": s, "label": v[0], "p": 1.0} for s, v in hits[:3]])
    sentence, (ans, qualifier) = hits[0]
    if frame.asks != "PROPERTY":
        # A sentence where the same actor does the same action the other way, even
        # to something else, makes it depend: "shall not disclose to any third
        # party" vs "may disclose to its legal counsel".
        bare = Frame(frame.asks, frame.actor, frame.action, [], actions=frame.actions, conds=frame.conds, only=frame.only)
        bare_slots = [(_action_forms(frame, lexicon), False)]
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
                       evidence=[{"text": sentence, "label": label, "p": 1.0}])
