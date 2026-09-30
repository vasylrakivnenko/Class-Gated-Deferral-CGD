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
import re
import time
from dataclasses import asdict, dataclass, field

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
    ("ACTOR", "RECEIVER"): ["receiving party", "the receiving party", "recipient", "the recipient", "receiving parties"],
    ("ACTOR", "DISCLOSER"): ["disclosing party", "the disclosing party", "discloser", "the discloser"],
    # modality; precedence NOT > MAY > MUST > DOES
    ("MODAL", "NOT"): ["not", "never", "no right", "cannot", "can't", "won't", "don't", "doesn't", "shan't",
                       "may in no event", "in no event", "exempt", "excused", "relieved", "released from", "waive",
                       "fail", "under no circumstances", "nor"],
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
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


_LEMMA = {}
for _phrases in LEXICON.values():
    for _phrase in _phrases:
        for _word in _phrase.split():
            for _form in _forms(_word):
                # "disclosing" is a form of "disclose" and of itself ("disclosing party"): the shortest base wins
                if _form not in _LEMMA or len(_word) < len(_LEMMA[_form]):
                    _LEMMA[_form] = _word
_LEMMA.update(IRREGULAR)


def normalize(token: str) -> str:
    return _LEMMA.get(token) or _crude(token)


def tokens(text: str) -> list:
    text = re.sub(r"'s\b", "", text.lower().replace("’", "'"))  # "Licensor's books" -> licensor books
    return [normalize(t) for t in _TOKEN.findall(text)]


def _build_trie(extra: dict) -> dict:
    trie = {}
    for concept, phrases in list(LEXICON.items()) + list(extra.items()):
        for phrase in phrases:
            node = trie
            for word in tokens(phrase):
                node = node.setdefault(word, {})
            node.setdefault("$", concept)
    return trie


_BASE_TRIE = _build_trie({})


@dataclass
class Tag:
    kind: str  # ACTOR | MODAL | ACTION | THING | PROP | BLOCK | WORD | BREAK
    name: str
    start: int  # token index


def tag(toks: list, trie: dict) -> list:
    """Longest-match concept tags; uncovered content words become WORD tags;
    NEUTRAL boilerplate is dropped."""
    out, i = [], 0
    while i < len(toks):
        node, j, best = trie, i, None
        while j < len(toks) and toks[j] in node:
            node = node[toks[j]]
            j += 1
            if "$" in node:
                best = (node["$"], j)
        if best:
            (kind, name), end = best
            if kind != "NEUTRAL":
                out.append(Tag(kind, name, i))
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
    """The question's frame, or a reason it has none."""
    raw = question.lower()
    toks = tokens(question)
    if not toks or toks[0] not in QUESTION_AUX:
        return "doesn't start with an auxiliary"
    tags = tag(toks[1:], trie)
    if any(t.kind == "BLOCK" or t.name in ("NOT", "NONE") for t in tags):
        return "negated or conditional question"
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
    return Frame(asks, actor.name, action.name, things, actions=[a.name for a in actions])


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
    return t.name == "UPON" and any(u.kind == "WORD" and u.name == "notice" and 0 < u.start - t.start <= 8 for u in tags)


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
    toks = tokens(sentence)
    tags = tag(toks, trie)
    if frame.asks == "PROPERTY":
        return _judge_property(frame, tags)
    for clause in _clauses(tags):
        wanted = set(frame.actions or [frame.action])
        composite = {v for a in wanted if a in COMPOSITE for v in COMPOSITE[a][0]
                     if any(t.kind == "THING" and t.name == COMPOSITE[a][1] for t in clause)}
        actions = [t for t in clause if t.kind == "ACTION" and (t.name in wanted or t.name in composite)]
        for action in actions:
            before = [t for t in clause if t.start < action.start]
            actors = [t for t in before if t.kind == "ACTOR"]
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
            if action.start and toks[action.start - 1] in _PASSIVE_AUX:
                continue
            if not all(_has_thing(th, [t for t in clause if t.start > action.start or th[0] == "ACTOR"])
                       for th in frame.things):
                continue
            if _blocked(tags):
                return "the sentence has a condition or exception"
            modals = {t.name for t in before if t.kind == "MODAL" and t.start > actor.start}
            # "..., nor shall either party use": a "nor" after the previous
            # action and before this actor negates it. (Only "nor": in "not
            # contributing with coverage the Sponsor may carry", "not" is not the Sponsor's.)
            prev = max((t.start for t in clause if t.kind == "ACTION" and t.start < actor.start), default=-1)
            negated = (actor.name == "NONE" or "NOT" in modals or "BAN" in modals
                       or any(t.name == "NEITHER" and t.start < actor.start for t in clause)
                       or any(t.kind == "MODAL" and toks[t.start] == "nor" and prev < t.start < actor.start for t in clause))
            if frame.asks == "PROHIBITED":
                if negated:
                    return ("yes", None)
                if "MAY" in modals:
                    return ("no", None)
                return "the text says it happens, not whether it is prohibited"
            if negated:
                return ("no", None)
            if "MAY" in modals:
                if frame.asks == "MUST":
                    return "the text permits it but doesn't require it"
                return ("yes", "may" if frame.asks == "DOES" else None)
            return ("yes", None)
    return None


def _judge_property(frame: Frame, tags: list):
    for clause in _clauses(tags):
        props = [t for t in clause if t.kind == "PROP" and t.name in frame.props]
        if not props or not all(_has_thing(th, clause) for th in frame.things):
            continue
        if _blocked(tags):
            return "the sentence has a condition or exception"
        prop = props[0]
        if any(t.kind == "MODAL" and t.name in ("NOT", "BAN") and 0 < prop.start - t.start <= 3 for t in clause):
            return "the property is negated"
        return ("yes", None)
    return None


_BOUNDARY = re.compile(r"[.!?\n]")
FEW_SENTENCES = 64  # below this, check candidate sentences one by one instead of rescanning


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
        firsts = _forms(words[0]) | {k for k, v in IRREGULAR.items() if v == words[0]}
        if len(words) == 1:
            out |= firsts
            continue
        lasts = _forms(words[-1]) | {k for k, v in IRREGULAR.items() if v == words[-1]}
        out |= {" ".join([f, *words[1:-1], l]) for f in firsts for l in lasts}
    return frozenset(out)


_PREFILTER_TABLE = str.maketrans({"-": " ", "\n": " ", "\t": " ", "’": "'", "\u2011": " ", "\u2013": " "})


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
    if isinstance(frame, str) or frame.asks == "PROPERTY":
        return None
    sentences, lowered = Sentences(document), _prefilter_text(document)
    found = set()
    for i in _candidates([(_action_forms(frame, LEXICON), False)], sentences, lowered):
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


def _answer(question: str, document: str, parties) -> FrameResult:
    q_trie = _build_trie(_party_lexicon(parties)) if parties else _BASE_TRIE
    frame = question_frame(question, q_trie)
    if isinstance(frame, str):
        return FrameResult(reason=f"no frame: {frame}")
    fd = asdict(frame)
    # Sentences also name actors a question can't ("SpringCo shall ..."): they
    # count as parties when the question asks about any party.
    names = named_actors(document) if frame.actor in ("ANY", "ALL") else []
    lexicon = {**LEXICON, **_party_lexicon(parties), **_party_lexicon(names)}
    trie = _build_trie(_party_lexicon(list(parties) + names)) if (parties or names) else _BASE_TRIE
    # Only sentences holding some form of every slot are tagged. Literal words
    # and things are usually rarer than actions and actors, so they go first
    # and an empty intersection stops early.
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
    sentences, lowered = Sentences(document), _prefilter_text(document)
    for i in _candidates(slots, sentences, lowered):
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
        bare = Frame(frame.asks, frame.actor, frame.action, [], actions=frame.actions)
        bare_slots = [(_action_forms(frame, lexicon), False)]
        if (actor := _actor_slot(frame, lexicon)) is not None:
            bare_slots.append(actor)
        for i in _candidates(bare_slots, sentences, lowered):
            other = _judge(bare, sentences.text(i), trie)
            if isinstance(other, tuple) and other[0] != ans:
                return FrameResult(reason="another sentence says otherwise about the same action", frame=fd,
                                   evidence=[{"text": sentence, "label": ans, "p": 1.0},
                                             {"text": sentences.text(i), "label": other[0], "p": 1.0}])
    label = f"{ans}, {qualifier}" if qualifier else ans
    return FrameResult(ans, qualifier, reason="one sentence has the whole frame", frame=fd,
                       evidence=[{"text": sentence, "label": label, "p": 1.0}])
