"""
Question shapes for Pre-Tier 0 (2026-10-01): rewrite the ways people ask a yes/no question into the one shape the
clause frames read (router/frames.py), "Aux Subject Verb Object [condition]", and find the question's subject.

    "Does the agreement require the Agent to pay the premiums?"  -> "Must the Agent pay the premiums?"
    "Does the lease allow the tenant to sublet?"                 -> "Can the tenant sublet?"
    "Does the NDA prohibit the recipient from copying it?"       -> "Is the recipient prohibited from copying it?"
    "If the buyer defaults, can the seller terminate?"           -> "Can the seller terminate if the buyer defaults?"
    "Can this Agreement be assigned by the Licensee?"            -> "Can the Licensee assigned this Agreement?"
    "Can this Agreement be assigned?"                            -> "Can a party assigned this Agreement?"
    "Is reverse engineering prohibited?"                         -> "Is a party prohibited from reverse engineering?"
(The frames read "assigned" as "assign": words are reduced to their base form before matching.)
Regular expressions only, so it costs microseconds; a question no shape fits is returned as it was.
"""
from __future__ import annotations

import re

_DOC = (r"(?:the |this |these |that |such )?(?:agreement|contract|clause|lease|licen[cs]e|policy|terms(?: of "
        r"(?:service|use))?|section|provision|document|nda|text|passage|amendment|plan|note|deed)s?")
_PREAMBLE = re.compile(r"^(?:(?:according to|under|per|based on|in|as (?:per|stated in))\s+" + _DOC + r"|generally speaking|"
                       r"in general|by default|legally speaking|in practice)\s*,\s*", re.I)
_LEAD_CONDITION = re.compile(r"^(?P<cue>if|when|whenever|once|after|before|upon|following|in the event(?: that)?|"
                             r"in case|unless|until|during|where|on|at)\s+(?P<cond>[^,?]{3,160}?)\s*,\s*(?P<main>.+)$", re.I)
_AUX = r"(?:is|are|was|were|am|do|does|did|can|could|will|would|shall|should|may|might|must|has|have)"
_REQUIRE = re.compile(rf"^(?:does|do|will|would|did)\s+{_DOC}\s+(?:require|requires|oblige|obliges|obligate|obligates|"
                      r"force|forces|compel|compels)\s+(?P<subj>.+?)\s+to\s+(?P<vp>.+)$", re.I)
_ALLOW = re.compile(rf"^(?:does|do|will|would|did)\s+{_DOC}\s+(?:allow|allows|permit|permits|entitle|entitles|"
                    r"authori[sz]e|authori[sz]es|enable|enables)\s+(?P<subj>.+?)\s+to\s+(?P<vp>.+)$", re.I)
_GIVE_RIGHT = re.compile(rf"^(?:does|do|will|would|did)\s+{_DOC}\s+(?:give|gives|grant|grants)\s+(?P<subj>.+?)\s+"
                         r"(?:the |a |any )?(?:right|option|ability|permission)\s+to\s+(?P<vp>.+)$", re.I)
_PROHIBIT = re.compile(rf"^(?:does|do|will|would|did)\s+{_DOC}\s+(?:prohibit|prohibits|forbid|forbids|bar|bars|"
                       r"prevent|prevents|preclude|precludes|restrict|restricts|stop|stops)\s+(?P<subj>.+?)\s+from\s+"
                       r"(?P<vp>.+)$", re.I)
_HAVE_TO = re.compile(r"^(?:does|do)\s+(?P<subj>.+?)\s+(?:have to|has to|need to|needs to|have an obligation to|"
                      r"have a duty to)\s+(?P<vp>.+)$", re.I)
_HAVE_RIGHT = re.compile(r"^(?:does|do)\s+(?P<subj>.+?)\s+(?:have|has|get|retain|keep)\s+(?:the |a |any )?"
                         r"(?:right|option|ability|permission)\s+to\s+(?P<vp>.+)$", re.I)
_ABLE = re.compile(rf"^(?P<aux>{_AUX})\s+(?P<subj>.+?)\s+(?:be\s+)?able\s+to\s+(?P<vp>.+)$", re.I)
# "Can this Agreement be assigned (by the Licensee) ...": the passive, with or without its agent
_PASSIVE = re.compile(r"^(?P<aux>can|could|may|might|must|shall|should|will|would)\s+(?P<obj>.+?)\s+be\s+(?:\w+ly\s+)?"
                      r"(?P<verb>[a-z]+(?:ed|en|wn|ld|nt|ade|ut|id|ung|ost))\b(?P<rest>.*)$", re.I)
_AGENT = re.compile(r"^(?P<pre>.*?)\bby\s+(?P<agent>(?:the |a |an |either |each |any |its |their |our |your )?"
                    r"(?:[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,3}|[a-z]+\s+part(?:y|ies)|[a-z]+(?:ee|or|er|ant)s?|"
                    r"we|us|you|me|them|him|her|it))(?P<post>\b.*)$")
# "Is reverse engineering prohibited?", "Is subletting allowed?", "Is assigning the lease required?"
_GERUND = re.compile(r"^(?:is|are)\s+(?P<ger>(?:(?!the\b|a\b|an\b|any\b|this\b|that\b|such\b)\w+\s+)?\w+ing\b"
                     r"(?:\s+(?!prohibited|forbidden|allowed|permitted|required|banned|barred)\w+){0,6})\s+"
                     r"(?P<mod>prohibited|forbidden|banned|barred|not allowed|not permitted|allowed|permitted|required)\b"
                     r"(?P<rest>.*)$", re.I)
_PASSIVE_NOT_VERBS = frozenset("seen been born then even often open garden token".split())


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


# Questions about the text itself, put the way the presence and topic readers take them (2026-10-01, from
# rewordings of the benchmark questions): "Is the governing law specified in the agreement?" -> "Does the agreement
# specify the governing law?", "Is the clause about X?" / "Does this provision cover X?" -> "Does ... discuss X?",
# "Is oral disclosure considered confidential information?" -> "Does confidential information include oral
# disclosure?".
_SPECIFIED_IN = re.compile(rf"^(?:is|are)\s+(?P<x>.+?)\s+(?:specified|stated|set out|set forth|spelled out|provided|"
                           rf"identified|indicated|mentioned|laid out|defined|described)\s+(?:in|by)\s+(?P<doc>{_DOC})$", re.I)
_ABOUT = re.compile(rf"^(?:is|are)\s+(?P<doc>{_DOC})\s+about\s+(?P<x>.+)$", re.I)
_COVERS = re.compile(rf"^(?:does|do)\s+(?P<doc>{_DOC})\s+(?:cover|concern|relate to|deal with|touch on|talk about)\s+(?P<x>.+)$", re.I)
_CONSIDERED = re.compile(r"^(?:is|are)\s+(?P<x>.+?)\s+(?:considered|deemed|treated as|regarded as|counted as)(?:\s+to\s+be)?"
                         r"\s+(?:part\s+of\s+)?(?P<y>(?:the\s+|any\s+)?(?:confidential|proprietary)\s+(?:information|material)s?)$",
                         re.I)
_INFO_COVERS = re.compile(r"^(?:does|do)\s+(?P<y>(?:the\s+)?(?:confidential|proprietary)\s+(?:information|material)s?)\s+"
                          r"(?:cover|encompass|extend to|take in)\s+(?P<x>.+)$", re.I)
# Noun-shaped questions put the action in a noun; put it back in a verb so the frames can check who does what
# (2026-10-02, from the passive/noun study on the open generated sets):
#   "Is there a requirement for notices to be in writing?"   -> "Must notices be in writing?"
#   "Is there a right for the Company to amend the Plan?"     -> "Can the Company amend the Plan?"
#   "Is there a prohibition on assignment of the agreement?"  -> "Is a party prohibited from assigning the agreement?"
#   "Is there a cap on the combined expenses?"               -> "Are the combined expenses capped?"
#   "Is assignment of this Agreement prohibited without consent?" -> "Is a party prohibited from assigning this ..."
_NEED_NOUN = re.compile(r"^(?:is|are)\s+there\s+(?:a|an|any)\s+(?:requirement|obligation|duty|need)\s+(?:(?:for|that|on)\s+"
                        r"(?P<subj>.+?)\s+(?:to|must|shall|will|should)\s+|to\s+)(?P<vp>.+)$", re.I)
_RIGHT_NOUN = re.compile(r"^(?:is|are)\s+there\s+(?:a|an|any)\s+(?:right|option|ability|permission)\s+(?:for\s+(?P<subj>.+?)\s+)?"
                         r"to\s+(?P<vp>.+)$", re.I)
_BAN_NOUN = re.compile(r"^(?:is|are)\s+there\s+(?:a|an|any)\s+(?:prohibition|ban|restriction|bar)\s+(?:on|against)\s+(?P<x>.+)$", re.I)
_CAP_NOUN = re.compile(r"^(?:is|are)\s+there\s+(?:a|an|any)\s+(?:cap|limit|ceiling|maximum|limitation)\s+(?:on|to|for)\s+(?P<x>.+)$",
                       re.I)
NOMINAL = {"assignment": "assign", "disclosure": "disclose", "termination": "terminate", "transfer": "transfer",
           "solicitation": "solicit", "use": "use", "reproduction": "reproduce", "modification": "modify",
           "renewal": "renew", "cancellation": "cancel", "payment": "pay", "sale": "sell", "sublicense": "sublicense",
           "sublease": "sublease", "competition": "compete", "audit": "audit", "inspection": "inspect",
           "amendment": "amend", "extension": "extend", "delivery": "deliver", "publication": "publish",
           "distribution": "distribute", "removal": "remove", "deletion": "delete", "retention": "retain",
           "return": "return", "destruction": "destroy", "indemnification": "indemnify", "notification": "notify",
           "collection": "collect", "storage": "store", "hiring": "hire", "employment": "employ", "purchase": "purchase"}
_NOMINAL_MOD = re.compile(r"^(?:is|are)\s+(?:the\s+|any\s+)?(?P<head>" + "|".join(NOMINAL) + r")\b(?:\s+of\s+(?P<obj>.+?))?\s+"
                          r"(?P<mod>prohibited|forbidden|banned|barred|not allowed|not permitted|allowed|permitted|required)"
                          r"\b(?P<rest>.*)$", re.I)


_REFRAIN = re.compile(r"^(?:must|shall|should|does|do|is|are)\s+(?P<subj>.+?)\s+(?:have to\s+|has to\s+|required to\s+|obligated to\s+)?"
                      r"(?:refrain|abstain)\s+from\s+(?P<vp>.+)$", re.I)
_SUBJECT_FORM = {"me": "I", "us": "we", "him": "he", "her": "she", "them": "they"}  # "a right for me to own" -> "Can I own"


def _gerund(verb: str) -> str:
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith("ee"):
        return verb[:-1] + "ing"
    if re.search(r"[^aeiou][aeiou][bdgmnprt]$", verb) and verb not in ("audit", "deliver", "cancel", "extend"):
        return verb + verb[-1] + "ing"  # transferring, permitting
    return verb + "ing"


def _verb_phrase(x: str) -> str:
    """"assignment of the agreement" -> "assigning the agreement"; a gerund or anything else stays."""
    m = re.match(r"^(?:the\s+|any\s+)?(?P<head>" + "|".join(NOMINAL) + r")\b(?:\s+of\s+(?P<obj>.+))?$", x, re.I)
    if not m:
        return x
    return f"{_gerund(NOMINAL[m.group('head').lower()])}{' ' + m.group('obj') if m.group('obj') else ''}"


_LIMIT_USE = re.compile(r"\b(?:limit|restrict|confine)\s+(?:the\s+|its\s+|their\s+|his\s+|her\s+)?use\s+of\s+(?P<x>.+?)\s+"
                        r"(?:to|for)\s+(?P<p>(?:the\s+)?(?:purpose|purposes|uses?)\b.*)$", re.I)


def canonical(question: str) -> str:
    """The question in the shape the clause frames read, or as it was."""
    q = " ".join(question.replace("’", "'").split()).rstrip(" ?.!")
    if not q:
        return question
    out = q
    if m := _NEED_NOUN.match(out):
        out = f"Must {_SUBJECT_FORM.get((m.group('subj') or '').lower(), m.group('subj') or 'a party')} {m.group('vp')}"
    elif m := _RIGHT_NOUN.match(out):
        out = f"Can {_SUBJECT_FORM.get((m.group('subj') or '').lower(), m.group('subj') or 'a party')} {m.group('vp')}"
    elif m := _BAN_NOUN.match(out):
        out = f"Is a party prohibited from {_verb_phrase(m.group('x'))}"
    elif m := _CAP_NOUN.match(out):
        x = m.group("x")
        out = f"{'Are' if re.search(r's\b', x.split(' of ')[0].split()[-1]) else 'Is'} {x} capped"
    elif m := _NOMINAL_MOD.match(out):
        verb = NOMINAL[m.group("head").lower()]
        obj = f" {m.group('obj')}" if m.group("obj") else ""
        mod = m.group("mod").lower()
        out = (f"Can a party {verb}{obj}{m.group('rest')}" if mod in ("allowed", "permitted") else
               f"Must a party {verb}{obj}{m.group('rest')}" if mod == "required" else
               f"Is a party prohibited from {_gerund(verb)}{obj}{m.group('rest')}")
    for rx, shape in ((_SPECIFIED_IN, "Does {doc} specify {x}"), (_ABOUT, "Does {doc} discuss {x}"),
                      (_COVERS, "Does {doc} discuss {x}"), (_CONSIDERED, "Does {y} include {x}"),
                      (_INFO_COVERS, "Does {y} include {x}")):
        if m := rx.match(out):
            out = shape.format(**m.groupdict())
            break
    for _ in range(2):  # a preamble, then a leading condition ("Under the lease, if I move out early, ...")
        out = _PREAMBLE.sub("", out)
        m = _LEAD_CONDITION.match(out)
        if m and re.match(rf"^{_AUX}\b", m.group("main"), re.I):
            cue = m.group("cue").lower()
            out = f"{m.group('main').rstrip(' ?')} {cue} {m.group('cond')}"
    if m := _REFRAIN.match(out):  # "Must the Participant refrain from disparaging ...?"
        out = f"Is {m.group('subj')} prohibited from {m.group('vp')}"
    for rx, lead in ((_REQUIRE, "Must"), (_ALLOW, "Can"), (_GIVE_RIGHT, "Can"), (_HAVE_TO, "Must"),
                     (_HAVE_RIGHT, "Can")):
        m = rx.match(out)
        if m:
            out = f"{lead} {m.group('subj')} {m.group('vp')}"
            break
    else:
        if m := _PROHIBIT.match(out):
            out = f"Is {m.group('subj')} prohibited from {m.group('vp')}"
        elif m := _ABLE.match(out):
            out = f"Can {m.group('subj')} {m.group('vp')}"
        elif m := _GERUND.match(out):
            mod = m.group("mod").lower()
            ger, rest = m.group("ger"), m.group("rest")
            if mod in ("allowed", "permitted"):
                out = f"Can a party {ger}{rest}"
            elif mod == "required":
                out = f"Must a party {ger}{rest}"
            else:
                out = f"Is a party prohibited from {ger}{rest}"
        elif (m := _PASSIVE.match(out)) and m.group("verb").lower() not in _PASSIVE_NOT_VERBS and (
                m.group("obj").lower() not in ("it", "they", "this", "that", "these", "those", "them")
                or re.search(r"\bby\b", m.group("rest"))):  # "can it be released?" has nothing to check; "by X" does
            obj, verb, rest = m.group("obj"), m.group("verb"), m.group("rest")
            a = _AGENT.match(rest)
            if a and len(a.group("pre").split()) <= 3:
                agent = a.group("agent")
                out = f"{_cap(m.group('aux'))} {agent} {verb} {obj}{(' ' + a.group('pre').strip()) if a.group('pre').strip() else ''}{a.group('post')}"
            elif not re.search(r"\bby\b", rest):
                out = f"{_cap(m.group('aux'))} a party {verb} {obj}{rest}"
    out = _LIMIT_USE.sub(lambda m: f"use {m.group('x')} only for {m.group('p')}", out)  # "limit the use of X to the purposes"
    # "... once the agreement ends": a condition ("once" alone isn't one in a contract: "no more than once per year")
    out = re.sub(r"\bonce\s+(?=(?:the|this|that|a|an|it|they|we|you|i|such|any|its|their|my|our)\b)", "when ", out)
    out = " ".join(out.split())
    return _cap(out) + "?" if out != q else question


_TWO_QUESTIONS = re.compile(r",\s*or\s+(?:is|are|can|may|must|does|do|will|would|should|not)\b", re.I)


def two_questions(question: str) -> bool:
    """"Can the receiving party disclose it, or is it prohibited?": two questions, which a yes or no can't answer."""
    return bool(_TWO_QUESTIONS.search(question))


_ARTICLES = frozenset("the a an any each every such this that".split())
_NOT_NAME = frozenset("i we you they it he she party parties either neither one both there agreement contract "
                      "lease section".split())


def subject(question: str) -> str | None:
    """The name right after the leading auxiliary of a canonical question when it names someone the way a contract
    names a party, capitalized: "Must the Agent pay ...?" -> "Agent", "Is the Secured Party required ...?" ->
    "Secured Party", "Can Macquarie ...?" -> "Macquarie". None for "the tenant", "a party", "it" and the like
    (frames.py knows those)."""
    words = question.strip().rstrip("?").split()
    if not words or not re.fullmatch(_AUX, words[0], re.I):
        return None
    i = 1
    while i < len(words) and words[i].lower() in _ARTICLES:
        i += 1
    name = []
    while i < len(words) and re.fullmatch(r"[A-Z][\w&.-]*", words[i]) and len(name) < 4:
        name.append(words[i])
        i += 1
    if not name or len(name) == 1 and name[0].lower() in _NOT_NAME or name[0] == "I":
        return None
    if name[-1].lower() in _DOC_NOUNS or i >= len(words):
        return None
    after = words[i].lower().strip(",")
    # what follows must be the verb's place: "Is the Agent required ...", "Can Macquarie terminate ...", not
    # "Is Delaware law the governing law?" or "Is this a Force Majeure clause?"
    if words[0].lower() in ("is", "are", "was", "were", "am"):
        return " ".join(name) if after in _PREDICATES else None
    return " ".join(name) if after not in _DOC_NOUNS and after not in _ARTICLES else None


_PREDICATES = frozenset("required allowed permitted entitled obligated obliged prohibited forbidden barred precluded "
                        "able responsible liable bound expected supposed authorized authorised free restricted not "
                        "also still ever expressly only subject".split())
_DOC_NOUNS = frozenset("agreement agreements contract contracts clause clauses provision provisions section sections "
                       "plan plans amendment amendments lease leases license licence note notes policy law laws court "
                       "courts state states rules rule act acts code regulations terms deed schedule exhibit "
                       "certificate".split())
