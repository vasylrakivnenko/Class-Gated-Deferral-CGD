"""
Legal equivalences for Pre-Tier 0 (2026-10-01): a few fixed formulas contracts use where the words differ from the
question's but the meaning is settled, hand-written and checked one by one (STATUS.md, NEXT STEPS item 3):

    "Can the receiving party independently develop similar information?"
        <- "Confidential Information does not include information that ... is independently developed by the
           Receiving Party"  (a carve-out from what the agreement protects: developing it is not restricted)
    "Can the receiving party obtain similar information from third parties?"
        <- "... information that is lawfully obtained from a third party who has the right to disclose it"
    "Is the receiving party prohibited from disclosing the existence of the agreement?"
        <- "The existence and terms of this Agreement shall be kept confidential"
    "Do the confidentiality obligations survive termination of the agreement?"
        <- "The obligations of confidentiality shall survive the termination of this Agreement"
    "Does the agreement say that no license to the confidential information is granted?"
        <- "Nothing in this Agreement shall be construed as granting any rights or license ..."

A rule answers only "yes", only when the clause frames (router/frames.py) defer, and never from a sentence its veto
matches (the contrary wording: "may disclose the existence", "shall not survive", "shall not independently develop").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from router import frames

_PERMIT_Q = r"^(?:can|may|could|is|are|am|does|do)\b(?!.*\b(?:not|prohibited|forbidden|barred|prevented|restricted)\b)"
_SUBJECT_INFO = r"(?:information|data|material|materials|know-?how|technology|technologies|products?|ideas?|works?)"


@dataclass(frozen=True)
class Rule:
    name: str
    question: re.Pattern  # on the question in canonical shape, lowercased
    sentence: re.Pattern  # what the deciding sentence must say
    veto: re.Pattern | None  # a sentence saying this doesn't count
    reason: str
    also: re.Pattern | None = None  # with question `also_if`, the sentence must say this too
    also_if: re.Pattern | None = None
    cue: tuple = ()  # word starts that every matching sentence has: one Aho-Corasick pass finds the few worth the pattern


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.I)


RULES = [
    Rule("independent development",
         _rx(_PERMIT_Q + r"(?=.*\b(?:independently (?:develop|create|produce|generate|build|design|invent)|"
             r"(?:develop|create|produce|generate|build|design|invent|come up with)\w*\b(?:\W+\w+){0,6}?\W+(?:independently|"
             r"on (?:its|their|my|our|his|her|your) own|by (?:itself|themselves|myself|ourselves)|without (?:using|use of)))|"
             r"(?=.*\bindependent(?:ly)? develop))"
             r"(?=.*\b(?:similar|comparable|competing|same|such|like|related|equivalent)\b)"),
         _rx(r"\b(?:independently developed|developed independently|independently conceived|developed\b(?:\W+\w+){0,10}?"
             r"\W+independently|independent(?:ly)? (?:development|created|generated)|developed by (?:or for )?(?:the )?"
             r"(?:receiving party|recipient|you|it|them|\w+)\s+without (?:use of|reference to|access to|the use of|"
             r"using|reliance on|benefit of))"),
         _rx(r"\b(?:not|never|nor)\s+(?:\w+\s+){0,3}independently develop|\bdeemed (?:to be )?confidential\b|"
             r"\bshall not be deemed\b[^.;]{0,60}independently"),
         "the agreement leaves information developed independently outside what it protects", cue=("independen", "develop")),
    Rule("acquired from a third party",
         _rx(_PERMIT_Q + r"(?=.*\b(?:obtain|acquire|receive|get|gather|collect|learn|buy|purchase|access)\w*\b)"
             r"(?=.*\bfrom\s+(?:a\s+|any\s+|another\s+|other\s+)?(?:third[- ]part(?:y|ies)|sources?|someone else|others|"
             r"other (?:people|persons|companies)|outside))"
             r"(?=.*\b(?:similar|comparable|same|such|that|this|the|related|equivalent)\b)"),
         _rx(r"\b(?:received|obtained|acquired|learned|furnished|made available|becomes? (?:known|available)|"
             r"comes? into (?:its |the )?possession|(?:is|was|been) disclosed to (?:\w+\s+){0,3}?)\b(?:\W+\w+){0,12}?"
             r"\W+(?:from|by)\s+(?:a\s+|any\s+|another\s+)?(?:third[- ]part(?:y|ies)|source other than|person other than|"
             r"third persons?|other sources?|another source|independent source)"),
         _rx(r"\b(?:shall|will|must)\s+not\s+(?:\w+\s+){0,3}(?:obtain|acquire|seek|solicit)\b|\bdeemed (?:to be )?confidential\b"),
         "the agreement leaves information received from a third party outside what it protects", cue=("third", "source", "other than", "receiv", "obtain", "acquir", "learn", "furnish", "available", "possession")),
    Rule("existence kept confidential",
         _rx(r"(?=.*\b(?:prohibited|forbidden|barred|not allowed|not permitted|restricted|must not|shall not|"
             r"(?:keep|hold|treat)\b(?:\W+\w+){0,8}?\W+(?:confidential|secret|in confidence)|refrain)\b)"
             r"(?=.*\b(?:existence|fact that|the fact|terms of (?:this|the) (?:agreement|contract)|"
             r"(?:this|the|an|any) agreement (?:exists|was signed|has been|is in place)|negotiations|discussions)\b)"
             r"(?!.*\b(?:other than|except)\b)"),
         _rx(r"\b(?:existence|fact)\b[^.;]{0,160}?\b(?:shall|will|must|are|is|to)\s+(?:\w+\s+){0,2}?(?:be\s+)?"
             r"(?:kept|held|treated|maintained|remain)\s+(?:\w+\s+){0,3}?(?:confidential|in (?:strict )?confidence|secret)"
             r"|\b(?:shall|will|must|agrees? to|agrees? that it will)\s+(?:not\s+(?:\w+\s+){0,2}?(?:disclose|reveal|"
             r"divulge|announce|publici[sz]e|make public)|keep\s+(?:\w+\s+){0,3}?(?:confidential|secret)|hold\s+"
             r"(?:\w+\s+){0,3}?in (?:strict )?confidence)\b[^.;]{0,160}?\b(?:existence|the fact that|the terms of this "
             r"agreement|terms and conditions of this agreement)\b"
             r"|\b(?:existence|fact)\b[^.;]{0,160}?\b(?:shall|will|must|may)\s+not\s+(?:\w+\s+){0,2}?be\s+(?:disclosed|"
             r"revealed|divulged|announced|released|made public)"),
         _rx(r"\bmay\s+(?:\w+\s+){0,3}(?:disclose|notify|announce|publici[sz]e|reveal)\b[^.;]{0,80}\bexistence|"
             r"\bexistence\b[^.;]{0,80}\bmay be (?:disclosed|announced)"),
         "the agreement keeps the existence or terms of the agreement confidential", cue=("existence", "fact", "terms of this agreement", "terms and conditions of this agreement")),
    Rule("survival",
         _rx(r"^(?=.*\b(?:obligations?|duties|duty|confidentiality|non-?disclosure|restrictions?|covenants?|terms|"
             r"provisions?|indemnif\w*|indemnity|representations?|warranties|sections?|clauses?)\b)(?:do|does|will|shall|must)\s+(?:the\s+|any\s+|all\s+|its\s+|their\s+|some\s+)?"
             r"(?:(?!(?:when|if|after|once|while|whether|before|until|unless|we|i|you|they)\b)[\w'-]+\s+){0,7}?"
             r"(?:survive|survives|continue|continues|remain|remains|last|lasts|stay|stays)\b(?!.*\b(?:not|no|exactly|period|years?|months?|days?|until|how long|"
             r"limit|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\b)(?!.*\d)"
             r"(?=.*\b(?:terminat\w*|expir\w*|end\w*|cancel\w*)\b)"),
         _rx(r"\bsurviv\w*\b(?:\W+\w+){0,14}?\W+(?:the\s+|any\s+|such\s+)?(?:terminat\w*|expir\w*|cancell?ation|end)\b"
             r"|\b(?:terminat\w*|expir\w*)\b[^.;]{0,100}?\bsurviv\w*"
             r"|\b(?:continue|remain)s?\b(?:\W+\w+){0,6}?\W+(?:in (?:full )?(?:force|effect)|binding|effective|applicable|"
             r"in place)\b(?:\W+\w+){0,12}?\W+(?:after|following|beyond|notwithstanding|despite)\s+(?:\w+\s+){0,3}?"
             r"(?:the\s+)?(?:terminat|expir)"),
         _rx(r"\b(?:shall|will|do|does)\s+not\s+surviv|\bno\b[^.;]{0,40}\bsurviv|\bsurviv\w*\s+(?:only\s+)?until\b"),
         "the agreement says these obligations survive its end",
         cue=("surviv", "continu", "remain"),
         also=_rx(r"\b(?:confiden\w*|non-?disclosure|nondisclosure|secrecy|proprietary|obligations?\b[^.;]{0,30}?\b"
                  r"(?:hereunder|under this agreement|of this agreement|herein)|(?:this|the) agreement shall survive)"),
         also_if=_rx(r"\bconfiden|\bnon-?disclosure")),
    Rule("changes in writing",
         # "Must any amendment be in writing?" <- "No provision may be amended ... except in a writing signed by ..."
         # only the question itself, whole: anything more ("without ...", "within 30 days", "for the Lender",
         # "every future amendment") is a detail the formula doesn't settle
         _rx(r"^(?:must|shall|do|does|is|are)\s+(?:any\s+|all\s+)?(?:amendments?|modifications?|changes?|waivers?)"
             r"(?:\s+(?:to|of)\s+(?:this|the)\s+(?:agreement|contract))?\s+(?:need\s+to\s+|have\s+to\s+|required\s+to\s+)?"
             r"be\s+(?:made\s+)?in\s+writing(?:\s+(?:and\s+)?signed(?:\s+by\s+(?:both|all|the)\s+parties)?)?\??$"),
         _rx(r"\b(?:no|not|nor)\b[^.;]{0,160}?\b(?:amend\w*|modif\w*|waive[sd]?|waiver|chang\w*|supplement\w*|alter\w*)"
             r"\b[^.;]{0,160}?\b(?:unless|except|other than|save)\b[^.;]{0,40}?\b(?:in writing|by (?:a |an )?(?:written|writing)|"
             r"in a writing|by a writing|by an instrument in writing|by (?:a |an )?(?:further |subsequent )?(?:written )?"
             r"(?:agreement|instrument),? in writing)"
             r"|\b(?:amend\w*|modif\w*|waive[sd]?|waiver|chang\w*|supplement\w*)\b[^.;]{0,100}?\b(?:shall|must|will)\s+"
             r"(?:only\s+)?be\s+(?:made\s+|effective\s+only\s+(?:if|when)\s+)?in writing"),
         _rx(r"\b(?:orally|oral)\b[^.;]{0,40}\b(?:may|can)\b"),
         "the agreement lets it be changed or waived only in writing",
         also=_rx(r"\b(?:sign\w*|execut\w*)\b"), also_if=_rx(r"\b(?:sign\w*|execut\w*)\b"),
         cue=("amend", "modif", "waive", "waiver", "chang", "supplement", "alter")),
    Rule("exhibits are part of it",
         # "Must the Exhibits be considered part of the agreement?" <- "The Exhibits constitute a part hereof"
         _rx(r"^(?:are|is|must|do|does)\s+(?:the\s+)?(?:exhibits?|schedules?|annex(?:es)?|appendi(?:x|ces)|attachments?|"
             r"recitals?)(?:\s+(?:to|of)\s+(?:this|the)\s+(?:agreement|contract|amendment))?\s+(?:be\s+)?(?:considered\s+|deemed\s+|"
             r"treated\s+as\s+)?(?:(?:a|an)\s+(?:integral\s+)?part\s+of\s+(?:the|this)\s+(?:agreement|contract|amendment)|"
             r"incorporated(?:\s+(?:into|in)\s+(?:the|this)\s+(?:agreement|contract|amendment))?(?:\s+by\s+reference)?)\??$"),
         _rx(r"\b(?:exhibits?|schedules?|annex(?:es)?|appendi(?:x|ces)|attachments?|recitals?)\b[^.;]{0,120}?\b(?:constitute|"
             r"form|are|is|shall be|be deemed|deemed)\s+(?:to be\s+)?(?:a\s+|an\s+)?(?:integral\s+)?part\b"
             r"|\b(?:exhibits?|schedules?|annex(?:es)?|appendi(?:x|ces)|attachments?|recitals?)\b[^.;]{0,120}?\bincorporated\b"),
         _rx(r"\bnot\s+(?:be\s+)?(?:deemed\s+)?(?:a\s+)?part\b|\bnot\s+incorporated\b"),
         "the agreement makes them part of it",
         cue=("exhibit", "schedule", "annex", "appendi", "attachment", "recital")),
    Rule("no license",
         _rx(r"(?=.*\b(?:no|not|never|nothing)\b(?:\W+\w+){0,6}?\W+(?:licen[cs]e|rights?)\b)"
             r"(?=.*\b(?:grant\w*|give|given|gives|confer\w*|transfer\w*|provid\w*)\b)"),
         _rx(r"\bno\s+(?:\w+\s+){0,4}?(?:licen[cs]es?|rights?)\b[^.;]{0,120}?\b(?:grant|confer|impl|transfer|convey|"
             r"creat|giv|vest)\w*"
             r"|\b(?:does|do|shall|will)\s+not\s+(?:\w+\s+){0,2}?(?:grant|confer|transfer|convey|give|create)\w*\s+"
             r"(?:\w+\s+){0,6}?(?:licen[cs]es?|rights?)\b"
             r"|\bnot\s+to\s+(?:grant|confer|transfer|convey|give|create)\w*\s+(?:\w+\s+){0,6}?(?:licen[cs]es?|rights?)\b"
             r"|\b(?:neither|no)\s+part(?:y|ies)\s+(?:\w+\s+){0,2}?(?:acquires?|obtains?|receives?|gets?|shall acquire|"
             r"will acquire)\s+(?:any\s+)?(?:\w+\s+){0,2}?(?:licen[cs]es?|rights?)\b"
             r"|\bnothing\b[^.;]{0,100}?\b(?:constitutes?|creates?|implies|confers?|grants?|gives?|transfers?)\s+"
             r"(?:to\s+\w+\s+)?(?:a\s+|any\s+)?(?:licen[cs]es?|rights?|title|interest)\b"
             r"|\bnothing\b[^.;]{0,120}?\b(?:construed|deemed|interpreted|intended)\b[^.;]{0,30}?\b(?:as|to)\s+"
             r"(?:\w+\s+){0,2}?(?:grant\w*|confer\w*|transfer\w*|convey\w*|creat\w*|giv\w*)\b[^.;]{0,200}?\b"
             r"(?:licen[cs]e|rights?|title|interest)\b"),
         _rx(r"\bhereby grants?\b|\bis hereby granted\b|\bgrants? (?:to )?(?:the )?(?:receiving party|recipient)\s+(?:a|an)\s+"
             r"(?:\w+\s+){0,3}licen|\bsublicen"),
         "the agreement says it grants no license or rights", cue=("license ", "license,", "license.", "licenses", "licence ", "licence,", "licence.", "licences", "nothing", "neither part", "no right", "no other right", "no rights", "not grant", "not confer", "not transfer", "not convey", "not give", "not create", "not vest", "not imply", "not to grant", "not to confer", "not be construed", "licensing", "licensed")),
]


_OWNED = re.compile(r"\b([a-z][\w-]*?)(?:'s|s')\s+(?:[\w-]+\s+){0,3}?(?:obligations?|duties|covenants?|representations|"
                    r"agreements|undertakings|commitments)\b|\b(?:obligations?|duties|covenants?)\s+of\s+(?:the\s+)?([a-z][\w-]*)")


def _owner_named(q: str, sentence: str) -> bool:
    """"Do the Lenders' obligations ... continue?" needs the Lenders' obligations, not the Borrower's."""
    m = _OWNED.search(q)
    owner = m and (m.group(1) or m.group(2))
    if not owner or owner in ("party", "parties", "agreement", "receiving", "disclosing", "each", "either", "both"):
        return True
    o = re.escape(owner.rstrip("s"))
    return bool(re.search(rf"\b{o}\w*(?:'s|s'|')?\s+(?:[\w-]+\s+){{0,3}}?(?:obligations?|duties|covenants?|representations|"
                          rf"agreements|undertakings|commitments)\b|\b(?:obligations?|duties|covenants?|representations|"
                          rf"warranties)\s+of\s+(?:the\s+)?{o}", sentence, re.I))


def answer(question: str, document: str) -> dict | None:
    """{"answer": "yes", "rule", "reason", "evidence"} when a rule's formula settles the question, else None."""
    q = " ".join(frames.qshapes.canonical(question).lower().replace("’", "'").split())
    for rule in RULES:
        if not rule.question.search(q):
            continue
        sentences, lowered = frames.indexed(document)
        seen = set()
        for pos in frames._occurs(frozenset(rule.cue), True, lowered):
            i, sentence = sentences.at(pos)
            if i in seen:
                continue
            seen.add(i)
            sentence = sentence.replace("’", "'")
            if not rule.sentence.search(sentence):
                continue
            if rule.veto is not None and rule.veto.search(sentence):
                continue
            if rule.also is not None and (rule.also_if is None or rule.also_if.search(q)) and not rule.also.search(sentence):
                continue
            if not _owner_named(q, sentence.replace("’", "'")):
                continue
            return {"answer": "yes", "rule": rule.name, "reason": rule.reason,
                    "evidence": [{"text": sentence, "label": "yes", "p": 1.0}]}
    return None
