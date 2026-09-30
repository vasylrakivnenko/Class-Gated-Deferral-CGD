"""The question tree: which kind of question this is, decided before any
answering (plan and milestones: router/STATUS.md).

Ordered tests, first match wins, so requests and requests for advice leave
before a lookup handler can see them:
    1. request     "Can you summarize the lease?"            -> not a question about the text
    2. judgmental  "Is the non-compete enforceable?"          -> advice or a legal conclusion
    3. branch      lookup ("Can the tenant sublet?") or reasoning ("Why...?")
    4. leaf        boolean / choice / span / backward / forward / process

Deterministic: one spaCy parse plus the cue lists in router/qtree_cues.py.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from router import qtree_cues as cues

_POLAR_START = re.compile(r"^(?:is|are|was|were|am|do|does|did|can|could|will|would|shall|should|may|might|must|"
                          r"has|have|had)\b", re.I)
_WH_START = re.compile(r"^(?:who|whom|whose|what|when|where|which|why|how)\b", re.I)


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner"])


@dataclass
class QuestionFrame:
    question: str  # as asked
    form: str  # polar | wh | other
    branch: str  # exit | lookup | reasoning
    leaf: str  # request | judgmental | boolean | choice | span | backward | forward | process | unclassified
    lehnert: str  # Lehnert's category, for logs only
    lookup: str = ""  # the question the answering tiers see: unwrapped ("Can you tell me whether X?" -> X)
    cue: str | None = None  # the words that decided an exit leaf ("enforceable", "should")
    wh: str | None = None
    answer_type: str | None = None  # span: DATE, DURATION, MONEY, PARTY, JURISDICTION, ...
    options: list = field(default_factory=list)  # choice: the alternatives, as written
    slots: dict = field(default_factory=dict)  # QA-SRL-style: aux, subject, verb, object, complement, preps
    negated: bool = False
    trace: list = field(default_factory=list)  # the rule that decided at each test

    def to_dict(self) -> dict:
        return asdict(self)


def request_cue(question: str, doc=None) -> str | None:
    """The task verb when the question asks for a task ("Can you list the
    deadlines?", "Summarize clause 4"); None for a question about the text,
    including one wrapped in a request ("Can you tell me whether...")."""
    q = question.strip()
    if cues.EMBEDDED.match(q):
        return None
    doc = doc if doc is not None else _nlp()(q)
    words = [t for t in doc if not t.is_punct]
    if not words:
        return None
    low = [t.lower_ for t in words]
    i = 1 if low[0] == "please" else 0
    # "Can/could/would/will you <task verb> ..."
    if len(low) > i + 2 and low[i] in ("can", "could", "would", "will") and low[i + 1] == "you":
        j = i + 2 + (low[i + 2] == "please")
        if j < len(low) and low[j] in cues.TASK_VERBS:
            return words[j].text
    # An imperative: "Summarize the lease.", "Please list the deadlines.", "Highlight every clause..."
    # (spaCy often tags a capitalized first verb as a noun, so the word decides, not its tag.)
    if i < len(low) and (low[i] in cues.TASK_VERBS or low[i] in cues.IMPERATIVE_ONLY) \
            and not _POLAR_START.match(low[i]) and not _WH_START.match(low[i]):
        return words[i].text
    return None


def _in_condition(low: str, start: int) -> bool:
    """Whether position `start` of the question is inside an if/unless/as...
    clause: "notify it if it is required by law to disclose"."""
    return bool(cues.CONDITION_BEFORE.search(low[:start]))


def _should_is_advice(doc) -> bool:
    """ "Should the tenant pay?" and "What should I do?" ask for advice; "Where
    should payments be sent?" asks what the text says."""
    if cues.WH_LEAD.match(doc.text) and not re.search(r"\bwhat (?:should|shall) (?:i|we) do\b", doc.text.lower()):
        return False  # "Who should I contact to file a complaint?" asks for a fact of the text
    for t in doc:
        if t.lower_ != "should":
            continue
        if t.i == 0:
            return True
        subj = next((c for c in t.head.children if c.dep_ in ("nsubj", "nsubjpass", "expl")), None)
        if subj is not None and subj.lower_ in cues.ADVICE_SUBJECTS:
            return True
    return False


def judgment_cue(question: str, doc=None) -> str | None:
    """The words that make the question ask for advice or a legal conclusion
    rather than what the text says; None for a lookup."""
    low = question.lower()
    doc = doc if doc is not None else _nlp()(question)
    for name, rx in cues.ADVICE + cues.LAW_CUES:
        m = rx.search(low)
        if m is None or (name in cues.CONDITION_EXEMPT and _in_condition(low, m.start())):
            continue
        if name == "should" and not _should_is_advice(doc):
            continue
        return name
    for t in doc:
        lemma = t.lemma_.lower()
        if t.lower_ in cues.LEGAL_ADVERBS and not _in_condition(low, t.idx):
            return t.text  # "Can the landlord legally enter?"
        if lemma in cues.EVALUATIVE and t.pos_ == "ADJ" and _appraises(t, low):
            return t.text
    if (m := cues.ENOUGH.search(low)) is not None and not _in_condition(low, m.start()):
        return m.group(0).split()[-2] + " enough" if "enough" in m.group(0) else "worth"
    m = cues.COPULA_APPRAISAL.search(low)
    deontic = [x.start() for x in (cues.OBLIGATION.search(low), cues.PERMISSION.search(low)) if x]
    if m and not any(d < m.start(1) for d in deontic) \
            and not (m.group(1) == "valid" and _valid_as_stated(low, doc)) and not _in_condition(low, m.start(1)):
        tok = next((t for t in doc if t.idx == m.start(1)), None)
        if tok is None or not any(a.lemma_.lower() in cues.MAKE_SO or _conditional(a)
                                  for a in [tok.head, *tok.ancestors]):
            return m.group(1)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is not None:
        verb = root.lemma_.lower()
        if verb in cues.CONCLUSION_VERBS:
            return root.text  # "Can they enforce the non-compete?", "Did the landlord breach the lease?"
        if verb == "comply" and _about_the_law(root):
            return root.text  # "Does the policy comply with GDPR?"
    return None


def _subject_of(adj):
    """The subject an adjective is predicated of: "Is [the fee] reasonable?",
    "Is [it] legal for X to Y?"."""
    head = adj.head
    for t in head.children:
        if t.dep_ in ("nsubj", "nsubjpass", "expl"):
            return t
    return None


def _appraises(adj, low: str) -> bool:
    """Whether an evaluative adjective is an appraisal the question asks for."""
    if adj.lemma_.lower() == "valid" and _valid_as_stated(low, adj.doc):
        return False
    if any(a.lemma_.lower() in cues.MAKE_SO or a.lower_ in cues.MAKE_SO for a in [adj.head, *adj.ancestors]):
        return False  # "How is my data kept safe?"
    if any(_conditional(a) for a in adj.ancestors) or _in_condition(low, adj.idx):
        return False  # "Can the landlord terminate if the tenant's conduct is unreasonable?"
    while adj.dep_ == "conj" and adj.head.pos_ == "ADJ":
        adj = adj.head  # "good or bad habit data": "bad" is what "good" is, a modifier
    if adj.dep_ == "amod" and adj.head.i > adj.i:
        # Only "Is this a fair contract?": a modifier of what "this/the clause" is.
        # (After its noun, "amod" is a misparse of the predicate: "Is the
        # limitation of liability valid?")
        noun = adj.head
        if noun.dep_ != "attr":
            return False
        subj = _subject_of(noun)
        return subj is not None and subj.lemma_.lower() in cues.DOCUMENT_NOUNS
    if adj.dep_ not in ("acomp", "attr", "oprd", "ROOT", "ccomp", "xcomp", "conj", "amod"):
        return False
    ob = cues.OBLIGATION.search(low)
    if ob and ob.start() < adj.idx:
        return False  # "Must the fee be reasonable?" (but "Is it legal for them to require...?" is a judgment)
    pm = cues.PERMISSION.search(low)
    if pm and pm.start() < adj.idx:
        subj = _subject_of(adj)
        return not (subj is not None and (subj.lemma_.lower() in cues.PARTY_NOUNS or subj.pos_ == "PROPN"))
    return True


def _valid_as_stated(low: str, doc) -> bool:
    """ "valid for 30 days" and "tickets are valid if bought from us" are terms
    the text states; "is the clause valid if...?" is still a legal conclusion."""
    if cues.VALID_FOR.search(low):
        return True
    if not cues.VALID_IF.search(low):
        return False
    subj = next((t for t in doc if t.dep_ in ("nsubj", "nsubjpass")), None)
    words = {t.lower_ for t in subj.subtree} | {t.lemma_.lower() for t in subj.subtree} if subj is not None else set()
    return not (words & (cues.DOCUMENT_NOUNS | cues.LEGAL_INSTRUMENTS))


def _conditional(tok) -> bool:
    """An if/unless/when clause (not every "advcl": spaCy also gives that to
    "the clause letting them use my photos")."""
    return tok.dep_ == "advcl" and any(c.dep_ == "mark" and c.lower_ in cues.CONDITION_MARKS for c in tok.children)


def _about_the_law(verb) -> bool:
    words = {t.lemma_.lower() for t in verb.subtree}
    subj = next((t for t in verb.children if t.dep_ in ("nsubj", "nsubjpass")), None)
    return bool(words & cues.LAW_WORDS) or (subj is not None and subj.lemma_.lower() in cues.DOCUMENT_NOUNS)




# ---- M1: the rest of the tree ---------------------------------------------------

_BE_HAVE_DO = {"be", "have", "do"}
_OPPOSITE_SET = set(cues.OPPOSITES)


def unwrap(question: str) -> str | None:
    """The question inside a polite wrapper, as a direct question: "Can you
    tell me whether the tenant can sublet?" -> "Can the tenant sublet?",
    "I want to know when rent is due" -> "When is rent due?". None if there's
    no wrapper."""
    m = cues.EMBEDDED.match(question.strip())
    if not m:
        return None
    clause = (m.group("q") or m.group("q2")).strip().rstrip("?.! ")
    first = clause.split()[0].lower()
    try:
        if first in ("whether", "if"):
            return _invert(clause.split(None, 1)[1] if " " in clause else "", "")
        return _invert_wh(clause)
    except Exception:
        return clause[:1].upper() + clause[1:] + "?"


def _finite_aux(tokens):
    """The first finite auxiliary or copula: "can", "is", "must", "has"."""
    for t in tokens:
        if t.tag_ == "MD" or (t.lemma_.lower() in _BE_HAVE_DO and t.pos_ == "AUX" and t.tag_ in ("VBZ", "VBP", "VBD")):
            return t
    return None


def _invert(clause: str, lead: str) -> str:
    """A declarative clause as a question, after `lead` (a wh-phrase or ""):
    "the tenant can sublet" -> "Can the tenant sublet?"; "the tenant pays rent"
    -> "Does the tenant pay rent?"."""
    doc = _nlp()(clause)
    aux = _finite_aux(doc)
    words = [t.text_with_ws for t in doc]
    if aux is not None:
        front = aux.text
        rest = "".join(w for i, w in enumerate(words) if i != aux.i).strip()
    else:
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if root is None or root.pos_ != "VERB":
            return (lead + " " + clause).strip().capitalize() + "?"
        front = {"VBZ": "does", "VBD": "did"}.get(root.tag_, "do")
        rest = "".join((root.lemma_ + root.whitespace_) if i == root.i else w for i, w in enumerate(words)).strip()
    out = f"{lead} {front} {rest}".strip() if lead else f"{front} {rest}"
    return out[:1].upper() + out[1:] + "?"


def _invert_wh(clause: str) -> str:
    """ "when rent is due" -> "When is rent due?"; "who pays for repairs" ->
    "Who pays for repairs?" (the wh-phrase is the subject: no inversion)."""
    doc = _nlp()(clause)
    wh = doc[0]
    end = wh.i
    if wh.dep_ in ("det", "poss", "advmod", "amod", "npadvmod") and wh.head.i > wh.i \
            and wh.head.pos_ in ("NOUN", "ADJ", "ADV", "PROPN", "NUM"):
        end = wh.head.i  # "which state's law", "how much", "what kind"
        while end + 1 < len(doc) and doc[end + 1].pos_ in ("NOUN", "PROPN", "ADP", "PART") and doc[end + 1].dep_ in (
                "compound", "prep", "pobj", "case", "poss"):
            end += 1  # "what kind of data"
    lead = doc[:end + 1].text
    rest = doc[end + 1:]
    subj_first = len(rest) and rest[0].pos_ in ("VERB", "AUX")
    if subj_first:
        out = f"{lead} {rest.text}"
        return out[:1].upper() + out[1:] + "?"
    return _invert(rest.text, lead)


def _normalized(question: str) -> str:
    q = cues.OPENERS.sub("", " ".join(question.split()))
    q = re.sub(r"([?!.])[?!.]+", r"\1", q)  # "??" is one question
    q = cues.WH_CONTRACTIONS.sub(lambda m: m.group(1) + " is", q)
    first = q.split(" ", 1)
    if first and first[0].lower() in cues.WH_TYPOS:
        q = cues.WH_TYPOS[first[0].lower()] + (" " + first[1] if len(first) > 1 else "")
    return q


def _main_clause(q: str) -> str:
    """The question after a leading condition or time clause: "If I breach it,
    what are the consequences?" -> "what are the consequences?"; "if i delete
    the app how long is my data stored" -> "how long is my data stored". The
    whole question stays the lookup: the condition matters for the answer."""
    if not cues.SUBORDINATE_LEAD.match(q) or cues.WH_AUX_NEXT.match(q):
        return q
    if "," in q:
        rest = q.split(",", 1)[1].strip()
        return _main_clause(rest) if rest else q
    words = q.split()
    for k in range(2, len(words)):
        w = words[k].lower().strip(",")
        if cues.WH_LEAD.match(w) and w not in ("that",):
            return " ".join(words[k:])
    for k in range(3, len(words)):
        if cues.POLAR_LEAD.match(words[k]) and not cues.POLAR_LEAD.match(words[k - 1]):
            return " ".join(words[k:])
    return q


def _form(q: str, doc) -> tuple[str, str | None]:
    m = cues.WH_LEAD.match(q)
    if m:
        return "wh", m.group(1).lower()
    if cues.POLAR_LEAD.match(q):
        return "polar", None
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
    if q.rstrip().endswith("?") and (has_verb or q.lower().startswith(("any ", "no "))):
        return "polar", None  # "Tenant allowed to sublet?", "Any late fees?"
    if has_verb and len(q.split()) >= 3 and not re.match(r"^(?:i|i'm|im|we|we're|my|our|me)\b", q, re.I):
        return "polar", None  # a statement to confirm: "this app shows my watching history."
    if not has_verb and 1 <= len(q.split()) <= 6:
        return "fragment", None  # "Audit rights?", "Late fee?", "Governing law?"
    return "other", None


def _span_type(q: str, wh: str, doc) -> str:
    low = q.lower()
    if wh == "how":
        words = low.split()
        i = next((k for k, w in enumerate(words) if w == "how"), 0)
        nxt = words[i + 1].strip("?,.") if i + 1 < len(words) else ""
        typ = cues.HOW_QUANT.get(nxt, "QUANTITY")
        after = words[i + 2].strip("?,.'’") if i + 2 < len(words) else ""
        if nxt in ("much", "many") and (after in cues.TIME_UNITS or after in ("notice", "time", "advance", "lead")):
            typ = "DURATION"  # "How many days' notice", "How much notice"
        elif nxt in ("much", "many") and after and after in cues.NOUN_TYPE and cues.NOUN_TYPE[after] != "DATE":
            typ = cues.NOUN_TYPE[after] if nxt == "much" or cues.NOUN_TYPE[after] != "MONEY" else typ
        return typ
    if wh in cues.WH_TYPES:
        return cues.WH_TYPES[wh]
    if re.search(r"\bwhat (?:does|do|is meant by|is the meaning of)\b.*\bmean\b|\bwhat is meant\b|\bdefin", low):
        return "DEFINITION"
    wh_tok = next((t for t in doc if t.lower_ == wh), None)
    noun = None
    if wh_tok is not None and wh_tok.dep_ in ("det", "poss", "amod") and wh_tok.head.pos_ in ("NOUN", "PROPN"):
        noun = wh_tok.head
    elif wh_tok is not None:
        # "What is the late fee?": the noun the copula links to "what"
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if root is not None and root.lemma_ == "be":
            noun = next((c for c in root.children if c.dep_ in ("nsubj", "attr") and c.i != wh_tok.i
                         and c.pos_ in ("NOUN", "PROPN")), None)
    if re.search(r"\b(?:interest|apr|uptime|percentage|percent|share|split)\b", low):
        return "PERCENT"  # "What interest rate applies?", "What uptime is guaranteed?"
    if re.search(r"\b(?:minimum|maximum|required|legal)\s+age\b|\bage\s+(?:limit|requirement)\b", low):
        return "AGE"  # "What is the minimum age to sign up?"
    if re.search(r"\bnumber of\b", low):
        return "CARDINAL"  # "What's the maximum number of guests allowed?"
    if re.search(r"\btime limit\b", low):
        return "DURATION"
    if noun is not None:
        for cand in [noun.lemma_.lower(), noun.lower_] + [c.lower_ for c in noun.children if c.dep_ == "compound"]:
            if cand in cues.NOUN_TYPE:
                return cues.NOUN_TYPE[cand]
    if re.search(r"\bwhat\b[^?]*\b(?:pay|pays|paid|cost|costs|charge|charged|owe|owes|spend)\b", low):
        return "MONEY"  # "What does the customer have to pay for onboarding?"
    for t in doc:  # any typed noun in the question: "the one-time implementation fee"
        if t.pos_ in ("NOUN", "PROPN") and t.i > 0 and (t.lemma_.lower() in cues.NOUN_TYPE or t.lower_ in cues.NOUN_TYPE):
            typ = cues.NOUN_TYPE.get(t.lemma_.lower()) or cues.NOUN_TYPE.get(t.lower_)
            if typ not in ("PARTY", "LOCATION"):
                return typ
    return "ENTITY"


def _trim_options(alt: list) -> list:
    """Parallel options: "the bonus guaranteed" / "discretionary" -> "guaranteed";
    "go to arbitration" / "to court" -> "to arbitration". Names stay whole ("New York")."""
    if len(alt) != 2:
        return alt
    out = []
    for i, text in enumerate(alt):
        other = alt[1 - i]
        toks = list(_nlp()(text))
        n_other = len(other.split())
        while len(toks) > n_other and len(toks) > 1 and toks[0].pos_ in ("DET", "AUX", "VERB", "NOUN", "PRON", "PART") \
                and not (toks[0].pos_ == "NOUN" and toks[1].pos_ in ("NOUN", "PROPN")) and not toks[0].like_num:
            toks = toks[1:]
        out.append("".join(t.text_with_ws for t in toks).strip())
    return out


def _options(q: str, doc, form: str, wh: str | None) -> list:
    """The alternatives of a choice question; [] when "or" joins things either
    of which answers yes ("Is the license irrevocable or perpetual?", "Must a
    party share revenue or profits?")."""
    low = q.lower()
    if " or " not in low or re.search(r"\bor not\s*[?.!]*$", low):
        return []  # "Is termination possible, or not?" is a yes/no question
    before, after = low.split(" or ", 1)
    or_tok = next((t for t in doc if t.lower_ == "or" and t.dep_ == "cc"), None)
    left = or_tok.head if or_tok is not None else None
    right = next((c for c in left.children if c.dep_ == "conj" and c.i > or_tok.i), None) if left is not None else None
    if left is not None and right is not None:
        # Each option is its conjunct's whole phrase: "in court" / "in arbitration", not "in" / "in".
        a_idx = [t.i for t in left.subtree if t.i < or_tok.i]
        b_idx = [t.i for t in right.subtree]
        alt = [doc[min(a_idx):max(a_idx) + 1].text, doc[min(b_idx):max(b_idx) + 1].text.rstrip("?.! ")]
    else:
        alt = [" ".join(before.split()[-3:]), " ".join(after.split()[:3]).rstrip("?.!")]
    if cues.PRESENCE_LEAD.match(low) and not cues.CLAUSE_ALTERNATIVE.search(low):
        return []  # "Is there a clause requiring arbitration or mediation?"
    if re.search(r"\bor not \w", low):
        return alt  # "Does the company sell Personal Data or not sell Personal Data?"
    if _parallel_values(alt, doc):
        return alt  # "New York law or Delaware law", "three months ... or six months", "once a year or twice a year"
    if cues.CLAUSE_ALTERNATIVE.search(low) or cues.TAIL_ALTERNATIVES.search(low):
        return alt  # "or only", "or does the vendor", "..., me or the insurer?", "or both"
    if left is not None and right is not None and left.dep_ in ("nsubj", "nsubjpass") \
            and not re.search(r"\beither\b", before):
        return alt  # "Does the client or the freelancer pay?": which of two parties
    # After "can/may/must" an "or" usually offers either ("Can either party terminate for
    # breach or for convenience?"); elsewhere a repeated preposition marks alternatives.
    modal = re.match(r"^(?:can|could|may|might|must|shall|should)\b", low)
    if (cues.REPEATED_PREP.search(low) or cues.REPEATED_MARK.search(low)) and not modal:
        return alt  # "Does title pass at shipment or at delivery?"
    value = lambda t: t.tag_ == "CD" or t.pos_ == "PROPN" or re.fullmatch(r"\d+(?:st|nd|rd|th)", t.lower_) \
        or t.lower_ in cues.FREQUENCY_WORDS or t.lemma_.lower() in cues.PARTY_ROLE_NOUNS
    if left is not None and right is not None and value(left) and value(right):
        return alt  # "the tenant or the landlord", "New York or Delaware", "the 1st or the 5th"
    if cues.NUMBER_WORD.search(before[-25:]) and cues.NUMBER_WORD.search(after.split("?")[0][:25]) \
            and not re.search(r"\bone or (?:more|two)\b", low):
        return alt  # "30 days or 60 days", "net 30 or net 60", "one year or two years", "one month's rent or two"
    freq = lambda chunk: {w for w in re.findall(r"[a-z]+", chunk)} & cues.FREQUENCY_WORDS
    fa, fb = freq(" ".join(before.split()[-3:])), freq(" ".join(after.split()[:3]))
    if (fa and fb and fa != fb) or (fa and re.match(r"per \w+", after)) or (fb and re.search(r"per \w+$", before)):
        return alt  # "monthly or annual", "monthly or per milestone"
    lex = lambda chunk: {w[:-2] if w.endswith("ly") and len(w) > 5 else w for w in
                         set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", chunk)) | set(re.findall(r"[a-z0-9]+", chunk))}
    wa = lex(" ".join(before.split()[-4:]))
    wb = lex(" ".join(after.split()[:4]))
    if left is not None and right is not None:
        wa |= {t.lemma_.lower() for t in doc[left.left_edge.i:left.i + 1]}
        wb |= {t.lemma_.lower() for t in doc[right.left_edge.i:right.i + 1]}
    negation = lambda x, y: y in ("non" + x, "non-" + x, "un" + x, "in" + x, "ir" + x, "dis" + x)
    # Opposites across the two options: "a flat amount or a percentage", "opt-in or opt-out".
    if any(x != y and (frozenset((x, y)) in _OPPOSITE_SET or negation(x, y) or negation(y, x)) for x in wa for y in wb):
        return alt
    return []


def _parallel_values(alt: list, doc) -> bool:
    """Two options that differ in a value (a number, a name, a frequency or period) or in
    being negated: alternatives even after "can/must" ("Can Buyer inspect once a year or
    twice a year?"). "first refusal or first offer" differs in nouns only: not values."""
    if len(alt) != 2:
        return False
    a, b = alt[0].lower(), alt[1].lower()
    wa, wb = set(re.findall(r"[\w$%.-]+", a)), set(re.findall(r"[\w$%.-]+", b))
    num = lambda ws: {w for w in ws if re.fullmatch(r"\$?\d[\d,.]*%?|\d+(?:st|nd|rd|th)", w) or w in (
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
        "eighteen nineteen twenty thirty forty forty-five fifty sixty seventy eighty ninety hundred thousand "
        "million").split()}
    if num(wa) and num(wb) and num(wa) != num(wb):
        return True
    diff_a, diff_b = wa - wb, wb - wa
    if (diff_a & cues.VALUE_WORDS) and (diff_b & cues.VALUE_WORDS):
        return True
    names = {t.lower_ for t in doc if t.pos_ == "PROPN"}
    if (wa & wb) and (diff_a & names) and (diff_b & names):
        return True  # "New York law or Delaware law"
    negation = lambda x, y: y in ("non" + x, "non-" + x, "un" + x, "in" + x, "ir" + x, "dis" + x)
    return any(negation(x, y) or negation(y, x) for x in wa for y in wb if len(x) > 3 and len(y) > 3)


def _slots(doc) -> dict:
    """QA-SRL-style slots (He et al., 2015): aux, subject, verb, object, complement, preps."""
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return {}
    verb = root
    if root.lemma_ in _BE_HAVE_DO and root.pos_ == "AUX":
        verb = next((c for c in root.children if c.pos_ == "VERB" and c.dep_ in ("xcomp", "ccomp", "acomp")), root)
    sub = lambda t: doc[t.left_edge.i:t.right_edge.i + 1].text
    aux = next((t for t in doc if t.tag_ == "MD" or (t.dep_ in ("aux", "auxpass") and t.i < verb.i)), None)
    subj = next((c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass", "expl")), None) or \
        next((c for c in root.children if c.dep_ in ("nsubj", "nsubjpass", "expl")), None)
    obj = next((c for c in verb.children if c.dep_ in ("dobj", "dative")), None)
    comp = next((c for c in root.children if c.dep_ in ("acomp", "attr", "oprd")), None)
    return {"aux": aux.lower_ if aux is not None else None, "subject": sub(subj) if subj is not None else None,
            "verb": verb.lemma_.lower(), "object": sub(obj) if obj is not None else None,
            "complement": sub(comp) if comp is not None else None,
            "preps": [sub(c) for c in verb.children if c.dep_ in ("prep", "agent")]}


_LEHNERT = {"boolean": "verification", "choice": "disjunctive", "forward": "causal consequent",
            "process": "instrumental/procedural", "request": "request", "judgmental": "judgmental",
            "unclassified": ""}


def classify(question: str) -> QuestionFrame:
    """Which kind of question this is. Ordered tests, first match wins."""
    q = " ".join(question.split())
    doc = _nlp()(q)
    trace = []
    if (cue := request_cue(q, doc)) is not None:
        trace.append(f"request: task verb '{cue}'")
        return QuestionFrame(q, "other", "exit", "request", "request", lookup="", cue=cue, trace=trace)
    trace.append("request: no")
    inner = unwrap(q)
    lq = _normalized(inner or q)
    if inner:
        trace.append(f"unwrapped: {inner}")
    ldoc = doc if lq == q else _nlp()(lq)
    mq = _main_clause(lq)  # the form is the main clause's; the lookup keeps any condition
    if mq != lq:
        trace.append(f"main clause: {mq}")
    mdoc = ldoc if mq == lq else _nlp()(mq)
    negated = any(t.dep_ == "neg" for t in mdoc)
    form, wh = _form(mq, mdoc)
    base = dict(lookup=lq, wh=wh, negated=negated, slots=_slots(mdoc))
    if (cue := judgment_cue(lq, ldoc) or (judgment_cue(q, doc) if inner else None)) is not None:
        trace.append(f"judgmental: '{cue}'")
        return QuestionFrame(q, form, "exit", "judgmental", "judgmental", cue=cue, trace=trace, **base)
    trace.append("judgmental: no")
    if cues.SECOND_QUESTION.search(lq.lower()):
        trace.append("more than one question")
        return QuestionFrame(q, form, "exit", "unclassified", "", trace=trace, **base)
    low = mq.lower()
    trace.append(f"form: {form}" + (f" ({wh})" if wh else ""))
    if form == "fragment":
        # A bare noun phrase: its head noun's answer type makes it a fact question
        # ("Late fee?" -> MONEY); otherwise it asks whether the text covers it ("Audit rights?").
        head = next((t for t in mdoc if t.dep_ == "ROOT"), None)
        typ = cues.NOUN_TYPE.get(head.lemma_.lower(), cues.NOUN_TYPE.get(head.lower_)) if head is not None else None
        if typ:
            trace.append(f"fragment: span {typ}")
            return QuestionFrame(q, form, "lookup", "span", "concept completion", answer_type=typ, trace=trace,
                                 **base)
        trace.append("fragment: does the text cover it (boolean)")
        return QuestionFrame(q, form, "lookup", "boolean", "verification", trace=trace, **base)
    if form == "other":
        return QuestionFrame(q, form, "exit", "unclassified", "", trace=trace, **base)
    if form == "wh":
        if wh == "why" or re.match(r"^how come\b", low):
            lehnert = "expectational" if negated else "goal orientation" if re.search(
                r"\bwhy (?:is|are|does|do|did) (?:there|the|this|it|they|we|you)\b.*\b(?:include|require|need|have|"
                r"say|allow|ask)", low) else "causal antecedent"
            trace.append("reasoning: why")
            return QuestionFrame(q, form, "reasoning", "backward", lehnert, trace=trace, **base)
        if cues.FORWARD.search(low) or cues.FORWARD_MORE.search(low) or cues.IF_WHAT_DO.search(lq.lower()):
            trace.append("reasoning: what happens if")
            return QuestionFrame(q, form, "reasoning", "forward", "causal consequent", trace=trace, **base)
        if cues.PROCESS.search(low) and not cues.STEPS_SPAN.search(low):
            trace.append("reasoning: what it takes")
            return QuestionFrame(q, form, "reasoning", "process", "enablement", trace=trace, **base)
        if wh == "how":
            nxt = low.split("how", 1)[1].split()[:1]
            if not nxt or nxt[0].strip("?,.") not in cues.HOW_QUANT:
                trace.append("reasoning: how (a way, not a quantity)")
                return QuestionFrame(q, form, "reasoning", "process", "instrumental/procedural", trace=trace, **base)
    options = _trim_options(_options(mq, mdoc, form, wh))
    if options:
        trace.append(f"choice: {' | '.join(options)}")
        return QuestionFrame(q, form, "lookup", "choice", "disjunctive", options=options, trace=trace, **base)
    if form == "polar":
        trace.append("boolean")
        return QuestionFrame(q, form, "lookup", "boolean", "verification", trace=trace, **base)
    typ = "ENTITY" if cues.STEPS_SPAN.search(low) else _span_type(mq, wh, mdoc)
    lehnert = "quantification" if typ in ("MONEY", "CARDINAL", "DURATION", "FREQUENCY", "QUANTITY", "PERCENT") \
        else "feature specification" if typ == "PROPERTY" else "concept completion"
    trace.append(f"span: {typ}")
    return QuestionFrame(q, form, "lookup", "span", lehnert, answer_type=typ, trace=trace, **base)
