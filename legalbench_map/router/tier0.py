"""
Tier 0: answer a yes/no question locally, before any LLM call, when one
sentence of the document settles it.

    1. Convert the question to a statement with spaCy's parse
       ("Can the tenant sublet?" -> "The tenant can sublet."). Questions that
       don't start with an auxiliary followed by their subject (wh-questions,
       odd word order) are not attempted.
    2. Split the document into sentences and keep the TOP_K that share the
       most content words with the statement.
    3. Score each kept sentence against the statement with an NLI
       cross-encoder (nli-deberta-v3-xsmall): entailment -> "yes",
       contradiction -> "no", at THRESHOLD or above, counting only sentences
       that share a word with the question's predicate (not just its subject).
    4. Defer if sentences disagree, if nothing clears the threshold, or if
       a deciding sentence carries an exception or condition ("unless",
       "except", "subject to", "without ... consent"): on those the model
       answers yes/no with full confidence while the right answer is "it
       depends".

On a 32-case legal yes/no set (/root/zadumai_nli_proto, hand-written) this
fires on 20 and gets all 20 right. Without the predicate-word rule it fired on
23 with one miss: an unrelated clause scored as a contradiction, which makes
"no" the riskier answer.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from router.harness import _SENTENCE_END
from router.pretier0 import PARTY_SCAN_CHARS, TOPIC_QUESTION, find_parties

MODEL = "cross-encoder/nli-deberta-v3-xsmall"
THRESHOLD = 0.9
TOP_K = 8
MIN_SENTENCE_CHARS = 15  # shorter pieces ("1. RENT.") join the next sentence
TORCH_THREADS = 4

AUXILIARIES = {"is", "are", "was", "were", "am", "does", "do", "did", "has", "have", "had", "can", "could",
               "will", "would", "shall", "should", "may", "might", "must"}
BE = {"is", "are", "was", "were", "am"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "expl", "csubj"}
IS_IT_TRUE = re.compile(r"^(is it (true|the case)|does it say|does the (contract|agreement|document) say) that\s+", re.I)

# Conditions and scope limits: with one of these the answer is "it depends".
# "upon ... notice" is a procedure, not a condition, so it doesn't count.
CUES = re.compile(
    r"\b(unless|except|excepting|provided that|provided, however|subject to|notwithstanding|"
    r"conditioned (?:up)?on|without (?:[\w'’]+\s+){0,5}?(?:consent|approval)|"
    r"if|only|in the event|during|within|sole discretion|"
    r"upon(?!\s+(?:[\w()'’,]+\s+){0,6}?notice))\b",
    re.I,
)
STOPWORDS = frozenset(
    "a an the and or of to in on for by with at from as is are was were be been being do does did has have had "
    "can could will would shall should may might must this that these those it its we our us you your they their "
    "them he she his her there any all each either neither no not such than then so if".split()
)
_WORD = re.compile(r"[a-z0-9]+")
_DOC_PRONOUNS = re.compile(r"\b(we|us|our|you|your)\b", re.I)
PRONOUN_SUBJECTS = {"we", "i", "you", "they"}
# A subject like "Neither party" or "the parties" speaks for every party.
_ALL_PARTIES = re.compile(r"\b(either|each|neither|both|any|all|no)\s+part(y|ies)\b|\bthe parties\b", re.I)


@dataclass
class Tier0Result:
    fired: bool
    answer: str | None = None  # "yes" | "no" when fired
    confidence: float = 0.0
    reason: str = ""
    hypothesis: str | None = None  # the question as a statement
    evidence: list = field(default_factory=list)  # [{"text", "label", "p"}], most decisive first
    n_units: int = 0  # sentences in the document
    n_scored: int = 0  # sentences sent to the NLI model

    def to_dict(self) -> dict:
        return asdict(self)


def sentences(document: str) -> list:
    """The document's sentences. Unlike the harness's units, short sentences
    stay whole: gluing "Tenant shall not sublet." to its neighbor would hand
    the NLI model, and the exception-cue check, a second clause."""
    pieces = [p for line in re.split(r"\n+", document) for p in _SENTENCE_END.split(" ".join(line.split()))]
    out, carry = [], ""
    for piece in pieces:
        piece = f"{carry} {piece}".strip()
        if len(piece) < MIN_SENTENCE_CHARS:
            carry = piece
            continue
        out.append(piece)
        carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


def _stems(text: str) -> set:
    """Content words cut to 5 letters, so "pays"/"payment" and "sublet"/"subletting" meet."""
    return {w[:5] for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1}


def _subject_end(doc) -> int | None:
    """Index just past the subject that follows the leading auxiliary. The
    small parser often hangs the predicate off the subject noun ("Is the
    landlord responsible": landlord <-amod- responsible), so subtrees overrun;
    instead a "be" question's subject ends at its head noun (plus an "of"
    phrase), and any other auxiliary's subject ends at the first verb."""
    if doc[0].lower_ in BE:
        heads = [t for t in doc[1:] if t.dep_ in SUBJECT_DEPS or t.dep_ == "attr"]
        if not heads or heads[0].left_edge.i != 1:
            return None
        end = heads[0].i + 1
        if end < len(doc) and doc[end].lower_ == "to":
            # "Is the right to audit books discussed": where the subject ends
            # can't be read off the small parser's tree.
            return None
        for child in heads[0].rights:
            if child.dep_ == "prep" and child.lower_ == "of":
                end = child.right_edge.i + 1
        return end
    verb = next((t for t in doc[1:] if t.pos_ in ("VERB", "AUX") and t.dep_ not in ("amod", "compound")), None)
    if verb is None:
        # "Can the licensee audit the books": the tagger reads "audit" as a noun,
        # but after a modal and its subject the parse root is still the verb.
        verb = next((t for t in doc[2:] if t.dep_ == "ROOT"), None)
        if verb is None:
            return None
    end = verb.i
    while end > 1 and doc[end - 1].dep_ in ("advmod", "neg"):  # "Can the landlord unreasonably withhold"
        end -= 1
    return end


def restate(question: str, nlp) -> tuple[str, str, str] | None:
    """(statement, subject, predicate): the yes/no question as a declarative
    sentence and its parts, or None if it isn't an auxiliary-first question
    with the subject right after the auxiliary."""
    q = " ".join(question.split()).rstrip("?").strip()
    if m := IS_IT_TRUE.match(q):
        rest = q[m.end():]
        return (rest[:1].upper() + rest[1:] + ".", "", rest) if rest else None
    doc = nlp(q)
    if len(doc) < 3 or doc[0].lower_ not in AUXILIARIES:
        return None
    end = _subject_end(doc)
    if end is None or not 1 < end < len(doc):
        return None
    # "Does the tenant pay rent?" -> "The tenant does pay rent.": emphatic "do"
    # keeps the sentence grammatical without re-inflecting the verb.
    statement = f"{doc[1:end].text} {doc[0].lower_} {doc[end:].text}"
    return statement[:1].upper() + statement[1:] + ".", doc[1:end].text, doc[end:].text


def to_statement(question: str, nlp) -> str | None:
    restated = restate(question, nlp)
    return restated[0] if restated else None


class Tier0:
    def __init__(self, nli=None, nlp=None, threshold: float = THRESHOLD, top_k: int = TOP_K, answer_no: bool = False):
        """`nli(premises, hypothesis)` returns one {"entailment", "neutral",
        "contradiction"} dict per premise; `nlp` is a spaCy pipeline. Both
        default to the real models."""
        if nlp is None:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        self.nlp = nlp
        self.nli = nli or _CrossEncoderNLI(MODEL)
        self.threshold = threshold
        self.top_k = top_k
        # On LegalBench (router eval, 2026-09-30) "yes" was 98% precise, "no" 46-60%:
        # a contradiction usually came from an unrelated sentence.
        self.answer_no = answer_no

    def answer(self, question: str, document: str) -> Tier0Result:
        if TOPIC_QUESTION.search(question):
            return Tier0Result(False, reason="asks whether the text covers a topic, not what it says about it")
        restated = restate(question, self.nlp)
        if restated is None:
            return Tier0Result(False, reason="not a yes/no question Tier 0 can restate")
        hypothesis, subject, predicate = restated
        if subject.lower() in PRONOUN_SUBJECTS and not _DOC_PRONOUNS.search(document):
            return Tier0Result(False, reason=f'can\'t tell which party "{subject}" is', hypothesis=hypothesis)
        units = sentences(document)
        q_stems = _stems(hypothesis)
        overlap = sorted(((len(q_stems & _stems(u)), i) for i, u in enumerate(units)), reverse=True)
        kept = [i for n, i in overlap[: self.top_k] if n > 0]
        base = dict(hypothesis=hypothesis, n_units=len(units), n_scored=len(kept))
        if not kept:
            return Tier0Result(False, reason="no sentence shares a content word with the question", **base)

        scores = self.nli([units[i] for i in kept], hypothesis)
        rows = [(units[i], s) for i, s in zip(kept, scores)]
        # A sentence can decide only if it shares a word with the predicate, not
        # just the subject: "Tenant shall not sublet" must not answer "Can the
        # tenant assign the lease?", though the model scores it a contradiction.
        p_stems = _stems(predicate) or q_stems
        on_topic = lambda r: len(p_stems & _stems(r[0]))
        # Nor if it is about another party: "Licensee shall not audit the books
        # of Licensor" says nothing about whether the Licensor can audit.
        parties = find_parties(document[:PARTY_SCAN_CHARS])
        q_parties = {p for p in parties if re.search(rf"\b{re.escape(p.lower())}\b", subject.lower())}
        mismatched = []

        def same_party(r):
            if not q_parties:
                return True
            actors = self._actors(r[0], parties)
            if actors is None or actors & q_parties:
                return True
            mismatched.append((r[0], actors))
            return False

        yes = sorted((r for r in rows if r[1]["entailment"] >= self.threshold and on_topic(r) and same_party(r)),
                     key=lambda r: (-on_topic(r), -r[1]["entailment"]))
        no = sorted((r for r in rows if r[1]["contradiction"] >= self.threshold and on_topic(r) and same_party(r)),
                    key=lambda r: (-on_topic(r), -r[1]["contradiction"]))

        def ev(rs, label, key):
            return [{"text": t, "label": label, "p": round(s[key], 3)} for t, s in rs]

        if yes and no:
            return Tier0Result(False, reason=f"sentences disagree ({len(yes)} say yes, {len(no)} say no)",
                               evidence=(ev(yes, "yes", "entailment") + ev(no, "no", "contradiction"))[:3], **base)
        if not yes and not no and mismatched:
            text, actors = mismatched[0]
            return Tier0Result(False, reason=f"the sentence that would settle it is about {' and '.join(sorted(actors))}, "
                                             f"not {' and '.join(sorted(q_parties))}",
                               evidence=[{"text": text, "label": "other party", "p": 0.0}], **base)
        if not yes and not no:
            best = max(rows, key=lambda r: max(r[1]["entailment"], r[1]["contradiction"]))
            label, key = ("yes", "entailment") if best[1]["entailment"] >= best[1]["contradiction"] else ("no", "contradiction")
            why = (f"{best[1][key]:.2f} < {self.threshold}" if best[1][key] < self.threshold
                   else "it shares no word with the question's predicate")
            return Tier0Result(False, reason=f"no sentence settles it (closest: {label} at {best[1][key]:.2f}; {why})",
                               evidence=ev([best], label, key), **base)
        answer, key, hits = ("yes", "entailment", yes) if yes else ("no", "contradiction", no)
        evidence = ev(hits, answer, key)[:3]
        for text, _ in hits:
            if m := CUES.search(text):
                return Tier0Result(False, reason=f"a deciding sentence has a condition or exception ('{m.group(0)}')",
                                   evidence=evidence, **base)
        if answer == "no" and not self.answer_no:
            return Tier0Result(False, reason="the model says no, and Tier 0's \"no\" answers are off",
                               evidence=evidence, **base)
        return Tier0Result(True, answer, round(hits[0][1][key], 3),
                           reason=f"one sentence {'entails' if answer == 'yes' else 'contradicts'} the question as a statement",
                           evidence=evidence, **base)


    def _actors(self, sentence: str, parties: list) -> set | None:
        """Parties named in the sentence's subjects (the agent, for a passive),
        or None if no subject names one or a subject speaks for all parties."""
        doc = self.nlp(sentence)
        spans = []
        for t in doc:
            if t.dep_ == "nsubj":
                spans.append(doc[t.left_edge.i:t.right_edge.i + 1].text)
            elif t.dep_ == "agent":
                spans.append(doc[t.i:t.right_edge.i + 1].text)
        text = " ".join(spans)
        if not text or _ALL_PARTIES.search(text):
            return None
        found = {p for p in parties if re.search(rf"\b{re.escape(p)}\b", text, re.I)}
        return found or None


class _CrossEncoderNLI:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(TORCH_THREADS)
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.idx = {k: labels[k] for k in ("entailment", "neutral", "contradiction")}

    def __call__(self, premises: list, hypothesis: str) -> list:
        with self._torch.inference_mode():
            enc = self.tok(premises, [hypothesis] * len(premises), return_tensors="pt",
                           padding=True, truncation=True, max_length=512)
            probs = self.model(**enc).logits.softmax(-1)
        return [{k: float(p[i]) for k, i in self.idx.items()} for p in probs]
