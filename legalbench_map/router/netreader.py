"""
Tier 0 (reader network): answer a yes/no question locally from the passage that settles it, before any LLM call,
with a small network to find and read the passage and Pre-Tier 0's tagger to check what it read.

    1. Units: the document's windows of whole sentences (at most WINDOW_CHARS; a short document is one unit).
    2. Read: in a long document, the units sharing the most word stems with the question (its lexicon synonyms
       count) and the ones a small embedder (bge-small-en-v1.5) puts closest, so a paraphrase can be found too.
       The units' vectors are cached per document.
    3. Network: a cross-encoder (DeBERTa-v3-xsmall, trained 2026-10-01 on contract questions written and checked
       by gpt-oss-120b and distilled from a DeBERTa-v3-large teacher; /root/zadumai_nli_proto/reader_net, see
       models/reader_net/PROVENANCE.txt) gives each unit p(no), p(yes) and p(doesn't settle it). The answer is the
       likelier of yes and no if it clears its threshold and no unit says the other at CONFLICT or more; the
       deciding sentence is that unit's sentence most probable for the answer. Thresholds differ by what was read:
       a one-unit text (T_YES), a document read whole (T_YES_DOC), a long document read through retrieval
       (T_YES_LONG, off: on whole contracts the network's "yes" was right only ~68-89%).
    4. Checks, on the deciding sentence with Pre-Tier 0's tagger (router/frames.py): the party the question asks
       about is the one acting; a "yes" isn't contradicted by "shall not" and has every number, amount and month
       the question names; a "no" has a negation (silence is not "no"); "must" isn't answered by "may"; a "no"
       with a condition defers; a "yes" left to someone's discretion needs the question to name that party.
       Any failed check defers.
    5. A "yes" from a sentence with a condition or exception is "yes, with a condition", quoting it.
"""
from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from router.frames import (CONDITIONAL, LEXICON, QUESTION_AUX, _BASE_TRIE, _actor_matches, _blocked, _build_trie,
                           _clauses, _negated_sentence, _party_lexicon, _raw_tokens, condition_text, named_actors,
                           normalize, tag, tokens, values)
from router.pretier0 import PARTY_SCAN_CHARS, ROLES, TOPIC_QUESTION, find_parties
from router.tier0 import sentences

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "reader_net"
EMBEDDER = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge's prefix for queries
UNIT_CHARS = 700  # a document this short is one unit
WINDOW_CHARS = 600  # longer ones are cut into windows of whole sentences
K_LEX, K_EMB = 6, 6  # units read in a long document: the top by shared stems, then by the embedder
# Set 2026-10-01 (STATUS.md, TIER 0 READER NETWORK): T_YES is the lowest threshold right on >=99.5% of fresh A and
# >=99% of the dev rows training never saw; T_YES_DOC kept the CUAD test contracts' answers right by CUAD or the
# teacher. "No" is off: on LegalBench-like rows the network's "no" was right 0-11% of the time.
T_YES, T_NO = 0.935, 1.01
T_YES_DOC = 0.94  # a document of several units, read whole (at most K_LEX + K_EMB units)
T_YES_LONG = 1.01  # a longer document, read through retrieval: off (sealed contracts: 68% right by CUAD, 89% by the teacher)
CONFLICT = 0.5  # a unit this sure of the other answer makes it depend
TORCH_THREADS = 4
NO, YES, NONE = 0, 1, 2  # the network's labels (its NLI head's order: contradiction, entailment, neutral)
_STOP = frozenset("a an the and or of to in on for by with at from as is are was were be been being do does did has "
                  "have had can could will would shall should may might must this that these those it its any all "
                  "each either neither no not such than then so if there their they them we our you your party "
                  "parties agreement other one".split())
_WORD = re.compile(r"[a-z0-9]+")


def units(document: str) -> list:
    text = " ".join(document.split())
    if len(text) <= UNIT_CHARS:
        return [text]
    out, cur = [], ""
    for s in sentences(text):
        if cur and len(cur) + 1 + len(s) > WINDOW_CHARS:
            out.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip()
    return out + [cur] if cur else out


def _stems(text: str) -> set:
    return {w[:5] for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _synonyms() -> dict:
    out = {}
    for phrases in LEXICON.values():
        group = set().union(*(_stems(p) for p in phrases)) if phrases else set()
        for p in phrases:
            for s in _stems(p):
                out.setdefault(s, set()).update(group)
    return out


_SYN = _synonyms()


@dataclass
class NetResult:
    fired: bool
    answer: str | None = None  # "yes" | "no" when fired
    confidence: float = 0.0  # the network's probability of the answer
    reason: str = ""
    hypothesis: str | None = None  # kept for the harness, which reads Tier 0's results the same way
    evidence: list = field(default_factory=list)  # [{"text", "label", "p"}]: the deciding sentence first
    n_units: int = 0  # units in the document
    n_scored: int = 0  # units the network read
    qualifier: str | None = None  # CONDITIONAL for a "yes, with a condition"
    condition: str = ""  # the condition it quotes
    checks: list = field(default_factory=list)  # the checks that blocked the answer, when one did

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# --- the checks: Pre-Tier 0's tagger on the deciding sentence -------------------------------------------------------

_SPECIFIC = lambda a: a is not None and a not in ("ANY", "ALL", "NONE", "IT", "OTHER")  # noqa: E731
_ASKS_FROM_MODAL = {"MAY": "CAN", "MUST": "MUST", "BAN": "PROHIBITED"}


def document_trie(document: str, names: bool = False):
    """The tagger's trie with the role words (so "Can the landlord sublet?" names a party even where the text
    never mentions one) and the document's parties; with `names`, also the capitalized names acting in it
    ("SpringCo shall"), which Pre-Tier 0 adds only for questions about any party: they can swallow a heading
    ("PETS. Tenant may"), which would hide the party a question names."""
    parties = ROLES + list(find_parties(document[:PARTY_SCAN_CHARS])) + (named_actors(document) if names else [])
    return _build_trie(_party_lexicon(list(dict.fromkeys(parties))))


def question_view(question: str, trie) -> tuple:
    """(asks: DOES | CAN | MUST | PROHIBITED | None, the party in subject position or None, negated)"""
    toks = tokens(question)
    if not toks:
        return None, None, False
    asks = QUESTION_AUX.get(toks[0])
    tags = tag(toks[1:], trie)
    negated = any(t.name in ("NOT", "NONE") for t in tags)
    actor = None
    for t in tags:  # up to the first content word: "Does assigning ... require the other party's consent" has none
        if t.kind == "ACTOR":
            actor = t.name
        elif t.kind == "MODAL":
            asks = _ASKS_FROM_MODAL.get(t.name, asks)  # "Is a party required to", "Is X prohibited from"
        elif not (t.kind == "WORD" and t.name == "have"):  # "Does X have to"
            break
    return asks, actor, negated


def sentence_view(sentence: str, trie) -> tuple:
    """([(acting party, its modals)], negated anywhere, has a condition, leaves it to someone's discretion)"""
    raw = _raw_tokens(sentence)
    tags = tag([normalize(w) for w in raw], trie)
    subjects = []
    for clause in _clauses(tags):
        for i, t in enumerate(clause):
            if t.kind != "ACTOR":
                continue
            rest = [u for u in clause[i + 1:] if not (u.kind == "WORD" and u.name.endswith("ly"))]
            run = []
            for u in rest:
                if u.kind != "MODAL":
                    break
                run.append(u.name)
            if not run and not (rest and rest[0].kind == "ACTION"):
                continue  # not the subject of what follows
            name = t.name
            if name == "IT":  # "it shall": the party named before it
                named = [u for u in tags if u.kind == "ACTOR" and u.name != "IT" and u.start < t.start]
                if not named:
                    continue
                name = named[-1].name
            subjects.append((name, set(run)))
    discretion = any(t.kind == "BLOCK" and "discretion" in raw[t.start:t.end] for t in tags)
    return subjects, _negated_sentence(raw, tags), _blocked(tags), discretion


def check(question: str, sentence: str, answer: str, document: str) -> tuple[list, str]:
    """(the checks that block `answer`, the condition a "yes, with a condition" quotes)."""
    trie = document_trie(document)
    asks, actor, q_negated = question_view(question, trie)
    if actor in (None, "ANY", "ALL"):
        trie = document_trie(document, names=True)
    subjects, negated, blocked, discretion = sentence_view(sentence, trie)
    fails = []
    mine = [m for a, m in subjects if actor is None or _actor_matches(actor, a)]
    if _SPECIFIC(actor) and subjects and not mine:
        fails.append("party")  # "Can the receiving party ...?" from "The Disclosing Party may ..."
    modals = set().union(*mine) if mine else set()
    neg_mine = bool(modals & {"NOT", "BAN"}) or any(a == "NONE" for a, _ in subjects)
    if answer == "yes":
        if not values(question) <= values(sentence):
            fails.append("values")  # "Is the rent $1,500?" from "$2,000": the numbers, amounts and months must be there
        if asks == "PROHIBITED":
            if not negated:
                fails.append("yes_ban")  # prohibited, from a sentence that bans nothing
        elif not q_negated and neg_mine:
            fails.append("yes_neg")  # "yes" while the sentence says that party shall not
        if asks == "MUST" and modals and modals <= {"MAY"}:
            fails.append("must_may")  # the text permits it but doesn't require it
        if discretion and (actor is None or not mine):
            # "The Company may, in its sole discretion, pay an annual bonus": "Can the Company pay one?" yes, but
            # "Is the employee entitled to one?" no, so the question has to name the party whose discretion it is
            fails.append("discretion")
    else:
        if asks == "PROHIBITED":
            if "MAY" not in modals:
                fails.append("no_may")  # not prohibited needs a "may"
        elif not q_negated and not negated:
            fails.append("no_neg")  # silence is not "no"
        if blocked:
            fails.append("no_cond")  # "shall not ... except": often the very case asked about
    return fails, condition_text(sentence) if answer == "yes" and blocked else ""


# --- the network and the reader ----------------------------------------------------------------------------------------

class NetReader:
    """`answer(question, document)` like router.tier0.Tier0's. Loads the network (and, on the first long
    document, the embedder) once; `net` can be any callable [(text, question)] -> [[p_no, p_yes, p_none]]."""
    name = "Tier 0 (local reader network)"

    def __init__(self, model_dir: str | Path = MODEL_DIR, net=None, t_yes: float = T_YES, t_no: float = T_NO,
                 t_yes_doc: float | None = T_YES_DOC, t_yes_long: float | None = T_YES_LONG,
                 conflict: float = CONFLICT, k_lex: int = K_LEX, k_emb: int = K_EMB, embedder=None):
        self.net = net or _CrossEncoder(Path(model_dir))
        self.t_yes, self.t_no, self.conflict = t_yes, t_no, conflict
        self.t_yes_doc = t_yes if t_yes_doc is None else t_yes_doc
        self.t_yes_long = self.t_yes_doc if t_yes_long is None else t_yes_long
        self.k_lex, self.k_emb = k_lex, k_emb
        self._embedder = embedder
        self._vectors = OrderedDict()  # document hash -> its units' vectors
        self._lock = threading.Lock()

    def _embed(self, texts: list):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDER, device="cpu")
        return self._embedder.encode(texts, batch_size=32, normalize_embeddings=True)

    def pick(self, question: str, document: str, us: list) -> list:
        """Indexes of the units to read."""
        if len(us) <= self.k_lex + self.k_emb:
            return list(range(len(us)))
        q = _stems(question)
        qx = set().union(q, *(_SYN.get(s, ()) for s in q))
        lex = sorted(range(len(us)), key=lambda i: -(2 * len(q & _stems(us[i])) + len(qx & _stems(us[i]))))
        picked = lex[:self.k_lex]
        if self.k_emb:
            key = hashlib.sha1(document.encode()).hexdigest()
            with self._lock:
                vecs = self._vectors.get(key)
                if vecs is None:
                    vecs = self._embed(us)
                    self._vectors[key] = vecs
                    while len(self._vectors) > 32:
                        self._vectors.popitem(last=False)
                else:
                    self._vectors.move_to_end(key)
            qv = self._embed([QUERY_PREFIX + question])[0]
            order = (-(vecs @ qv)).argsort()
            picked += [int(i) for i in order if int(i) not in picked][:self.k_emb]
        return picked

    def answer(self, question: str, document: str) -> NetResult:
        if TOPIC_QUESTION.search(question):
            return NetResult(False, reason="asks whether the text covers a topic, not what it says about it")
        us = units(document)
        if len(us) > self.k_lex + self.k_emb and self.t_yes_long > 1 and self.t_no > 1:
            # long documents are off: don't spend the embedding and the network on an answer that can't be given
            return NetResult(False, reason="a long document, read through retrieval: the network doesn't answer "
                                           "there (on whole contracts its yes was right only ~68-89%)", n_units=len(us))
        picked = self.pick(question, document, us)
        if not picked:
            return NetResult(False, reason="the document has no text", n_units=len(us))
        probs = self.net([(us[i], question) for i in picked])
        base = dict(n_units=len(us), n_scored=len(picked))
        best = {a: max(range(len(picked)), key=lambda k: probs[k][a]) for a in (YES, NO)}
        ans = YES if probs[best[YES]][YES] >= probs[best[NO]][NO] else NO
        p = probs[best[ans]][ans]
        other = NO if ans == YES else YES
        label = "yes" if ans == YES else "no"
        unit = us[picked[best[ans]]]
        t_yes = self.t_yes if len(us) == 1 else self.t_yes_doc if len(us) <= self.k_lex + self.k_emb else self.t_yes_long
        if p < (t_yes if ans == YES else self.t_no):
            why = f"no passage settles it (closest: {label} at {p:.2f})"
            if ans == YES and t_yes == self.t_yes_long and p >= self.t_yes_doc:
                why = (f"a long document, read through retrieval: the network's yes ({p:.2f}) isn't trusted there "
                       f"(on whole contracts it was right only ~68-89%)")
            return NetResult(False, reason=why, evidence=[{"text": unit, "label": label, "p": round(p, 3)}], **base)
        if probs[best[other]][other] >= self.conflict:
            return NetResult(False, reason=f"passages disagree (yes {probs[best[YES]][YES]:.2f}, no {probs[best[NO]][NO]:.2f})",
                             evidence=[{"text": us[picked[best[a]]], "label": "yes" if a == YES else "no",
                                        "p": round(probs[best[a]][a], 3)} for a in (ans, other)], **base)
        sents = sentences(unit)
        sent = unit
        if len(sents) > 1:
            sp = self.net([(s, question) for s in sents])
            sent = sents[max(range(len(sents)), key=lambda k: sp[k][ans])]
        fails, cond = check(question, sent, label, document)
        evidence = [{"text": sent, "label": f"{label}, {CONDITIONAL}" if cond else label, "p": round(p, 3)}]
        if fails:
            return NetResult(False, reason=f"the network says {label} ({p:.2f}), but the sentence fails the "
                                           f"{', '.join(fails)} check", evidence=evidence, checks=fails, **base)
        reason = f"the network reads {label} in one passage ({p:.2f}) and the sentence passes the party, " \
                 f"negation and modality checks"
        if cond:
            reason += f", {CONDITIONAL}: \"{cond}\""
        return NetResult(True, label, round(p, 3), reason, evidence=evidence, qualifier=CONDITIONAL if cond else None,
                         condition=cond, **base)


class _CrossEncoder:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"no reader network at {model_dir} (see STATUS.md, TIER 0 READER NETWORK)")
        torch.set_num_threads(TORCH_THREADS)
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()

    def __call__(self, pairs: list) -> list:
        with self._torch.inference_mode():
            enc = self.tok([t for t, _ in pairs], [q for _, q in pairs], truncation="only_first", max_length=256,
                           padding=True, return_tensors="pt")
            return self.model(**enc).logits.softmax(-1).tolist()
