"""The contract map (2026-10-02): what kind of document this is, what its sections are about, and which questions
the local tiers leave to the LLM.

Measured in /root/zadumai_nli_proto/contract_map/bench/RESULTS.md (models built by ../build_models.py there):
- Kind: TF-IDF + logistic regression on the first 4,000 characters -> commercial / nda / privacy policy /
  terms of service / lease / not a contract. 95.1% on 1,156 held-out documents; 99.1% on the 75% it gives
  p >= 0.9. Trained on whole documents, so texts under MIN_DOC_CHARS (a clause, a paragraph) get no kind.
- Sections: one tagger per source of labels, chosen by the kind. Commercial: LEDGAR's 77 provision types (91.5% right
  on the 97% it tags) and CUAD's 37 review categories (F1 0.60); NDA: ContractNLI's 17 (0.68); privacy policies:
  OPP-115's 9 (0.77); terms of service: UNFAIR-ToS's 8 (0.74); leases: a fine-tuned MiniLM (0.28 on all 8,057 test
  paragraphs, TF-IDF 0.21; run in the
  background because it's slow). Jev may read sections whose score is near the threshold (+0.015-0.02 F1 on CUAD,
  ToS and privacy policies), at most LLM_UNSURE_MAX per document.
- Who answers doesn't change (the user's call, 2026-10-02: "local first"). A rule that sent questions about a
  high-risk business clause (the CUAD types tagged below F1 0.75, meta.json `to_llm`) and every lease question
  straight to the LLM was built and measured, then left off: on the open sets it took 10% of user-style questions
  (57-70% of LegalBench's) away from the local tiers, which answered them 99.6% right (Pre-Tier 0 604/604, the
  network 469/472) where the LLM got 88.4% (499 of them, Jev). ROUTE_HIGH_RISK / ROUTE_LEASES turn it back on.
  Questions the local tiers leave still go to the LLM, as before. The decision is reported on each answer.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "contract_map"
MIN_DOC_CHARS = 3000  # shorter texts get no kind (the router read whole documents)
KIND_MIN_P = 0.9  # act on the kind only this sure (99.1% right at >= 0.9 on held-out documents)
CONSUMER = ("privacy policy", "terms of service")  # high-risk business clause rule doesn't apply to these
LLM_UNSURE_MAX = 8  # Jev calls per document on sections near a tagger's threshold
UNSURE_BAND = 0.2  # "near" = within this of the threshold
CACHE_DOCS = 64
MIN_SECTION_CHARS = 40
MAX_SECTION_CHARS = 4000
TAGGERS = {"commercial": ("ledgar", "cuad"), "nda": ("contractnli",), "privacy policy": ("opp115",),
           "terms of service": ("tos",)}
LLM_TAGGERS = ("cuad", "opp115", "tos")  # where Jev on the unsure band helped
ROUTE_HIGH_RISK = False  # True: high-risk business clause questions skip the local tiers (measured worse, see above)
ROUTE_LEASES = False  # True: every question about a lease skips them (unmeasured: no lease questions in our sets)

# Which high-risk business clause a question is about: hand-written, read on the lowercased question. A miss
# leaves the question on today's path; a false match sends it to the LLM (slower, not wrong).
QUESTION_TRIGGERS = {
    "Affiliate License-Licensee": r"\baffiliates?\b.{0,60}\blicen[cs]|\blicen[cs].{0,60}\baffiliates?\b",
    "Affiliate License-Licensor": r"\baffiliates?\b.{0,60}\blicen[cs]|\blicen[cs].{0,60}\baffiliates?\b",
    "Anti-Assignment": r"\bassign(?:ment|able|ed|s)?\b(?!.{0,40}\b(?:intellectual property|ip|inventions?|patents?|"
                       r"copyrights?|work product)\b)|\btransfer (?:this|the) (?:agreement|contract)\b",
    "Audit Rights": r"\baudit|\binspect (?:the |its |their )?(?:books|records)\b|\bbooks and records\b",
    "Cap On Liability": r"\b(?:cap|caps|capped|limit|limits|limited|limitation|maximum|ceiling)\b.{0,30}\bliabilit|"
                        r"\bliabilit.{0,30}\b(?:cap|capped|limited|limit)\b",
    "Change Of Control": r"\bchange (?:of|in) control\b|\bmerger\b|\bacquisition of\b|\bis acquired\b|\bgets acquired\b",
    "Competitive Restriction Exception": r"\bexceptions?\b.{0,40}\b(?:non-?compet|exclusiv|restrict)",
    "Covenant Not To Sue": r"\bcovenant not to sue\b|\bnot (?:to )?sue\b|\bchallenge (?:the )?(?:validity|ownership)\b",
    "Exclusivity": r"\bexclusiv",
    "Insurance": r"\binsurance\b|\binsured\b|\binsure\b",
    "Ip Ownership Assignment": r"\b(?:own|owns|ownership)\b.{0,40}\b(?:intellectual property|ip|inventions?|work product|"
                               r"patents?|copyrights?)\b|\bassign.{0,40}\b(?:intellectual property|ip|inventions?|"
                               r"patents?|copyrights?|work product)\b",
    "Irrevocable Or Perpetual License": r"\b(?:irrevocable|perpetual)\b",
    "Joint Ip Ownership": r"\bjoint(?:ly)? own|\bco-?own",
    "License Grant": r"\blicen[cs]e[sd]?\b|\bsublicen",
    "Liquidated Damages": r"\bliquidated damages\b|\bpenalt(?:y|ies)\b|\bbreak-?up fee\b|\btermination fee\b",
    "Minimum Commitment": r"\bminimum (?:purchase|order|commitment|quantit|volume|amount|spend)|\b(?:buy|purchase|order) at least\b",
    "Most Favored Nation": r"\bmost[- ]favou?red\b|\bmfn\b|\bbest price\b|\bno less favou?rable\b",
    "No-Solicit Of Customers": r"\bsolicit.{0,40}\b(?:customer|client)s?\b",
    "No-Solicit Of Employees": r"\bsolicit.{0,40}\b(?:employee|staff|personnel|worker)s?\b|"
                               r"\b(?:hire|poach|recruit).{0,30}\b(?:employee|staff|personnel)s?\b",
    "Non-Compete": r"\bnon-?compet|\bcompet(?:e|es|ing|itor|itors)\b",
    "Non-Disparagement": r"\bdisparag|\bnegative (?:statement|comment|review)s?\b|\bbad-?mouth|\bcriticiz",
    "Non-Transferable License": r"\b(?:non-?transferable|transfer\w*)\b.{0,30}\blicen[cs]|\blicen[cs]\w*.{0,30}\btransfer",
    "Notice Period To Terminate Renewal": r"\bnotice\b.{0,40}\b(?:renew|non-?renew)|\b(?:renew|non-?renew)\w*.{0,40}\bnotice\b",
    "Post-Termination Services": r"\bpost-?termination\b|\btransition (?:services|period|assistance)\b",
    "Price Restrictions": r"\b(?:raise|increase|change|adjust)\w*\s+(?:the |its |their |our |your )?(?:prices?|fees?|rates?)\b|"
                          r"\bprice (?:increase|change|cap|restriction)s?\b",
    "Renewal Term": r"\brenew",
    "Revenue/Profit Sharing": r"\b(?:revenue|profit)s?[- ]shar|\bshare (?:of )?(?:the |its |their )?(?:revenue|profit|net sales)|\broyalt",
    "Rofr/Rofo/Rofn": r"\bright of first\b|\bfirst (?:refusal|offer|negotiation)\b|\brof[rno]\b",
    "Source Code Escrow": r"\bsource code\b|\bescrow\b",
    "Termination For Convenience": r"\bterminat\w*.{0,40}\b(?:for convenience|without cause|for any reason|at any time|"
                                   r"without (?:a )?reason)|\bcancel\w*.{0,30}\b(?:at any time|anytime|for any reason)",
    "Third Party Beneficiary": r"\bthird[- ]part(?:y|ies)\b.{0,20}\bbeneficiar|\bbeneficiar",
    "Uncapped Liability": r"\b(?:uncapped|unlimited)\s+liabilit|\bliabilit\w*.{0,20}\b(?:uncapped|unlimited|not (?:be )?(?:capped|limited))",
    "Volume Restriction": r"\b(?:volume|usage) (?:limit|restriction|cap)s?\b|\bmaximum (?:number|quantity|volume)\b",
    "Warranty Duration": r"\bwarrant(?:y|ies)\b.{0,40}\b(?:period|last|long|duration|expire|months?|years?|days?)\b|"
                         r"\bhow long\b.{0,30}\bwarrant",
}
_TRIGGERS = {k: re.compile(v) for k, v in QUESTION_TRIGGERS.items()}


@dataclass
class DocMap:
    kind: str | None  # None: too short, or the router isn't sure
    kind_p: float
    kind_top: str  # the router's pick even when not sure
    sections: list = field(default_factory=list)  # [{"i", "start", "chars", "text", "types": [{"type", "p", "by"}]}]
    pending: bool = False  # the lease encoder is still reading it

    def summary(self, n: int = 12) -> dict:
        counts = {}
        for s in self.sections:
            for t in s["types"]:
                counts[t["type"]] = counts.get(t["type"], 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:n]
        return {"kind": self.kind, "kind_p": round(self.kind_p, 3), "kind_top": self.kind_top,
                "sections": len(self.sections), "tagged": sum(bool(s["types"]) for s in self.sections),
                "types": dict(top), "pending": self.pending}


@dataclass
class Decision:
    to_llm: bool
    reason: str
    question_types: list  # high-risk clause types the question names
    document: dict  # DocMap.summary()

    def to_dict(self) -> dict:
        return asdict(self)


def split_sections(text: str) -> list:
    """Paragraphs: blank-line blocks (single lines when the text has few blank lines); a short piece (a heading,
    a number) joins the next one; long ones are cut. Returns [(start, text)]."""
    sep = r"\n\s*\n" if len(re.findall(r"\n\s*\n", text)) >= max(3, text.count("\n") // 4) else r"\n"
    pieces, pos = [], 0
    for m in re.finditer(sep, text + "\n\n"):
        pieces.append((pos, text[pos:m.start()])); pos = m.end()
    out, carry, carry_start = [], "", None
    for start, p in pieces:
        p = re.sub(r"\s+", " ", p).strip()
        if not p: continue
        if carry: p, start = carry + " " + p, carry_start
        if len(p) < MIN_SECTION_CHARS:
            carry, carry_start = p, start; continue
        carry = ""
        for k in range(0, len(p), MAX_SECTION_CHARS): out.append((start + k, p[k:k + MAX_SECTION_CHARS]))
    if carry: out.append((carry_start, carry))
    return out


def question_types(question: str) -> list:
    q = question.lower().replace("’", "'")
    return [t for t, rx in _TRIGGERS.items() if rx.search(q)]


class _Tagger:
    def __init__(self, path: Path):
        import joblib
        d = joblib.load(path)
        self.vec, self.classes, self.W, self.b, self.th = d["vec"], d["classes"], d["W"], d["b"], d["threshold"]
        self.labels = d["labels"]  # score class -> the dataset's own label
        self.single = "ledgar" in path.name  # one type per provision: argmax

    def scores(self, texts: list) -> np.ndarray:
        z = self.vec.transform(texts) @ self.W + self.b
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class _LeaseEncoder:
    def __init__(self, d: Path, threads: int = 2):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch  # threads: the process-wide setting (netreader sets it)
        m = json.loads((d / "meta.json").read_text())
        self.classes, self.cls, self.th, self.labels = m["classes"], m["score_classes"], m["threshold"], m["labels"]
        self.tok = AutoTokenizer.from_pretrained(str(d)); self.enc = AutoModel.from_pretrained(str(d)).eval()
        self.head = torch.nn.Linear(self.enc.config.hidden_size, len(self.classes))
        self.head.load_state_dict(torch.load(d / "head.pt", map_location="cpu")); self.head.eval()

    def scores(self, texts: list) -> np.ndarray:
        torch = self.torch; out = np.zeros((len(texts), len(self.cls))); idx = [self.classes.index(c) for c in self.cls]
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                e = self.tok(texts[i:i + 32], truncation=True, max_length=256, padding=True, return_tensors="pt")
                h = self.enc(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1).float()
                p = torch.sigmoid(self.head((h * m).sum(1) / m.sum(1).clamp(min=1))).numpy()
                out[i:i + 32] = p[:, idx]
        return out


class ContractMap:
    def __init__(self, model_dir: str | Path = MODEL_DIR, lease_encoder: bool = True):
        import joblib
        d = Path(model_dir)
        if not (d / "router.joblib").exists():
            raise FileNotFoundError(f"no contract map models at {d} (build: /root/zadumai_nli_proto/contract_map/build_models.py)")
        self.meta = json.loads((d / "meta.json").read_text())
        r = joblib.load(d / "router.joblib"); self.rvec, self.rlr = r["vec"], r["lr"]
        self.taggers = {n: _Tagger(d / f"tagger_{n}.joblib") for n in {t for ts in TAGGERS.values() for t in ts}}
        self.to_llm_types = set(self.meta["to_llm"]["cuad_types"])
        self.lease = None
        if lease_encoder and (d / "lease" / "head.pt").exists():
            self.lease = _LeaseEncoder(d / "lease")
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._bg = cf.ThreadPoolExecutor(1)  # the lease encoder, off the request path

    # ---------- documents ----------
    def kind(self, document: str) -> tuple:
        """(kind or None, p, the router's top pick)."""
        P = self.rlr.predict_proba(self.rvec.transform([document[:4000]]))[0]
        top = self.rlr.classes_[int(P.argmax())]; p = float(P.max())
        ok = len(document) >= MIN_DOC_CHARS and p >= KIND_MIN_P
        return (top if ok else None), p, top

    def analyze(self, document: str, llm=None) -> DocMap:
        key = hashlib.sha256(document.encode("utf-8", "ignore")).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key); return self._cache[key]
        kind, p, top = self.kind(document)
        secs = split_sections(document)
        dm = DocMap(kind, p, top, [{"i": i, "start": s, "chars": len(t), "text": t[:160], "types": []} for i, (s, t) in enumerate(secs)])
        texts = [t for _, t in secs]
        if kind in TAGGERS and texts:
            for name in TAGGERS[kind]:
                self._tag(dm, texts, self.taggers[name], llm if name in LLM_TAGGERS else None)
        elif kind == "lease" and self.lease is not None and texts:
            dm.pending = True
            self._bg.submit(self._tag_lease, dm, texts)
        with self._lock:
            self._cache[key] = dm
            while len(self._cache) > CACHE_DOCS: self._cache.popitem(last=False)
        return dm

    def _tag(self, dm: DocMap, texts: list, tg: _Tagger, llm) -> None:
        S = tg.scores(texts)
        unsure = []
        for i, s in enumerate(S):
            if tg.single:
                j = int(s.argmax())
                if s[j] >= tg.th: dm.sections[i]["types"].append({"type": tg.labels[tg.classes[j]], "p": round(float(s[j]), 3), "by": "tfidf"})
                continue
            for j in np.where(s >= tg.th)[0]:
                dm.sections[i]["types"].append({"type": tg.labels[tg.classes[j]], "p": round(float(s[j]), 3), "by": "tfidf"})
            if llm is not None and abs(float(s.max()) - tg.th) < UNSURE_BAND:
                unsure.append((abs(float(s.max()) - tg.th), i))
        if llm is not None and unsure:
            crit = {tg.labels[c]: tg.labels[c] for c in tg.classes}; crit["NONE"] = "none of the clause types above"
            todo = [i for _, i in sorted(unsure)[:LLM_UNSURE_MAX]]
            def ask(i):
                try:
                    return i, llm.choice(texts[i][:4000], "Which type of clause is this text? If it is none of these, pick NONE.", crit)
                except Exception:
                    return i, None
            with cf.ThreadPoolExecutor(4) as ex:
                for i, out in ex.map(ask, todo):
                    if out is None: continue
                    keep = [t for t in dm.sections[i]["types"] if t["type"] not in crit]  # Jev's pick replaces this tagger's
                    if out["choice"] != "NONE":
                        keep.append({"type": out["choice"], "p": round(float(out["probabilities"].get(out["choice"], 0)), 3), "by": "jev"})
                    dm.sections[i]["types"] = keep

    def _tag_lease(self, dm: DocMap, texts: list) -> None:
        try:
            S = self.lease.scores(texts)
            for i, s in enumerate(S):
                dm.sections[i]["types"] = [{"type": self.lease.labels[self.lease.cls[j]], "p": round(float(s[j]), 3), "by": "encoder"}
                                           for j in np.where(s >= self.lease.th)[0]]
        finally:
            dm.pending = False

    # ---------- questions ----------
    def decide(self, question: str, document: str, llm=None) -> Decision:
        """Whether the local tiers should leave this question to the LLM."""
        dm = self.analyze(document, llm=llm)
        named = [t for t in question_types(question) if t in self.to_llm_types]
        if ROUTE_LEASES and dm.kind == "lease":
            return Decision(True, f"the document is a lease (p={dm.kind_p:.2f}); the local tiers can't find lease clauses "
                                  f"reliably, so the LLM reads it", named, dm.summary())
        if ROUTE_HIGH_RISK and named and dm.kind not in CONSUMER:
            return Decision(True, f"the question is about {', '.join(named[:3])}, a high-risk business clause the local tiers "
                                  f"can't find reliably, so the LLM reads it", named, dm.summary())
        return Decision(False, "", named, dm.summary())
