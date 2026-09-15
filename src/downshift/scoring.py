"""Pluggable scorers, so a task can be something other than pick-one-of-N.

Everything in the harness so far assumed single-label classification: the
program pinned its output to `Literal[labels]` and the metric did string
equality. That works for sentiment and intent, and it is exactly why both of
those benchmarks were winnable by TF-IDF -- if a task can be expressed as
"choose one of N", a linear model over n-grams can usually learn it given
enough labels, and the LLM has no structural advantage to demonstrate.

The tasks where an LLM is genuinely the right tool produce something a
bag-of-words model cannot even represent: a span of text, a composed answer,
a JSON object. Those need scorers that are not `==`.

Every scorer here is deterministic code. That is deliberate and load-bearing:
scoring with an LLM judge would make a cheap-vs-expensive comparison circular,
since the judge is itself one of the models under test (or a more expensive
one whose agreement we would then be assuming). If a task cannot be scored by
code, that is a strike against using it, not a reason to reach for a judge.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter

# ── normalisation ──────────────────────────────────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
# Built once, like _ARTICLES. Inline, `ch not in set(string.punctuation)`
# reconstructed a 32-element set for EVERY CHARACTER of every answer scored.
_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lowercase, strip articles/punctuation/extra space.

    This is the long-standing convention for span-answer QA, and using the
    standard version matters: it is what published numbers on these datasets
    are computed with, so our results stay comparable to them.
    """
    if s is None:
        return ""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in _PUNCTUATION)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


# ── scorers: each returns (score in [0,1], short explanation) ──────────────

def exact_match(pred, gold, **_) -> tuple[float, str]:
    """Strict equality after normalisation."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    return (1.0 if p == g else 0.0), ("exact match" if p == g else f"expected '{g}', got '{p}'")


def token_f1(pred, gold, **_) -> tuple[float, str]:
    """Token-overlap F1 — the standard partial-credit metric for span QA.

    Exact match alone is brutally unforgiving on free-form answers ("Rome" vs
    "in Rome" scores zero), which would compress every model toward the floor
    and hide the differences we are trying to measure. F1 is reported alongside
    it by every QA paper for exactly this reason.
    """
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt:
        return (1.0 if pt == gt else 0.0), "empty answer"
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0, f"no token overlap with '{' '.join(gt)}'"
    precision, recall = same / len(pt), same / len(gt)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, f"F1={f1:.2f} (p={precision:.2f} r={recall:.2f}) vs '{' '.join(gt)}'"


def json_schema_valid(pred, gold=None, required_keys: tuple[str, ...] = (), **_):
    """Does the output parse as JSON and carry the required keys?

    Reported separately from accuracy because they are different failure
    modes with different fixes. A model that returns malformed JSON needs a
    format fix (adapter, retry, constrained decoding); a model that returns
    valid JSON with wrong values needs a better prompt or a better model.
    Collapsing both into one number is what makes "reliability" claims mushy.
    """
    text = str(pred or "").strip()
    m = re.search(r"\{.*\}", text, re.S)     # tolerate prose around the object
    if not m:
        return 0.0, "no JSON object found in output"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return 0.0, f"invalid JSON: {e.msg}"
    if not isinstance(obj, dict):
        return 0.0, "JSON is not an object"
    missing = [k for k in required_keys if k not in obj]
    if missing:
        return 0.0, f"missing keys: {missing}"
    return 1.0, "schema valid"


def field_values_equal(a, b) -> bool:
    """Compare two extracted FIELD values with type awareness.

    `normalize_answer` is the SQuAD convention and is kept exactly as it is,
    because published span-QA numbers are computed with it. But it strips all
    punctuation, which is wrong for the values extraction tasks actually pull
    out of documents:

        "100"        vs "1.00"        -> both "100"       -> scored EQUAL
        "2024-01-02" vs "20240102"    -> both "20240102"  -> scored EQUAL
        "$1,000"     vs "$1.000"      -> both "1000"      -> scored EQUAL

    An invoice-amount field that scores 100 and 1.00 as a perfect match is not
    a partially-correct scorer, it is a broken one, and it fails in the
    flattering direction. So: numbers compare numerically, and anything that
    survives digit-and-separator stripping as a pure digit string compares
    exactly. Everything else falls back to the SQuAD text convention.
    """
    sa, sb = ("" if a is None else str(a)).strip(), ("" if b is None else str(b)).strip()
    if sa == sb:
        return True

    def as_number(s):
        t = s.replace(",", "").replace("$", "").replace("%", "").strip()
        try:
            return float(t)
        except ValueError:
            return None

    # Date-ish / identifier-ish FIRST. It has to precede the numeric branch:
    # "20240102" parses as a float while "2024-01-02" does not, so the
    # mixed-numeric rejection below would call the same date a mismatch. The
    # len >= 6 floor is what keeps this from swallowing "100" vs "1.00", whose
    # digit strings are only 3 long.
    da = "".join(ch for ch in sa if ch.isdigit())
    db = "".join(ch for ch in sb if ch.isdigit())
    if da and da == db and len(da) >= 6:
        sep_a = "".join(ch for ch in sa if not ch.isdigit())
        sep_b = "".join(ch for ch in sb if not ch.isdigit())
        if set(sep_a) <= set("-/. ") and set(sep_b) <= set("-/. "):
            return True

    na, nb = as_number(sa), as_number(sb)
    if na is not None and nb is not None:
        return na == nb
    # One numeric and one not is a mismatch, never a text-normalised "hit".
    if (na is None) != (nb is None):
        return False

    return normalize_answer(sa) == normalize_answer(sb)


def json_field_f1(pred, gold, required_keys: tuple[str, ...] = (), **_):
    """Field-level F1 between a predicted JSON object and a gold one.

    Partial credit per field, so a model that gets 4 of 5 fields right is not
    scored the same as one that returns garbage -- which is the distinction a
    buyer actually cares about when deciding whether a cheaper model is usable.
    """
    ok, why = json_schema_valid(pred, required_keys=required_keys)
    if ok == 0.0:
        return 0.0, why
    obj = json.loads(re.search(r"\{.*\}", str(pred), re.S).group(0))
    gold_obj = gold if isinstance(gold, dict) else json.loads(str(gold))
    keys = set(obj) | set(gold_obj)
    if not keys:
        return 1.0, "both empty"
    # Compared once per key. `hits` and `wrong` were two separate passes, so
    # every field went through field_values_equal (which normalises, digit-
    # strips and float-parses both sides) twice.
    wrong = [k for k in keys
             if not field_values_equal(obj.get(k, ""), gold_obj.get(k, ""))]
    hits = len(keys) - len(wrong)
    score = hits / len(keys)
    return score, f"{hits}/{len(keys)} fields correct" + (f"; wrong: {wrong[:4]}" if wrong else "")


def any_of_exact(pred, gold, **_) -> tuple[float, str]:
    """Gold is a list of acceptable answers (QA datasets often ship several)."""
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    p = normalize_answer(pred)
    hit = any(p == normalize_answer(g) for g in golds)
    return (1.0 if hit else 0.0), ("exact match" if hit
                                   else f"expected one of {[normalize_answer(g) for g in golds][:3]}, got '{p}'")


def any_of_f1(pred, gold, **_) -> tuple[float, str]:
    """Best token-F1 against any acceptable gold answer."""
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    best, why = 0.0, ""
    for g in golds:
        s, w = token_f1(pred, g)
        if s > best:
            best, why = s, w
    return best, why or "no overlap"


SCORERS = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "any_of_exact": any_of_exact,
    "any_of_f1": any_of_f1,
    "json_schema_valid": json_schema_valid,
    "json_field_f1": json_field_f1,
}


def get_scorer(name: str):
    if name not in SCORERS:
        raise KeyError(f"unknown scorer {name!r}; known: {sorted(SCORERS)}")
    return SCORERS[name]
