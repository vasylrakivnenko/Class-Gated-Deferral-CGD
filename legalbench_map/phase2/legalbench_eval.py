"""Verbatim port of the LegalBench scorer used for our three Phase 2 tasks.

Source: https://github.com/HazyResearch/legalbench/blob/main/evaluation.py
(cached locally as data/legalbench_evaluation.py). Only the two functions on
the exact-match balanced-accuracy path are ported: ``normalize`` and
``evaluate_exact_match_balanced_accuracy`` (exposed here as
``balanced_accuracy_exact_match``). Bodies are copied character-for-character
from the cached file; the ONLY edits are:

  * the ``PorterStemmer`` import is deferred into the ``stem=True`` branch.
    The exact-match path calls ``normalize(x, stem=False)`` (see
    ``evaluate_exact_match_balanced_accuracy`` in the source), so nltk is
    never touched for our tasks. nltk is not installed in this environment
    and we install nothing; ``stem=True`` raises ImportError with a clear
    message. Our three tasks' labels are single words ("yes"/"no"), for which
    stemming would be a no-op anyway.

tests/test_phase2_runner.py asserts that ``normalize`` here returns the same
output as the function in data/legalbench_evaluation.py on tricky inputs.
"""
from __future__ import annotations

import string
from typing import List

from sklearn.metrics import balanced_accuracy_score


def normalize(text: str, stem: bool = False) -> str:
    """
    Normalizes strings.

    Args:
        - text: text to normalize
        - stem: whether or not to apply a stemmer

    Returns: normalized text
    """

    # Remove punctuation
    text = str(text).translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = text.strip()

    # Make lower case
    text = text.lower()

    # Stem
    if stem:
        try:
            from nltk.stem.porter import PorterStemmer
        except ImportError as e:  # pragma: no cover - nltk deliberately not installed
            raise ImportError(
                "normalize(stem=True) needs nltk, which is not installed; the "
                "LegalBench exact-match path uses stem=False and never needs it"
            ) from e
        text = PorterStemmer().stem(text)
    return text


def balanced_accuracy_exact_match(generations: List[str], answers: List[str]):
    """
    Evaluates exact match using balanced_accuracy.
    (Ported from evaluate_exact_match_balanced_accuracy.)
    """
    normalized_answers = [normalize(a, stem=False) for a in answers]
    normalized_gens = [normalize(g, stem=False) for g in generations]
    return balanced_accuracy_score(normalized_answers, normalized_gens)


# Alias under the upstream name so grep-by-source-name still works.
evaluate_exact_match_balanced_accuracy = balanced_accuracy_exact_match
