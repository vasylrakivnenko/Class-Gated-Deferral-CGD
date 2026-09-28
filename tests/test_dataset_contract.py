"""The pipeline's input contract.

The pipeline grew around four OCL streams, so its label table, its loader and
its provenance pins were all reachable from inside the analysis. Adding
banking77 then required a runner that mutated the pipeline's own state before
calling it (`RP.LABEL_NAMES["banking77"] = names`). Each test here pins one half
of the inversion: a dataset is adapted into `Dataset`, and the pipeline is edited
for a bug or a method change, never to accommodate a source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from downshift.dataset import Dataset, DatasetError, available, load, register  # noqa: E402


def _ok(**kw):
    base = dict(name="t", texts=np.array(["a", "b", "c", "d"], dtype=object),
                gold=np.array([0, 1, 0, 1]), label_names={0: "neg", 1: "pos"})
    base.update(kw)
    return Dataset(**base)


def test_a_valid_dataset_reports_itself():
    d = _ok(source="unit test")
    assert (d.n, d.n_classes, d.has_expert) == (4, 2, False)
    assert d.describe()["source"] == "unit test"


def test_texts_and_labels_must_be_the_same_length():
    """A filter applied to one array and not the other shifts every label by the
    number of dropped rows, and nothing downstream can detect it."""
    with pytest.raises(DatasetError, match="texts against"):
        _ok(texts=np.array(["a", "b"], dtype=object))


def test_labels_must_be_contiguous_from_zero():
    """Every per-class array in the pipeline indexes by position, so a gap in the
    ids silently reads the wrong class."""
    with pytest.raises(DatasetError, match="not 0\\.\\."):
        _ok(gold=np.array([0, 2, 0, 2]), label_names={0: "a", 2: "b"})


def test_every_class_needs_a_name():
    """Names are used verbatim as NLI hypotheses. A missing one is a quietly
    weaker candidate rather than an error at run time."""
    with pytest.raises(DatasetError, match="no label name"):
        _ok(label_names={0: "neg"})
    with pytest.raises(DatasetError, match="empty label name"):
        _ok(label_names={0: "neg", 1: "  "})


def test_expert_must_align_and_stay_inside_the_label_set():
    with pytest.raises(DatasetError, match="expert predictions against"):
        _ok(expert=np.array([0, 1]))
    with pytest.raises(DatasetError, match="do not appear in gold"):
        _ok(expert=np.array([0, 1, 0, 7]))


def test_registry_rejects_unknown_names_with_the_list():
    with pytest.raises(DatasetError, match="registered:"):
        load("not-a-dataset")


def test_an_adapter_must_return_a_dataset():
    register("bad-adapter")(lambda: {"texts": []})
    with pytest.raises(DatasetError, match="not Dataset"):
        load("bad-adapter")


def _code_only(path: Path) -> str:
    """Source with comments and docstrings removed.

    Comments naming a dataset are wanted -- "measured on banking77" records where
    a rule came from and is the house style. What must not appear is a dataset
    name the code branches on.
    """
    import io as _io, tokenize
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and tok.line.strip().startswith(tok.string[:3]):
                continue                      # a docstring on its own line
            out.append(tok.string)
    return "\n".join(out)


def test_the_engine_holds_no_dataset_specific_knowledge():
    """The check that keeps the pipeline fixed. If a source name is reachable
    from engine code -- not from a comment recording a measurement -- the
    inversion has leaked back."""
    src = _code_only(ROOT / "src" / "downshift" / "engine.py")
    for token in ("imdb", "hatespeech", "isear", "fever", "banking77",
                  "reconstruct", "LABEL_NAMES", "ISEAR_TO_ID", "load_streams"):
        assert token not in src, f"engine.py branches on {token!r}"


def test_adapters_carry_the_dataset_specific_knowledge():
    src = (ROOT / "ocl_compare" / "adapters.py").read_text()
    for token in ("banking77", "hatespeech", "ISEAR_TO_ID"):
        assert token in src, token
    # and the registry is the only way in
    assert "@register(" in src or "register(" in src


def test_train_on_llm_refuses_a_dataset_with_no_teacher():
    from downshift import engine as E
    d = _ok(name="noexpert")
    with pytest.raises(ValueError, match="needs expert predictions"):
        E.analyse(d, train_on="llm")
