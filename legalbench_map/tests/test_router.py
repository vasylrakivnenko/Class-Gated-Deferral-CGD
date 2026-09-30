"""The router harness's decisions (gate, text check, policy, splitting,
aggregation) against a scripted LLM and stub classifiers: no network, no
fitted bank needed."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from router import menu
from router.harness import MAX_UNIT_CHARS, Harness, aggregate, route_scores, split_units
from router.menu import CRITERIA, TEXT_TYPES, family

BINARY = ["no", "yes"]
ABERCROMBIE = ["arbitrary", "descriptive", "fanciful", "generic", "suggestive"]


class FakeLLM:
    """Stands in for SystemOne. Routing calls (criteria = the menu) get
    `route`; other choice calls get `choice`; noul calls are answered by the
    first `noul` key the instructions start with."""

    def __init__(self, route=None, noul=None, choice=None):
        self.route_probs = route or {}
        self.noul_answers = noul or {}
        self.choice_probs = choice or {}
        self.calls = 0
        self.model = "fake"
        self.model_version = "fake-1"
        self.log = []

    def choice(self, state, instructions, criteria):
        self.calls += 1
        probs = self.route_probs if criteria is CRITERIA else self.choice_probs
        self.log.append(("choice", instructions, sorted(criteria)))
        return {"choice": max(probs, key=probs.get), "probabilities": dict(probs)}

    def noul(self, state, instructions):
        self.calls += 1
        self.log.append(("noul", instructions, state))
        for prefix, p in self.noul_answers.items():
            if instructions.startswith(prefix):
                return p
        raise AssertionError(f"unexpected noul question: {instructions!r}")


class StubModel:
    def __init__(self, classes, fn):
        self.classes = classes
        self._fn = fn

    def predict_proba(self, texts):
        return np.array([self._fn(t) for t in texts], dtype=float)


class StubBank:
    def __init__(self, models=None):
        self._models = models or {}
        self.manifest = {
            "cuad_audit_rights": _info("cuad", "free", BINARY),
            "personal_jurisdiction": _info("personal_jurisdiction", "llm", BINARY),
            "abercrombie": _info("abercrombie", "llm", ABERCROMBIE),
        }

    def model(self, task):
        return self._models[task]


def _info(fam, policy, classes):
    return {"family": fam, "policy": policy, "candidate": "tfidf_logreg", "classes": classes,
            "cv_metric": "balanced_accuracy", "cv_mean": 0.9, "published_best": 0.95,
            "published_best_model": "GPT-4", "gap_points": 5.0}


def _route(task, p_task=0.95, p_none=0.01, runner_up=None, p_runner=0.0):
    probs = {task: p_task, "none_of_these": p_none}
    if runner_up:
        probs[runner_up] = p_runner
    return probs


def _audit_model():
    return StubModel(BINARY, lambda t: [0.1, 0.9] if "audit" in t else [0.95, 0.05])


CONTRACT = ("1. TERM\n\nThis Agreement runs for five years.\n\n"
            "Licensee may audit Licensor's books and records once a year to confirm compliance.\n\n"
            "Each party shall keep the other's information confidential.")


def test_route_scores_counts_diversity_as_one_option():
    probs = {"diversity_1": 0.3, "diversity_2": 0.3, "diversity_3": 0.3, "none_of_these": 0.05, "hearsay": 0.05}
    p_none, margin = route_scores(probs)
    assert p_none == 0.05
    assert abs(margin - 0.85) < 1e-9


def test_classifier_path_splits_the_contract_and_finds_the_audit_clause():
    llm = FakeLLM(route=_route("cuad_audit_rights"), noul={"Is this text": 0.97})
    a = Harness(llm, llm, StubBank({"cuad_audit_rights": _audit_model()})).answer("Can we inspect their books?", CONTRACT)
    assert (a.path, a.answer, a.task) == ("classifier", "yes", "cuad_audit_rights")
    assert a.n_units == 3  # "1. TERM" is merged into the paragraph after it
    assert "audit" in a.evidence[0]["text"] and a.evidence[0]["p"] == 0.9
    assert a.asked == CRITERIA["cuad_audit_rights"]
    assert a.llm_calls == 2  # route + text check
    assert llm.log[1][1] == f"Is this text {TEXT_TYPES['cuad']}?"


def test_unsure_router_falls_back_to_the_users_question():
    llm = FakeLLM(route=_route("cuad_audit_rights", p_task=0.7, p_none=0.3), noul={"Can we": 0.8})
    a = Harness(llm, llm, StubBank()).answer("Can we inspect their books?", CONTRACT)
    assert (a.path, a.answer, a.asked) == ("llm_fallback", "yes", "Can we inspect their books?")
    assert "router unsure" in a.reason


def test_near_tie_between_two_tasks_falls_back():
    probs = _route("cuad_audit_rights", p_task=0.5, p_none=0.02, runner_up="cuad_cap_on_liability", p_runner=0.4)
    llm = FakeLLM(route=probs, noul={"Can we": 0.3})
    a = Harness(llm, llm, StubBank()).answer("Can we inspect their books?", CONTRACT)
    assert (a.path, a.answer) == ("llm_fallback", "no")
    assert abs(a.confidence - 0.7) < 1e-9


def test_off_menu_question_falls_back():
    llm = FakeLLM(route={"none_of_these": 0.9, "cuad_audit_rights": 0.1}, noul={"Is this contract": 0.2})
    a = Harness(llm, llm, StubBank()).answer("Is this contract governed by Delaware law?", CONTRACT)
    assert a.path == "llm_fallback" and "none of our tasks" in a.reason
    assert a.llm_calls == 2


def test_wrong_kind_of_document_falls_back():
    llm = FakeLLM(route=_route("cuad_audit_rights"), noul={"Is this text": 0.1, "Can we": 0.4})
    a = Harness(llm, llm, StubBank({"cuad_audit_rights": _audit_model()})).answer("Can we inspect their books?", "lol cats")
    assert a.path == "llm_fallback" and "doesn't look like" in a.reason
    assert a.asked == "Can we inspect their books?"


def test_weak_task_is_answered_by_the_llm_with_the_tasks_own_question():
    llm = FakeLLM(route=_route("personal_jurisdiction"), noul={"Given a fact pattern": 0.8})
    a = Harness(llm, llm, StubBank()).answer("Can Texas courts hear this case against me?", "Facts...")
    assert (a.path, a.answer, a.asked) == ("llm_task", "yes", CRITERIA["personal_jurisdiction"])
    assert a.llm_calls == 2  # route + answer, no text check


def test_weak_multiclass_task_uses_choice_over_its_labels():
    llm = FakeLLM(route=_route("abercrombie"), choice={"suggestive": 0.7, "arbitrary": 0.3})
    a = Harness(llm, llm, StubBank()).answer("Is 'Igloo' for coolers suggestive?", "Igloo for coolers")
    assert (a.path, a.answer, a.confidence) == ("llm_task", "suggestive", 0.7)
    assert llm.log[-1][2] == ABERCROMBIE


def test_separate_llm_calls_are_counted_once_each():
    router = FakeLLM(route=_route("cuad_audit_rights"))
    reader = FakeLLM(noul={"Is this text": 0.9})
    a = Harness(router, reader, StubBank({"cuad_audit_rights": _audit_model()})).answer("Audit rights?", CONTRACT)
    assert a.llm_calls == 2 and router.calls == 1 and reader.calls == 1


def test_split_units_paragraphs_lines_and_long_paragraphs():
    two_lines = "The first clause of this contract is long enough.\nThe second clause of this contract is long enough too."
    assert split_units(two_lines, "paragraph") == two_lines.split("\n")  # no blank lines: split on single newlines
    long_para = " ".join(f"Sentence number {i} says something about the contract." for i in range(80))
    units = split_units(long_para, "paragraph")
    assert len(units) > 1 and all(len(u) <= MAX_UNIT_CHARS for u in units)
    opinion = ("The court agrees with the reasoning of the panel below. "
               "We therefore overrule Smith v. Jones in its entirety. It is so ordered.")
    assert split_units(opinion, "sentence") == [  # no break after "v."; the short last sentence joins its neighbor
        "The court agrees with the reasoning of the panel below.",
        "We therefore overrule Smith v. Jones in its entirety. It is so ordered."]


def test_aggregate_binary_no_and_multiclass_skips_other():
    answer, conf, ev, probs = aggregate(["a", "b"], np.array([[0.8, 0.2], [0.6, 0.4]]), BINARY)
    assert (answer, round(conf, 3), ev[0]["text"], probs) == ("no", 0.6, "b", {"yes": 0.4, "no": 0.6})
    classes = ["arbitration", "other"]
    answer, conf, ev, probs = aggregate(["x", "y"], np.array([[0.1, 0.9], [0.7, 0.3]]), classes)
    assert (answer, conf, [e["text"] for e in ev], probs) == ("arbitration", 0.7, ["y"], {"arbitration": 0.7, "other": 0.3})


def test_every_menu_task_has_a_text_type():
    assert all(family(t) in TEXT_TYPES for t in CRITERIA if t != "none_of_these")


def test_routing_tests_use_the_harness_menu():
    path = Path(__file__).resolve().parent.parent / "results_jev_pilot" / "routing_menu.py"
    spec = importlib.util.spec_from_file_location("routing_menu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CRITERIA is menu.CRITERIA and module.INSTRUCTIONS is menu.INSTRUCTIONS
