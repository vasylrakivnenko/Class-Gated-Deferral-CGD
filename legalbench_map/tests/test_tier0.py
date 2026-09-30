"""Tier 0's question rewriting and decisions against a scripted NLI model:
spaCy's small English model is needed, the NLI model is not."""
from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")

from router.harness import Harness
from router.tier0 import Tier0, to_statement
from tests.test_router import CONTRACT, FakeLLM, StubBank, _route

NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

LEASE = ("Tenant shall pay rent on the first day of each calendar month.\n\n"
         "Tenant shall not sublet the premises.\n\n"
         "Tenant shall not assign this Lease without Landlord's prior written consent.")

YES = {"entailment": 0.97, "neutral": 0.02, "contradiction": 0.01}
NO = {"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97}
UNSURE = {"entailment": 0.3, "neutral": 0.6, "contradiction": 0.1}


class FakeNLI:
    """Scores a premise by the first `rules` key it contains; UNSURE otherwise."""

    def __init__(self, rules):
        self.rules = rules
        self.premises = []

    def __call__(self, premises, hypothesis):
        self.premises += premises
        return [next((v for k, v in self.rules.items() if k in p), UNSURE) for p in premises]


def tier0(rules, answer_no=True):
    return Tier0(nli=FakeNLI(rules), nlp=NLP, answer_no=answer_no)


@pytest.mark.parametrize("question, statement", [
    ("Can the tenant sublet the premises?", "The tenant can sublet the premises."),
    ("Does the tenant have to pay rent monthly?", "The tenant does have to pay rent monthly."),
    ("Is the landlord responsible for repairing the roof?", "The landlord is responsible for repairing the roof."),
    ("Are pets allowed on the premises?", "Pets are allowed on the premises."),
    ("Is the term of the lease three years?", "The term of the lease is three years."),
    ("Is there a termination fee?", "There is a termination fee."),
    ("Can the landlord unreasonably withhold consent?", "The landlord can unreasonably withhold consent."),
    ("Is it true that pets are not allowed?", "Pets are not allowed."),
    ("Can the licensee audit the licensor's books?", "The licensee can audit the licensor's books."),
    ("Can either party terminate for convenience?", "Either party can terminate for convenience."),
])
def test_yes_no_questions_become_statements(question, statement):
    assert to_statement(question, NLP) == statement


@pytest.mark.parametrize("question", ["What is the governing law?", "Who pays for repairs?", "Rent?", "Tell me about rent."])
def test_other_questions_are_not_attempted(question):
    assert to_statement(question, NLP) is None
    r = tier0({}).answer(question, LEASE)
    assert not r.fired and "can restate" in r.reason


def test_contradicting_sentence_answers_no_with_evidence():
    r = tier0({"shall not sublet": NO}).answer("Can the tenant sublet the premises?", LEASE)
    assert (r.fired, r.answer, r.confidence) == (True, "no", 0.97)
    assert r.evidence[0]["text"] == "Tenant shall not sublet the premises."
    assert r.n_units == 3


def test_entailing_sentence_answers_yes():
    r = tier0({"pay rent": YES}).answer("Does the tenant have to pay rent monthly?", LEASE)
    assert (r.fired, r.answer) == (True, "yes")


def test_exception_in_the_deciding_sentence_defers():
    r = tier0({"assign": NO}).answer("Can the tenant assign the lease?", LEASE)
    assert not r.fired and "without Landlord's prior written consent" in r.reason
    assert r.evidence[0]["label"] == "no"


def test_disagreeing_sentences_defer():
    doc = "Tenant may sublet the premises to an affiliate.\n\nTenant shall not sublet the premises."
    r = tier0({"may sublet": YES, "not sublet": NO}).answer("Can the tenant sublet the premises?", doc)
    assert not r.fired and "disagree" in r.reason


def test_nothing_over_the_threshold_defers():
    r = tier0({}).answer("Can the tenant sublet the premises?", LEASE)
    assert not r.fired and "no sentence settles it" in r.reason


def test_only_sentences_sharing_a_content_word_are_scored():
    nli = FakeNLI({})
    Tier0(nli=nli, nlp=NLP).answer("Can the tenant keep pets?", "Rent is due monthly.\n\nTenant may keep one cat.")
    assert nli.premises == ["Tenant may keep one cat."]
    r = Tier0(nli=nli, nlp=NLP).answer("Is there a termination fee?", "Rent is due monthly.")
    assert not r.fired and r.n_scored == 0


def test_harness_answers_from_tier0_without_llm_calls():
    llm = FakeLLM()
    a = Harness(llm, llm, StubBank(), tier0({"shall not sublet": NO})).answer("Can the tenant sublet the premises?", LEASE)
    assert (a.path, a.answer, a.llm_calls, a.route) == ("tier0", "no", 0, None)
    assert a.asked == "The tenant can sublet the premises." and a.tier0["fired"]


def test_harness_falls_through_when_tier0_defers_and_keeps_its_trace():
    llm = FakeLLM(route={"none_of_these": 0.9, "cuad_audit_rights": 0.1}, noul={"Is this contract": 0.2})
    a = Harness(llm, llm, StubBank(), tier0({})).answer("Is this contract governed by Delaware law?", CONTRACT)
    assert a.path == "llm_fallback" and a.llm_calls == 2
    assert a.tier0 is not None and not a.tier0["fired"]


def test_a_sentence_sharing_only_the_subject_cannot_decide():
    # The subletting clause "contradicts" the assignment question; the
    # assignment clause, the one on topic, carries an exception.
    r = tier0({"sublet": {**NO, "contradiction": 0.99}, "assign": NO}).answer("Can the tenant assign the lease?", LEASE)
    assert not r.fired and "condition or exception" in r.reason
    r = tier0({"sublet": NO}).answer("Can the tenant pay rent late?", LEASE)
    assert not r.fired


AUDIT = ("Licensee shall not have the right to audit the books and records of Licensor relating to this "
         "Agreement, no more than once per calendar year.")


def test_a_sentence_about_another_party_cannot_decide():
    r = tier0({"audit": NO}).answer("Can the licensor audit the licensee's books?", AUDIT)
    assert not r.fired and "about Licensee, not Licensor" in r.reason
    r = tier0({"audit": NO}).answer("Can the licensee audit the licensor's books?", AUDIT)
    assert (r.fired, r.answer) == (True, "no")


def test_a_sentence_about_every_party_can_decide_for_one():
    doc = "Licensor and Licensee are the parties. Neither party may assign this Agreement."
    r = tier0({"assign": NO}).answer("Can the licensee assign the agreement?", doc)
    assert (r.fired, r.answer) == (True, "no")


def test_pronoun_subject_defers_when_the_document_names_parties():
    r = tier0({"audit": NO}).answer("Can we audit their books?", AUDIT)
    assert not r.fired and 'which party "we"' in r.reason


@pytest.mark.parametrize("question", ["Is the right to audit books discussed here?", "Does the clause mention audit rights?",
                                      "Is there an audit clause?", "Does the lease say anything about pets?"])
def test_topic_questions_are_not_attempted(question):
    r = tier0({"audit": NO}).answer(question, AUDIT)
    assert not r.fired and "covers a topic" in r.reason


def test_subject_with_an_infinitive_is_not_split():
    assert to_statement("Is the obligation to pay rent suspended during repairs?", NLP) is None
    assert to_statement("Is the tenant required to pay rent?", NLP) == "The tenant is required to pay rent."


@pytest.mark.parametrize("sentence", ["Tenant may sublet the premises only with Landlord's approval.",
                                      "During the first year, Tenant may sublet the premises.",
                                      "Tenant may sublet the premises if Landlord agrees.",
                                      "Tenant may sublet the premises within the first year.",
                                      "Tenant may, in its sole discretion, sublet the premises.",
                                      "Upon a change of control, Tenant may sublet the premises.",
                                      "Tenant may not sublet the premises without the respective Landlord's prior written consent."])
def test_scope_limits_defer(sentence):
    r = tier0({"sublet": YES, "not sublet": NO}).answer("Can the tenant sublet the premises?", sentence)
    assert not r.fired and "condition or exception" in r.reason


def test_upon_notice_is_not_a_condition():
    doc = "Tenant may, upon thirty (30) days' prior written notice, sublet the premises."
    assert tier0({"sublet": YES}).answer("Can the tenant sublet the premises?", doc).fired


def test_no_answers_are_off_by_default():
    r = Tier0(nli=FakeNLI({"shall not sublet": NO}), nlp=NLP).answer("Can the tenant sublet the premises?", LEASE)
    assert not r.fired and "answers are off" in r.reason and r.evidence[0]["label"] == "no"
