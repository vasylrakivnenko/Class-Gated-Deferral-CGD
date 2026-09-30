"""The question tree (router/qtree.py): requests and requests for advice or a
legal conclusion leave before any lookup can answer them."""
from __future__ import annotations

import pytest

pytest.importorskip("spacy")

from router.harness import Harness
from router.qtree import classify
from tests.test_router import FakeLLM, StubBank


@pytest.mark.parametrize("question", [
    "Should the tenant pay?",  # the spec's trap
    "Should I sign this lease?",
    "Is the non-compete enforceable?",  # answered "yes" by Pre-Tier 0 before the gate existed
    "Is the limitation of liability valid?",  # spaCy parses "valid" as a modifier of "liability"
    "Is the fee reasonable?",
    "Is it legal for the landlord to enter the premises?",  # Tier 0 said yes at 0.955
    "Can the landlord legally enter the premises?",
    "Is this a fair contract?",
    "Could the non-compete be unenforceable?",
    "Does the policy comply with GDPR?",
    "Can they enforce the non-compete?",
    "Would you recommend signing?",
    "Do I have a case against the landlord?",
    "Is this arbitration clause standard for consumer loans?",  # "standard" parsed as a noun
    "Is the termination clause unusually harsh compared to typical licenses?",
    "Could this app be hacked?",
    "how secure is my data?",
])
def test_advice_and_legal_conclusions_are_judgmental(question):
    f = classify(question)
    assert (f.leaf, f.branch) == ("judgmental", "exit"), f.trace


@pytest.mark.parametrize("question", [
    "When must plaintiff pay?",  # the spec's trap: a lookup
    "Is the offer valid for 30 days?",  # a stated term
    "Must the fee be reasonable?",  # asks what the text requires
    "Does the contract require the notice period to be reasonable?",
    "Can the landlord unreasonably withhold consent?",
    "Do we have to give notice?",  # first person, still a lookup
    "Can we audit their books?",
    "Does the tenant have to comply with the building rules?",
    "Is normal wear and tear excluded?",  # "normal" modifies a noun
    "Is fair market value used?",
    "Is the license irrevocable or perpetual?",
    "Must the receiving party notify the disclosing party if it is required by law to disclose confidential information?",
    "Where should payments be sent?",  # "should" about a fact of the text
    "Has the app ever been hacked?",  # history, not a prediction
    "Who bears the risk of loss?",  # a contract term
    "Can you tell me whether the tenant can sublet?",  # a question inside a request
])
def test_lookups_are_not_exits(question):
    f = classify(question)
    assert f.leaf not in ("judgmental", "request"), (f.leaf, f.cue, f.trace)


@pytest.mark.parametrize("question", [
    "Can you list the deadlines?",  # the spec's trap
    "Summarize the lease.",
    "Please list all deadlines",
    "Highlight every clause that mentions termination.",  # spaCy tags "Highlight" as a noun
    "Can you explain what the release clause means in plain English?",
])
def test_tasks_are_requests(question):
    assert classify(question).leaf == "request"


def test_you_as_the_company_is_not_a_request():
    assert classify("Will you send me spam email?").leaf != "request"


class RecordingTier0:
    def answer(self, question, document):
        self.question = question
        raise AssertionError("a judgmental question must not reach Tier 0")


def test_judgmental_questions_skip_the_lookup_tiers():
    llm = FakeLLM(noul={"": 0.7})
    doc = "Employee agrees that the non-compete in this Section 7 is reasonable and enforceable."
    a = Harness(llm, llm, StubBank(), RecordingTier0(), pretier0=True, classifiers=False).answer(
        "Is the non-compete enforceable?", doc)
    assert (a.path, a.llm_calls, a.pretier0, a.tier0) == ("llm_judgment", 1, None, None)
    assert '"enforceable"' in a.reason and a.qtree["leaf"] == "judgmental"


def test_lookups_carry_their_kind():
    llm = FakeLLM()
    doc = "Licensee may audit the books and records of Licensor once per calendar year."
    a = Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False).answer(
        "Can the licensee audit the licensor's books?", doc)
    assert a.path == "pretier0" and a.qtree["leaf"] == "boolean"


# ---- M1: the rest of the tree

@pytest.mark.parametrize("question, leaf, answer_type", [
    ("When is rent due?", "span", "DATE"),
    ("How long is the notice period?", "span", "DURATION"),
    ("How much is the security deposit?", "span", "MONEY"),
    ("How many days' notice must the tenant give?", "span", "DURATION"),
    ("How often can the licensee audit?", "span", "FREQUENCY"),
    ("Who pays for repairs?", "span", "PARTY"),
    ("Which state's law governs the agreement?", "span", "JURISDICTION"),
    ("What is the late fee?", "span", "MONEY"),
    ("What does 'Confidential Information' mean?", "span", "DEFINITION"),
    ("Why can the landlord terminate early?", "backward", None),
    ("What happens if I pay rent late?", "forward", None),
    ("If I breach the confidentiality clause, what are the consequences?", "forward", None),
    ("How do I terminate the lease?", "process", None),
    ("What do I need to do to get my deposit back?", "process", None),
    ("Does the tenant or the landlord pay for water?", "choice", None),
    ("Is the governing law New York or Delaware?", "choice", None),
    ("Is the notice period 30 days or 60 days?", "choice", None),
    ("Is the bonus guaranteed or discretionary?", "choice", None),
    ("Do disputes go to arbitration or to court?", "choice", None),
    ("Is the license irrevocable or perpetual?", "boolean", None),  # either one: yes/no (LegalBench)
    ("Must a party share revenue or profits with the other party?", "boolean", None),
    ("Can either party terminate for breach or for convenience?", "boolean", None),
    ("Does a party have a right of first refusal or first offer?", "boolean", None),
    ("Is there a clause requiring disputes to go to arbitration or mediation?", "boolean", None),
    ("What is the rent and when is it due?", "unclassified", None),  # two questions
])
def test_leaves(question, leaf, answer_type):
    f = classify(question)
    assert (f.leaf, f.answer_type) == (leaf, answer_type), f.trace


@pytest.mark.parametrize("question, lookup", [
    ("Can you tell me whether the tenant can sublet?", "Can the tenant sublet?"),
    ("Tell me if the tenant pays for water", "Does the tenant pay for water?"),
    ("I want to know when rent is due", "When is rent due?"),
    ("So, can the tenant sublet?", "can the tenant sublet?"),
])
def test_polite_wrappers_are_unwrapped(question, lookup):
    f = classify(question)
    assert f.lookup == lookup and f.leaf != "request"


def test_choice_keeps_its_options():
    assert classify("Does the tenant or the landlord pay for water?").options == ["the tenant", "the landlord"]


def test_requests_and_deferred_kinds_cost_no_llm_call():
    llm = FakeLLM(noul={"": 0.9})
    h = Harness(llm, llm, StubBank(), RecordingTier0(), pretier0=True, classifiers=False)
    for q, path in [("Summarize the lease.", "declined"), ("Why can the landlord terminate early?", "deferred"),
                    ("What is the rent and when is it due?", "deferred")]:
        a = h.answer(q, "Tenant shall pay rent of $2,400 on the first day of each month.")
        assert (a.path, a.llm_calls, a.answer) == (path, 0, "not answered"), q
    assert llm.calls == 0


def test_unwrapped_questions_reach_the_lookup_tiers():
    llm = FakeLLM()
    doc = "Licensee may audit the books and records of Licensor once per calendar year."
    a = Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False).answer(
        "Can you tell me whether the licensee can audit the licensor's books?", doc)
    assert (a.path, a.answer, a.asked) == ("pretier0", "yes", "Can the licensee audit the licensor's books?")
    assert a.question == "Can you tell me whether the licensee can audit the licensor's books?"


# ---- M2: fact questions (router/spans.py)

LEASE_FACTS = ("This Lease Agreement is made between Landlord and Tenant as of January 1, 2025.\n"
               "1. TERM. The term of this Lease is twelve (12) months commencing on January 1, 2025.\n"
               "2. RENT. Tenant shall pay rent of $2,400 on the first day of each calendar month.\n"
               "3. LATE FEE. Any rent not received by the fifth day of the month shall incur a late fee of $100.\n"
               "5. UTILITIES. Tenant shall pay for electricity; Landlord shall pay for water.\n"
               "8. GOVERNING LAW. This Lease is governed by the laws of the State of New York.")


@pytest.mark.parametrize("question, answer", [
    ("When is rent due?", "the first day of each calendar month"),
    ("How much is the rent?", "$2,400"),  # the RENT clause, not the late-fee clause
    ("What is the late fee?", "$100"),
    ("How long is the lease term?", "twelve (12) months"),
    ("Who pays for water?", "Landlord"),
    ("Which state's law governs the lease?", "the State of New York"),
    ("What is the date of this agreement?", "January 1, 2025"),
])
def test_fact_questions_answered_by_rule_without_llm(question, answer):
    llm = FakeLLM()
    a = Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False).answer(question, LEASE_FACTS)
    assert (a.path, a.answer, a.llm_calls) == ("span", answer, 0), a.reason


@pytest.mark.parametrize("question", ["When does the lease expire?", "What is the pet deposit?"])
def test_facts_the_text_does_not_state_are_deferred(question):
    llm = FakeLLM(choice={"none of these": 0.9})
    a = Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False).answer(question, LEASE_FACTS)
    assert a.path == "deferred" and a.answer == "not answered", a.reason


def test_the_llm_picks_among_candidates_when_the_rule_cannot():
    doc = LEASE_FACTS + "\n9. PETS. A pet fee of $300 applies; the pet deposit is $500 per animal."
    llm = FakeLLM(choice={"$500": 0.93, "$300": 0.05, "none of these": 0.02})
    a = Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False).answer(
        "How much is the pet deposit per animal?", doc)
    assert (a.path, a.answer, a.llm_calls) in (("llm_span", "$500", 1), ("span", "$500", 0))


# ---- M3: choice questions (router/choice.py)

CHOICES = ("5. UTILITIES. Tenant shall pay for electricity; Landlord shall pay for water.\n"
           "6. NOTICE. Either party may terminate this Lease upon thirty (30) days' prior written notice.\n"
           "7. BONUS. The annual bonus is discretionary and is not guaranteed.\n"
           "9. GOVERNING LAW. This Lease is governed by the laws of the State of New York.")


@pytest.mark.parametrize("question, answer", [
    ("Does the tenant or the landlord pay for water?", "the landlord"),
    ("Who pays for electricity, the tenant or the landlord?", "the tenant"),
    ("Is the notice period 30 days or 60 days?", "30 days"),
    ("Is the bonus guaranteed or discretionary?", "discretionary"),  # "not guaranteed" doesn't state "guaranteed"
    ("Is the governing law New York or Delaware?", "New York"),  # "New" isn't the qualifier "new"
])
def test_choice_questions_answered_by_rule(question, answer):
    llm = FakeLLM()
    a = Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False).answer(question, CHOICES)
    assert (a.path, a.answer, a.llm_calls) == ("choice", answer, 0), a.reason


def test_choice_not_stated_is_deferred():
    llm = FakeLLM(choice={"neither / the text doesn't say": 0.95})
    a = Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False).answer(
        "Is the deposit refundable or nonrefundable?", CHOICES)
    assert a.path == "deferred"


def test_prepositional_options_are_whole_phrases():
    assert classify("Do disputes end up in court or in arbitration?").options == ["in court", "in arbitration"]


def test_an_age_is_not_a_duration():
    doc = ("We will promptly delete information associated with any account of a registered user under eighteen "
           "years of age. We retain account data for two (2) years after the account is closed.")
    a = Harness(FakeLLM(), FakeLLM(), StubBank(), None, pretier0=True, classifiers=False).answer(
        "How long do you retain account data?", doc)
    assert a.answer == "two (2) years"


@pytest.mark.parametrize("question, answer_type", [
    ("How old do you have to be to buy a Premium subscription?", "AGE"),
    ("What is the minimum age to open an account?", "AGE"),
    ("What's the maximum number of guests allowed at an event?", "CARDINAL"),
    ("What's the starting credit limit on the account?", "MONEY"),
    ("What is the time limit to file a claim?", "DURATION"),
])
def test_more_answer_types(question, answer_type):
    assert classify(question).answer_type == answer_type


def test_age_answers():
    doc = "Users must be at least sixteen (16) years of age to buy Premium. Premium costs $9.99 per month."
    a = Harness(FakeLLM(), FakeLLM(), StubBank(), None, pretier0=True, classifiers=False).answer(
        "How old must users be to buy Premium?", doc)
    assert (a.path, a.answer) == ("span", "sixteen (16) years of age")
