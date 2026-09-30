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


def test_a_termination_date_is_not_the_start_date():
    doc = ("This Sponsorship Agreement is made between Constellation and the HOF Entities.\n"
           "4. TERMINATION. Constellation may terminate this Agreement, effective as of December 31, 2023, in the "
           "event that the investment is not recovered.")
    llm = FakeLLM(choice={"none of these": 0.95})
    a = Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False).answer(
        "When does this agreement become effective?", doc)
    assert a.answer != "December 31, 2023", a.reason
    started = Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False).answer(
        "When does this agreement become effective?", doc.replace("4. TERMINATION.", "This Agreement shall "
                                                                  "commence on January 1, 2020.\n4. TERMINATION."))
    assert (started.path, started.answer) == ("span", "January 1, 2020"), started.reason
    assert "effective date" in started.reason


def test_the_llm_picks_among_candidates_when_the_rule_cannot():
    doc = LEASE_FACTS + "\n9. PETS. A pet fee of $300 applies; the pet deposit is $500 per animal."
    llm = FakeLLM(choice={"$500": 0.93, "$300": 0.05, "none of these": 0.02})
    a = Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False).answer(
        "How much is the pet deposit per animal?", doc)
    assert (a.path, a.answer, a.llm_calls) in (("llm_span", "$500", 1), ("span", "$500", 0))


# ---- Tier 2: a reader copies deferred facts from the text, Jev checks them (router/reader.py)

PETS = LEASE_FACTS + "\n9. PETS. If Tenant keeps a pet, Tenant shall pay a pet deposit of $500 per animal."  # a condition: the rule defers


class FakeReader:
    name = "fake reader"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply, error, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


def _tier2(reader, noul=0.95):
    llm = FakeLLM(choice={"none of these": 0.9}, noul={"Question about this text": noul})
    return Harness(llm, llm, StubBank(), None, pretier0=True, classifiers=False, reader=reader), llm


def test_tier2_answers_a_deferred_fact_when_jev_confirms_it():
    h, llm = _tier2(FakeReader('{"answer": "$500 per animal", "clause": null}'))
    a = h.answer("How much is the pet deposit?", PETS)
    assert (a.path, a.answer, a.confidence) == ("reader", "$500 per animal", 0.95), a.reason
    assert "9. PETS" in a.evidence[0]["text"] and a.reader["check"] == 0.95
    assert a.llm_calls == 2  # Jev's pick among candidates (none), then its check


def test_tier2_defers_when_jev_doubts_the_clause_states_it():
    h, _ = _tier2(FakeReader('{"answer": "$500 per animal", "clause": null}'), noul=0.3)
    a = h.answer("How much is the pet deposit?", PETS)
    assert a.path == "deferred" and "Jev doubts" in a.reason and a.reader["check"] == 0.3


def test_tier2_drops_an_answer_not_copied_from_the_text():
    h, llm = _tier2(FakeReader('{"answer": "$450", "clause": 1}'))
    a = h.answer("How much is the pet deposit?", PETS)
    assert a.path == "deferred" and "word for word" in a.reason
    assert not [c for c in llm.log if c[0] == "noul"]  # nothing to check


def test_tier2_outage_defers_instead_of_failing():
    from router.reader import ReaderError
    h, _ = _tier2(FakeReader(error=ReaderError("Fireworks failed: HTTP 503")))
    a = h.answer("How much is the pet deposit?", PETS)
    assert a.path == "deferred" and "unavailable" in a.reason


EFFECTIVE = "When does this agreement become effective?"


@pytest.mark.parametrize("answer, clause, question, answer_type, fits", [
    # the document's own date: the sentence names the document, and no other event is dated
    ("the Closing Date", "This Agreement shall become effective on the Closing Date.", EFFECTIVE, "DATE", True),
    ("July 1, 2019", "AGREEMENT PERIOD Effective Date: July 1, 2019", EFFECTIVE, "DATE", True),
    ("December 9, 1996", "This amendment to Section 2 of the Co-Branding Agreement is made effective December 9, 1996 "
                         "by and between PC Quote, Inc. and HyperFeed.", EFFECTIVE, "DATE", True),
    ("effective as of the date hereof", 'The term of this Agreement, unless mutually extended by the Parties or unless '
                                        'sooner terminated as provided herein, shall commence effective as of the date '
                                        'hereof.', EFFECTIVE, "DATE", True),
    ("effective upon creation", "Turpin does hereby assign and transfer to the Company, effective upon creation, all "
                                "right, title and interest in the Work Product.", EFFECTIVE, "DATE", False),
    ("Effective as of the Closing Date", "Effective as of the Closing Date, Equifax agrees to transfer to Certegy all "
                                         "right, title and interest in the Assets.", EFFECTIVE, "DATE", False),
    ("as of the Effective Date", 'This License Agreement (the "Agreement") is made as of the Effective Date.', EFFECTIVE,
     "DATE", False),  # restates the question
    ("December 31, 2023", "Constellation may terminate this Agreement, effective as of December 31, 2023.", EFFECTIVE,
     "DATE", False),
    # other types: a value of the type asked for, redacted or in capitals
    ("for the duration of this Agreement", "CBC warrants it will retain all approvals for the duration of this "
                                           "Agreement.", "How long does the warranty last?", "DURATION", False),
    ("[* * *] days", "Either party may give notice [* * *] days before the end of the term.",
     "How much notice is needed to prevent automatic renewal?", "DURATION", True),
    ("THE STATE OF DELAWARE", "THIS AGREEMENT IS GOVERNED BY THE LAWS OF THE STATE OF DELAWARE.",
     "Which state's law governs this agreement?", "JURISDICTION", True),
    ("no more than 5%", "The Processor may raise the monthly fee each year by no more than 5%.",
     "By how much can the Processor raise the monthly fee each year?", "MONEY", True),
])
def test_tier2_answers_must_fit_the_question(answer, clause, question, answer_type, fits):
    from router.reader import fits as fits_
    assert fits_(answer, clause, question, answer_type) is fits


def test_tier2_only_reads_fact_types_it_was_measured_on():
    reader = FakeReader('{"answer": "Tenant", "clause": null}')
    h, _ = _tier2(reader)
    a = h.answer("What color is the front door?", PETS)
    assert a.path != "reader" and not reader.prompts


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


# ---- "or" questions the wording doesn't settle: the document decides

GALLERY = ("3. DELIVERY OF IMAGES. The Photographer shall deliver a sneak peek of at least twenty-five (25) edited "
           "images within seventy-two (72) hours after the wedding. The full gallery of edited images shall be "
           "delivered via an online gallery within sixty (60) days after the wedding date.\n"
           "5. ACCOUNTINGS. Additional accountings within the same calendar year cost $25.00 each.")


def test_the_document_decides_which_one_when_it_states_exactly_one():
    f = classify("Does the Photographer deliver the images on a USB drive or through an online gallery?")
    assert f.leaf == "boolean" and f.maybe_options == ["on a USB drive", "through an online gallery"]
    a = Harness(FakeLLM(), FakeLLM(), StubBank(), None, pretier0=True, classifiers=False).answer(f.question, GALLERY)
    assert (a.path, a.answer, a.llm_calls) == ("choice", "through an online gallery", 0)


def test_an_option_stated_about_something_else_does_not_decide():
    # "$25.00" is what ADDITIONAL accountings cost; the first one isn't named here, so yes/no it stays
    a = Harness(FakeLLM(noul={"": 0.5}), FakeLLM(noul={"": 0.5}), StubBank(), None, pretier0=True,
                classifiers=False).answer("Is the first accounting of disclosures each year free or $25.00?", GALLERY)
    assert a.path != "choice"


@pytest.mark.parametrize("question", [
    "Is the license irrevocable or perpetual?",  # LegalBench: either one answers yes... but no permission word,
])
def test_either_questions_keep_yes_no_when_the_text_states_both(question):
    doc = "Licensor grants Licensee a perpetual, irrevocable license to use the Software."
    a = Harness(FakeLLM(), FakeLLM(), StubBank(), None, pretier0=True, classifiers=False).answer(question, doc)
    assert a.path != "choice"


@pytest.mark.parametrize("question", [
    "Can the receiving party share confidential information with its consultants or advisors?",
    "Must a party share revenue or profits with the other party?",
    "Is a party entitled to liquidated damages or a termination fee?",
    "Is there a clause requiring arbitration or mediation?",
    "Is termination on 30 days' notice possible, or not?",
])
def test_permission_obligation_and_presence_questions_are_never_undecided(question):
    assert classify(question).maybe_options == []
