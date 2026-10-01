"""Pre-Tier 0's regex checks: topic questions it answers, parties, and
rewording questions into a we/you document's voice."""
from __future__ import annotations

import json

import pytest

from router.harness import Harness
from router.pretier0 import check, find_parties, to_document_voice
from router.tier0 import Tier0Result
from tests.test_router import FakeLLM, StubBank

AUDIT = ("Licensee shall have the right, upon thirty (30) days' prior written notice, to audit the books and "
         "records of Licensor relating to this Agreement, no more than once per calendar year.")
NDA = 'This Agreement is between Acme Corp. ("Acme") and Beta Labs LLC ("Beta"). Each party shall keep the other\'s information confidential.'
POLICY = "We may share information about you with our advertising partners."


class RecordingTier0:
    def answer(self, question, document):
        self.question = question
        return Tier0Result(True, "yes", 0.99, "stub")


def test_find_parties_roles_and_defined_company_names():
    assert find_parties(AUDIT) == ["Licensee", "Licensor"]
    assert find_parties(NDA) == ["Acme", "Beta"]
    assert find_parties('the consulting firm (the "Consultant") and the "Effective Date" (the "Effective Date")') == ["Consultant"]
    assert find_parties("the tenant pays rent") == []  # lower-case role words aren't defined parties


def test_questions_are_turned_to_the_documents_voice():
    assert to_document_voice("Do they share my data with advertisers?") == "Do we share your data with advertisers?"
    assert to_document_voice("Do you sell my data?") == "Do we sell your data?"
    assert to_document_voice("Am I allowed to opt out?") == "Are you allowed to opt out?"
    assert to_document_voice("Can you delete your account?") is None  # "you" alone may mean "one"
    r = check("Can I delete my account?", POLICY)
    assert not r.fired and r.rewritten == "Can you delete your account?"


def test_we_is_read_as_the_one_party_that_does_the_action():
    r = check("Can we audit their books?", AUDIT)
    assert r.rewritten == "Can the Licensee audit the Licensor's books?" and r.parties == ["Licensee", "Licensor"]
    assert (r.fired, r.answer) == (True, "yes")  # and the frame settles it
    lease = "Tenant shall pay rent monthly. Landlord shall maintain the roof."
    assert check("Do we have to repair the roof?", lease).rewritten == "Does the Landlord have to repair the roof?"


def test_we_is_left_alone_when_no_single_party_does_the_action():
    both = AUDIT + " Licensor may audit Licensee's royalty reports once a year."
    assert check("Can we audit their books?", both).rewritten is None
    assert check("Can we terminate?", "Tenant shall pay rent monthly. Landlord shall maintain the roof.").rewritten is None


def test_harness_hands_every_question_to_tier0_in_the_documents_voice():
    llm = FakeLLM()
    t0 = RecordingTier0()
    a = Harness(llm, llm, StubBank(), t0, pretier0=True).answer("Can I delete my account?", POLICY)
    assert t0.question == "Can you delete your account?" and a.question == "Can I delete my account?"
    assert a.path == "tier0" and a.pretier0 is not None and not a.pretier0["fired"]
    a = Harness(llm, llm, StubBank(), t0, pretier0=True).answer("Can we audit their books?", AUDIT)
    assert (a.path, a.answer, a.asked) == ("pretier0", "yes", "Can the Licensee audit the Licensor's books?")


NO_AUDIT = ("Licensee shall have no right to audit the books and records of Licensor relating to this Agreement, "
            "no more than once per calendar year.")


@pytest.mark.parametrize("question", ["Is the right to audit books discussed here?", "Does the clause mention auditing of records?"])
def test_topic_is_discussed_when_one_sentence_has_every_word(question):
    r = check(question, "Rent is due monthly.\n\n" + NO_AUDIT)
    assert (r.fired, r.answer) == (True, "yes") and r.evidence[0]["text"] == NO_AUDIT


@pytest.mark.parametrize("question", ["Is termination discussed?",  # absent: may be a synonym, so defer
                                      "Is the right to audit books not discussed here?",  # negated
                                      "Can the licensee audit books?"])  # not a topic question
def test_topic_check_defers(question):
    assert "topic word" not in check(question, NO_AUDIT).reason  # (the frames may still answer)


def test_topic_words_must_share_one_sentence():
    assert not check("Is the right to audit rent discussed?", "Rent is due monthly. " + NO_AUDIT).fired


def test_harness_answers_topic_questions_without_models():
    llm = FakeLLM()
    t0 = RecordingTier0()
    a = Harness(llm, llm, StubBank(), t0, pretier0=True).answer("Is the right to audit books discussed here?", NO_AUDIT)
    assert (a.path, a.answer, a.llm_calls) == ("pretier0", "yes", 0) and not hasattr(t0, "question")


def test_covered_needs_a_word_for_the_text():
    doc = "Landlord shall provide Tenant with two parking spaces."
    assert not check("Are the parking spaces covered?", doc).fired
    assert check("Are parking spaces covered in this lease?", doc).fired


def test_result_is_json_serializable():
    """The page gets this dict as JSON; a frame's Items hold frozen sets."""
    r = check("Is the purpose of life discussed there?", AUDIT)
    assert not r.fired
    assert json.loads(json.dumps(r.to_dict()))["frames"]["frame"]["alts"][0][0]["stems"] == ["purpos"]


def test_a_verb_the_lexicon_reads_as_a_condition_is_not_dropped():
    # "save" is a condition word in the lexicon ("save as provided"); here it's the question's verb,
    # and "is health data mentioned?" is not the question. (PrivacyQA, M4 2026-09-30)
    doc = "We will ask for your explicit consent to share any sensitive personal information such as health."
    assert not check("does it save any of my health data?", doc).fired


# ---- topic words (2026-10-01): word families, not a 6-letter prefix; terms of art as terms

ROLES_DOC = ("The Employee shall report to the Chief Executive Officer. The Licensee may change the control panel "
             "settings at any time. Neither party shall assign this Agreement without consent.")


@pytest.mark.parametrize("question, fires", [
    ("Is the employer discussed here?", False),  # the 6-letter prefix read "employer" as "employee"
    ("Is the employee discussed here?", True),
    ("Is the assignor mentioned in the agreement?", False),  # "assignor" is a party, "assign" the act
    ("Is assignment discussed here?", True),  # assign / assignment: one family
    ("Is change of control discussed here?", False),  # a term of art, not "change" and "control" anywhere
])
def test_topic_words_keep_roles_apart_and_terms_whole(question, fires):
    assert check(question, ROLES_DOC).fired is fires


@pytest.mark.parametrize("question, doc", [
    ("Is change of control discussed here?", "Upon a change in effective control of Licensee, Licensor may terminate."),
    ("Is a minimum purchase commitment discussed here?", "Distributor will make purchases at least equal to the "
                                                         "Guaranteed Minimum Purchase amounts through committed orders."),
    ("Is the warranty period discussed here?", "The above warranties are valid for a period of one year."),
    ("Is termination for convenience discussed here?", "NETTAXI may terminate this Agreement for its convenience."),
    ("Does the clause mention third-party beneficiaries?", "Duval is an intended third party creditor beneficiary hereof."),
])
def test_topic_words_match_their_family(question, doc):
    assert check(question, doc).fired


def test_is_there_a_provision_isnt_answered_by_a_denial():
    # "Is there an audit clause?" asks whether the contract provides audits; NO_AUDIT denies them (2026-10-01, v6)
    assert not check("Is there an audit clause?", "Rent is due monthly.\n\n" + NO_AUDIT).fired
    assert check("Is there an audit clause?", "Licensee may audit the books and records of Licensor once a year.").answer == "yes"
