"""Pre-Tier 0 v2's frame matching (router/frames.py): lexicon, lemmas,
modality, blockers, and the answers Pre-Tier 0 gives from them."""
from __future__ import annotations

import pytest

from router.frames import answer, tokens
from router.pretier0 import check

POLICY = ("We may share information about you with our advertising partners and analytics providers so that they "
          "can show you ads that are more relevant to your interests. We do not sell your personal information. "
          "You can delete your account at any time from the settings page.")
AUDIT = ("Licensee shall have the right, upon thirty (30) days' prior written notice, to audit the books and records "
         "of Licensor relating to this Agreement, no more than once per calendar year.")
LEASE = ("Tenant shall pay rent on the first day of each calendar month. Tenant shall not sublet the premises. "
         "Tenant shall not assign this Lease without Landlord's prior written consent. Landlord shall maintain the roof.")
NDA = "The Receiving Party shall not disclose Confidential Information to any third party."


def run(question, document):
    """Pre-Tier 0 v2's answer, as (answer, qualifier)."""
    fr = check(question, document).frames
    return fr["answer"], fr["qualifier"]


@pytest.mark.parametrize("document, question, expected", [
    (POLICY, "Do they share my data with advertisers?", ("yes", "may")),
    (POLICY, "Do they share my data with third parties?", ("yes", "may")),  # advertisers are third parties
    (POLICY, "Do they sell my data?", ("no", None)),
    (POLICY, "Can I delete my account?", ("yes", None)),
    (AUDIT, "Can the Licensee audit the Licensor's books?", ("yes", None)),
    (LEASE, "Can the tenant sublet the premises?", ("no", None)),
    (LEASE, "Does the tenant have to pay rent?", ("yes", None)),
    (LEASE, "Does the landlord have to repair the roof?", ("yes", None)),  # repair ~ maintain
    (NDA, "Can the receiving party disclose confidential information?", ("no", None)),
])
def test_answers(document, question, expected):
    assert run(question, document) == expected


@pytest.mark.parametrize("document, question", [
    (POLICY, "Must they share my data with advertisers?"),  # "may" isn't "must"
    (AUDIT, "Can the Licensor audit the Licensee's books?"),  # the other party
    (LEASE, "Can the tenant assign the lease?"),  # "without Landlord's consent"
    (LEASE, "Does the landlord have to repair the elevator?"),  # no sentence names the elevator
    (LEASE, "Is the tenant exempt from paying rent?"),  # a negating word in the question
    (LEASE, "Can the tenant not sublet?"),
    (LEASE, "What rent does the tenant pay?"),
])
def test_defers(document, question):
    assert run(question, document) == (None, None)


def test_sentences_that_disagree_defer():
    doc = "Tenant may sublet the premises. Tenant shall not sublet the premises."
    assert check("Can the tenant sublet the premises?", doc).frames["reason"] == "sentences disagree"


def test_lemmas_cover_inflections_and_irregular_forms():
    assert tokens("shares sharing shared paid sold Licensor's") == ["share", "share", "share", "pay", "sell", "licensor"]


def test_a_settled_frame_answers():
    r = check("Do they sell my data?", POLICY)
    assert (r.fired, r.answer) == (True, "no") and r.frames["ms"] < 10 and "sell" in r.evidence[0]["text"]


@pytest.mark.parametrize("sentence", ["Tenant may sublet the premises only with approval.",
                                      "During the first year, Tenant may sublet the premises.",
                                      "Upon a change of control, Tenant may sublet the premises."])
def test_scope_limits_block(sentence):
    assert run("Can the tenant sublet the premises?", sentence) == (None, None)


def test_another_sentence_about_the_same_action_defers():
    doc = ("The Recipient shall not disclose Confidential Information to any third party. "
           "The Recipient may disclose Confidential Information to its legal counsel.")
    assert run("Can the recipient disclose confidential information to third parties?", doc) == (None, None)


def test_two_actions_defer():
    assert run("Can the licensee use and distribute the software?", "Licensee may use and copy the Software.") == (None, None)


def test_we_read_as_the_acting_party_reaches_v2():
    assert run("Can we audit their books?", AUDIT) == ("yes", None)


@pytest.mark.parametrize("document, question, expected", [
    # "a party" is any party; "prohibited from" asks about a ban
    ("Distributor shall not solicit any customers of Company.", "Is a party prohibited from soliciting the other party's customers?", ("yes", None)),
    ("Neither party shall disparage the other party.", "Is a party prohibited from disparaging the other party?", ("yes", None)),
    ("Supplier may solicit customers of Buyer.", "Is a party prohibited from soliciting the other party's customers?", ("no", None)),
    # "or" joins alternative actions
    ("Consultant shall not hire any employees of Client.", "Is a party prohibited from soliciting or hiring the other party's employees?", ("yes", None)),
    # properties
    ("Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the Software.", "Is the license non-transferable?", ("yes", None)),
    ("In no event shall either party's aggregate liability exceed the fees paid.", "Is a party's liability capped?", ("yes", None)),
    # a name the lexicon doesn't know, a pronoun actor, a verb + object action, whole-agreement boilerplate
    ("SpringCo shall maintain commercial general liability insurance.", "Is a party required to maintain insurance?", ("yes", None)),
    ("Each Party agrees that it shall carry adequate insurance.", "Is a party required to maintain insurance?", ("yes", None)),
    ("Subject to the terms and conditions of this Agreement, Licensor grants Licensee an irrevocable, perpetual license.", "Is the license irrevocable or perpetual?", ("yes", None)),
    ("Either party may terminate this Agreement at any time without cause upon thirty (30) days' written notice.", "Can a party terminate the agreement without cause?", ("yes", None)),
    ("The Receiving Party shall not disclose the existence of this Agreement to any third party.", "Is the receiving party prohibited from disclosing the existence of the agreement?", ("yes", None)),
])
def test_legal_frames(document, question, expected):
    assert run(question, document) == expected


@pytest.mark.parametrize("document, question", [
    # the party is the object of a passive, not the actor
    ("All copies in the possession of the Receiving Party shall be returned.", "Must the receiving party return or destroy the confidential information?"),
    # the actor isn't the subject of the action
    ("Franchisee shall protect the rights of Franchisor which may disparage nothing.", "Is a party prohibited from disparaging the other party?"),
    # another action sits between actor and action
    ("CytoDyn may terminate this Agreement if Vyera challenges any CytoDyn Patents.", "Is a party prohibited from challenging the other party's intellectual property?"),
    # a quantity idiom is not a negation, but "not" before the property is
    ("The license shall not be non-transferable.", "Is the license non-transferable?"),
    # a specific section is still a condition
    ("Subject to Section 9, Licensor grants Licensee a perpetual license.", "Is the license irrevocable or perpetual?"),
    # two actions joined by more than "or"
    ("Recipient may retain copies of Confidential Information.", "Can the receiving party retain copies of confidential information after returning or destroying it?"),
])
def test_legal_frames_defer(document, question):
    assert run(question, document) == (None, None)


def test_quantity_idioms_are_not_negations():
    doc = "Company shall, for not less than four years, maintain general liability insurance."
    assert run("Is a party required to maintain insurance?", doc) == ("yes", None)


def test_lemma_prefers_the_shortest_base():
    assert tokens("disclosing sharing") == ["disclose", "share"]  # not "disclosing" from "disclosing party"


def test_neither_nor_negates():
    doc = "Neither Acme nor Beta will grant to any third party any license to the jointly developed software."
    assert run("Does one party grant a license to the other party?", doc) != ("yes", None)


def test_nor_before_the_actor_negates():
    doc = "Neither party shall disclose the terms of this Agreement, nor shall either party use the other party's name in advertising."
    assert run("Can a party use the other party's name in advertising?", doc) == ("no", None)


def test_not_before_another_actor_does_not_negate_it():
    doc = ("All policies of insurance procured by Racing herein shall be written as primary policies, "
           "not contributing with or in excess of coverage that the Sponsor may carry.")
    assert check("Is a party required to maintain insurance?", doc).answer != "no"


# --- presence frames ("Does the agreement specify ...", "Is there ...", gerund subjects)

GOV = "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware."


def test_presence_specify_is_yes_when_one_sentence_names_everything():
    assert run("Does the agreement specify which law governs it?", "Rent is due monthly. " + GOV) == ("yes", None)
    assert run("Does the agreement specify which law governs it?", "Each party shall comply with applicable law.") == (None, None)


def test_presence_negated_sentence_defers_and_negative_question_needs_negation():
    assert run("Is there a third-party beneficiary to the agreement?",
               "There are no third party beneficiaries to this Agreement.") == (None, None)
    assert run("Does the agreement say that no license to the confidential information is granted to the receiving party?",
               "Nothing in this Agreement grants the Receiving Party any license to the Confidential Information.") == ("yes", None)


def test_presence_require_reads_not_without_consent_as_a_requirement():
    q = "Does assigning the agreement require the other party's consent?"
    assert run(q, "Neither party may assign this Agreement without the prior written consent of the other party.") == ("yes", None)
    assert run(q, "Either party may assign this Agreement without the consent of the other party.") == (None, None)


def test_when_questions_need_the_term_stated_as_a_date_or_duration():
    q = "Does the agreement specify when its initial term expires?"
    assert run(q, "The initial term of this Agreement shall be three (3) years from the Effective Date.") == ("yes", None)
    assert run(q, "Either party shall give notice at least ninety (90) days prior to the expiration of the Term.") == (None, None)
    assert run(q, "The Consultant may terminate the engagement by giving 120 days prior written notice.") == (None, None)


def test_catch_all_presence_leaves_questions_about_a_named_party_alone():
    doc = "Employer shall reimburse Employee for reasonable travel expenses."
    assert run("Must the employee reimburse the employer for travel expenses?", doc) == (None, None)


def test_catch_all_presence_defers_on_conditions():
    doc = "Employee shall receive severance equal to six months' salary if terminated without Cause."
    assert run("Is a party entitled to severance?", doc) == (None, None)


# --- values: numbers, amounts and dates must match

@pytest.mark.parametrize("question, doc", [
    ("Is the purchase price $500,000?", "The Purchase Price is Five Million Dollars ($5,000,000), payable at Closing."),
    ("Does the agreement expire on December 31, 2025?", "This Agreement shall expire on December 31, 2026."),
    ("Can the agreement be terminated with 30 days' notice?", "Either party may terminate this Agreement upon ninety (90) days' prior written notice."),
    ("Must the landlord repair the heating system within 24 hours?", "Landlord shall repair the heating system within 48 hours of notice."),
])
def test_values_in_the_question_must_be_in_the_sentence(question, doc):
    assert run(question, doc) == (None, None)


# --- conditions in the question are matched, not rejected

def test_question_condition_matches_the_clause_condition():
    q = "Can a party terminate the agreement if the other party undergoes a change of control?"
    assert run(q, "Licensor may terminate this Agreement upon a change of control of Licensee.") == ("yes", None)
    assert run(q, "Licensor may terminate this Agreement for convenience.") == (None, None)
    assert run(q, "Licensor may terminate this Agreement upon a change of control of Licensee, unless Licensee cures.") == (None, None)


def test_only_question_accepts_not_except():
    q = "Must the receiving party use the confidential information only for the purposes of the agreement?"
    assert run(q, "The Receiving Party shall use the Confidential Information solely for the purpose of evaluating the deal.") == ("yes", None)
    assert run(q, "The Receiving Party agrees not to use the Confidential Information except for the purpose of evaluating the deal.") == ("yes", None)


# --- grammar: passive agents, unlisted capitalized subjects

def test_passive_with_an_agent_is_the_agents_action():
    q = "Can a party terminate the agreement without cause?"
    assert run(q, "This Agreement may be terminated at any time without cause by either Acme or Beta on written notice.") == ("yes", None)


def test_passive_modality_is_the_verbs_own():
    doc = "The insurance shall not be limited in any way by reason of any insurance which may be maintained by Pretzel Time."
    assert check("Is a party required to maintain insurance?", doc).answer != "no"


def test_capitalized_subject_counts_as_a_party_for_a_party_question():
    doc = "Customer specifically agrees to maintain insurance coverage for any finished Products."
    assert run("Is a party required to maintain insurance?", doc) == ("yes", None)


# --- the LLM-grown lexicon and the long-document budget

def test_extra_lexicon_is_loaded_and_never_remaps_curated_forms():
    from router.frames import LEXICON, tokens
    assert "cede" in LEXICON[("ACTION", "ASSIGN")]  # from lexicon_extra.json
    assert tokens("accounts expires") == ["accounts", "expir"]


def test_too_many_candidate_sentences_defer():
    doc = "Either party may terminate this Agreement upon written notice. " * 80
    assert "too many" in check("Can a party terminate the agreement?", doc).frames["reason"]


def test_condition_is_judged_in_the_actions_clause():
    q = "Must the receiving party use the confidential information only for the purposes of the agreement?"
    doc = ("The Receiving Party shall limit disclosure to its employees who need to know, and only for that purpose; "
           "the Receiving Party agrees to use the same degree of protection it uses for its own information.")
    assert run(q, doc) == (None, None)


def test_date_questions_need_a_date():
    q = "Does the agreement specify the date on which it becomes effective?"
    assert run(q, "This Agreement shall become effective on the signing date.") == (None, None)
    assert run(q, "This Agreement shall become effective on January 1, 2021.") == ("yes", None)


def test_head_noun_is_the_noun_not_its_adjective():
    q = "Does the agreement specify when its initial term expires?"
    assert run(q, "This initial order shall be received no later than April 1, 2000.") == (None, None)
    assert run(q, "This Agreement will renew for successive terms of one (1) year each.") == (None, None)


# ---- 2026-10-01: structure fixes from LegalBench dev misses (each with the case it must still refuse)

EXPIRES = "Does the agreement specify when its initial term expires?"
SHARE_EMPLOYEES = "Can the receiving party share confidential information with its employees?"


@pytest.mark.parametrize("document, question, fires", [
    # the agreement itself as the subject of its term's end, past an "unless earlier terminated" clause
    ("This Agreement shall commence on the Effective Date and shall terminate on December 31, 2022.", EXPIRES, True),
    ("The term of this Agreement starts on the Effective Date and, unless this Agreement is earlier terminated in "
     "accordance with its provisions, will expire ten (10) years from the Effective Date.", EXPIRES, True),
    ("This Agreement will take effect on the Effective Date and remain in effect for a period of 1 year.", EXPIRES, True),
    ("This Agreement may be terminated by either party upon thirty (30) days' prior written notice.", EXPIRES, False),
    ("This Agreement shall automatically renew for successive one (1) year periods.", EXPIRES, False),
    # recipients listed as the exception to a ban may receive it; a circumstance as the exception permits nothing
    ("The Receiving Party shall not disclose Confidential Information to any person other than to its directors, "
     "officers and employees who need to know it.", SHARE_EMPLOYEES, True),
    ("The Receiving Party shall not, except with the prior written consent of the Disclosing Party, disclose "
     "Confidential Information to its employees.", SHARE_EMPLOYEES, False),
    ("Each Recipient Party shall not disclose Confidential Information to any person other than to its officers and "
     "employees.", SHARE_EMPLOYEES, True),  # "Recipient Party" is the receiving party
    # a one-way NDA that names its receiving party and never calls it that
    ("The Contractor shall not disclose Confidential Information to any person other than its employees.",
     SHARE_EMPLOYEES, True),
    # a carve-out pointing elsewhere, and scope conditions after the license, don't undo the property
    ("Licensor grants to Licensee a non-exclusive, non-transferable (except in accordance with Section 14) license "
     "to reproduce the Software only for installation on Licensee's servers.", "Is the license non-transferable?", True),
    ("Upon expiration of this Agreement, the license granted under Section 3 will become perpetual.",
     "Is the license perpetual?", False),  # a condition before the property can undo it
    ("Umbrella/Excess Liability with limits of not less than $5,000,000 in excess of the Commercial General "
     "Liability.", "Is a party's liability capped?", False),  # insurance limits, not a liability cap
    # a limit isn't a ban (found in the fresh held-out's half B)
    ("You shall not make more copies of the Evaluation Material than are reasonably necessary.",
     "Can the receiving party make copies of the confidential information?", False),
])
def test_structure_fixes(document, question, fires):
    r = check(question, document)
    assert (r.fired and r.answer == "yes") is fires, (r.reason, r.frames and r.frames.get("reason"))


def test_a_real_ban_on_copies_is_still_a_no():
    r = check("Can the receiving party make copies of the confidential information?",
              "The Receiving Party shall not make any copies of the Confidential Information.")
    assert (r.fired, r.answer) == (True, "no")
