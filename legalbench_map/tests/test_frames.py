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
