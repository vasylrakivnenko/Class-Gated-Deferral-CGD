"""Tier 0's reader network (router/netreader.py): units, the decision, and the rule checks on the deciding sentence.
The network is a stand-in here (`FakeNet`); the trained one is scored by /root/zadumai_nli_proto/reader_net."""
from __future__ import annotations

import pytest

from router.netreader import NetReader, check, units

LEASE = ("1. TERM. The term of this Lease is twelve (12) months. 4. PETS. Tenant may keep one cat on the premises. "
         "No dogs are permitted. 5. SUBLETTING. Tenant shall not sublet the premises.")


class FakeNet:
    """p(no), p(yes), p(doesn't settle it) by the first rule whose words are all in the text."""

    def __init__(self, rules):
        self.rules = rules  # [(words in the text, words in the question, (p_no, p_yes, p_none))]

    def __call__(self, pairs):
        out = []
        for text, question in pairs:
            t, q = text.lower(), question.lower()
            out.append(next((list(p) for tw, qw, p in self.rules if all(w in t for w in tw) and all(w in q for w in qw)),
                            [0.02, 0.02, 0.96]))
        return out


def reader(rules, **kw):
    kw = {"t_yes": 0.9, "t_no": 0.9, "t_yes_doc": None, "t_yes_long": None, **kw}  # the tests' own thresholds
    return NetReader(net=FakeNet(rules), k_emb=0, **kw)


def test_short_documents_are_one_unit_and_long_ones_windows_of_sentences():
    assert units(LEASE) == [LEASE]
    long = " ".join(f"Clause {i}. The Supplier shall deliver the goods number {i} on time." for i in range(40))
    us = units(long)
    assert len(us) > 1 and all(len(u) <= 600 for u in us) and " ".join(us) == " ".join(long.split())


def test_a_yes_the_checks_pass_answers_with_the_deciding_sentence():
    r = reader([(["cat"], ["cat"], (0.01, 0.97, 0.02))]).answer("Can the tenant keep a cat?", LEASE)
    assert (r.fired, r.answer) == (True, "yes") and "cat" in r.evidence[0]["text"] and r.qualifier is None


def test_below_the_threshold_defers():
    r = reader([(["cat"], ["cat"], (0.01, 0.80, 0.19))]).answer("Can the tenant keep a cat?", LEASE)
    assert not r.fired and "closest: yes at 0.80" in r.reason


def test_documents_of_several_units_have_their_own_yes_threshold():
    doc = "Tenant may keep one cat on the premises. " + "Lorem ipsum dolor sit amet clause. " * 30
    rules = [(["may keep one cat"], ["cat"], (0.01, 0.97, 0.02))]
    assert reader(rules).answer("Can the tenant keep a cat?", doc).fired
    assert not reader(rules, t_yes_doc=0.98).answer("Can the tenant keep a cat?", doc).fired
    # a text of one unit keeps t_yes
    assert reader(rules, t_yes_doc=0.98).answer("Can the tenant keep a cat?", "Tenant may keep one cat on the premises.").fired


def test_long_documents_read_through_retrieval_have_their_own_threshold():
    long = "Tenant may keep one cat on the premises. " + "Lorem ipsum dolor sit amet clause number. " * 200
    rules = [(["may keep one cat"], ["cat"], (0.01, 0.97, 0.02))]
    assert len(units(long)) > 6 and reader(rules).answer("Can the tenant keep a cat?", long).fired
    r = reader(rules, t_yes_long=1.01).answer("Can the tenant keep a cat?", long)
    assert not r.fired and r.reason.startswith("a long document")
    short = "Tenant may keep one cat on the premises. " + "Lorem ipsum dolor sit amet clause. " * 30  # read whole
    assert reader(rules, t_yes_long=1.01).answer("Can the tenant keep a cat?", short).fired


def test_units_that_disagree_defer():
    doc = "Tenant may keep one cat on the premises. " + "Lorem ipsum dolor sit amet clause. " * 30 + \
          "Tenant shall not keep any cat on the premises."
    rules = [(["may keep one cat"], ["cat"], (0.01, 0.97, 0.02)), (["not keep any cat"], ["cat"], (0.95, 0.01, 0.04))]
    r = reader(rules).answer("Can the tenant keep a cat?", doc)
    assert not r.fired and r.reason.startswith("passages disagree")


@pytest.mark.parametrize("question, sentence, answer, fails", [
    ("Can the tenant keep a cat?", "Tenant may keep one cat on the premises.", "yes", []),
    # the sentence says that party shall not
    ("Can the tenant sublet?", "Tenant shall not sublet the premises.", "yes", ["yes_neg"]),
    # another party: the landlord is a role word even where the text never names one
    ("Can the landlord sublet?", "Tenant shall not sublet the premises.", "no", ["party"]),
    ("Can the receiving party share confidential information with its employees?",
     "The Disclosing Party may share Confidential Information with its employees.", "yes", ["party"]),
    # silence is not "no"
    ("Can a party assign the agreement?", "This Agreement shall be governed by the laws of Delaware.", "no", ["no_neg"]),
    ("Can the tenant keep a dog?", "No dogs are permitted.", "no", []),
    # "must" isn't answered by "may"
    ("Must the licensee maintain insurance?", "The Licensee may maintain insurance at its own cost.", "yes", ["must_may"]),
    # prohibited: "yes" needs a ban, "no" needs a "may"
    ("Is a party prohibited from soliciting employees?", "Neither party shall solicit the other party's employees.", "yes", []),
    ("Is a party prohibited from soliciting employees?", "Each party shall use reasonable efforts to retain employees.", "yes", ["yes_ban"]),
    ("Is the tenant prohibited from keeping a cat?", "Tenant may keep one cat on the premises.", "no", []),
    # the question's numbers must be in the sentence for a "yes"
    ("Is the rent $1,500 per month?", "Tenant shall pay rent of $2,000 per month.", "yes", ["values"]),
    ("Is the rent $2,000 per month?", "Tenant shall pay rent of $2,000 per month.", "yes", []),
    # a "no" with a condition defers
    ("Can a party assign the agreement?", "Neither party may assign this Agreement without the prior written consent "
                                          "of the other party.", "no", ["no_cond"]),
    # someone's discretion: a yes for that party only (adv-38)
    ("Is the employee entitled to an annual bonus?", "The Company may, in its sole discretion, pay an annual bonus to "
                                                     "the Employee.", "yes", ["discretion"]),
    ("Can the company pay the employee an annual bonus?", "The Company may, in its sole discretion, pay an annual "
                                                          "bonus to the Employee.", "yes", []),
])
def test_checks(question, sentence, answer, fails):
    assert check(question, sentence, answer, sentence)[0] == fails


def test_a_yes_with_a_condition_quotes_it():
    s = "Tenant may sublet the premises with Landlord's consent, unless Tenant is in default."
    assert check("Can the tenant sublet?", s, "yes", s) == ([], "unless Tenant is in default")
    r = reader([(["sublet"], ["sublet"], (0.01, 0.97, 0.02))]).answer("Can the tenant sublet?", s)
    assert (r.fired, r.answer, r.qualifier, r.condition) == (True, "yes", "with a condition", "unless Tenant is in default")


def test_a_failed_check_defers_and_says_which():
    r = reader([(["sublet"], ["sublet"], (0.01, 0.97, 0.02))]).answer("Can the tenant sublet?", LEASE)
    assert not r.fired and r.checks == ["yes_neg"] and "sublet" in r.evidence[0]["text"]


def test_topic_questions_are_left_to_pre_tier_0():
    assert not reader([]).answer("Is subletting discussed here?", LEASE).fired
