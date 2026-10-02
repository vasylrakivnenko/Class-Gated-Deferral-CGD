"""The contract map (router/contract_map.py): which questions name a high-risk business clause, how a document is cut
into sections, the shipped models' kinds on plain examples, and the harness sending a question to the LLM."""
from __future__ import annotations

import pytest

from router import contract_map as cm
from router.contract_map import ContractMap, Decision, question_types, split_sections
from router.harness import Harness
from tests.test_router import FakeLLM, StubBank


@pytest.mark.parametrize("q, t", [
    ("Is there a cap on liability?", "Cap On Liability"),
    ("Is the vendor's liability limited to the fees paid?", "Cap On Liability"),
    ("Can the supplier compete with us after the agreement ends?", "Non-Compete"),
    ("Is there a most favored nation clause?", "Most Favored Nation"),
    ("Who owns the inventions made under this agreement?", "Ip Ownership Assignment"),
    ("Can we terminate for convenience?", "Termination For Convenience"),
    ("Does the agreement renew automatically?", "Renewal Term"),
    ("Is the distributor's territory exclusive?", "Exclusivity"),
    ("What happens on a change of control?", "Change Of Control"),
    ("Can either party assign this agreement?", "Anti-Assignment"),
    ("Do we have to keep insurance?", "Insurance"),
    ("Is there a minimum purchase commitment?", "Minimum Commitment"),
])
def test_questions_naming_a_high_risk_clause(q, t):
    assert t in question_types(q)


@pytest.mark.parametrize("q", [
    "Is the agreement governed by Delaware law?",
    "Can they share my personal data with advertisers?",
    "How long do you keep my data?",
    "When is rent due?",
    "Must the receiving party return the confidential information?",
    "Can I delete my account?",
    "Do they assign the intellectual property to the company?",  # IP ownership, not anti-assignment
])
def test_questions_naming_none(q):
    assert "Anti-Assignment" not in question_types(q)
    if "intellectual" not in q:
        assert question_types(q) == []


def test_sections_are_paragraphs_and_headings_join_the_next_one():
    text = "1. Term\n\nThis Agreement starts on the Effective Date and lasts two years unless terminated.\n\n" \
           "2. Fees\n\nCustomer shall pay the fees within thirty (30) days of each invoice date."
    secs = split_sections(text)
    assert len(secs) == 2
    assert secs[0][1].startswith("1. Term This Agreement") and secs[1][1].startswith("2. Fees Customer")
    assert text[secs[1][0]:].startswith("2. Fees")


class FakeMap:
    def __init__(self, to_llm: bool):
        self.to_llm = to_llm
        self.seen = []

    def decide(self, question, document, llm=None):
        self.seen.append(question)
        return Decision(self.to_llm, "the question is about Non-Compete, a high-risk business clause" if self.to_llm else "",
                        ["Non-Compete"], {"kind": "commercial", "kind_p": 0.99})


DOC = "The Supplier shall not, during the Term, sell competing products in the Territory. " * 3


def test_a_high_risk_question_skips_the_local_tiers():
    llm = FakeLLM(noul={"Can the supplier": 0.95})
    a = Harness(llm, llm, StubBank(), pretier0=True, classifiers=False, contract_map=FakeMap(True)).answer(
        "Can the supplier sell competing products?", DOC)
    assert a.path == "llm_fallback" and a.answer == "yes"
    assert a.pretier0 is None and a.tier0 is None and a.netreader is None  # none of them ran
    assert a.reason.startswith("the question is about Non-Compete") and a.llm_calls == 1
    assert a.contract_map["to_llm"] is True


def test_other_questions_take_the_usual_path():
    llm = FakeLLM(noul={"Can the supplier": 0.95})
    a = Harness(llm, llm, StubBank(), pretier0=True, classifiers=False, contract_map=FakeMap(False)).answer(
        "Can the supplier sell competing products?", DOC)
    assert a.pretier0 is not None  # Pre-Tier 0 ran
    assert a.contract_map["to_llm"] is False


def test_judgment_and_requests_dont_consult_the_map():
    llm = FakeLLM(noul={"Is the non-compete enforceable": 0.4})
    m = FakeMap(True)
    Harness(llm, llm, StubBank(), pretier0=True, classifiers=False, contract_map=m).answer("Is the non-compete enforceable?", DOC)
    assert m.seen == []


needs_models = pytest.mark.skipif(not (cm.MODEL_DIR / "router.joblib").exists(), reason="no contract map models")

PRIVACY = ("Privacy Policy. This Privacy Policy describes how we collect, use and share your personal information when you "
           "use our website and mobile app. We collect information you provide to us, such as your name, email address "
           "and payment details, and information collected automatically, such as your IP address, device identifiers, "
           "cookies and browsing activity. We use your information to provide and improve our services, to communicate "
           "with you, and for advertising. We may share your information with service providers, advertising partners "
           "and law enforcement when required by law. You can access, correct or delete your personal information and opt "
           "out of marketing emails. We retain your information for as long as your account is active. We use reasonable "
           "security measures to protect your information. Children under 13 may not use our services. We may update this "
           "Privacy Policy from time to time and will notify you of material changes.\n\n") * 4
LEASE = ("LEASE AGREEMENT. This Lease is made between the Landlord and the Tenant. The Landlord lets the Premises at Unit 4, "
         "Riverside Park to the Tenant for a term of five years from the Term Commencement Date. The Tenant shall pay the "
         "Rent of 45,000 per annum by equal quarterly payments in advance on the usual quarter days. The Rent shall be "
         "reviewed on each Review Date to the open market rent. The Tenant shall keep the Premises in good and substantial "
         "repair and shall not use the Premises other than for offices. The Tenant shall not assign, underlet or charge "
         "the whole or any part of the Premises without the consent of the Landlord. The Landlord may re-enter the "
         "Premises if the Rent is unpaid for 21 days. The Tenant may terminate this Lease on the Break Date by giving "
         "not less than six months' notice.\n\n") * 4


@needs_models
def test_the_shipped_models_tell_plain_documents_apart():
    m = ContractMap(lease_encoder=False)
    assert m.kind("Short clause about fees.")[0] is None  # too short for a kind
    assert m.kind(PRIVACY)[0] == "privacy policy"
    assert m.kind(LEASE)[0] == "lease"
    d = m.decide("Can the tenant terminate early?", LEASE)
    assert not d.to_llm and d.document["kind"] == "lease"  # local first (the user's call, 2026-10-02)
    assert not m.decide("Is there a cap on liability?", PRIVACY * 1).to_llm


@needs_models
def test_the_rule_when_turned_on(monkeypatch):
    m = ContractMap(lease_encoder=False)
    monkeypatch.setattr(cm, "ROUTE_HIGH_RISK", True); monkeypatch.setattr(cm, "ROUTE_LEASES", True)
    d = m.decide("Can the tenant terminate early?", LEASE)
    assert d.to_llm and "lease" in d.reason
    assert not m.decide("Can they share my data with advertisers?", PRIVACY).to_llm
    assert not m.decide("Can they terminate my account at any time?", PRIVACY).to_llm  # consumer policy: rule off
    assert m.decide("Is there a cap on liability?", "Fees are due monthly.").to_llm  # short text: no kind, rule applies
