"""Pre-Tier 0 "next level" (2026-10-01, router/STATUS.md): question shapes (router/qshapes.py), the precision
guards for user-style questions, open-vocabulary actions, legal equivalences (router/equivalences.py) and the
document's own names for its confidential information. Each case is a pattern a generated or real question showed."""
import pytest

from router import frames, qshapes
from router.pretier0 import check


@pytest.mark.parametrize("question, shape", [
    ("Does the agreement require the Agent to pay the insurance premiums?", "Must the Agent pay the insurance premiums?"),
    ("Does the lease allow the tenant to sublet the premises?", "Can the tenant sublet the premises?"),
    ("Does the NDA prohibit the recipient from copying the information?",
     "Is the recipient prohibited from copying the information?"),
    ("If the buyer defaults, can the seller terminate the agreement?",
     "Can the seller terminate the agreement if the buyer defaults?"),
    ("Can this Agreement be assigned by the Licensee without consent?",
     "Can the Licensee assigned this Agreement without consent?"),
    ("Is reverse engineering prohibited?", "Is a party prohibited from reverse engineering?"),
    ("Under the lease, if I move out early, do I have to pay rent?", "Must I pay rent if I move out early?"),
    ("Can the Company assign its rights?", "Can the Company assign its rights?"),  # already canonical
])
def test_canonical_shapes(question, shape):
    assert qshapes.canonical(question) == shape


@pytest.mark.parametrize("question, subject", [
    ("Must the Agent pay the premiums?", "Agent"),
    ("Is the Secured Party required to pay expenses?", "Secured Party"),
    ("Is Delaware law the governing law?", None),
    ("Is this a Force Majeure clause?", None),
    ("Can the tenant sublet?", None),
])
def test_subject(question, subject):
    assert qshapes.subject(question) == subject


@pytest.mark.parametrize("question, text", [
    # the named subject must be the one acting
    ("Is the Secured Party required to pay the reasonable expenses of enforcement?",
     "Each Guarantor shall pay on demand all reasonable expenses relating to the enforcement of any Secured Party's rights."),
    ("Does the agreement require the Agent to pay the insurance premiums?",
     "The Agent shall have received evidence of the payment of all Insurance Premiums."),
    # nothing the question asks about may come before its actor
    ("Is there a limit on the number of subsidiaries to which the Company can assign its rights?",
     "The Company may assign its rights under this Agreement to any subsidiary."),
    # condition cues must be the same kind
    ("Is the Consultant required to continue providing services after July 31, 2017?",
     "Consultant shall provide services during the period beginning on May 1, 2017 through July 31, 2017."),
    ("Can a party assign its rights without consent after a change of control?",
     "Except in the case of a change of control, neither Party shall assign any of its rights without the prior "
     "written consent of the other Party."),
    # presence frames keep the question's relations
    ("Is the Covered Person a corporation?", 'Gupta (the "Covered Person") and Progress Software Corporation, a Delaware corporation.'),
    ("Are conflict-of-law provisions applied to this Agreement?",
     "This Agreement will be governed by the laws of the State of Ohio, excluding conflict of law provisions."),
    ("Is the Agreement effective before October 10, 2016?", 'This Agreement is effective as of October 10, 2016.'),
    ("Is there a specified interest rate for the Base Rate Advances?",
     "Interest at the Base Rate shall be calculated on the basis of a 365-day year."),
    ("Do the fees include tax?", "All Fees are exclusive of applicable sales, use and other taxes."),
    ("Is confidential information limited to technical information?",
     '"Confidential Information" includes, but is not limited to, technical and business information.'),
    # a "must" isn't answered by a description
    ("Must confidential information be expressly identified as confidential by the disclosing party?",
     "Confidential Information means any information that (a) is marked as confidential or (b) is identified "
     "orally by the disclosing Party as confidential."),
    # "the shares" is a noun; "entitled to severance" takes a noun
    ("Must the shares be exercised immediately after a Change in Control?",
     "In the event of a Change in Control, all Shares shall become immediately vested and shall remain exercisable."),
    # a payee isn't the one acting
    ("Can the Grantor use its own intellectual property without paying royalties?",
     "Each Grantor grants to Agent a license (exercisable without payment of royalty or other compensation to such "
     "Grantor) to use any of the Intellectual Property owned by such Grantor."),
    # survival: the question's own verb, no durations, same owner
    ("Must the parties renegotiate the surviving covenants after the agreement terminates?",
     "The covenants contained in paragraphs 5 and 8 shall survive the expiration or termination of this Agreement."),
    ("Do the Lenders' obligations under Article III continue after the commitments are terminated?",
     "All of the Borrower's obligations under this Article III shall survive termination of the Commitments of the Lenders."),
    ("Is there a non-compete clause restricting the Executive?",
     "Executive represents that she is not subject to any non-competition or similar restrictions."),
    # from v4's sealed half, after it was run: "its rights" is the object; "discuss" by a person isn't a topic
    # question; boilerplate isn't a limit; a value on one side; permission vs a prohibition; who is responsible
    ("Can a party sell or transfer its rights under this agreement?",
     "Any signature transmitted by facsimile shall be binding upon the party transmitting its signature by facsimile."),
    ("Can the Employee discuss the Agreement with a current employee of the Company?",
     "Employee agrees to keep the terms of this Agreement confidential, except that Employee may tell Employee's family."),
    ("Is there a limit on the amount the Corporation can withhold?",
     "The Corporation shall withhold all taxes required to be withheld, including, but not limited to, garnishments."),
    ("Can the Borrower use more than $175,000,000 of Revolving Loans proceeds for the Acquisition?",
     "The Borrower will use up to $175,000,000 of proceeds of Revolving Loans to finance the Acquisition."),
    ("Is illegal practice permitted?", "you agree to abide by rules prohibiting illegal and other practices."),
    ("Is the company responsible for my messaging fees?",
     "you are responsible for any messaging or data fees you may be charged by your wireless carrier."),
    ("Is there any right for me to own the game code?",
     "all rights, title and interest in and to the service, including games and computer code, are owned by the company."),
])
def test_defers(question, text):
    assert not check(question, text).fired


@pytest.mark.parametrize("question, text, answer", [
    ("Does the lease allow the tenant to sublet the premises?", "Tenant may sublet the Premises with notice to Landlord.", "yes"),
    ("If the agreement terminates, must the receiving party return the confidential information?",
     "Upon termination of this Agreement, the Receiving Party shall return all Confidential Information.", "yes"),
    ("Must the Holder surrender the Note at the closing of any conversion?",
     "The Holder shall surrender the Note at the closing of any conversion.", "yes"),  # a verb outside the lexicon
    ("Is the receiving party prohibited from reverse engineering the confidential information?",
     "Recipient shall not reverse engineer, decompile or disassemble any of the Confidential Information.", "yes"),
    ("Must the receiving party notify the disclosing party if it is required by law to disclose confidential information?",
     "If the Receiving Party is required by law to disclose any Confidential Information, the Receiving Party shall "
     "promptly notify the Disclosing Party.", "yes"),
    ("Is the receiving party prohibited from disclosing the existence of the agreement?",
     "The existence of this Agreement cannot be disclosed to any third party.", "yes"),  # passive, by anyone
    ("Can a party audit the other party's books and records?",
     "Licensee shall grant access to Licensor to audit the books and records of Licensee.", "yes"),
    ("Is a party entitled to severance?",
     "Employee shall receive severance equal to six months' salary if terminated without Cause.", "yes"),
    ("Can either party transfer its rights under this Agreement?",
     "Either party may transfer its rights under this Agreement to an affiliate.", "yes"),
])
def test_answers(question, text, answer):
    r = check(question, text)
    assert (r.fired, r.answer) == (True, answer)


@pytest.mark.parametrize("question, text", [
    ("Can the receiving party independently develop similar information?",
     "Confidential Information does not include information that is independently developed by the Receiving Party."),
    ("Can the receiving party obtain similar information from third parties?",
     "This Agreement shall not apply to information lawfully obtained from a third party who has the right to disclose it."),
    ("Is the receiving party prohibited from disclosing the existence of the agreement?",
     "The existence and terms of this Agreement shall be kept confidential by the parties."),
    ("Do the confidentiality obligations survive termination of the agreement?",
     "The obligations of confidentiality under this Agreement shall survive the termination of this Agreement."),
    ("Does the agreement say that no license to the confidential information is granted to the receiving party?",
     "Nothing in this Agreement shall be construed as granting any rights or license under any patent or copyright."),
])
def test_equivalences(question, text):
    r = check(question, text)
    assert (r.fired, r.answer) == (True, "yes")  # by the frames or by a formula


@pytest.mark.parametrize("question, text", [
    ("Can the receiving party independently develop similar information?",
     "The Recipient shall not independently develop any product similar to the Discloser's products."),
    ("Is the receiving party prohibited from disclosing the existence of the agreement?",
     "Either party may disclose the existence of this Agreement to its investors."),  # (the frames say "no")
    ("Do the confidentiality obligations survive termination of the agreement?",
     "The confidentiality obligations shall not survive the termination of this Agreement."),
])
def test_equivalence_vetoes(question, text):
    assert check(question, text).answer != "yes"


def test_document_names_its_confidential_information():
    doc = ('"Information" means any non-public, confidential or proprietary information disclosed by Company. '
           "Recipient may share the Information with its attorneys and accountants.")
    assert frames.info_aliases(doc) == ["information"]
    r = check("Can the receiving party share confidential information with its consultants or advisors?", doc)
    assert (r.fired, r.answer) == (True, "yes")
