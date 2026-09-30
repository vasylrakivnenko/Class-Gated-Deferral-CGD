"""The 68-question synthetic routing test set and the near-miss set (legal
questions just off the menu, which must route to none_of_these), used by both
the Kev and Jev routing-scale checks. The 38-option menu itself (37 tasks +
none_of_these) lives in router/menu.py, shared with the harness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.menu import CRITERIA, DIVERSITY_FAMILY, INSTRUCTIONS  # noqa: E402,F401


# (expected_task_or_family, paraphrase text)
TEST_CASES = [
    ("cuad_anti-assignment", "Do we need to get the other party's sign-off before we can hand this contract off to someone else?"),
    ("cuad_audit_rights", "Can one company inspect the other's financial records to check they're following the agreement?"),
    ("cuad_cap_on_liability", "Is there a maximum dollar amount either side can be forced to pay if they breach this deal?"),
    ("cuad_change_of_control", "If our company gets acquired or merges with someone else, does the other party get to walk away from this contract, or do they need to be notified first?"),
    ("cuad_covenant_not_to_sue", "Are we barred from ever challenging who legally owns this trademark, or suing them over something unrelated to this deal?"),
    ("cuad_exclusivity", "Does this deal lock us into only buying from this one supplier, or stop us from working with their competitors?"),
    ("cuad_expiration_date", "When does this agreement's initial term actually end?"),
    (DIVERSITY_FAMILY, "We're suing someone in a different state for more than $75,000 and none of us share a home state -- can we bring this in federal court?"),
    ("hearsay", "The witness wants to testify that a bystander told her 'that guy ran the red light' -- can she say that in court to prove he ran the light?"),
    ("learned_hands_consumer", "I bought a used couch online and it showed up broken, seller won't refund me, can I sue in small claims?"),
    ("learned_hands_crime", "My cousin got arrested last night, what happens at his arraignment tomorrow?"),
    ("learned_hands_employment", "My boss fired me the week after I filed a workers comp claim, is that legal?"),
    ("learned_hands_family", "My ex and I are fighting over who gets custody of our kid after the divorce."),
    ("learned_hands_housing", "My landlord is trying to kick me out and won't fix anything, what do I do?"),
    ("learned_hands_torts", "My neighbor's dog bit my kid in our front yard, can I make them pay the medical bills?"),
    ("opp115_data_retention", "How long does this app keep my browsing history before deleting it?"),
    ("opp115_data_security", "How does this app keep my personal info safe from hackers?"),
    ("opp115_first_party_collection_use", "Why does this website need my email address and what do they use it for?"),
    ("opp115_international_and_specific_audiences", "Does this privacy policy have separate rules just for kids or people in Europe?"),
    ("opp115_policy_change", "If they change their privacy policy later, will they actually tell users about it?"),
    ("opp115_third_party_sharing_collection", "Does this company sell or hand my data over to other companies?"),
    ("opp115_user_access,_edit_and_deletion", "Can I ask this company to delete or correct the data they have on me?"),
    ("overruling", "This court opinion says the old ruling from 1997 no longer applies -- is that the court overturning that earlier case?"),
    ("personal_jurisdiction", "Can I be sued in Texas for a car accident that happened in Texas even though I live in Oklahoma?"),
    ("supply_chain_disclosure_best_practice_accountability", "Does this company's statement describe an actual internal compliance mechanism for human trafficking standards, not just a requirement that suppliers comply with the law?"),
    ("supply_chain_disclosure_best_practice_audits", "Does the company say it actually audits its suppliers, or at least reserves the right to?"),
    ("supply_chain_disclosure_best_practice_certification", "Does the company require its suppliers to sign off certifying they follow labor and anti-trafficking laws?"),
    ("supply_chain_disclosure_best_practice_training", "Does the company train its own staff on spotting human trafficking or slavery risks?"),
    ("supply_chain_disclosure_best_practice_verification", "Does the company say it checks the Department of Labor's list or otherwise verifies supplier risk?"),
    ("supply_chain_disclosure_disclosed_accountability", "Does this company's statement say whether, and to what extent, it has internal accountability procedures for staff who violate anti-trafficking standards?"),
    ("unfair_tos", "This terms of service clause says they can change the rules whenever they want without telling you -- what category of unfair clause is that?"),
    ("abercrombie", "Is the brand name 'Apple' for a computer company generic, descriptive, suggestive, arbitrary, or fanciful?"),
    ("none_of_these", "What's the weather going to be like tomorrow in Austin?"),
    ("none_of_these", "Can you recommend a good pizza place near downtown?"),

    # --- round 2: different phrasing/scenarios, same 32 distinct tasks + 2 new controls ---
    ("cuad_anti-assignment", "Am I allowed to sell or transfer my rights under this contract to another company without checking with you first?"),
    ("cuad_audit_rights", "Do we have the ability to send someone to physically inspect their warehouse and verify they're meeting the contract terms?"),
    ("cuad_cap_on_liability", "If we mess up and breach this agreement, is our exposure capped at some specific number, or could we owe unlimited damages?"),
    ("cuad_change_of_control", "We're about to sell 100% of our stock to a private equity firm -- does that trigger any termination rights for our vendor under this supply agreement?"),
    ("cuad_covenant_not_to_sue", "Once we sign this, are we giving up our right to ever dispute their patent ownership down the line?"),
    ("cuad_exclusivity", "Are we contractually forbidden from sourcing this component from any other manufacturer while this agreement is active?"),
    ("cuad_expiration_date", "What's the original end date written into this lease before any renewal options kick in?"),
    (DIVERSITY_FAMILY, "A plaintiff from New York is suing two defendants, one from New Jersey and one from Connecticut, for $200,000 in a state law breach of contract claim -- does a federal court have jurisdiction here?"),
    ("hearsay", "Can a police officer testify at trial about what a 911 caller said on the phone, to prove that the defendant was actually at the scene?"),
    ("learned_hands_consumer", "The gym keeps charging my card every month even after I cancelled my membership in writing, what can I do?"),
    ("learned_hands_crime", "I got pulled over and they found something in my car that wasn't mine, am I going to get charged?"),
    ("learned_hands_employment", "They cut my hours to under 20 a week right after I told them I was pregnant, is that discrimination?"),
    ("learned_hands_family", "How do I change my last name back to my maiden name after the divorce is finalized?"),
    ("learned_hands_housing", "The city is trying to condemn the building I live in and I don't know what my rights are as a tenant."),
    ("learned_hands_torts", "Someone rear-ended me at a stop sign and now my neck hurts, how do I get them to cover my medical bills?"),
    ("opp115_data_retention", "Once I delete my account, how many days do they actually hold onto my old messages before wiping them?"),
    ("opp115_data_security", "What encryption or safeguards does the company use to stop my payment info from being stolen?"),
    ("opp115_first_party_collection_use", "What's the actual reason they're asking for my phone number when I sign up?"),
    ("opp115_international_and_specific_audiences", "Are there extra protections in this policy specifically for users under 13 years old?"),
    ("opp115_policy_change", "Will I get an email notification the next time they update their privacy terms?"),
    ("opp115_third_party_sharing_collection", "Do advertisers get access to my browsing activity from this site?"),
    ("opp115_user_access,_edit_and_deletion", "How do I request a copy of all the data this company has stored about me, or get it removed?"),
    ("overruling", "The appellate court's opinion explicitly states that the precedent set in Smith v. Jones is no longer good law -- does that count as overruling it?"),
    ("personal_jurisdiction", "A company based entirely in California shipped a defective product to me in Ohio and it hurt me here -- can I sue them in an Ohio court?"),
    # Relabeled from ..._best_practice_accountability: "process for employees who violate" is
    # LegalBench's disclosed_accountability wording ("procedures for employees or contractors
    # failing to meet company standards"); the best-practice criteria never mention employees.
    ("supply_chain_disclosure_disclosed_accountability", "Does the company's statement mention any real consequences or internal process for employees who violate their anti-slavery policy?"),
    ("supply_chain_disclosure_best_practice_audits", "Does the disclosure mention conducting site visits or inspections of supplier factories?"),
    ("supply_chain_disclosure_best_practice_certification", "Do their suppliers have to sign paperwork attesting they don't use forced labor?"),
    ("supply_chain_disclosure_best_practice_training", "Does the company teach its purchasing managers how to spot trafficking red flags in the supply chain?"),
    ("supply_chain_disclosure_best_practice_verification", "Does the disclosure say they cross-check suppliers against any government watch lists?"),
    ("supply_chain_disclosure_disclosed_accountability", "Did the company say anything at all about having internal consequences for staff who violate trafficking rules, even vaguely?"),
    ("unfair_tos", "This clause says any dispute has to go through binding arbitration instead of a court -- which unfair-clause bucket does that fall into?"),
    ("abercrombie", "Is 'Igloo' as a brand name for a cooler company suggestive, descriptive, or something else on the trademark spectrum?"),
    ("none_of_these", "Can you help me write a cover letter for a marketing job?"),
    ("none_of_these", "What's the capital of France?"),
]

# Near misses: each question belongs to a real LegalBench task (or a legal question outside
# LegalBench) that has no classifier on the menu, and is worded to sit next to one menu task.
# Routing it to that neighbor would return a confident answer to a different question.
# Comments: true task -> tempting menu neighbor.
NEAR_MISS_CASES = [
    ("none_of_these", "After this agreement ends, are we barred from opening a competing business in the same region for two years?"),  # cuad_non-compete -> cuad_exclusivity
    ("none_of_these", "Can either side walk away from this contract at any time, for no reason, just by giving 30 days' notice?"),  # cuad_termination_for_convenience -> cuad_change_of_control
    ("none_of_these", "Does this agreement renew automatically once the first term is over, and for how long?"),  # cuad_renewal_term -> cuad_expiration_date
    ("none_of_these", "On what date does this agreement actually take effect?"),  # cuad_effective_date -> cuad_expiration_date
    ("none_of_these", "If we cancel early, does the contract make us pay a set termination fee?"),  # cuad_liquidated_damages -> cuad_cap_on_liability
    ("none_of_these", "Does the contract require the vendor to carry liability insurance that covers us?"),  # cuad_insurance -> cuad_cap_on_liability
    ("none_of_these", "Who will own the code the contractor writes for us under this agreement, us or them?"),  # cuad_ip_ownership_assignment -> cuad_covenant_not_to_sue
    ("none_of_these", "Are we allowed to hire any of their engineers after the project ends, or does the contract forbid it?"),  # cuad_no-solicit_of_employees -> cuad_exclusivity
    ("none_of_these", "If they decide to sell this product line, do we get the first chance to buy it before any outsider?"),  # cuad_rofr-rofo-rofn -> cuad_change_of_control
    ("none_of_these", "Do we have to buy a minimum number of units every year under this supply deal?"),  # cuad_minimum_commitment -> cuad_exclusivity
    ("none_of_these", "Which state's law governs this contract if we end up in a dispute?"),  # cuad_governing_law -> personal_jurisdiction / unfair_tos
    ("none_of_these", "Does this website respect my browser's Do Not Track setting?"),  # opp115_do_not_track -> opp115_first_party_collection_use
    ("none_of_these", "Can I opt out of getting their marketing emails?"),  # opp115_user_choice_control -> opp115_user_access,_edit_and_deletion
    ("none_of_these", "Under this NDA, can we share their confidential information with our outside consultants?"),  # contract_nli_sharing_with_third-parties -> opp115_third_party_sharing_collection
    ("none_of_these", "When this NDA ends, do we have to return or destroy the documents they gave us?"),  # contract_nli_return_of_confidential_information -> opp115_data_retention
    ("none_of_these", "Do we still have to keep their secrets after this NDA expires?"),  # contract_nli_survival_of_obligations -> cuad_expiration_date
    ("none_of_these", "I got a notice to appear in immigration court about my green card, what should I do?"),  # learned_hands_immigration -> learned_hands_crime
    ("none_of_these", "I got a speeding ticket last week and want to fight it, is it worth going to court?"),  # learned_hands_traffic -> learned_hands_crime
    ("none_of_these", "I want to write a will leaving my savings to my niece, do I need a lawyer to make it valid?"),  # learned_hands_estates -> learned_hands_family
    ("none_of_these", "My food stamps were cut off after I missed a phone interview, how do I get them back?"),  # learned_hands_benefits -> learned_hands_consumer
    ("none_of_these", "My son's school suspended him for two weeks without any hearing, are they allowed to do that?"),  # learned_hands_education -> learned_hands_family
    ("none_of_these", "What licenses do I need to open a small bakery, and should I register it as an LLC?"),  # learned_hands_business -> learned_hands_consumer
    ("none_of_these", "We're suing under a federal civil-rights statute -- does that alone let us file in federal court?"),  # federal-question jurisdiction -> diversity_*
    # A question about the asker's own criminal trial is also a fair learned_hands_crime match,
    # the way the 68-question set scores similar help requests.
    ({"none_of_these", "learned_hands_crime"}, "Can the prosecutor bring up my 2015 fraud conviction to make the jury doubt my testimony?"),  # impeachment by prior conviction -> hearsay
    ("none_of_these", "Is our brand name 'Zapple' too close to Apple's trademark, could they sue us for infringement?"),  # likelihood of confusion -> abercrombie
    ("none_of_these", "Which earlier cases does this court opinion rely on as precedent?"),  # citation extraction -> overruling
    ("none_of_these", "Translate this privacy policy into Spanish."),  # not a classification question -> opp115_*
    ("none_of_these", "Summarize the main terms of this supply agreement in three bullet points."),  # not a classification question -> cuad_*
]



def is_correct(expected, choice: str) -> bool:
    if isinstance(expected, set):
        return choice in expected
    return choice == expected
