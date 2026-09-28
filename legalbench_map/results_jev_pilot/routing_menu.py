"""Shared 38-option routing menu (37 tasks + none_of_these) and 34-question
synthetic test set, used by both the Kev and Jev routing-scale checks."""

CRITERIA = {
    "cuad_anti-assignment": "Does the clause require consent or notice of a party if the contract is assigned to a third party?",
    "cuad_audit_rights": "Does the clause give a party the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?",
    "cuad_cap_on_liability": "Does the clause specify a cap on liability upon the breach of a party's obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.",
    "cuad_change_of_control": "Does the clause give one party the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?",
    "cuad_covenant_not_to_sue": "Is a party restricted from contesting the validity of the counterparty's ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?",
    "cuad_exclusivity": "Does the clause specify an exclusive dealing commitment with the counterparty, such as requiring all purchases from one party, or prohibiting selling to or working with third parties?",
    "cuad_expiration_date": "Does the clause specify the date upon which the initial term expires?",
    "diversity_1": "Given a lawsuit's fact pattern (which parties are plaintiffs/defendants, their states of citizenship, and the amount(s) in controversy), determine whether the case satisfies federal diversity jurisdiction (complete diversity between all plaintiffs and defendants, and amount in controversy over $75,000).",
    "diversity_2": "Given a lawsuit's fact pattern (which parties are plaintiffs/defendants, their states of citizenship, and the amount(s) in controversy), determine whether the case satisfies federal diversity jurisdiction (complete diversity between all plaintiffs and defendants, and amount in controversy over $75,000).",
    "diversity_3": "Given a lawsuit's fact pattern (which parties are plaintiffs/defendants, their states of citizenship, and the amount(s) in controversy), determine whether the case satisfies federal diversity jurisdiction (complete diversity between all plaintiffs and defendants, and amount in controversy over $75,000).",
    "diversity_4": "Given a lawsuit's fact pattern (which parties are plaintiffs/defendants, their states of citizenship, and the amount(s) in controversy), determine whether the case satisfies federal diversity jurisdiction (complete diversity between all plaintiffs and defendants, and amount in controversy over $75,000).",
    "diversity_5": "Given a lawsuit's fact pattern (which parties are plaintiffs/defendants, their states of citizenship, and the amount(s) in controversy), determine whether the case satisfies federal diversity jurisdiction (complete diversity between all plaintiffs and defendants, and amount in controversy over $75,000).",
    "diversity_6": "Given a lawsuit's fact pattern (which parties are plaintiffs/defendants, their states of citizenship, and the amount(s) in controversy), determine whether the case satisfies federal diversity jurisdiction (complete diversity between all plaintiffs and defendants, and amount in controversy over $75,000).",
    "hearsay": "Given a description of testimony or evidence, determine whether it counts as hearsay (an out-of-court statement offered to prove the truth of what it asserts).",
    "learned_hands_consumer": "Does the post discuss issues people face regarding money, insurance, consumer goods and contracts, taxes, and small claims about quality of service?",
    "learned_hands_crime": "Does the post discuss issues in the criminal system including when people are charged with crimes, go to a criminal trial, go to prison, or are a victim of a crime?",
    "learned_hands_employment": "Does the post discuss issues related to working at a job, including discrimination and harassment, worker's compensation, workers rights, unions, getting paid, pensions, being fired, and more?",
    "learned_hands_family": "Does the post discuss issues that arise within a family, like divorce, adoption, name change, guardianship, domestic violence, child custody, and other issues?",
    "learned_hands_housing": "Does the post discuss issues with paying your rent or mortgage, landlord-tenant issues, housing subsidies and public housing, eviction, and other problems with your apartment, mobile home, or house?",
    "learned_hands_torts": "Does the post discuss problems that one person has with another person (or animal), like a car accident, a dog bite, bullying or possible harassment, or neighbors treating each other badly?",
    "opp115_data_retention": "Does the clause describe how long user information is stored?",
    "opp115_data_security": "Does the clause describe how user information is protected?",
    "opp115_first_party_collection_use": "Does the clause describe how and why a service provider collects user information?",
    "opp115_international_and_specific_audiences": "Does the clause describe practices that pertain only to a specific group of users (e.g., children, Europeans, or California residents)?",
    "opp115_policy_change": "Does the clause describe if and how users will be informed about changes to the privacy policy?",
    "opp115_third_party_sharing_collection": "Does the clause describe how user information may be shared with or collected by third parties?",
    "opp115_user_access,_edit_and_deletion": "Does the clause describe if and how users may access, edit, or delete their information?",
    "overruling": "Does the sentence contain language overruling a previous case?",
    "personal_jurisdiction": "Given a fact pattern about a defendant's contacts with a state and where a lawsuit is filed, determine whether that state's court has personal jurisdiction over the defendant.",
    "supply_chain_disclosure_best_practice_accountability": "Does the statement disclose whether the retailer/manufacturer maintains internal compliance procedures on human trafficking and slavery standards (an actual internal accountability mechanism, not just requiring suppliers to comply with the law)?",
    "supply_chain_disclosure_best_practice_audits": "Does the statement disclose whether the retailer/manufacturer performs any type of audit, or reserves the right to audit, its suppliers?",
    "supply_chain_disclosure_best_practice_certification": "Does the statement disclose whether the retailer/manufacturer requires direct suppliers to certify compliance with labor and anti-trafficking laws?",
    "supply_chain_disclosure_best_practice_training": "Does the statement disclose whether the retailer/manufacturer provides training to employees on human trafficking and slavery risks?",
    "supply_chain_disclosure_best_practice_verification": "Does the statement disclose whether the retailer/manufacturer engages in supplier verification/auditing, or assesses supplier risk via the US Dept. of Labor's list?",
    "supply_chain_disclosure_disclosed_accountability": "Does the statement disclose to what extent, if any, the retailer/manufacturer maintains internal accountability standards for employees/contractors who fail to meet anti-trafficking standards?",
    "unfair_tos": "Given a clause from a Terms of Service agreement, classify it into one of: Arbitration, Unilateral change, Content removal, Jurisdiction, Choice of law, Limitation of liability, Unilateral termination, Contract by using, or Other.",
    "abercrombie": "Given a trademark/brand name and the product or service it names, classify its trademark distinctiveness category: generic, descriptive, suggestive, arbitrary, or fanciful.",
}
CRITERIA["none_of_these"] = "The request does not match any of the other listed questions."

DIVERSITY_FAMILY = {"diversity_1", "diversity_2", "diversity_3", "diversity_4", "diversity_5", "diversity_6"}

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
    ("supply_chain_disclosure_best_practice_accountability", "Does the company's statement mention any real consequences or internal process for employees who violate their anti-slavery policy?"),
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

INSTRUCTIONS = (
    "A user submitted a free-text request. Which ONE of the following predefined "
    "legal classification questions does the user's request match? If none of them "
    "match, choose none_of_these."
)


def is_correct(expected, choice: str) -> bool:
    if isinstance(expected, set):
        return choice in expected
    return choice == expected
