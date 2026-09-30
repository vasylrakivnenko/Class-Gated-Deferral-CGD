"""
The routing menu Jev picks from, and what kind of text each task expects.

CRITERIA and INSTRUCTIONS are the exact texts the routing gate in
router/harness.py was calibrated on (results_jev_pilot/, 68 on-menu and 28
near-miss questions); results_jev_pilot/routing_menu.py re-exports them for
those tests. Editing either one means re-running those tests and re-checking
the gate thresholds.
"""

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

INSTRUCTIONS = (
    "A user submitted a free-text request. Which ONE of the following predefined "
    "legal classification questions does the user's request match? If none of them "
    "match, choose none_of_these."
)

# The kind of text each family's classifier was trained on, completing the
# yes/no check "Is this text <type>?" that runs before a classifier is trusted
# with the user's document.
TEXT_TYPES = {
    "cuad": "a commercial contract, or a clause or section from one",
    "opp115": "a website or app privacy policy, or a passage from one",
    "learned_hands": "a person describing their own legal problem or asking for legal help",
    "supply_chain_disclosure": "a company's statement about slavery or human trafficking risks in its supply chain",
    "overruling": "a court opinion, or a sentence or passage from one",
    "unfair_tos": "a website or app terms of service, or a clause from one",
    "hearsay": "a description of evidence or testimony offered in a legal case",
    "diversity": "a fact pattern about a lawsuit: who sues whom, where each party is a citizen, and how much is claimed",
    "personal_jurisdiction": "a fact pattern about a defendant's contacts with a state where a lawsuit is filed",
    "abercrombie": "a brand name or trademark together with the product or service it names",
}

# Families trained on single clauses or sentences (median 150-450 characters).
# A longer document is split into units of that size and each unit is
# classified; every other family was trained on whole documents.
SPLIT_UNIT = {"cuad": "paragraph", "opp115": "paragraph", "unfair_tos": "sentence", "overruling": "sentence"}

_FAMILY_PREFIXES = ("cuad", "opp115", "learned_hands", "supply_chain_disclosure", "diversity")


def family(task: str) -> str:
    for prefix in _FAMILY_PREFIXES:
        if task.startswith(prefix + "_"):
            return prefix
    return task
