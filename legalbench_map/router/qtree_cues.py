"""Every cue word the question tree (router/qtree.py) decides by. Kept out of
the code so the lists can be read, reviewed and extended in one place.

Judgmental questions ask for advice or a legal conclusion ("Should I sign?",
"Is the non-compete enforceable?"). The text can't settle those: a contract
that calls itself enforceable isn't thereby enforceable. Questions about what
the text allows or requires stay lookups, first person included ("Do we have
to give notice?", "Must the fee be reasonable?").
"""
from __future__ import annotations

import re

# Advice and prediction, matched on the lowercased question (after unwrapping).
ADVICE = [(name, re.compile(rx)) for name, rx in [
    ("should", r"\bshould(?:n't| not)?\b"),
    ("ought to", r"\bought(?:n't| not)? to\b"),
    ("had better", r"\bhad better\b"),
    ("is it wise", r"\b(?:is|would|will) it (?:be )?(?:better|best|wiser|wise|advisable|smart|a good idea|a bad idea|"
                   r"worth it|worth|risky|safe|okay|ok|fine|a problem|a mistake)\b"),
    ("recommend", r"\b(?:do|would|can|could) you (?:recommend|advise|suggest|think)\b"),
    ("recommend", r"\bwhat (?:do|would) you (?:recommend|advise|suggest|think)\b"),
    ("in your opinion", r"\bin your (?:opinion|view)\b"),
    ("worth", r"\bworth (?:it|signing|suing|fighting|challenging|negotiating)\b"),
    ("my options", r"\bwhat are (?:my|our|his|her|their) (?:options|chances|rights)\b"),
    ("what should I do", r"\bwhat (?:should|can|could|do) (?:i|we) do\b"),
    ("have a case", r"\b(?:do|does|would) (?:i|we|my client|they|he|she) have a (?:good |strong )?(?:case|claim)\b"),
    ("win", r"\b(?:will|would|could|can) (?:i|we|they|he|she|my client) win\b"),
    ("chances", r"\b(?:chances|odds|likelihood) (?:of|that)\b"),
    ("sue", r"\b(?:can|could|should|may) (?:i|we|my client) (?:sue|take (?:them|him|her|it) to court)\b"),
    ("in court", r"\b(?:hold|stand) up in court\b|\bwould a (?:court|judge|jury)\b|\bwill a (?:court|judge|jury)\b"),
    # Risk and trust: predictions the text can't settle ("Could this app be hacked?").
    # ("risk of loss" is a contract term, so a bare "risk of" doesn't count.)
    ("risk", r"\b(?:at risk|is there (?:a|any) risk|any risks?\b|red flags?|worst[- ]case|repercussions?|"
             r"dangers? of|dangerous)\b"),
    # A prediction or fear ("Could this app be hacked?", "fear of being hacked"), not
    # history ("Has the app ever been hacked?" asks for a fact).
    ("hacked", r"\b(?:will|would|could|can|might|may)\b[^?.]{0,40}\b(?:be|get) (?:\w+ly )?(?:hacked|stolen|leaked|breached|"
               r"compromised|misused|exposed)\b|\bbeing (?:hacked|stolen|leaked|breached|compromised|misused)\b|"
               r"\bhackable\b"),
    ("afraid", r"\b(?:fear|afraid|scared|worried) (?:of|about|that)\b"),
    ("how well", r"\bhow (?:well|securely|safely) (?:secured|protected|guarded|kept|handled|encrypted|stored|is|are|"
                 r"does|do|will)\b"),
    ("trust", r"\b(?:un)?trustworthy\b|\b(?:can|should|could) (?:i|we) trust\b"),
    ("how secure", r"\bhow (?:secure|safe|private|protected|reliable|trustworthy|vulnerable|risky|fair|reasonable|"
                   r"enforceable|strict|harsh|likely|legal|standard) (?:is|are|was|were|would|will|do|does|can|my|the|"
                   r"this|your|our)\b"),
    # Degree and comparison with the world outside the text.
    ("too", r"\b(?:too|overly|unusually|excessively|unfairly|unreasonably) (?:broad|long|short|high|low|strict|harsh|"
            r"restrictive|vague|one-sided|expensive|much|little|many|few|aggressive|onerous|generous|lenient)\b"),
    ("compared to", r"\bcompared (?:to|with)\b|\brelative to (?:other|typical|most)\b|\bthan (?:typical|usual|"
                    r"normal|average|most|other|standard|market)\b|\b(?:industry|market)[- ](?:standard|practice|rate)s?\b"),
]]
# The law as the question's topic: "Is the landlord required by law to...?",
# "Is this allowed under California law?". Not inside a condition, where it
# describes the text: "...notify it if it is required by law to disclose?"
LAW_CUES = [(name, re.compile(rx)) for name, rx in [
    ("under the law", r"\b(?:under|by|against) (?:the )?(?:law|laws)\b|\bunder (?:[a-z]+ )?(?:state|federal|local|us|"
                      r"u\.s\.|eu|english|california|new york) law\b"),
    ("under a statute", r"\bunder (?:the )?(?:gdpr|ccpa|cpra|hipaa|coppa|ferpa|flsa|ada|fcra|tcpa|can-spam)\b"),
]]
# Cues that describe the text, not the question, inside a condition:
# "...if it is required by law...", "Will you notify me if my data gets hacked?"
CONDITION_EXEMPT = frozenset(["under the law", "under a statute", "hacked", "risk"])
CONDITION_BEFORE = re.compile(r"\b(?:if|unless|when|whenever|where|once|as|except|to the extent|in case)\b(?:\W+\w+){0,6}\W*$")
# "should" in a question about a fact of the text is a lookup: "Where should
# payments be sent?", "When should the tenant give notice?". Advice has a
# person as its subject ("What should I do?", "Should we sign?") or is a
# yes/no question ("Should the tenant pay?").
ADVICE_SUBJECTS = frozenset("i we you my our me us".split())

# Adjectives that appraise or conclude: a question predicating one of them of
# the text or a term ("Is the fee reasonable?", "Is it legal for X to Y?") is
# judgmental. As a plain modifier ("normal wear and tear") or inside a
# deontic question ("Must the fee be reasonable?") they aren't.
EVALUATIVE = frozenset("""
enforceable unenforceable valid invalid void voidable legal illegal lawful unlawful legit legitimate
fair unfair reasonable unreasonable unconscionable excessive predatory abusive exploitative one-sided
standard normal typical customary usual common uncommon unusual market favorable favourable unfavorable
unfavourable advantageous disadvantageous good bad better worse risky dangerous safe unsafe harmful acceptable
unacceptable constitutional unconstitutional compliant noncompliant non-compliant likely unlikely
harsh onerous burdensome overbroad draconian lenient secure insecure trustworthy untrustworthy vulnerable
reliable unreliable
""".split())
# A backstop for parses that miss the predicate ("Is this arbitration clause
# standard for consumer loans?" parses "standard" as a noun): a yes/no "is/are"
# question whose predicate ends in an evaluative word, i.e. one followed by the
# end or a preposition, not by a noun ("Is fair market value used?" stays a lookup).
COPULA_APPRAISAL = re.compile(
    r"^(?:is|are|was|were|isn't|aren't)\b[^?]*?\b(" + "|".join(sorted(EVALUATIVE - {"market", "common", "good", "bad",
                                                                                  "better", "worse", "likely"}))
    + r")\b\s*(?=[?.!]|$|(?:for|given|in|under|compared|to|by|at|here|there|enough|or not)\b)")
# ...unless it's what the text should make so: "how is my data kept safe?",
# "what does the app do to make sure my data is secure?"
MAKE_SO = frozenset("keep kept keeps keeping make makes made ensure ensures ensured remain remains stay stays".split())
# "valid for 30 days", "valid until June": a stated term, not a legal conclusion.
VALID_IF = re.compile(r"\bvalid (?:if|when|unless|only|once|as long as)\b")  # a stated condition, unless of the contract itself
LEGAL_INSTRUMENTS = frozenset("non-compete noncompete compete covenant waiver release restriction limitation indemnity "
                              "arbitration guarantee warranty lien".split())
VALID_FOR = re.compile(r"\bvalid (?:for|until|till|through|thru|from|after|before) (?:\d+|a|an|one|two|three|four|"
                       r"five|six|seven|eight|nine|ten|twelve|fifteen|thirty|sixty|ninety|the (?:first|initial|term|"
                       r"duration|period|life)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\b")
# Adverbs that ask about the law rather than the text: "Can the landlord legally enter?"
LEGAL_ADVERBS = frozenset("legally lawfully validly illegally unlawfully".split())
# Verbs whose question is a legal conclusion or a finding of fact outside the text.
CONCLUSION_VERBS = frozenset("enforce uphold breach violate infringe contravene invalidate void".split())
# "comply" is a conclusion only about the law ("Does the policy comply with GDPR?"),
# not an obligation in the text ("Does the tenant have to comply with the rules?").
LAW_WORDS = frozenset("""
law laws statute statutes regulation regulations legislation constitution gdpr ccpa cpra hipaa coppa ferpa
flsa ada fcra tcpa code
""".split())

# An appraisal inside an obligation asks what the text requires: "Must the fee be
# reasonable?", "Is the landlord required to be reasonable?".
OBLIGATION = re.compile(r"\b(?:must|shall|supposed to|require|requires|requiring|required|obligated|obliged|bound|(?:have|has|had|need|needs) to)\b")
# Inside a permission it does too when a party is the subject ("Can the landlord be
# unreasonable about consent?"), not when a term is ("Could the non-compete be unenforceable?").
PERMISSION = re.compile(r"\b(?:can|could|may|might|allowed|permitted)\b")
PARTY_NOUNS = frozenset("""
i we you they he she landlord tenant lessor lessee licensor licensee party parties buyer seller purchaser vendor
supplier customer client contractor subcontractor employer employee company provider user member owner
distributor manufacturer franchisor franchisee borrower lender guarantor recipient discloser consultant
""".split())
# Nouns for the text itself: "Is this a fair contract?" appraises the document.
DOCUMENT_NOUNS = frozenset("""
this that it contract agreement clause lease policy term terms provision provisions deal nda document section
arrangement offer proposal
""".split())

# Requests: the asker wants a task done, not a question answered. After "can/
# will you" only unambiguous assistant tasks count: in a privacy policy "you"
# is often the company ("Will you send me spam?", "Do you share my data?").
TASK_VERBS = frozenset("""
list give draft summarize summarise explain show find write tell check highlight extract translate rewrite
simplify compare review analyze analyse identify outline describe paraphrase quote cite redraft calculate compute
""".split())
# Imperatives may also start with these: "Create a checklist", "Make a summary".
IMPERATIVE_ONLY = frozenset("create generate make prepare edit fix count point mark flag break walk".split())
# Polite wrappers around a real question: "Can you tell me whether X?", "I want to know if X".
EMBEDDED = re.compile(
    r"^(?:please[,\s]+)?(?:(?:can|could|would|will) you(?: please)? )?"
    r"(?:tell me|let me know|check|find out|confirm|say)(?: for me)?[,:]?\s+"
    r"(?P<q>(?:whether|if|who|whom|whose|what|when|where|which|why|how)\b.*)$"
    r"|^(?:i(?:'d| would) like to know|i want to know|i need to know|i wonder|i am wondering|i'm wondering|"
    r"do you know|does anyone know)[,:]?\s+(?P<q2>(?:whether|if|who|whom|whose|what|when|where|which|why|how)\b.*)$",
    re.I,
)


# ---- M1: form, branch and leaf -------------------------------------------------

# "What's", "whats", "who's": the wh-word with "is" glued on.
WH_CONTRACTIONS = re.compile(r"^(what|who|where|when|how|why|which)(?:'s|s|’s)\b", re.I)
# Openers that don't change the question: "So, can the tenant sublet?"
OPENERS = re.compile(r"^(?:(?:so|ok|okay|and|but|also|then|hi|hey|hello|quick question|question|one more thing)"
                     r"[,:!.]?\s+)+", re.I)
# A wh-word, maybe after a preposition: "By when must notice be given?", "To whom is it paid?"
WH_LEAD = re.compile(r"^(?:(?:by|until|till|to|from|for|in|on|at|with|under|within|after|before|since|of|about)\s+)?"
                     r"(who|whom|whose|what|when|where|which|why|how)\b", re.I)
POLAR_LEAD = re.compile(r"^(?:is|are|was|were|am|do|does|did|can|could|will|would|shall|should|may|might|must|"
                        r"has|have|had|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|cannot|couldn't|"
                        r"won't|wouldn't|shouldn't|mustn't|hasn't|haven't|hadn't|is there|are there)\b", re.I)

# "how" + one of these asks for a quantity (a fact); other "how"s ask for a way.
HOW_QUANT = {"much": "MONEY", "many": "CARDINAL", "long": "DURATION", "often": "FREQUENCY", "frequently": "FREQUENCY",
             "soon": "DATE", "early": "DATE", "late": "DATE", "far": "QUANTITY", "old": "AGE", "big": "QUANTITY",
             "large": "QUANTITY", "high": "QUANTITY", "low": "QUANTITY"}
# Units that make "how many" a duration: "How many days' notice?"
TIME_UNITS = frozenset("second seconds minute minutes hour hours day days business week weeks month months year years".split())

# Reasoning about consequences and procedures (the spec's Forward and Process leaves).
FORWARD = re.compile(r"\bwhat (?:happens|would happen|will happen|could happen|happened|if)\b|\bwhat are the "
                     r"(?:consequences|implications|results|effects|penalties) (?:of|if|for)\b|\bwhat is the "
                     r"(?:consequence|implication|result|effect) (?:of|if)\b|\bwhat (?:would|will) (?:be|happen)\b"
                     r"|\bwhat do(?:es)? (?:i|we|you|they|the \w+) (?:get|lose|owe) if\b")
PROCESS = re.compile(r"\bwhat (?:do|does|would|should|must|will) (?:i|we|you|one|they|the \w+|he|she|a \w+) "
                     r"(?:need|have|has) to do\b|\bwhat (?:steps|actions)\b|\bwhat (?:is|are) the (?:process|procedure|"
                     r"steps|requirements)\b|\bwhat (?:is|are) (?:required|needed|necessary) (?:to|for|in order)\b"
                     r"|\bwhat do(?:es)? (?:i|we|you|one) need to\b|\bwhat (?:do|does|must) it take\b")

# Answer types for "what/which + noun" and "what is the <noun>" spans, by the noun.
NOUN_TYPES = {
    "DATE": "date day deadline time year month start end beginning expiration effective commencement".split(),
    "DURATION": "period term duration length notice timeframe window term".split(),
    "MONEY": "fee fees price cost costs rent amount deposit penalty salary payment payments charge charges fine cap ceiling "
             "limit coverage "
             "compensation bonus rate wage wages budget premium deductible royalty royalties".split(),
    "PERCENT": "percentage percent interest apr".split(),
    "JURISDICTION": "law laws state country jurisdiction court courts venue forum county city".split(),
    "PARTY": "party parties company person who entity business employer landlord tenant owner provider "
             "licensor licensee buyer seller vendor".split(),
    "FREQUENCY": "frequency often".split(),
    "LOCATION": "address place location where".split(),
    "PROPERTY": "kind type sort types kinds sorts form forms".split(),
}
NOUN_TYPE = {w: t for t, ws in NOUN_TYPES.items() for w in ws}
WH_TYPES = {"who": "PARTY", "whom": "PARTY", "whose": "PARTY", "when": "DATE", "where": "LOCATION"}

# Alternatives vs "either will do": "Is the bonus guaranteed or discretionary?" asks
# which; "Is the license irrevocable or perpetual?" asks whether either holds.
CLAUSE_ALTERNATIVE = re.compile(r"\bor (?:only|just|merely|instead|rather|is it|are they|is that|is this|does|do|did|"
                                r"can|could|will|would|must|should|are|is|has|have|was|were|am i|do i|does it|do they|"
                                r"do we|do you)\b")
OPPOSITES = [frozenset(p.split("/")) for p in """
guaranteed/discretionary fixed/variable fixed/floating flat/percentage flat/percent monthly/weekly monthly/annually
monthly/yearly monthly/quarterly weekly/daily annually/quarterly full/partial full/part whole/part exclusive/nonexclusive
mandatory/optional required/optional paid/unpaid refundable/nonrefundable written/oral written/verbal automatic/manual
temporary/permanent public/private joint/several before/after gross/net cash/credit hourly/salaried
employee/contractor tenant/landlord landlord/tenant buyer/seller licensor/licensee employer/employee company/user
lessor/lessee customer/provider vendor/customer arbitration/court arbitrate/litigate state/federal renew/expire
increase/decrease upfront/installments online/offline opt-in/opt-out""".split()]
FREQUENCY_WORDS = frozenset("daily weekly biweekly monthly quarterly annually yearly semiannually hourly".split())
# More than one question: "What is the rent and when is it due?"
SECOND_QUESTION = re.compile(r"\?\s*\S|\b(?:and|also|plus|but)\s+(?:(?:who|what|when|where|why|how)\s+(?:is|are|was|"
                             r"were|do|does|did|can|could|will|would|must|should|may|much|many|long|often)\b|(?:is|are|"
                             r"can|could|does|do|did|must|will|would|should|may)\s+(?:the|a|an|i|we|you|they|it|this|that|"
                             r"he|she|my|our|your|their|there)\b)")

# Misspelled wh-words seen in real questions: "hwere is it saved".
WH_TYPOS = {"hwere": "where", "wher": "where", "whre": "where", "waht": "what", "wat": "what", "wht": "what",
            "hwo": "who", "whn": "when", "wen": "when", "hwy": "why", "whay": "why", "hw": "how", "hwat": "what"}
# A condition or time clause before the question: "If I breach it, what are the consequences?"
SUBORDINATE_LEAD = re.compile(r"^(?:if|when|whenever|once|after|before|in case|as long as|as|since|while|unless|"
                              r"assuming|suppose|supposing)\b", re.I)
# "when"/"where" + auxiliary is a question word ("When is rent due?"), otherwise a subordinator.
WH_AUX_NEXT = re.compile(r"^(?:when|where)\s+(?:is|are|was|were|do|does|did|can|could|will|would|shall|should|may|"
                         r"might|must|has|have|had)\b", re.I)
# "how" asks for a procedure when the asker (or anyone) is the one doing it; "How is
# my data stored?" asks how the other side does something: a fact the text states.
HOW_PROCESS = re.compile(r"^how (?:to|do|can|could|should|would|must|may|might) (?:i|we|one|a user|users|someone|"
                         r"customers?|members?|employees?|tenants?|the tenant|the user|the customer|the employee|"
                         r"the member|the buyer|the licensee|me)\b|^how to\b")
HOW_YOU_PROCESS = re.compile(r"^how (?:do|can|could|should|would|must) you (?:cancel|terminate|opt|unsubscribe|delete|"
                             r"request|file|apply|renew|dispute|appeal|withdraw|claim|get|contact|report|return|"
                             r"exercise|close|access|end|stop|change|update|correct|transfer|assign|sublet|"
                             r"submit|obtain|recover)\b")
# "What steps does the developer take?" describes their practice (a span); "What
# steps do I take?" is a procedure.
STEPS_SPAN = re.compile(r"\bwhat (?:steps|measures|actions|precautions) (?:does|do|did|will|has|have) "
                        r"(?!i\b|we\b|one\b|me\b)")
# Alternatives spelled with a repeated preposition: "at shipment or at delivery".
REPEATED_PREP = re.compile(r"\b(at|to|in|by|on|upon|from|before|after|within|under|via|through|with|for) [\w'’ -]{1,40}? or "
                           r"\1 ")
OPPOSITES = OPPOSITES + [frozenset(p.split("/")) for p in """
arbitration/litigation shipment/delivery immediately/after immediately/later now/later one-time/subscription
opt-in/opt-out in/out personalized/global personalized/globally domestic/international locally/remotely local/remote
device/server inside/outside online/in-person email/mail electronic/paper renewal/termination buy/lease buy/rent
rent/own purchase/lease sell/license license/assignment individual/joint individually/jointly""".split()]
HOW_QUANT.update({"quickly": "DURATION", "fast": "DURATION"})
FORWARD_MORE = re.compile(r"\bwhat\b[^?]{0,60}\b(?:happens?|happened)\b|\b(?:consequences?|penalt(?:y|ies)|"
                          r"implications?)\s*[?.!]*$")
# Parties by role, for choice options ("the tenant or the landlord"); pronouns aren't options.
PARTY_ROLE_NOUNS = PARTY_NOUNS - {"i", "we", "you", "they", "he", "she"}
WH_TYPOS.update({"wha": "what", "whta": "what", "whats": "what is"})
FREQUENCY_WORDS = FREQUENCY_WORDS | frozenset("annual semiannual biannual weekly daily hourly yearly quarterly".split())
OPPOSITES = OPPOSITES + [frozenset(p.split("/")) for p in """
mediation/arbitration mediation/litigation internally/externally internal/external paid/free free/paid""".split()]
# "..., me or the insurer?", "..., the tenant or the landlord?" after a wh-question;
# "X, Y, or both/neither/either?".
TAIL_ALTERNATIVES = re.compile(r",\s*[\w'’ -]{1,40}\s+or\s+[\w'’ -]{1,40}[?.!]*$|\bor (?:both|neither|either)\b")
# A leading condition with "what ... do/happen" in the main clause asks what follows.
IF_WHAT_DO = re.compile(r"^(?:if|in case|in the event|should)\b.*\bwhat\b[^?]*\b(?:do|does|did|happen|happens)\b")
# "strong enough to protect us", "worth anything in practice"
ENOUGH = re.compile(r"^(?:is|are|was|were|would|will)\b[^?]*\b(?!(?:long|much|many|often|early|soon)\b)[a-z]+ enough\b"
                    r"|\bworth (?:anything|much|the|a lot|paying|keeping)\b")
# Words that open a condition or time clause (spaCy's "mark" of an advcl).
CONDITION_MARKS = frozenset("if unless when whenever once where while because since although though after before until "
                            "provided".split())
ADVICE.extend((name, re.compile(rx)) for name, rx in [
    ("guarantee", r"\b(?:can|could|will|would|do|does) (?:you|they|it|the app|the company) guarantee\b"),
    ("concern", r"\b(?:create|creates|cause|causes|raise|raises|pose|poses) (?:a |any )?(?:privacy|security|safety|"
                r"legal) (?:concerns?|risks?|issues?|problems?)\b|\b(?:should|do|does|would) (?:i|we) (?:have|need to "
                r"have|be) (?:any )?concern(?:s|ed)?\b|\bconcerns? (?:about|with)\b"),
    ("repercussions", r"\brepercu\w*\b"),
    ("secure from", r"\b(?:secure|secured|safe|protected) (?:from|against) (?:hackers?|attacks?|breaches|theft|"
                    r"viruses?|malware|cyber)\b"),
    ("will it stay safe", r"^(?!how|what)[^?]*\bwill\b[^?]{0,40}\b(?:keep|stay|remain|be kept)\b[^?]{0,30}\b(?:safe|"
                          r"secure|private|protected|confidential)\b"),
])
HOW_QUANT.update({"constantly": "FREQUENCY", "regularly": "FREQUENCY"})
# "Is there a clause requiring arbitration or mediation?": a presence question; its "or" is content.
PRESENCE_LEAD = re.compile(r"^(?:is there|are there|does (?:the|this|my) (?:contract|agreement|policy|lease|nda|document|"
                           r"license|plan|terms?) (?:have|contain|include|say|mention|require|specify|state|provide))\b")
REPEATED_MARK = re.compile(r"\b(when|if|after|before|until|once|while) [^?]{1,60}? or \1\b")
NUMBER_WORD = re.compile(r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|"
                         r"thirty|forty|fifty|sixty|ninety|hundred)\b")

# Words that make two otherwise parallel options different values: "New York law or Delaware
# law", "$350 per quarter or $350 per month", "twice a month or once a month".
VALUE_WORDS = frozenset("once twice thrice day days week weeks month months quarter quarters year years hour hours "
                        "daily weekly monthly quarterly annually yearly biweekly".split())
