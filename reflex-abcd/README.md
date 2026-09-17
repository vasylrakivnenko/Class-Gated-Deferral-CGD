# REFLEXIVE v1 — skeleton-first response selection with a confidence gate, on ABCD

The fast path does **not generate text**. For each agent turn it recognizes the
situation from the conversation so far, picks a response **skeleton** (an ordered
list of dialogue acts the training data has seen), picks one stored **sentence
template** per act, fills the template slots from known values — or picks an
**action** (a button call) with values instead of speaking — and, when it is not
confident, **escalates** the turn to an LLM.

Success is a two-arm experiment on identical turns: Arm A = LLM only, Arm B =
fast path + escalation to the same LLM. The headline result is the reflex rate
(share of turns handled without the LLM) and the quality gap between arms.

Implementation spec: REFLEXIVE v1, sections referenced throughout as "spec 6.2",
"spec 11.3" and so on.

---

## Start here

| file | what it is |
|---|---|
| `src/reflex/contracts.py` | **The frozen module-boundary contract.** Every public function of every spec Section 4 module, with full type hints, the MECE ownership map, and three boundary rulings. Read it before writing any code. |
| `src/reflex/schemas.py` | Spec Section 5 schemas as frozen dataclasses. Field names are normative. |
| `configs/default.yaml` | Spec Section 12 verbatim, plus marked additions. Spec 10: no threshold, model id, path or price may appear in code. |
| `tests/test_contracts.py` | Mechanically enforces the contract. Passes while modules are stubs; keeps passing as they land. |

Twelve modules are stubs that re-export from `contracts.py`. `src/reflex/config.py`
is **already implemented**, because all twelve need it on day one.

## Two hard rules

**1. Zero paid API calls.** `llm.enabled` ships `false`. `src/reflex/llm_agent.py`
is the only place that may call a hosted model, and it must raise
`LLMDisabledError` while disabled. Act labeling (spec 6.2 step 3a) defaults to
`compile.act_labeler: rules_plus_embed`, a free local labeler; the `llm` labeler
sits behind the same interface, off. `llm.strong`, `llm.cheap`, the price table
and `price_list_date` ship as `<fill: ...>` placeholders — **do not invent model
ids or prices**; `config.require_filled` raises rather than guessing.
`tests/test_no_paid_calls.py` enforces all of this and starts biting the moment
`llm_agent` is implemented.

**2. Config-driven and deterministic.** Read `cfg`, never a literal (spec 10).
Same context → same Decision, always.

---

## Setup

The venv used for this work is `/Users/vasyl/zadumai/.venv`.

```bash
# Option A — no install, what the commands below assume
export PYTHONPATH=src

# Option B — editable install, gives you the `reflex` console script
pip install -e .

# Dependencies are pinned in requirements.txt. Two are NOT in the venv yet
# and must be installed before `reflex compile` or anything touching the
# novelty index:
pip install faiss-cpu==1.12.0 pysbd==0.3.4
```

ABCD is already downloaded; `data.abcd_dir` in `configs/default.yaml` points at
it. To fetch from scratch: `scripts/download_abcd.sh <dest>`, then update
`data.abcd_dir`.

## How to reproduce

Spec 7 fixes the execution order: **E0 → E1 → E2 → E5 → E3 → E4 → E6.**

```bash
# E0 sanity — the mandatory spec 3.3 verification (this is what produced the
# "Dataset notes" output below)
python scripts/inspect_abcd.py            # or: python -m reflex inspect

# Build the BANK from train only (spec 6.2) -> outputs/compile/
python -m reflex compile

# Train all three seeds (spec 6.4 requires the mean and spread of all three)
for s in 1 2 3; do python -m reflex train --seed "$s"; done

# Gate thresholds from dev, seen subflows only (spec 6.8)
python -m reflex calibrate --checkpoint outputs/checkpoints/seed1.pt --seed 1

# E2 Arm B. (E1 Arm A needs llm.enabled: true and a filled price table, so it
# cannot run under the zero-paid-calls rule as shipped.)
python -m reflex run --arm B --model strong --split test_seen --alpha 0.02
python -m reflex run --arm B --model strong --split test_novel

# Score and report
python -m reflex evaluate --run-id <run_id> --baseline-run-id <armA_run_id>
python -m reflex report --run-ids <run_id> ... --out report.md

# E5 / E3 / E4 / E6 matrices
python -m reflex sweep --experiment E5

# Any config value is overridable, repeatably (spec 10)
python -m reflex run --arm B --split test_seen --set gate.alpha=0.05 --set data.context_turns_K=10

pytest
```

---

## License

**ABCD data and official code: MIT** (ASAPP Research), per the project brief.

Recorded per spec 3.5, with one caveat stated plainly: **the local download does
not contain a LICENSE file**, and the vendored `README.md` has no license
section — only a citation block. So the MIT attribution above is taken from the
brief and from the upstream GitHub repository page, and was **not verifiable from
the downloaded artifact**. Before any public use of results, read the LICENSE in
a full clone of `github.com/asappresearch/abcd` and replace this paragraph with
the verified text.

Cite the dataset:

> Derek Chen, Howard Chen, Yi Yang, Alex Lin, Zhou Yu. "Action-Based
> Conversations Dataset: A Corpus for Building More In-Depth Task-Oriented
> Dialogue Systems." NAACL-HLT 2021, pp. 3002–3017.
> <https://www.aclweb.org/anthology/2021.naacl-main.239>

---

## Dataset notes

Spec 3.3 makes this section mandatory: "Record any deviation from 3.2 in
README.md under 'Dataset notes'." Everything below is the **real, unedited
output** of `python scripts/inspect_abcd.py --sample 20 --seed 0`, which checks
every structural assumption of spec 3.2 over all 220,983 turns rather than
sampling. The only edit is to section 7, where 17 of the 20 sampled turns are
omitted for length — rerun the command for the full listing.

### The eight deviations, in one table

| # | spec 3.2 says / implies | verified truth | consequence |
|---|---|---|---|
| **D1** | "a processed turn list", unnamed | it is `conversation["delexed"]`. `len(original) == len(delexed)` on **10,042/10,042** conversations, index-aligned 1:1 | use `delexed`; `original[i][1]` is the lexicalized twin |
| **D2** | `targets[4]` is a `utt_rank`/id (official `process.py` unpacks it as `utt_id`) | it is a **RANK into this turn's `candidates`**. `utterances[candidates[targets[4]]]` matches the original text on **95,127/95,129 (99.998%)**; the global reading `utterances[targets[4]]` matches **25/95,129 (0.026%)** | `gold_utt_id = candidates[targets[4]]`. `NormalizedTurn` carries **both** `utt_rank` (raw, what the official evaluator wants) and `utt_id` (resolved, for text lookup) |
| **D3** | `end_conversation` is one of three nextsteps | it **never appears in the raw data** (0 turns). `utils/process.py` synthesizes one extra example per conversation: `end_targets[1]="end_conversation"; end_targets[4]=-1` | must be replicated (`data.synthesize_end_conversation`), +10,042 turns, or nextstep accuracy is not comparable to published numbers |
| **D4** | (not mentioned) | `delexed` text is **already partially delexicalized**: **21,297** `<slot>` markers over 17,448 turns, drawn from `ontology["values"]["non_enumerable"]` | ABCD's own `<slot>` vocabulary is the **slot-registry backbone** (spec 6.3). Do not build a delexicalizer from scratch or invent parallel names |
| **D5** | `utterances.json` = "list of agent utterance strings" | the pool is **LEXICALIZED**: only **209/95,288** entries carry a marker, vs 21,297 markers in `delexed` text | delexicalize the 100 candidates before comparing them to composed template text (spec 6.5 step 5), or the right answer scores near zero |
| **D6** | (not mentioned) | **357/21,297 (1.68%)** markers are **glued to an adjacent alphanumeric** — ABCD did partial-substring replacement: `"$1<amount>"` for `$164`, `"<order_id>3"`, `"4<street_address>"`, `"$4.<amount>"` for `$4.99` | those spans are not clean slots and are **not reversible by the filler**. `compile.guard_glued_markers` leaves them literal; emitting `{amount}` there would produce `"$1123"` |
| **D7** | "55 subflows (intents) under 10 flows" | `ontology["intents"]["subflows"]` is a **dict keyed by the 10 flows**, not a flat list of 55 | flatten in file order (flow, then subflow) — `data.subflow_list` is the one place that does it. `kb.json`'s 55 keys match exactly |
| **D8** | "actions (buttons)" | `ontology["actions"]` is a **dict of 3 categories** (`kb_query` 6, `interaction` 10, `faq_policy` 14) = 30 buttons | flatten in file order — matches the "intent mask should be size of 30 long" comment in the official `cds_report` |

**On D2, the brief's hypothesis was wrong in an instructive way.** The 376 turns
where the resolved gold string does not equal the `delexed` text are **not
duplicate-utterance collisions**. They are exactly the turns where ABCD's own
delexicalizer fired: the candidate pool keeps the literal value, the turn text
carries the marker. Checked against `original[i][1].lower()` instead, the rank
hypothesis holds on 95,127/95,129, and **0 turns match neither**. The 2 residuals
are pipe-character artifacts in `original` — ABCD replaced `"...now. |What
product is it"` with `"...now. andwhat product is it"`. So the rank reading is
not "mostly right"; it is right, and D5 explains every apparent exception.

### A ninth finding: the official metrics do not import as downloaded

Spec 13 forbids reimplementing the official AST/CDS metrics, and spec 8.2 says
to use them. But `utils/evaluate.py` opens with
`from components.systems import Application`, and **the local download contains
only `data/` and `utils/` — there is no `components/` package**, so a bare import
raises `ModuleNotFoundError`.

Resolved without a rewrite: `evaluate.load_official_metrics` installs stub
`components.*` modules into `sys.modules` before importing, where
`Application.prepare_masks` raises. All four report functions only touch
`Application` on the `kb_labels is not None` path, so every metric this project
needs works untouched. **Verified: `ast_report` and `cds_report` both run to
completion under the shim.** The cost is that the KB-masked variant is
unavailable, so `eval.use_kb_labels` must stay `false` and the report must say
the numbers are unmasked. Vendoring `components/` from a full clone would lift
this.

### Verbatim output

```text

==============================================================================
0. FILES
==============================================================================
raw             121.55 MB  /private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad/abcd/data/abcd_v1.1.json
utterances        5.23 MB  /private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad/abcd/data/utterances.json
ontology          0.01 MB  /private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad/abcd/data/ontology.json
kb                0.01 MB  /private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad/abcd/data/kb.json
guidelines        0.10 MB  /private/tmp/claude-501/-Users-vasyl-zadumai/12e40f00-be9f-4f43-9226-7a491b668aef/scratchpad/abcd/data/guidelines.json

config     : /Users/vasyl/zadumai/reflex-abcd/configs/default.yaml
turn list  : conversation['delexed']  (spec 3.2 does not name it)

==============================================================================
1. CONVERSATIONS PER SPLIT (spec 3.3)
==============================================================================
split keys: ['dev', 'test', 'train']
  train    8034 conversations
  dev      1004 conversations
  test     1004 conversations
  TOTAL   10042

==============================================================================
2. EXACT KEY NAMES (spec 3.3)
==============================================================================
conversation keys : ['convo_id', 'delexed', 'original', 'scenario']
  expected (3.2)  : ['convo_id', 'delexed', 'original', 'scenario']
scenario keys     : ['flow', 'order', 'personal', 'product', 'subflow']
turn keys         : ['candidates', 'speaker', 'targets', 'text', 'turn_count']
  expected (3.2)  : ['candidates', 'speaker', 'targets', 'text', 'turn_count']

targets layout    : [intent, nextstep, action, values, utt_rank]  (len 5)
ontology keys     : ['actions', 'intents', 'next_steps', 'values', 'vocabulary']
  intents keys    : ['flows', 'subflows']
  values keys     : ['enumerable', 'non_enumerable']
  next_steps      : ['retrieve_utterance', 'take_action', 'end_conversation']
  next_steps order OK (NORMATIVE: cds_report branches on label == 0/1/2)

example conversation 3592, first 4 turns of 'delexed':
  [0] speaker=agent    turn_count=  1 n_candidates=100
      text    = 'hi!'
      targets = ['return_size', 'retrieve_utterance', None, [], 87]
      original[0] = ['agent', 'Hi!']
  [1] speaker=agent    turn_count=  2 n_candidates=100
      text    = 'how can i help you?'
      targets = ['return_size', 'retrieve_utterance', None, [], 35]
      original[1] = ['agent', 'How can I help you?']
  [2] speaker=customer turn_count=  3 n_candidates=  0
      text    = 'hi! i need to return an item, can you help me with that?'
      targets = ['return_size', None, None, [], -1]
      original[2] = ['customer', 'Hi! I need to return an item, can you help me with that?']
  [3] speaker=agent    turn_count=  4 n_candidates=100
      text    = 'sure, may i have your name please?'
      targets = ['return_size', 'retrieve_utterance', None, [], 21]
      original[3] = ['agent', 'sure, may I have your name please?']

==============================================================================
3. TURN-TYPE COUNTS (spec 3.3)
==============================================================================
split        agent  customer    action     total
train        75985     71259     29190    176434
dev           9600      9096      3684     22380
test          9544      9017      3608     22169
ALL          95129     89372     36482    220983

nextstep counts (targets[1], all splits):
  retrieve_utterance     95129
  None                   89372
  take_action            36482

speaker -> nextstep pairing (all splits):
  action    -> take_action            36482
  agent     -> retrieve_utterance     95129
  customer  -> None                   89372
  => speaker determines nextstep 1:1.

'end_conversation' in raw data: 0 turns
  !! DEVIATION: end_conversation NEVER appears in raw ABCD.
     utils/process.py synthesizes ONE extra example per conversation after
     the last turn: end_targets = turn['targets'].copy();
     end_targets[1]='end_conversation'; end_targets[4]=-1.
     Replicating it adds 10042 turns (data.synthesize_end_conversation).

candidate-list length by speaker: {'action:0': 36482, 'agent:100': 95129, 'customer:0': 89372}
targets tuple lengths           : {5: 220983}
len(values) on action turns     : {0: 11327, 1: 19057, 3: 6098}
len(original) == len(delexed) : 10042/10042 conversations

==============================================================================
4. SUBFLOWS (spec 3.3)
==============================================================================
ontology['intents']['subflows'] is a dict of len 10
  !! DEVIATION: it is a DICT KEYED BY THE 10 FLOWS, not a flat list of 55.
     Flatten it (flow order, then file order) to get the canonical 55.

10 flows, 55 subflows:
  account_access         ( 3) recover_username, recover_password, reset_2fa
  manage_account         ( 8) status_service_added, status_service_removed, status_shipping_question, status_credit_missing, manage_change_address, manage_change_name, manage_change_phone, manage_payment_method
  order_issue            ( 8) status_mystery_fee, status_delivery_time, status_payment_method, status_quantity, manage_upgrade, manage_downgrade, manage_create, manage_cancel
  product_defect         ( 6) refund_initiate, refund_update, refund_status, return_stain, return_color, return_size
  purchase_dispute       ( 8) bad_price_competitor, bad_price_yesterday, out_of_stock_general, out_of_stock_one_item, promo_code_invalid, promo_code_out_of_date, mistimed_billing_already_returned, mistimed_billing_never_bought
  shipping_issue         ( 4) status, manage, missing, cost
  single_item_query      ( 4) boots, shirt, jeans, jacket
  storewide_query        ( 4) pricing, membership, timing, policy
  subscription_inquiry   ( 6) status_active, status_due_amount, status_due_date, manage_pay_bill, manage_extension, manage_dispute_bill
  troubleshoot_site      ( 4) credit_card, shopping_cart, search_results, slow_speed

kb.json entries: 55; keys == flattened subflows: True
ontology['actions'] categories : {'kb_query': 6, 'interaction': 10, 'faq_policy': 14}
flattened action (button) list : 30 names
  (matches the 'intent mask should be size of 30 long' comment in cds_report)

==============================================================================
5. targets[4]: RANK OR GLOBAL UTTERANCE ID? (full-corpus check)
==============================================================================
utterances.json: 95288 strings (index == the id used in `candidates`)
agent turns with candidates: 95129
  utterances[candidates[targets[4]]] == delexed text :  94753 / 95129  (99.605%)
  utterances[candidates[targets[4]]] == ORIGINAL text:  95127 / 95129  (99.998%)  <-- the truth
  utterances[targets[4]]             == delexed text :     25 / 95129  (0.026%)  <-- global reading fails

  => targets[4] IS A RANK into `candidates`. gold_utt_id = candidates[targets[4]].
     The 376 apparent mismatches against the DELEXED text are NOT duplicate
     collisions: utterances.json is the LEXICALIZED pool, so the gold string keeps
     the literal value where ABCD's delexicalizer replaced it in `delexed`.
     Turns matching NEITHER delexed nor original text: 0

     train convo=3647 turn=12 rank=22 -> utt_id=6973
       utterances[utt_id] = 'the rate is for 39.99 for items 5 lbs or less. and 54.99 for items of 10 lbs or lesss and 69'
       delexed text       = 'the rate is for 39.99 for items 5 lbs or less. and <amount>.99 for items of 10 lbs or lesss '

     train convo=10015 turn=26 rank=85 -> utt_id=95034
       utterances[utt_id] = "i've refunded your account $54. is there anything else i can help you with?"
       delexed text       = "i've refunded your account $<amount>. is there anything else i can help you with?"

     train convo=8749 turn=27 rank=72 -> utt_id=95016
       utterances[utt_id] = '2311 1st ave san mateo, ny 60315'
       delexed text       = '<street_address> san mateo, ny <zip_code>'

==============================================================================
6. `delexed` IS ALREADY PARTIALLY DELEXICALIZED
==============================================================================
markers in `delexed` text : 21297 over 17448 / 220983 turns
  <order_id>           4613
  <account_id>         3075
  <username>           2772
  <email>              2619
  <zip_code>           2411
  <amount>             2228
  <street_address>     1781
  <name>                810
  <phone>               667
  <pin_number>          321

ontology['values']['non_enumerable'] flattened (11): ['account_id', 'amount', 'email', 'full_address', 'name', 'order_id', 'phone', 'pin_number', 'street_address', 'username', 'zip_code']
  => THIS IS ABCD'S OWN SLOT VOCABULARY and the registry BACKBONE (spec 6.3).
     utils/load.py adds exactly [f'<{slot}>' for these] to its tokenizer.
  markers observed in text (10): ['account_id', 'amount', 'email', 'name', 'order_id', 'phone', 'pin_number', 'street_address', 'username', 'zip_code']
  declared but never seen as a marker: ['full_address']

utterances.json strings containing a marker: 209 / 95288 (0.22%)
  => the candidate pool is LEXICALIZED. Delexicalize candidates before comparing
     them to composed template text (spec 6.5 step 5).

markers GLUED to an adjacent alphanumeric: 357 / 21297 (1.68%)
  ABCD's own delexicalizer did partial-substring replacement, so these spans are
  NOT clean slots and are NOT reversible by a filler:
    <order_id>       in ...order id: <order_id>3...
    <amount>         in ...ade for the amount of $1<amount>....
    <amount>         in ...ur refund will now be $1<amount>....
    <amount>         in ...ade for the amount of $1<amount>....
    <street_address> in ...4<street_address>  la fayette, ...
    <street_address> in ... has been updated with 4<street_address> la fayette, c...

==============================================================================
7. 20 RANDOM AGENT UTTERANCES WITH TARGETS (spec 3.3)
==============================================================================
(sampling 20 of 95129 agent turns, seed=0)

[ 1] train convo=2251 turn_index=26 turn_count=27
     text (delexed) : 'thank you!'
     text (original): 'Thank You!'
     targets        : intent='status' nextstep='retrieve_utterance' action=None values=[] utt_rank=55
     resolved utt_id: 443   (candidates[55])
     utterances[443] = 'thank you!'
     scenario       : flow='shipping_issue' subflow='status'

[ 2] train convo=1402 turn_index=4 turn_count=5
     text (delexed) : 'give me your full name or account id'
     text (original): 'Give me your Full name or account id'
     targets        : intent='shopping_cart' nextstep='retrieve_utterance' action=None values=[] utt_rank=34
     resolved utt_id: 3166   (candidates[34])
     utterances[3166] = 'give me your full name or account id'
     scenario       : flow='troubleshoot_site' subflow='shopping_cart'

[ 3] train convo=9277 turn_index=18 turn_count=19
     text (delexed) : 'ok great!'
     text (original): 'ok great!'
     targets        : intent='search_results' nextstep='retrieve_utterance' action=None values=[] utt_rank=32
     resolved utt_id: 16117   (candidates[32])
     utterances[16117] = 'ok great!'
     scenario       : flow='troubleshoot_site' subflow='search_results'

[... 17 more sampled turns omitted from this README excerpt; rerun the command for all 20 ...]

==============================================================================
8. SUMMARY OF DEVIATIONS FROM SPEC 3.2 (record these in README)
==============================================================================
D1 The processed turn list is conversation['delexed']; spec 3.2 does not name it.
   len(original) == len(delexed) on 10042/10042 conversations, index-aligned 1:1.
D2 targets[4] is a RANK into `candidates`, not a global utterance id.
   gold_utt_id = candidates[targets[4]] resolves to the original text on 95127/95129 (99.998%).
D3 'end_conversation' never appears in the raw data; utils/process.py synthesizes
   one extra example per conversation (+10042 turns). Must be replicated.
D4 `delexed` text is ALREADY partially delexicalized: 21297 <slot> markers
   over 17448 turns, drawn from ontology['values']['non_enumerable'].
D5 utterances.json is LEXICALIZED (209/95288 entries carry a marker), unlike `delexed` text.
D6 357/21297 markers are glued to adjacent alphanumerics (partial-substring
   replacement by ABCD), so they are not clean, reversible slots.
D7 ontology['intents']['subflows'] is a dict keyed by the 10 flows, not a flat list of 55.
D8 ontology['actions'] is a dict of 3 categories; flatten it for the 30-button list.

```

---

## Repository layout (spec 11.1)

```
reflex-abcd/
  README.md                      # setup, dataset notes (3.3), license note (3.5), reproduce
  requirements.txt               # PINNED (spec 10)
  pyproject.toml                 # + not in 11.1; makes `python -m reflex` work after install
  pytest.ini
  configs/default.yaml           # Section 12 verbatim + marked additions
  prompts/agent_A.txt            # Arm A prompt — DRAFT, not yet frozen (6.9 step 3)
  prompts/act_labeling.txt       # offline act labeling prompt (unused by default)
  scripts/download_abcd.sh
  scripts/inspect_abcd.py        # Section 3.3 — IMPLEMENTED, output above
  src/reflex/__init__.py
  src/reflex/__main__.py         # Section 11.2 CLI — IMPLEMENTED
  src/reflex/schemas.py          # Section 5 — FROZEN
  src/reflex/contracts.py        # the module-boundary contract — FROZEN
  src/reflex/config.py           # + not in 11.1; IMPLEMENTED
  src/reflex/data.py             # 4.1      stub
  src/reflex/compile.py          # 4.2 + 6.3 stub
  src/reflex/models.py           # 6.4      stub
  src/reflex/train.py            # 4.3      stub
  src/reflex/calibrate.py        # 4.4      stub
  src/reflex/select.py           # 4.5      stub
  src/reflex/gate.py             # 4.6      stub
  src/reflex/fill.py             # 4.7      stub
  src/reflex/llm_agent.py        # 4.8      stub
  src/reflex/run.py              # arm orchestration  stub
  src/reflex/evaluate.py         # 4.9      stub
  src/reflex/report.py           # 4.10     stub
  tests/
  outputs/
    compile/                     # bank/, labels/, delex_check.md, act_check.md
    calibration/gate.json
    runs/<run_id>/               # manifest.json, decisions.jsonl, metrics.json, report.md, figures/
```

## Implementation status

| module | spec | status |
|---|---|---|
| `schemas.py` | 5 | **frozen** |
| `contracts.py` | 4 | **frozen** |
| `config.py` | — | **implemented** |
| `__main__.py` | 11.2 | **implemented** (dispatches to contracts; unimplemented commands exit 3) |
| `scripts/inspect_abcd.py` | 3.3 | **implemented, run, output above** |
| `tests/test_contracts.py`, `tests/test_no_paid_calls.py` | 10 | **implemented** |
| `data` `compile` `models` `train` `calibrate` `select` `gate` `fill` `llm_agent` `run` `evaluate` `report` | 4.1–4.10 | stubs re-exporting `contracts.py` |

Each stub lists its owned names in `__all__`. To implement one: delete the
`from reflex.contracts import ...` block and write real functions with
**byte-identical signatures**. `pytest tests/test_contracts.py` enforces it.

## Deviations from the spec

Recorded rather than silently absorbed.

1. **Python 3.12.8, not 3.11.** Spec 10 asks for 3.11; `/Users/vasyl/zadumai/.venv`
   is 3.12.8. `requirements.txt` pins to that venv's versions to avoid churn.
   `pyproject.toml` declares `>=3.11`.
2. **Two files added beyond spec 11.1.** `src/reflex/config.py` — every module
   needs `cfg` and spec 10 bans literals, so config loading had to live
   somewhere; spec Section 4 assigns it to no module. `pyproject.toml` — so
   `python -m reflex` works (spec 11.2) without `PYTHONPATH`.
3. **`faiss-cpu` and `pysbd` are not installed** in that venv. Both are required
   (spec 10 lists them; `compile.sentence_splitter: pysbd`, and the spec 6.6
   novelty index is FAISS). Pinned in `requirements.txt`; install before
   `compile`.
4. **`eval.use_kb_labels` is forced `false`** — the `components/` package is
   missing from the download, so the KB-masked official metrics are unreachable.
   See "A ninth finding" above.
5. **Arm A cannot run as shipped.** `llm.enabled: false` and the model ids and
   prices are `<fill>` placeholders, so spec 7's E1 — and therefore the spec 9
   parity criterion, which is defined against E1 — cannot be produced without a
   deliberate, funded change. Arm B, the reflex rate, the fast-path error rates
   and the novelty numbers are all measurable for free. **This is a real limit on
   what v1 can conclude, not a bug:** spec 9's PASS requires a parity comparison
   against an arm that costs money.
6. **`prompts/agent_A.txt` is a draft, not frozen.** Spec 6.9 step 3 allows two
   revisions on dev before freezing. Revisions used: 0. It cannot be revised on
   dev without running the LLM, so freezing is blocked by item 5.
7. **`NormalizedTurn` adds three fields** beyond spec 5.1: `utt_rank` (D2),
   `turn_count` (the official `cds_report` needs it in `ci_and_tc`) and
   `is_synthetic_end` (D3). `Decision` adds `candidate_rank`,
   `exact_template_match` (required by spec 6.5 step 5 but absent from spec 5.5)
   and `cache_hit`. All spec-5 field names are unchanged.
8. **Eleven schemas added** that the spec describes in prose but never gives a
   JSON shape (`ContextWindow`, `SlotRegistry`, `Bank`, `Calibration`, …), so
   twelve modules agree on them. Each is marked `REFLEX-ADDED`.

## Three boundary rulings

Made once, in `contracts.py`, so nobody has to guess:

1. **Novelty distance** is computed in `select` and only *thresholded* in `gate`;
   the FAISS index is built by `calibrate`. Computing a cosine is not deciding.
2. **Missing-slot detection** lives in `fill.check_availability` (4.7 owns slot
   logic); `gate` consumes the list and turns it into `unavailable_slot`.
3. **The early-stopping score** lives in `train.dev_selection_score` and is
   explicitly *not* a reported metric — spec 4.9 makes `evaluate` the only place
   Section 8 numbers are computed.
