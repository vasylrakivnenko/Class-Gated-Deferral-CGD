# REFLEXIVE v1 — skeleton-first response selection with a confidence gate, on ABCD

The fast path does **not generate text**. For each agent turn it recognizes the
situation from the conversation so far, picks a response **skeleton** (an ordered
list of dialogue acts the training data has seen), picks one stored **sentence
template** per act, fills the template slots from known values — or picks an
**action** (a button call) with values instead of speaking — and, when it is not
confident, **escalates** the turn to an LLM.

Success was *designed* as a two-arm experiment on identical turns: Arm A = LLM
only, Arm B = fast path + escalation to the same LLM. The headline would be the
reflex rate (share of turns handled without the LLM) and the quality gap between
arms.

Implementation spec: REFLEXIVE v1, sections referenced throughout as "spec 6.2",
"spec 11.3" and so on.

---

## STATUS — read this before any number in this file (2026-09-20)

**The `run → calibrate → gate → evaluate → report` pipeline has never been
executed.** `outputs/runs/` holds only `.gitkeep`; `outputs/calibration/` is
empty; `outputs/checkpoints/` does not exist. There is **no reflex rate, no
Arm A/Arm B comparison, no containment number and no spec 9 verdict** anywhere in
this repository, and nothing in this README should be read as reporting one. The
only stage of that pipeline that has ever run is `reflex compile`, whose output
is in `outputs/compile/`.

**Every published result in this repo comes from two side tracks, not from
`src/reflex/`:**

| where | what it produced |
|---|---|
| `probes/run_response_probe.py`, `outputs/probes/` | the TF-IDF+logreg "learned cache" head probes (H5 skeleton, H7 template), the committee gate, recall@k |
| `sft/eval/`, `sft/eval/data/` | the SFT arms (Qwen3-4B, qwen3-0.6B, SmolLM2), the retrieval / n-gram / RNN / LangCache baselines and every LLM-judge sample |

The results table is **`Chat_Leaderboard.md`** (verified 2026-09-20 against the
artifacts on disk). `DECISIONS.md` is a running log whose prose has repeatedly
disagreed with the artifacts — where the two differ, `Chat_Leaderboard.md` and
the JSON under `outputs/` are correct.

The `src/reflex/` modules below are implemented and unit-tested (561 tests pass),
but "implemented and tested" here means *will not crash on first run*, not
*has produced a measured result*.

---

## Start here

| file | what it is |
|---|---|
| `src/reflex/contracts.py` | **The frozen module-boundary contract.** Every public function of every spec Section 4 module, with full type hints, the MECE ownership map, and three boundary rulings. Read it before writing any code. |
| `src/reflex/schemas.py` | Spec Section 5 schemas as frozen dataclasses. Field names are normative. |
| `configs/default.yaml` | Spec Section 12 verbatim, plus marked additions. Spec 10: no threshold, model id, path or price may appear in code. |
| `tests/test_contracts.py` | Mechanically enforces the contract: every module defines exactly the public names it owns, with byte-identical signatures. 228 of the suite's 561 tests. |
| `Chat_Leaderboard.md` | **The results.** Every measured number this project has. Verified 2026-09-20 against the artifacts. |

All twelve spec Section 4 modules are **implemented** — `data`, `compile`,
`models`, `train`, `calibrate`, `select`, `gate`, `fill`, `llm_agent`, `run`,
`evaluate`, `report`; none is a stub any more. Two more exist beyond them:
`src/reflex/featurize.py` (the shared feature builder the probes and the runtime
both use) and `src/reflex/arm_b0.py` (the bank-only Arm B0 variant). With
`config.py`, `contracts.py` and `schemas.py` that is 21,107 lines under
`src/reflex/`. `config.py` was implemented first, because every other module
needs it.

## Two hard rules

**1. Zero paid API calls.** `llm.enabled` ships `false`. `src/reflex/llm_agent.py`
is the only place that may call a hosted model, and it must raise
`LLMDisabledError` while disabled. Act labeling (spec 6.2 step 3a) defaults to
`compile.act_labeler: rules_plus_embed`, a free local labeler; the `llm` labeler
sits behind the same interface, off. `llm.strong`, `llm.cheap`, the price table
and `price_list_date` ship as `<fill: ...>` placeholders — **do not invent model
ids or prices**; `config.require_filled` raises rather than guessing.
`tests/test_no_paid_calls.py` (19 tests) enforces all of this. `llm_agent.py` is
now fully implemented, so those tests are load-bearing, not aspirational.

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

# Dependencies are pinned in requirements.txt. As of 2026-09-20, pysbd 0.3.4 IS
# installed in that venv; faiss-cpu is still MISSING and must be installed before
# anything touches the spec 6.6 novelty index (calibrate / run / sweep):
pip install faiss-cpu==1.12.0
```

Checked 2026-09-20 in `/Users/vasyl/zadumai/.venv`: `import pysbd` → 0.3.4 OK,
`import faiss` → `ModuleNotFoundError`. `reflex compile` therefore runs today
(it needs only pysbd) and has been run; anything that builds the novelty index
has not.

ABCD is already downloaded; `data.abcd_dir` in `configs/default.yaml` points at
it. To fetch from scratch: `scripts/download_abcd.sh <dest>`, then update
`data.abcd_dir`.

## How to reproduce

Spec 7 fixes the execution order: **E0 → E1 → E2 → E5 → E3 → E4 → E6.**

**Only the first two steps below have ever been run.** Everything from `train`
onwards is the intended sequence, not a record of one; see STATUS above. The
commands were re-checked against `src/reflex/__main__.py` on 2026-09-20 and the
flags shown are the ones argparse actually accepts.

```bash
# [RUN] E0 sanity — the mandatory spec 3.3 verification (this is what produced
# the "Dataset notes" output below)
python scripts/inspect_abcd.py            # or: python -m reflex inspect

# [RUN] Build the BANK from train only (spec 6.2) -> outputs/compile/
python -m reflex compile

# [NEVER RUN] Train all three seeds (spec 6.4 requires all of train.seeds = [1,2,3]).
# Writes train.checkpoint_dir = outputs/checkpoints/seed<N>.pt, which does not exist.
for s in 1 2 3; do python -m reflex train --seed "$s"; done

# [NEVER RUN] Gate thresholds from dev, seen subflows only (spec 6.8).
# outputs/calibration/ is empty. gate.py needs six quantiles including the H4
# "value" head (see outputs/REQUIRED_SLOTS_GATE_FIX_NOTE.md); calibrate.py:135
# now emits all six, so this no longer blocks a run.
python -m reflex calibrate --checkpoint outputs/checkpoints/seed1.pt --seed 1

# [NEVER RUN] E2 Arm B. --checkpoint is REQUIRED for --arm B (the CLI exits 1
# without it). E1 Arm A needs llm.enabled: true and a filled price table, so it
# cannot run under the zero-paid-calls rule as shipped.
python -m reflex run --arm B --model strong --split test_seen --alpha 0.02 \
    --checkpoint outputs/checkpoints/seed1.pt
python -m reflex run --arm B --model strong --split test_novel \
    --checkpoint outputs/checkpoints/seed1.pt

# [NEVER RUN] THE $0 MODE: the gate's real verdict is logged but escalated turns
# are recorded UNANSWERED, so spec 8.1 routing / 8.4 novelty come out with
# llm.enabled still false. It cannot produce arm quality or spec 8.7 parity.
python -m reflex run --arm B --split test_seen --forced-reflex \
    --checkpoint outputs/checkpoints/seed1.pt

# [NEVER RUN] Score and report
python -m reflex evaluate --run-id <run_id> --baseline-run-id <armA_run_id>
python -m reflex report --run-ids <run_id> ... --out report.md

# [NEVER RUN] E5 only. run_sweep raises NotImplementedError for E3, E4 and E6
# (they need ablation checkpoints / a learning curve / a passing E2), and E5
# reads the checkpoint from the REFLEX_CHECKPOINT environment variable.
REFLEX_CHECKPOINT=outputs/checkpoints/seed1.pt python -m reflex sweep --experiment E5

# Any config value is overridable, repeatably (spec 10)
python -m reflex run --arm B --split test_seen --checkpoint outputs/checkpoints/seed1.pt \
    --set gate.alpha=0.05 --set data.context_turns_K=10

pytest        # 561 passed, 0 failed (2026-09-20)
```

The probe and SFT tracks that produced every published number are run
separately: `PYTHONPATH=src python -m probes.run_response_probe
{conformance|audit|validate|measure|select}` and the scripts under `sft/eval/`.
See `Chat_Leaderboard.md` for which artifact backs which row.

> **Do not re-run `probes.run_response_probe validate` yet.** A 2026-09-20 code
> review found a half-applied edit at `probes/run_response_probe.py:548` (the
> winner-ranking key was changed to `(cells_in_band, headline_in_band)` while the
> certification predicate at line 563 still reads `best["headline_in_band"]`).
> Replayed against the committed `outputs/probes/response/validate.json` it now
> picks a different vectorizer and writes `certified: false`, after which
> `measure` and `select` refuse unless `--force-uncertified` is passed. The
> committed `certification.json` (`certified: true`, `min_df=1`,
> `sublinear_tf=true`) is what every published probe number was produced with.

---

## License

**ABCD data and official code: MIT** — `Copyright (c) 2021 ASAPP Research`.

Recorded per spec 3.5. **Verified 2026-09-20 from the artifact, not from the
brief:** the clone at `data.abcd_dir` (`/Users/vasyl/zadumai/data/abcd`, a full
`git clone` of `github.com/asappresearch/abcd` at commit `6b8700c`) contains a
21-line `LICENSE` whose first two lines are `MIT License` /
`Copyright (c) 2021 ASAPP Research`. The vendored `README.md` still has no
license section — only a citation block — but the LICENSE file settles it.

(An earlier version of this section said the download had no LICENSE and that the
MIT attribution was "not verifiable from the downloaded artifact". Whatever was
true of the 2026-09-16 download, it is not true of the clone `data.abcd_dir`
resolves to today — `git remote -v` there is
`https://github.com/asappresearch/abcd`, and the LICENSE is present.)

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

**Re-run 2026-09-20 and diffed line by line against the block below: every
number, count and percentage is unchanged.** The only difference was section 0's
five file paths, which had been captured from a temporary 2026-09-16 download
directory; they have been replaced with the paths `data.abcd_dir` resolves to
today. Sizes and all downstream counts are identical, so it is the same corpus.

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

### A ninth finding: the official metrics still do not import bare — but not for the reason first recorded

Spec 13 forbids reimplementing the official AST/CDS metrics, and spec 8.2 says
to use them. `utils/evaluate.py` opens with
`from components.systems import Application`.

**Corrected 2026-09-20.** This section used to say the download contained only
`data/` and `utils/` and had no `components/` package. That is no longer true:
the full clone at `/Users/vasyl/zadumai/data/abcd` **does** ship
`components/{__init__,datasets,features,models,systems,tools}.py`. A bare import
of `utils/evaluate.py` nevertheless still raises — verified by execution —
`ModuleNotFoundError: No module named 'tensorboardX'`, because
`components/tools.py:5` does `from tensorboardX import SummaryWriter` and
`tensorboardX` is not in the venv and is not in `requirements.txt`.

The resolution is unchanged and still in force:
`evaluate.load_official_metrics` installs stub `components.*` modules into
`sys.modules` before importing, so the real `components/` (and therefore
`tensorboardX`) is never touched; the stub's `Application.prepare_masks` raises.
All four report functions only touch `Application` on the `kb_labels is not None`
path. **Verified 2026-09-20 by execution: `load_official_metrics(cfg)` returns
the four real functions from `/Users/vasyl/zadumai/data/abcd/utils/evaluate.py`.**
The cost is that the KB-masked variant is unavailable, so `eval.use_kb_labels`
must stay `false` (`evaluate.py:218` raises `ContractViolation` if it is true)
and the report must say the numbers are unmasked. Lifting this now needs
`pip install tensorboardX` plus dropping the shim, not vendoring — and
`configs/default.yaml:388`'s comment "components/ is not vendored (README)"
should be corrected when someone owns that file.

### Verbatim output

```text

==============================================================================
0. FILES
==============================================================================
raw             121.55 MB  /Users/vasyl/zadumai/data/abcd/data/abcd_v1.1.json
utterances        5.23 MB  /Users/vasyl/zadumai/data/abcd/data/utterances.json
ontology          0.01 MB  /Users/vasyl/zadumai/data/abcd/data/ontology.json
kb                0.01 MB  /Users/vasyl/zadumai/data/abcd/data/kb.json
guidelines        0.10 MB  /Users/vasyl/zadumai/data/abcd/data/guidelines.json

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
  src/reflex/data.py             # 4.1      IMPLEMENTED
  src/reflex/compile.py          # 4.2 + 6.3 IMPLEMENTED
  src/reflex/models.py           # 6.4      IMPLEMENTED
  src/reflex/train.py            # 4.3      IMPLEMENTED
  src/reflex/calibrate.py        # 4.4      IMPLEMENTED
  src/reflex/select.py           # 4.5      IMPLEMENTED
  src/reflex/gate.py             # 4.6      IMPLEMENTED
  src/reflex/fill.py             # 4.7      IMPLEMENTED
  src/reflex/llm_agent.py        # 4.8      IMPLEMENTED (inert: llm.enabled false)
  src/reflex/run.py              # arm orchestration  IMPLEMENTED
  src/reflex/evaluate.py         # 4.9      IMPLEMENTED
  src/reflex/report.py           # 4.10     IMPLEMENTED
  src/reflex/featurize.py        # + not in 11.1; shared feature builder
  src/reflex/arm_b0.py           # + not in 11.1; bank-only Arm B0 variant
  tests/                         # 16 test modules, 561 tests
  probes/                        # + not in 11.1; run_response_probe.py + probe.yaml
  sft/                           # + not in 11.1; SFT data builders and sft/eval/ arms
  DECISIONS.md                   # + running decision log (prose; superseded by the artifacts)
  Chat_Leaderboard.md            # + THE RESULTS, verified 2026-09-20
  AUDIT_2026-09-19.md            # + the 187-defect audit
  DEFECTS_OPEN.md                # + open-defect register
  PROPOSED_CONFIG_B0.yaml        # + Arm B0 proposal; not in configs/, but tests/test_arm_b0.py
                                 #   loads it so it cannot drift from arm_b0.py
  outputs/
    compile/                     # bank/, labels/train.jsonl, compile_summary.md,
                                 #   delex_check.md, act_check.md   [POPULATED]
    probes/response/             # every probe artifact behind the leaderboard [POPULATED]
    probes/analysis/             # coverage_gap.json, min_count_elasticity.json [POPULATED]
    act_audit/                   # act_audit_sample.md (UNFILLED), key, scorer
    *_FIX_NOTE.md                # three fix notes for src/reflex/ defects
    calibration/                 # EMPTY — gate.json has never been written
    runs/                        # EMPTY except .gitkeep — no run has ever happened
    checkpoints/                 # DOES NOT EXIST — no model has ever been trained
```

## Implementation status

"Implemented" below means the code exists and its unit tests pass. It does **not**
mean the module has ever produced a result on real data — see STATUS.

| module | spec | code | has it ever run on real data? |
|---|---|---|---|
| `schemas.py` | 5 | **frozen** | n/a |
| `contracts.py` | 4 | **frozen** | n/a |
| `config.py` | — | implemented | yes |
| `__main__.py` | 11.2 | implemented | yes (`inspect`, `compile`) |
| `scripts/inspect_abcd.py` | 3.3 | implemented | **yes** — output above, re-run 2026-09-20 |
| `data` | 4.1 | implemented | yes (loaded by `compile` and the probes) |
| `compile` | 4.2 + 6.3 | implemented | **yes** — `outputs/compile/` |
| `featurize` | — (added) | implemented | yes (the probes use it) |
| `models` `train` | 6.4, 4.3 | implemented | **no** — `outputs/checkpoints/` does not exist |
| `calibrate` | 4.4 | implemented | **no** — `outputs/calibration/` is empty |
| `select` `gate` `fill` | 4.5–4.7 | implemented | **no** — reachable only from a run |
| `llm_agent` | 4.8 | implemented | **no**, by design (`llm.enabled: false`) |
| `run` `arm_b0` | — | implemented | **no** — `outputs/runs/` is empty |
| `evaluate` `report` | 4.9, 4.10 | implemented | **no** — no `metrics.json` exists |
| `tests/` | 10 | 16 modules, 561 tests | **yes — 561 passed, 0 failed (2026-09-20)** |

Every module still imports its types from `contracts.py`, and
`pytest tests/test_contracts.py` (228 tests) enforces byte-identical signatures
against the frozen contract.

## Deviations from the spec

Recorded rather than silently absorbed.

1. **Python 3.12.8, not 3.11.** Spec 10 asks for 3.11; `/Users/vasyl/zadumai/.venv`
   is 3.12.8. `requirements.txt` pins to that venv's versions to avoid churn.
   `pyproject.toml` declares `>=3.11`.
2. **Files added beyond spec 11.1.** `src/reflex/config.py` — every module needs
   `cfg` and spec 10 bans literals, so config loading had to live somewhere;
   spec Section 4 assigns it to no module. `pyproject.toml` — so
   `python -m reflex` works (spec 11.2) without `PYTHONPATH`. Since then,
   `src/reflex/featurize.py` and `src/reflex/arm_b0.py`, plus the whole
   `probes/` and `sft/` trees, which are where the published results come from.
3. **`faiss-cpu` is not installed** in that venv (checked 2026-09-20). It backs
   the spec 6.6 novelty index and is pinned in `requirements.txt`; install it
   before `calibrate` / `run` / `sweep`. `pysbd` 0.3.4 (the spec 12
   `compile.sentence_splitter`) **is** installed, which is why `reflex compile`
   has been able to run.
4. **`eval.use_kb_labels` is forced `false`.** The reason changed: the full clone
   now *does* ship `components/`, but `components/tools.py` imports
   `tensorboardX`, which is absent, so `evaluate.load_official_metrics` still
   shims `components.*` and the shim's `Application.prepare_masks` raises. The
   KB-masked official metrics remain unreachable. See "A ninth finding" above.
   (`configs/default.yaml:388` still carries the old "components/ is not
   vendored" comment and is wrong.)
5. **Arm A cannot run as shipped.** `llm.enabled: false` and the model ids and
   prices are `<fill>` placeholders, so spec 7's E1 — and therefore the spec 9
   parity criterion, which is defined against E1 — cannot be produced without a
   deliberate, funded change. Arm B, the reflex rate, the fast-path error rates
   and the novelty numbers are all measurable for free (`--forced-reflex`) —
   **but none of them has been measured: no Arm B run exists.** **This is a real
   limit on what v1 can conclude, not a bug:** spec 9's PASS requires a parity
   comparison against an arm that costs money.
6. **`prompts/agent_A.txt` is a draft, not frozen.** Spec 6.9 step 3 allows two
   revisions on dev before freezing. Revisions used: 0. It cannot be revised on
   dev without running the LLM, so freezing is blocked by item 5.
7. **`NormalizedTurn` adds three fields** beyond spec 5.1: `utt_rank` (D2),
   `turn_count` (the official `cds_report` needs it in `ci_and_tc`) and
   `is_synthetic_end` (D3). `Decision` adds `candidate_rank`,
   `exact_template_match` (required by spec 6.5 step 5 but absent from spec 5.5)
   and `cache_hit`. All spec-5 field names are unchanged.
8. **Twelve schemas added** that the spec describes in prose but never gives a
   JSON shape — `ContextWindow`, `SlotSpec`, `SlotRegistry`, `Bank`, `TurnLabel`,
   `SelectorScores`, `Selection`, `Calibration`, `SlotSources`, `LLMDecision`,
   `Partitions`, `RunManifest` (counted from `src/reflex/schemas.py`, which lists
   exactly these; the README used to say eleven). Each is marked `REFLEX-ADDED`.

## Three boundary rulings

Made once, in `contracts.py`, so nobody has to guess:

1. **Novelty distance** is computed in `select` and only *thresholded* in `gate`;
   the FAISS index is built by `calibrate`. Computing a cosine is not deciding.
2. **Missing-slot detection** lives in `fill.check_availability` (4.7 owns slot
   logic); `gate` consumes the list and turns it into `unavailable_slot`.
3. **The early-stopping score** lives in `train.dev_selection_score` and is
   explicitly *not* a reported metric — spec 4.9 makes `evaluate` the only place
   Section 8 numbers are computed.
