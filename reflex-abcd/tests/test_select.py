"""Tests for 4.5 ``reflex.select`` -- scores only, no decisions.

These run WITHOUT a checkpoint and without loading any encoder: a stub model
exposing exactly the public surface ``reflex.models.build_model`` promises
(``class_orders``, ``tokenizer``, ``__call__``/forward, ``score_templates``,
``template_features``, ``value_candidate_tokens``) stands in for the real one.
That is deliberate -- it also proves ``select`` touches nothing outside that
documented surface.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from reflex import select as S
from reflex.config import load_config
from reflex.schemas import (
    ActionPattern,
    Bank,
    Calibration,
    ContextWindow,
    NormalizedTurn,
    Skeleton,
    SlotRegistry,
    SlotSpec,
    Template,
)

CONFIG = "configs/default.yaml"

VALUE_LIST = ["guest", "bronze", "silver", "gold", "jeans", "shirt", "crystal minh", "alessandro phoenix"]
SUBFLOWS = ["return_size", "status_delivery_time", "manage_change_address"]
ACTIONS = ["pull-up-account", "verify-identity", "membership"]
ACTS = ["ACK", "VERIFY", "ASK", "INFORM", "INSTRUCT", "CONFIRM", "OFFER", "CLOSE", "OTHER"]


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


class _Tok:
    """Minimal tokenizer: deterministic ids from the characters of the text."""

    pad_token_id = 0
    truncation_side = "right"

    def __call__(self, texts, padding=True, truncation=True, max_length=32, return_tensors="pt"):
        rows = []
        for text in texts:
            ids = [(ord(ch) % 97) + 1 for ch in text][:max_length]
            rows.append(ids or [1])
        width = max(len(r) for r in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [(sum(ord(c) for c in t) % 97) + 1 for t in tokens]


class _Model:
    """A stand-in for ``_ReflexModel`` with deterministic, input-dependent heads."""

    query_dim = 4

    def __init__(self, bank, n_templates):
        self.tokenizer = _Tok()
        self.class_orders = {
            "next_steps": ["retrieve_utterance", "take_action", "end_conversation"],
            "subflows": list(SUBFLOWS),
            "actions": list(ACTIONS),
            "skeleton_ids": [s.skeleton_id for s in bank.skeletons],
            "acts": list(ACTS),
            "value_list": list(VALUE_LIST),
            "template_ids": [t.template_id for t in bank.templates],
        }
        self.enumerable = {
            "membership_level": ["guest", "bronze", "silver", "gold"],
            "product": ["jeans", "shirt"],
            "customer_name": ["crystal minh", "alessandro phoenix"],
        }
        self.value_by_action = {
            "pull-up-account": ["customer_name"],
            "verify-identity": ["customer_name", "account_id", "order_id"],
            "membership": ["membership_level"],
        }
        self.template_features = torch.arange(n_templates * 4, dtype=torch.float32).reshape(n_templates, 4)
        self.refreshed = 0
        self.forward_calls = 0
        self.force_nextstep = None  # pin H1's argmax so a branch can be exercised
        self._n_templates = n_templates

    # -- the documented surface ------------------------------------------------ #
    def eval(self):
        return self

    def refresh_template_cache(self, texts=None, batch_size=None):
        self.refreshed += 1
        return self.template_features

    def __call__(self, batch):
        self.forward_calls += 1
        ids = batch["input_ids"].to(torch.float32)
        seed = float(ids.sum().item())
        query = torch.tensor([[math.sin(seed), math.cos(seed), math.sin(seed / 3.0), 0.5]], dtype=torch.float32)
        out = {
            "query": query,
            "pooled": query,
            "nextstep_logits": torch.tensor([[seed % 3.0, (seed / 2.0) % 3.0, 0.1]], dtype=torch.float32),
            "intent_logits": torch.tensor([[(seed + i) % 5.0 for i in range(len(SUBFLOWS))]], dtype=torch.float32),
            "action_logits": torch.tensor([[(seed * (i + 1)) % 7.0 for i in range(len(ACTIONS))]], dtype=torch.float32),
            "skeleton_logits": torch.tensor([[(seed + 2 * i) % 4.0 for i in range(len(self.class_orders["skeleton_ids"]))]], dtype=torch.float32),
            "act_logits": torch.zeros((1, len(ACTS)), dtype=torch.float32),
        }
        width = len(VALUE_LIST) + 100  # official AST layout: values | model.value_context_len
        value = torch.tensor([[(seed + 0.5 * i) % 3.0 for i in range(width)]], dtype=torch.float32)
        if "context_input_ids" in batch:
            value = value + float(batch["context_input_ids"].sum().item()) % 1.0
        out["value_logits"] = value
        if self.force_nextstep is not None:
            pinned = [0.0, 0.0, 0.0]
            pinned[self.force_nextstep] = 10.0
            out["nextstep_logits"] = torch.tensor([pinned], dtype=torch.float32)
        return out

    def score_templates(self, query, act_ids, template_features):
        normalized_q = torch.nn.functional.normalize(query, dim=-1)
        normalized_t = torch.nn.functional.normalize(template_features, dim=-1)
        return normalized_q @ normalized_t.t()

    def value_candidate_tokens(self, context_texts, action):
        tokens = []
        for text in context_texts:
            for token in text.split():
                if len(token) > 2 and token not in tokens:
                    tokens.append(token)
        return tokens


class _Index:
    dim = 4
    size = 3

    def max_cosine(self, vectors):
        return [float(min(1.0, abs(float(row[0])))) for row in vectors]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def cfg_lex():
    """The candidate backend pinned explicitly, so a config flip cannot silently
    turn these unit tests into a model download. `lexical` is also the shipped
    default (measured: 51.6% vs 41.6% recall@1 against MiniLM on a near-miss
    probe), and these tests exercise the ranking rules, not the embedder.
    """
    return load_config(CONFIG, ["select.candidate_match_backend=lexical"])


@pytest.fixture(scope="module")
def bank():
    templates = [
        Template(template_id="T000", act="ACK", text_delex="thank you for waiting.", slots=[], count=9),
        Template(template_id="T001", act="ACK", text_delex="no problem at all.", slots=[], count=7),
        Template(template_id="T002", act="ASK", text_delex="what is your account id?", slots=[], count=5),
        Template(
            template_id="T003",
            act="INFORM",
            text_delex="your order {order_id} ships today.",
            slots=["order_id"],
            count=4,
        ),
    ]
    registry = SlotRegistry(
        slots={
            "order_id": SlotSpec(name="order_id", type="id", source="customer_utterance", abcd_marker="<order_id>"),
            "account_id": SlotSpec(
                name="account_id", type="id", source="customer_utterance", abcd_marker="<account_id>"
            ),
            # compile.delex_regex_slots names these two; the registry is the only
            # source of slot names, so a registry without them is rejected.
            "email": SlotSpec(name="email", type="email", source="customer_utterance", abcd_marker="<email>"),
            "phone": SlotSpec(name="phone", type="phone", source="customer_utterance", abcd_marker="<phone>"),
        },
        marker_to_slot={
            "<order_id>": "order_id",
            "<account_id>": "account_id",
            "<email>": "email",
            "<phone>": "phone",
        },
    )
    return Bank(
        templates=templates,
        skeletons=[
            Skeleton(skeleton_id="S000", acts=["ACK"], count=10),
            Skeleton(skeleton_id="S001", acts=["ACK", "ASK"], count=6),
            Skeleton(skeleton_id="S002", acts=["INFORM"], count=3),
        ],
        actions=[
            ActionPattern(action="pull-up-account", required_slots=["customer_name"], count=5),
            ActionPattern(action="verify-identity", required_slots=["customer_name", "account_id"], count=4),
            ActionPattern(action="membership", required_slots=["membership_level"], count=3),
        ],
        slot_registry=registry,
        templates_by_act={"ACK": ["T000", "T001"], "ASK": ["T002"], "INFORM": ["T003"]},
        bank_hash="test",
    )


def _selector(cfg, bank, *, plan=None, active=False, utterances=None):
    """Assemble the private bundle directly -- no checkpoint, no encoder."""
    model = _Model(bank, len(bank.templates))
    orders = model.class_orders
    bundle = S._Selector(
        cfg=cfg,
        model=model,
        tokenizer=model.tokenizer,
        bank=bank,
        ontology={},
        calibration=Calibration(alpha=0.02, novelty_index_path="x", checkpoint_path="y"),
        checkpoint_path="y",
        metadata={},
        novelty_index=_Index(),
        utterances=list(utterances or []),
        next_steps=list(orders["next_steps"]),
        subflows=list(orders["subflows"]),
        actions=list(orders["actions"]),
        skeleton_ids=list(orders["skeleton_ids"]),
        acts=list(orders["acts"]),
        value_list=list(orders["value_list"]),
    )
    bundle.skeleton_acts = {s.skeleton_id: list(s.acts) for s in bank.skeletons}
    bundle.template_id_by_index = [t.template_id for t in bank.templates]
    bundle.template_text_by_id = {t.template_id: t.text_delex for t in bank.templates}
    by_act = {}
    for i, template in enumerate(bank.templates):
        by_act.setdefault(template.act, []).append(i)
    bundle.template_index_by_act = by_act
    bundle.required_slots = {p.action: list(p.required_slots) for p in bank.actions}
    bundle.enumerable = dict(model.enumerable)
    bundle.value_by_action = dict(model.value_by_action)
    bundle.value_index = {v: i for i, v in enumerate(VALUE_LIST)}
    bundle.max_len = 512
    bundle.truncation_side = "left"
    bundle.infonce_temperature = 0.05
    bundle.multi_value_actions = frozenset({"verify-identity"})
    bundle.slot_aliases = {"shipping_status": "shipping_option", "name": "product"}
    bundle.value_slots_source = "bank_then_union"
    bundle.value_restrict_to_action = True
    bundle.compose_join = " "
    bundle.normalizers = ("casefold", "collapse_whitespace")
    bundle.exact_match_similarity = 1.0
    bundle.cache_limit = 1000
    bundle.candidate_backend = "lexical"
    bundle.context_k = "full"
    bundle.plan = plan or S._inherit_plan()
    bundle.requested_plan = bundle.plan
    bundle.head_context_active = active
    return bundle


def _context():
    turns = [
        "agent|hello, how can i help you today?",
        "customer|i need my order status please",
        "action|pull-up-account (crystal minh)",
        "customer|my account id is <account_id>",
    ]
    state = "state|disclosed: order.order_id=3348917502 || actions: pull-up-account(crystal minh)"
    text = "\n".join(turns + [state])
    return ContextWindow(
        convo_id=42,
        turn_index=4,
        text=text,
        turns=turns,
        disclosed={"order.order_id": "3348917502"},
        actions_so_far=[{"action": "pull-up-account", "values": ["crystal minh"]}],
        context_hash="deadbeefdeadbeef",
    )


def _turn(candidates=None):
    return NormalizedTurn(
        convo_id=42,
        turn_index=4,
        speaker="agent",
        text="your order will ship today.",
        nextstep="retrieve_utterance",
        intent="return_size",
        candidates=list(candidates or []),
        utt_rank=1,
        utt_id=(candidates or [0, 0])[1] if candidates else None,
    )


# --------------------------------------------------------------------------- #
# prediction_set (spec 6.6 signal 1)
# --------------------------------------------------------------------------- #


def test_prediction_set_uses_the_contract_inequality():
    probs = [0.90, 0.07, 0.03]
    assert S.prediction_set(probs, 0.10) == [0]      # threshold 0.90
    assert S.prediction_set(probs, 0.95) == [0, 1]   # threshold 0.05 -- 0.03 is out
    assert S.prediction_set(probs, 0.98) == [0, 1, 2]


def test_prediction_set_may_be_empty_and_is_not_patched_with_the_argmax():
    """An empty set is a real low_confidence outcome; inserting argmax breaks coverage."""
    assert S.prediction_set([0.34, 0.33, 0.33], 0.02) == []


# --------------------------------------------------------------------------- #
# Per-head context rendering (D2b / D7)
# --------------------------------------------------------------------------- #


def test_inherited_context_is_byte_identical_to_the_trained_input(cfg, bank):
    bundle = _selector(cfg, bank)
    context = _context()
    assert S._render_variant(context, S._HeadContext(), bundle) == context.text


def test_k6_window_keeps_the_state_line_last_and_drops_the_oldest_turns(cfg, bank):
    bundle = _selector(cfg, bank)
    context = _context()
    rendered = S._render_variant(context, S._HeadContext(k=2), bundle)
    lines = rendered.split("\n")
    assert len(lines) == 3
    assert lines[:2] == context.turns[-2:]
    assert lines[-1].startswith("state|"), "the state line must stay last (D13 left-truncation)"


def test_recency_tagging_is_ordered_and_bucketed(cfg, bank):
    bundle = _selector(cfg, bank)
    context = _context()
    spec = S._HeadContext(k=3, tag_recency=True, recency_buckets=1, recency_mode="word")
    lines = S._render_variant(context, spec, bundle).split("\n")
    assert lines[-2].startswith("customer|r0|"), "the most recent turn is bucket r0"
    assert lines[0].startswith("customer|r1|"), "everything older saturates at MAXB"
    assert "r2|" not in "\n".join(lines), "recency_buckets=1 must emit only r0/r1"


def test_turn_mode_tags_the_line_once(cfg, bank):
    bundle = _selector(cfg, bank)
    spec = S._HeadContext(k=1, tag_recency=True, recency_buckets=3, recency_mode="turn")
    rendered = S._render_variant(_context(), spec, bundle)
    assert rendered.split("\n")[0] == "customer|r0|my account id is <account_id>"


def test_plan_is_read_from_config_not_hard_coded(cfg):
    """D2b/D7's per-head plan must come out of configs/default.yaml."""
    plan = S._resolve_head_plan(cfg)
    assert plan["intent"].k == "full" and not plan["intent"].tag_recency
    assert plan["nextstep"].k == 6 and plan["nextstep"].tag_recency
    assert plan["action"].k == 6 and plan["action"].tag_recency
    assert plan["skeleton"].is_inherited() and plan["template"].is_inherited()


def test_per_head_context_is_inactive_without_a_matching_checkpoint(cfg, bank):
    """The guard that stops a silent train/test mismatch (module docstring)."""
    bundle = _selector(cfg, bank)
    bundle.requested_plan = S._resolve_head_plan(cfg)
    bundle.metadata = {"extra": {}}
    with pytest.warns(RuntimeWarning, match="per-head context"):
        effective = S._effective_plan(bundle, cfg)
    assert all(spec.is_inherited() for spec in effective.values())
    assert bundle.head_context_active is False


def test_per_head_context_activates_when_the_checkpoint_records_the_same_plan(cfg, bank):
    bundle = _selector(cfg, bank)
    bundle.requested_plan = S._resolve_head_plan(cfg)
    bundle.metadata = {"extra": {"head_context": S._plan_fingerprint(bundle.requested_plan)}}
    effective = S._effective_plan(bundle, cfg)
    assert effective["intent"].k == "full"
    assert effective["nextstep"].k == 6
    assert bundle.head_context_active is True


def test_active_plan_runs_one_encoder_pass_per_distinct_context(cfg, bank):
    """Heads with different windows must NOT be collapsed onto one pass."""
    plan = S._resolve_head_plan(cfg)
    bundle = _selector(cfg, bank, plan=plan, active=True)
    scores = S.score_turn(bundle, _context(), _turn(), cfg)
    # nextstep/action/values share ONE k6+recency-tagged string; intent asks for the
    # full thread, which at data.context_turns_K=full renders byte-identically to
    # ContextWindow.text and so shares the pass with skeleton/template/novelty.
    # Two distinct strings -> two encoder passes, and the local heads are NOT
    # collapsed onto the global one.
    assert bundle.model.forward_calls == 2
    assert len(scores.nextstep_probs) == 3
    tagged = S._render_variant(_context(), plan["nextstep"], bundle)
    assert tagged != _context().text and "r0|" in tagged


# --------------------------------------------------------------------------- #
# score_turn
# --------------------------------------------------------------------------- #


def test_score_turn_returns_probabilities_not_logits(cfg, bank):
    bundle = _selector(cfg, bank)
    scores = S.score_turn(bundle, _context(), _turn(), cfg)
    assert abs(sum(scores.nextstep_probs) - 1.0) < 1e-9
    assert abs(sum(scores.intent_probs) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in scores.nextstep_probs + scores.intent_probs)
    assert len(scores.intent_probs) == len(SUBFLOWS)


def test_score_turn_is_deterministic_for_the_same_context(cfg, bank):
    """Spec 10: same context -> same scores, always."""
    bundle = _selector(cfg, bank)
    context = _context()
    first = S.score_turn(bundle, context, _turn(), cfg)
    second = S.score_turn(bundle, context, _turn(), cfg)
    assert first.to_dict() == second.to_dict()
    assert first.context_hash == context.context_hash


def test_score_turn_branches_on_the_predicted_nextstep_not_the_gold_one(cfg, bank):
    """Reading turn.nextstep would be leakage; the branch must follow H1's argmax."""
    bundle = _selector(cfg, bank)
    context = _context()
    scores = S.score_turn(bundle, context, _turn(), cfg)
    predicted = S.NEXT_STEPS[S._argmax(scores.nextstep_probs)]
    assert bool(scores.action_probs) == (predicted == "take_action")
    assert bool(scores.skeleton_probs) == (predicted == "retrieve_utterance")

    gold_flipped = NormalizedTurn(
        convo_id=42, turn_index=4, speaker="action", text="", nextstep="take_action",
        action="membership", values=["gold"], candidates=[],
    )
    same = S.score_turn(bundle, context, gold_flipped, cfg)
    assert same.nextstep_probs == scores.nextstep_probs
    assert same.action_probs == scores.action_probs, "gold nextstep must not change the scores"


def test_novelty_distance_is_one_minus_max_cosine(cfg, bank):
    bundle = _selector(cfg, bank)
    scores = S.score_turn(bundle, _context(), _turn(), cfg)
    assert 0.0 <= scores.novelty_distance <= 2.0
    assert scores.novelty_distance == pytest.approx(1.0 - abs(math.sin(_seed(bundle))), abs=1e-6)


def _seed(bundle):
    encoded = bundle.tokenizer([_context().text], max_length=bundle.max_len)
    return float(encoded["input_ids"].to(torch.float32).sum().item())


def test_template_candidates_cover_the_skeletons_act_positions(cfg, bank):
    bundle = _selector(cfg, bank)
    bundle.score_inapplicable_heads = True
    scores = S.score_turn(bundle, _context(), _turn(), cfg)
    skeleton = bundle.skeleton_ids[S._argmax(scores.skeleton_probs)]
    acts = bundle.skeleton_acts[skeleton]
    assert len(scores.template_probs) == len(acts)
    for position, act in enumerate(acts):
        ids = scores.template_candidates[position]
        assert ids == [bundle.template_id_by_index[i] for i in bundle.template_index_by_act[act]]
        assert abs(sum(scores.template_probs[position]) - 1.0) < 1e-9


def test_value_candidates_are_restricted_per_slot_and_renormalized(cfg, bank):
    bundle = _selector(cfg, bank)
    probs, candidates = S._score_values(bundle, _context(), _context().text, "membership")
    assert candidates == [["guest", "bronze", "silver", "gold"]]
    assert abs(sum(probs[0]) - 1.0) < 1e-9


def test_value_slots_give_the_buttons_arity(cfg, bank):
    bundle = _selector(cfg, bank)
    _, candidates = S._score_values(bundle, _context(), _context().text, "verify-identity")
    assert len(candidates) == 2, "verify-identity keeps two supported slots in this bank"


def test_value_slots_bank_then_union_no_longer_drops_ontology_only_slots(cfg, bank):
    """Regression for the reviewer-confirmed bug: bank_then_union returned
    bank_slots ALONE whenever non-empty, silently dropping any argument the
    filler could not source but the ontology still lists (e.g. `username` for
    `validate-purchase` in the real ontology; `order_id` for `verify-identity`
    in this test bank, whose ActionPattern only records
    `[customer_name, account_id]`). The fix unions bank slots (kept first,
    since the filler-availability signal was tuned against that order) with
    the full ontology slot list.
    """
    bundle = _selector(cfg, bank)
    bundle.ontology = {
        "actions": {
            "account": {
                "verify-identity": ["customer_name", "account_id", "order_id"],
            }
        }
    }
    assert bundle.required_slots["verify-identity"] == ["customer_name", "account_id"], (
        "bank-derived required_slots must stay untouched -- it still drives "
        "the availability/unavailable_slot gate signal, D-4"
    )
    slots = S._value_slots(bundle, "verify-identity")
    assert slots == ["customer_name", "account_id", "order_id"], (
        "bank_then_union must now union with the ontology, bank order first, "
        "not return bank_slots alone"
    )


def test_value_slots_bank_then_union_falls_back_to_bank_when_ontology_is_empty(cfg, bank):
    """No ontology (e.g. the test bundles elsewhere that pass ontology={})
    must behave exactly as before: bank_slots, unchanged."""
    bundle = _selector(cfg, bank)
    assert bundle.ontology == {}
    slots = S._value_slots(bundle, "verify-identity")
    assert slots == ["customer_name", "account_id"]


def test_score_values_now_attempts_a_slot_bank_alone_used_to_drop(cfg, bank):
    """End-to-end: with the ontology wired, H4 now attempts order_id (via its
    <order_id> copy marker) for verify-identity, which bank_slots alone
    (customer_name, account_id) never asked for -- the actual behavioral
    change the fix is meant to produce, not just a list-equality check."""
    bundle = _selector(cfg, bank)
    bundle.ontology = {
        "actions": {"account": {"verify-identity": ["customer_name", "account_id", "order_id"]}}
    }
    turns = [
        "agent|hello, how can i help you today?",
        "customer|i need my order status please",
        "action|pull-up-account (crystal minh)",
        "customer|my account id is <account_id>",
        "customer|my order id is <order_id>",
    ]
    state = "state|disclosed: order.order_id=3348917502 || actions: pull-up-account(crystal minh)"
    context = ContextWindow(
        convo_id=42, turn_index=4, text="\n".join(turns + [state]), turns=turns,
        disclosed={"order.order_id": "3348917502"},
        actions_so_far=[{"action": "pull-up-account", "values": ["crystal minh"]}],
        context_hash="deadbeefdeadbeef",
    )
    _, candidates = S._score_values(bundle, context, context.text, "verify-identity")
    assert len(candidates) == 3, (
        "order_id's <order_id> copy marker is present in context -- H4 must now "
        "attempt it, where bank_slots alone silently never would have"
    )


# --------------------------------------------------------------------------- #
# select_from_scores
# --------------------------------------------------------------------------- #


def test_select_from_scores_composes_templates_and_maps_to_a_candidate(cfg_lex, bank):
    utterances = [
        "thank you for waiting. what is your account id?",  # id 0: the exact composition
        "hello there",
        "your order 3348917502 ships today.",
    ]
    bundle = _selector(cfg_lex, bank, utterances=utterances)
    bundle.model.force_nextstep = 0  # pin H1 to retrieve_utterance
    context = _context()
    scores = S.score_turn(bundle, context, _turn(), cfg_lex)
    assert scores.skeleton_probs, "H5 must be populated when retrieve_utterance is predicted"
    selection = S.select_from_scores(bundle, scores, _turn([2, 0, 1]), cfg_lex)
    assert selection.nextstep == "retrieve_utterance"
    assert selection.composed_text_delex is not None
    assert selection.candidate_rank >= 0
    assert selection.candidate_utt_id == [2, 0, 1][selection.candidate_rank]
    assert selection.skeleton_id is not None and selection.template_ids
    assert selection.action is None and selection.values is None
    assert selection.exact_template_match in (True, False)
    composed_from_ids = " ".join(bundle.template_text_by_id[t] for t in selection.template_ids)
    assert selection.composed_text_delex == composed_from_ids


def test_selection_reports_both_rank_and_id_and_the_rank_indexes_candidates(cfg_lex, bank):
    candidates = [11, 22, 33]
    texts = ["nope", "thank you for waiting.", "also nope"]
    utt_id, rank, similarity = S.map_to_candidate("thank you for waiting.", candidates, texts, None, cfg_lex)
    assert (utt_id, rank) == (22, 1), "rank indexes candidates; the id is candidates[rank]"
    assert similarity == 1.0


def test_map_to_candidate_prefers_an_exact_match_over_a_higher_cosine(cfg_lex, bank):
    candidates = [5, 6]
    texts = ["thank you for waiting, sir, and thank you again.", "thank you for waiting."]
    utt_id, rank, _ = S.map_to_candidate("thank you for waiting.", candidates, texts, None, cfg_lex)
    assert (utt_id, rank) == (6, 1)


def test_map_to_candidate_breaks_ties_toward_the_lowest_rank(cfg_lex, bank):
    candidates = [7, 8, 9]
    texts = ["thank you for waiting.", "thank you for waiting.", "unrelated"]
    assert S.map_to_candidate("thank you for waiting.", candidates, texts, None, cfg_lex)[1] == 0


def test_map_to_candidate_with_no_candidates_returns_the_contract_sentinel(cfg_lex):
    assert S.map_to_candidate("anything", [], [], None, cfg_lex) == (None, -1, 0.0)


def test_map_to_candidate_refuses_to_guess_when_nothing_was_composed(cfg_lex):
    """An empty composition must not silently map to rank 0."""
    assert S.map_to_candidate("", [1, 2], ["a", "b"], None, cfg_lex) == (None, -1, 0.0)


def test_map_to_candidate_is_deterministic(cfg_lex):
    candidates = [1, 2, 3]
    texts = ["what is your account id?", "thanks for waiting", "no problem"]
    first = S.map_to_candidate("thank you for waiting.", candidates, texts, None, cfg_lex)
    second = S.map_to_candidate("thank you for waiting.", candidates, texts, None, cfg_lex)
    assert first == second


# --------------------------------------------------------------------------- #
# delexicalize_candidates
# --------------------------------------------------------------------------- #


def test_delexicalize_candidates_converts_abcd_markers(cfg, bank):
    utterances = ["your order <order_id> is on its way.", "hello!"]
    out = S.delexicalize_candidates([0, 1], utterances, bank.slot_registry, cfg)
    assert out[0] == "your order {order_id} is on its way."
    assert "<" not in out[0]
    assert out[1] == "hello!"


def test_delexicalize_candidates_is_aligned_and_cached(cfg, bank):
    utterances = ["a <order_id> b", "c", "d"]
    first = S.delexicalize_candidates([2, 0], utterances, bank.slot_registry, cfg)
    second = S.delexicalize_candidates([2, 0], utterances, bank.slot_registry, cfg)
    assert first == second == ["d", "a {order_id} b"]


def test_delexicalize_candidates_rejects_an_out_of_range_id(cfg, bank):
    with pytest.raises(Exception):
        S.delexicalize_candidates([99], ["only one"], bank.slot_registry, cfg)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_a_foreign_selector_is_refused(cfg):
    with pytest.raises(Exception):
        S.score_turn(object(), _context(), _turn(), cfg)


def test_select_from_scores_take_action_branch_fills_action_and_values(cfg_lex, bank):
    bundle = _selector(cfg_lex, bank)
    bundle.model.force_nextstep = 1  # pin H1 to take_action
    scores = S.score_turn(bundle, _context(), _turn(), cfg_lex)
    assert scores.action_probs and not scores.skeleton_probs
    selection = S.select_from_scores(bundle, scores, _turn(), cfg_lex)
    assert selection.nextstep == "take_action"
    assert selection.action in ACTIONS
    assert selection.values is not None and len(selection.values) == len(scores.value_probs)
    assert selection.skeleton_id is None and selection.template_ids is None
    assert selection.candidate_rank == -1 and selection.candidate_utt_id is None


def test_select_from_scores_end_conversation_branch_is_nextstep_only(cfg_lex, bank):
    bundle = _selector(cfg_lex, bank)
    bundle.model.force_nextstep = 2  # pin H1 to end_conversation
    scores = S.score_turn(bundle, _context(), _turn(), cfg_lex)
    selection = S.select_from_scores(bundle, scores, _turn(), cfg_lex)
    assert selection.nextstep == "end_conversation"
    assert selection.action is None and selection.values is None
    assert selection.skeleton_id is None and selection.template_ids is None
    assert selection.composed_text_delex is None
    assert selection.exact_template_match is None
