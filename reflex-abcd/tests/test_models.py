"""Spec 6.4 -- unit tests for the encoder and heads H1-H7.

Everything here runs on a TINY randomly-initialized BERT built into a temp
directory, so the suite is offline, deterministic and fast; no training happens
and no real checkpoint is downloaded. One test (marked) touches the real
``model.encoder`` from ``configs/default.yaml`` and skips if the HF cache does
not have it.

What is pinned:
  * head widths come from the data, and H1's order is the normative one;
  * H4's index space is the official AST one (``len(value_list) + 100``) and
    ``value_target_index`` reproduces ``utils/process.py::value_to_id``;
  * H3/H4 are masked to take_action turns and H5/H7 to retrieve_utterance ones;
  * H7 is act-conditioned cosine and its template features carry no gradient;
  * checkpoints round-trip and REFUSE to load against a different class order.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from reflex.config import load_config  # noqa: E402
from reflex.contracts import ContractViolation  # noqa: E402
from reflex.models import build_encoder, build_model, load_checkpoint, save_checkpoint  # noqa: E402
from reflex.schemas import ACT_INVENTORY, NEXT_STEPS, ActionPattern, Bank, Skeleton, Template  # noqa: E402

# --------------------------------------------------------------------------- #
# Tiny fixtures
# --------------------------------------------------------------------------- #

_VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "hi", "hello", "there", "the", "your", "order", "will", "arrive", "please",
    "name", "account", "thanks", "bye", "sure", "may", "have", "ship", "size",
]


@pytest.fixture(scope="session")
def tiny_encoder_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A 2-layer, 32-dim randomly-initialized BERT saved to disk. No network.

    The tokenizer is built from an explicit ``tokenizers`` WordPiece model rather
    than ``BertTokenizerFast(vocab_file=...)``: on transformers 5.x the latter
    silently ignores ``vocab_file`` and yields a 5-token special-tokens-only
    vocabulary, which turns every test string into ``[UNK]`` and makes the
    ``<slot>``-marker and copy-position tests vacuous.
    """
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    directory = tmp_path_factory.mktemp("tiny_encoder")
    vocab_path = directory / "vocab.txt"
    vocab_path.write_text("\n".join(_VOCAB) + "\n", encoding="utf-8")

    config = transformers.BertConfig(
        vocab_size=len(_VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=2,
    )
    transformers.BertModel(config).save_pretrained(directory)

    backend = tokenizers.Tokenizer(
        tokenizers.models.WordPiece(
            vocab={word: idx for idx, word in enumerate(_VOCAB)}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    ).save_pretrained(directory)
    return str(directory)


@pytest.fixture()
def cfg(tiny_encoder_dir: str) -> dict:
    """The real config, pointed at the tiny encoder and a temp checkpoint dir."""
    return load_config(
        overrides=[
            f"model.encoder={tiny_encoder_dir}",
            f"model.small_encoder={tiny_encoder_dir}",
            "model.query_mlp=[16, 8]",
            "model.value_context_len=12",
            "model.template_max_len=8",
            "data.max_len=16",
        ]
    )


#: ``data.subflow_list`` refuses any ontology that does not flatten to exactly 55
#: subflows ("the ontology file is not ABCD v1.1"), and iterates
#: ``intents["flows"]`` rather than the ``subflows`` dict. A fixture ontology
#: therefore has to carry both keys and the real count; only the NAMES are
#: miniature. The three real ones come first so the value/action fixtures below
#: still read as ABCD.
_SUBFLOWS = {
    "shipping": ["status_delivery", "manage_change"],
    "returns": ["return_size"],
    "synthetic": [f"subflow_{i:02d}" for i in range(52)],
}


@pytest.fixture()
def ontology() -> dict:
    """A miniature ABCD ontology with the same SHAPE as the real one."""
    return {
        "intents": {"flows": list(_SUBFLOWS), "subflows": {k: list(v) for k, v in _SUBFLOWS.items()}},
        "actions": {
            "kb_query": {"verify-identity": ["account_id", "name"], "validate-purchase": ["order_id"]},
            "api_call": {"update-order": ["change_option"], "log-out-in": []},
        },
        "values": {
            "enumerable": {
                "change_option": ["Shipping Option", "Payment Method"],
                "name": ["Alessandro Phoenix"],
            },
            "non_enumerable": {"personal": ["account_id"], "order": ["order_id"]},
        },
        "next_steps": list(NEXT_STEPS),
    }


@pytest.fixture()
def bank() -> Bank:
    templates = [
        Template("T000001", "ACK", "Thanks for waiting.", [], 9),
        Template("T000002", "ASK", "May I have your {name}?", ["name"], 7),
        Template("T000003", "INFORM", "Your order {order_id} ships today.", ["order_id"], 4),
        Template("T000004", "ASK", "May I have your {account_id}?", ["account_id"], 3),
    ]
    return Bank(
        templates=templates,
        skeletons=[
            Skeleton("S0001", ["ACK", "ASK"], 12),
            Skeleton("S0002", ["INFORM"], 5),
            Skeleton("S0003", ["ACK", "INFORM", "CLOSE"], 2),
        ],
        actions=[ActionPattern("verify-identity", ["account_id"], 6)],
        templates_by_act={"ACK": ["T000001"], "ASK": ["T000002", "T000004"], "INFORM": ["T000003"]},
        source_fraction=1.0,
        bank_hash="deadbeefdeadbeef",
    )


@pytest.fixture()
def model(cfg: dict, bank: Bank, ontology: dict):
    torch.manual_seed(0)
    return build_model(cfg, bank, ontology, size="base")


def _batch(model, n: int = 4, seed: int = 0) -> dict:
    """Tiny random tensors: 2 retrieve_utterance rows then 2 take_action rows."""
    generator = torch.Generator().manual_seed(seed)
    vocab = len(model.tokenizer)
    retrieve = NEXT_STEPS.index("retrieve_utterance")
    take_action = NEXT_STEPS.index("take_action")
    half = n // 2
    nextstep = torch.tensor([retrieve] * half + [take_action] * (n - half))
    return {
        "input_ids": torch.randint(0, vocab, (n, 7), generator=generator),
        "attention_mask": torch.ones(n, 7, dtype=torch.long),
        "token_type_ids": torch.zeros(n, 7, dtype=torch.long),
        "context_input_ids": torch.randint(0, vocab, (n, model.value_context_len), generator=generator),
        "context_attention_mask": torch.ones(n, model.value_context_len, dtype=torch.long),
        "nextstep_labels": nextstep,
        "intent_labels": torch.zeros(n, dtype=torch.long),
        "act_labels": torch.zeros(n, dtype=torch.long),
        "action_labels": torch.tensor([-1] * half + [1] * (n - half)),
        "value_labels": torch.tensor([-1] * half + [0] * (n - half)),
        "skeleton_labels": torch.tensor([0] * half + [-1] * (n - half)),
        "act_ids": torch.zeros(n, dtype=torch.long),
        "template_features": torch.randn(n, 3, model.hidden_size, generator=generator),
        "template_gold_index": torch.arange(n),
        "template_mask": nextstep == retrieve,
    }


# --------------------------------------------------------------------------- #
# build_encoder
# --------------------------------------------------------------------------- #


def test_build_encoder_returns_id_and_tokenizer(cfg: dict, tiny_encoder_dir: str) -> None:
    encoder, tokenizer, resolved = build_encoder(cfg, "base")
    assert resolved == tiny_encoder_dir
    assert encoder.config.hidden_size == 32
    assert tokenizer.tokenize("hello there") == ["hello", "there"]


def test_build_encoder_rejects_unknown_size(cfg: dict) -> None:
    with pytest.raises(ValueError, match="unknown encoder size"):
        build_encoder(cfg, "gigantic")


def test_build_encoder_falls_back_when_the_first_id_will_not_load(cfg: dict, tiny_encoder_dir: str) -> None:
    """A silent swap would make two runs incomparable, so the id is returned.

    Both ids are local so the test is hermetic: pointing ``fallback_encoder`` at
    the real ``microsoft/deberta-v3-base`` would make this test a network probe.
    """
    broken = dict(cfg)
    broken["model"] = dict(
        cfg["model"],
        encoder="reflex-test/does-not-exist",
        fallback_encoder=tiny_encoder_dir,
        local_files_only=True,
    )
    _encoder, _tokenizer, resolved = build_encoder(broken, "base")
    assert resolved == tiny_encoder_dir, "the fallback that fired must be reported, not hidden"


def test_build_encoder_reports_every_id_it_tried_when_none_load(cfg: dict) -> None:
    """The error names each candidate and why it failed, so a bad id is diagnosable."""
    from reflex.contracts import ReflexError

    broken = dict(cfg)
    broken["model"] = dict(
        cfg["model"],
        encoder="reflex-test/does-not-exist",
        fallback_encoder="reflex-test/also-missing",
        local_files_only=True,
    )
    with pytest.raises(ReflexError) as excinfo:
        build_encoder(broken, "base")
    message = str(excinfo.value)
    assert "reflex-test/does-not-exist" in message
    assert "reflex-test/also-missing" in message


def test_configured_encoder_is_loadable_in_this_environment() -> None:
    """The real ``model.encoder``. Skips when the HF cache does not have it."""
    real_cfg = load_config()
    try:
        _encoder, _tokenizer, resolved = build_encoder(real_cfg, "base")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no encoder loadable here: {exc}")
    assert resolved in (real_cfg["model"]["encoder"], real_cfg["model"]["fallback_encoder"])


def test_real_encoder_runs_every_head(bank: Bank) -> None:
    """End-to-end forward on the CONFIGURED encoder, not the tiny stand-in.

    The tiny fixture is a classic BERT; the configured default is ModernBERT,
    which has no ``token_type_ids`` and no pooler. This is the test that would
    catch ``d`` being hard-coded, the segment-id path crashing, or the real
    tokenizer refusing the ``<slot>`` markers.
    """
    real_cfg = load_config()
    real_ontology_path = os.path.join(real_cfg["eval"]["official_utils_dir"], "data", "ontology.json")
    if not os.path.exists(real_ontology_path):  # pragma: no cover - environment dependent
        pytest.skip("real ABCD ontology not present")
    import json

    with open(real_ontology_path, encoding="utf-8") as handle:
        real_ontology = json.load(handle)
    try:
        model = build_model(real_cfg, bank, real_ontology)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no encoder loadable here: {exc}")

    # Widths come from the real data: 3 / 55 / 30 / 126+100 / |skeletons| / 9.
    assert model.h1_nextstep.out_features == 3
    assert model.h2_intent.out_features == 55
    assert model.h3_action.out_features == 30
    assert model.value_head_size == 126 + real_cfg["model"]["value_context_len"]
    assert model.h5_skeleton.out_features == len(bank.skeletons)
    assert model.h6_act.out_features == len(ACT_INVENTORY)
    assert model.query_dim == real_cfg["model"]["query_mlp"][-1]
    assert model.hidden_size == model.encoder.config.hidden_size
    assert model.slot_marker_tokens and all(
        model.tokenizer.tokenize(marker) == [marker] for marker in model.slot_marker_tokens
    )

    encoded = model.tokenizer(
        ["customer | where is my order <order_id>?", "agent | let me check that for you"],
        padding=True,
        truncation=True,
        max_length=real_cfg["data"]["max_len"],
        return_tensors="pt",
    )
    batch = dict(encoded)
    # ModernBERT has no segment embeddings: this must warn, not crash.
    batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out["query"].shape == (2, model.query_dim)
    assert out["value_logits"].shape == (2, model.value_head_size)
    for key, tensor in out.items():
        assert torch.isfinite(tensor).all(), key

    features = model.refresh_template_cache()
    assert features.shape == (len(bank.templates), model.hidden_size)
    scores = model.score_templates(out["query"], torch.zeros(2, dtype=torch.long), features)
    assert scores.shape == (2, len(bank.templates))
    assert scores.abs().max() <= 1.0 + 1e-5


# --------------------------------------------------------------------------- #
# Head widths and class orders
# --------------------------------------------------------------------------- #


def test_head_widths_come_from_the_data(model, bank: Bank, ontology: dict) -> None:
    assert model.h1_nextstep.out_features == len(NEXT_STEPS) == 3
    assert model.h2_intent.out_features == 55  # data.subflow_list's fixed width
    assert model.h3_action.out_features == 4  # flattened buttons
    assert model.h5_skeleton.out_features == len(bank.skeletons)
    assert model.h6_act.out_features == len(ACT_INVENTORY) == 9
    assert model.h7_act_projection.shape == (9, model.query_dim, model.query_dim)


def test_nextstep_order_is_normative_and_enforced(cfg: dict, bank: Bank, ontology: dict) -> None:
    """H1's order is refused if it ever stops being cds_report's 0/1/2.

    ``reflex.data.nextstep_list`` owns the check and raises first;
    ``models`` keeps its own guard for the case where the ontology reaches it by
    another route. Either message is acceptable -- the refusal is the contract.
    """
    assert model_next_steps(cfg, bank, ontology) == list(NEXT_STEPS)
    scrambled = dict(ontology, next_steps=["take_action", "retrieve_utterance", "end_conversation"])
    with pytest.raises(ContractViolation, match="normative order|branches on the order"):
        build_model(cfg, bank, scrambled)


def model_next_steps(cfg: dict, bank: Bank, ontology: dict) -> list[str]:
    return build_model(cfg, bank, ontology).next_steps


def test_query_mlp_is_d_512_256_gelu(cfg: dict, bank: Bank, ontology: dict) -> None:
    """Widths are read from ``model.query_mlp``; the default really is d->512->256."""
    default_widths = load_config()["model"]["query_mlp"]
    assert default_widths == [512, 256]
    built = build_model(cfg, bank, ontology)  # overridden to [16, 8] to stay tiny
    kinds = [type(layer).__name__ for layer in built.query_mlp]
    assert kinds == ["Linear", "GELU", "Dropout", "Linear"]
    assert built.query_mlp[0].in_features == built.hidden_size
    assert built.query_mlp[0].out_features == 16
    assert built.query_mlp[-1].out_features == built.query_dim == 8


def test_empty_bank_is_refused(cfg: dict, ontology: dict) -> None:
    with pytest.raises(ContractViolation, match="bank.skeletons is empty"):
        build_model(cfg, Bank(), ontology)


def test_slot_markers_are_single_tokens(model) -> None:
    """utils/load.py adds them; without this H4 can never hit a copy target."""
    assert model.slot_marker_tokens == ["<account_id>", "<order_id>"]
    for marker in model.slot_marker_tokens:
        assert model.tokenizer.tokenize(marker) == [marker]
    assert model.encoder.get_input_embeddings().weight.shape[0] >= len(model.tokenizer)


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #


def test_forward_shapes_with_tiny_tensors(model) -> None:
    batch = _batch(model)
    out = model(batch)
    n = batch["input_ids"].shape[0]
    assert out["query"].shape == (n, model.query_dim)
    assert out["nextstep_logits"].shape == (n, 3)
    assert out["intent_logits"].shape == (n, 55)
    assert out["action_logits"].shape == (n, 4)
    assert out["skeleton_logits"].shape == (n, 3)
    assert out["act_logits"].shape == (n, 9)
    assert out["value_logits"].shape == (n, model.value_head_size)
    assert out["template_cosine"].shape == (n, 3)
    assert torch.isfinite(out["value_logits"]).all()
    for key, tensor in out.items():
        assert torch.isfinite(tensor).all(), key


def test_value_head_uses_the_official_ast_index_space(model) -> None:
    """``len(value_list) + model.value_context_len`` -- utils/process.py's layout."""
    assert model.value_list == ["shipping option", "payment method", "alessandro phoenix"]
    assert model.value_head_size == len(model.value_list) + model.value_context_len
    assert model.h4_value_enumerable.out_features == len(model.value_list)


def test_value_logits_mask_padded_copy_positions(model) -> None:
    batch = _batch(model)
    batch["context_attention_mask"][:, 5:] = 0
    logits = model(batch)["value_logits"]
    copy_part = logits[:, len(model.value_list) :]
    assert (copy_part[:, 5:] < -1e8).all()
    assert (copy_part[:, :5] > -1e8).all()


def test_value_copy_logits_are_all_masked_without_a_context_input(model) -> None:
    batch = _batch(model)
    batch.pop("context_input_ids")
    batch.pop("context_attention_mask")
    logits = model(batch)["value_logits"]
    assert (logits[:, len(model.value_list) :] < -1e8).all()
    assert torch.isfinite(logits).all()


def test_value_target_index_matches_official_value_to_id(model) -> None:
    """Enumerable hit, copy hit, and the -1 miss, as ``value_to_id`` returns them."""
    context = ["your order <order_id> will ship", "thanks"]
    target, tokens = model.value_target_index(context, "update-order", "shipping option")
    assert target == model.value_list.index("shipping option")

    target, tokens = model.value_target_index(context, "validate-purchase", "anything")
    assert "<order_id>" in tokens
    assert target == len(model.value_list) + tokens.index("<order_id>")

    target, _tokens = model.value_target_index(["hello there"], "validate-purchase", "anything")
    assert target == -1, "no <order_id> in context -> the official code yields -1"

    target, _tokens = model.value_target_index(context, "log-out-in", "anything")
    assert target == -1, "an action with no value slots can never produce a target"


def test_context_tokens_are_truncated_the_official_way(model) -> None:
    long_context = [" ".join(_VOCAB[5:]) for _ in range(5)]
    tokens = model.value_candidate_tokens(long_context, "verify-identity")
    budget = model.value_context_len - (len(model.tokenizer.tokenize("verify-identity")) + 3)
    assert len(tokens) <= budget
    assert len(tokens) == len(set(tokens)), "value_to_id de-duplicates, preserving order"
    assert all(len(token) > 2 for token in tokens)


def test_forward_is_deterministic_in_eval_mode(model) -> None:
    model.eval()
    batch = _batch(model)
    with torch.no_grad():
        first = model(batch)
        second = model(batch)
    for key in first:
        assert torch.equal(first[key], second[key]), key


# --------------------------------------------------------------------------- #
# H7
# --------------------------------------------------------------------------- #


def test_score_templates_is_cosine_and_act_conditioned(model) -> None:
    torch.manual_seed(1)
    query = torch.randn(3, model.query_dim)
    features = torch.randn(3, 5, model.hidden_size)
    acts_a = torch.zeros(3, dtype=torch.long)
    acts_b = torch.full((3,), 2, dtype=torch.long)

    scores = model.score_templates(query, acts_a, features)
    assert scores.shape == (3, 5)
    assert scores.abs().max() <= 1.0 + 1e-5, "cosine must stay in [-1, 1]"

    # Identity init => act-agnostic at t=0; that is deliberate, not a bug.
    assert torch.allclose(scores, model.score_templates(query, acts_b, features), atol=1e-6)
    with torch.no_grad():
        model.h7_act_projection[2].mul_(0.0).add_(torch.randn(model.query_dim, model.query_dim))
    assert not torch.allclose(scores, model.score_templates(query, acts_b, features), atol=1e-4)


def test_score_templates_supports_a_shared_pool_and_act_positions(model) -> None:
    query = torch.randn(2, model.query_dim)
    pool = torch.randn(7, model.hidden_size)
    assert model.score_templates(query, torch.zeros(2, dtype=torch.long), pool).shape == (2, 7)
    positions = torch.zeros(2, 3, dtype=torch.long)
    per_position = torch.randn(2, 3, 4, model.hidden_size)
    assert model.score_templates(query, positions, per_position).shape == (2, 3, 4)


def test_template_features_come_from_the_frozen_encoder(model) -> None:
    features = model.refresh_template_cache(batch_size=2)
    assert features.shape == (len(model.template_ids), model.hidden_size)
    assert features.requires_grad is False
    assert torch.equal(model.template_features, features)

    # The frozen twin only moves when explicitly synced.
    with torch.no_grad():
        for param in model.encoder.parameters():
            param.add_(0.5)
    assert torch.allclose(model.refresh_template_cache(batch_size=2), features)
    model.sync_frozen_encoder()
    assert not torch.allclose(model.refresh_template_cache(batch_size=2), features)


def test_frozen_encoder_never_receives_gradients(model) -> None:
    batch = _batch(model)
    model.compute_losses(batch)["total"].backward()
    assert all(param.grad is None for param in model.frozen_encoder.parameters())
    assert model.h7_act_projection.grad is not None
    assert model.h7_template_projection.weight.grad is not None


def test_every_head_shares_one_encoder_pass_over_the_context(model) -> None:
    """H7 must read the query :meth:`forward` produced, not re-encode the context.

    Re-encoding costs a third full encoder pass per training step and, because
    dropout is live in training, trains H7 against a query sample H1-H6 never
    saw. Exactly two live-encoder passes are expected: the context, and the
    official AST copy-context input H4 points into.
    """
    calls: list[int] = []
    original = model.encoder.forward

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    model.encoder.forward = counting
    try:
        model.train()
        model.compute_losses(_batch(model))
        assert sum(calls) == 2, f"{sum(calls)} live-encoder passes per step; expected 2"

        calls.clear()
        outputs = model(_batch(model))
        before = sum(calls)
        model.compute_losses(_batch(model), outputs)
        assert sum(calls) == before, "passing `outputs` must not trigger another encoder pass"
    finally:
        model.encoder.forward = original


def test_h7_scores_the_same_query_the_other_heads_saw(model) -> None:
    """With dropout live, a second encode would hand H7 a different query."""
    model.train()
    torch.manual_seed(7)
    batch = _batch(model)
    outputs = model(batch)

    # Dropout makes a re-encode observably different, so "same query" is a real
    # claim here rather than a coincidence of eval mode.
    assert not torch.allclose(model(batch)["query"], outputs["query"])

    seen: list[torch.Tensor] = []
    original = model.score_templates

    def capturing(query, act_ids, template_features):
        seen.append(query)
        return original(query, act_ids, template_features)

    model.score_templates = capturing
    try:
        model.compute_losses(batch, outputs)
    finally:
        model.score_templates = original

    assert seen, "H7 never scored anything"
    for query in seen:
        assert query is outputs["query"]


# --------------------------------------------------------------------------- #
# Masked losses (spec 6.4)
# --------------------------------------------------------------------------- #


def test_losses_are_masked_to_the_applicable_turns(model) -> None:
    batch = _batch(model)
    losses = model.compute_losses(batch)
    assert set(losses) == {
        "nextstep", "intent", "action", "values", "skeleton", "template", "act", "total"
    }
    for value in losses.values():
        assert torch.isfinite(value), losses

    only_retrieve = _batch(model)
    only_retrieve["action_labels"] = torch.full_like(only_retrieve["action_labels"], -1)
    only_retrieve["value_labels"] = torch.full_like(only_retrieve["value_labels"], -1)
    only_retrieve["nextstep_labels"] = torch.full_like(
        only_retrieve["nextstep_labels"], NEXT_STEPS.index("retrieve_utterance")
    )
    only_retrieve["skeleton_labels"] = torch.zeros_like(only_retrieve["skeleton_labels"])
    only_retrieve["template_mask"] = torch.ones_like(only_retrieve["template_mask"])
    masked = model.compute_losses(only_retrieve)
    assert masked["action"].item() == 0.0
    assert masked["values"].item() == 0.0
    assert masked["skeleton"].item() > 0.0
    assert masked["template"].item() > 0.0


def test_template_loss_is_zero_when_no_row_is_retrieve_utterance(model) -> None:
    batch = _batch(model)
    batch["nextstep_labels"] = torch.full_like(
        batch["nextstep_labels"], NEXT_STEPS.index("take_action")
    )
    batch["skeleton_labels"] = torch.full_like(batch["skeleton_labels"], -1)
    batch["action_labels"] = torch.zeros_like(batch["action_labels"])
    batch["value_labels"] = torch.zeros_like(batch["value_labels"])
    batch["template_mask"] = torch.zeros_like(batch["template_mask"])
    losses = model.compute_losses(batch)
    assert losses["template"].item() == 0.0
    assert losses["skeleton"].item() == 0.0
    assert losses["action"].item() > 0.0


def test_a_head_label_on_the_wrong_turn_type_is_refused(model) -> None:
    batch = _batch(model)
    batch["action_labels"] = torch.zeros_like(batch["action_labels"])  # also on agent turns
    with pytest.raises(ContractViolation, match="gold action label"):
        model.compute_losses(batch)


def test_loss_weights_come_from_config(model, cfg: dict) -> None:
    assert model.loss_weights == {k: float(v) for k, v in cfg["train"]["loss_weights"].items()}
    batch = _batch(model)
    losses = model.compute_losses(batch)
    expected = sum(
        losses[name] * model.loss_weights[name]
        for name in ("nextstep", "intent", "action", "values", "skeleton", "template", "act")
    )
    assert torch.allclose(losses["total"], expected)


def test_missing_loss_weight_is_refused(cfg: dict, bank: Bank, ontology: dict) -> None:
    trimmed = dict(cfg)
    weights = dict(cfg["train"]["loss_weights"])
    weights.pop("template")
    trimmed["train"] = dict(cfg["train"], loss_weights=weights)
    built = build_model(trimmed, bank, ontology)
    with pytest.raises(ContractViolation, match="no weight for head"):
        built.compute_losses(_batch(built))


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def test_checkpoint_round_trips(cfg: dict, bank: Bank, ontology: dict, tmp_path) -> None:
    cfg = dict(cfg)
    cfg["train"] = dict(cfg["train"], checkpoint_dir=str(tmp_path / "ckpts"))
    torch.manual_seed(3)
    original = build_model(cfg, bank, ontology)
    path = save_checkpoint(original, cfg, seed=7, extra={"epochs_run": 2})

    assert os.path.basename(path) == "seed7.pt"
    loaded, metadata = load_checkpoint(path, cfg, bank, ontology)
    assert metadata["seed"] == 7
    assert metadata["extra"] == {"epochs_run": 2}
    assert metadata["resolved_encoder"] == original.resolved_encoder
    assert metadata["skeleton_ids"] == [s.skeleton_id for s in bank.skeletons]
    assert metadata["value_list"] == original.value_list
    assert loaded.training is False

    original.eval()
    batch = _batch(original)
    with torch.no_grad():
        before = original(batch)["nextstep_logits"]
        after = loaded(batch)["nextstep_logits"]
    assert torch.equal(before, after)


def test_checkpoint_excludes_the_frozen_twin_but_resyncs_it(cfg: dict, bank: Bank, ontology: dict, tmp_path) -> None:
    cfg = dict(cfg)
    cfg["train"] = dict(cfg["train"], checkpoint_dir=str(tmp_path / "ckpts"))
    model = build_model(cfg, bank, ontology)
    path = save_checkpoint(model, cfg, seed=1)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert not any(key.startswith("frozen_encoder.") for key in payload["state_dict"])
    loaded, _metadata = load_checkpoint(path, cfg, bank, ontology)
    for live, frozen in zip(loaded.encoder.parameters(), loaded.frozen_encoder.parameters()):
        assert torch.equal(live, frozen)
    assert loaded.template_features.numel() == 0, "the cache is stale by construction"


def test_checkpoint_refuses_a_different_class_order(cfg: dict, bank: Bank, ontology: dict, tmp_path) -> None:
    cfg = dict(cfg)
    cfg["train"] = dict(cfg["train"], checkpoint_dir=str(tmp_path / "ckpts"))
    path = save_checkpoint(build_model(cfg, bank, ontology), cfg, seed=1)

    other_bank = Bank(
        templates=bank.templates,
        skeletons=list(bank.skeletons) + [Skeleton("S0004", ["OFFER"], 1)],
        actions=bank.actions,
        templates_by_act=bank.templates_by_act,
    )
    with pytest.raises(ContractViolation, match="skeleton_ids"):
        load_checkpoint(path, cfg, other_bank, ontology)

    # Still 55 subflows, still a valid ontology -- only the FLOW order moved, so
    # H2's classes are renumbered. That is the silent failure this guard exists
    # for: the load must refuse rather than mislabel every intent.
    reordered = dict(ontology)
    flows = list(reversed(ontology["intents"]["flows"]))
    reordered["intents"] = {"flows": flows, "subflows": ontology["intents"]["subflows"]}
    with pytest.raises(ContractViolation, match="subflows"):
        load_checkpoint(path, cfg, bank, reordered)


def test_load_checkpoint_reports_a_missing_file(cfg: dict, bank: Bank, ontology: dict, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "nope.pt"), cfg, bank, ontology)


def test_checkpoint_name_encodes_size_and_fraction(cfg: dict, bank: Bank, ontology: dict, tmp_path) -> None:
    cfg = dict(cfg)
    cfg["train"] = dict(cfg["train"], checkpoint_dir=str(tmp_path / "ckpts"))
    quarter = Bank(
        templates=bank.templates,
        skeletons=bank.skeletons,
        actions=bank.actions,
        templates_by_act=bank.templates_by_act,
        source_fraction=0.25,
    )
    path = save_checkpoint(build_model(cfg, quarter, ontology, size="small"), cfg, seed=2)
    assert os.path.basename(path) == "seed2_small_frac0.25.pt"
