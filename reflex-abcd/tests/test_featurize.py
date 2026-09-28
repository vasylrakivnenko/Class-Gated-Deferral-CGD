"""Tests for reflex.featurize -- the recency-tagging TF-IDF featurizer.

SYNTHETIC FIXTURES ONLY. Nothing here parses the ABCD corpus: the corpus is
being re-downloaded and heavy jobs are serialized through the coordinator.

The tests are shaped by DECISIONS D7's own warning that two earlier order
controls were INCAPABLE OF FAILING and were nearly reported as evidence. So the
central test here asserts both halves of the contrast: plain unigram TF-IDF IS
permutation-invariant (the vacuous control, asserted so it stays vacuous and
nobody re-reads a null from it as "order does not matter"), and recency-tagged
TF-IDF is NOT.
"""

from __future__ import annotations

import os
import re
from collections import Counter

import pytest

from reflex.config import load_config
from reflex.contracts import ContractViolation
from reflex.data import build_context
from reflex.featurize import (
    FeaturizeConfig,
    RecencySpec,
    RecencyTfidfFeaturizer,
    TfidfSpec,
    WindowSpec,
    bucket_sequence,
    build_document,
    build_documents,
    split_context,
    window_turns,
)
from reflex.schemas import NormalizedTurn

CONFIG = os.environ.get("REFLEX_TEST_CONFIG", "configs/default.yaml")

# A ten-turn synthetic thread. Distinct content words per turn so a test can
# tell which turn a feature came from.
LINES = [
    "agent|hello thanks for contacting acme support",
    "customer|my package never arrived",
    "agent|sorry about that could you confirm your order",
    "customer|order id is 3348917502",
    "action|pull-up-account",
    "agent|thanks i have located your account",
    "customer|i would like a refund please",
    "action|validate-purchase",
    "agent|checking the purchase records now",
    "customer|how long will the refund take",
]

STATE_LINE = "state|disclosed: order.order_id=3348917502 || actions: pull-up-account(crystal minh)"

TAG_RE = re.compile(r"^r(\d+)\|")


def tagged(**kw) -> RecencySpec:
    base = {"enabled": True, "max_bucket": 3, "mode": "word"}
    base.update(kw)
    return RecencySpec(**base)


def plain() -> RecencySpec:
    return RecencySpec(enabled=False)


def tags_of(document: str) -> list[str]:
    return [m.group(0) for m in (TAG_RE.match(tok) for tok in document.split()) if m]


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #


def test_full_window_keeps_every_turn_in_order() -> None:
    assert window_turns(LINES, WindowSpec.full()) == LINES


def test_k6_selects_the_last_six_turns() -> None:
    chosen = window_turns(LINES, WindowSpec.last_k(6))
    assert chosen == LINES[4:]
    assert len(chosen) == 6
    # oldest-first ordering survives, and the most recent turn is last
    assert chosen[-1] == LINES[-1]
    assert chosen[0] == LINES[4]


def test_k6_on_a_short_thread_takes_what_exists() -> None:
    assert window_turns(LINES[:3], WindowSpec.last_k(6)) == LINES[:3]
    assert window_turns([], WindowSpec.last_k(6)) == []


def test_first_n_and_head_tail_windows_match_d2b_rows() -> None:
    assert window_turns(LINES, WindowSpec.first_n(2)) == LINES[:2]
    assert window_turns(LINES, WindowSpec.first_last(2, 6)) == LINES[:2] + LINES[4:]


def test_head_tail_never_duplicates_the_middle() -> None:
    short = LINES[:7]
    chosen = window_turns(short, WindowSpec.first_last(2, 6))
    assert chosen == short, "first+last >= n must return the thread contiguously"
    assert len(chosen) == len(set(chosen))


def test_window_parse_accepts_every_config_spelling() -> None:
    assert WindowSpec.parse("full") == WindowSpec.full()
    assert WindowSpec.parse(None) == WindowSpec.full()
    assert WindowSpec.parse(6) == WindowSpec.last_k(6)
    assert WindowSpec.parse("k6") == WindowSpec.last_k(6)
    assert WindowSpec.parse("last6") == WindowSpec.last_k(6)
    assert WindowSpec.parse("first2+last6") == WindowSpec.first_last(2, 6)
    assert WindowSpec.parse({"first": 2, "last": 6}) == WindowSpec.first_last(2, 6)


def test_window_parse_rejects_booleans_and_junk() -> None:
    # D18: bool is an int subclass and a bare YAML scalar can arrive as one.
    with pytest.raises(ContractViolation):
        WindowSpec.parse(True)
    with pytest.raises(ContractViolation):
        WindowSpec.parse("last-6")
    with pytest.raises(ContractViolation):
        WindowSpec(last=-1)


# --------------------------------------------------------------------------- #
# Recency buckets: MAXB = 0 / 1 / 3
# --------------------------------------------------------------------------- #


def test_bucket_sequence_matches_the_documented_granularity() -> None:
    assert bucket_sequence(5, 3) == [3, 3, 2, 1, 0]
    assert bucket_sequence(5, 1) == [1, 1, 1, 1, 0]
    assert bucket_sequence(5, 0) == [0, 0, 0, 0, 0]
    # r0 is always the most recent turn, whatever MAXB is
    for maxb in (0, 1, 2, 3, 8):
        assert bucket_sequence(7, maxb)[-1] == 0


def test_maxb0_is_a_bijective_relabelling_of_plain() -> None:
    """MAXB=0 is one bucket: every token gets r0, so no recency survives."""
    cfg0 = FeaturizeConfig(window=WindowSpec.full(), recency=tagged(max_bucket=0))
    doc0 = build_document(LINES, None, cfg0)
    doc_plain = build_document(LINES, None, FeaturizeConfig(window=WindowSpec.full(), recency=plain()))

    assert set(tags_of(doc0)) == {"r0|"}
    stripped = " ".join(TAG_RE.sub("", tok) for tok in doc0.split())
    assert stripped == doc_plain


def test_maxb1_is_exactly_one_bit_of_recency() -> None:
    """D7: MAXB=1 is 'is this the most recent turn, yes or no' -- and that one
    bit buys +3.1 full / +2.5 k6, most of a 12-point order effect."""
    cfg1 = FeaturizeConfig(window=WindowSpec.full(), recency=tagged(max_bucket=1))
    doc1 = build_document(LINES, None, cfg1)

    assert set(tags_of(doc1)) == {"r0|", "r1|"}, "two buckets, i.e. one bit"

    r0_tokens = {tok[len("r0|"):] for tok in doc1.split() if tok.startswith("r0|")}
    last_turn_words = set(re.findall(r"(?u)\b\w\w+\b", LINES[-1].partition("|")[2]))
    assert r0_tokens == last_turn_words, "r0 covers exactly the most recent turn"


def test_maxb3_gives_four_buckets_capped_at_r3() -> None:
    cfg3 = FeaturizeConfig(window=WindowSpec.full(), recency=tagged(max_bucket=3))
    doc3 = build_document(LINES, None, cfg3)
    assert set(tags_of(doc3)) == {"r0|", "r1|", "r2|", "r3|"}

    for distance, expected in ((0, "r0|"), (1, "r1|"), (2, "r2|"), (3, "r3|"), (4, "r3|"), (9, "r3|")):
        line = LINES[len(LINES) - 1 - distance]
        word = re.findall(r"(?u)\b\w\w+\b", line.partition("|")[2])[0]
        assert f"{expected}{word}" in doc3.split(), f"turn at distance {distance} must carry {expected}"


def test_granularity_saturates_on_a_six_turn_window() -> None:
    """D7's own sanity check that the bucketing does what it claims: 'k6 at MAXB
    6/8/12 is byte-identical, because a 6-turn window only has 6 positions'."""
    docs = {
        maxb: build_document(
            LINES, None, FeaturizeConfig(window=WindowSpec.last_k(6), recency=tagged(max_bucket=maxb))
        )
        for maxb in (6, 8, 12)
    }
    assert docs[6] == docs[8] == docs[12]
    # ... and it is NOT byte-identical to a granularity the window can resolve
    assert docs[6] != build_document(
        LINES, None, FeaturizeConfig(window=WindowSpec.last_k(6), recency=tagged(max_bucket=3))
    )


# --------------------------------------------------------------------------- #
# Order sensitivity -- the control that CAN fail, and the one that cannot
# --------------------------------------------------------------------------- #


def _rows_for(documents, config):
    fx = RecencyTfidfFeaturizer(config)
    matrix = fx.fit_transform(documents)
    return matrix, fx


def test_plain_unigram_tfidf_is_permutation_invariant() -> None:
    """The vacuous control, asserted so it stays vacuous.

    D7: permuting turn lines cannot change a multiset of word unigrams -- that
    control is 'exactly 0.00% order-sensitive' and a null from it means nothing.
    This test pins that arithmetic in place so a future null is never re-read as
    evidence that order does not matter.
    """
    config = FeaturizeConfig(
        window=WindowSpec.full(), recency=plain(), tfidf=TfidfSpec(ngram_range=(1, 1))
    )
    natural = build_document(LINES, None, config)
    shuffled = build_document(list(reversed(LINES)), None, config)

    assert Counter(natural.split()) == Counter(shuffled.split())
    matrix, _ = _rows_for([natural, shuffled], config)
    assert (matrix[0] != matrix[1]).nnz == 0


def test_recency_tagging_is_not_permutation_invariant() -> None:
    """The control that CAN fail: shuffling turns must change the features."""
    config = FeaturizeConfig(window=WindowSpec.full(), recency=tagged(max_bucket=3))
    natural = build_document(LINES, None, config)
    shuffled = build_document(list(reversed(LINES)), None, config)

    assert natural != shuffled
    assert Counter(natural.split()) != Counter(shuffled.split())
    matrix, _ = _rows_for([natural, shuffled], config)
    assert (matrix[0] != matrix[1]).nnz > 0


def test_tagging_survives_at_maxb1_which_is_where_most_of_the_effect_lives() -> None:
    config = FeaturizeConfig(window=WindowSpec.full(), recency=tagged(max_bucket=1))
    natural = build_document(LINES, None, config)
    shuffled = build_document(list(reversed(LINES)), None, config)
    assert Counter(natural.split()) != Counter(shuffled.split())


def test_maxb0_is_order_blind_like_plain() -> None:
    """MAXB=0 tags everything r0, so it must be as order-blind as plain TF-IDF."""
    config = FeaturizeConfig(
        window=WindowSpec.full(), recency=tagged(max_bucket=0), tfidf=TfidfSpec(ngram_range=(1, 1))
    )
    natural = build_document(LINES, None, config)
    shuffled = build_document(list(reversed(LINES)), None, config)
    assert Counter(natural.split()) == Counter(shuffled.split())


# --------------------------------------------------------------------------- #
# The intent path is genuinely untagged
# --------------------------------------------------------------------------- #


def test_intent_path_emits_no_recency_tag_anywhere() -> None:
    """D2b/D7: intent wants the full thread UNTAGGED -- tagging hurts it.

    'Plain' must mean plain: not one tagged token, and no stray bucket token.
    """
    config = FeaturizeConfig(window=WindowSpec.full(), recency=plain())
    doc = build_document(LINES, STATE_LINE, config)

    assert tags_of(doc) == []
    assert not any(re.fullmatch(r"r\d+", tok) for tok in doc.split())
    assert "|" not in doc, "no token may carry a bucket separator on the plain path"


def test_plain_and_tagged_differ_only_by_the_tags() -> None:
    cfg_plain = FeaturizeConfig(window=WindowSpec.full(), recency=plain())
    cfg_tagged = FeaturizeConfig(window=WindowSpec.full(), recency=tagged())
    stripped = " ".join(TAG_RE.sub("", tok) for tok in build_document(LINES, None, cfg_tagged).split())
    assert stripped == build_document(LINES, None, cfg_plain)


def test_turn_mode_tags_the_first_whitespace_chunk_of_each_line() -> None:
    """`turn` mode tags the line once. select.py renders that as
    "speaker|r0|body words here", so the bucket is glued to the FIRST WHITESPACE
    TOKEN of the body and the rest of the line is plain.

    Two consequences are pinned here because they are easy to get wrong and both
    are properties of the SHIP rendering, not of this module:

    * a hyphenated first chunk ("validate-purchase") yields TWO tagged tokens;
    * a first chunk with fewer than two word characters ("i would like...")
      loses the line's bucket entirely.

    `turn` mode is unmeasured on TF-IDF -- D7 measured `word` -- and this is the
    kind of thing that would quietly weaken a turn-mode arm rather than break it.
    """
    config = FeaturizeConfig(window=WindowSpec.last_k(4), recency=tagged(mode="turn"))
    doc = build_document(LINES, None, config)
    tokens = doc.split()

    assert not any(re.fullmatch(r"r\d+", tok) for tok in tokens), "no bare bucket token"
    assert "r1|checking" in tokens and "r0|how" in tokens
    assert ["r2|validate", "r2|purchase"] == [t for t in tokens if t.startswith("r2|")]
    assert not any(t.startswith("r3|") for t in tokens), (
        "the r3 line starts with the one-character token 'i', so the ship rendering "
        "drops its bucket"
    )


# --------------------------------------------------------------------------- #
# Tokenization: the tag must not be severed from its token
# --------------------------------------------------------------------------- #


def test_the_bucket_tag_survives_vectorization_as_one_feature() -> None:
    """Guards the failure mode named in the module docstring.

    sklearn's default token_pattern would split ``r0|refund`` into ``r0`` and
    ``refund`` and destroy every bit of recency while still reporting a
    plausible number. The featurizer owns its tokenization so it cannot.
    """
    config = FeaturizeConfig(window=WindowSpec.last_k(2), recency=tagged(), tfidf=TfidfSpec(ngram_range=(1, 1)))
    fx = RecencyTfidfFeaturizer(config)
    fx.fit(fx.build_documents([LINES]))
    names = set(fx.feature_names())

    assert any(name.startswith("r0|") for name in names)
    assert "r0" not in names, "a bare bucket token means the tag was severed"
    assert "refund" not in names, "an untagged content token means the tag was severed"


def test_plain_mode_matches_sklearn_default_analyzer(cfg) -> None:
    """With recency off, the document is exactly what sklearn's own default
    analyzer would extract from ContextWindow.text -- so 'plain TF-IDF' here is
    the plain TF-IDF of the measured record, not a near-miss variant."""
    from sklearn.feature_extraction.text import CountVectorizer

    context = build_context(CONVERSATION, 7, SCENARIO, cfg)
    config = FeaturizeConfig(window=WindowSpec.full(), recency=plain(), include_state=True)
    doc = build_document(*split_context(context), config)

    reference = CountVectorizer().build_analyzer()(context.text)
    assert doc.split() == list(reference)


# --------------------------------------------------------------------------- #
# ContextWindow integration
# --------------------------------------------------------------------------- #


SCENARIO = {
    "personal": {"customer_name": "crystal minh", "account_id": "kwzrzrwfye"},
    "order": {"order_id": "3348917502", "street_address": "9137 brushwick dr"},
    "flow": "storewide_query",
    "subflow": "timing_4",
}


def _turn(index: int, speaker: str, text: str, **kw) -> NormalizedTurn:
    nextstep = {"agent": "retrieve_utterance", "action": "take_action", "customer": None}[speaker]
    return NormalizedTurn(
        convo_id=1,
        turn_index=index,
        speaker=speaker,
        text=text,
        nextstep=kw.pop("nextstep", nextstep),
        intent="timing_4",
        **kw,
    )


CONVERSATION = [
    _turn(0, "agent", "hello thanks for contacting acme support"),
    _turn(1, "customer", "my package never arrived"),
    _turn(2, "agent", "sorry about that could you confirm your order"),
    _turn(3, "customer", "order id is 3348917502"),
    _turn(4, "action", "pull-up-account", action="pull-up-account", values=["crystal minh"]),
    _turn(5, "agent", "thanks i have located your account"),
    _turn(6, "customer", "i would like a refund please"),
    _turn(7, "agent", "checking the purchase records now"),
    _turn(8, "customer", "how long will the refund take"),
]


@pytest.fixture()
def cfg() -> dict:
    return load_config(CONFIG)


def test_split_context_recovers_turns_and_state(cfg) -> None:
    context = build_context(CONVERSATION, 7, SCENARIO, cfg)
    turns, state = split_context(context)
    assert turns == list(context.turns)
    assert state is not None and state.startswith("state|")
    assert "\n".join(turns + [state]) == context.text


def test_context_window_featurizes_end_to_end(cfg) -> None:
    contexts = [build_context(CONVERSATION, i, SCENARIO, cfg) for i in (2, 5, 7)]
    config = FeaturizeConfig(window=WindowSpec.last_k(6), recency=tagged())
    fx = RecencyTfidfFeaturizer(config)
    matrix = fx.fit_transform_contexts(contexts)
    assert matrix.shape[0] == 3
    assert fx.n_features > 0
    assert fx.describe()["recency"]["max_bucket"] == 3


def test_state_line_is_excluded_by_default_and_never_tagged(cfg) -> None:
    """include_state defaults to false: build_context's state line summarizes the
    WHOLE prefix, so folding it in would leak full-thread information into every
    k6 condition and dissolve the contrast D2b measures."""
    context = build_context(CONVERSATION, 7, SCENARIO, cfg)
    turns, state = split_context(context)
    assert state is not None

    excluded = build_document(turns, state, FeaturizeConfig(recency=tagged()))
    included = build_document(turns, state, FeaturizeConfig(recency=tagged(), include_state=True))

    assert "disclosed" not in excluded.split()
    assert "disclosed" in included.split(), "the state line is included when asked for"
    assert "r0|disclosed" not in included.split(), "the state line is never bucketed"


def test_featurizer_refuses_a_window_the_global_k_cannot_supply() -> None:
    """ContextWindow.turns is already truncated to data.context_turns_K; a wider
    featurizer window would silently get fewer turns than it asked for and report
    a number for a configuration that never ran (the D6 failure mode)."""
    narrow = load_config(CONFIG, ["data.context_turns_K=4"])
    fx = RecencyTfidfFeaturizer(FeaturizeConfig(window=WindowSpec.last_k(6)))
    with pytest.raises(ContractViolation) as excinfo:
        fx.assert_window_reachable(narrow)
    assert "context_turns_K" in str(excinfo.value)

    RecencyTfidfFeaturizer(FeaturizeConfig(window=WindowSpec.last_k(4))).assert_window_reachable(narrow)
    RecencyTfidfFeaturizer(FeaturizeConfig(window=WindowSpec.full())).assert_window_reachable(
        load_config(CONFIG)
    )


def test_full_window_is_rejected_against_a_truncating_global_k() -> None:
    narrow = load_config(CONFIG, ["data.context_turns_K=6"])
    with pytest.raises(ContractViolation):
        RecencyTfidfFeaturizer(FeaturizeConfig(window=WindowSpec.full())).assert_window_reachable(narrow)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_documents_are_deterministic() -> None:
    config = FeaturizeConfig(window=WindowSpec.last_k(6), recency=tagged())
    first = build_documents([LINES, LINES[:4]], config)
    second = build_documents([LINES, LINES[:4]], config)
    assert first == second


def test_fitting_twice_gives_the_same_vocabulary_and_matrix() -> None:
    config = FeaturizeConfig(window=WindowSpec.last_k(6), recency=tagged())
    docs = build_documents([LINES, LINES[:6], LINES[2:]], config)

    a = RecencyTfidfFeaturizer(config)
    b = RecencyTfidfFeaturizer(config)
    xa = a.fit_transform(docs)
    xb = b.fit_transform(docs)

    assert a.feature_names() == b.feature_names()
    assert (xa != xb).nnz == 0


def test_transform_is_stable_across_calls() -> None:
    config = FeaturizeConfig(window=WindowSpec.full(), recency=tagged())
    fx = RecencyTfidfFeaturizer(config)
    fx.fit(fx.build_documents([LINES, LINES[:5]]))
    first = fx.transform_contexts([LINES[:7]])
    second = fx.transform_contexts([LINES[:7]])
    assert (first != second).nnz == 0


def test_unfitted_featurizer_raises_rather_than_returning_zeros() -> None:
    fx = RecencyTfidfFeaturizer(FeaturizeConfig())
    with pytest.raises(ContractViolation):
        fx.transform(["a b c"])
    with pytest.raises(ContractViolation):
        _ = fx.n_features


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #


def test_from_mapping_reads_the_select_style_spelling() -> None:
    config = FeaturizeConfig.from_mapping({"K": 6, "tag_recency": True, "recency_buckets": 3})
    assert config.window == WindowSpec.last_k(6)
    assert config.recency.enabled and config.recency.max_bucket == 3

    intent = FeaturizeConfig.from_mapping({"K": "full", "tag_recency": False})
    assert intent.window.is_full() and not intent.recency.enabled


def test_from_mapping_rejects_unknown_keys() -> None:
    with pytest.raises(ContractViolation):
        FeaturizeConfig.from_mapping({"K": 6, "recency_bukets": 3})


def test_describe_carries_the_whole_configuration() -> None:
    """D6: a baseline is only a baseline WITH its configuration attached."""
    config = FeaturizeConfig(window=WindowSpec.last_k(6), recency=tagged())
    described = config.describe()
    assert described["window"]["label"] == "last6"
    assert described["recency"] == {
        "enabled": True,
        "max_bucket": 3,
        "mode": "word",
        "tag_speaker": False,
        "sep": "|",
    }
    assert described["include_state"] is False
    assert described["tfidf"]["ngram_range"] == [1, 2]


# --------------------------------------------------------------------------- #
# Conformance with the probes/ harness
# --------------------------------------------------------------------------- #


def _probe_api():
    """The probe harness, if the concurrent agent's package is on disk."""
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return pytest.importorskip("probes.featurizer_api")


def test_probe_harness_conformance_report_passes() -> None:
    """The probes/ harness requires render_context to be BYTE-IDENTICAL to
    reflex.select._render_variant, the ship path's renderer.

    Its own words: a probe measured on a rendering reflex.select cannot
    reproduce is not transferable to the ship path. So the harness's check is
    the authority here, not a re-implementation of it in this file.
    """
    api = _probe_api()
    from reflex.featurize import build_featurizer

    report = api.conformance_report(build_featurizer())
    assert report["ok"], report
    assert report["n_failures"] == 0
    assert report["checks_run"] > 0
    assert report["notes"] == []


def test_the_two_paths_tokenize_identically() -> None:
    """Arm B0 builds documents directly; the probe harness renders to a string
    and vectorizes that. The two must produce the same tokens, or the arm and
    the probe measure different representations under the same name (D6)."""
    api = _probe_api()
    from reflex.featurize import build_featurizer, tokenize_rendered

    featurizer = build_featurizer()
    for spec in api.CONFORMANCE_SPECS:
        window = WindowSpec.full() if spec.k in ("full", "inherit") else WindowSpec.last_k(int(spec.k))
        config = FeaturizeConfig(
            window=window,
            recency=RecencySpec(
                enabled=spec.tag_recency,
                max_bucket=spec.recency_buckets,
                mode=spec.recency_mode,
            ),
            include_state=True,
            lowercase=False,
        )
        for context in api.synthetic_contexts():
            direct = build_document(*split_context(context), config)
            via_render = tokenize_rendered(featurizer.render_context(context, spec))
            assert direct.split() == via_render, (spec.label(), context.convo_id)


def test_a_stock_vectorizer_would_have_destroyed_the_tagging() -> None:
    """The silent failure this featurizer exists to prevent, demonstrated.

    A stock TfidfVectorizer over the ship rendering applies sklearn's default
    token_pattern, which splits `r0|refund` into `r0` and `refund`: the bucket
    becomes its own feature, every content word goes back to being untagged, and
    the run still completes and reports a plausible accuracy.
    """
    api = _probe_api()
    from sklearn.feature_extraction.text import TfidfVectorizer

    from reflex.featurize import build_featurizer

    featurizer = build_featurizer()
    spec = api.RenderSpec(k=6, tag_recency=True, recency_buckets=3, recency_mode="word")
    rendered = featurizer.render_context(api.synthetic_contexts()[3], spec)

    naive = set(TfidfVectorizer(ngram_range=(1, 1)).fit([rendered]).get_feature_names_out())
    assert "r0" in naive and "refund" in naive, "the stock analyzer severs the tag"
    assert not any(name.startswith("r0|") for name in naive)

    ours = set(
        featurizer.make_vectorizer(api.VectorizerSpec(ngram_range=(1, 1), min_df=1))
        .fit([rendered])
        .get_feature_names_out()
    )
    assert any(name.startswith("r0|") for name in ours)
    assert "r0" not in ours


def test_the_order_shuffle_is_deterministic_and_seed_sensitive() -> None:
    """The shuf->shuf arm refits on shuffled train rows and scores shuffled dev
    rows; if a row permuted differently between the two passes the control would
    measure noise instead of order."""
    api = _probe_api()
    from reflex.featurize import build_featurizer

    featurizer = build_featurizer()
    context = api.synthetic_contexts()[3]
    spec = api.RenderSpec(k=6, shuffle="within_window", shuffle_seed=17)

    assert featurizer.render_context(context, spec) == featurizer.render_context(context, spec)
    other_seed = api.RenderSpec(k=6, shuffle="within_window", shuffle_seed=18)
    assert featurizer.render_context(context, spec) != featurizer.render_context(context, other_seed)

    natural = featurizer.render_context(context, api.RenderSpec(k=6))
    assert featurizer.render_context(context, spec) != natural


def test_within_window_shuffle_does_not_change_which_turns_are_visible() -> None:
    """D19's lesson as a test: `pre_window` would change WHICH turns are visible
    as well as their order, which is a confounded control. `within_window`
    permutes only the chosen lines, so the multiset of visible turns is fixed."""
    api = _probe_api()
    from reflex.featurize import build_featurizer

    featurizer = build_featurizer()
    context = api.synthetic_contexts()[3]
    natural = featurizer.render_context(context, api.RenderSpec(k=6)).split("\n")
    shuffled = featurizer.render_context(
        context, api.RenderSpec(k=6, shuffle="within_window", shuffle_seed=3)
    ).split("\n")

    assert Counter(natural) == Counter(shuffled), "same turns, different order"
    assert natural != shuffled
    assert natural[-1] == shuffled[-1], "the state line is held fixed"
