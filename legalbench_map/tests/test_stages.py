"""The /admin Routing Pipeline switches (router/stages.py).

These run anywhere: router/stages.py deliberately imports nothing heavy, so the
rules /admin enforces can be checked without spaCy or the NLI model.
"""
import pytest

from router.stages import TIER2, TIER2_OFF, Stages


def test_default_is_the_live_shape():
    st = Stages()
    assert (st.pretier0, st.tier0, st.tier1) == (True, True, True)
    assert st.classifiers is False  # the bank is on standby
    assert st.tier2 == TIER2_OFF
    assert st.problem() is None


@pytest.mark.parametrize("flag,expected", [("standard", "gpt-oss-120b"), ("priority", "gpt-oss-120b-priority")])
def test_service_flag_aliases_still_work(flag, expected):
    """The live systemd unit passes --tier2 priority; it must keep meaning gpt-oss-120b."""
    assert Stages.from_dict({"tier2": flag}).tier2 == expected


def test_from_dict_only_overrides_what_it_is_given():
    base = Stages(pretier0=False, tier2="gemma-4-26b")
    got = Stages.from_dict({"tier0": False}, base)
    assert (got.pretier0, got.tier0, got.tier2) == (False, False, "gemma-4-26b")


def test_llm_only_is_allowed():
    st = Stages.from_dict({"pretier0": False, "tier0": False, "tier1": False, "tier2": "gpt-oss-120b"})
    assert st.problem() is None
    assert st.reader_id == "gpt-oss-120b"


def test_everything_off_is_refused():
    """Tier 1 and Tier 2 both off leaves nothing to answer what the rules miss."""
    st = Stages.from_dict({"tier1": False})
    assert st.problem() is not None
    assert "Tier 2" in st.problem()


def test_classifiers_need_jev_to_route():
    st = Stages.from_dict({"tier1": False, "classifiers": True, "tier2": "gpt-oss-120b"})
    assert "Tier 1" in st.problem()


def test_unknown_reader_is_refused():
    assert Stages.from_dict({"tier2": "llama-9000"}).problem() is not None


def test_reader_id_is_none_when_off():
    assert Stages().reader_id is None


def test_mask_is_stable_and_records_the_reader():
    st = Stages.from_dict({"tier0": False, "tier2": "gemma-4-26b"})
    assert st.mask() == "p1 t0 j1 c0 r:gemma-4-26b"


def test_summary_names_the_tiers_in_order():
    st = Stages.from_dict({"tier2": "gpt-oss-120b"})
    assert st.summary() == "Pre-Tier 0 (regex) -> Tier 0 (local NLI) -> Tier 1 (Jev) -> gpt-oss-120b (Fireworks)"


def test_every_reader_has_a_label_and_a_check_floor():
    for rid, (label, floor) in TIER2.items():
        assert label and 0 < floor <= 1, rid


def test_stages_are_hashable_and_frozen():
    st = Stages()
    with pytest.raises(Exception):
        st.tier1 = False
