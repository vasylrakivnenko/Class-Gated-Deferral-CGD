"""The /admin Routing Pipeline switches (router/stages.py).

These run anywhere: router/stages.py deliberately imports nothing heavy, so the
rules /admin enforces can be checked without spaCy or the NLI model. Tier 2's
reader is switched by router/tier2.py (tests/test_tier2.py).
"""
import pytest

from router.stages import Stages


def test_default_is_the_live_shape():
    st = Stages()
    assert (st.pretier0, st.tier0, st.tier1) == (True, True, True)
    assert st.classifiers is False  # the bank is on standby
    assert st.problem(tier2_on=False) is None


def test_from_dict_only_overrides_what_it_is_given():
    base = Stages(pretier0=False)
    got = Stages.from_dict({"tier0": False}, base)
    assert (got.pretier0, got.tier0, got.tier1) == (False, False, True)


def test_from_dict_ignores_what_is_not_a_switch():
    """Tier 2 is router/tier2.py's; an old saved row or a stray key changes nothing here."""
    assert Stages.from_dict({"tier2": "gpt-oss-120b", "nonsense": 1}) == Stages()


def test_llm_only_is_allowed():
    st = Stages.from_dict({"pretier0": False, "tier0": False, "tier1": False})
    assert st.problem(tier2_on=True) is None


def test_everything_off_is_refused():
    """Tier 1 and Tier 2 both off leaves nothing to answer what the rules miss."""
    st = Stages.from_dict({"tier1": False})
    assert "Tier 2" in st.problem(tier2_on=False)


def test_classifiers_need_jev_to_route():
    st = Stages.from_dict({"tier1": False, "classifiers": True})
    assert "Tier 1" in st.problem(tier2_on=True)


def test_mask_is_stable_and_records_the_reader():
    st = Stages.from_dict({"tier0": False})
    assert st.mask("openrouter-gemma") == "p1 t0 j1 c0 r:openrouter-gemma"
    assert Stages().mask("off") == "p1 t1 j1 c0 r:off"


def test_summary_names_the_tiers_in_order():
    assert Stages().summary("gpt-oss-120b (Fireworks)") == \
        "Pre-Tier 0 (regex) -> Tier 0 (local NLI) -> Tier 1 (Jev) -> Tier 2: gpt-oss-120b (Fireworks)"
    assert Stages(pretier0=False, tier0=False, tier1=False).summary(None) == "nothing (every question defers)"


def test_stages_are_frozen():
    st = Stages()
    with pytest.raises(Exception):
        st.tier1 = False
