"""Guard the ZERO-PAID-API-CALLS rule at the repo level.

These tests are deliberately written to pass while :mod:`reflex.llm_agent` is
still a stub and to START ENFORCING the moment it is implemented. Whoever
implements module 4.8 must keep them green.

The rule: ``llm.enabled`` ships ``false``; the LLM agent is fully implemented but
refuses to run while disabled; act labeling defaults to a free local labeler; and
no module reaches a hosted endpoint by any other route.
"""

from __future__ import annotations

import os
import re

import pytest

from reflex import contracts
from reflex.config import load_config, repo_root

SRC = os.path.join(repo_root(), "src", "reflex")

#: Client libraries and hostnames that would indicate a paid call path outside
#: the sanctioned llm_agent seam.
FORBIDDEN_PATTERNS = (
    r"\bimport\s+openai\b",
    r"\bfrom\s+openai\b",
    r"\bimport\s+anthropic\b",
    r"\bfrom\s+anthropic\b",
    r"\bimport\s+cohere\b",
    r"\bimport\s+litellm\b",
    r"api\.openai\.com",
    r"api\.anthropic\.com",
    r"api\.fireworks\.ai",
    r"api\.cohere\.ai",
    r"generativelanguage\.googleapis\.com",
)


def _module_sources() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in sorted(os.listdir(SRC)):
        if name.endswith(".py"):
            with open(os.path.join(SRC, name), "r", encoding="utf-8") as fh:
                out[name] = fh.read()
    return out


def test_llm_enabled_defaults_to_false() -> None:
    """The kill switch ships off. Changing this default is a deliberate act."""
    cfg = load_config()
    assert cfg["llm"]["enabled"] is False, (
        "configs/default.yaml must ship llm.enabled: false -- zero paid API calls"
    )


def test_act_labeler_defaults_to_free_local() -> None:
    """Spec 6.2 step 3a's LLM labeler is available but must not be the default."""
    cfg = load_config()
    assert cfg["compile"]["act_labeler"] == "rules_plus_embed", (
        "compile.act_labeler must default to the free local labeler"
    )


def test_model_ids_and_prices_ship_unfilled() -> None:
    """No invented model ids or prices. They stay ``<fill: ...>`` until a human fills them."""
    cfg = load_config()
    assert "<fill" in cfg["llm"]["strong"]
    assert "<fill" in cfg["llm"]["cheap"]
    assert "<fill" in str(cfg["llm"]["prices_usd_per_million"])
    assert "<fill" in cfg["llm"]["price_list_date"]


def test_require_filled_refuses_placeholders() -> None:
    """Reading an unfilled placeholder raises instead of returning a guess."""
    from reflex.config import require_filled

    cfg = load_config()
    for key in ("llm.strong", "llm.cheap", "llm.prices_usd_per_million", "llm.price_list_date"):
        with pytest.raises(contracts.PlaceholderConfigError):
            require_filled(cfg, key)


@pytest.mark.parametrize("pattern", FORBIDDEN_PATTERNS)
def test_no_hosted_client_outside_llm_agent(pattern: str) -> None:
    """Only ``llm_agent.py`` may reference a hosted inference client (spec 4.8).

    ``llm_agent.py`` is the ONLY sanctioned seam, and even there the call is
    gated by :func:`reflex.contracts.build_llm_agent`'s kill switch.
    """
    rx = re.compile(pattern)
    offenders = [
        name
        for name, src in _module_sources().items()
        if name != "llm_agent.py" and rx.search(src)
    ]
    assert not offenders, (
        f"hosted-inference reference {pattern!r} found outside llm_agent.py: {offenders}"
    )


def test_build_llm_agent_refuses_while_disabled() -> None:
    """``build_llm_agent`` must raise ``LLMDisabledError`` when ``llm.enabled`` is false.

    Skips while the module is a stub; enforces from the first real
    implementation onward. Failing at CONSTRUCTION rather than at the first call
    is deliberate -- a misconfigured run should die in a second, not after an
    hour of setup.
    """
    from reflex import llm_agent

    if llm_agent.build_llm_agent is contracts.build_llm_agent:
        pytest.skip("reflex.llm_agent is still the contract stub")
    cfg = load_config()
    assert cfg["llm"]["enabled"] is False
    with pytest.raises(contracts.LLMDisabledError):
        llm_agent.build_llm_agent(cfg, "strong")


def test_llm_decide_refuses_while_disabled() -> None:
    """``llm_decide`` must also refuse, so no code path can bypass the switch."""
    from reflex import llm_agent

    if llm_agent.llm_decide is contracts.llm_decide:
        pytest.skip("reflex.llm_agent is still the contract stub")
    cfg = load_config()
    with pytest.raises((contracts.LLMDisabledError, TypeError, AttributeError)):
        llm_agent.llm_decide(None, None, None, [], cfg)


def test_llm_act_labeler_refuses_while_disabled() -> None:
    """The ``llm`` act labeler must refuse at construction too (spec 6.2 step 3a)."""
    from reflex import compile as compile_mod

    if compile_mod.get_act_labeler is contracts.get_act_labeler:
        pytest.skip("reflex.compile is still the contract stub")
    cfg = load_config(overrides=["compile.act_labeler=llm"])
    assert cfg["llm"]["enabled"] is False
    with pytest.raises(contracts.LLMDisabledError):
        compile_mod.get_act_labeler(cfg)


def test_prompts_are_present_but_marked_unfrozen() -> None:
    """Spec 6.9 step 3: the Arm A prompt is committed; freezing is a later, explicit act."""
    path = os.path.join(repo_root(), "prompts", "agent_A.txt")
    assert os.path.exists(path), "prompts/agent_A.txt is a spec 11.1 deliverable"
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert "NOT FROZEN" in text, "the draft prompt must say so until it is frozen on dev"
    assert "{context_text}" in text, "the user block must carry the encoder's context string"
