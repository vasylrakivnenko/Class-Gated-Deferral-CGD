"""Adversarial tests for ``configs/default.yaml`` itself.

Three DISTINCT silent YAML failure modes were found in this one file inside an
hour, each invisible until something downstream broke oddly:

1. An unquoted ``#`` inside a key (``account #: account_id``) makes YAML read the
   line as a bare key with no colon. The whole file stops parsing, and the error
   points at the NEXT line.
2. A bare ``off`` is parsed by YAML 1.1 as the BOOLEAN ``False``, so
   ``gate.affect_signal`` never equals the string ``"off"`` any code compares it
   to. That one would have raised on every single turn.
3. A duplicate key -- at any depth -- is silently resolved by PyYAML in favour of
   the LAST one. Two agents each add a block, each believes theirs is live, one
   is dead, nothing errors. This is the worst of the three because the config
   still loads and the run still completes.

A config file that twelve modules read deserves the same adversarial treatment
as code. These tests are cheap, permanent, and make all three impossible to
reintroduce silently.
"""

from __future__ import annotations

import collections
import os
from typing import Any

import pytest
import yaml

from reflex.config import get_dotted, load_config, repo_root

CONFIG_PATH = os.path.join(repo_root(), "configs", "default.yaml")


class _DuplicateRejectingLoader(yaml.SafeLoader):
    """A SafeLoader that records every repeated mapping key instead of dropping it."""

    duplicates: list[tuple[str, int]] = []


def _construct_mapping(loader: _DuplicateRejectingLoader, node: Any, deep: bool = False) -> dict:
    seen: collections.Counter = collections.Counter()
    for key_node, _value_node in node.value:
        seen[loader.construct_object(key_node, deep=True)] += 1
    for key, count in seen.items():
        if count > 1:
            _DuplicateRejectingLoader.duplicates.append((str(key), node.start_mark.line + 1))
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_DuplicateRejectingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def test_config_has_no_duplicate_keys_at_any_depth() -> None:
    """PyYAML silently keeps the LAST duplicate. That is a phantom-configuration bug.

    Two agents each add a ``fill:`` block; the first one is silently dead; the
    module reading it behaves as if its settings were never written, and nothing
    anywhere reports an error. Reject duplicates outright.
    """
    _DuplicateRejectingLoader.duplicates = []
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        yaml.load(handle, _DuplicateRejectingLoader)
    assert not _DuplicateRejectingLoader.duplicates, (
        "configs/default.yaml has duplicate keys; PyYAML keeps only the LAST of each, "
        "so an earlier block is silently dead: "
        + ", ".join(f"{key!r} near line {line}" for key, line in _DuplicateRejectingLoader.duplicates)
    )


def test_config_parses_at_all() -> None:
    """A ``#`` inside an unquoted key truncates the line and breaks the whole file."""
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, dict) and cfg, "configs/default.yaml did not parse into a mapping"


def test_keys_containing_hash_are_quoted() -> None:
    """``account #: account_id`` parses as the bare key ``account``. Quote such keys.

    The file currently parses, so this guards the reintroduction rather than the
    original defect: any key whose text contains ``#`` must survive the round
    trip, which it only does when quoted.
    """
    cfg = load_config(CONFIG_PATH)
    terms = get_dotted(cfg, "compile.merge_field_terms")
    hashed = [key for key in terms if "#" in str(key)]
    assert hashed, (
        "compile.merge_field_terms no longer contains a '#'-bearing key. If that is "
        "deliberate, delete this test; if a key was silently truncated by an unquoted "
        "'#', quote it instead."
    )
    for key in hashed:
        assert str(key).strip() != "", f"key {key!r} parsed as empty"


@pytest.mark.parametrize(
    "key",
    [
        "gate.affect_signal",
        "compile.act_labeler",
        "model.pooling",
        "data.turn_list_key",
        "train.truncation_side",
        "calibrate.novelty_index_backend",
        "llm.tokenizer",
    ],
)
def test_enum_like_keys_are_not_yaml_booleans(key: str) -> None:
    """YAML 1.1 coerces bare ``on``/``off``/``yes``/``no`` to booleans.

    ``gate.affect_signal: off`` is a string in the author's head and ``False`` in
    the parser's. Code that compares it to ``"off"`` rejects the shipped config
    and raises on every turn. Either quote the value or make the reader accept
    both spellings -- this test just makes the coercion visible.
    """
    cfg = load_config(CONFIG_PATH)
    value = get_dotted(cfg, key)
    if isinstance(value, bool):
        pytest.fail(
            f"config key {key!r} parsed as the BOOLEAN {value!r}, not a string. YAML 1.1 "
            "coerces bare on/off/yes/no. Quote it in configs/default.yaml, or confirm the "
            "reader handles both spellings and remove this key from the list."
        )


def test_every_config_key_read_by_a_module_exists() -> None:
    """A module reading a key this file does not define is a runtime KeyError.

    Twenty such keys were found at once in ``train.py`` and ``llm_agent.py``,
    which is how we learned those modules had only ever been imported, never
    executed. Scanning the source is crude but it catches the whole class.
    """
    import re

    cfg = load_config(CONFIG_PATH)

    def flatten(node: dict, prefix: str = "") -> set[str]:
        out: set[str] = set()
        for key, value in node.items():
            path = f"{prefix}{key}"
            out.add(path)
            if isinstance(value, dict):
                out |= flatten(value, path + ".")
        return out

    present = flatten({k: v for k, v in cfg.items() if not str(k).startswith("_")})
    pattern = re.compile(
        r'(?:get_dotted|resolve_path|require_filled)\(\s*cfg\s*,\s*"([^"]+)"\s*\)'
    )
    source_dir = os.path.join(repo_root(), "src", "reflex")
    missing: dict[str, list[str]] = {}
    for filename in sorted(os.listdir(source_dir)):
        if not filename.endswith(".py"):
            continue
        with open(os.path.join(source_dir, filename), "r", encoding="utf-8") as handle:
            text = handle.read()
        absent = sorted(k for k in set(pattern.findall(text)) if k not in present)
        if absent:
            missing[filename] = absent
    assert not missing, (
        "modules read config keys that configs/default.yaml does not define; each one is a "
        f"KeyError the first time that code path runs: {missing}"
    )
