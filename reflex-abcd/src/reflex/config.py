"""Cross-cutting: config loading and ``--set`` override handling.

NOT in spec 11.1. Added because every module needs ``cfg`` and spec 10 forbids
thresholds, model ids, paths and prices in code. Recorded as a deviation in
README.

THIS MODULE IS ALREADY IMPLEMENTED -- do not stub it out. Nine agents depend on
it working on day one. Its contract is frozen in :mod:`reflex.contracts`.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Optional, Sequence

import yaml

from reflex.contracts import PlaceholderConfigError

__all__ = [
    "load_config",
    "apply_overrides",
    "resolve_path",
    "require_filled",
    "repo_root",
    "get_dotted",
]

#: Marker for a config value the spec deliberately leaves unfilled.
_FILL_MARKER = "<fill"


def repo_root() -> str:
    """Return the repository root: the parent of ``src/``.

    Relative config paths resolve against this, not the process cwd, so a run
    started from any directory writes to the same place.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_dotted(cfg: dict[str, Any], key: str) -> Any:
    """Return ``cfg`` value at a dotted ``key``.

    Args:
        cfg: Resolved config.
        key: e.g. ``"gate.alpha"``.

    Returns:
        The value.

    Raises:
        KeyError: if any path component is missing.
    """
    node: Any = cfg
    for i, part in enumerate(key.split(".")):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config key not found: {key!r} (failed at {'.'.join(key.split('.')[:i + 1])!r})")
        node = node[part]
    return node


def load_config(path: str = "configs/default.yaml", overrides: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Load the YAML config and apply ``--set`` overrides.

    See :func:`reflex.contracts.load_config` for the frozen contract.
    """
    resolved = path if os.path.isabs(path) else os.path.join(repo_root(), path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"config not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping, got {type(cfg).__name__}: {resolved}")
    cfg["_config_path"] = resolved
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a new config with ``"dotted.key=value"`` overrides applied.

    See :func:`reflex.contracts.apply_overrides` for the frozen contract.
    Overriding an unknown key raises: a typo that silently adds a dead key is
    worse than a crash.
    """
    out = copy.deepcopy(cfg)
    applied: list[str] = list(out.get("_overrides", []))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"malformed --set override (expected key=value): {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"malformed --set override (empty key): {item!r}")
        parts = key.split(".")
        node: Any = out
        for i, part in enumerate(parts[:-1]):
            if not isinstance(node, dict) or part not in node:
                raise ValueError(
                    f"unknown config key path in override {item!r}: "
                    f"{'.'.join(parts[:i + 1])!r} does not exist"
                )
            node = node[part]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise ValueError(f"unknown config key in override {item!r}: {key!r} does not exist")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:  # pragma: no cover - malformed scalar
            raise ValueError(f"could not parse override value in {item!r}: {exc}") from exc
        node[leaf] = value
        applied.append(item)
    out["_overrides"] = applied
    return out


def resolve_path(cfg: dict[str, Any], key: str) -> str:
    """Resolve a dotted config key holding a path into an absolute path.

    See :func:`reflex.contracts.resolve_path` for the frozen contract.
    """
    value = get_dotted(cfg, key)
    if not isinstance(value, str):
        raise ValueError(f"config key {key!r} is not a path string: {value!r}")
    if _FILL_MARKER in value:
        raise PlaceholderConfigError(f"config key {key!r} is still an unfilled placeholder: {value!r}")
    return value if os.path.isabs(value) else os.path.join(repo_root(), value)


def _has_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return _FILL_MARKER in value
    if isinstance(value, dict):
        return any(_has_placeholder(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_placeholder(v) for v in value)
    return False


def require_filled(cfg: dict[str, Any], key: str) -> Any:
    """Return ``cfg[key]``, refusing to return an unfilled placeholder.

    See :func:`reflex.contracts.require_filled` for the frozen contract. The
    ``<fill: ...>`` placeholders in ``llm.*`` are deliberate; never substitute a
    default.
    """
    value = get_dotted(cfg, key)
    if _has_placeholder(value):
        raise PlaceholderConfigError(
            f"config key {key!r} is still an unfilled placeholder ({value!r}). "
            "Fill it in configs/default.yaml (or override with --set) before a run "
            "that needs it. Do not guess a value."
        )
    return value
