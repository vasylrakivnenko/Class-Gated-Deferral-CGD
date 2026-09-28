"""REFLEXIVE v1 -- hierarchical response selection with a confidence gate, on ABCD.

The fast path does NOT generate text. Per agent turn it recognizes the situation,
picks a response SKELETON (an ordered list of dialogue acts seen in training),
picks one stored SENTENCE TEMPLATE per act, fills the template slots from known
values -- or picks an ACTION with values instead of speaking -- and, when not
confident, ESCALATES the turn to an LLM.

WHERE TO START
--------------
:mod:`reflex.contracts` is the frozen module-boundary contract: every public
function of every spec Section 4 module, with full type hints, the MECE
ownership map, and three boundary rulings. Read it before writing any code.

:mod:`reflex.schemas` holds the spec Section 5 schemas as frozen dataclasses.
Field names are normative.

TWO HARD RULES
--------------
* **ZERO PAID API CALLS.** :mod:`reflex.llm_agent` is fully implemented but
  disabled while ``llm.enabled`` is false, and raises
  :class:`reflex.contracts.LLMDisabledError` if invoked. Act labeling defaults to
  the free local ``rules_plus_embed`` labeler.
* **No numeric threshold, model id, path or price in code** (spec 10). Read
  ``cfg``, loaded by :mod:`reflex.config` from ``configs/default.yaml``.
"""

from __future__ import annotations

__version__ = "1.0.0.dev0"

#: The twelve spec Section 4 modules, plus the cross-cutting config module.
#: Import them lazily -- importing :mod:`reflex.models` pulls in torch.
MODULES: tuple[str, ...] = (
    "config",
    "data",
    "compile",
    "models",
    "train",
    "calibrate",
    "select",
    "gate",
    "fill",
    "llm_agent",
    "evaluate",
    "report",
    "run",
)

__all__ = ["__version__", "MODULES"]
