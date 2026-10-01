"""Which tiers may answer a question: the switches /admin sets, kept in usage.db.

Order: Pre-Tier 0 (regex) -> Tier 0 (local NLI) -> Tier 1 (Jev) -> Tier 2 (hosted LLM).
The numbering is /admin's. In the code Jev has no tier number (router/harness.py calls it
`llm`) and "Tier 0" is router/tier0.py's NLI model, so Tier 1 == Jev here.

Tier 1 is not an ordinary link in that chain. Jev also answers every yes/no question the
first two tiers defer (harness `_fallback`), gives the reading on judgment questions, picks
the facts and choices the regex rules miss, and checks what Tier 2 reads. Turning it off
hands all of that to Tier 2, unchecked (router/reader.py `decide`), which the bake-off never
measured -- it only scored fact reading. With Tier 1 and Tier 2 both off nothing is left to
answer a question the rules miss, so `problem()` refuses that combination.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

# The Tier 2 options /admin offers: id -> (label, the Jev check floor the bake-off set for it).
# The clients themselves are in router/reader.py (READERS maps these ids to factories); this
# module stays free of heavy imports so it can be read without spaCy or the NLI model.
TIER2 = {
    "gpt-oss-120b": ("gpt-oss-120b (Fireworks)", 0.7),
    "gpt-oss-120b-priority": ("gpt-oss-120b (Fireworks, priority)", 0.7),
    "gemma-4-26b": ("Gemma 4 26B-A4B (OpenRouter)", 0.8),
}

TIER2_OFF = "off"
TIER2_CHOICES = (TIER2_OFF, *TIER2)
# Back-compat for `ask_ui.py --tier2 standard|priority`, which the live systemd unit passes.
TIER2_ALIASES = {"standard": "gpt-oss-120b", "priority": "gpt-oss-120b-priority"}

LABELS = {"pretier0": "Pre-Tier 0 (regex)", "tier0": "Tier 0 (local NLI)", "tier1": "Tier 1 (Jev)",
          "classifiers": "Free classifiers", "tier2": "Tier 2 (hosted LLM)"}


@dataclass(frozen=True)
class Stages:
    pretier0: bool = True
    tier0: bool = True
    tier1: bool = True  # Jev
    classifiers: bool = False  # the task classifiers in the bank; on standby by default
    tier2: str = TIER2_OFF  # TIER2_OFF or a key of reader.READERS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, base: "Stages | None" = None) -> "Stages":
        """Only the keys present override `base` (or the defaults), so /admin can send one switch."""
        base = base or cls()
        fields = {k: bool(d[k]) for k in ("pretier0", "tier0", "tier1", "classifiers") if k in d}
        if "tier2" in d:
            t2 = str(d["tier2"] or TIER2_OFF)
            fields["tier2"] = TIER2_ALIASES.get(t2, t2)
        return replace(base, **fields)

    def problem(self) -> str | None:
        """Why this combination can't be used, or None. /admin and /api reject on this."""
        if self.tier2 not in TIER2_CHOICES:
            return f"Tier 2 must be one of {', '.join(TIER2_CHOICES)}."
        if not self.tier1 and self.tier2 == TIER2_OFF:
            return ("Turn on Tier 1 (Jev) or Tier 2: with both off nothing answers a question the "
                    "regex rules miss, so every question would defer.")
        if self.classifiers and not self.tier1:
            return "The free classifiers need Tier 1 (Jev): it routes the question to one and checks the document's type."
        return None

    @property
    def reader_id(self) -> str | None:
        return None if self.tier2 == TIER2_OFF else self.tier2

    def summary(self) -> str:
        """One line for the admin page and the request log."""
        on = [LABELS[k] for k in ("pretier0", "tier0", "tier1") if getattr(self, k)]
        if self.classifiers:
            on.append(LABELS["classifiers"])
        if self.reader_id:
            on.append(TIER2[self.reader_id][0])
        return " -> ".join(on) if on else "nothing (every question defers)"

    def mask(self) -> str:
        """A short, stable string for usage.db, so a day's numbers can be read back
        against the configuration that produced them: "p1 t0 j1 c0 r:gpt-oss-120b"."""
        return (f"p{int(self.pretier0)} t{int(self.tier0)} j{int(self.tier1)} "
                f"c{int(self.classifiers)} r:{self.tier2}")
