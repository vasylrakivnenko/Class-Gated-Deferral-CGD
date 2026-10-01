"""Which tiers may answer a question: the switches /admin sets, kept in usage.db.

Order: Pre-Tier 0 (regex) -> Tier 0 (local NLI) -> Tier 1 (Jev) -> Tier 2 (hosted LLM).
The numbering is /admin's. In the code Jev has no tier number (router/harness.py calls it
`llm`) and "Tier 0" is router/tier0.py's NLI model, so Tier 1 == Jev here. Tier 2 is not
switched here: router/tier2.py picks its reader (or off), with a test read first, and keeps
that choice under its own key. Both show in /admin's Routing Pipeline tab.

Tier 1 is not an ordinary link in that chain. Jev also answers every yes/no question the
first two tiers defer (harness `_fallback`), gives the reading on judgment questions, picks
the facts and choices the regex rules miss, and checks what Tier 2 reads. Turning it off
hands all of that to Tier 2, unchecked (router/reader.py `decide`), which the bake-off never
measured -- it only scored fact reading. With Tier 1 and Tier 2 both off nothing is left to
answer a question the rules miss, so `problem()` refuses that combination.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

SETTING = "stages"  # its key in usage.db `settings`
SWITCHES = ("pretier0", "tier0", "tier1", "classifiers")
LABELS = {"pretier0": "Pre-Tier 0 (regex)", "tier0": "Tier 0 (local NLI)", "tier1": "Tier 1 (Jev)",
          "classifiers": "Free classifiers"}


@dataclass(frozen=True)
class Stages:
    pretier0: bool = True
    tier0: bool = True
    tier1: bool = True  # Jev
    classifiers: bool = False  # the task classifiers in the bank; on standby by default

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, base: "Stages | None" = None) -> "Stages":
        """Only the keys present override `base` (or the defaults), so /admin can send one switch."""
        return replace(base or cls(), **{k: bool(d[k]) for k in SWITCHES if k in d})

    def problem(self, tier2_on: bool) -> str | None:
        """Why this combination can't be used, or None. /admin and /api reject on this."""
        if not self.tier1 and not tier2_on:
            return ("Turn on Tier 1 (Jev) or Tier 2: with both off nothing answers a question the "
                    "regex rules miss, so every question would defer.")
        if self.classifiers and not self.tier1:
            return "The free classifiers need Tier 1 (Jev): it routes the question to one and checks the document's type."
        return None

    def summary(self, tier2: str | None) -> str:
        """One line for the admin page: the tiers that are on, in order. `tier2` is the
        live reader's name, None when Tier 2 is off."""
        on = [LABELS[k] for k in SWITCHES if getattr(self, k)]
        if tier2:
            on.append(f"Tier 2: {tier2}")
        return " -> ".join(on) if on else "nothing (every question defers)"

    def mask(self, tier2: str) -> str:
        """A short, stable string for usage.db, so a day's numbers can be read back
        against the configuration that produced them: "p1 t0 j1 c0 r:fireworks-priority".
        `tier2` is router/tier2.py's option id, or "off"."""
        return (f"p{int(self.pretier0)} t{int(self.tier0)} j{int(self.tier1)} "
                f"c{int(self.classifiers)} r:{tier2}")
