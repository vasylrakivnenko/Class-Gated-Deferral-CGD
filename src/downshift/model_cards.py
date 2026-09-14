"""Official vendor descriptions for the models in the registry.

Two jobs, and they pull in different directions, which is why this is a
separate module rather than another field on `ModelSpec`:

1. **Reference.** When a row on the chart looks wrong, the first question is
   what the model actually is -- how big, trained to when, tuned for what. That
   belongs next to the price, not in a browser tab.
2. **Candidate selection.** When a new dataset arrives, the first sweep should
   not be "every model we own". These descriptions are the input to picking a
   shortlist: a multilingual task should prefer a model whose card claims
   multilingual coverage, a long-document task one whose context fits.

`text` is the vendor's own wording, stored **verbatim and unedited**. That is
deliberate. A paraphrase would be our claim about the model dressed as the
vendor's, and the whole point of keeping it is to have an unfiltered statement
of what its makers say it is -- including the marketing, which is itself a
signal. Every card carries a `source` for the same reason prices do.

The structured fields are transcribed from the same card, never inferred. If a
card does not state a data cutoff, the field is empty rather than guessed.

`caveat` is the one field that is OURS rather than the vendor's, and it exists
because transcription kept turning up discrepancies that a card cannot express.
A vendor often publishes different numbers in two places (Gemma 3's HuggingFace
card says 8,192 max output, Google's own page says 128K), and a model is often
served to us by someone other than its maker on different terms (Qwen 3.7 Plus
states a 1M context on Alibaba's endpoint; the Fireworks route we actually call
publishes 262k). Recording only one number would be picking a side silently.
So the card keeps what the cited source says, and `caveat` says what else is
true. Read it before using `context` or `max_output` to decide anything.

Coverage is partial. `ModelSpec.card` returns None for models without an entry,
and every caller must handle that -- an absent card means "nobody has
transcribed it yet", never "this model has no documentation". Call
`missing_cards()` for the current gap list.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCard:
    params: str = ""                       # as the vendor states it, e.g. "3.8B"
    context: int = 0                        # tokens; 0 = not stated
    max_output: int = 0                     # tokens; 0 = not stated
    data_cutoff: str = ""                   # ISO-ish, e.g. "2024-06"
    license: str = ""
    source: str = ""                        # where `text` was copied from
    text: str = ""                          # VERBATIM vendor description
    languages: tuple[str, ...] = field(default_factory=tuple)
    caveat: str = ""                        # OUR note, not the vendor's -- see below

    @property
    def is_multilingual(self) -> bool:
        """More than English claimed. Selection-relevant, cheap to answer."""
        return len(self.languages) > 1

    def digest(self, width: int = 340) -> str:
        """One-paragraph form for a selection prompt.

        Deliberately lossy: a shortlist step wants the gist of thirty models,
        not thirty full cards. Read `text` when the gist is not enough.
        """
        head = " · ".join(p for p in (
            self.params and f"{self.params} params",
            self.context and f"{self.context // 1000}K context",
            self.data_cutoff and f"data to {self.data_cutoff}",
            self.license,
        ) if p)
        body = " ".join(self.text.split())
        if len(body) > width:
            body = body[:width].rsplit(" ", 1)[0] + "…"
        return f"{head}. {body}" if head else body


def _load() -> dict[str, ModelCard]:
    """Read the cards from `model_cards.json`.

    A data file rather than a literal in this module: the text is verbatim
    vendor prose, several paragraphs each, and twenty-odd of them inlined would
    make a 30 KB Python file whose diffs are unreadable and whose string
    escaping invites exactly the silent edits this module exists to prevent.
    JSON round-trips the vendor's own curly quotes, em dashes and non-ASCII
    intact, which a hand-maintained Python literal does not.

    Unknown keys raise rather than being dropped, so a typo in the data file is
    a startup error and not a silently missing field.
    """
    raw = json.loads((pathlib.Path(__file__).parent / "model_cards.json")
                     .read_text(encoding="utf-8"))
    cards = {}
    for key, kw in raw.items():
        kw = dict(kw)
        kw["languages"] = tuple(kw.get("languages") or ())
        cards[key] = ModelCard(**kw)
    return cards


CARDS: dict[str, ModelCard] = _load()


def missing_cards() -> list[str]:
    """Registry keys with no transcribed card, so the gap stays visible.

    Imported lazily to keep this module free of a circular dependency on
    `models`, which imports it.
    """
    from .models import ALL_MODELS
    return sorted(m.key for m in ALL_MODELS
                  if m.key not in CARDS and m.tier not in ("trivial", "encoder"))


def selection_digest(keys: list[str] | None = None) -> str:
    """Render cards as a block a model-selection step can read.

    Models without a card are listed by name under a heading that says so,
    rather than silently dropped -- a shortlist that quietly ignored half the
    registry would be worse than no shortlist.
    """
    from .models import BY_KEY
    keys = keys if keys is not None else sorted(BY_KEY)
    described, undescribed = [], []
    for k in keys:
        spec = BY_KEY.get(k)
        if spec is None:
            continue
        card = CARDS.get(k)
        if card:
            described.append(f"- {spec.label} (`{k}`): {card.digest()}")
        else:
            undescribed.append(f"{spec.label} (`{k}`)")
    out = "\n".join(described)
    if undescribed:
        out += ("\n\nNo vendor description on file for these -- judge them on "
                "measured results only:\n- " + "\n- ".join(undescribed))
    return out
