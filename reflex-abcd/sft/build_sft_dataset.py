"""Serialize ABCD agent turns into OpenAI-style `messages` JSONL for Fireworks SFT.

WHAT THIS IS FOR
----------------
The UNCONSTRAINED arm. The learned cache answers a turn by picking a skeleton and
one stored template per act; it can only ever say something the bank contains.
Measured coverage of that bank on held-out chat is ~45% (dev 45.22 / test_seen
44.83 / test_novel 44.27). The open question is whether that 45% is the ceiling
of the TECHNIQUE or the ceiling of the BANK.

A model that generates freely is not bounded by the bank. If it clears 45% by a
wide margin on the same population under the same normalization, the bank is the
bottleneck and the fix is `compile.min_template_count`, not a better selector.
If it does not, the bank is not what is holding the fast path back.

FIVE DECISIONS, EACH ONE DELIBERATE
-----------------------------------
1. THE PROMPT IS `ContextWindow.text`, NOT A HAND-ROLLED RENDERING.
   Spec 6.1 step 4 / 6.9 step 1: that string is "the single rendered string fed
   to the encoder AND to the LLM agent". Building our own would compare two arms
   on two different inputs, which is DECISIONS D6 in a new costume. It also
   inherits the leakage guard for free: the state line carries only scenario
   fields the customer has ALREADY DISCLOSED, never the full scenario.

2. POPULATION = retrieve_utterance TURNS ONLY.
   take_action turns emit a button plus values, not text; they are a different
   task with a different metric. The ~45% coverage figure this arm is being
   compared against is measured over retrieve turns, so the denominators match
   by construction. `--include-actions` exists to override this; it makes the
   comparison to 45% invalid and says so.

3. THE TARGET IS THE LEXICALIZED UTTERANCE (`utterances[turn.utt_id]`), NOT
   `turn.text`.
   schemas.NormalizedTurn is emphatic that utt_id and utt_rank are different
   things -- utt_rank is a 0-based index into the turn's own candidate list and
   belongs to the official evaluator; utt_id is the resolved global id and is
   what fetches gold text. Never swapped here.
   `--target delex` trains on `turn.text` (slot markers intact) instead, which
   puts the model in the bank's own space. Default is `lexical`: we want the
   model doing the WHOLE job, and scoring delexicalizes both sides afterwards.

4. NO BANK, NO TEMPLATES, NO ACTS IN THE PROMPT.
   This arm exists to answer a question about the bank. Showing it the bank
   would answer a different one (that is the `constrained` arm, not this).

5. EVERY FILE SHIPS A PROVENANCE SIDECAR.
   D6: a number is only interpretable with its configuration attached. The
   sidecar records dataset hash, split, context window, target mode, counts and
   the exact config keys that shaped the prompt, so a result can always be tied
   back to the data that produced it.

USAGE
-----
    PYTHONPATH=src python -m sft.build_sft_dataset --split train --out sft/data
    PYTHONPATH=src python -m sft.build_sft_dataset --split test_seen --out sft/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Iterator

SYSTEM_PROMPT = (
    "You are a customer service agent. Given the conversation so far and the "
    "facts the customer has already disclosed, write the agent's next message. "
    "Reply with that message only -- no preamble, no explanation, no quotes."
)


def _iter_examples(cfg: dict[str, Any], split: str, target: str,
                   include_actions: bool) -> Iterator[dict[str, Any]]:
    """Yield one OpenAI-style `messages` record per agent turn of `split`."""
    from reflex.data import (build_context, build_partitions, iter_agent_turns,
                             load_raw_abcd, load_utterances)

    raw = load_raw_abcd(cfg)
    utterances = load_utterances(cfg)
    partitions = build_partitions(cfg)
    partition = getattr(partitions, split)

    # scenario lives on the raw conversation, not on the partition
    scenarios: dict[int, dict] = {}
    for convos in raw.values():
        for convo in convos:
            scenarios[int(convo["convo_id"])] = convo.get("scenario", {}) or {}

    skipped_no_gold = 0
    skipped_action = 0

    for convo_id, turn_index, turn in iter_agent_turns(partition):
        if turn.nextstep != "retrieve_utterance":
            # take_action and the synthetic end_conversation turn emit no text
            skipped_action += 1
            if not include_actions:
                continue

        if target == "delex":
            gold = (turn.text or "").strip()
        else:
            uid = turn.utt_id  # RESOLVED global id -- never utt_rank
            gold = utterances[uid].strip() if uid is not None and 0 <= uid < len(utterances) else ""

        if not gold:
            skipped_no_gold += 1
            continue

        turns = partition[convo_id]
        context = build_context(turns, turn_index, scenarios.get(int(convo_id), {}), cfg)

        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context.text},
                {"role": "assistant", "content": gold},
            ],
            # not sent to the trainer; kept so a generation can be joined back
            "_meta": {"convo_id": int(convo_id), "turn_index": int(turn_index)},
        }

    _iter_examples.skipped_no_gold = skipped_no_gold  # type: ignore[attr-defined]
    _iter_examples.skipped_action = skipped_action  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train",
                    help="train | dev | test_seen | test_novel")
    ap.add_argument("--out", default="sft/data", help="output directory")
    ap.add_argument("--target", choices=("lexical", "delex"), default="lexical",
                    help="lexical: utterances[utt_id] (default). delex: turn.text")
    ap.add_argument("--include-actions", action="store_true",
                    help="also emit take_action turns; INVALIDATES the comparison "
                         "against the ~45%% bank-coverage figure")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="config override, e.g. --set data.context_turns_K=full")
    ap.add_argument("--limit", type=int, default=0, help="stop after N records (smoke test)")
    args = ap.parse_args(argv)

    from reflex.config import load_config

    cfg = load_config()
    for ov in args.overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        cur = node[parts[-1]]
        node[parts[-1]] = type(cur)(val) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else val

    os.makedirs(args.out, exist_ok=True)
    stem = f"abcd_{args.split}_{args.target}"
    path = os.path.join(args.out, stem + ".jsonl")

    n = 0
    hasher = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as fh:
        for rec in _iter_examples(cfg, args.split, args.target, args.include_actions):
            meta = rec.pop("_meta")
            line = json.dumps(rec, ensure_ascii=False)
            fh.write(line + "\n")
            hasher.update(line.encode("utf-8"))
            n += 1
            if args.limit and n >= args.limit:
                break

    sidecar = {
        "file": path,
        "n_examples": n,
        "split": args.split,
        "target": args.target,
        "include_actions": bool(args.include_actions),
        "content_sha256": hasher.hexdigest()[:16],
        "system_prompt": SYSTEM_PROMPT,
        "prompt_source": "reflex.data.build_context(...).text  (spec 6.9 step 1)",
        "gold_source": ("utterances[turn.utt_id]" if args.target == "lexical" else "turn.text"),
        "config": {
            "data.abcd_dir": cfg["data"]["abcd_dir"],
            "data.context_turns_K": cfg["data"]["context_turns_K"],
            "data.synthesize_end_conversation": cfg["data"].get("synthesize_end_conversation"),
        },
        "skipped_no_gold": getattr(_iter_examples, "skipped_no_gold", None),
        "compare_against": (
            "bank template coverage on the same split: dev 45.22%, test_seen 44.83%, "
            "test_novel 44.27% (turns fully reconstructible). Score a generation by "
            "delexicalizing + normalizing BOTH sides with reflex.compile's own "
            "functions -- raw exact match holds the model to a harsher standard "
            "than the bank and is not comparable."
        ),
    }
    with open(os.path.join(args.out, stem + ".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)

    print(json.dumps({"wrote": path, "n_examples": n,
                      "sha": sidecar["content_sha256"],
                      "skipped_no_gold": sidecar["skipped_no_gold"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
