"""Serialize ABCD agent turns into OpenAI-style `messages` JSONL for the
STRUCTURED SFT arm: the model reproduces the cache's own skeleton + templates,
not a free-text utterance.

WHY A SECOND ARM (see sft/build_sft_dataset.py for the unconstrained one)
--------------------------------------------------------------------------
The unconstrained arm (free-text utterance) answers "is the BANK the ceiling?"
This arm answers "is the SELECTOR the ceiling?" -- same LLM capacity, aimed at
the exact task H5/H7 measure, so it is directly comparable to those numbers:
same label space (skeleton_id, per-act template_id), same metric family.

TARGET IS TEXT, NOT TEMPLATE IDS
---------------------------------
"T003418" is an arbitrary identifier; making a 0.6B model memorize 4,489 of them
wastes capacity on a lookup table it has no way to generalize. The target is the
DELEXICALIZED SENTENCE TEXT for each act in the skeleton, one line per act:

    ACK | thank you.
    ASK | can i have your account id and order id?

Scoring then exact-matches the generated line back to the bank's
(act, text_delex) -> template_id table to recover an id -- unambiguous, unlike
the unconstrained arm's delex-then-normalize comparison.

GOLD SOURCE, AND ITS CAVEAT
----------------------------
skeleton_id / template_ids come from outputs/compile/labels/train.jsonl, i.e.
FROM THE SAME `compile.act_labeler` PIPELINE THAT H5/H7 SCORE AGAINST. This arm
therefore inherits that labeller's caveat: a score here measures agreement with
the labeller's output, not correctness against what the human agent did. See
DECISIONS D15. Do not report a number from this arm without that caveat
attached, same as any H5 number.

PROMPT IS THE SAME ContextWindow.text AS THE UNCONSTRAINED ARM
-----------------------------------------------------------------
Both arms must see the same input for a fair comparison. Only the SUPERVISION
differs (free text vs. structured skeleton+templates).

USAGE
-----
    PYTHONPATH=src python -m sft.build_structured_sft_dataset --split train --out sft/data
    PYTHONPATH=src python -m sft.build_structured_sft_dataset --split test_seen --out sft/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Iterator

SYSTEM_PROMPT = (
    "You are a customer service agent's response planner. Given the "
    "conversation so far and the facts the customer has already disclosed, "
    "output the plan for the agent's next message: one line per dialogue act, "
    "each line formatted as 'ACT | sentence'. Use only the acts ACK, ASK, "
    "CLOSE, CONFIRM, INFORM, INSTRUCT, OFFER, OTHER, VERIFY. Reply with the "
    "plan only -- no preamble, no explanation."
)


def _load_bank_text(bank_path: str) -> dict[str, str]:
    """template_id -> text_delex, so structured targets read as sentences."""
    out: dict[str, str] = {}
    with open(bank_path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["template_id"]] = d["text_delex"]
    return out


def _load_labels(labels_path: str) -> dict[str, dict[str, Any]]:
    """turn_id -> label row (skeleton_id, template_ids, ...)."""
    out: dict[str, dict[str, Any]] = {}
    with open(labels_path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["turn_id"]] = d
    return out


def _skeleton_acts(skeleton_id: str, skeletons_path: str,
                   _cache: dict[str, list[str]] = {}) -> list[str]:
    if not _cache:
        with open(skeletons_path, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                _cache[d["skeleton_id"]] = d["acts"]
    return _cache.get(skeleton_id, [])


def _iter_examples(cfg: dict[str, Any], split: str,
                   bank_text: dict[str, str], labels: dict[str, dict],
                   skeletons_path: str) -> Iterator[dict[str, Any]]:
    from reflex.data import build_context, build_partitions, iter_agent_turns

    partitions = build_partitions(cfg)
    partition = getattr(partitions, split)

    scenarios: dict[int, dict] = {}
    from reflex.data import load_raw_abcd
    raw = load_raw_abcd(cfg)
    for convos in raw.values():
        for convo in convos:
            scenarios[int(convo["convo_id"])] = convo.get("scenario", {}) or {}

    skipped_no_labels = 0
    skipped_no_template_text = 0

    for convo_id, turn_index, turn in iter_agent_turns(partition):
        if turn.nextstep != "retrieve_utterance":
            continue  # same population as the unconstrained arm

        turn_id = f"{split}:{convo_id}:{turn_index}"
        label = labels.get(turn_id)
        if label is None or not label.get("skeleton_id"):
            skipped_no_labels += 1
            continue

        acts = _skeleton_acts(label["skeleton_id"], skeletons_path)
        tids = label.get("template_ids") or []
        if not acts or len(acts) != len(tids):
            skipped_no_labels += 1
            continue

        lines = []
        ok = True
        for act, tid in zip(acts, tids):
            text = bank_text.get(tid, "")
            if not text:
                ok = False
                break
            lines.append(f"{act} | {text}")
        if not ok:
            skipped_no_template_text += 1
            continue

        turns = partition[convo_id]
        context = build_context(turns, turn_index, scenarios.get(int(convo_id), {}), cfg)

        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context.text},
                {"role": "assistant", "content": "\n".join(lines)},
            ],
            "_meta": {"convo_id": int(convo_id), "turn_index": int(turn_index),
                      "skeleton_id": label["skeleton_id"], "template_ids": tids},
        }

    _iter_examples.skipped_no_labels = skipped_no_labels  # type: ignore[attr-defined]
    _iter_examples.skipped_no_template_text = skipped_no_template_text  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="sft/data")
    ap.add_argument("--bank", default="outputs/compile/bank/templates.jsonl")
    ap.add_argument("--skeletons", default="outputs/compile/bank/skeletons.jsonl")
    ap.add_argument("--labels", default="outputs/compile/labels/train.jsonl",
                    help="label file to join against (train-only bank; see caveat below "
                         "if scoring a non-train split -- dev/test golds come from a "
                         "DIFFERENT derivation, reflex.train._derive_turn_labels, and are "
                         "not yet wired here)")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
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

    if args.split != "train":
        print(json.dumps({
            "refused": True,
            "reason": ("labels/train.jsonl only covers the TRAIN split's own gold. "
                       "Scoring a held-out split needs the dev-style derivation "
                       "(reflex.train._derive_turn_labels), which probes/response_labels.py "
                       "already implements for H5/H7 -- reuse that, don't re-derive it here. "
                       "Refusing rather than silently joining nothing."),
        }, indent=2))
        return 1

    bank_text = _load_bank_text(args.bank)
    labels = _load_labels(args.labels)

    os.makedirs(args.out, exist_ok=True)
    stem = f"abcd_{args.split}_structured"
    path = os.path.join(args.out, stem + ".jsonl")

    n = 0
    hasher = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as fh:
        for rec in _iter_examples(cfg, args.split, bank_text, labels, args.skeletons):
            rec.pop("_meta")
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
        "content_sha256": hasher.hexdigest()[:16],
        "system_prompt": SYSTEM_PROMPT,
        "prompt_source": "reflex.data.build_context(...).text  (same as the unconstrained arm)",
        "target_format": "one 'ACT | delexicalized sentence' line per skeleton position",
        "gold_source": "outputs/compile/labels/train.jsonl (skeleton_id, template_ids) "
                       "joined against outputs/compile/bank/templates.jsonl for text",
        "caveat": ("Gold is compile.act_labeler's output (DECISIONS D15) -- a score here is "
                  "agreement with the labeller, not correctness against the human agent. "
                  "Ships with every reported number from this dataset, same as any H5/H7 "
                  "number."),
        "skipped_no_labels": getattr(_iter_examples, "skipped_no_labels", None),
        "skipped_no_template_text": getattr(_iter_examples, "skipped_no_template_text", None),
        "compare_against": ("probes H5 (skeleton top-1), H7 (per-act template top-1/recall@k), "
                           "compose@1 -- same label space, same population (retrieve_utterance "
                           "turns), directly comparable."),
    }
    with open(os.path.join(args.out, stem + ".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)

    print(json.dumps({"wrote": path, "n_examples": n,
                      "sha": sidecar["content_sha256"],
                      "skipped_no_labels": sidecar["skipped_no_labels"],
                      "skipped_no_template_text": sidecar["skipped_no_template_text"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
