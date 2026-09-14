#!/usr/bin/env python
"""Backfill `n_calls` and `cost_per_1k_items` into result files written before them.

Every `runs/*/results.json` row on disk carries `cost_per_1k_calls`, which is the
cost of one *billed LM call*. Both charts label that number "cost per 1,000
classifications", and those are not the same thing: DSPy's `ChatAdapter` re-sends
an item through `JSONAdapter` when its own field format fails to parse, so one
classification can be two billed calls. This adds the per-classification number
next to the per-call one instead of overwriting it -- "what does the bill say per
call" is still a real question, it is just not the one the axis was asking.

The call count is recoverable exactly, without re-running anything:

    n_calls = round(total_in_tokens / mean_in_tokens)

because `mean_in_tokens` was itself computed as `total_in_tokens / n_calls`. The
rounding only undoes float division. Rows with `mean_in_tokens == 0` made no
measurable LM call (majority / tfidf / frozen-embed / finetuned-encoder), and for
those the file no longer holds the true count; they fall back to `n`, which is
the value that makes per-item cost equal per-call cost -- the rule that matters
for those rows, since both are $0.

Idempotent: a row that already has both keys is left byte-identical. Safe to run
after every experiment.

    python migrate_results.py [--dry-run] [runs/...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

# A row whose per-item cost is within this of its per-call cost is a clean run
# (one call per item, give or take a single transient retry). Above it, the two
# numbers are answering different questions and the chart was plotting the wrong
# one, so it gets printed.
REPORT_THRESHOLD = 0.005


def derive_n_calls(row: dict) -> int:
    """Billed LM calls for a stored row, recovered from its own token totals.

    Zero input tokens means no LM was involved at all -- the majority, TF-IDF,
    frozen-embedding and fine-tuned-encoder rows. Those made no billed calls, so
    0 is the honest answer and it is what a fresh `encoders.py` run records.
    Returning `n` here instead would put "251 calls" in the tooltip of a row
    that never touched an API.
    """
    mean_in = row["mean_in_tokens"]
    if mean_in > 0:
        return round(row["total_in_tokens"] / mean_in)
    return 0


def per_item_cost(row: dict, n_calls: int) -> float:
    """Cost per 1,000 classifications: the per-call rate times calls per item."""
    cost_calls = row["cost_per_1k_calls"]
    n = row["n"]
    # No billed calls (or no items) means there is no per-call-to-per-item
    # conversion to make; the per-call figure stands, NaN and 0.0 included.
    if not n_calls or not n:
        return cost_calls
    return cost_calls * n_calls / n


def _same(a, b) -> bool:
    """Equality that counts NaN as unchanged.

    `float("nan") != float("nan")`, so a plain `==` would call an
    unknown-price row dirty on every pass and the script would rewrite the
    same bytes forever while reporting work it did not do.
    """
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


def migrate_row(row: dict) -> tuple[dict, bool]:
    """Return the row with both keys present, plus whether anything changed.

    Rebuilt key-by-key rather than mutated, so the two new keys land next to
    `cost_per_1k_calls` (where `EvalResult.to_dict` writes them) instead of
    after the 250-element prediction arrays.
    """
    n_calls = derive_n_calls(row)
    cost_items = per_item_cost(row, n_calls)
    if (_same(row.get("n_calls"), n_calls)
            and _same(row.get("cost_per_1k_items"), cost_items)):
        return row, False

    out: dict = {}
    for key, value in row.items():
        if key in ("n_calls", "cost_per_1k_items"):
            continue
        out[key] = value
        if key == "cost_per_1k_calls":
            out["n_calls"] = n_calls
            out["cost_per_1k_items"] = cost_items
    if "n_calls" not in out:  # no cost_per_1k_calls key at all: append instead
        out["n_calls"] = n_calls
        out["cost_per_1k_items"] = cost_items
    return out, True


def migrate_file(path: Path, dry_run: bool = False) -> list[dict]:
    payload = json.loads(path.read_text())
    results = payload.get("results")
    if not isinstance(results, dict):
        print(f"{path}: no 'results' object, skipped")
        return []

    changed_rows, notable = 0, []
    for key, row in list(results.items()):
        new_row, changed = migrate_row(row)
        results[key] = new_row
        changed_rows += changed
        cost_calls, cost_items = new_row["cost_per_1k_calls"], new_row["cost_per_1k_items"]
        if cost_calls > 0 and abs(cost_items / cost_calls - 1.0) > REPORT_THRESHOLD:
            notable.append({"task": path.parent.name, "key": key, "n": new_row["n"],
                            "n_calls": new_row["n_calls"], "cost_calls": cost_calls,
                            "cost_items": cost_items, "ratio": cost_items / cost_calls})

    if changed_rows and not dry_run:
        path.write_text(json.dumps(payload, indent=2, default=str))
    verb = "would update" if dry_run else "updated"
    print(f"{path}: {len(results)} rows, {verb} {changed_rows}"
          f"{' (already current)' if not changed_rows else ''}")
    return notable


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    args = [a for a in argv if not a.startswith("--")]
    paths = ([Path(a).resolve() for a in args] if args
             else sorted(RUNS.glob("*/results.json")))
    if not paths:
        print("no results.json found")
        return 1

    notable: list[dict] = []
    for path in paths:
        notable += migrate_file(path, dry_run=dry_run)

    if notable:
        print(f"\nRows where the charted per-call cost understates the real cost "
              f"per classification by >{REPORT_THRESHOLD:.1%}:")
        header = f"  {'task':<20} {'row':<28} {'calls/items':>12} {'$/1k calls':>11} {'$/1k items':>11} {'ratio':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for task in sorted({r["task"] for r in notable}):
            for r in [r for r in notable if r["task"] == task]:
                print(f"  {r['task']:<20} {r['key']:<28} "
                      f"{r['n_calls']:>5} /{r['n']:>5} "
                      f"{r['cost_calls']:>11.5f} {r['cost_items']:>11.5f} "
                      f"{r['ratio']:>7.3f}x")
    else:
        print("\nno row's per-item cost differs from its per-call cost by "
              f">{REPORT_THRESHOLD:.1%}")

    for path in paths:
        print(f"  sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
