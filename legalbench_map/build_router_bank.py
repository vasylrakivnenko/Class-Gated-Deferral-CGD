"""
Fit and save the router's free classifiers into router/bank/ (gitignored),
plus manifest.json with every menu task's policy and CV score. Takes a few
minutes; rerun whenever results/tasks.csv changes.

Usage:
    /Users/vasyl/zadumai/.venv/bin/python /Users/vasyl/zadumai/legalbench_map/build_router_bank.py
    ... --tasks cuad_audit_rights,overruling    (refit only these; the manifest keeps only them)
"""
from __future__ import annotations

import argparse
import logging

from router.bank import build


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="", help="comma-separated task ids (default: all registered tasks)")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    build([t for t in args.tasks.split(",") if t] or None)


if __name__ == "__main__":
    main()
