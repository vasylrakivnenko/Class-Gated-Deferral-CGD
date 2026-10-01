"""
Ask a yes/no question about a document. The router (router/harness.py)
answers with one of our free classifiers when the question is one we cover,
and with the LLM otherwise. Build the classifiers first with
build_router_bank.py.

Usage:
    /Users/vasyl/zadumai/.venv/bin/python /Users/vasyl/zadumai/legalbench_map/ask.py \\
        "Can we inspect their books?" --doc contract.txt
    ... --text "Licensee may audit Licensor's records..."   (document inline)
    ... --llm kev    (the local Kev reads the document; only the question goes to Jev)
    ... --json
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from router.bank import Bank
from router.harness import Harness
from router.systemone import SystemOne


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--doc", type=Path, help="text file holding the document")
    src.add_argument("--text", help="the document itself")
    p.add_argument("--llm", choices=["jev", "kev"], default="jev",
                   help="who reads the document when no classifier can answer (default: jev)")
    p.add_argument("--json", action="store_true", help="print the full answer as JSON")
    args = p.parse_args()

    document = args.doc.read_text() if args.doc else args.text
    router = SystemOne.jev()
    llm = router if args.llm == "jev" else SystemOne.kev()
    a = Harness(router, llm, Bank(), pretier0=True).answer(args.question, document)

    if args.json:
        print(json.dumps(a.to_dict(), indent=1))
        return
    print(f"Answer:       {a.answer.upper()}  (confidence {a.confidence:.2f})")
    print(f"Answered by:  {a.answered_by}")
    print(f"Why:          {a.path}: {a.reason}")
    print(f"Question answered: {a.asked}")
    if a.evidence:
        unit = "units" if a.n_units != 1 else "unit"
        print(f"Closest {min(len(a.evidence), a.n_units)} of {a.n_units} {unit}:")
        for e in a.evidence:
            print(f"  {e['p']:.2f} {e['label']}: " + textwrap.shorten(e["text"], 110))
    print(f"LLM calls:    {a.llm_calls}")


if __name__ == "__main__":
    main()
