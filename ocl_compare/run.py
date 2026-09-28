"""The only runner. `python ocl_compare/run.py [--datasets ...]`

    run.py                          every registered dataset
    run.py --datasets banking77     one of them
    run.py --train-on llm           OCL's no-gold setting, where a teacher exists

It resolves names through the registry in `adapters.py`, hands each `Dataset` to
`downshift.engine.analyse`, and writes one artifact. It contains no knowledge of
any dataset, and the engine contains no knowledge of any source -- which is the
whole point: a new dataset is a function in `adapters.py`, not an edit here or
in the pipeline.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import _provenance as P                                   # noqa: E402
import adapters                                           # noqa: E402,F401  (registers)
from downshift import engine as E                         # noqa: E402
from downshift.dataset import available, load             # noqa: E402

E.EMB_CACHE = os.path.join(P.CACHE, "emb")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help=f"registered: {', '.join(available())}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", nargs="*", default=(),
                    help="candidate names to remove, for ablations")
    ap.add_argument("--with-reranker", action="store_true")
    ap.add_argument("--train-on", choices=("gold", "llm"), default="gold")
    ap.add_argument("--out", default="pipeline", help="artifact basename")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    a = ap.parse_args()

    if a.list:
        for n in available():
            print(n)
        return 0

    names = a.datasets or available()
    unknown = [n for n in names if n not in available()]
    if unknown:
        print(f"unknown dataset(s) {unknown}; registered: {', '.join(available())}",
              file=sys.stderr)
        return 2

    t0 = time.time()
    with P.Tee(os.path.join(P.RESULTS, a.out + ".log")):
        out = []
        for n in names:
            ds = load(n)
            if a.train_on == "llm" and not ds.has_expert:
                print(f"\nskipping {n}: --train-on llm needs expert predictions "
                      f"and this dataset carries none", flush=True)
                continue
            print(f"\nloaded {ds.name}: {ds.n:,} items, {ds.n_classes} classes, "
                  f"expert={'yes' if ds.has_expert else 'no'}\n  source: {ds.source}",
                  flush=True)
            r = E.analyse(ds, seed=a.seed, drop=tuple(a.drop),
                          with_reranker=a.with_reranker, train_on=a.train_on)
            print(E.render(r), flush=True)
            out.append(r)
        P.emit(a.out, {"what": "downshift classification pipeline",
                       "seed": a.seed, "train_on": a.train_on,
                       "dropped_candidates": list(a.drop),
                       "datasets": [r["dataset"] for r in out],
                       "tasks": out})
        print(f"\nwrote {os.path.join(P.RESULTS, a.out + '.json')} "
              f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
