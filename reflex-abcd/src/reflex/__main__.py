"""The single CLI entry point: ``python -m reflex <command>`` (spec 11.2).

Commands, exactly as spec 11.2 lists them::

    reflex inspect                                              # Section 3.3
    reflex compile   --config configs/default.yaml               # -> outputs/compile/
    reflex train     --config ... --seed 1                       # -> model checkpoint
    reflex calibrate --config ... --checkpoint ...               # -> calibration/gate.json
    reflex run       --arm A --model strong --split test_seen     # -> decisions.jsonl
    reflex run       --arm B --model strong --split test_seen --alpha 0.02
    reflex evaluate  --run-id ...                                # -> metrics.json
    reflex report    --run-ids ... --out report.md               # -> the one-page report
    reflex sweep     --experiment E3|E4|E5|E6                    # the spec 7 matrix

``--config`` and ``--set key=value`` are accepted by EVERY subcommand, per spec
10 ("all live in configs/default.yaml and are overridable by --set key=value on
the CLI"). ``--set`` repeats and applies in order.

This module only parses arguments and dispatches. All behaviour lives in the
spec Section 4 modules behind :mod:`reflex.contracts`, so an unimplemented
command exits 3 with a clear "not implemented" message rather than a traceback.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from reflex import __version__
from reflex.config import apply_overrides, load_config

#: Exit codes. 0 ok / 1 user error / 2 argparse / 3 not implemented / 4 refused.
EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_NOT_IMPLEMENTED = 3
EXIT_REFUSED = 4


def build_parser() -> argparse.ArgumentParser:
    """Build the spec 11.2 argument surface."""
    parser = argparse.ArgumentParser(
        prog="reflex",
        description="REFLEXIVE v1 -- skeleton-first response selection with a confidence gate, on ABCD.",
        epilog="Every subcommand accepts --config and repeatable --set key=value (spec 10).",
    )
    parser.add_argument("--version", action="version", version=f"reflex {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default="configs/default.yaml",
        metavar="PATH",
        help="YAML config (spec Section 12). Default: configs/default.yaml",
    )
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key, e.g. --set gate.alpha=0.05. Repeatable.",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    p = sub.add_parser("inspect", parents=[common], help="Dataset verification report (spec 3.3).")
    p.add_argument("--sample", type=int, default=20, metavar="N",
                   help="Random agent utterances to print with their targets. Default: 20")
    p.add_argument("--seed", type=int, default=0, metavar="N", help="Sampling seed. Default: 0")

    p = sub.add_parser("compile", parents=[common], help="Build the BANK from train (spec 6.2) -> outputs/compile/.")
    p.add_argument("--fraction", type=float, default=1.0, metavar="F",
                   help="Learning-curve fraction of train to compile from (spec 3.4 / E4). Default: 1.0")

    p = sub.add_parser("train", parents=[common], help="Train the encoder and heads H1-H7 (spec 6.4).")
    p.add_argument("--seed", type=int, required=True, metavar="N",
                   help="Training seed. Spec 6.4 requires all of train.seeds to be run.")
    p.add_argument("--fraction", type=float, default=1.0, metavar="F",
                   help="Must match the bank's source_fraction. Default: 1.0")
    p.add_argument("--size", choices=["base", "small"], default="base",
                   help="Encoder size; 'small' is ablation E3c. Default: base")

    p = sub.add_parser("calibrate", parents=[common], help="Compute gate thresholds on dev (spec 6.8).")
    p.add_argument("--checkpoint", required=True, metavar="PATH",
                   help="Checkpoint to calibrate. Quantiles are not transferable across checkpoints.")
    p.add_argument("--seed", type=int, required=True, metavar="N", help="That checkpoint's seed.")

    p = sub.add_parser("run", parents=[common], help="Run one arm over one split -> decisions.jsonl.")
    p.add_argument("--arm", choices=["A", "B"], required=True,
                   help="A = LLM only (spec 6.9). B = fast path + escalation.")
    p.add_argument("--model", dest="model_key", choices=["strong", "cheap"], default="strong",
                   help="Which llm.* model id to use. Default: strong")
    p.add_argument("--split", choices=["dev", "test_seen", "test_novel"], required=True,
                   help="Partition to score (spec 3.4).")
    p.add_argument("--seed", type=int, default=1, metavar="N", help="Training seed. Default: 1")
    p.add_argument("--checkpoint", default=None, metavar="PATH", help="Required for --arm B.")
    p.add_argument("--alpha", type=float, default=None, metavar="A",
                   help="Override gate.alpha for this run (spec 7 E5).")
    p.add_argument("--forced-reflex", dest="forced_reflex", action="store_true",
                   help="THE $0 MODE. Arm B only. The gate still runs and its real verdict is "
                        "logged, but an escalated turn is NOT sent to the LLM -- it is recorded "
                        "unanswered. Makes every routing metric (spec 8.1 reflex rate, "
                        "containment, escalation reasons; spec 8.4 novel escalation rate) "
                        "obtainable while llm.enabled is false, which is the only way to measure "
                        "this build today. It CANNOT produce arm-level answer quality or any "
                        "spec 8.7 parity number: nobody answered the escalated turns. "
                        "Equivalent to --set run.forced_reflex=true.")

    p = sub.add_parser("evaluate", parents=[common], help="Score a run -> metrics.json (spec Section 8).")
    p.add_argument("--run-id", dest="run_id", required=True, metavar="ID", help="Run to score.")
    p.add_argument("--baseline-run-id", dest="baseline_run_id", default=None, metavar="ID",
                   help="Arm A run over the same turns, for the spec 8.7 parity statistics.")

    p = sub.add_parser("report", parents=[common], help="Render the one-page report (spec 11.3).")
    p.add_argument("--run-ids", dest="run_ids", nargs="+", required=True, metavar="ID",
                   help="Runs whose metrics.json feed the report.")
    p.add_argument("--out", default="report.md", metavar="PATH", help="Output path. Default: report.md")

    p = sub.add_parser("sweep", parents=[common], help="Run an experiment matrix (spec Section 7).")
    p.add_argument("--experiment", choices=["E3", "E4", "E5", "E6"], required=True,
                   help="E3 ablations / E4 learning curve / E5 coverage-error / E6 growth loop.")

    return parser


def _dispatch(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    """Dispatch one parsed command to the owning module. Imports are lazy."""
    import os

    command = args.command

    if command == "inspect":
        # scripts/inspect_abcd.py is the spec 3.3 deliverable and is standalone,
        # so the CLI shells out to it rather than duplicating its report.
        import runpy

        from reflex.config import repo_root

        script = os.path.join(repo_root(), "scripts", "inspect_abcd.py")
        sys.argv = [
            script,
            "--config", str(cfg.get("_config_path", args.config)),
            "--sample", str(args.sample),
            "--seed", str(args.seed),
        ]
        runpy.run_path(script, run_name="__main__")
        return EXIT_OK

    if command == "compile":
        from reflex.compile import compile_bank, write_act_check, write_bank, write_delex_check

        bank = compile_bank(cfg, fraction=args.fraction)
        written = write_bank(bank, cfg)
        write_delex_check(bank, cfg)
        write_act_check(bank, cfg)
        for name, path in written.items():
            print(f"{name}: {path}")
        return EXIT_OK

    if command == "train":
        from reflex.train import train

        result = train(cfg, seed=args.seed, fraction=args.fraction, size=args.size)
        print(f"checkpoint: {result['checkpoint_path']}")
        return EXIT_OK

    if command == "calibrate":
        from reflex.calibrate import calibrate, write_calibration

        calibration = calibrate(cfg, checkpoint_path=args.checkpoint, seed=args.seed)
        print(f"calibration: {write_calibration(calibration, cfg)}")
        return EXIT_OK

    if command == "run":
        from reflex.run import run_arm

        if args.arm == "B" and not args.checkpoint:
            print("error: --arm B requires --checkpoint", file=sys.stderr)
            return EXIT_USER_ERROR
        if args.forced_reflex:
            if args.arm != "B":
                print("error: --forced-reflex applies to --arm B only", file=sys.stderr)
                return EXIT_USER_ERROR
            # run_arm's signature is frozen by the contract, so the mode travels
            # through cfg -- the same route --set run.forced_reflex=true takes.
            cfg = apply_overrides(cfg, ["run.forced_reflex=true"])
            print(
                "forced-reflex: escalated turns will be recorded UNANSWERED. Routing metrics "
                "are valid; arm-level answer quality and spec 8.7 parity are NOT.",
                file=sys.stderr,
            )
        run_id = run_arm(
            cfg,
            arm=args.arm,
            model_key=args.model_key,
            split=args.split,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            alpha=args.alpha,
        )
        print(f"run_id: {run_id}")
        return EXIT_OK

    if command == "evaluate":
        from reflex.evaluate import evaluate_run

        evaluate_run(cfg, run_id=args.run_id, baseline_run_id=args.baseline_run_id)
        print(f"metrics written for run {args.run_id}")
        return EXIT_OK

    if command == "report":
        from reflex.config import resolve_path
        from reflex.report import render_report

        runs_dir = resolve_path(cfg, "paths.runs_dir")
        metrics_paths = [os.path.join(runs_dir, rid, "metrics.json") for rid in args.run_ids]
        print(f"report: {render_report(metrics_paths, args.out, cfg)}")
        return EXIT_OK

    if command == "sweep":
        from reflex.run import run_sweep

        for run_id in run_sweep(cfg, experiment=args.experiment):
            print(run_id)
        return EXIT_OK

    print(f"error: unknown command {command!r}", file=sys.stderr)  # pragma: no cover
    return EXIT_USER_ERROR


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, load the config, dispatch, and map errors onto exit codes."""
    from reflex.contracts import LLMDisabledError, ReflexError

    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides)
    except (FileNotFoundError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_USER_ERROR

    try:
        return _dispatch(args, cfg)
    except NotImplementedError:
        print(
            f"'reflex {args.command}' is not implemented yet.\n"
            f"Its contract is frozen in src/reflex/contracts.py; implement the owning "
            f"module under src/reflex/ with byte-identical signatures.",
            file=sys.stderr,
        )
        return EXIT_NOT_IMPLEMENTED
    except LLMDisabledError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except ReflexError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USER_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
