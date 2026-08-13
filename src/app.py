"""One entry point for the whole thing.

    python -m src.app list                          # what has been measured
    python -m src.app run --model resnet18_cifar --mock
    python -m src.app run --model resnet18_cifar    # needs a Raspberry Pi
    python -m src.app dashboard --run <run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model_index import known_models
from src.pipeline import STAGE_NAMES, run_pipeline, summarize
from src.runs import RUNS_DIR, create_run, list_runs, load_run, run_id_for
from src.search.version import DEFAULT_POPULATION_SIZE, study_name

DEFAULT_BUDGET_PT = 0.2

PI_TARGET = Path("pi_target.json")


def choose_venue(requested_mock: bool) -> str:
    """Mock unless a device is configured, so a fresh clone runs out of the box."""
    if requested_mock:
        return "mock"
    if not PI_TARGET.exists():
        raise SystemExit(
            f"No {PI_TARGET}, so there is no device to measure on.\n"
            "Either set one up (docs/raspberry-pi_setup.md) or run with --mock."
        )
    return "pi"


def command_run(args: argparse.Namespace) -> None:
    venue = choose_venue(args.mock)
    if args.model not in known_models():
        raise SystemExit(
            f"{args.model!r} is not a known model. Known: {', '.join(known_models())}\n"
            "Import one with: python -m src.import_onnx --onnx <file> --name <name>"
        )

    # Mock run ids carry the suffix so .gitignore can keep simulated results out
    # of the repo without anyone having to remember not to commit them.
    default_id = run_id_for(args.model, args.budget_pt) + ("_mock" if venue == "mock" else "")

    run = create_run(
        model=args.model,
        budget_pt=args.budget_pt,
        trials=args.trials,
        venue=venue,
        study=study_name(args.model, args.budget_pt, venue == "mock", args.population_size),
        run_id=args.run_id or default_id,
        runs_dir=args.runs_dir,
        population_size=args.population_size,
        max_rss_mb=args.max_rss_mb,
    )

    if venue == "mock":
        from src.pipeline_mock import make_executor

        print("SIMULATED RUN. No device is involved and no number below is real.\n")
    else:
        from src.pipeline_device import make_executor

    budget = "budget from the plan" if run.trials is None else f"{run.trials} trials"
    print(f"{run.run_id}  model {run.model}  venue {run.venue}  {budget}")
    outcomes = run_pipeline(run, make_executor(run), tuple(args.only) if args.only else None, args.force)

    print(f"\n{summarize(outcomes)}")
    if run.paths.dashboard.exists():
        print(f"Page: {run.paths.dashboard}")


def command_list(args: argparse.Namespace) -> None:
    runs = list_runs(args.runs_dir)
    if not runs:
        print("No runs yet. Make one with: python -m src.app run --model resnet18_cifar --mock")
        return
    for run in runs:
        page = "page" if run.paths.dashboard.exists() else "no page"
        trials = "auto" if run.trials is None else f"{run.trials:3d}"
        print(f"{run.run_id:34s} {run.venue:5s} {trials:>4s} trials  {run.source:8s}  {page}")


def command_dashboard(args: argparse.Namespace) -> None:
    from src.pipeline import local_command, execute

    run = load_run(args.run, args.runs_dir)
    execute(local_command("dashboard", run))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="src.app", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    # On each subcommand rather than the parent, so `run --runs-dir X` works in
    # the order anyone would actually type it.
    def with_runs_dir(command: argparse.ArgumentParser) -> argparse.ArgumentParser:
        command.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
        return command

    runner = with_runs_dir(sub.add_parser("run", help="run a campaign end to end"))
    runner.add_argument("--model", required=True)
    runner.add_argument("--mock", action="store_true", help="simulate the device")
    runner.add_argument("--trials", type=int, default=None,
                        help="max candidate evaluations; sized from the reduced space if absent")
    runner.add_argument("--budget-pt", type=float, default=DEFAULT_BUDGET_PT)
    runner.add_argument("--max-rss-mb", type=float, default=None, help="reject candidates above it")
    # Only reachable when the reduced space is too large to enumerate, which no
    # bundled model is; an imported one can be.
    runner.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    runner.add_argument("--run-id", default=None)
    runner.add_argument("--only", nargs="+", choices=STAGE_NAMES, help="run just these stages")
    runner.add_argument("--force", action="store_true", help="redo stages already done")
    runner.set_defaults(handler=command_run)

    lister = with_runs_dir(sub.add_parser("list", help="every run in runs/"))
    lister.set_defaults(handler=command_list)

    page = with_runs_dir(sub.add_parser("dashboard", help="rebuild one run's page"))
    page.add_argument("--run", required=True)
    page.set_defaults(handler=command_dashboard)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
