"""Build the Pareto dashboard page from the measured result JSONs.

Every figure on the page is extracted here and inlined into the template, which
carries no literal numbers of its own (PLAN.md forbids hand-copied figures).

    python -m scripts.build_dashboard
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from src.quant.baselines import BASELINE_CONFIGS
from src.quant.config import DeploymentConfig, QuantConfig, RunConfig

SEARCH_DIR = Path("artifacts/search")
REPORT_DIR = Path("artifacts/reports")
DEVICE_REPORT_DIR = Path("artifacts/reports_pi")
CACHE_DIR = Path("artifacts/bench_cache")
TEMPLATE = Path("scripts/dashboard_template.html")
DEFAULT_OUT = Path("artifacts/dashboard/index.html")

MODEL = "resnet18_cifar"
STUDY = "resnet18_cifar_budget0.2_v2"
SENTINEL = "sentinel_resnet18_cifar_20260808.json"

# The runtime the global baselines were measured under in Phases 3-4, and what
# the study seeds them with. Reference marks must be read back under it or they
# are not the same measurement the campaign compared against.
REFERENCE_RUN = RunConfig(intra_op_num_threads=4, graph_optimization_level="all")

DATA_PLACEHOLDER = "/*__DASHBOARD_DATA__*/"

# The four objectives, in the order the page lays them out. `sense` is which
# direction is better, which is what the normalizer needs to invert.
OBJECTIVES: tuple[dict[str, str], ...] = (
    {"key": "latency_ms", "label": "Latency", "unit": "ms", "sense": "min"},
    {"key": "size_mb", "label": "Model size", "unit": "MB", "sense": "min"},
    {"key": "peak_rss_mb", "label": "Peak RAM", "unit": "MB", "sense": "min"},
    {"key": "top1", "label": "Top-1", "unit": "%", "sense": "max"},
)

# Slider granularity, shared by the page and by the reachability enumeration
# below. One constant so the "no slider setting selects this row" claim is made
# against the grid the sliders actually offer, not an idealised continuous one.
SLIDER_STEP = 5
SLIDER_MAX = 100


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- config text


def parse_quant_describe(text: str) -> QuantConfig:
    """Rebuild the config behind one of `QuantConfig.describe`'s strings.

    The search summary stores configs only as these labels, and the study's
    sqlite file that holds the raw parameters is gitignored. Parsing the label
    keeps every panel buildable from a fresh clone; `front_rows` asserts the
    round trip, so a format change fails the build rather than mislabelling a
    config on the page.
    """
    if text == "fp32":
        return QuantConfig(quant_type="none")

    tokens = text.split()
    quant_type = tokens[0]
    fields: dict[str, Any] = {"quant_type": quant_type, "per_channel": "per-channel" in tokens}
    for token in tokens[1:]:
        if token.startswith("act:"):
            fields["activation_type"] = token[len("act:") :]
        elif token.startswith("fp32:"):
            fields["excluded_groups"] = tuple(sorted(token[len("fp32:") :].split("+")))
        elif "/" in token:
            method, size = token.split("/")
            fields["calibration_method"] = method
            fields["calibration_size"] = int(size)
    return QuantConfig(**fields)


def parse_run_describe(text: str) -> RunConfig:
    """Rebuild the runtime config behind one of `RunConfig.describe`'s strings."""
    tokens = text.split()
    return RunConfig(
        intra_op_num_threads=int(tokens[0].rstrip("t")),
        graph_optimization_level=tokens[1][len("opt:") :],
        enable_cpu_mem_arena="no-arena" not in tokens,
        allow_intra_op_spinning="no-spin" not in tokens,
    )


# ------------------------------------------------------------------ measured


def measurements_by_config(model: str, cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Every admissible cached Pi measurement, keyed by its *deployment* hash.

    Keying on the quant hash instead is what `sensitivity.cost_benefit` does and
    is no longer safe here: Phase 6 made the runtime knobs searchable, so one
    artifact now has several admissible 4-thread measurements and the quant hash
    silently returns whichever was read last -- which reads a 3.4 ms baseline
    back as 24 ms. The deployment hash covers the runtime too.
    """
    found: dict[str, dict[str, Any]] = {}
    for path in sorted((cache_dir / model).glob("*.json")):
        result = read_json(path)
        if result.get("status") != "ok" or not result.get("admissible"):
            continue
        found.setdefault(result["config_hash"], result)
    return found


def reference_marks(
    quant_baselines: dict[str, Any],
    measured: dict[str, dict[str, Any]],
    front: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The four global baselines, pinned on the scatter as the bar to beat.

    A baseline can also *be* a front member -- the search seeds all four, and
    global per-channel INT8 wins latency outright. `same_as` records that by
    artifact hash rather than by comparing plotted values, so the page can say
    so instead of drawing two marks on one spot.
    """
    marks: list[dict[str, Any]] = []
    for name, config in BASELINE_CONFIGS.items():
        baseline = quant_baselines["baselines"].get(name, {})
        bench = measured.get(DeploymentConfig(quant=config, run=REFERENCE_RUN).hash)
        if baseline.get("status") != "ok" or bench is None:
            continue
        twin = next(
            (
                row["id"]
                for row in front
                if row["quant_hash"] == config.hash
                and row["describe_run"] == REFERENCE_RUN.describe()
            ),
            None,
        )
        marks.append(
            {
                "name": name,
                "describe": config.describe(),
                "latency_ms": bench["latency"]["median_ms"],
                "size_mb": baseline["bytes"] / 1e6,
                "peak_rss_mb": bench["peak_rss_mb"],
                "top1": baseline["accuracy"]["top1"],
                "same_as": twin,
            }
        )
    return marks


def noise_floor(sentinel: dict[str, Any]) -> dict[str, float]:
    """Repeat-to-repeat spread of one identical config, as a percentage."""
    rss = [result["peak_rss_mb"] for result in sentinel["results"]]
    return {
        "trials": sentinel["admissible_trials"],
        "latency_percent": sentinel["spread_percent"],
        "rss_percent": round((max(rss) - min(rss)) / (sum(rss) / len(rss)) * 100, 4),
        "size_percent": 0.0,
    }


# ---------------------------------------------------------------- front rows


def front_rows(summary: dict[str, Any], baseline_top1: float, samples: int) -> list[dict[str, Any]]:
    """One row per Pareto member, ordered by latency, with its config unpacked."""
    rows: list[dict[str, Any]] = []
    for member in summary["pareto"]:
        result = member["result"]
        if result["status"] != "ok":
            continue
        quant = parse_quant_describe(member["describe"])
        run = parse_run_describe(member["describe_run"])
        # A silent parse slip would relabel a config on the page, so make it loud.
        assert quant.describe() == member["describe"], member["describe"]
        assert run.describe() == member["describe_run"], member["describe_run"]

        rows.append(
            {
                "describe": member["describe"],
                "describe_run": member["describe_run"],
                "quant_hash": member["quant_hash"],
                "quant_type": quant.quant_type,
                "per_channel": quant.per_channel,
                "activation_type": quant.activation_type if quant.quant_type == "static" else None,
                "calibration": (
                    f"{quant.calibration_method}/{quant.calibration_size}"
                    if quant.quant_type == "static"
                    else None
                ),
                "excluded_groups": list(quant.excluded_groups),
                "run": run.as_dict(),
                "latency_ms": result["latency_ms"],
                "size_mb": result["size_bytes"] / 1e6,
                "size_bytes": result["size_bytes"],
                "peak_rss_mb": result["peak_rss_mb"],
                "top1": result["top1"],
                "screen_top1": result["screen_top1"],
                # The accuracy gap in images rather than points: 0.27 pt of a
                # 3,000-image split is 8 pictures, which is the honest scale.
                "images_vs_baseline": round((result["top1"] - baseline_top1) / 100 * samples),
            }
        )
    rows.sort(key=lambda row: row["latency_ms"])
    for index, row in enumerate(rows):
        row["id"] = f"C{index + 1}"
    return rows


def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Min-max each objective across the front, oriented so 1.0 is always best.

    A single-valued objective normalizes to 1.0 for every row: it separates
    nothing, so it must not push the ranking around either.
    """
    for row in rows:
        row["norm"] = {}
    for objective in OBJECTIVES:
        key = objective["key"]
        values = [row[key] for row in rows]
        low, high = min(values), max(values)
        span = high - low
        for row in rows:
            if span == 0:
                row["norm"][key] = 1.0
                continue
            fraction = (row[key] - low) / span
            row["norm"][key] = fraction if objective["sense"] == "max" else 1.0 - fraction
    return rows


def score(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[float]:
    """Weighted sum of the normalized objectives, in row order.

    The page's sliders recompute exactly this line in JavaScript; keeping the
    normalization here and shipping `norm` with the data leaves only the sum
    itself duplicated.
    """
    return [
        sum(weights.get(objective["key"], 0.0) * row["norm"][objective["key"]] for objective in OBJECTIVES)
        for row in rows
    ]


def rank(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[int]:
    """Row indices, best first, under one weighting."""
    scores = score(rows, weights)
    return sorted(range(len(rows)), key=lambda index: -scores[index])


def unreachable_rows(rows: list[dict[str, Any]], step: int = SLIDER_STEP) -> list[str]:
    """Rows that no slider setting can lift to rank 1.

    A weighted sum over fixed points can only ever select members of their
    convex hull, so a front member can be genuinely optimal and still never win
    a slider contest. Enumerating the slider grid rather than solving the hull
    answers the question the reader actually has -- can I get this row by moving
    these controls -- and needs no solver.
    """
    keys = [objective["key"] for objective in OBJECTIVES]
    columns = [[row["norm"][key] for row in rows] for key in keys]
    reachable: set[int] = set()

    for weights in product(range(0, SLIDER_MAX + 1, step), repeat=len(keys)):
        if not any(weights):
            continue  # every row scores zero; the page forbids this setting too
        scores = [
            sum(weight * column[index] for weight, column in zip(weights, columns, strict=True))
            for index in range(len(rows))
        ]
        best = max(scores)
        reachable.update(
            index for index, value in enumerate(scores) if value >= best - 1e-12
        )
        if len(reachable) == len(rows):
            break
    return [row["id"] for index, row in enumerate(rows) if index not in reachable]


# ------------------------------------------------------- sensitivity vs cost


def cost_benefit_rows(cost_benefit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-group benefit and price, carrying both rankings so the gap is visible."""
    by_share = sorted(cost_benefit, key=lambda row: -row["recovery_share"])
    efficiency_order = [row["group"] for row in cost_benefit]  # already share-per-ms sorted

    rows: list[dict[str, Any]] = []
    for kl_rank, row in enumerate(by_share, start=1):
        rows.append(
            {
                "group": row["group"],
                "recovery_share": row["recovery_share"],
                "cost_ms": row["cost_ms"],
                "share_per_ms": row["share_per_ms"],
                "kl_rank": kl_rank,
                "efficiency_rank": efficiency_order.index(row["group"]) + 1,
            }
        )
    return rows


# ------------------------------------------------------------ trial history


def attach_test_scores(front: list[dict[str, Any]], path: Path) -> dict[str, Any] | None:
    """Fold the sealed test-set scores onto the front, if they have been run.

    Display only, and never an objective: the test set is scored once, after the
    search has already chosen, so letting it into the ranking would be exactly
    the leak the split exists to prevent. Absent file, the page simply says the
    evaluation has not run.
    """
    if not path.exists():
        return None

    report = read_json(path)
    by_label = {result["label"]: result for result in report["results"] if result["status"] == "ok"}
    for row in front:
        scored = by_label.get(f"{row['describe']} | {row['describe_run']}")
        row["test_top1"] = scored["test"]["top1"] if scored else None

    reference = by_label.get("fp32 (reference)")
    return {
        "split": report["eval_split"],
        "samples": next(iter(by_label.values()))["test"]["samples"],
        "fp32_top1": reference["test"]["top1"] if reference else None,
        "scored": sum(1 for row in front if row["test_top1"] is not None),
    }


def trial_rows(db_path: Path, study_name: str, front: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every measured trial from the study, or an empty list if it is not here.

    The sqlite study is gitignored -- it is rewritten on every resume and the
    JSON summary beside it carries the results -- so a fresh clone builds every
    other panel and simply omits this one.
    """
    if not db_path.exists():
        return []

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path.as_posix()}")
    on_front = {(row["quant_hash"], row["describe_run"]) for row in front}

    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        result = trial.user_attrs.get("result")
        if not result:
            continue
        rows.append(
            {
                "number": trial.number,
                "status": result["status"],
                "top1": result["top1"],
                "screen_top1": result["screen_top1"],
                "latency_ms": result["latency_ms"],
                "describe": trial.user_attrs.get("describe", ""),
                "on_front": (
                    trial.user_attrs.get("quant_hash"),
                    trial.user_attrs.get("describe_run"),
                )
                in on_front,
            }
        )
    return rows


# ------------------------------------------------------------------- payload


def build_payload(
    search_dir: Path, report_dir: Path, device_dir: Path, cache_dir: Path
) -> dict[str, Any]:
    summary = read_json(search_dir / f"{STUDY}.json")
    baseline = read_json(report_dir / f"{MODEL}_baseline.json")
    quant_baselines = read_json(device_dir / f"{MODEL}_quant_baselines.json")
    sensitivity = read_json(device_dir / f"{MODEL}_sensitivity.json")
    cost_benefit = read_json(device_dir / "cost_benefit.json")[MODEL]
    sentinel = read_json(device_dir / SENTINEL)

    samples = quant_baselines["baselines"]["fp32"]["accuracy"]["samples"]
    rows = normalize(front_rows(summary, summary["baseline_top1"], samples))
    test = attach_test_scores(rows, device_dir / f"{MODEL}_final_test.json")

    return {
        "model": MODEL,
        "study": summary["study"],
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": {
            "name": quant_baselines["eval_split"],
            "samples": samples,
            "fingerprint": sensitivity["split_fingerprint"],
        },
        "device": {
            "platform": quant_baselines["host"]["platform"],
            "onnxruntime": quant_baselines["onnxruntime"],
            "threads": REFERENCE_RUN.intra_op_num_threads,
        },
        "budget": {
            "baseline_top1": summary["baseline_top1"],
            "budget_pt": summary["accuracy_budget_pt"],
            "threshold_top1": summary["threshold_top1"],
        },
        "objectives": list(OBJECTIVES),
        "slider": {"step": SLIDER_STEP, "max": SLIDER_MAX},
        "front": rows,
        "test": test,
        "unreachable": unreachable_rows(rows),
        "references": reference_marks(
            quant_baselines, measurements_by_config(MODEL, cache_dir), rows
        ),
        "noise": noise_floor(sentinel),
        "cost_benefit": cost_benefit_rows(cost_benefit),
        "groups": summary["groups"],
        "trials": trial_rows(search_dir / f"{STUDY}.db", STUDY, rows),
        "trials_declared": summary["trials"],
        "provenance": {
            "checkpoint_sha256": baseline["checkpoint"]["sha256"][:16],
            "onnx_sha256": baseline["onnx"]["sha256"][:16],
            "torch": baseline["package_versions"]["torch"],
        },
    }


def render(payload: dict[str, Any], template: Path) -> str:
    text = template.read_text(encoding="utf-8")
    if DATA_PLACEHOLDER not in text:
        raise SystemExit(f"{template} has no {DATA_PLACEHOLDER} to fill")
    return text.replace(DATA_PLACEHOLDER, json.dumps(payload, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--search-dir", type=Path, default=SEARCH_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--device-dir", type=Path, default=DEVICE_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = build_payload(args.search_dir, args.report_dir, args.device_dir, args.cache_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(payload, args.template), encoding="utf-8")

    print(f"{payload['study']}  split {payload['split']['name']} ({payload['split']['samples']})")
    print(f"  {'id':4s}{'latency':>10s}{'size MB':>9s}{'RSS MB':>9s}{'top-1':>8s}  config")
    for row in payload["front"]:
        print(
            f"  {row['id']:4s}{row['latency_ms']:10.4f}{row['size_mb']:9.3f}"
            f"{row['peak_rss_mb']:9.3f}{row['top1']:8.3f}  {row['describe']} | {row['describe_run']}"
        )
    print(f"\n  unreachable by any slider setting: {', '.join(payload['unreachable']) or 'none'}")
    print(f"  trial history: {len(payload['trials'])} of {payload['trials_declared']} trials")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
