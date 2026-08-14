"""Build the Pareto dashboard page from the measured result JSONs.

Every figure on the page is extracted here and inlined into the template, which
carries no literal numbers of its own (PLAN.md forbids hand-copied figures).

    python -m scripts.build_dashboard
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import groupby, permutations
from pathlib import Path
from typing import Any

from src.quant.baselines import BASELINE_CONFIGS
from src.quant.config import DeploymentConfig, QuantConfig, RunConfig
from src.runs import RUNS_DIR, Run, load_run
from src.search.version import SPACE_VERSION, study_name

SEARCH_DIR = Path("artifacts/search")
REPORT_DIR = Path("artifacts/reports")
DEVICE_REPORT_DIR = Path("artifacts/reports_pi")
CACHE_DIR = Path("artifacts/bench_cache")
TEMPLATE = Path("scripts/dashboard_template.html")
DEFAULT_OUT = Path("artifacts/dashboard/index.html")

DEFAULT_MODEL = "resnet18_cifar"

DEFAULT_BUDGET_PT = 0.2

# The runtime the global baselines were measured under in Phases 3-4, and what
# the study seeds them with. Reference marks must be read back under it or they
# are not the same measurement the campaign compared against.
REFERENCE_RUN = RunConfig(intra_op_num_threads=4, graph_optimization_level="all")

DATA_PLACEHOLDER = "/*__DASHBOARD_DATA__*/"

# Pi 5 firmware soft-limits the clock at 80 C. The panel draws the axis up to it
# so the distance between the measurements and the limit is the visible claim.
THROTTLE_SOFT_C = 80.0

# The four objectives, in the order the page lays them out. `sense` is which
# direction is better; `decimals` is how many digits the page prints, which is
# also the finest difference it is allowed to rank on.
OBJECTIVES: tuple[dict[str, Any], ...] = (
    {"key": "latency_ms", "label": "Latency", "unit": "ms", "sense": "min", "decimals": 3,
     "blurb": "Median time to classify one image"},
    {"key": "size_mb", "label": "Model size", "unit": "MB", "sense": "min", "decimals": 2,
     "blurb": "Size of the .onnx file on disk"},
    {"key": "peak_rss_mb", "label": "Peak RAM", "unit": "MB", "sense": "min", "decimals": 1,
     "blurb": "Most memory the process held while running"},
    {"key": "top1", "label": "Top-1", "unit": "%", "sense": "max", "decimals": 2,
     "blurb": "Share of images classified correctly"},
)

# Where the reader's priority list starts before they touch it.
DEFAULT_PRIORITY = ("latency_ms", "top1", "peak_rss_mb", "size_mb")

# Display names for the models trained here. An imported model has no entry and
# falls back to its own slug, which is what the importer was told to call it.
MODEL_LABELS = {
    "resnet18_cifar": "ResNet-18",
    "mobilenetv2_cifar": "MobileNetV2",
    "custom_cnn": "Custom CNN",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Sources:
    """Every file one page is built from, named individually.

    Two layouts produce the same page: a run directory, where stages are named
    by what they are, and the original per-phase directories, where they carry
    the model name. Naming each file here is what lets both work without the
    payload builder knowing which layout it is reading.
    """

    model: str
    study: str
    summary: Path
    baseline: Path
    quant_baselines: Path
    sensitivity: Path
    cost_benefit: Path
    cache_dir: Path
    final_test: Path | None = None
    sentinel: Path | None = None
    study_db: Path | None = None
    venue: str = "pi"

    @classmethod
    def legacy(
        cls,
        model: str,
        study: str,
        search_dir: Path,
        report_dir: Path,
        device_dir: Path,
        cache_dir: Path,
        sentinel: Path | None = None,
    ) -> Sources:
        final_test = device_dir / f"{model}_final_test.json"
        return cls(
            model=model,
            study=study,
            summary=search_dir / f"{study}.json",
            baseline=report_dir / f"{model}_baseline.json",
            quant_baselines=device_dir / f"{model}_quant_baselines.json",
            sensitivity=device_dir / f"{model}_sensitivity.json",
            cost_benefit=device_dir / "cost_benefit.json",
            cache_dir=cache_dir,
            final_test=final_test if final_test.exists() else None,
            sentinel=sentinel if sentinel is None or sentinel.exists() else None,
            study_db=search_dir / f"{study}.db",
        )

    @classmethod
    def from_run(cls, run: Run, cache_dir: Path, search_dir: Path) -> Sources:
        stage = run.paths.stage
        optional = {name: stage(name) for name in ("final_test", "sentinel")}
        return cls(
            model=run.model,
            study=run.study,
            summary=stage("study"),
            baseline=stage("baseline"),
            quant_baselines=stage("quant_baselines"),
            sensitivity=stage("sensitivity"),
            cost_benefit=stage("cost_benefit"),
            cache_dir=cache_dir,
            final_test=optional["final_test"] if optional["final_test"].exists() else None,
            sentinel=optional["sentinel"] if optional["sentinel"].exists() else None,
            # The sqlite study stays where optuna wrote it: it is gitignored and
            # rewritten on every resume, so a run copies the summary, not the db.
            study_db=search_dir / f"{run.study}.db",
            venue=run.venue,
        )


def configs_for(member: dict[str, Any]) -> tuple[QuantConfig, RunConfig]:
    """This front member's configs, read back rather than reconstructed.

    A study/2 summary stores them in full. Only the pre-study/2 campaigns need
    the label parsers, and those are measured history that still has to render.
    """
    stored = member.get("config")
    if stored:
        config = DeploymentConfig.from_dict(stored)
        return config.quant, config.run

    quant = parse_quant_describe(member["describe"])
    run = parse_run_describe(member["describe_run"])
    # A silent parse slip would relabel a config on the page, so make it loud.
    assert quant.describe() == member["describe"], member["describe"]
    assert run.describe() == member["describe_run"], member["describe_run"]
    return quant, run


def parse_quant_describe(text: str) -> QuantConfig:
    """Rebuild the config behind one of `QuantConfig.describe`'s strings.

    Only reached for a campaign measured before the summary carried full
    configs. `configs_for` asserts the round trip, so a format change fails the
    build rather than mislabelling a config on the page.
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


def thermal_rows(model: str, cache_dir: Path) -> dict[str, Any] | None:
    """Every cached measurement's thermal state, in the order the device took them.

    Comparing two latencies only means something if the device was in the same
    state for both, so the page has to show that state rather than assert it.
    Reads the whole cache, not just this run's front: a candidate that was built
    and rejected still cost a measurement, and its temperature is evidence too.
    """
    samples: list[dict[str, Any]] = []
    for path in sorted((cache_dir / model).glob("*.json")):
        result = read_json(path)
        before = result.get("device_before") or {}
        after = result.get("device_after") or {}
        if result.get("status") != "ok" or before.get("temperature_c") is None:
            continue
        throttle = (after.get("throttle") or {}).get("sticky") or []
        samples.append(
            {
                "at": result["created_at_utc"],
                "before_c": before["temperature_c"],
                "after_c": after.get("temperature_c", before["temperature_c"]),
                "clock_mhz": before.get("clock_mhz"),
                "pinned": bool(before.get("governor_is_pinned")),
                "throttled": bool(throttle or result.get("throttle_events")),
                "stable": bool((result.get("latency") or {}).get("stable")),
            }
        )
    if not samples:
        return None

    samples.sort(key=lambda sample: sample["at"])
    peaks = [sample["after_c"] for sample in samples]
    return {
        "soft_limit_c": THROTTLE_SOFT_C,
        "samples": samples,
        "count": len(samples),
        "min_c": min(sample["before_c"] for sample in samples),
        "max_c": max(peaks),
        "median_rise_c": round(
            sorted(sample["after_c"] - sample["before_c"] for sample in samples)[len(samples) // 2], 2
        ),
        "throttled": sum(1 for sample in samples if sample["throttled"]),
        "unpinned": sum(1 for sample in samples if not sample["pinned"]),
        "unstable": sum(1 for sample in samples if not sample["stable"]),
    }


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


def find_sentinel(device_dir: Path, model: str) -> Path | None:
    """The most recent repeat-spread report for this model, if one was ever run.

    Only resnet18 has one; the other two campaigns never repeated a config. The
    build degrades rather than borrowing another model's figure, because a spread
    measured on a different graph is an estimate, not a measurement.
    """
    found = sorted(device_dir.glob(f"sentinel_{model}_*.json"))
    return found[-1] if found else None


def unmeasured_noise() -> dict[str, Any]:
    """The noise floor for a run that never repeated a config.

    Zero tolerance is not a claim that the device is perfectly repeatable; it is
    the page refusing to invent one. Ties then fall back to printed digits alone,
    which understates how many rows are really indistinguishable, and the page
    says so.
    """
    return {
        "measured": False,
        "trials": 0,
        "latency_percent": 0.0,
        "rss_percent": 0.0,
        "percent": {objective["key"]: 0.0 for objective in OBJECTIVES},
    }


def noise_floor(sentinel: dict[str, Any]) -> dict[str, Any]:
    """Repeat-to-repeat spread of one identical config, as a percentage.

    Size and top-1 are zero rather than unmeasured: the same artifact is the same
    bytes, and rescoring a fixed split with a deterministic graph returns the same
    number. Their only resolution limit is how many digits the page prints.
    """
    rss = [result["peak_rss_mb"] for result in sentinel["results"]]
    rss_percent = round((max(rss) - min(rss)) / (sum(rss) / len(rss)) * 100, 4)
    return {
        "measured": True,
        "trials": sentinel["admissible_trials"],
        "latency_percent": sentinel["spread_percent"],
        "rss_percent": rss_percent,
        "percent": {
            "latency_ms": sentinel["spread_percent"],
            "size_mb": 0.0,
            "peak_rss_mb": rss_percent,
            "top1": 0.0,
        },
    }


def front_rows(summary: dict[str, Any], baseline_top1: float, samples: int) -> list[dict[str, Any]]:
    """One row per Pareto member, ordered by latency, with its config unpacked."""
    rows: list[dict[str, Any]] = []
    for member in summary["pareto"]:
        result = member["result"]
        if result["status"] != "ok":
            continue
        quant, run = configs_for(member)
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


def tie_groups(rows: list[dict[str, Any]], noise: dict[str, float]) -> dict[str, list[list[str]]]:
    """Per objective, the rows the page cannot honestly tell apart.

    Two configs are indistinguishable on an objective when they print the same
    digits or sit inside its measured repeat noise. Grouping chains from each
    group's leader rather than comparing every pair, because a tolerance applied
    pairwise is not transitive -- a~b and b~c without a~c -- and sorting on a
    non-transitive comparison gives an order that depends on which comparisons
    the sort happened to make.
    """
    groups: dict[str, list[list[str]]] = {}
    for objective in OBJECTIVES:
        key, decimals = objective["key"], objective["decimals"]
        tolerance = noise[key] / 100
        ordered = sorted(rows, key=lambda row: row[key], reverse=objective["sense"] == "max")

        buckets: list[list[str]] = []
        leaders: list[float] = []
        for row in ordered:
            value = row[key]
            same_digits = bool(buckets) and f"{leaders[-1]:.{decimals}f}" == f"{value:.{decimals}f}"
            inside_noise = bool(buckets) and abs(value - leaders[-1]) <= abs(leaders[-1]) * tolerance
            if same_digits or inside_noise:
                buckets[-1].append(row["id"])
            else:
                buckets.append([row["id"]])
                leaders.append(value)
        groups[key] = buckets
    return groups


def group_index(groups: dict[str, list[list[str]]]) -> dict[str, dict[str, int]]:
    """Objective -> row id -> its place in that objective's ordered groups."""
    return {
        key: {row_id: place for place, bucket in enumerate(buckets) for row_id in bucket}
        for key, buckets in groups.items()
    }


def lexicographic(
    rows: list[dict[str, Any]], order: tuple[str, ...], index: dict[str, dict[str, int]]
) -> list[dict[str, Any]]:
    """Rank under one priority order, splitting each tie on the next priority."""

    def resolve(subset: list[dict[str, Any]], depth: int) -> list[dict[str, Any]]:
        # Out of priorities: hold the incoming order, which is latency-sorted.
        if len(subset) < 2 or depth >= len(order):
            return subset
        place = index[order[depth]]
        ranked = sorted(subset, key=lambda row: place[row["id"]])
        out: list[dict[str, Any]] = []
        for _, bucket in groupby(ranked, key=lambda row: place[row["id"]]):
            out.extend(resolve(list(bucket), depth + 1))
        return out

    return resolve(rows, 0)


def orderings(
    rows: list[dict[str, Any]], groups: dict[str, list[list[str]]]
) -> dict[str, list[dict[str, Any]]]:
    """Every priority order the reader can build, resolved here rather than on the page.

    Four objectives is 24 permutations, so shipping all of them costs a few
    kilobytes and leaves the page with a lookup instead of a second copy of the
    ranking rule that could drift from this one.
    """
    index = group_index(groups)
    resolved: dict[str, list[dict[str, Any]]] = {}
    for order in permutations(objective["key"] for objective in OBJECTIVES):
        ranked = lexicographic(rows, order, index)
        # Which priority separated each row from the one above it -- the page
        # shows it per row, so a tie-break never looks like an arbitrary choice.
        entries = [{"id": ranked[0]["id"], "decided_by": None}]
        for above, row in zip(ranked, ranked[1:], strict=False):
            entries.append(
                {
                    "id": row["id"],
                    "decided_by": next(
                        (key for key in order if index[key][above["id"]] != index[key][row["id"]]),
                        None,
                    ),
                }
            )
        resolved[",".join(order)] = entries
    return resolved


def unreachable_rows(resolved: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]]) -> list[str]:
    """Rows that no priority order puts first.

    A lexicographic ranking always starts from a best, so a config that is never
    the best -- nor tied for best and then ahead on the next priority -- cannot
    lead any ordering. It can still be genuinely un-dominated, which is why it is
    on the front at all, so the page marks it rather than dropping it.
    """
    leaders = {entries[0]["id"] for entries in resolved.values()}
    return [row["id"] for row in rows if row["id"] not in leaders]


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


def trial_rows(
    summary: dict[str, Any], db_path: Path, study: str, front: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every evaluated candidate, from the summary when it carries them.

    A `study/2` summary records each evaluation in full and is committed, so the
    panel survives a fresh clone. Only the pre-`study/2` campaigns fall back to
    the gitignored sqlite file, and there the panel is simply dropped when it is
    not present.
    """
    evaluated = summary.get("evaluated_configs")
    if evaluated is not None:
        on_front = {(row["quant_hash"], row["describe_run"]) for row in front}
        return [
            {
                "number": number,
                "status": row["result"]["status"],
                "top1": row["result"]["top1"],
                "screen_top1": row["result"]["screen_top1"],
                "latency_ms": row["result"]["latency_ms"],
                "describe": row["describe"],
                "round": row.get("round"),
                "on_front": (row["quant_hash"], row["describe_run"]) in on_front,
            }
            for number, row in enumerate(evaluated)
        ]

    if not db_path.exists():
        return []

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    loaded = optuna.load_study(study_name=study, storage=f"sqlite:///{db_path.as_posix()}")
    on_front = {(row["quant_hash"], row["describe_run"]) for row in front}

    rows: list[dict[str, Any]] = []
    for trial in loaded.trials:
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


def build_payload(sources: Sources) -> dict[str, Any]:
    model = sources.model
    summary = read_json(sources.summary)
    baseline = read_json(sources.baseline)
    quant_baselines = read_json(sources.quant_baselines)
    sensitivity = read_json(sources.sensitivity)

    cost_benefit = read_json(sources.cost_benefit)
    # The legacy file holds every model at once; a run's copy holds only its own.
    if isinstance(cost_benefit, dict):
        cost_benefit = cost_benefit[model]

    samples = quant_baselines["baselines"]["fp32"]["accuracy"]["samples"]
    rows = front_rows(summary, summary["baseline_top1"], samples)
    test = attach_test_scores(rows, sources.final_test) if sources.final_test else None
    noise = noise_floor(read_json(sources.sentinel)) if sources.sentinel else unmeasured_noise()
    resolved = orderings(rows, tie_groups(rows, noise["percent"]))

    return {
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "venue": sources.venue,
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
        "default_priority": list(DEFAULT_PRIORITY),
        "front": rows,
        "test": test,
        "orderings": resolved,
        "unreachable": unreachable_rows(resolved, rows),
        "references": reference_marks(
            quant_baselines, measurements_by_config(model, sources.cache_dir), rows
        ),
        "noise": noise,
        "thermal": thermal_rows(model, sources.cache_dir),
        "cost_benefit": cost_benefit_rows(cost_benefit),
        "groups": summary["groups"],
        "trials": trial_rows(summary, sources.study_db, sources.study, rows),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=None, help="run id; writes into that run's directory")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--study", default=None, help=f"defaults to <model>_budget0.2_{SPACE_VERSION}")
    parser.add_argument(
        "--sentinel", type=Path, default=None, help="repeat-spread report; auto-discovered by model"
    )
    parser.add_argument("--search-dir", type=Path, default=SEARCH_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--device-dir", type=Path, default=DEVICE_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run:
        run = load_run(args.run, args.runs_dir)
        sources = Sources.from_run(run, args.cache_dir, args.search_dir)
        out = args.out or run.paths.dashboard
    else:
        study = args.study or study_name(args.model, DEFAULT_BUDGET_PT)
        sources = Sources.legacy(
            args.model,
            study,
            args.search_dir,
            args.report_dir,
            args.device_dir,
            args.cache_dir,
            args.sentinel or find_sentinel(args.device_dir, args.model),
        )
        out = args.out or DEFAULT_OUT

    payload = build_payload(sources)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload, args.template), encoding="utf-8")

    print(f"{payload['study']}  split {payload['split']['name']} ({payload['split']['samples']})")
    print(f"  {'id':4s}{'latency':>10s}{'size MB':>9s}{'RSS MB':>9s}{'top-1':>8s}  config")
    for row in payload["front"]:
        print(
            f"  {row['id']:4s}{row['latency_ms']:10.4f}{row['size_mb']:9.3f}"
            f"{row['peak_rss_mb']:9.3f}{row['top1']:8.3f}  {row['describe']} | {row['describe_run']}"
        )
    floor = payload["noise"]
    print(f"\n  repeat noise: "
          + (f"latency {floor['latency_percent']}% over {floor['trials']} repeats"
             if floor["measured"] else "not measured, ties on printed digits only"))
    print(f"  never first under any of the {len(payload['orderings'])} priority orders: "
          f"{', '.join(payload['unreachable']) or 'none'}")
    print(f"  trial history: {len(payload['trials'])} of {payload['trials_declared']} trials")
    heat = payload["thermal"]
    if heat:
        print(f"  thermal: {heat['count']} measurements, {heat['min_c']}-{heat['max_c']} C, "
              f"{heat['throttled']} throttled, {heat['unpinned']} unpinned, {heat['unstable']} unstable")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
