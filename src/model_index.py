"""What the app knows about each model, without opening its ONNX graph."""

# The graphs are gitignored (61 MB), so a fresh clone cannot call `build_group_map`.
# This index carries the few facts the search and the simulator need, costs a few
# kilobytes, and is what an imported model registers itself in.
#
#     python -m src.model_index --rebuild

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INDEX_PATH = Path("artifacts/models/index.json")
INDEX_SCHEMA = "model-index/1"

ONNX_DIR = Path("artifacts/onnx")
REPORT_DIR = Path("artifacts/reports")


@dataclass(frozen=True)
class ModelEntry:
    """One model the app can run a campaign for."""

    name: str
    source: str  # "trained" here, or "imported" for a graph brought from outside
    groups: tuple[str, ...]
    group_depth: int
    baseline_top1: float
    fp32_bytes: int
    classes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "groups": list(self.groups),
            "group_depth": self.group_depth,
            "baseline_top1": self.baseline_top1,
            "fp32_bytes": self.fp32_bytes,
            "classes": self.classes,
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> ModelEntry:
        return cls(
            name=document["name"],
            source=document["source"],
            groups=tuple(document["groups"]),
            group_depth=document["group_depth"],
            baseline_top1=document["baseline_top1"],
            fp32_bytes=document["fp32_bytes"],
            classes=document["classes"],
        )


def load_index(path: Path = INDEX_PATH) -> dict[str, ModelEntry]:
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != INDEX_SCHEMA:
        raise SystemExit(f"{path} is schema {document.get('schema')!r}, expected {INDEX_SCHEMA}")
    return {name: ModelEntry.from_dict(entry) for name, entry in document["models"].items()}


def save_index(entries: dict[str, ModelEntry], path: Path = INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": INDEX_SCHEMA,
        "models": {name: entry.as_dict() for name, entry in sorted(entries.items())},
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def register(entry: ModelEntry, path: Path = INDEX_PATH) -> None:
    entries = load_index(path)
    entries[entry.name] = entry
    save_index(entries, path)


def known_models(path: Path = INDEX_PATH) -> tuple[str, ...]:
    """Every model the app can run, index first so imports are included."""
    from src.quant.config import EXPORTED_MODELS

    return tuple(sorted(set(EXPORTED_MODELS) | set(load_index(path))))


def entry_for(model: str, path: Path = INDEX_PATH) -> ModelEntry:
    entries = load_index(path)
    if model not in entries:
        raise SystemExit(
            f"{model!r} is not in {path}. Rebuild it with: python -m src.model_index --rebuild"
        )
    return entries[model]


def describe_trained(model: str) -> ModelEntry:
    """Read one trained model's facts back out of its graph and baseline report."""
    from src.quant.groups import build_group_map, group_depth_for, quantizable_groups
    from src.quant.quantize import ModelPaths

    paths = ModelPaths.resolve(model, ONNX_DIR)
    baseline = json.loads((REPORT_DIR / f"{model}_baseline.json").read_text(encoding="utf-8"))
    return ModelEntry(
        name=model,
        source="trained",
        groups=tuple(quantizable_groups(build_group_map(paths.quant_ready, model))),
        group_depth=group_depth_for(model),
        baseline_top1=baseline["metrics"]["optval_top1"],
        fp32_bytes=baseline["onnx"]["bytes"],
        classes=100,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="re-read every trained model")
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    args = parser.parse_args()

    if not args.rebuild:
        for name, entry in sorted(load_index(args.index).items()):
            print(f"{name:20s} {entry.source:9s} {len(entry.groups):2d} groups  "
                  f"baseline {entry.baseline_top1:.2f}%  {entry.fp32_bytes / 1e6:.1f} MB")
        return

    from src.quant.config import EXPORTED_MODELS

    entries = load_index(args.index)
    for model in EXPORTED_MODELS:
        entries[model] = describe_trained(model)
        print(f"{model:20s} {len(entries[model].groups):2d} groups: {', '.join(entries[model].groups)}")
    save_index(entries, args.index)
    print(f"\nWrote {args.index}")


if __name__ == "__main__":
    main()
