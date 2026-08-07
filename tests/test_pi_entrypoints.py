"""The device-side entry points must import without torch.

The Pi installs numpy, onnxruntime and onnx only -- never torch (see
`requirements-pi.txt`), because a torch install there invites accidental
torch-vs-ORT comparisons that do not reflect the deployment target. That makes
"does this module import cleanly?" a real invariant rather than a style
preference: a single `from src.models.registry import ...` anywhere in the
transitive import graph makes the module unrunnable on the only machine whose
numbers are admissible.

Checking it needs a subprocess. Torch *is* installed on the dev box, so an
in-process import would succeed whether or not the dependency is there; only a
fresh interpreter can tell us whether importing the module pulled torch in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.quant.config import EXPORTED_MODELS

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "artifacts" / "reports"
BASELINE_REPORTS = sorted(REPORT_DIR.glob("*_baseline.json"))

# Entry points that run on the Pi. Each transitively covers the quantization
# spine (config, quantize, calibration, groups, evaluate) and src.portable, so
# a torch dependency reintroduced anywhere below them fails this test.
PI_ENTRYPOINTS = (
    "src.quant.measure",
    "src.quant.baselines",
    "src.quant.saturation_probe",
    # The Phase 3 agent. `src.bench.remote` is deliberately absent: it is the
    # host half of the pair and never runs on the device.
    "src.bench.agent",
)


@pytest.mark.parametrize("module", PI_ENTRYPOINTS)
def test_entrypoint_imports_without_torch(module: str) -> None:
    probe = (
        "import importlib, sys; "
        f"importlib.import_module({module!r}); "
        "print('TORCH_LOADED' if 'torch' in sys.modules else 'TORCH_FREE')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"{module} failed to import:\n{result.stderr}"
    assert "TORCH_FREE" in result.stdout, (
        f"{module} pulled torch into sys.modules, so it cannot run on the Pi. "
        "Check what was added to its import graph."
    )


def test_exported_models_match_the_baseline_reports() -> None:
    """The campaign's models are those Phase 2 froze a baseline for.

    `artifacts/reports/<model>_baseline.json` is the Stage 0 -> Stage 1 contract
    and is committed, so this check is meaningful on the Pi too. It catches the
    drift that matters: a model exported and baselined but never added here
    would be silently unmeasurable.
    """
    baselined = tuple(sorted(p.name.removesuffix("_baseline.json") for p in BASELINE_REPORTS))
    assert baselined, f"no baseline reports found under {REPORT_DIR}"
    assert EXPORTED_MODELS == baselined


def test_exported_models_are_registered_architectures() -> None:
    """Every name here must be a real registered model -- but not the reverse.

    The registry is deliberately a superset: `resnet34_cifar` is registered and
    never trained, and `EXPORTED_MODELS` names only what the campaign measures.
    So this is a subset check, and a rename in the registry still trips it.

    Skipped where torch is absent: on the Pi the registry is unimportable by
    design, which is the whole reason the constant exists.
    """
    pytest.importorskip("torch", reason="registry needs torch; PC-only check")

    from src.models import available_models  # noqa: PLC0415 -- torch-gated import

    assert set(EXPORTED_MODELS) <= set(available_models())
