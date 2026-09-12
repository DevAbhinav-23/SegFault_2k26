"""Level S — the off-device `aircc` path end to end. Spec: design/03-lld-M6-toolchain.md §6.2.

Marked `slow`, so it is excluded by the default `addopts`; run it with `pytest -m slow`. Only
one `aircc` may run at a time (the tool parallelises internally and the machine is shared).

Stub written by B at P0c to unblock; **Person C owns this file**.
"""

from __future__ import annotations

import importlib.util
from importlib.resources import files
from pathlib import Path

import pytest

from spatial import m6_tools as m6

PROBE = Path(str(files("tests"))).parent / "vendor" / "probes" / "q" / "q7_a.py"
"""The upstream-API W1 shape (`M=N=K=64, TM=TN=32, TK=16, PI=PJ=2`), **not** our DSL.

`vendor/probes/` is git-ignored, so this test skips wherever the probe sources were not copied.
It is the toolchain half of gate G1: the pinned wheel compiles a W1-shaped module off-device.
"""


def _emit_probe_module(destination: Path) -> Path:
    """Import the probe in-process (never a subprocess: invariant I-1) and write its AIR text."""
    spec = importlib.util.spec_from_file_location("q7_a_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination.write_text(str(module.build().build(target="npu1")), encoding="utf-8")
    return destination


@pytest.mark.slow
@pytest.mark.requires_aircc
@pytest.mark.fr("FR-T2")
def test_probe_w1_aircc_none(tmp_path):
    """`aircc --device npu1 --output-format=none` on a W1-shaped module returns `air_project`."""
    if not PROBE.is_file():
        pytest.skip("probe sources not present (git-ignored)")
    if not m6.tool_available("aircc"):
        pytest.skip("aircc is not installed; install the [aie] extra (07-environment.md §2)")

    mlir_path = _emit_probe_module(tmp_path / "w1_base.mlir")
    out = m6.artifact(str(mlir_path), "npu1", "none", workdir=tmp_path)

    assert out == str(tmp_path / "air_project")
    assert Path(out).is_dir(), "aircc's --tmpdir is the compile-only witness for output none"
