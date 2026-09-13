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


# --------------------------------------------------------------------------------------------
# W1 as **our emitter** writes it. Spec: design/04-test-plan.md §3.5. Added by B at P1.
# Only one `aircc` may run at a time; pytest is serial, so do not add xdist to this file.
# --------------------------------------------------------------------------------------------


def _w1_module(tmp_path: Path, target: str) -> Path:
    from spatial import m5_emit
    from tests.fixtures.plans import w1_plan

    path = tmp_path / f"w1.base.{target}.air.mlir"
    path.write_text(m5_emit.emit(w1_plan.plan(target), target).mlir, encoding="utf-8")
    return path


def _require_aircc() -> None:
    if not m6.tool_available("aircc"):
        pytest.skip("aircc is not installed; install the [aie] extra (07-environment.md §2)")


@pytest.mark.slow
@pytest.mark.requires_aircc
@pytest.mark.parametrize("target", ["npu1", "npu2"])
@pytest.mark.fr("FR-T2")
def test_W1_aircc_none(tmp_path, target):
    """`aircc --device <target> --output-format=none` on M5's own W1 module.

    `m6.verdict` is what makes this a verdict rather than an exit code: a diagnostic on stderr
    fails it whatever `aircc` exited with (FR-T5).
    """
    _require_aircc()
    out = m6.artifact(str(_w1_module(tmp_path, target)), target, "none", workdir=tmp_path)
    assert out == str(tmp_path / "air_project")
    assert Path(out).is_dir(), "aircc's --tmpdir is the compile-only witness for output none"


@pytest.mark.slow
@pytest.mark.requires_aircc
@pytest.mark.fr("FR-T2", "FR-T3")
def test_W1_aircc_pdi(tmp_path):
    """One `pdi` build, which needs neither XRT nor a device (VF §G.5)."""
    _require_aircc()
    out = m6.artifact(str(_w1_module(tmp_path, "npu1")), "npu1", "pdi", workdir=tmp_path)
    assert out == str(tmp_path / "air.pdi")
    assert Path(out).is_file()


# --------------------------------------------------------------------------------------------
# W3 as **our emitter** writes it. Spec: design/04-test-plan.md §3.5. Added by B at P4.
# Only one `aircc` may run at a time; pytest is serial, so do not add xdist to this file.
# --------------------------------------------------------------------------------------------


def _w3_module(tmp_path: Path, target: str) -> Path:
    from spatial import m4_mapping as m4, m5_emit
    from tests.fixtures.mappings import w3_legal

    path = tmp_path / f"w3.base.{target}.air.mlir"
    path.write_text(m5_emit.emit(m4.plan(w3_legal.legal(target)), target).mlir, encoding="utf-8")
    return path


@pytest.mark.slow
@pytest.mark.requires_aircc
@pytest.mark.parametrize("target", ["npu1", "npu2"])
@pytest.mark.fr("FR-T2", "FR-M5")
def test_W3_aircc_none(tmp_path, target):
    """`aircc --device <target> --output-format=none` on M5's own wavefront module.

    This is what makes ruling **R-W3-1** a measurement rather than a design argument: the
    segment-scope source puts `S[i, 0:1]` and the drain gets `S[i, NR:NR+1]` — two L3 endpoints
    on the kernel's own written tensor, one of them an address `SOut` also covers — and the
    default pipeline runs `air-verify-hierarchy-locality{strict=true}` on the **placed** IR
    (`tools/aircc/aircc.cpp:1213-1218`, `cl::init(PIV_error)`), so a race between them would be
    a hard error here.
    """
    _require_aircc()
    out = m6.artifact(str(_w3_module(tmp_path, target)), target, "none", workdir=tmp_path)
    assert out == str(tmp_path / "air_project")
    assert Path(out).is_dir(), "aircc's --tmpdir is the compile-only witness for output none"


@pytest.mark.slow
@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-M5", "FR-T5")
def test_W3_hierarchy_locality(tmp_path):
    """The locality verifier run **explicitly**, so its verdict is recorded and not inferred.

    `aircc` runs it inside a 22-stage build whose exit code says nothing about which pass was
    happy; this runs the one pass on its own, in both strictness settings, and `m6.verdict`
    fails on any `error:` line whatever the exit code (FR-T5).
    """
    if not m6.tool_available("air-opt"):
        pytest.skip("air-opt is not resolvable; install the toolchain (07-environment.md §2)")
    module = _w3_module(tmp_path, "npu1")
    for strict in ("false", "true"):
        pipeline = f"builtin.module(air-verify-hierarchy-locality{{strict={strict}}})"
        run = m6.invoke([m6.tool("air-opt"), module, f"-pass-pipeline={pipeline}",
                         "-o", tmp_path / f"verified.{strict}.mlir"], cwd=tmp_path)
        m6.verdict(run)
        assert run.stderr.strip() == "", run.stderr
