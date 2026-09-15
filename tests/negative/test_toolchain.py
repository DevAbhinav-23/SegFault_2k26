"""Level N — the toolchain corpus (M6). Owner: Person A (the corpus); M6 is Person C's.

All six `toolchain` codes of `06-interfaces.md` §6.3, through M6's public functions, with
**no toolchain binary, no network and no device**: every case that would otherwise need one
patches `spatial.m6_tools` with `unittest.mock.patch` inside the function itself, so nothing
leaks into another test. `tests/unit/test_m6_tools.py` asserts what these messages carry; what
this module owes is a raiser per code (FR-D3) whose diagnostic `test_D1_schema` can read
(FR-D1).

`TOOL-AIRCC-FAILED` and `TOOL-DIAGNOSTIC` go through `verdict(ToolRun(...))` — a recorded tool
result rather than a patched subprocess, because `ToolRun` is exactly what `invoke` returns and
the verdict is the half under test (FR-T5: **stderr is read before the exit code**).
"""

from __future__ import annotations

from unittest import mock

import pytest

from spatial import m6_tools as m6
from spatial.model import ToolchainError
from tests.negative import _corpus

_AIR_OPT_DIAGNOSTIC = ("module.mlir:12:3: error: 'air.channel.put' op found channel op not in "
                       "pairs")
"""One recorded `air-opt` diagnostic (`tests/fixtures/stderr/exit0_with_error.txt`'s shape).
`air-opt` calls `emitOpError` without `signalPassFailure()`, so it prints this and exits 0."""


def raises_tool_target():
    """`aircc`'s own `--device` default is `xcvc1902`, which is a declared non-goal: M6 accepts
    `npu1`, `npu2` and `auto` and nothing else (FR-T1)."""
    with mock.patch.object(m6, "check_pin"):          # the pin is FR-T6's check, not this one
        m6.artifact("kernel.mlir", "xcvc1902", "none")


def raises_tool_missing_xclbinutil():
    """`output_format="xclbin"` needs XRT's `xclbinutil`, and says so **before** spending two
    minutes in `aircc` (FR-T3)."""
    with mock.patch.object(m6, "check_pin"), \
            mock.patch.object(m6.shutil, "which", return_value=None):
        m6.artifact("kernel.mlir", "npu1", "xclbin")


def raises_tool_aircc_failed():
    """A tool that exits non-zero with no `error:` line on stderr."""
    m6.verdict(m6.ToolRun(argv=("aircc", "--device", "npu1"), returncode=1,
                          stdout="", stderr="aborted\n"))


def raises_tool_diagnostic():
    """FR-T5: an `error:` line **with exit code 0** is still a failure."""
    m6.verdict(m6.ToolRun(argv=("air-opt", "module.mlir"), returncode=0, stdout="",
                          stderr=_AIR_OPT_DIAGNOSTIC + "\n"))


def raises_tool_version_pin():
    """FR-T6: a golden file is valid only for the pinned wheel, so a mismatch is a rejection.

    The pin itself is patched rather than the installed wheels, so the case is the same on a
    machine with the toolchain and on one without.
    """
    with mock.patch.object(m6, "PIN", {"mlir_air": "0.0.0.not-the-pin"}):
        m6.check_pin()


def raises_tool_no_device():
    """A device run with no `/dev/accel*`: the off-device path is named in the fix (FR-T4)."""
    with mock.patch.object(m6, "has_device", return_value=False):
        m6.run("kernel.xclbin", [], "npu1", "gemm")


def _module():
    import sys
    return sys.modules[__name__]


@pytest.mark.fr("FR-D3")
def test_toolchain_codes():
    """Every corpus function raises the `ToolchainError` code its name spells."""
    _corpus.assert_codes(_module(), ToolchainError)


@pytest.mark.fr("FR-T5")
def test_stderr_is_read_before_the_exit_code():
    """FR-T5, restated as the corpus sees it: the two verdicts do not swap.

    Exit 0 with a diagnostic is `TOOL-DIAGNOSTIC`; a non-zero exit with none is
    `TOOL-AIRCC-FAILED`. Reading the exit code first would give the first case a pass.
    """
    with pytest.raises(ToolchainError) as excinfo:
        raises_tool_diagnostic()
    assert excinfo.value.diagnostic.details["returncode"] == 0
    assert excinfo.value.diagnostic.details["diagnostics"] == (_AIR_OPT_DIAGNOSTIC,)

    with pytest.raises(ToolchainError) as excinfo:
        raises_tool_aircc_failed()
    assert excinfo.value.diagnostic.details["returncode"] == 1

    # a clean run is not a failure at all
    assert m6.verdict(m6.ToolRun(argv=("aircc",), returncode=0, stdout="ok\n", stderr="")) is None
