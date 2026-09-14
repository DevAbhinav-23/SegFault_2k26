"""M6's off-device wrappers. Spec: design/03-lld-M6-toolchain.md §3.1-§3.3, §3.7, §3.9, §7.

No subprocess runs here except the one deliberate `sleep` of `test_invoke_timeout`: the three
stderr samples are recorded (`tests/fixtures/stderr/README.md`), which is what keeps
`test_stderr_parser` in the default suite (`03-lld-M7-tests.md` §3.8).

Stub written by B at P0c to unblock; **Person C owns this file**.
"""

from __future__ import annotations

import shutil
import time
from importlib import metadata
from importlib.resources import files
from pathlib import Path

import pytest

from spatial import m6_tools as m6
from spatial.model import ToolchainError
from tests.helpers.diagnostics import assert_diagnostic
STDERR = Path(str(files("tests"))) / "fixtures" / "stderr"

# (sample, returncode) — the exit codes are recorded in tests/fixtures/stderr/README.md
SAMPLES = {"clean": 0, "exit0_with_error": 0, "exit1_with_error": 1}


def _run(sample: str, tool: str = "air-opt") -> m6.ToolRun:
    return m6.ToolRun(argv=(tool, "<module>.mlir"), returncode=SAMPLES[sample], stdout="",
                      stderr=(STDERR / f"{sample}.txt").read_text(encoding="utf-8"))


@pytest.mark.fr("FR-T5")
def test_stderr_parser():
    """All three verdict branches, from the recorded samples (§3.2, 04-test-plan §2)."""
    assert m6.verdict(_run("clean")) is None

    with pytest.raises(ToolchainError) as excinfo:
        m6.verdict(_run("exit0_with_error"))
    assert_diagnostic(excinfo, code="TOOL-DIAGNOSTIC", clause="build()",
                      mentions=("found channel op not in pairs",),
                      details_keys=("argv", "returncode", "diagnostics", "stderr"))
    # The whole of FR-T5: a diagnostic is a failure even though the tool exited 0.
    assert excinfo.value.diagnostic.details["returncode"] == 0
    assert "exited 0" in excinfo.value.diagnostic.reason

    with pytest.raises(ToolchainError) as excinfo:
        m6.verdict(_run("exit1_with_error", tool="aircc"))
    # Through the DIAGNOSTIC branch, so the message is the diagnostic, not "exit 1" (§7).
    assert_diagnostic(excinfo, code="TOOL-DIAGNOSTIC", clause="build()",
                      mentions=("use of undeclared SSA value name",))
    assert excinfo.value.diagnostic.details["returncode"] == 1


@pytest.mark.fr("FR-T5")
def test_T5_error_without_exit_code():
    """A synthetic exit-0 run with one `error:` line still fails (invariant I-3)."""
    with pytest.raises(ToolchainError) as excinfo:
        m6.verdict(m6.ToolRun(argv=("air-opt",), returncode=0,
                              stderr="some chatter\nerror: x\n"))
    assert_diagnostic(excinfo, code="TOOL-DIAGNOSTIC")
    assert excinfo.value.diagnostic.details["diagnostics"] == ("error: x",)


def test_verdict_nonzero_exit_with_clean_stderr():
    """No `error:` line and a non-zero exit is `TOOL-AIRCC-FAILED`, with tails (§5)."""
    with pytest.raises(ToolchainError) as excinfo:
        m6.verdict(m6.ToolRun(argv=("aircc",), returncode=2, stdout="a\n", stderr="b\n"))
    assert_diagnostic(excinfo, code="TOOL-AIRCC-FAILED",
                      details_keys=("argv", "returncode", "stdout_tail", "stderr_tail"))


def test_classify_table():
    """Each substring of §3.2's table maps to its own fix hint, and nothing else does."""
    assert set(m6.FIX_FOR) == {"found channel op not in pairs", "exceeded available memory",
                               "xclbinutil"}
    for substring, hint in m6.FIX_FOR.items():
        line = f"design.mlir:1:1: error: something {substring} something"
        assert m6.classify(line) == substring
        assert m6.FIX_FOR[m6.classify(line)] == hint
    assert m6.classify("error: an unrecognised complaint") is None


@pytest.mark.fr("FR-T1")
def test_T1_targets(monkeypatch, tmp_path):
    """A target outside `{npu1, npu2, auto}` is rejected before any subprocess runs."""
    def _no_subprocess(*args, **kwargs):
        raise AssertionError("artifact() must reject the target before invoking a tool")

    monkeypatch.setattr(m6, "invoke", _no_subprocess)
    with pytest.raises(ToolchainError) as excinfo:
        m6.artifact(str(tmp_path / "w1.mlir"), "xcvc1902", "none", workdir=tmp_path)
    assert_diagnostic(excinfo, code="TOOL-TARGET", clause="build()",
                      mentions=("xcvc1902", "npu1", "npu2"),
                      details_keys=("given", "accepted", "why"))

    with pytest.raises(ToolchainError) as excinfo:
        m6.artifact(str(tmp_path / "w1.mlir"), "npu1", "elf", workdir=tmp_path)
    assert_diagnostic(excinfo, code="TOOL-TARGET", mentions=("elf", "none", "pdi"))


@pytest.mark.fr("FR-T3")
def test_T3_xclbin_message(monkeypatch, tmp_path):
    """FR-T3 / invariant I-4: named parts, and raised before `aircc` could have run."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(m6, "invoke", lambda *a, **k: pytest.fail("aircc must not run"))
    started = time.monotonic()
    with pytest.raises(ToolchainError) as excinfo:
        m6.artifact(str(tmp_path / "w1.mlir"), "npu1", "xclbin", workdir=tmp_path)
    assert time.monotonic() - started < 1.0
    assert_diagnostic(excinfo, code="TOOL-MISSING-XCLBINUTIL",
                      clause='build(output_format="xclbin")',
                      mentions=("xclbinutil", "XRT", "none", "pdi"),
                      details_keys=("looked_for", "provided_by", "alternatives", "evidence"))
    assert excinfo.value.diagnostic.reason == \
        "xclbin packaging needs xclbinutil and it is not on PATH"


@pytest.mark.fr("FR-T6")
def test_T6_versions(monkeypatch):
    """The pin holds in this venv; a mismatch names the reinstall command (§3.9)."""
    assert m6.check_pin() is None
    assert m6.PIN["mlir_air"] == metadata.version("mlir_air")

    monkeypatch.setattr(metadata, "version",
                        lambda name: "0.0.0" if name == "mlir_air" else m6.PIN[name])
    with pytest.raises(ToolchainError) as excinfo:
        m6.check_pin()
    assert_diagnostic(excinfo, code="TOOL-VERSION-PIN", clause="build()",
                      mentions=("--find-links vendor/wheels", "0.0.0"),
                      details_keys=("expected", "installed", "mismatched"))


def test_invoke_timeout(tmp_path):
    """A tool that overruns its timeout is `TOOL-AIRCC-FAILED`, not a `TimeoutExpired` (§3.2)."""
    with pytest.raises(ToolchainError) as excinfo:
        m6.invoke(["sleep", "5"], cwd=tmp_path, timeout=0.2)
    assert_diagnostic(excinfo, code="TOOL-AIRCC-FAILED", mentions=("sleep", "0.2"),
                      details_keys=("argv", "timeout_s"))


def test_invoke_wraps_an_os_error(tmp_path):
    """A missing tool path is a `ToolchainError`, never a `FileNotFoundError` (§5, NFR-7)."""
    with pytest.raises(ToolchainError) as excinfo:
        m6.invoke([tmp_path / "no-such-tool"], cwd=tmp_path, timeout=30)
    assert_diagnostic(excinfo, code="TOOL-AIRCC-FAILED", mentions=("no-such-tool",),
                      details_keys=("argv", "cwd", "original"))
    with pytest.raises(ValueError, match="must name a tool"):
        m6.invoke([], cwd=tmp_path)


def test_invoke_stringifies_argv(tmp_path):
    """`ToolRun.argv` is all `str`, so every `details` payload stays JSON-serialisable."""
    run = m6.invoke(["true", tmp_path], cwd=tmp_path, timeout=30)
    assert run.argv == ("true", str(tmp_path))
    assert run.returncode == 0 and run.seconds >= 0.0
    assert m6.verdict(run) is None


def test_pipelines_and_extractors_agree():
    """Every fact names a pipeline that exists and an extractor that reads it (§3.7)."""
    assert set(m6.PIPELINE_OF) == set(m6.EXTRACTORS)
    assert set(m6.PIPELINE_OF.values()) <= set(m6.PIPELINES)
    assert m6.PIPELINES["pingpong"].format(target="npu1") == \
        "builtin.module(air-dependency,air-label-scf-for-to-ping-pong{device=npu1})"
    assert m6.PIPELINES["pairs"] == \
        "builtin.module(air-dependency,air-dependency-canonicalize)"
    assert m6.PIPELINES["transform"].endswith(
        ",air-label-scf-for-to-ping-pong{{device={target}}},"
        "air-ping-pong-transform,canonicalize,cse)")
    assert m6.EXTRACTORS["pingpong_unroll"]("scf.for ... {unroll = 2 : i32}") == 2
    assert m6.EXTRACTORS["pingpong_unroll"]("nothing here") == 0
    # `cascade_channels` counts the **physical links** `air-to-aie` built, not the attribute on
    # the declaration (ruling R-F-3, `03-lld-M5-emitter.md` §3.8). The lowered W1-flip carries
    # `channel_type = "npu_cascade"` four times — `@CascadeK [3]` plus the three `@channel_N`
    # bundles the pass splits it into — and `aie.cascade_flow` three times, which is the number
    # that matters. This sample is the shape of both.
    assert m6.PIPELINE_OF["cascade_channels"] == "aie"
    assert m6.EXTRACTORS["cascade_channels"](
        'air.channel @CascadeK [3] {channel_type = "npu_cascade"}\n'
        'air.channel @channel_0 [1, 1] {channel_type = "npu_cascade"}\n'
        "aie.cascade_flow(%tile_0_2, %tile_1_2)\n"
        "aie.cascade_flow(%tile_1_2, %tile_2_2)\n"
        "aie.cascade_flow(%tile_2_2, %tile_3_2)\n") == 3
    assert m6.EXTRACTORS["cascade_channels"](
        'air.channel @C [3] {channel_type = "npu_cascade"}') == 0
    assert m6.EXTRACTORS["lock_init_histogram"](
        "init = 0\ninit = 1\ninit = 1") == {"0": 1, "1": 2}


def test_ir_facts_rejects_an_unknown_fact(tmp_path):
    """An unknown fact name is a programmer error, named with the known set."""
    with pytest.raises(ValueError, match="pingpong_unroll"):
        m6.ir_facts(str(tmp_path / "x.mlir"), "npu1", ["not_a_fact"], workdir=tmp_path)


@pytest.mark.fr("FR-T4")
def test_T4_device_diff():
    """FR-T4: device diff is exact; without a device this skips with the reason (R-09).

    It gates on `has_device()` rather than carrying `requires_device`, deliberately. That
    marker is deselected by the default `addopts`, and a deselected test prints nothing — but
    04-test-plan §8 item 10 and R-09 want the *absence of the device* to appear in the default
    run's skip list, because that list is what the honest-limits slide says out loud.
    """
    if not m6.has_device():
        pytest.skip("no NPU: /dev/accel* is absent, so level D cannot run "
                    "(04-test-plan.md §1)")
    pytest.skip("device present but no xclbin artifact committed; D5 device window")


def test_trace_disclaimer():
    """I-8: anything M6 prints about air-runner carries the timing-model disclaimer."""
    import inspect

    src = inspect.getsource(m6.trace)
    assert "NOT A CORRECTNESS ORACLE" in src


def test_diff_denominator():
    """I-7: `diff` raises nothing and always reports `count_mismatched` of `total`."""
    import numpy as np

    ok = m6.diff([np.zeros((2, 2))], [np.zeros((2, 2))], 0.0)
    assert (ok.matched, ok.max_abs_err, ok.count_mismatched, ok.total) == (True, 0.0, 0, 4)
    bad = m6.diff([np.ones((2, 2))], [np.zeros((2, 2))], 0.0)
    assert (bad.matched, bad.count_mismatched, bad.total) == (False, 4, 4)
    assert bad.first_mismatch == (0, 0)
    shape = m6.diff([np.zeros((2,))], [np.zeros((3,))], 0.0)
    assert (shape.matched, shape.count_mismatched, shape.total) == (False, 3, 3)

    # A short result list used to be truncated by `zip` and reported as a match: a kernel
    # that returned one buffer instead of two passed. Every oracle element is charged.
    missing = m6.diff([np.zeros((2, 2))], [np.zeros((2, 2)), np.ones((3,))], 0.0)
    assert (missing.matched, missing.count_mismatched, missing.total) == (False, 7, 7)
    assert m6.diff([], [np.zeros((2,))], 0.0).matched is False
    assert m6.diff([], [], 0.0).matched is True
