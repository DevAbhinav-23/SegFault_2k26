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
    assert m6.EXTRACTORS["cascade_channels"](
        'air.channel @C [3] {channel_type = "npu_cascade"}') == 1
    assert m6.EXTRACTORS["lock_init_histogram"](
        "init = 0\ninit = 1\ninit = 1") == {"0": 1, "1": 2}


def test_ir_facts_rejects_an_unknown_fact(tmp_path):
    """An unknown fact name is a programmer error, named with the known set."""
    with pytest.raises(ValueError, match="pingpong_unroll"):
        m6.ir_facts(str(tmp_path / "x.mlir"), "npu1", ["not_a_fact"], workdir=tmp_path)


def test_device_path_is_still_a_stub():
    """§3.4-§3.6 and §3.8 are Person C's; they must fail loudly, not silently."""
    for name in ("has_device", "run", "diff", "trace"):
        with pytest.raises(NotImplementedError):
            getattr(m6, name)(*[None] * getattr(m6, name).__code__.co_argcount)
