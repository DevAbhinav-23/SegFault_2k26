"""M6 — toolchain & runtime driver. Owner: Person C. LLD: design/03-lld-M6-toolchain.md.

Entry points per design/06-interfaces.md §7.2. The `air` package is imported lazily, inside a
function, so importing this module never requires the toolchain (FR-S20). Any stderr line
matching `error:` is a failure regardless of exit status (FR-T5, design/07-environment.md §4).

The off-device half — §3.1 (environment), §3.2 (the one subprocess chokepoint and the FR-T5
verdict), §3.3 (`artifact` for `none`/`pdi` and the `xclbin` pre-check), §3.7 (`ir_facts`) and
§3.9 (`check_pin`) — was written by B at P0c to unblock M4/M5; **Person C owns this module**.
§3.4-§3.6 and §3.8 (device detection, `run`, `diff`, `trace`) are still stubs.

`subprocess` is called in exactly one place, `invoke` (invariant I-1), and every `ToolRun` it
returns passes through `verdict` before its output is read (I-2).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sysconfig
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from spatial.model import Diagnostic, Target, ToolchainError

_NOT_BUILT = "m6_tools: not built yet; see design/03-lld-M6-toolchain.md"

_CLAUSE = "build()"
"""Every toolchain `Diagnostic` needs a clause (design/03-lld-M7-tests.md §3.2); this is it."""

PIN = {
    "mlir_air": "0.0.1.2026091204+ff95a9b",
    "mlir_aie": "1.4.3.dev55+g10767b5",
    "llvm_aie": "22.0.0.2026091201+386ca5c6",
}
"""The pinned wheel versions (FR-T6, design/03-lld-M6-toolchain.md §3.9)."""

ACCEPTED_TARGETS = ("npu1", "npu2", "auto")
"""Exactly what `resolve_target` accepts (design/03-lld-M6-toolchain.md §3.3)."""

ACCEPTED_FORMATS = ("none", "pdi", "xclbin")
"""The three `--output-format` values we drive (design/03-lld-M6-toolchain.md §3.3)."""

_TAIL_LINES = 40
"""How much of a 22-stage aiecc log reaches `details` (design/03-lld-M6-toolchain.md §5)."""


def _fail(code: str, reason: str, fix: str, details: Mapping[str, object],
          clause: str = _CLAUSE) -> ToolchainError:
    """Build the `ToolchainError` for `code`; every one carries all four message parts."""
    return ToolchainError(Diagnostic(code=code, stage="toolchain", clause=clause,
                                     reason=reason, fix=fix, location=None, details=details))


def _name(argv0: str) -> str:
    return os.path.basename(str(argv0))


def _tail(text: str, lines: int = _TAIL_LINES) -> list[str]:
    return text.splitlines()[-lines:]


# --------------------------------------------------------------------------------------------
# §3.9 check_pin — FR-T6
# --------------------------------------------------------------------------------------------


def check_pin() -> None:
    """Raise `ToolchainError(TOOL-VERSION-PIN)` unless the installed wheels match `PIN`."""
    from importlib import metadata

    installed = {}
    for name in PIN:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    mismatched = {n: [PIN[n], installed[n]] for n in PIN if installed[n] != PIN[n]}
    if not mismatched:
        return
    raise _fail(
        "TOOL-VERSION-PIN",
        "the installed toolchain is not the pinned one; "
        "every golden file is valid only for the pin",
        "pip install --no-index --find-links vendor/wheels 'mlir_air[aie]'",
        {"expected": PIN, "installed": installed, "mismatched": mismatched},
    )


# --------------------------------------------------------------------------------------------
# §3.1 Environment resolution
# --------------------------------------------------------------------------------------------


def _purelib() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _bundled(name: str) -> Path:
    """Where the wheel puts a native tool (`python/air/tools.py:39-58` looks here first)."""
    return _purelib() / "mlir_air" / "bin" / name


def tool(name: str) -> Path:
    """Resolve a native tool, the bundled wheel binary first and `PATH` second.

    Delegates to the toolchain's own `resolve_tool` rather than reimplementing it
    (design/03-lld-M6-toolchain.md §3.1). The import is lazy, so this module stays importable
    with no toolchain installed (FR-S20).
    """
    from air.tools import resolve_tool  # noqa: PLC0415 — lazy by FR-S20

    try:
        return Path(resolve_tool(name))
    except RuntimeError as exc:
        raise _fail(
            "TOOL-AIRCC-FAILED",
            f"MLIR-AIR tool {name!r} is not installed",
            "pip install --no-index --find-links vendor/wheels 'mlir_air[aie]'",
            {"searched": [str(_bundled(name)), "PATH"], "original": str(exc)},
        ) from None


def tool_available(name: str) -> bool:
    """True when `tool(name)` would resolve. Used by the `requires_*` skip predicates."""
    try:
        tool(name)
    except (ToolchainError, ImportError):
        return False
    return True


def tool_env() -> tuple[dict[str, str], Path | None]:
    """The environment `aircc` needs — derived, never read from the user's shell.

    Returns `(env, peano)`. `aircc` shells out to `aiecc`, which the wheel puts under
    `mlir_aie/bin`, and Peano must be locatable; nothing else in the project needs an
    environment variable at all (design/07-environment.md §2, §4).
    """
    purelib = _purelib()
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(purelib / "mlir_air" / "bin"),
                                   str(purelib / "mlir_aie" / "bin"),
                                   env.get("PATH", "")])
    peano = Path(env.get("PEANO_INSTALL_DIR") or (purelib / "llvm-aie"))
    if not peano.is_dir():
        return env, None
    env["PEANO_INSTALL_DIR"] = str(peano)
    return env, peano


# --------------------------------------------------------------------------------------------
# §3.2 The one subprocess chokepoint, and the FR-T5 verdict
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolRun:
    """One completed tool invocation. A `CompletedProcess` never escapes this module (I-1)."""

    argv: tuple[str, ...] = ()
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0


ERROR_LINE = re.compile(
    r'^(?P<loc>(?:loc\("[^"]*":\d+:\d+\)|[^\s:]+:\d+:\d+):\s*)?error:\s*(?P<msg>.*)$',
    re.MULTILINE)
"""Any `error:` line on stderr, with or without a source location.

Widened from design/03-lld-M6-toolchain.md §3.2 by **one alternative** in the `loc` group, and
only there: `aircc` spells its diagnostics `loc("<file>":<line>:<col>): error: ...` while
`air-opt` spells them `<file>:<line>:<col>: error: ...`, so §3.2's pattern as written matches no
`aircc` diagnostic at all. Both spellings are in `tests/fixtures/stderr/`, recorded live; the
measurement and the proposed spec edit are in `tests/fixtures/stderr/README.md`. Missing a real
diagnostic is precisely the failure FR-T5 exists to prevent, so the pattern is widened rather
than the evidence discarded.
"""

XCLBINUTIL_FIX = ('build with output_format="none" or output_format="pdi" — both are measured '
                  'to work with no XRT and no device — or install XRT')
"""The FR-T3 fix line (design/03-lld-M6-toolchain.md §5, verbatim)."""

XCLBINUTIL_CLAUSE = 'build(output_format="xclbin")'
"""The clause §5's verbatim FR-T3 message carries, which is more specific than `build()`."""

FIX_FOR: dict[str, str] = {
    "found channel op not in pairs":
        "the module has an unmatched channel op; M4's balance self-check should have caught "
        "this — run s.plan() and read the per-key table",
    "exceeded available memory":
        "the herd's L1 working set does not fit; reduce a `tile` factor or drop a "
        "`double_buffer`",
    "xclbinutil": XCLBINUTIL_FIX,
}
"""Diagnostic substring → fix hint (design/03-lld-M6-toolchain.md §3.2). Grows by a row."""

_DEFAULT_FIX = "read the diagnostics below and fix the emitted module"


def classify(diagnostic: str) -> str | None:
    """The `FIX_FOR` key a diagnostic line matches, or `None`. A table, never prose matching."""
    for substring in FIX_FOR:
        if substring in diagnostic:
            return substring
    return None


def invoke(argv: Sequence[object], cwd: str | os.PathLike[str],
           env: Mapping[str, str] | None = None, timeout: float = 600) -> ToolRun:
    """Run one tool. **The only `subprocess` call in the package** (invariant I-1).

    Every element of `argv` is stringified, so a `ToolRun` — and therefore every `details`
    payload built from one — stays JSON-serialisable. A timeout is a `TOOL-AIRCC-FAILED`, and so
    is an `OSError` from the tool path: no `FileNotFoundError` and no `CalledProcessError` ever
    reaches the caller (design/03-lld-M6-toolchain.md §5, NFR-7).
    """
    args = tuple(str(a) for a in argv)
    if not args:
        raise ValueError("m6.invoke: argv must name a tool")
    started = monotonic()
    try:
        done = subprocess.run(args, cwd=str(cwd), env=dict(env) if env is not None else None,
                              capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise _fail(
            "TOOL-AIRCC-FAILED",
            f"{args[0]} did not finish within {timeout}s",
            "re-run with a smaller fixture, or raise the timeout in m6.invoke",
            {"argv": list(args), "timeout_s": timeout},
        ) from None
    except OSError as exc:
        raise _fail(
            "TOOL-AIRCC-FAILED",
            f"{_name(args[0])} could not be run",
            "check the tool path and the working directory; reinstall with "
            "pip install --no-index --find-links vendor/wheels 'mlir_air[aie]'",
            {"argv": list(args), "cwd": str(cwd), "original": str(exc)},
        ) from None
    return ToolRun(args, done.returncode, done.stdout, done.stderr, monotonic() - started)


def verdict(run: ToolRun) -> None:
    """Raise unless `run` succeeded. **Stderr is read before the exit code** (FR-T5).

    A tool that printed an `error:` line fails whatever it exited with — `air-opt` calls
    `emitOpError` without `signalPassFailure()` (`mlir/lib/Util/Dependency.cpp:2063-2066`), so
    the diagnostic is the informative half and the exit code is the half that lies.
    """
    hits = [match.group(0) for match in ERROR_LINE.finditer(run.stderr)]
    argv0 = _name(run.argv[0]) if run.argv else "the tool"
    if hits:
        raise _fail(
            "TOOL-DIAGNOSTIC",
            f"{argv0} printed {len(hits)} diagnostic{'' if len(hits) == 1 else 's'} "
            f"and exited {run.returncode}",
            FIX_FOR.get(classify(hits[0]) or "", _DEFAULT_FIX),
            {"argv": list(run.argv), "returncode": run.returncode,
             "diagnostics": hits, "stderr": run.stderr},
        )
    if run.returncode != 0:
        missing_xclbinutil = "xclbinutil" in run.stdout + run.stderr
        raise _fail(
            "TOOL-MISSING-XCLBINUTIL" if missing_xclbinutil else "TOOL-AIRCC-FAILED",
            f"{argv0} exited {run.returncode} with no error: line",
            XCLBINUTIL_FIX if missing_xclbinutil else
            "re-run with --verbose and read the last aiecc stage",
            {"argv": list(run.argv), "returncode": run.returncode,
             "stdout_tail": _tail(run.stdout), "stderr_tail": _tail(run.stderr)},
            clause=XCLBINUTIL_CLAUSE if missing_xclbinutil else _CLAUSE,
        )


# --------------------------------------------------------------------------------------------
# §3.3 artifact() — the off-device build
# --------------------------------------------------------------------------------------------


def _xclbinutil_error() -> ToolchainError:
    return _fail(
        "TOOL-MISSING-XCLBINUTIL",
        "xclbin packaging needs xclbinutil and it is not on PATH",
        XCLBINUTIL_FIX,
        {"looked_for": "xclbinutil",
         "provided_by": "XRT (Xilinx Runtime)",
         "alternatives": ["none", "pdi"],
         "evidence": "VF §G.5 — aircc fails at step 40/41 with tool 'xclbinutil' not found "
                     "in search paths or PATH"},
        clause=XCLBINUTIL_CLAUSE,
    )


def artifact(mlir_path: str, target: Target, output_format: str,
             workdir: str | os.PathLike[str] | None = None) -> str:
    """Drive `aircc`; `output_format` is `"none"`, `"pdi"` or `"xclbin"`.

    Returns the artifact path — the `air_project` directory for `"none"`, the `.pdi` or
    `.xclbin` file otherwise. `aircc` never writes into the repository: `--tmpdir` and `cwd` are
    both inside `workdir` (invariant I-5). Raises `ToolchainError`.
    """
    check_pin()                                             # FR-T6, first, always
    if target not in ACCEPTED_TARGETS:
        raise _fail(
            "TOOL-TARGET",
            f"target {target!r} is not an AIR target",
            'pass target="npu1" or target="npu2"',
            {"given": target, "accepted": list(ACCEPTED_TARGETS),
             "why": "air.api's resolve_target accepts only these (_trace.py:164-183); "
                    "aircc's own --device default xcvc1902 is a declared non-goal"},
        )
    if output_format not in ACCEPTED_FORMATS:
        raise _fail(
            "TOOL-TARGET",
            f"output_format {output_format!r} is not one we drive",
            'pass output_format="none", "pdi" or "xclbin"',
            {"given": output_format, "accepted": list(ACCEPTED_FORMATS),
             "why": "aircc's --output-format also takes txn and elf, which no requirement "
                    "asks for (tools/aircc/aircc.cpp:266-268)"},
        )

    if target == "auto":
        # G-12: this is the ONLY place "auto" is resolved, and it spawns xrt-smi. The checker
        # resolves it internally to npu2 with no subprocess (decision D-13, M3 §3.11).
        from air.backend.xrt import detect_target_device  # noqa: PLC0415 — lazy by FR-S20

        resolved = detect_target_device(default="npu2")
    else:
        resolved = target

    if output_format == "xclbin" and shutil.which("xclbinutil") is None:
        raise _xclbinutil_error()                           # FR-T3, before spending 2 minutes

    work = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="m6-"))
    env, peano = tool_env()
    argv: list[object] = [tool("aircc"), "--device", resolved,
                          "--output-format", output_format,
                          "--tmpdir", work / "air_project"]
    if peano is not None:
        argv += ["--peano", peano, "--no-xchesscc", "--no-xbridge"]
    if output_format == "xclbin":
        out = work / "air.xclbin"
        argv += ["-o", out, "-i", work / "insts.txt"]
    elif output_format == "pdi":
        out = work / "air.pdi"
        argv += ["--pdi-name", out, "-i", work / "insts.txt"]
    else:
        out = work / "air_project"                          # the compile-only witness
    argv += [mlir_path]

    verdict(invoke(argv, cwd=work, env=env))                # FR-T5
    if output_format != "none" and not out.is_file():
        raise _fail(
            "TOOL-AIRCC-FAILED",
            "aircc reported success but produced no artifact",
            "re-run with --verbose",
            {"expected": str(out), "argv": [str(a) for a in argv]},
        )
    return str(out)


# --------------------------------------------------------------------------------------------
# §3.7 ir_facts() — the air-opt inspection pipeline
# --------------------------------------------------------------------------------------------

PIPELINES: dict[str, str] = {
    "pingpong": "builtin.module(air-dependency,"
                "air-label-scf-for-to-ping-pong{{device={target}}})",
    "pingpong_full": "builtin.module(air-dependency,air-annotate-packet-ids{{assign=true}},"
                     "air-hoist-dma-in-accum-pattern,air-dma-to-channel,canonicalize,cse,"
                     "air-dependency-canonicalize,canonicalize,cse,"
                     "air-isolate-async-dma-loop-nests{{scope=launch}},canonicalize,cse,"
                     "air-fuse-channels,canonicalize,cse,func.func(air-fuse-alloc-dealloc),"
                     "func.func(air-shrink-memref-sizes-by-access),"
                     "air-label-scf-for-to-ping-pong{{device={target}}})",
    "broadcast": "builtin.module(air-dependency,air-broadcast-detection)",
    "pairs": "builtin.module(air-dependency,air-dependency-canonicalize)",
    "aie": "builtin.module(air-place-herds{{num-rows={rows} num-cols={cols} "
           "row-anchor={row_anchor} col-anchor=0}},air-to-aie{{device={target}}})",
}
"""Pass pipeline per named inspection, so "which pipeline produced this number" is never a
question (design/03-lld-M6-toolchain.md §3.7). `{target}`, `{cols}`, `{rows}` and
`{row_anchor}` are filled by `str.format`; `{{`/`}}` are the literal braces of a pass option.
`pingpong_full` is the **unused recorded fallback**.

`air-to-aie` alone is **not** enough: on an unplaced module every herd keeps its logical origin,
so a 1-D `grid(4)` herd asks for columns 1..4 and the pass fails with `'aie.tile' op column index
(4) must be less than the number of columns in the device (4)` (measured on W3). `aircc` places
first (`tools/aircc/aircc.cpp:1158-1162`), and §3.7's pipeline now does the same with that call's
own geometry, so the number `ir_facts` reads is the number `aircc` compiled. Cross-checked on
W3/npu1: this pipeline and `aircc`'s own `air_project/aie.*.mlir` give the identical
`lock_init_histogram` `{{"0": 20, "1": 16, "2": 4}}` (design/PROGRESS-B.md, phase P4)."""

AIE_GEOMETRY: dict[str, dict[str, int]] = {
    "npu1": {"cols": 4, "rows": 6, "row_anchor": 2},
    "npu2": {"cols": 8, "rows": 6, "row_anchor": 2},
}
"""`air-place-herds`' geometry per target, as `aircc` resolves it when no flag overrides it
(`tools/aircc/aircc.cpp:1003-1022`: npu1 → 4 columns, npu2 → 8, both 6 rows anchored at row 2)."""

PIPELINES["transform"] = (PIPELINES["pingpong_full"][:-1]
                          + ",air-ping-pong-transform,canonicalize,cse)")


def _max_unroll(text: str) -> int:
    return max((int(n) for n in re.findall(r"unroll = (\d+) : i32", text)), default=0)


def _max_tokens_per_scf_for(text: str) -> int:
    """The largest number of `!air.async.token` iter_args on any one `scf.for` header line."""
    return max((line.count("!air.async.token") for line in text.splitlines()
                if "scf.for" in line), default=0)


def _lock_init_histogram(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in re.findall(r"init = (\d+)", text):
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


EXTRACTORS: dict[str, object] = {
    "pingpong_unroll": _max_unroll,
    "hoist_alloc_count": lambda text: text.count("hoist_alloc = true"),
    "broadcast_pattern_count": lambda text: text.count("broadcast_pattern"),
    "cascade_channels": lambda text: text.count("aie.cascade_flow"),
    "pingpong_iter_args": _max_tokens_per_scf_for,
    "lock_init_histogram": _lock_init_histogram,
}
"""One fact name → the function that reads it off the post-pass IR text (§3.7)."""

PIPELINE_OF: dict[str, str] = {
    "pingpong_unroll": "pingpong",
    "hoist_alloc_count": "pingpong",
    "cascade_channels": "aie",
    "pingpong_iter_args": "transform",
    "broadcast_pattern_count": "broadcast",
    "lock_init_histogram": "aie",
}
"""Which pipeline each fact is read from, exactly as §3.7's table fixes it.

**`cascade_channels` counts `aie.cascade_flow` ops after `air-to-aie`** (ruling R-F-3), which is
what `03-lld-M5-emitter.md` §3.8 specifies and what actually matters: one bundle of
`size=(PK-1,)` prints `channel_type = "npu_cascade"` **once** before lowering, and P0c's
reading 5 counted that string on the `pingpong` pipeline. Measured on the lowered flip probe
(`vendor/probes/review/ap_asc/aie.flip1d_asc.mlir`), the string occurs **4** times — the
`@CascadeK [3]` declaration plus the three `@channel_N [1, 1]` bundles `air-to-aie` splits it
into — while `aie.cascade_flow` occurs **3**, which is the number of physical links."""


def ir_facts(mlir_path: str, target: Target, facts: Iterable[str],
             workdir: str | os.PathLike[str] | None = None) -> dict[str, object]:
    """Run `air-opt` once per needed pipeline and extract the named facts (FR-E3, FR-E5).

    Every result also carries `_pipeline_<name>`, the exact pipeline string it came from, so a
    golden can never be compared against a number produced by a different pipeline (D-8).
    """
    wanted = list(facts)
    unknown = [f for f in wanted if f not in PIPELINE_OF]
    if unknown:
        raise ValueError(f"unknown ir fact(s) {unknown}; known: {sorted(PIPELINE_OF)}")
    work = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="m6-"))
    air_opt = tool("air-opt")
    geometry = AIE_GEOMETRY.get(target, AIE_GEOMETRY["npu1"])
    out: dict[str, object] = {}
    for name, template in PIPELINES.items():                # a fixed order, so runs compare
        group = [f for f in wanted if PIPELINE_OF[f] == name]
        if not group:
            continue
        pipeline = template.format(target=target, **geometry)
        destination = work / f"{name}.mlir"
        verdict(invoke([air_opt, mlir_path, f"-pass-pipeline={pipeline}", "-o", destination],
                       cwd=work))                           # FR-T5 applies here too
        text = destination.read_text(encoding="utf-8")
        for fact in group:
            out[fact] = EXTRACTORS[fact](text)              # type: ignore[operator]
        out[f"_pipeline_{name}"] = pipeline
    return out


# --------------------------------------------------------------------------------------------
# §3.4-§3.6, §3.8 — the device and runner path. Person C.
# --------------------------------------------------------------------------------------------


def has_device() -> bool:
    """True when a `/dev/accel*` character device exists. Raises nothing."""
    raise NotImplementedError(_NOT_BUILT)  # Person C, M6 §3.4-§3.6, §3.8


def run(artifact: str, inputs: Sequence[object]) -> list[object]:
    """Execute on a device. Raises ToolchainError."""
    raise NotImplementedError(_NOT_BUILT)  # Person C, M6 §3.4-§3.6, §3.8


def diff(device: Sequence[object], oracle: Sequence[object], tol: float) -> object:
    """Compare device output against the oracle. Raises nothing."""
    raise NotImplementedError(_NOT_BUILT)  # Person C, M6 §3.4-§3.6, §3.8


def trace(mlir_path: str, model_json: str) -> str:
    """Drive air-runner, a performance model only. Raises ToolchainError."""
    raise NotImplementedError(_NOT_BUILT)  # Person C, M6 §3.4-§3.6, §3.8
