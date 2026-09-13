# LLD M6 — Toolchain & runtime driver

*Phase 1, 2026-09-12. Owner: **Person C**. Reads on top of [`01-requirements.md`](01-requirements.md)
§3.5, [`02-hld.md`](02-hld.md) §2 (M6) and §4, [`06-interfaces.md`](06-interfaces.md) §7.2 and
§6.3, and [`07-environment.md`](07-environment.md). Citation convention as in
`01-requirements.md` §0: `VF §X` = `../hackathon/VERIFIED-AIR-FACTS.md`; `PC §X` =
`../hackathon/01-paradigm-comparison.md` r3; a bare `path:line` is the mlir-air clone at commit
`ff95a9b35b692b4cfbdf1d52bf69e6ffe01facde` (`/home/adi/Projects/Honours/mlir-air`).*

**New measurements in this document.** Everything marked **[M]** was run on 2026-09-12 against
the installed wheel `mlir_air 0.0.1.2026091204+ff95a9b` in the verifier's scratchpad venv
(Python 3.14.7, no `/dev/accel*`, no XRT). Everything else is source-read or copied from VF.
Nothing was written into the repository by those runs.

---

## 1. Purpose and FR IDs satisfied

M6 is the only module that shells out. It turns the AIR text M5 produced into an artifact, runs
that artifact on a device when one exists, and reports whether any of that worked — **without
trusting an exit code**, because `air-opt` prints `error:` and exits 0 (VF §B.6, §S11;
`mlir/lib/Util/Dependency.cpp:2063-2066`).

| FR | What M6 owes it |
|---|---|
| **FR-T1** | `build(target=)` accepts `npu1`/`npu2`/`auto`; anything else becomes `ToolchainError(TOOL-TARGET)` naming the two accepted values |
| **FR-T2** | drives `aircc --device <t> --output-format=none\|pdi` and treats both as off-device success |
| **FR-T3** | `output_format="xclbin"` with no `xclbinutil` fails with a three-part `ToolchainError(TOOL-MISSING-XCLBINUTIL)` |
| **FR-T4** | device compile + run + oracle diff with an explicit per-fixture tolerance; `DiffReport` names its denominator |
| **FR-T5** | every `air-opt`/`aircc` invocation captures stderr and treats any `error:` line as failure **regardless of exit status** |
| **FR-T6** | the pin is verified before anything else runs; a mismatch is `TOOL-VERSION-PIN` with the reinstall command |
| *(support)* | **FR-E7** (`s.emit(path)` round-trips through `air-opt`), **FR-S13/E3** (the `air-opt` pipeline that exposes `unroll = 2`), **FR-E5** (the `air-broadcast-detection` count), **NFR-4** (no raw MLIR diagnostic in a top-level message), **NFR-6** (no network at test time), **R-09/R-11/R-13/R-14** |

**Non-responsibilities.** M6 makes **no correctness claim**. `air-runner` is a timing model
(`docs/AIRRunner.md:3`; VF §S7), and every string M6 prints about it says so. M6 does not run
passes of its own choosing on the artifact path — the optimisation pipeline is the stock C++ one
inside `aircc` (`tools/aircc/aircc.cpp:905-995`), which we never modify (G3).

---

## 2. Public entry points / CLI surface

### 2.1 Python entry points (frozen in `06-interfaces.md` §7.2)

```
m6.artifact(mlir_path: str, target: Target,
            output_format: "none" | "pdi" | "xclbin") -> str      raises ToolchainError
m6.run(artifact: str, inputs: Sequence[ndarray]) -> list[ndarray]  raises ToolchainError
m6.diff(device: Sequence[ndarray], oracle: Sequence[ndarray],
        tol: float) -> DiffReport                                  raises nothing
m6.trace(mlir_path: str, model_json: str) -> str                   raises ToolchainError
```

`DiffReport = (matched: bool, max_abs_err: float, first_mismatch: tuple[int, ...] | None,
count_mismatched: int, total: int)`.

Four helpers are M6-internal but contractual for M7, because M7's tests call them directly:

```
m6.tool(name: str) -> pathlib.Path                 # resolve a native tool, bundled first
m6.invoke(argv: Sequence[str], cwd: str) -> ToolRun # the one subprocess chokepoint
m6.verdict(run: ToolRun) -> None                    # raises ToolchainError per FR-T5
m6.check_pin() -> None                              # raises ToolchainError(TOOL-VERSION-PIN)
m6.ir_facts(mlir_path, target, facts: Sequence[str]) -> dict[str, object]
m6.has_device() -> bool
```

`ToolRun` is a frozen dataclass `(argv, returncode, stdout, stderr, seconds)`. It is the only
thing any other module ever sees of a subprocess; a `CompletedProcess` never escapes M6.

`Schedule.build(target)` (`06-interfaces.md` §7.1) stays in M5 and does **not** call `aircc`; it
traces and verifies only (`python/air/api/_compile.py:106-160`, VF §D.9). M6 begins after that.

### 2.2 CLI surface we drive

Confirmed by running `--help` in the verifier's venv **[M]**. Only the flags below are used.

| Tool | Invocation | Evidence |
|---|---|---|
| `aircc` | `aircc --device <npu1\|npu2> --output-format <none\|pdi\|xclbin> --tmpdir <dir> [--peano <dir> --no-xchesscc --no-xbridge] [-o <out> -i <insts>] <in.mlir>` | `aircc --help` **[M]**: `--device=<string>`, `--output-format=<value>` with the five values `xclbin, txn, elf, pdi, none`, `--tmpdir=<string>`, `--peano=<string>`, `--no-xchesscc`, `--no-xbridge`, `-o <string>`, `-i <string>`, `--verbose`. `--output-format=xclbin` is the **default** (`tools/aircc/aircc.cpp:266-268`) and `--device`'s own default is `xcvc1902` (`aircc.cpp:165`), so **both flags are mandatory for us** |
| `air-opt` | `air-opt <in.mlir> -pass-pipeline='builtin.module(...)' -o <out.mlir>` | `air-opt --version` → `LLVM version 24.0.0` **[M]**; both invocations run clean in §6 below |
| `air-runner` | `air-runner <in.mlir> -f <func> -m <arch.json> -o <trace.json> [-g herd\|core] [-l single\|all] [-v]` | `air-runner --help` **[M]** lists exactly `-f <function>`, `-m <filename>`, `-o <filename>`; the other three are in `tools/air-runner/air-runner.cpp:44-68`. Defaults: `-m arch.json`, `-f graph`, `-g herd`, `-l all` |

Two spelling corrections to `07-environment.md` §4, found by running `--help` **[M]**:

1. The ping-pong opt-out is `--omit-ping-pong-transform[=<''\|L1\|L2\|all>]`, **not**
   `--omit-pingpong`. We never pass it; recorded so nobody copies the wrong flag from VF §E.1.
2. `--placed-ir-verifiers=<off|warn|error>` exists and defaults to `error`
   (`aircc.cpp:264` `cl::init(PIV_error)`; VF §S8). We never pass it either — the default is the
   strict one and that is what we want.

`--tmpdir` defaults to `air_project` **in the current working directory**
(`tools/aircc/aircc.cpp:88-90`). M6 always passes an explicit `--tmpdir` inside a caller-supplied
scratch directory, so that (a) `aircc` never writes into the repository, and (b) two smoke tests
cannot collide on one directory.

---

## 3. Internal design

### 3.1 Environment resolution

Tools are resolved the way `air.api` itself resolves them: the binary bundled in the wheel at
`<purelib>/mlir_air/bin/<tool>` first, `PATH` second
(`python/air/tools.py:39-58` — `resolve_tool`, `bundled_tool.is_file()` then `shutil.which`,
then `RuntimeError`). We reuse that function rather than reimplement it (VF "Install path" note;
07-environment §2).

`aircc` additionally shells out to `aiecc`, which is **not** bundled under `mlir_air/bin` — it
comes from `mlir_aie/bin` — and Peano must be locatable. Those two are the only environment
variables in the project.

```pseudo
CONST TOOLS_NEEDING_ENV = {"aircc"}

FUNCTION tool(name):
    # air.tools.resolve_tool: bundled wheel binary first, PATH second.
    TRY
        RETURN air.tools.resolve_tool(name)
    CATCH RuntimeError AS e:
        RAISE ToolchainError(code="TOOL-AIRCC-FAILED", clause=None,
            reason = f"MLIR-AIR tool {name!r} is not installed",
            fix    = "pip install --no-index --find-links vendor/wheels 'mlir_air[aie]'",
            details = {"searched": [bundled_path(name), "PATH"], "original": str(e)})

FUNCTION tool_env():
    # Returns the environment aircc needs, derived, never read from the user's shell.
    purelib = sysconfig.get_paths()["purelib"]
    env     = dict(os.environ)
    env["PATH"] = os.pathsep.join([purelib/"mlir_air"/"bin",
                                   purelib/"mlir_aie"/"bin",
                                   env.get("PATH", "")])
    peano = env.get("PEANO_INSTALL_DIR") OR (purelib/"llvm-aie")
    IF is_dir(peano): env["PEANO_INSTALL_DIR"] = peano
    RETURN env, peano
```

`air-opt`, `air-runner` and `import air` need **no** environment variables at all (VF §G.3,
measured; `07-environment.md` §2). Only `aircc` does. `tool_env()` is therefore applied to the
`aircc` invocation and to nothing else — which keeps `test_stderr_parser` and every IR-inspection
test free of environment setup.

Peano is passed **explicitly** as well as through the environment, mirroring what `XRTBackend`
does (`python/air/backend/xrt.py:556-566`: `--peano <dir> --no-xchesscc --no-xbridge`, with the
fallback branch printing a Vitis warning). Passing it explicitly means a stale
`PEANO_INSTALL_DIR` in someone's shell cannot silently route the build through Chess.

### 3.2 The one subprocess chokepoint, and the FR-T5 verdict

Every shell-out goes through `invoke`. Nothing else in the package calls `subprocess`.

```pseudo
FROZEN DATACLASS ToolRun: argv, returncode, stdout, stderr, seconds

FUNCTION invoke(argv, cwd, env=None, timeout=600):
    t0 = monotonic()
    TRY
        p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                           text=True, timeout=timeout)
    CATCH subprocess.TimeoutExpired:
        RAISE ToolchainError(code="TOOL-AIRCC-FAILED",
            reason = f"{argv[0]} did not finish within {timeout}s",
            fix    = "re-run with a smaller fixture, or raise the timeout in m6.invoke",
            details = {"argv": argv, "timeout_s": timeout})
    RETURN ToolRun(argv, p.returncode, p.stdout, p.stderr, monotonic() - t0)

# The FR-T5 rule, in one place. Order matters: the diagnostic is reported before
# the exit code, because the diagnostic is the informative half and the exit
# code is the half that lies.
ERROR_LINE = compile(
    r'^(?P<loc>(?:loc\("[^"]*":\d+:\d+\)|[^\s:]+:\d+:\d+):\s*)?error:\s*(?P<msg>.*)$',
    MULTILINE)
# The `loc` group carries two alternatives because the two tools spell diagnostics
# differently: `aircc` prints MLIR's loc(...) form, measured at P0c as
# `loc("malformed.mlir":15:39): error: use of undeclared SSA value name`
# (`tests/fixtures/stderr/exit1_with_error.txt`, line 1), while `air-opt` prints
# `<file>:<line>:<col>: error: ...`. The single-alternative pattern this document
# carried before CONTRACT_VERSION 4 matched no `aircc` diagnostic at all.

FUNCTION verdict(run):
    hits = [m.group(0) for m in ERROR_LINE.finditer(run.stderr)]
    IF hits:
        RAISE ToolchainError(code="TOOL-DIAGNOSTIC",
            reason  = f"{basename(run.argv[0])} printed {len(hits)} diagnostic(s)"
                      f" and exited {run.returncode}",
            fix     = FIX_FOR.get(classify(hits[0]), "read the diagnostics below and "
                                  "fix the emitted module"),
            details = {"argv": run.argv, "returncode": run.returncode,
                       "diagnostics": hits, "stderr": run.stderr})
    IF run.returncode != 0:
        RAISE ToolchainError(code = "TOOL-MISSING-XCLBINUTIL"
                                    IF "xclbinutil" IN run.stdout + run.stderr
                                    ELSE "TOOL-AIRCC-FAILED",
            reason  = f"{basename(run.argv[0])} exited {run.returncode} with no error: line",
            fix     = XCLBINUTIL_FIX IF "xclbinutil" IN ... ELSE
                      "re-run with --verbose and read the last aiecc stage",
            details = {"argv": run.argv, "returncode": run.returncode,
                       "stdout_tail": last_n_lines(run.stdout, 40),
                       "stderr_tail": last_n_lines(run.stderr, 40)})
```

Three properties of this design, each with a test in §7:

* **A clean run with `returncode == 0` and no `error:` line passes.**
* **`returncode == 0` with an `error:` line fails** — this is the case FR-T5 exists for, and it
  is the one a naive driver gets wrong. Reproduced **[M]** in §6.1.
* **`returncode != 0` with an `error:` line fails through the first branch**, so the message the
  user sees is the diagnostic, not "exit 1".

`classify()` maps a diagnostic to a fix hint with a small table, not with prose matching. The
table has three entries today and grows by a row, not by a branch:

| Substring in the diagnostic | Fix hint |
|---|---|
| `found channel op not in pairs` | "the module has an unmatched channel op; M4's balance self-check should have caught this — run `s.plan()` and read the per-key table" |
| `exceeded available memory` | "the herd's L1 working set does not fit; reduce a `tile` factor or drop a `double_buffer`" (mirrors `air.api`'s own `_annotate_l1_failure`, `python/air/api/_compile.py:268-283`) |
| `xclbinutil` | the FR-T3 message (§5) |

### 3.3 `artifact()` — the off-device and on-device build

```pseudo
CONST ACCEPTED_TARGETS  = ("npu1", "npu2", "auto")
CONST ACCEPTED_FORMATS  = ("none", "pdi", "xclbin")

FUNCTION artifact(mlir_path, target, output_format, workdir=None):
    check_pin()                                  # FR-T6, first, always
    IF target NOT IN ACCEPTED_TARGETS:
        RAISE ToolchainError(code="TOOL-TARGET",
            reason  = f"target {target!r} is not an AIR target",
            fix     = 'pass target="npu1" or target="npu2"',
            details = {"given": target, "accepted": ["npu1", "npu2", "auto"],
                       "why": "air.api's resolve_target accepts only these "
                              "(_trace.py:164-183); aircc's own --device default "
                              "xcvc1902 is a declared non-goal"})
    IF output_format NOT IN ACCEPTED_FORMATS:
        RAISE ToolchainError(code="TOOL-TARGET", ...)

    # G-12: m6.artifact is the ONLY place "auto" is resolved, and only outside a test (I-6).
    # The checker resolves "auto" internally to "npu2" with a recorded note and spawns no
    # subprocess (decision D-13, 03-lld-M3-checker.md §3.11); this call does spawn one.
    resolved = target IF target != "auto" ELSE air.backend.xrt.detect_target_device(
                                                    default="npu2")
    IF output_format == "xclbin" AND shutil.which("xclbinutil") IS None:
        RAISE xclbinutil_error()                 # FR-T3, §5 — before spending 2 minutes

    work = workdir OR mkdtemp(prefix="m6-")
    env, peano = tool_env()
    argv = [tool("aircc"), "--device", resolved,
                           "--output-format", output_format,
                           "--tmpdir", work/"air_project"]
    IF peano: argv += ["--peano", peano, "--no-xchesscc", "--no-xbridge"]
    IF output_format == "xclbin":
        out   = work/"air.xclbin"; insts = work/"insts.txt"
        argv += ["-o", out, "-i", insts]
    ELIF output_format == "pdi":
        out   = work/"air.pdi";   insts = work/"insts.txt"
        argv += ["--pdi-name", out, "-i", insts]
    ELSE:                                         # none
        out = work/"air_project"                  # the compile-only witness
    argv += [mlir_path]

    run = invoke(argv, cwd=work, env=env)
    verdict(run)                                  # FR-T5
    IF output_format != "none" AND NOT is_file(out):
        RAISE ToolchainError(code="TOOL-AIRCC-FAILED",
            reason="aircc reported success but produced no artifact",
            fix="re-run with --verbose", details={"expected": out, "argv": argv})
    RETURN str(out)
```

The flag order and the `-o`/`-i`/`--pdi-name` split are copied from `XRTBackend.compile`
(`python/air/backend/xrt.py:474-490`, `:540-566`), which is the shape upstream itself uses.
Two deliberate divergences:

1. **We pass `--tmpdir` and an absolute input path; upstream writes `air.mlir` into the process's
   cwd and passes a relative name** (`xrt.py:588-591`). Upstream's shape makes two concurrent
   compiles in one directory clobber each other, and leaves `air.mlir` and `air_project/` behind
   in whatever directory the test ran from.
2. **We do not pass `-o`/`-i` for `--output-format=none`.** Upstream's `else` branch does
   (`xrt.py:488-490`) because `none` shares a branch with `xclbin`. VF §G.5 measured
   `aircc --device npu1 --output-format=none design.mlir` with no `-o` and it exits 0.

### 3.4 Device detection and `requires_device`

```pseudo
FUNCTION has_device():
    # Cheap, no subprocess, no import of pyxrt. /dev/accel* is the amdxdna
    # character device; VF §G.3's probe machine had none.
    RETURN bool(glob("/dev/accel/accel*") OR glob("/dev/accel*"))

# tests/conftest.py (M7 owns the file; M6 owns the predicate)
FUNCTION pytest_runtest_setup(item):
    IF item.get_closest_marker("requires_device") AND NOT m6.has_device():
        skip("no NPU: /dev/accel* is absent, so level D cannot run "
             "(04-test-plan.md §1). This line is read aloud at the D7 rehearsal.")
    IF item.get_closest_marker("requires_aircc") AND NOT m6.tool_available("aircc"):
        skip("aircc is not installed; install the [aie] extra (07-environment.md §2)")
```

`has_device()` deliberately does **not** call `xrt-smi`. `air.api`'s `"auto"` does
(`python/air/api/_trace.py:182` → `python/air/backend/xrt.py:99-109`, a 10-second-timeout
subprocess), which is why D-13 makes every test pass an explicit target. When a test *must* pin
the generation without a probe, set `AIR_TARGET_DEVICE` — it short-circuits `detect_target_device`
before `xrt-smi` is reached (`xrt.py:88-97`), and it is the mechanism upstream's own lit configs
use.

### 3.5 `run()` — the device harness

**The harness reuses `XRTBackend`, not `air.api`'s `CompiledKernel`.** That is a deliberate
choice with a measured reason:

`CompiledKernel.__call__` (`python/air/api/_compile.py:318-347`) builds its argument list as
`fn(*args, *outputs)` where `outputs = [np.zeros(t.shape, ...) for t in self.launch.outputs]` and
`inputs = [t for t in self.tensors if not t.is_output]` (`:98-103`). A tensor the kernel writes
has `is_output = True` (set at `python/air/api/ops.py:310`), so it is **excluded from `inputs`
and handed a zero buffer**. W1's `C`, W2's `U` and W3's `S` are all read *and* written — `C` and
`S` want to start at zero, but **W2's `U` carries the initial field**, and through
`CompiledKernel` it would arrive on the device as zeros and produce a plausible, wrong answer.
`inout=True` does not fix it: it exempts the tensor from the inputs-then-outputs ordering rule
and nothing more (`python/air/api/_trace.py:1803-1815`), and it simultaneously removes the
tensor from `_check_interface`'s output set (`_compile.py:226-232`), so a kernel whose only
written tensor is `inout` fails to build at all.

`XRTBackend.load(artifact)` returns an `invoker(*arrays)` that maps **every** argument to a
buffer object, uploads **every** one, runs, and syncs **every** one back
(`python/air/backend/xrt.py:863-925`). That is exactly in-place semantics, so it is what we use.

```pseudo
FUNCTION run(artifact, inputs, target, kernel_name):
    IF NOT has_device():
        RAISE ToolchainError(code="TOOL-NO-DEVICE",
            reason="a device run was requested but /dev/accel* is absent",
            fix="run with output_format='none' for the off-device path",
            details={"looked_for": "/dev/accel*"})

    FROM air.backend.xrt IMPORT XRTBackend       # imports pyxrt lazily (xrt.py:717-727)
    backend = XRTBackend(target_device=target, output_format="xclbin",
                         omit_while_true_loop=False, instance_name=kernel_name)
    # inputs are already in the launch's tensor-declaration order; the xclbin
    # path caps the arity at five (xrt.py:863-867) and all three workloads use
    # three or fewer.
    ASSERT len(inputs) <= 5
    shapes = [a.shape for a in inputs]
    WITH filelock(tempdir/"npu.lock"):           # one device, N pytest workers
        invoke_fn = backend.load(XRTCompileArtifact(artifact, kernel_name, insts_of(artifact)))
        TRY
            flat = invoke_fn(*[ascontiguousarray(a) for a in inputs])
        FINALLY
            backend.unload()                     # release order matters (xrt.py:946-958)
    # XRT hands buffers back flat; restore the declared shape.
    RETURN [asarray(r).reshape(s) for r, s in zip(flat, shapes)]
```

The `filelock` on `<tmp>/npu.lock` is upstream's own convention for serialising device access
(`python/air/api/_compile.py:339`) and `filelock` is already an installed transitive dependency
of the wheel, so it adds nothing to NFR-2's list.

### 3.6 `diff()` — oracle versus device (FR-T4, and the Q-4 resolution)

```pseudo
FUNCTION diff(device, oracle, tol):
    total = mismatched = 0; max_abs = 0.0; first = None
    FOR d, o IN zip(device, oracle):
        d = asarray(d, dtype=float64); o = asarray(o, dtype=float64)
        IF d.shape != o.shape:
            RETURN DiffReport(False, inf, None, o.size, o.size)   # shape is a mismatch
        err  = abs(d - o)
        bad  = err > tol
        total       += o.size
        mismatched  += int(bad.sum())
        max_abs      = max(max_abs, float(err.max(initial=0.0)))
        IF first IS None AND bad.any(): first = tuple(argwhere(bad)[0])
    RETURN DiffReport(mismatched == 0, max_abs, first, mismatched, total)
```

`diff` raises nothing (06-interfaces §7.2) — a mismatch is data, not an exception, so a device
test can report the number rather than a traceback. The caller asserts `matched` and prints
`max_abs_err` with `count_mismatched` **of** `total`, which is the "every count names its
denominator" rule.

**Q-4 — RESOLVED. Integer-valued fixtures for the device diff, `tol = 0.0` everywhere except
W2, where `tol = 1e-5` absolute; no `bf16` device fixture exists, so the question's literal
subject does not arise.** The reasoning, which is the part that matters:

| Fixture | dtype | Values | Device `tol` | Why it is exact, or why it is not |
|---|---|---|---|---|
| **W1**, **W1-flip** | `f32` | integers drawn from `[-8, 8)` by `default_rng(seed).integers(-8, 8)` then `astype(f32)` | **0.0** | every product is an integer with `\|p\| ≤ 64`; the sum over `K = 64` terms has `\|C\| ≤ 4096`. Every integer of magnitude `< 2^24` is exactly representable in `f32` and `f32` addition of such integers is exact, so the result is **independent of summation order** — which is the property that matters, because the flip reassociates the reduction across four PEs and a cascade. For the large smoke fixture (`K = 256`) the bound is `64 · 256 = 16 384`, still far under `2^24` |
| **W3** | `i32` | scores | **0.0** | integer arithmetic; `max` is exact |
| **W2** | `f32` | integers in `[-8, 8)` | **1e-5** absolute, `rtol = 0` | the kernel multiplies by `0.2`, which is **not** a dyadic rational, so no choice of integer inputs makes the result exactly representable. `\|U\| ≤ 8` throughout (the update is a convex combination scaled by `1.0`), `ulp(8) = 2^-20 ≈ 9.5e-7`, and `T = 4` timesteps × 6 roundings each bounds the accumulated error well below `1e-5`. The number is stated, not fitted |
| **bf16 anything** | — | — | — | **no bf16 fixture is shipped.** `bf16` appears in exactly one place in the suite: FR-L9's `L1-CAPACITY` negative test, which is a *checker* test that never emits IR, never reaches `aircc` and never reaches a device. Q-4 asked for a `bf16` W1 tolerance; the answer is that no requirement produces one, and inventing one would be an unmeasured number on a slide |

The tolerance lives in `meta.json`'s `tol` field (06-interfaces §9), so `diff` never chooses it —
the fixture does. `test_T4_device_diff` asserts `report.matched` and prints
`max_abs_err = …, mismatched = 0 of 4096`.

### 3.7 `ir_facts()` — the `air-opt` inspection pipeline (Q-5 resolution)

This is the FR-S13/FR-E3/FR-E5 path and the producer of `*.ir_facts.json` (D-8).

```pseudo
# One table, so that "which pipeline produced this number" is never a question.
PIPELINES = {
  "pingpong":  "builtin.module(air-dependency,"
               "air-label-scf-for-to-ping-pong{{device={target}}})",
  "pingpong_full":
               "builtin.module(air-dependency,air-annotate-packet-ids{{assign=true}},"
               "air-hoist-dma-in-accum-pattern,air-dma-to-channel,canonicalize,cse,"
               "air-dependency-canonicalize,canonicalize,cse,"
               "air-isolate-async-dma-loop-nests{{scope=launch}},canonicalize,cse,"
               "air-fuse-channels,canonicalize,cse,func.func(air-fuse-alloc-dealloc),"
               "func.func(air-shrink-memref-sizes-by-access),"
               "air-label-scf-for-to-ping-pong{{device={target}}})",
  "transform": PIPELINES["pingpong_full"][:-1] + ",air-ping-pong-transform,canonicalize,cse)",
  "broadcast": "builtin.module(air-dependency,air-broadcast-detection)",
  "pairs":     "builtin.module(air-dependency,air-dependency-canonicalize)",
  "aie":       "builtin.module(air-place-herds{{num-rows={rows} num-cols={cols} "
               "row-anchor={row_anchor} col-anchor=0}},air-to-aie{{device={target}}})",
}

EXTRACTORS = {
  "pingpong_unroll":         lambda txt: max_int(findall(r"unroll = (\d+) : i32", txt), default=0),
  "hoist_alloc_count":       lambda txt: txt.count("hoist_alloc = true"),
  "broadcast_pattern_count": lambda txt: txt.count("broadcast_pattern"),
  "cascade_channels":        lambda txt: txt.count("aie.cascade_flow"),   # R-F-3, from "aie"
  "pingpong_iter_args":      lambda txt: max_count(r"!air\.async\.token", per_scf_for(txt)),
  "lock_init_histogram":     lambda txt: histogram(findall(r"init = (\d+)", txt)),
}

FUNCTION ir_facts(mlir_path, target, facts, workdir):
    out = {}
    FOR pipeline_name, fact_names IN group_by_pipeline(facts):
        pipe = PIPELINES[pipeline_name].format(target=target)
        dst  = workdir / f"{pipeline_name}.mlir"
        run  = invoke([tool("air-opt"), mlir_path, f"-pass-pipeline={pipe}", "-o", dst],
                      cwd=workdir)
        verdict(run)                              # FR-T5 applies here too
        txt = read(dst)
        FOR f IN fact_names: out[f] = EXTRACTORS[f](txt)
        out[f"_pipeline_{pipeline_name}"] = pipe  # recorded in the golden, see below
    RETURN out
```

*Erratum, 2026-09-13 (ruling **R-F-3**).* `cascade_channels` counts **`aie.cascade_flow` ops
after the `aie` pipeline**, which is what `03-lld-M5-emitter.md` §3.8's own table specifies, and
`PIPELINE_OF["cascade_channels"]` is therefore `"aie"` and not `"pingpong"`. Counting the
`channel_type = "npu_cascade"` **string** answers a different question: one bundle of
`size=(PK-1,)` prints the attribute once before lowering, and `air-to-aie` splits it into three
`@channel_N [1, 1]` bundles beside the surviving `@CascadeK [3]` declaration, so the string
occurs **4** times in the lowered W1-flip and `aie.cascade_flow` occurs **3** — the number of
physical links, which is the fact FR-M6 and FR-E8 are about (`03-lld-B-open-questions.md` §4).
Measured on `vendor/probes/review/ap_asc/aie.flip1d_asc.mlir` and again on our own emitted
module. Two consequences, both recorded in `design/PROGRESS-B.md` phase P6: the fact is not
readable for **W1**, whose module does not lower through `air-place-herds,air-to-aie` alone
(its `C2L3` bundle index goes through the `repeats` strip-mine `affine_map`, which the pass
cannot fold, and it then fails *'air.channel.get' op failed to get MM2S tile for L3
allocation*); and the `aie` pipeline gained its `air-place-herds` prefix at P4 for the same
family of reasons, which the table above now carries.

**Q-5 — RESOLVED, measured [M]. The short pipeline suffices: use
`builtin.module(air-dependency,air-label-scf-for-to-ping-pong{device=npu1})`.** Run on VF §E.5's
own probe module (`air-probe/e4.mlir`, the `air.api` GEMM at `M=N=K=256`, `TM=TN=TK=64`):

| Pipeline | exit | stderr | `unroll = 2` hits | `hoist_alloc` hits |
|---|---|---|---|---|
| VF §E.5's full 16-pass prefix | 0 | empty | **1** | **2** |
| `air-dependency,air-dma-to-channel,air-label-scf-for-to-ping-pong{device=npu1}` | 0 | empty | **1** | **2** |
| `air-dependency,air-label-scf-for-to-ping-pong{device=npu1}` | 0 | empty | **1** | **2** |

All three agree, so the two-pass form is what `test_I_pingpong_labels` uses. Two supporting
notes: `air-dma-to-channel` is a no-op for **our** modules, because our emitter writes
`air.channel.put/get` directly and that pass converts `air.dma_memcpy_nd`
(`tools/aircc/aircc.cpp:924`; VF §H.3); and `air-dependency` is *not* optional, because
`air-label-scf-for-to-ping-pong` requires an `air.execute`-wrapped `memref.alloc` as a direct
child, i.e. the IR must already be asynchronous (`mlir/include/air/Transform/Passes.td:970-971`;
VF §E.2 item 2).

`PIPELINES["pingpong_full"]` is kept, unused, as the **recorded fallback**: if the short form
ever reports `pingpong_unroll == 0` on a module whose plan asserts `double_buffer`, the test's
failure message says "retry with `pingpong_full`; if that finds the label, the difference is
`air-fuse-alloc-dealloc` and R-05 has bitten". Each `ir_facts.json` stores the pipeline string
it was produced by (`_pipeline_<name>`), so a golden can never be compared against a number
produced by a different pipeline.

Also measured **[M]** on the same module: `builtin.module(air-dependency,air-broadcast-detection)`
exits 0 with one `broadcast_pattern` attribute — the "bonus, unasked for" of VF §E.5, now with a
number. Our emitted W1 declares `broadcast_shape` explicitly and should therefore have a
*different* count; whatever it is, it is frozen in the golden, which is what R-04 asks for.

### 3.8 `trace()` — `air-runner`, and why it is not in the pitch

```pseudo
FUNCTION trace(mlir_path, model_json, function, workdir):
    # air-runner's input must (a) contain an air.launch and (b) be asynchronous.
    async_ir = workdir/"async.mlir"
    verdict(invoke([tool("air-opt"), mlir_path,
                    "-pass-pipeline=builtin.module(air-dependency)",
                    "-o", async_ir], cwd=workdir))
    IF "air.launch" NOT IN read(async_ir):
        RAISE ToolchainError(code="TOOL-AIRCC-FAILED",
            reason="air-runner needs an air.launch and this module has none",
            fix="emit air.launch (FR-E1); air-runner dereferences the LaunchOp "
                "without a null check and segfaults otherwise",
            details={"citation": "mlir/lib/Util/Runner.cpp:547-551"})
    out = workdir/"trace.json"
    run = invoke([tool("air-runner"), async_ir, "-f", function,
                  "-m", model_json, "-o", out], cwd=workdir)
    verdict(run)
    RETURN str(out) + "   # PERFORMANCE MODEL, NOT A CORRECTNESS ORACLE (VF §S7)"
```

**Which JSON ships in the wheel? None.** Searched the whole installed tree **[M]**:
`find <purelib> -name 'arch*.json'` returns nothing, and `find <purelib>/mlir_air -name '*.json'`
returns nothing. The only resource models in the project are the sixteen `arch.json` files under
`mlir/test/Util/Runner/` **in the source clone**, none of which ships. The canonical one,
`mlir/test/Util/Runner/arch.json`, describes `"devicename": "testdevice"` with
`"dus": {"count": [4, 4], …, "tiles": {"count": [1, 4], "memory": {"L1", 32768}}}` — a **32 KB**
L1, against the 65 536 B `air.api` budgets for a real AIE core (`python/air/api/_trace.py:100`).
It is not `npu1` and it is not `npu2`.

Three measurements that together decide the demo question **[M]**:

1. **Positive control.** `air-runner mlir/test/Util/Runner/channel.mlir -f test -m arch.json -o t.json`
   → exit 0, `Latency (all-iterations mode): 9.067us`, a 442 KB Chrome trace whose process names
   are `air.launch` and `air.segment[4, 4]`. The tool works.
2. **A module with no `air.launch` segfaults.** Running it on VF §E.5's probe module (a
   `func.func @gemm` containing `air.herd` directly, no launch) crashes in
   `xilinx::air::LaunchOp::getSizeOperands()` called from
   `AIRRunner::AIRRunner_impl::scheduleFunction`. The source is a `dyn_cast_if_present` whose
   result is dereferenced unconditionally (`mlir/lib/Util/Runner.cpp:547-551`). Our emitter always
   emits `air.launch` (FR-E1), so this does not bite us — but the guard above is one line and the
   failure mode is a stack dump in front of a jury.
3. **Our compute would be priced at nothing.** `docs/AIRRunner.md` costs an op three ways — a
   per-datatype rate for a `linalg` body, a constant for `air.custom`, or a `cycles` expression —
   and states that *"scalar ops in a `linalg` body that the rate model does not know are counted
   as free"*. FR-E2 makes our per-PE bodies native `air.sequential` loops emitting
   `memref.load`/`memref.store`/`arith`, with **no `linalg` op at all**. So a trace of our module
   would show channel and DMA time against ~zero compute time.

**Decision (M6's call, recorded here so nobody reopens it at D6): `m6.trace` is implemented and
tested, and the trace is NOT shown in the five-minute pitch.** It would require presenting a
`testdevice` model as if it were an NPU, on a module whose compute the model prices at zero, in a
talk that claims no performance numbers (G5, R-16). `05-work-breakdown.md` §4 already lists it as
a stretch item whose cut cost is "a nice visual, nothing more"; this is the evidence for cutting
it. It survives as one line on the honest-limits slide: *"mlir-air ships `air-runner`; it is a
timing model, it prices `linalg` bodies and ours are scalar loops, and no NPU resource model
ships with the wheel — so we do not show a trace."* If a judge asks, that sentence is the answer,
and it is stronger than a trace would have been.

### 3.9 `check_pin()` — FR-T6

```pseudo
PIN = {"mlir_air": "0.0.1.2026091204+ff95a9b",
       "mlir_aie": "1.4.3.dev55+g10767b5",
       "llvm-aie": "22.0.0.2026091201+386ca5c6"}

FUNCTION check_pin(strict=True):
    got = {name: importlib.metadata.version(name) for name IN PIN}
    bad = {n: (PIN[n], got[n]) for n IN PIN IF got[n] != PIN[n]}
    IF NOT bad: RETURN
    RAISE ToolchainError(code="TOOL-VERSION-PIN",
        reason  = "the installed toolchain is not the pinned one; "
                  "every golden file is valid only for the pin",
        fix     = "pip install --no-index --find-links vendor/wheels 'mlir_air[aie]'",
        details = {"expected": PIN, "installed": got, "mismatched": bad})
```

M7 calls `check_pin()` once per session and turns a mismatch into a **skip** for every golden and
IR-fact test, not a failure (06-interfaces §8 rule 2) — a wrong wheel is an environment problem,
and a red suite would hide the real one. `test_T6_versions` itself **fails** on a mismatch, so
exactly one test goes red and it is the one that explains why.

### 3.10 Wheel pinning and the sha256 cache (R-13)

`07-environment.md` §2 says to do this at D0 and leaves four blank sha256 lines. The procedure,
so that it is a checklist and not a judgement call:

```pseudo
# D0, on the machine that already has a working install. Network is allowed
# here and only here (NFR-6 disables it after the install step).
STEP 1  mkdir -p vendor/wheels
STEP 2  pip download 'mlir_air[aie]' numpy \
          -d vendor/wheels \
          --only-binary=:all: \
          -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels \
          -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4 \
          -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
STEP 3  sha256sum vendor/wheels/*.whl | tee vendor/wheels/SHA256SUMS
STEP 4  paste the four lines into 07-environment.md §2 and commit BOTH in one commit
STEP 5  verify the cache actually stands alone, on a second machine:
          python -m venv v && v/bin/pip install --no-index \
            --find-links vendor/wheels 'mlir_air[aie]'
          v/bin/air-opt --version && v/bin/aircc --help
STEP 6  from D1 onward every install, CI included, is STEP 5's command. The -f
        URLs never appear again in the project.
```

Notes that make the difference between this working and not:

* **`--only-binary=:all:`** — a source distribution would defeat the whole point.
* **Wheels are platform- and Python-tagged.** `pip download` on one machine fetches only that
  machine's tags. If the three team machines differ in Python minor version or OS, run STEP 2 on
  each and merge the directory; the sha256 list then has more than four lines and
  `07-environment.md` §2 records one line per file.
* **`vendor/wheels/` is ~2.3 GB** (VF §G.3). It must be in `.gitignore` and distributed out of
  band — a shared drive, or a USB stick at the D0 stand-up. `SHA256SUMS` **is** committed; the
  wheels are not. `07-environment.md` §2's phrasing ("Download the four wheels to
  `vendor/wheels/`") should be read as "to each machine's `vendor/wheels/`", and the committed
  artifact is the checksum file. **Flagged for the architect** as a one-line edit to
  `07-environment.md`.
* **Verify before trusting**: `sha256sum -c vendor/wheels/SHA256SUMS` is the first line of the
  install instructions from D1 on.

**Was this run now?** No. It is a network download of ~2.3 GB into the repository's tree, which
is outside the scratchpad and outside this document's remit. The scratchpad venv the
measurements above used was installed from the live `-f` URLs by the verifier on 2026-09-12 and
already resolves to the pinned versions (VF §G.3), so STEP 2 at D0 will fetch the same four
files — but that is an expectation, not a measurement, until someone runs it.

---

## 4. Invariants

| # | Invariant | Enforced by |
|---|---|---|
| **I-1** | No module outside M6 calls `subprocess`, and no `CompletedProcess` escapes M6 | `test_no_subprocess_outside_m6`, an AST lint over `spatial/` |
| **I-2** | Every tool invocation passes through `invoke` and every `invoke` result passes through `verdict` before its output is read | `test_stderr_parser`; an AST lint asserting `invoke(` and `verdict(` are paired in `m6_tools.py` |
| **I-3** | A tool is failed if stderr matches `error:`, **whatever the exit code**; the exit code is only consulted when stderr is clean | FR-T5, `test_T5_error_without_exit_code` |
| **I-4** | `artifact()` validates `target` and `output_format` and checks for `xclbinutil` **before** spending the two minutes of an `aircc` run | `test_T3_xclbin_message` asserts the error arrives in under a second |
| **I-5** | `aircc` never writes into the repository: `--tmpdir` is always inside a caller-supplied scratch directory, and `cwd` is that directory | `test_aircc_leaves_no_files` — a golden run with the repo root as cwd leaves `git status` clean |
| **I-6** | No test spawns `xrt-smi`. `target` is always literal at a call site in `tests/` | D-13; an AST lint over `tests/` for `target="auto"` |
| **I-7** | `m6.diff` raises nothing and always reports `count_mismatched` **of** `total` | signature, `test_diff_denominator` |
| **I-8** | Anything M6 prints about `air-runner` carries the words "timing model, not a correctness oracle" | `test_trace_disclaimer` greps the returned string and the honest-limits slide |
| **I-9** | `check_pin()` runs before any golden or IR-fact comparison in a session | M7's session fixture ordering; `test_T6_versions` |
| **I-10** | Every `ToolchainError` carries a `code` from 06-interfaces §6.3, a non-empty `reason` and `fix`, and `details` that names the `argv` | FR-D1, `test_D1_schema` |

---

## 5. Error paths

Six codes, all already in the frozen catalogue (06-interfaces §6.3). Rendering is the four-part
shape of HLD §4.2. **Toolchain failures are made legible by parsing diagnostics, never by
reporting an exit code** — the exit code appears only in `details`.

| Code | Raised when | `reason` | `fix` | `details` |
|---|---|---|---|---|
| `TOOL-TARGET` | `target ∉ {npu1, npu2, auto}`, or `output_format` outside its three values | `target 'xcvc1902' is not an AIR target` | `pass target="npu1" or target="npu2"` | `given`, `accepted`, `why` (the `_trace.py:164-183` citation) |
| `TOOL-MISSING-XCLBINUTIL` | `output_format="xclbin"` and `shutil.which("xclbinutil") is None`; also raised post-hoc when `aircc`'s output mentions `xclbinutil` | `xclbin packaging needs xclbinutil and it is not on PATH` | `install XRT, or build with output_format="none" or "pdi" — both are verified to work with no XRT and no device` | `looked_for: "xclbinutil"`, `provided_by: "XRT (Xilinx Runtime)"`, `alternatives: ["none", "pdi"]`, `evidence: "VF §G.5 — aircc fails at step 40/41 with tool 'xclbinutil' not found in search paths or PATH"` |
| `TOOL-AIRCC-FAILED` | non-zero exit **with clean stderr**, a timeout, a missing artifact, or a missing tool | `aircc exited 1 with no error: line` | `re-run with --verbose and read the last aiecc stage` | `argv`, `returncode`, `stdout_tail`, `stderr_tail` (40 lines each, so a 22-stage aiecc log does not become the message) |
| `TOOL-DIAGNOSTIC` | **any** `error:` line on stderr, *regardless of exit status* — FR-T5 | `air-opt printed 1 diagnostic and exited 0` | from `classify()`'s table (§3.2) | `argv`, `returncode`, `diagnostics` (the matched lines), `stderr` |
| `TOOL-VERSION-PIN` | `importlib.metadata.version` differs from `PIN` | `the installed toolchain is not the pinned one; every golden file is valid only for the pin` | the `--no-index --find-links vendor/wheels` command | `expected`, `installed`, `mismatched` |
| `TOOL-NO-DEVICE` | `run()` with no `/dev/accel*` | `a device run was requested but /dev/accel* is absent` | `run with output_format='none' for the off-device path` | `looked_for` |

**What is never raised to the user** (HLD §4.3): a `CalledProcessError`, a `FileNotFoundError`
from a tool path, an `AirBackendError` from `XRTBackend`, or a pyxrt exception. Each is caught
and re-raised as a `ToolchainError` with the original text preserved in
`details["original"]` — `air.api`'s own messages are good (`_annotate_l1_failure` explains an L1
overflow in terms of tile size, `python/air/api/_compile.py:268-283`) so they are kept, not
discarded.

**The three-part FR-T3 message, verbatim**, because it is the one a judge may see:

```
TOOL-MISSING-XCLBINUTIL: xclbin packaging needs xclbinutil and it is not on PATH
  in clause: build(output_format="xclbin")
  because:   aircc's last step (41 of 41) packages the design into an .xclbin
             container by calling xclbinutil, which is part of XRT (the Xilinx
             Runtime). It is a packaging tool, not a device: compilation,
             placement and codegen all completed.
  fix:       build with output_format="none" or output_format="pdi" — both are
             measured to work with no XRT and no device — or install XRT.
```

---

## 6. Worked examples

### 6.1 The exit-code trap, reproduced [M]

The whole of FR-T5 rests on one observable. Re-run on the installed wheel against
`air-probe/e3_i.mlir` (one `air.channel.put`, no matching `get`):

```
$ air-opt e3_i.mlir -pass-pipeline='builtin.module(air-dependency,air-dependency-canonicalize)' \
      -o /dev/null ; echo "exit=$?"
e3_i.mlir:9:7: error: 'air.channel.put' op found channel op not in pairs
      air.channel.put @C[%c0b, %c0b] (%alloc[] [] []) : (memref<32xi32, 1>)
      ^
e3_i.mlir:9:7: note: see current operation: %3 = "air.channel.put"(...)
exit=0
```

`verdict()` on that `ToolRun` raises `TOOL-DIAGNOSTIC` with
`reason = "air-opt printed 1 diagnostic and exited 0"` and
`fix = "the module has an unmatched channel op; M4's balance self-check should have caught this
— run s.plan() and read the per-key table"`.

A second recorded sample, for the parser's other branch: a module with a genuine parse error
(`air-probe/e3_i_unmatched_put.mlir`, which has an undeclared SSA name) gives
`exit=1` **and** `error: use of undeclared SSA value name`, so the first branch fires and the
user sees the parse error rather than "exit 1". Both samples are committed as
`tests/fixtures/stderr/{exit0_with_error,exit1_with_error,clean}.txt` and are what
`test_stderr_parser` feeds the parser — no subprocess, so the test is instant and runs in the
default suite.

### 6.2 W1 through the off-device path

```
m6.check_pin()                                  # 0.01 s
path = s.emit(tmp/"w1.npu1.air.mlir")           # M5
m6.artifact(path, target="npu1", output_format="none", workdir=tmp)
  → aircc --device npu1 --output-format none --tmpdir <tmp>/air_project \
           --peano <purelib>/llvm-aie --no-xchesscc --no-xbridge <tmp>/w1.npu1.air.mlir
  → ToolRun(returncode=0, stderr="")  → verdict passes  → returns "<tmp>/air_project"
m6.ir_facts(path, "npu1", ["pingpong_unroll", "hoist_alloc_count",
                           "broadcast_pattern_count"], workdir=tmp)
  → {"pingpong_unroll": 2, "hoist_alloc_count": 2, "broadcast_pattern_count": <golden>,
     "_pipeline_pingpong": "builtin.module(air-dependency,air-label-scf-for-to-ping-pong{device=npu1})",
     "_pipeline_broadcast": "builtin.module(air-dependency,air-broadcast-detection)"}
```

The `pingpong_unroll: 2` line is the one shown in the pitch (05-wbs §5 item 5): *our emitter
produced the loop shape the upstream pass requires, and the pass says so in its own IR.*

### 6.3 W1 on a device, if one arrives

```
art = m6.artifact(path, "npu1", "xclbin", workdir=tmp)      # needs xclbinutil
out = m6.run(art, [A, B, C_zeros], target="npu1", kernel_name="gemm")
rep = m6.diff([out[2]], [C_oracle], tol=0.0)
print(f"max_abs_err={rep.max_abs_err} mismatched={rep.count_mismatched} of {rep.total}")
→ max_abs_err=0.0 mismatched=0 of 4096
```

`out[2]` is `C`: `run` returns every buffer in the launch's tensor-declaration order, because the
invoker syncs every buffer back (`python/air/backend/xrt.py:918-925`). With no device the same
call raises `TOOL-NO-DEVICE` and `test_T4_device_diff` skips with that reason printed.

---

## 7. Tests table

Owner column: the person who writes the test. M6's own tests are C's.

| Test id | Level | Mark | FR / invariant | Asserts |
|---|---|---|---|---|
| `test_T1_targets` | U | — | FR-T1 | `npu1` and `npu2` both succeed with no NPU; `"xcvc1902"` raises `ToolchainError(TOOL-TARGET)` whose message names `npu1` and `npu2`; `"auto"` is accepted by the signature but never used in a test (D-13) |
| `test_T2_aircc_none` | S | `slow`, `requires_aircc` | FR-T2 | W1/W1-flip/W2/W3 × `{npu1, npu2}` through `--output-format=none`: exit 0 **and** no `error:` line. One variant also through `--output-format=pdi`, asserting the `.pdi` file exists |
| `test_T3_xclbin_message` | U | — | FR-T3, I-4 | with `xclbinutil` absent, the error names `xclbinutil`, names XRT, and names `none`/`pdi`; and it is raised in < 1 s (i.e. before `aircc` ran). Skipped when XRT is present |
| `test_T4_device_diff` | D | `requires_device` | FR-T4 | `DiffReport.matched`; prints `max_abs_err` and `count_mismatched` of `total` |
| `test_T5_error_without_exit_code` | U | — | FR-T5, I-3 | the recorded `exit0_with_error` sample produces a failure verdict |
| `test_T6_versions` | U | — | FR-T6, I-9 | `importlib.metadata.version("mlir_air") == PIN["mlir_air"]`, and the failure message contains the reinstall command |
| `test_stderr_parser` | U | — | FR-T5, I-2 | three recorded samples → three verdicts: clean passes, `exit0_with_error` fails, `exit1_with_error` fails **through the diagnostic branch** (the message is the diagnostic, not the exit code) |
| `test_I_pingpong_labels` | I | `requires_air_opt` | FR-S13, FR-E3 | `ir_facts(W1, "npu1", ["pingpong_unroll"])["pingpong_unroll"] == 2`, against `w1.npu1.ir_facts.json` |
| `test_I_pingpong_transform` | I | `requires_air_opt` | FR-E3 | the `transform` pipeline doubles the K loop's step and yields four `!air.async.token` `iter_args` |
| `test_I_broadcast_count` | I | `requires_air_opt` | FR-E5, R-04 | `broadcast_pattern_count` equals the golden, for all four variants |
| `test_I_unpaired_channel` | I | `requires_air_opt` | FR-T5 | the committed broken module produces `found channel op not in pairs` on stderr **and exit 0**; characterisation test of the toolchain, not of us |
| `test_I_lock_inits` *(optional, D6)* | I | `requires_air_opt`, `slow` | VF §C.7 | `lock_init_histogram` matches the golden; no producer lock has `init = 0` |
| `test_aircc_leaves_no_files` | S | `slow`, `requires_aircc` | I-5 | a smoke run with the repo root as cwd leaves `git status --porcelain` empty |
| `test_no_subprocess_outside_m6` | U | — | I-1 | AST lint over `spatial/`: `subprocess` is imported only by `m6_tools.py` |
| `test_diff_denominator` | U | — | I-7 | a hand-built mismatch reports the right `count_mismatched` and `total`, and `diff` raises nothing on a shape mismatch |
| `test_trace_disclaimer` | U | — | I-8 | `m6.trace`'s return value and `demo/honest_limits.md` both contain "timing model" and "not a correctness oracle" |
| `test_trace_requires_launch` | U | — | §3.8 note 2 | a module with no `air.launch` raises `ToolchainError` naming `Runner.cpp:547-551`, rather than segfaulting a child process |
| `test_no_auto_target_in_tests` | U | — | I-6, D-13 | AST lint over `tests/`: no literal `"auto"` reaches a `target=` argument |

---

## 8. Dependencies

### 8.1 CLI flags, with `--help` evidence [M]

| Flag | Tool | Confirmed by |
|---|---|---|
| `--device=<string>` | `aircc` | `aircc --help`; default `xcvc1902` at `tools/aircc/aircc.cpp:165` |
| `--output-format=<xclbin\|txn\|elf\|pdi\|none>` | `aircc` | `aircc --help` lists all five with one-line descriptions; enum at `tools/aircc/aircc.cpp:266` |
| `--tmpdir=<string>` | `aircc` | `aircc --help`; default `air_project` at `tools/aircc/aircc.cpp:88-90` |
| `--peano=<string>`, `--no-xchesscc`, `--no-xbridge` | `aircc` | `aircc --help`; the same trio is passed by `python/air/backend/xrt.py:556-560` |
| `-o <string>`, `-i <string>`, `--pdi-name=<string>`, `--verbose` | `aircc` | `aircc --help`; the `-o`/`-i`/`--pdi-name` split is `python/air/backend/xrt.py:481-490` |
| `--omit-ping-pong-transform[=<''\|L1\|L2\|all>]` | `aircc` | `aircc --help` — **not** `--omit-pingpong`; never passed |
| `--placed-ir-verifiers=<off\|warn\|error>` | `aircc` | `aircc --help`; default `error` (`aircc.cpp:264`); never passed |
| `-pass-pipeline=<string>`, `-o <file>`, `--version` | `air-opt` | `air-opt --version` → `LLVM version 24.0.0`; both pipelines in §3.7 run clean |
| `-f <function>`, `-m <filename>`, `-o <filename>` | `air-runner` | `air-runner --help`; `-g <herd\|core>`, `-l <single\|all>`, `-v` at `tools/air-runner/air-runner.cpp:56-68` with defaults `arch.json`, `graph`, `herd`, `all` |

### 8.2 Files in the clone

| Concern | `path:line` |
|---|---|
| tool resolution, bundled first then `PATH` | `python/air/tools.py:39-58` |
| `resolve_target` accepts `None`/`auto`/`npu1`/`npu2` only | `python/air/api/_trace.py:164-183` |
| `NO_DEVICE_TARGET = "npu2"`; `PHYSICAL_HERD`; `L1_BYTES = 65536` | `python/air/api/_trace.py:88-91`, `:96`, `:100` |
| `AIR_TARGET_DEVICE` short-circuits the `xrt-smi` probe | `python/air/backend/xrt.py:88-97`; the probe itself at `:99-109` |
| `build()` runs `_check_interface` then `module.operation.verify()`; no pass pipeline | `python/air/api/_compile.py:106-160`, `:157`, `:226-240` |
| `mlir()` is `str(self.build())` | `python/air/api/_compile.py:242` |
| a written tensor gets `is_output = True` | `python/air/api/ops.py:310`; initialised `False` at `python/air/api/_value.py:376` |
| `inout=True` exempts only the ordering rule | `python/air/api/_trace.py:1803-1815` |
| `CompiledKernel.__call__` zeroes every output buffer | `python/air/api/_compile.py:318-347` |
| `_annotate_l1_failure` — keep its text | `python/air/api/_compile.py:268-283` |
| the `aircc` argv upstream builds | `python/air/backend/xrt.py:474-566` |
| `subprocess.run` + `AirBackendError` on non-zero exit | `python/air/backend/xrt.py:593-600` |
| `pyxrt` imported lazily inside `load()` | `python/air/backend/xrt.py:717-727` |
| the xclbin load path, the 5-argument cap, the invoker's upload/sync-back | `python/air/backend/xrt.py:836-861`, `:863-867`, `:868-925` |
| `unload()` release order | `python/air/backend/xrt.py:946-958` |
| the Python `aircc` driver is retired | `python/air/compiler/aircc/main.py:10-11` — **the wheel uses the C++ binary**, so M6 drives that, not a Python API |
| `air.compiler.util.Runner` (the Python air-runner wrapper) | `python/air/compiler/util.py:64-131` — accepts a JSON string, object, or filename |
| `air-runner` dereferences the `LaunchOp` with no null check | `mlir/lib/Util/Runner.cpp:543-551` |
| `air-runner` is a performance simulator; unbalanced pairs stall rather than fail | `docs/AIRRunner.md:3`; `mlir/lib/Util/Runner/RunnerNode.cpp:1447` |
| the only resource models in the project (**none ship in the wheel**) | `mlir/test/Util/Runner/arch.json` and 15 siblings |
| `emitOpError` without `signalPassFailure()` — the FR-T5 root cause | `mlir/lib/Util/Dependency.cpp:2063-2066` |
| the stock `aircc` pipeline we never modify | `tools/aircc/aircc.cpp:905-995`; ping-pong at `:966-980` |
| `output_format: "none"` in an upstream example | `programming_examples/matrix_multiplication/i8/run.py:573` |

### 8.3 Python dependencies

None beyond NFR-2's list. `filelock` is already a transitive dependency of `mlir_air` (it is
installed in the probe venv and `python/air/api/_compile.py:319` imports it), so §3.5's device
lock adds nothing. `subprocess`, `shutil`, `sysconfig`, `importlib.metadata`, `re`, `glob`,
`tempfile`, `hashlib` are stdlib.

---

## 9. Implementation order, effort, definition of done

Effort figures are **estimates**, consistent with `05-work-breakdown.md` §7's 1.0 pd for M6.

| Step | Day | Est. | Output | Done when |
|---|---|---|---|---|
| 1. `ToolRun`, `invoke`, `verdict`, `classify`, the three recorded stderr samples | D2 am | 0.2 pd | `m6_tools.py` core | `test_stderr_parser` green, including the exit-0 case |
| 2. `tool`, `tool_env`, `check_pin`, `has_device` | D2 am | 0.1 pd | — | `test_T6_versions`, `test_T1_targets` green |
| 3. `artifact()` for `none` | D2 pm | 0.2 pd | the G2 gate's other half | `aircc --device npu1 --output-format=none` on B's W1 text exits 0 with no `error:` line |
| 4. `ir_facts()` + `PIPELINES`/`EXTRACTORS` | D3 | 0.2 pd | `w1.npu1.ir_facts.json` | `test_I_pingpong_labels` reports `unroll = 2` on W1 |
| 5. `diff()` and the `DiffReport` contract | D3 | 0.05 pd | — | `test_diff_denominator` green |
| 6. `artifact()` for `pdi`/`xclbin` + the FR-T3 message | D5 am | 0.1 pd | — | `test_T3_xclbin_message` green on a machine without XRT |
| 7. `run()` — the device harness | D5 (the device window) | 0.1 pd | — | either a device result, or `TOOL-NO-DEVICE` and a recorded skip reason |
| 8. `trace()` + the launch guard + the disclaimer | D6 | 0.05 pd | — | `test_trace_requires_launch`, `test_trace_disclaimer` green |

**Definition of done for M6:**

1. Every FR-T1…T6 test in §7 is green or skipped **with a printed reason** (04-test-plan §8
   item 9).
2. `test_stderr_parser` covers all three verdict branches, including exit 0 with a diagnostic.
3. All four variants × both targets pass `--output-format=none` with no `error:` line
   (04-test-plan §8 item 4), and one variant produces a `.pdi`.
4. `ir_facts` reports `pingpong_unroll == 2` for W1 and a frozen `broadcast_pattern_count` for
   all four variants (item 6).
5. `git status --porcelain` is empty after a full `slow` run (I-5).
6. `vendor/wheels/SHA256SUMS` is committed, and a second machine installs from
   `--no-index --find-links vendor/wheels` and passes `air-opt --version` and `aircc --help`
   (R-13).
7. No `subprocess` import outside `m6_tools.py` (I-1); no `target="auto"` in `tests/` (I-6).

---

## 10. Open questions owned, and their resolutions

| # | Question | Verdict |
|---|---|---|
| **Q-4** | absolute tolerance for the `bf16` W1 device diff | **RESOLVED — §3.6.** Integer-valued fixtures make W1/W1-flip/W3 exact (`tol = 0.0`) *independently of summation order*, which is what the cascade flip needs. W2 is `1e-5` absolute with a stated bound. **No `bf16` device fixture exists** — `bf16` appears only in FR-L9's checker-level `L1-CAPACITY` test — so the question's literal subject is void, and no unmeasured tolerance goes on a slide |
| **Q-5** | full `aircc` prefix or the short one for `test_E3_pingpong_fires` | **RESOLVED — §3.7, measured [M].** `builtin.module(air-dependency,air-label-scf-for-to-ping-pong{device=npu1})` reports `unroll = 2` and two `hoist_alloc = true`, identical to VF §E.5's 16-pass prefix, on VF's own probe module. The full prefix is retained in `PIPELINES` as a named fallback and the pipeline string is stored in every `ir_facts.json` |
| **(new) Q-C1** | does the demo show an `air-runner` Chrome trace? | **RESOLVED — no, §3.8.** Measured: no arch model ships in the wheel; the only one in the project is a `testdevice` with a 32 KB L1; `air-runner` prices `linalg` bodies and FR-E2 makes ours scalar loops; and it segfaults on a launch-less module. `m6.trace` ships and is tested; the pitch gets one honest sentence instead |
| **(new) Q-C2** | does the device harness go through `air.api`'s `CompiledKernel` or through `XRTBackend.load`? | **RESOLVED — `XRTBackend.load`, §3.5.** `CompiledKernel` zeroes every written tensor, which would silently feed W2 an all-zero initial field. Cited at `python/air/api/_compile.py:318-347` and `:98-103` |
| **(new) Q-C3** | `vendor/wheels/` is 2.3 GB — is it committed? | **RESOLVED — no, §3.10.** `SHA256SUMS` is committed; the wheels are distributed out of band and verified with `sha256sum -c`. **One-line edit flagged for the architect** to `07-environment.md` §2 |
| **(open) Q-C4** | is `--placed-ir-verifiers=error` (the default, `aircc.cpp:264`) ever the thing that fails one of our four variants at D2–D6? | **Owner C, due D2.** We pass the flag's default and never override it. If a variant fails there, the fallback is **not** to lower the verifier to `warn` — it is to fix the plan, because `air-verify-hierarchy-locality{strict=true}` is one of only two real checks upstream runs (VF §S8) and silencing it would undo G2. Recorded so the temptation is named in advance |
