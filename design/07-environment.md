# Environment — pinned toolchain, off-device workflow

*Phase 1, 2026-09-12. Owner: Person C. Everything here is copied from
[`../hackathon/VERIFIED-AIR-FACTS.md`](../hackathon/VERIFIED-AIR-FACTS.md) §G and its
"Install path" section, which was **run** on a Linux laptop with no NPU, no XRT and no Vitis
(5 min 43 s, ~2.3 GB). Nothing in this file is inferred.*

---

## 1. The pin

| Component | Version | Source |
|---|---|---|
| `mlir-air` | `0.0.1.2026091204+ff95a9b` | built from commit `ff95a9b35b692b4cfbdf1d52bf69e6ffe01facde`, which is the commit every fact in `VERIFIED-AIR-FACTS.md` cites |
| `mlir-aie` | `1.4.3.dev55+g10767b5` | matches `utils/clone-mlir-aie.sh:17`'s pin `10767b50e5beec9b2ec97ce92298e3507a33490c` |
| `llvm-aie` (Peano) | `22.0.0.2026091201+386ca5c6` | from the nightly index |
| `numpy` | `2.5.3` | whatever the wheel resolves; pin it in the lockfile |
| Python | upstream supports 3.11 – 3.14; **this project pins `3.12`** on all three laptops and in CI | `docs/buildingRyzenLin.md:9-11`. The wheels are Python- **and** platform-tagged (`cp3XX-…-manylinux_…`), so a machine on a different interpreter resolves a different file — or none. The measured install used 3.14.7; the pin is a decision, not a measurement, and Person C re-downloads the cache on the pinned interpreter at D0 (risk R-13, plan risk P-6) |

**Every golden file in the test suite is valid only against this pin** (`06-interfaces.md` §8
rule 2). `test_T6_versions` runs first and skips the golden tests with a clear reason if the
installed version differs.

---

## 2. Install

```bash
python3 -m venv airenv
source airenv/bin/activate          # fish: source airenv/bin/activate.fish
pip install --upgrade pip

pip install 'mlir_air[aie]' \
  -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels \
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4 \
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

Nothing is on PyPI proper; resolution goes entirely through those three GitHub release pages,
so the `-f` flags are **mandatory** (`docs/buildingRyzenLin.md:28-32`).

**Pin the tag, and cache the wheels.**
`.github/workflows/pruneAIRReleaseAssets.yml:95-97` prunes release assets, so
`latest-air-wheels` is not a reproducible reference. Two actions, both at D0 (risk R-13):

1. Substitute a `v*.*.*` release tag for `latest-air-wheels` once the team knows which tag
   carries `0.0.1.2026091204+ff95a9b`.
2. **Download the four wheels to `vendor/wheels/` and record their sha256 in this file.** CI and
   every re-install then use `pip install --no-index --find-links vendor/wheels 'mlir_air[aie]'`.
   A pruned asset mid-week then costs nothing.
3. `vendor/wheels/` is **git-ignored** — it is ~2.3 GB. `vendor/wheels/SHA256SUMS` **is**
   committed, and the wheels themselves are distributed out of band (a shared drive or a USB
   stick handed round at D0). CI restores them from its own cache keyed on that file.
4. The wheels are Python- and platform-tagged, so the cache is valid only for the pinned
   interpreter of §1 and for `manylinux` x86-64. A second platform needs a second cache; the
   three laptops and the CI runner are all on the pinned pair.

```
# to be filled in at D0 by Person C:
# vendor/wheels/mlir_air-0.0.1.2026091204+ff95a9b-*.whl   sha256: ...
# vendor/wheels/mlir_aie-1.4.3.dev55+g10767b5-*.whl        sha256: ...
# vendor/wheels/llvm_aie-22.0.0.2026091201+386ca5c6-*.whl  sha256: ...
# vendor/wheels/numpy-2.5.3-*.whl                          sha256: ...
```

### Verify — with no environment variables set at all

```bash
air-opt --version        # → LLVM version 24.0.0
aircc --help             # → exit 0
python -c "import air.ir, air.passmanager; from air.dialects import air; print('ok')"
python -c "from air import api as air; print('api ok')"
```

All four work with **no** environment variables and **no** XRT
(`python/air/tools.py:47-49` resolves the bundled `mlir_air/bin/<tool>` before falling back to
`PATH`). The env-var block in `docs/buildingRyzenLin.md:38-50` is **not** needed for these.

### The core-only install is not enough

`pip install mlir_air` without `[aie]` takes 1 min 45 s / 972 MB and gives `air-opt`,
`air-runner` and `import air` — but `aircc` cannot compile anything, because it needs `aiecc`
from `mlir_aie`. Install the `[aie]` extra.

### Do not build from source

Three tiers exist (`docs/buildingRyzenLin.md:3`, `:88`, `:290-341`). We use tier 1 only. Source
pins, recorded for reference and not to be used: LLVM is the ROCm fork at
`56bcc1871734e6c375a254dec0ec74eb18d04a2e` (`utils/clone-llvm.sh:21-23`); `cmake>=4.4.2,<5.0`,
`ninja!=1.13.0` (`utils/requirements_dev.txt:2-6`). Vitis/`xchesscc` is needed **only** for the
legacy VCK5000 path (`docs/buildingVCK5000.md:36`).

---

## 3. What works without XRT and without a device

| Capability | Works? | Evidence |
|---|---|---|
| `import air`, `from air import api as air` | **yes** | VF §G.6 — `air` is an implicit namespace package on a `.pth`; XRT is imported lazily inside `xrt.py:717-719`; no `ctypes.CDLL` on the import path |
| `air-opt` with any pass pipeline | **yes** | VF §G.4 — a plain `MlirOptMain` driver linking no XRT; `ldd` shows only libc/libm/libz/libstdc++ |
| `launch.build(target="npu1"\|"npu2")` → MLIR module + `str()` | **yes** | VF §D.9 — `build()` contains no pass pipeline at all: it traces, replays into a `func.func`, and calls `module.operation.verify()` |
| `aircc --output-format=none` | **yes** | VF §G.5, measured: exit 0, all 22 aiecc stages through Peano, core ELF produced |
| `aircc --output-format=pdi` | **yes** | VF §G.5, measured: exit 0, `aie.pdi` produced |
| `aircc --output-format=xclbin` | **no** | VF §G.5, measured: exit 1 at step 40/41 — `aiecc: ShellCommand: tool 'xclbinutil' not found in search paths or PATH`. `xclbinutil` comes from **XRT**; it is not a device requirement, it is a packaging tool |
| `output_format="elf"` on an `npu1` target | **no** | `python/air/backend/xrt.py:410-414` rejects it explicitly and suggests `pdi` |
| running a kernel | **no** | needs XRT + `/dev/accel*` |
| `air-runner` | **yes, but it has no model to run** | it is a **timing** simulator, not a functional oracle (VF §S7, `docs/AIRRunner.md:3`) — **and no resource model ships in the wheel**: `find <purelib> -name 'arch*.json'` returns nothing (measured). The only models in the project are under `mlir/test/Util/Runner/`, and the canonical `arch.json` there is a `"devicename": "testdevice"` with `L1 = 32768`, which is neither `npu1` nor `npu2`. Writing one is out of scope; `Runner.cpp:543-551` also dereferences a `dyn_cast_if_present<air::LaunchOp>` with no null check, so a module without a launch crashes it |

**Consequence for the plan**: everything in `04-test-plan.md` except level **D** runs on a stock
CPU machine. The only capability the team lacks without a Ryzen AI machine is the `xclbin`
container step and actually executing a kernel.

---

## 4. Driving `aircc` end to end

`aircc` shells out to `aiecc`, so `aiecc` must be on `PATH` and Peano must be locatable:

```bash
set -l SP (python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")   # fish
# bash:  SP=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")

export PATH=$SP/mlir_air/bin:$SP/mlir_aie/bin:$PATH
export PEANO_INSTALL_DIR=$SP/llvm-aie

aircc --device npu1 --output-format=none design.mlir     # or --output-format=pdi
```

`--output-format=xclbin` is the **default**, so the flag is not optional for us off-device.
`programming_examples/matrix_multiplication/i8/run.py:573` does the same thing from Python:
`"output_format": "none",  # Skip xclbin generation (no xrt dependencies)`.

**Read stderr, not the exit code.** `air-opt` prints
`error: 'air.channel.put' op found channel op not in pairs` and **exits 0**
(`mlir/lib/Util/Dependency.cpp:2063-2066` calls `emitOpError` without `signalPassFailure()`;
VF §B.6, §S11). M6's wrapper treats any line matching `error:` as a failure regardless of exit
status (FR-T5), and `test_stderr_parser` pins that behaviour.

### Useful `air-opt` invocations

```bash
# parse + verify only
air-opt design.mlir -o /dev/null

# the ping-pong labelling prefix — the MEASURED two-pass form (03-lld-M6-toolchain.md §3.7)
air-opt design.mlir \
  -pass-pipeline='builtin.module(air-dependency,air-label-scf-for-to-ping-pong{device=npu1})' \
  | grep -n 'unroll\|hoist_alloc'

# the 16-pass VF §E.5 prefix, kept as the recorded FALLBACK if the two-pass form ever
# stops labelling.  Identical assertions; slower; use only when the short form fails.
air-opt design.mlir -pass-pipeline='builtin.module(air-dependency,\
air-annotate-packet-ids{assign=true},air-hoist-dma-in-accum-pattern,air-dma-to-channel,\
canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,\
air-isolate-async-dma-loop-nests{scope=launch},canonicalize,cse,air-fuse-channels,\
canonicalize,cse,func.func(air-fuse-alloc-dealloc),\
func.func(air-shrink-memref-sizes-by-access),\
air-label-scf-for-to-ping-pong{device=npu1})' | grep -n 'unroll\|hoist_alloc'

# broadcast detection, to count what fires unasked (risk R-04)
air-opt design.mlir -pass-pipeline='builtin.module(air-dependency,air-broadcast-detection)' \
  | grep -c broadcast_pattern

# the emitted locks, to confirm no producer lock starts at 0 (VF §C.7)
air-opt design.mlir -pass-pipeline='builtin.module(air-to-aie{device=npu1})' \
  | grep -o 'init = [0-9]*' | sort | uniq -c
```

The full `aircc` pipeline is built in C++ at `tools/aircc/aircc.cpp:905-995` and is quoted
verbatim in VF §D.9. We never modify it.

**The ping-pong opt-out flag is `--omit-ping-pong-transform[=<''|L1|L2|all>]`**, re-verified
against `aircc --help` on 2026-09-12. `VERIFIED-AIR-FACTS.md` §E.1 originally spelled it without
the hyphens and without the `-transform` suffix; that line is corrected and the wrong spelling is
accepted by no build. **We never pass the flag** — it is recorded only so nobody copies the wrong
one out of a quoted source snippet.

### `air-runner`

```bash
air-runner design.mlir -f <top-level-function-name> -m <arch-model>.json -o trace.json
```

or from Python:

```python
import air.compiler.util
runner = air.compiler.util.Runner(arch)      # arch = a JSON resource model
trace = runner.run(air_module, "<module name>")
```

(`docs/AIRRunner.md`, Usage.) It emits a Chrome trace. **It is a performance model. It is not a
correctness oracle and must never be presented as one** (VF §S7) — it models channel stalls,
so "an unbalanced put/get pair still stalls instead of" failing
(`mlir/lib/Util/Runner/RunnerNode.cpp:1447`).

---

## 5. Targets

`air.api`'s `resolve_target` accepts exactly `None`, `"auto"`, `"npu1"`, `"npu2"` and raises
`ValueError` for anything else (`_trace.py:164-183`). `"auto"` shells out to `xrt-smi` and falls
back to `NO_DEVICE_TARGET = "npu2"` when there is no device (`_trace.py:96`, `:182`).

**Pass an explicit target in tests** so no subprocess is spawned.

`aircc`'s own `--device` default is `xcvc1902` (`aircc.cpp:164-166`), and `air-to-aie`'s pass
option defaults to the same (`Conversion/Passes.td:194-196`). A Versal build would therefore
have to bypass `LaunchContext.compile` entirely and drive `aircc --device xcvc1902` on a module
built by hand or through `air.dialects`. It is a declared non-goal
(`01-requirements.md` §2).

Physical array caps, which govern the herd shape (`_trace.py:88-91`):

```
PHYSICAL_HERD = {"npu1": {1: (4,), 2: (1, 4)},
                 "npu2": {1: (8,), 2: (2, 4)}}
```

A larger logical grid is strip-mined; `shape=` must divide it exactly, and the default is the
largest divisor of each extent not exceeding the cap.

Memory budgets, also from `air.api` itself: `L1_BYTES = 65536` per core (`_trace.py:100`),
`L2_BYTES_PER_MEMTILE = 512 * 1024` × `DEVICE_COLUMNS = {"npu1": 4, "npu2": 8}`
(`_trace.py:117-118`).

---

## 6. Repository layout (created at D0)

```
spatial/          the package: model.py, m1_frontend.py, m2_schedule.py, m3_legality.py,
                  m4_mapping.py, m4_selfcheck.py, m5_emit.py, m6_tools.py
kernels/          w1_gemm.py, w2_jacobi.py, w3_sw.py  — the three sources and their schedules
                  rejections.py                      — the illegal schedules the demo shows
tests/            conftest.py, helpers/, unit/, negative/, golden/, integration/, fixtures/
demo/             the demo script and its recorded transcript (03-lld-M8-kernels-demo.md §5)
vendor/wheels/    the four cached wheels (R-13) — GIT-IGNORED; SHA256SUMS is committed
design/           this document set
```

`spatial/` imports `air` **only** from `m5_emit.py` and `m6_tools.py` (FR-S20). A test asserts
that (`test_no_air_import_until_build`), so the surface, the checker and the mapper all remain
usable on a machine with no toolchain at all.
