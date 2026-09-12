# Recorded tool stderr — the `test_stderr_parser` corpus

Three raw stderr captures, recorded live on the pinned wheel so that `test_stderr_parser`
(`design/03-lld-M6-toolchain.md` §7, `design/04-test-plan.md` §2 "M6 toolchain") exercises all
three branches of `m6.verdict` with **no subprocess**, and therefore runs in the default suite.

Recorded **2026-09-13** (local, IST; `date -u` read `2026-09-12`) on machine 1 of 3, against
`mlir_air 0.0.1.2026091204+ff95a9b` / `mlir_aie 1.4.3.dev55+g10767b5` /
`llvm_aie 22.0.0.2026091201+386ca5c6` (`m6.PIN`). Each file holds the **raw stderr only**;
stdout and the exit code are recorded here.

Stub written by B at P0c to unblock; **C owns these files** (`design/03-lld-M7-tests.md` §3.8).

| File | Tool | argv | exit | stderr |
|---|---|---|---|---|
| `clean.txt` | `air-opt` | `air-opt <w1_base.mlir> -o /dev/null` | **0** | empty (0 bytes) |
| `exit0_with_error.txt` | `air-opt` | `air-opt tests/fixtures/modules/unpaired_channel.mlir -pass-pipeline='builtin.module(air-dependency,air-dependency-canonicalize)' -o /dev/null` | **0** | 1 `error:` line + 1 `note:` line, 576 bytes |
| `exit1_with_error.txt` | `aircc` | `aircc --device npu1 --output-format=none --tmpdir air_project malformed.mlir` | **1** | 1 `error:` line + 1 plain line, 109 bytes |

Reproducing each, from the repository root with `.venv` active:

1. **`clean.txt`** — the valid module is the W1-shaped probe, regenerated rather than committed
   (`vendor/probes/` is git-ignored):
   ```
   .venv/bin/python vendor/probes/q/q7_a.py > $SCRATCH/w1_base.mlir
   air-opt $SCRATCH/w1_base.mlir -o /dev/null
   ```
   Exit 0, stderr empty. The file is therefore 0 bytes **by measurement**, not by omission.

2. **`exit0_with_error.txt`** — the FR-T5 trap, and the whole reason `m6.verdict` reads stderr
   before the exit code. The module is `tests/fixtures/modules/unpaired_channel.mlir`; the
   command is the `PIPELINES["pairs"]` pipeline, verbatim from the table above. `air-opt` prints
   `error: 'air.channel.put' op found channel op not in pairs` **and exits 0**
   (`mlir/lib/Util/Dependency.cpp:2063-2066` calls `emitOpError` without `signalPassFailure()`).

3. **`exit1_with_error.txt`** — the other branch: a non-zero exit whose message must still be the
   diagnostic, not "exit 1". The input is the same fixture with one deliberate corruption,
   reproducible in one line:
   ```
   sed 's/%alloc\[\] \[\] \[\]/%undefined[] [] []/' \
       tests/fixtures/modules/unpaired_channel.mlir > $SCRATCH/malformed.mlir
   cd $SCRATCH && aircc --device npu1 --output-format=none --tmpdir air_project malformed.mlir
   ```
   Run from `$SCRATCH` with a *relative* input name so the recorded text carries no absolute
   path. `aircc` was on `PATH` via `source scripts/airenv.sh`.

**Finding, recorded at P0c and flagged for the architect.** `aircc` spells its diagnostics
`loc("<file>":<line>:<col>): error: …` — the MLIR `loc(...)` form — while `air-opt` spells them
`<file>:<line>:<col>: error: …`. The `ERROR_LINE` pattern of `03-lld-M6-toolchain.md` §3.2 only
accepts the second spelling, so as written it matches **no `aircc` diagnostic at all**.
`spatial/m6_tools.py` therefore accepts both spellings in its `loc` group, keeps the group names
and the rest of the pattern unchanged, and says so at the definition. Measured evidence: the two
`aircc` runs above, plus a third — `aircc --device npu1 --output-format=none` on the *unmodified*
`unpaired_channel.mlir` exits **0** and prints `loc("-":9:7): error: 'air.channel.put' op found
channel op not in pairs`, i.e. `aircc` reproduces the FR-T5 trap as well.
