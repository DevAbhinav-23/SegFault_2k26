# Honest limits — read aloud, not skipped (05-wbs §5 beat 6, M8 §5.2)

- **One lowering path, two device flags.** `air-to-aie` → MLIR-AIE, with `npu1`
  (Phoenix, AIE2) and `npu2` (Strix, AIE2P). Versal values are accepted by the
  *pass* but not by `air.api`'s `resolve_target`
  (`python/air/api/_trace.py:164-183`), so we do not claim them.
- **Compute is scalar.** Per-PE bodies are native `air.sequential` loops with
  element load/store. An `air.extern` vectorised kernel would be faster — and
  would *disable* the ping-pong passes on any buffer it touches first, because
  an opaque callee is not a definite write
  (`AIRDependencyScheduleOpt.cpp:1620-1630`). Future work, with that cost stated.
- **No performance numbers.** Optimisation passes are out of scope. We measured
  nothing and claim nothing.
- **`air-runner` is a timing model, not a correctness oracle.** mlir-air ships
  one (`docs/AIRRunner.md:3`); it prices `linalg` bodies and ours are scalar
  loops, and no NPU resource model ships in the wheel — the only one in the
  project describes a `testdevice` with a 32 KB L1. So we do not show a trace,
  rather than showing one that means less than it looks like it does.
- **Semantics off-device are established structurally, not by executing the
  emitted IR.** `air.api` has no interpreter, and the CPU backend JITs the
  *lowered* module, which tests the compiler's output rather than the user's
  source. Three structural checks stand in: access-region reconstruction,
  compute-node replay, and write-domain coverage. **Only the device run closes
  that loop.** (No device run yet: `/dev/accel*` absent, so the end-to-end
  claim stops at `aircc --output-format=none`.)
- **W4 FFT is out of scope**, and the reason is not a schedule limit: the
  stage-parameterised partner map `p ↦ p ⊕ 2^s` is affine in neither `p` nor
  `s`, so this surface cannot state it without an escape hatch.
- **The checker's idea is not ours; its implementation is.** AMD specified it
  in their own repository — `docs/AIRCorrectnessChecker.md`, properties P1
  (channel balance), P2 (deadlock freedom), P3 (resource constraints), P4
  (token-constraint consistency) — and did not build it.
- **The Tenstorrent beat is a functional simulator, not silicon, and not a
  timing claim.** No Tenstorrent hardware was touched at any point; ttsim's
  "bit-exact vs silicon" is *their* claim, not our measurement, and only the
  value of `TT_METAL_SIMULATOR` separates the two paths. One generation,
  Wormhole B0. The compute is scalar C++ on **one data-movement RISC-V** — the
  Tensix matrix and vector engines, tile layouts, double buffering and
  `f16`/`bf16` are all untouched — and ttsim is **not** an oracle for ordering
  hazards (measured: it does not flag a semaphore increment moved before the
  write barrier). What it does prove is narrow and real: the same unchanged
  `MappingPlan` drives two unrelated device models to the same numbers on four
  workloads, each with a negative control. **Tenstorrent is not an AIR
  target**, and nothing here says it is (`design/08-tt-backend.md` §8).
- **Nearest neighbours, named**: `amd/Triton-XDNA` (SPMD → MLIR-AIR for
  AIE2/AIE2P), **Dato** (typed streams → MLIR-AIE, already rejects deadlock
  and inconsistent put/get), **AIEHalide** (PACT 2026, ignorable directives
  and derived halos, targets MLIR-AIE), **IRON/ObjectFIFO**, **ARIES**; on the
  Tenstorrent side **TileLoom**, `tenstorrent/tt-lang` and
  `kernelize-ai/triton-tenstorrent`, and `qualcomm/hexagon-mlir` for the
  "why not Hexagon?" question (one DSP core, no PE grid).
- **Out of scope**: multi-kernel fusion, autotuning, GPUs.
