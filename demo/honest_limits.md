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
- **Nearest neighbours, named**: `amd/Triton-XDNA` (SPMD → MLIR-AIR for
  AIE2/AIE2P), **Dato** (typed streams → MLIR-AIE, already rejects deadlock
  and inconsistent put/get), **AIEHalide** (PACT 2026, ignorable directives
  and derived halos, targets MLIR-AIE), **IRON/ObjectFIFO**, **ARIES**.
- **Out of scope**: multi-kernel fusion, autotuning, GPUs.
