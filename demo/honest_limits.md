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
- **The herd shapes the checker refuses, and why.** Two of them, and neither is
  in an upstream document — both were traced to a line of mlir-air or mlir-aie
  and turned into a counted rule that fires before codegen
  (`design/PROGRESS-B.md` B-P36, B-P37). A herd with **more rows than columns**
  whose fill multicasts along the column axis is mis-lowered upstream: the
  affine guard `air-specialize-dma-broadcast` builds bounds the *row*
  coordinate by the **column** count (`AIRMiscPasses.cpp:265-271`, the argument
  is `herd.getNumCols()`), so on npu2's own 2×4 herd two rows fall into the
  wrong arm and `air-to-aie` reports *"'air.channel.put' op failed to get S2MM
  tile for L3 allocation"*. A **five-flow fill into one column** exhausts a
  switchbox arbiter's four master selects, because the multicast's
  `{DMA, North}` port set only partially overlaps each point-to-point flow's
  and none of them may share one. Both counts were measured on the routed IR,
  including the shapes that sit exactly on the cap and compile.
- **So npu2's whole 8-PE 2-D herd is not reachable as `grid(2, 4)`.** That is
  an upstream defect, not a device limit, and it is stated rather than papered
  over: the checker refuses it with the arithmetic, and a herd at least as wide
  as it is tall compiles — measured `(3, 2)` and `(4, 2)` at exit 0 with the
  physical cap lifted. Adopting that wider cap is an open item, not a claim.
- **Out of scope**: multi-kernel fusion, autotuning, GPUs.
