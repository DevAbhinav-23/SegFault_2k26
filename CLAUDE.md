# SegFault_2k26 — project instructions

## What this repo is
Entry for **SegFault 2026**, the hackathon of the Innovations In Compiler Technology (IICT)
workshop (segfault.compilertech.org). Final online evaluation listed **Sept 19–20 2026**;
grand finale **Oct 2–3 2026 at IISc Bengaluru**. Judging = **5-minute pitch + 2–3 min Q&A**;
**no published rubric**. Our entry sits in the open theme *"Domain specific compilers and
languages"*: a **Spatial DSL for NPUs lowering to MLIR-AIR**.

Parent research repo: **amd-npus** (`/home/adi/Projects/Honours/amd-npus`). This repo is
**independent** of it — context here was copied, not linked. Do not write to amd-npus.

## Scope rules (binding)
- **NPUs only. GPUs are never in scope** — do not introduce GPU content, backends, or comparisons.
- **DSL + lowering only.** Performance-optimisation passes are out of scope.
- **3–4 dense kernels**: GEMM, Jacobi/stencil, Smith-Waterman DP, optional FFT.
- **"2–3 backends via AIR"** honestly means AIE generations **NPU1 (Phoenix/AIE2)** and
  **NPU2 (Strix/AIE2P)** on one `air-to-aie` lowering path; possibly **Versal** (unverified).
  **Qualcomm Hexagon** and **Tenstorrent** are separate backend builds, **not AIR targets**.
- **Tenstorrent (Wormhole, functional simulator ttsim) is B's second backend via a second emitter from the backend-neutral MappingPlan — not an AIR target, no performance claims; spec in design/08-tt-backend.md. Qualcomm Hexagon stays out (single DSP core, disjoint stack).

## Where the state lives
- `hackathon/HANDOFF.md` — **read first**.
- `hackathon/01-paradigm-comparison.md` **rev r3** — the paradigm decision: a **pragma surface
  over a declared-intent SPMD per-PE core**, emitted through mlir-air's `air.api`. Rival option:
  ship the **legality checker alone**. §9 lists experiments **E1–E5** to run before day 1.
- `hackathon/REDTEAM-round{1,2}.md`, `hackathon/RESPONSE-round{1,2}.md` — adversarial review trail.

## Verified MLIR-AIR facts (as of 2026-09-12)
- Compute model requires an **acyclic channel graph** and **put/get balance** — stated as a
  compile-time condition, but **no enforcing pass was found in source**.
- `air_ChannelOp` has **no depth attribute**; double-buffering comes from ping-pong passes.
- `air.api` provides `launch`/`segment`/`herd` context managers, `channel(size=, broadcast_shape=)`,
  put/get, and `build(target="npu1"/"npu2")`.
- `air-broadcast-detection` derives multicast.
- **air-runner is a performance simulator only — not a correctness oracle.**

## Nearest neighbours (name these in any pitch)
amd/Triton-XDNA (Triton SPMD → MLIR-AIR, AIE2/AIE2P) · Dato (Cornell, typed streams → MLIR-AIE) ·
AIEHalide (the group's own accepted PACT 2026 paper; Halide → MLIR-AIE with ignorable directives) ·
IRON/ObjectFIFO · ARIES · hexagon-mlir (Qualcomm, Triton→Hexagon) · TileLoom / tt-lang /
triton-tenstorrent (Triton→Tenstorrent) · triton-ascend.

## Working rules
- **Research integrity:** no fabricated citations; verify or mark `[UNVERIFIED]`; every count names
  its denominator; estimates are labelled as estimates.
- **Workflow:** Fable acts as **architect** (briefs, verification, adjudication); **Opus subagents**
  do drafting, searching, and coding. **Red-team anything that proposes what to pursue.**
- **Style:** terse prose; minimal code; never skimp on validation, error handling, or security.
