# spatial-dsl

A design exploration: **a pragma-based (OpenMP-style) programming model that treats
spatial accelerators as first-class targets** — clean algorithm, spatial schedule as
ignorable annotations, lowering to a space-time-map IR. Targeting AMD AIE (Ryzen-AI
NPU / Versal) first.

Motivated by dissatisfaction with the ARIES (FPGA '25) tile programming model — see
`../docs/notes.html` (ARIES tab) for the paper notes that sparked this.

## Contents
- [`01-design-overview.md`](01-design-overview.md) — initial design thoughts: motivation, strawman pragma set,
  first-class concepts, prior art, open problems, next steps.
- [`02-spacetime-ir-and-reduction-legality.md`](02-spacetime-ir-and-reduction-legality.md)
  — the worked-out IR core: space-time `(σ,π)` map, stationarity-as-derived, the
  reduction-legality theorem (temporal/spatial/hybrid), AIE realization, and an
  end-to-end GEMM lowering.
- [`03-user-stories-beyond-matmul.md`](03-user-stories-beyond-matmul.md) — stencil,
  conv2d/3d, CNNs worked through the pragma model; pins the pragma surface (Python:
  plain-function algorithm + schedule builder). Shows the IR core is not matmul-shaped.
- [`04-spatial-triton-dsl.md`](04-spatial-triton-dsl.md) — an **alternative surface**: a
  hardware-specific DSL (CUDA/Triton-style) with **PE-tile** as the unit, first-class
  inter-PE streams + stationarity + **multicast** (the reuse trichotomy), and an
  end-to-end GEMM. Lowers to the same `(σ,π)` engine as the pragma surface.
- [`05-composition-and-fusion.md`](05-composition-and-fusion.md) — multi-op / multi-layer
  programs in Spatial-Triton: GEMM→ReLU epilogue fusion, GEMM→GEMM in three regimes
  (inline-fused / temporal / spatial-pipeline). Composition = the reuse trichotomy applied
  to intermediates; graph-level primitives (`@sp.graph`, `g.stay`, `g.pipeline`).
- [`REFERENCES.md`](REFERENCES.md) — priority-ordered reading list for the above.

Status: **active design**, 2026-06-21.
