# A pragma-based programming model for spatial accelerators — initial thoughts

*Started 2026-06-18. Early brainstorming; nothing here is settled.*

## 0. Where this came from

While reading **ARIES (FPGA '25)** we grew dissatisfied with its programming model.
The critique, in one breath:

- ARIES sells itself as Halide-like (decoupled algorithm/schedule), but its
  `@task_tile` body is **half-schedule already** — it contains a tile decomposition
  and explicit `A_L1`/`B_L1`/`C_L1` staging. Naming L1 buffers is a memory-hierarchy
  commitment that, in real Halide, would be a *schedule* directive, never the algorithm.
- It is therefore in an **uncanny valley**: not the honest architecture-aware imperative
  model of CUDA, and not the clean algorithm/schedule split of Halide.
- The tile is **shaped to fit the compiler** ("one tile → one core"), not the user's
  mental model — the frontend is reverse-engineered from the IR.
- Dataflow is **emergent, not named**: output-stationary arises as a *side effect* of
  which index you omit from the `.to()` placement expression (`NPU[i%4, 2+j%2]` uses
  `i,j` but not `k`). You cannot read a placement line and know the dataflow.
- Explicit placement hard-codes **physical core coordinates**, so it is
  topology-coupled and non-portable (breaks moving 4×5 → 8×4).
- The whole evaluation (GEMM, TTMc, MTTKRP, ResNet, MLP) is **dense affine
  reduction nests** — the polyhedral class. "General applications" is overclaim;
  the cited stencil work (SPARTA) runs on mlir-aie, not ARIES.

Taxonomy that frames the goal:

| Model | Algorithm | Schedule | Verdict |
|---|---|---|---|
| CUDA | architecture-aware (honest) | n/a (manual) | pain, but honest |
| Halide | architecture-**agnostic**, pure math | architecture-dependent directives | clean separation |
| ARIES | tile-shaped, half-schedule | decoupled-ish primitives | muddy / oversold |
| **(goal)** | **pure math, runs sequentially** | **spatial pragmas, ignorable** | **want this** |

There is a contemporary proof that clean separation is possible on this exact hardware:
**AIEHalide** (PACT '26) — *"Compiling Halide to AMD NPU Spatial Dataflow with
Algorithm-Schedule Separation."* So ARIES's conflated frontend is a *choice*, not a
necessity.

## 1. Thesis: why pragmas (not a new DSL)

OpenMP's underrated property: **strip every pragma and the program still compiles and
runs — sequentially, correctly.** Annotations are *ignorable*.

For a spatial target that is exactly the property we want, because spatial hardware is
where the nasty bugs live: partial-sum reduction hazards, stream deadlock, routing
capacity overflow. A pragma model gives a **sequential reference oracle for free** — the
un-annotated nest *is* the spec. ARIES/Triton tile-kernels do not run as plain code, so
they cannot offer this. This is the strongest single argument for the pragma approach.

## 2. Strawman: clean algorithm + spatial schedule pragmas

> **Surface locked (2026-06-19): Python**, not C — plain-function algorithm + schedule
> builder (the schedule method calls *are* the pragmas). See
> `03-user-stories-beyond-matmul.md` §0 for the decision and rationale. The C `#pragma`
> sketch below is kept as the original motivation; treat it as pseudocode for the intent,
> with the Python surface as the real target.

```c
// ── Algorithm: pure math, architecture-agnostic, runs sequentially as the reference ──
void matmul(i16 A[M][K], i16 B[K][N], i32 C[M][N]) {
#pragma spatial target(npu) grid(virtual)            // virtual PE grid; compiler folds to physical
#pragma spatial parallel(i, j) reduce(k)             // i,j → space; k declared as a reduction
#pragma spatial stationary(C)                        // dataflow is NAMED, not an emergent side-effect
#pragma spatial place(i -> px, j -> py)              // affine placement onto the virtual grid
#pragma spatial stream(A: forward(px), B: forward(py))  // systolic shift along rows/cols
#pragma spatial reside(A:L2, B:L2, C:L1) double_buffer(A,B)
#pragma spatial pipeline(k) ii(auto)                 // PE-local pipelining; auto ⇒ compiler searches II
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < K; k++)
        C[i][j] += A[i][k] * B[k][j];                // ← the real math, untouched
}
```

Wins over ARIES, concretely:

- Loop body is **pure math** — no `A_L1` staging, no `tile_ranks()`. (Fixes
  "algorithm is half-schedule.")
- Dataflow is **declared** (`stationary(C)`), not inferred from an omitted index.
  (Kills the output-stationary-by-omission footgun.)
- Placement targets a **virtual** grid; compiler folds to 4×5 / 8×4 physical.
  (Kills `NPU[i%4, 2+j%2]` topology-coupling.)
- `reduce(k)` is explicit, so the compiler **owns** the don't-evacuate-C-until-the-
  K-sweep-finishes hazard.

## 3. First-class citizens of a spatial schedule

The real design question: what vocabulary makes "spatial" native, not bolted on.

| Concept | Pragma surface | Controls |
|---|---|---|
| **Space** (allocation π) | `place(idx → coord)` | which PE runs which iteration |
| **Time** (schedule σ) | implied by `stationary` + `pipeline` | logical issue order |
| **Stream / channel** | `stream(buf: pattern)` | typed PE→PE edges: `broadcast`, `forward` (systolic), `cascade` (reduce) |
| **Stationarity** | `stationary(operand)` | which operand stays resident in a PE |
| **Memory residence** | `reside(buf: L1\|L2\|L3)` + `double_buffer` | explicit scratchpad placement |
| **Boundary IO** | `io_budget(...)` | shim/PLIO bandwidth the compiler must *prove* it does not exceed |

### The principled core: a space-time affine map

`place` + `time` together are a **space-time transform** — σ for time, π for space, the
classic systolic-array synthesis foundation. This is the IR these pragmas should lower
to. It buys legality checks for free:

- if the reduction index `k` is in the **time** map → temporal accumulation
  (output-stationary);
- if `k` is in the **space** map → the compiler must synthesize a **cascade / tree
  reduction network**.

The model *knows* which, so the partial-sum hazard becomes a compiler invariant rather
than a user footgun. Strictly cleaner than ARIES.

## 4. Prior art to build on (don't reinvent)

- **Space-time mapping theory** (Quinton, Rau, Lamport): the σ/π affine-map
  foundation. Principled but user-hostile to write directly — pragmas are the friendly
  surface.
- **T2S / SuSy** (Rong et al., Intel): *Halide + a space-time-transform directive* →
  systolic arrays on FPGA. Closest existing "Halide-for-spatial"; validates the thesis.
- **Spatial** (Koeplinger et al., PLDI '18): DSL with `SRAM`/`FIFO`/`Stream`/`Foreach`/
  `Reduce`/`FSM` first-class. Closest "DSL-for-spatial"; source for the escape hatch.
- **Allo / HeteroCL** (in this repo): decoupled algorithm/schedule/datatype in MLIR —
  the substrate to actually build this on.
- **Exo**: user-schedulable language via verified rewrites — good model for proving
  schedule correctness.
- **OpenMP** `target` / `teams` / `distribute` + `map(to:/from:)` clauses: the
  pragma+offload precedent to extend (but it assumes coherent memory, has no
  placement/routing/streams/stationarity).

**The gap nobody has nailed:** a pragma surface with the *ignorable-reference* property,
over a *space-time-map IR*, targeting *AIE specifically* (L1/L2/memtile/shim quirks +
inter-core forwarding). Thesis-sized hole.

## 5. Open problems (the fights we still need to have)

1. **Deadlock & routing legality.** Non-adjacent streams route through switches with
   bounded FIFOs → deadlock risk. The model must let the compiler *prove*
   deadlock-freedom (Kahn process networks: bounded + monotonic channels). ARIES's
   "if it exceeds channel limits, gather in L2" is an ad-hoc patch; a principled model
   surfaces it as a checkable constraint or a hard error. **Open:** do we accept
   "compiler may reject your schedule as un-routable"?
2. **Expressiveness cliff.** Pragma-over-loop-nest inherits the **affine/polyhedral
   domain** — great for dense tensor algebra (the NPU's job), useless for data-dependent
   control flow (attention masking, sparsity, sorting). **Leaning:** scope the clean
   model to dense-affine and *say so* (opposite of ARIES overclaiming), with a
   Spatial-style explicit-dataflow escape hatch (`channels` + `FSM`) for the irregular
   tail. **Open:** two-tier model, or one surface that eats the complexity?
3. **Is stationarity primitive or derived?** It is *implied* by the space-time map
   (stationary operand = the one whose access projection is constant under σ). So is
   `stationary(C)` real or just sugar for a particular σ/π? **Leaning:** sugar, but worth
   keeping — names beat affine maps for humans. **Open:** does naming it invite
   inconsistency with the underlying map?

## 6. Next steps (pick one)

- [ ] Pin down the **pragma set as a concrete spec** for the AIE (every clause, grammar,
      defaults, what each lowers to). *(partially drafted in
      `02-spacetime-ir-and-reduction-legality.md` §7 — the lowering map; still needs full
      grammar + defaults.)*
- [x] Drill into the **space-time-map IR + reduction legality** → worked out in
      [`02-spacetime-ir-and-reduction-legality.md`](02-spacetime-ir-and-reduction-legality.md):
      `(σ,π)` map + legality (L1/L2), stationarity derived from `ker Sπ`, the
      reduction-legality theorem (temporal/spatial/hybrid), AIE realization (cascade /
      AXIS / shared-L1 / memtile), end-to-end GEMM lowering. Reading list in
      [`REFERENCES.md`](REFERENCES.md).
- [~] Argue the **scope boundary** — decide honestly what is in (dense affine) and out
      (data-dependent), and design the escape hatch if any. *(advanced in
      `03-user-stories-beyond-matmul.md`: stencil/conv2d/conv3d/CNN trunk are all in-scope
      with no new theory; pooling = `max` A/C reduction is in-scope; **softmax** is the
      first genuine escape. Escape-hatch mechanism still to design.)*
- [ ] Read **AIEHalide** and **T2S/SuSy** closely and write a comparison: what they got
      right, where the gap for this design actually is.
- [ ] **(new, surfaced by the IR work)** AIE-specific routing & deadlock legality for the
      spatial-reduction network on cascade-vs-AXIS-vs-shared-L1 (bounded-FIFO / Kahn
      proof). The generic theory assumes routability; AIE's finite channels + unidirectional
      cascade do not. *This is the real open contribution.*
