# Spatial-Triton — a hardware-specific DSL with PE-tile as the unit

*2026-06-19. A second **surface** for the `spatial-dsl` engine: instead of clean
algorithm + separable pragmas (doc 02/03), embrace architecture-awareness in the kernel,
CUDA/Triton-style. Lowers to the **same** `(σ,π)` space-time IR
([doc 02](02-spacetime-ir-and-reduction-legality.md)).*

## 1. What this is (and what it is not)

A *domain* DSL abstracts a problem domain. CUDA and Triton are something else: a
**faithful-but-abstracted execution model you program *to*.** Their power is that each
committed to **one mental model of the machine**:

- **CUDA = SIMT**: write one *thread's* scalar program; reason about thread/block/grid +
  memory hierarchy + sync; compiler/hardware own warp scheduling, SM choice, coalescing.
- **Triton = block-level**: raise the unit to a *tile*; write one *block's* program with
  `tl.load`/`tl.store`; compiler owns everything *within* the block.

> **Spatial-Triton = Triton, but blocks (PEs) can talk to each other through first-class
> typed hardware streams, and you name what stays put (stationarity) and what fans out
> (multicast).** That delta — *inter-PE channels + stationarity + multicast* — is the
> entire difference from Triton, and it is exactly what the spatial fabric is *for*.

### Why ARIES is "neither here nor there" (the thing we are fixing)
- **No crisp unit of execution** — tile? core? task? blurry.
- **Dataflow is emergent** (output-stationary falls out of which index you omit from
  `.to()`), so you cannot *think* in it.
- Yet it **leaks physical coordinates** (`NPU[i%4, 2+j%2]`) — lower than CUDA ever forces.

So ARIES is simultaneously too high (dataflow hidden) and too low (placement exposed).
CUDA/Triton are disciplined about the expose/hide line; Spatial-Triton must be too.

## 2. Design decisions

- **Unit of execution = PE-tile** (locked). One PE's worth of work on a tile — matches the
  AIE core + its L1 + its vector unit; lets the compiler own *within*-PE optimization
  (vectorize/VLIW) exactly like Triton.
- **Expose** (architecture-aware): the PE grid, inter-PE **streams + directions**, what is
  **stationary**, what is **multicast**, the **memory levels** (L1/L2/L3).
- **Hide** (compiler owns): logical→physical placement, stream routing through switches,
  **generating + overlapping the forwarding code** (the *systolic tax* — recall the notes'
  "Ideal MCSA": overlapped forwarding recovered ~all lost performance), vectorization,
  double-buffering, lock/sync insertion. **No physical coordinates in the surface.**
- **Surface ⟂ engine.** Spatial-Triton lowers to the same `(σ,π)` IR as the pragma surface
  (doc 02). This is a *second frontend on one engine* (like C and Rust on LLVM), not a
  competing project.
- **Reference oracle preserved.** Every primitive has **sequential fallback semantics**
  (`pe_id` iterates, `stream` is a loop, `load` is a slice, `acc` is a buffer), so the
  kernel **runs in plain Python as the spec** (Triton's interpreter-mode trick).
- **Commit to one dataflow model** (leaning): *tiled-SPMD + typed neighbor streams +
  multicast*. Arbitrary free-topology dataflow is the escape-hatch tier, not the language.

## 3. The reuse trichotomy — where broadcast/multicast becomes first-class

This is the core. For operand `a` with access linear part `M_a`, its **reuse space** is
`L_a = ker M_a` (directions along which `a` is constant). Classify each loop axis
direction `r` by (i) is it **spatial** (mapped by `π`) and (ii) is `a` constant along it
(`r ∈ L_a`):

| `r ∈ L_a`? | `r` spatial (`π(r)≠0`)? | realization | AIE mechanism |
|---|---|---|---|
| yes | **no** (temporal) | **stationary** — stays in PE, reused over time | local L1 / accumulator register |
| yes | **yes** | **multicast** — same value to all PEs along `r` | AXIS broadcast / mem-tile fan-out down a row or column |
| no | yes | **stream / flow** — data moves PE→PE along `r` | shared-L1 neighbor fwd / cascade (if reduction) |

> **Multicast is not bolted on — it is the "spatial + constant-access" cell.** An operand
> is multicast along a row/column exactly when *it does not index that spatial axis*. In
> GEMM (below): `A[i,k]` does not index `j` → `A` multicasts along the **row**; `B[k,j]`
> does not index `i` → `B` multicasts along the **column**.

**Refinement of doc 02 §3.** Doc 02 split reuse under `π` into just *stationary*
(`π(r)=0`) vs *streams* (`π(r)≠0`). That conflated two distinct `π(r)≠0` cases: if `a`
is **constant** along the spatial `r` it is **multicast** (replicated, value unchanged);
only if `a` **varies** is it a true systolic **stream**. The trichotomy above is the
corrected picture.

**The multicast ⇄ systolic-forward lever.** When `r ∈ L_a` and `r` is spatial, delivery
can be either **multicast** (one source fans out to all PEs on the line — high source
fan-out/bandwidth, low latency) **or** **systolic shift** (PE forwards to neighbor —
low fan-out, +latency). This is a *real performance knob*, and it is exactly the MCSA
result in the notes (VCK5000 tab): pure-systolic wavefront = 62% eff, **multicast** MCSA
= 77%, ideal overlapped = 89%. So the surface must let you **default to derived multicast
but override to systolic forward** (and vice-versa).

## 4. Core primitives (surface vocabulary)

| Primitive | Meaning | Sequential fallback |
|---|---|---|
| `@sp.kernel(grid=(PI,PJ))` | the decorated fn is the per-PE-tile program over a logical PE grid | loop over all `(pi,pj)` |
| `sp.pe_id() -> (pi,pj)` | this PE's logical coords | the loop indices |
| `sp.acc(shape, stationary=True)` | accumulator tile resident in this PE's L1 | zeroed buffer |
| `sp.stream(lo,hi,step)` | a streamed axis (reduction/feed) | `range` loop |
| `sp.load(slice, bcast=?, flow=?)` | bring an operand tile to this PE; `bcast`/`flow` optionally pin delivery (default **derived** from access) | array slice |
| `sp.dot(a,b)` / ops | tile compute; compiler vectorizes (SIMD/VLIW) | numpy-style op |
| `sp.store(slice, tile)` | write result out | assignment |
| `sp.flow(slice, dir="W→E")` | force systolic forward instead of multicast | slice |

`bcast="row"` = multicast across `pj` for fixed `pi`; `bcast="col"` = across `pi` for
fixed `pj`.

These are the **kernel-level** (single-op) primitives. The **graph-level** primitives for
composing kernels (`@sp.graph`, `g.stay`, `g.pipeline`) — i.e. fusion across ops/layers —
are in [doc 05](05-composition-and-fusion.md).

## 5. End-to-end GEMM in Spatial-Triton (output-stationary)

```python
import spatial as sp

# Logical PE grid PI×PJ; PE(pi,pj) owns a TM×TN output tile of C.
@sp.kernel(grid=(PI, PJ))
def gemm(A: sp.f16[M, K], B: sp.f16[K, J], C: sp.f32[M, J]):
    pi, pj = sp.pe_id()
    TM, TN, TK = M // PI, J // PJ, 64

    c = sp.acc((TM, TN), stationary=True)            # C-tile lives in this PE (output-stationary)

    for kk in sp.stream(0, K, TK):                   # reduction over K, streamed → temporal accumulate
        a = sp.load(A[pi*TM:(pi+1)*TM, kk:kk+TK])    # indexes pi (row) + k, NOT pj  → multicast along ROW
        b = sp.load(B[kk:kk+TK, pj*TN:(pj+1)*TN])    # indexes pj (col) + k, NOT pi  → multicast along COL
        c += sp.dot(a, b)                            # TM×TK · TK×TN ; compiler vectorizes

    sp.store(C[pi*TM:(pi+1)*TM, pj*TN:(pj+1)*TN], c)
```

**What the programmer sees** (architecture-aware): a PE grid; each PE owns a C-tile that is
*stationary*; A and B *streamed* over K; and — without writing any directive — the
**access patterns imply the multicast** (A has no `pj` → fans out along the row; B has no
`pi` → fans out along the column). No physical coordinates, no hand-written forwarding.

**What the compiler does** (hidden): derive the reuse trichotomy → C stationary (local L1
accumulator), A multicast along each row, B multicast along each column, K accumulated
temporally; then place logical PEs → physical columns, choose multicast realization
(AXIS broadcast vs mem-tile fan-out vs systolic shift), vectorize `sp.dot`, double-buffer
the A/B loads against compute, insert locks.

### How it lowers to the doc-02 engine
```
Iteration (i,j,k);  π = (i↦pi, j↦pj)  spatial;  k temporal (in ker π);  reduction f kills k.
  C: M_C access (i,j) → L_C = span{e_k};  e_k temporal  → STATIONARY (output-stationary) ✓
  A: M_A access (i,k) → L_A = span{e_j};  e_j spatial    → MULTICAST along row (pj) ✓
  B: M_B access (k,j) → L_B = span{e_i};  e_i spatial    → MULTICAST along col (pi) ✓
  Reduction R = span{e_k} ⊆ ker π  → R_time = R, R_space = ∅  → local accumulate, no cascade.
Evacuate C[pi,pj] only after the full K-stream (doc 02 R1 hazard, discharged).
```

### Flipping the dataflow (the one-knob change)
Same algorithm, weight-stationary instead — keep `B` resident and spread the K-reduction
across a PE column:

```python
    b = sp.acc((TK, TN), stationary=True)     # B now stationary
    # ...stream A and partial-C; put k on the grid:
@sp.kernel(grid=(PK, PJ))                      # reduction K is now SPATIAL
```

Now `R = span{e_k}` is **spatial** → `R_space ≠ ∅` → the compiler synthesizes a **cascade
reduction** down the column (doc 02 §5/R2, AIE cascade stream), and `A` becomes the
multicast/streamed operand. The *kernel math is unchanged*; only the residency + grid
declaration moved. That is the CUDA/Triton bargain — you rewrite intent minimally, the
compiler re-derives all the plumbing.

### Sequential reference (the oracle, for free)
Running `gemm` with the fallback semantics — `pe_id` iterating all `(pi,pj)`, `stream` a
`range`, `load` a slice, `acc` a buffer — executes a plain triple-loop GEMM in CPython.
So the *same source* is both the spatial program and its correctness spec.

## 6. Open questions (think-together queue)
1. **Multicast vs systolic-forward defaulting.** Derive multicast by default and override
   to systolic shift — but what's the cost model that picks for the user when they don't
   override? (Array size, source fan-out limit, AXIS channel count — ties to the MCSA
   efficiency numbers.)
2. **Where does the K-tile size `TK` live** — programmer (architecture-aware) or compiler
   (autotuned)? Triton makes it a `tl.constexpr` the autotuner sweeps; lean same.
3. **Stream typing.** Should channels be typed by *element + capacity* (Kahn bounded-FIFO,
   feeding the deadlock-legality open problem) at the surface, or inferred?
4. **One model commit.** Is "tiled-SPMD + neighbor streams + multicast" enough to express
   the workloads in doc 03 (stencil halos, conv), or does conv's sliding window force a
   richer stream type (overlapping windows = neither clean multicast nor clean forward)?
