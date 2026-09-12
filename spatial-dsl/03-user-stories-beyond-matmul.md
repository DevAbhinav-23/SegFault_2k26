# User stories beyond matmul — stencil, conv2d/3d, CNNs

*2026-06-19. Stress-tests the `(σ,π)` IR from
[`02-spacetime-ir-and-reduction-legality.md`](02-spacetime-ir-and-reduction-legality.md)
on real workloads, and pins the surface language (Python). Each story is chosen to expose
a **new wrinkle** matmul did not.*

---

## 0. Surface decision: Python, plain-function algorithm + schedule builder

The "pragma" *spirit* is (i) **ignorable** annotations that (ii) leave a **runnable
reference**. In Python both are achievable and in fact *stronger* than C:

> A plain typed Python nested-loop kernel **executes in CPython as-is** — that is the
> sequential reference oracle, for free, with no build. C `#pragma` only gets you the
> oracle after a compile.

Three candidate surfaces:

| Surface | Algorithm | Schedule | Verdict |
|---|---|---|---|
| C `#pragma` (01-design-overview.md strawman) | C loop nest | `#pragma spatial …` | truly ignorable, but no cheap reference run; HLS heritage |
| Python **comment-pragma** | `def` nest | `# pragma: spatial …` | keeps OpenMP look, but stringly-typed, poor tooling |
| **Python decorator + schedule builder** ✅ | `@sp.kernel def …` | `s = sp.schedule(k); s.place(...)` | idiomatic (Triton/Exo/Allo), real handles, reference runs |

**DECISION (locked 2026-06-19): the third — Python, plain-function algorithm + schedule
builder.** The schedule *method calls* **are** the pragmas — same clauses (`place`,
`reduce`, `stationary`, `stream`, `reside`, `pipeline`), now first-class Python objects
instead of strings. Strip the schedule → the `@sp.kernel` function is plain Python = the
oracle. The C `#pragma` strawman in `01-design-overview.md` §2 is retained only as historical
motivation; all new design work targets this Python surface.

Surface used below (syntax illustrative):

```python
import spatial as sp

@sp.kernel
def k(...): ...                         # pure typed Python; runs as reference

s = sp.schedule(k, target=sp.NPU)       # the "pragmas" live here, separately
ax = s.axes()                           # loop handles
s.reduce(ax.k); s.place(px=ax.i, py=ax.j); s.stationary("C"); s.pipeline(ax.k)
```

Everything in §1–§4 lowers to the same `(σ,π) + REDUCE` IR; only `(D, F_a, f)` change.

---

## 1. Stencil — 2-D Jacobi (and iterative time-tiling)

```python
@sp.kernel
def jacobi2d(In: sp.f32[H, W], Out: sp.f32[H, W]):
    for i in range(1, H-1):
        for j in range(1, W-1):
            Out[i, j] = 0.2 * (In[i,j] + In[i-1,j] + In[i+1,j] + In[i,j-1] + In[i,j+1])
```

- **Domain** `D = [1,H-1)×[1,W-1)`; access maps are *shifts*: `In` is read at
  `{(i,j),(i±1,j),(i,j±1)}` — five **uniform (constant-offset) dependences**.
- **No separate reduction axis** — the 5-point sum is a fixed, fully-unrolled window
  (`R = ∅`). So the legality work here is *not* about reductions.

**New wrinkle: overlapping reuse → halo + line-buffer.** Adjacent outputs share input
points (`Out[i,j]` and `Out[i,j+1]` both read `In[i,j]`). The reuse is along `i,j` but
**shifted**, so it doesn't vanish into a clean `ker M` — instead it becomes a **sliding
window**. Two consequences the surface must name:

```python
s = sp.schedule(jacobi2d, target=sp.NPU)
s.place(px=ax.i, py=ax.j)            # tile rows→PEs (a strip-mined block per PE column)
s.window(In, dims=(ax.i, ax.j), halo=1)   # ← NEW: declare the ±1 ghost region
s.linebuffer(In, along=ax.j)        # ← NEW: shift-register reuse along the fast axis
```

- `window(..., halo=1)` makes the **ghost/halo region** explicit, so neighboring PEs
  exchange a 1-element boundary (AIE: **shared-L1 neighbor access**, notes/AIE-arch).
- `linebuffer(In, along=j)` says "keep a sliding line in L1 and shift" — the classic
  stencil-on-spatial pattern (line buffers + shift registers). Uniform dependences ⟹ the
  `(σ,π)` legality (L2 causality) is trivially satisfiable with a skewed `σ`.

**Iterative stencil (Jacobi over `T` timesteps)** adds **temporal reuse across `t`** →
**time-tiling** (overlapped/diamond tiling; Bandishti et al., SC'12). The surface needs
one more clause:

```python
s.time_tile(ax.t, factor=Tt, shape="diamond")   # ← NEW: legal skew that fuses timesteps
```

This is where the polyhedral pedigree pays off: time-tiling legality is exactly an affine
`σ`-skew condition, already solved in the references.

> **Takeaway:** stencils need no new *theory* — uniform-dependence space-time mapping
> covers them — but they add surface vocabulary: `window/halo`, `linebuffer`, `time_tile`.

---

## 2. Conv2D — where stencil **⊕** matmul meet

```python
@sp.kernel
def conv2d(In: sp.f16[IC, IH, IW], Wt: sp.f16[OC, IC, KH, KW], Out: sp.f32[OC, OH, OW]):
    for oc in range(OC):
        for oh in range(OH):
            for ow in range(OW):
                acc = 0.0
                for ic in range(IC):
                    for kh in range(KH):
                        for kw in range(KW):
                            acc += In[ic, oh*S+kh-P, ow*S+kw-P] * Wt[oc, ic, kh, kw]
                Out[oc, oh, ow] = acc
```

- **6-D domain** `(oc, oh, ow, ic, kh, kw)`. `S`=stride, `P`=pad.
- **Reduction** `Out[oc,oh,ow] = REDUCE(+, f, In*Wt)` with `f` projecting away
  `(ic, kh, kw)` ⟹ **`R = span{e_ic, e_kh, e_kw}` — a *rank-3* reduction** (vs. rank-1 in
  GEMM). The reduction-legality theorem (§5 of doc 02) applies unchanged; there are just
  more reduction directions to split into `R_time / R_space`.

**The key insight — conv2d = stencil ⊕ matmul:**
- the `ic` reduction is the **matmul-like** part (channel contraction);
- the `(oh,ow)` vs `(kh,kw)` overlap is the **stencil-like** part (sliding window over the
  spatial dims — same halo/line-buffer reuse as §1);
- the IR handles it because `R` is "just a bigger kernel" and reuse is "just more
  directions." Nothing new is needed at the theory level — strong evidence the framework
  is **not** matmul-shaped.

**New wrinkles:**

1. **Strides & padding.** `oh*S + kh - P` is still affine (good), but `P` makes the
   read domain non-rectangular at the boundary → polyhedral handles it; the surface adds
   `s.pad(In, P)` (and the reference oracle runs on a padded view).
2. **Many named dataflows = which 2 of 6 axes go to `π`.** Doc 02's "drop an index"
   result generalizes; the CNN-accelerator names fall out:

   | `place(px=, py=)` | stationary operand | named dataflow |
   |---|---|---|
   | `oc, ow` | weights `Wt` (reused over `oh,ow`… partial) | **weight-stationary** |
   | `oh, ow` | outputs `Out` | **output-stationary** |
   | `ic, oc` | inputs `In` (reused over `oc`) | **input-stationary** |
   | row of `kh`↔`oh` folded | filter+input rows | **row-stationary (Eyeriss)** |

   Row-stationary (Chen et al., ISCA'16) is a *particular* `(σ,π)` that folds a filter row
   and input row onto each PE — i.e. it is **not** a fresh idea, it's one point in our map
   space. That our IR can *name and derive* it (instead of hand-architecting it) is the
   pitch.
3. **Direct conv vs im2col.** im2col lowers conv → GEMM but explodes memory (`KH·KW×`
   input replication). Our IR runs **direct conv** (it's affine), keeping reuse explicit
   and avoiding the blowup — a concrete advantage to state.
4. **Channel reduction choice meets the AIE cascade.** Put `ic` in `R_time` (output-
   stationary, local accumulator) **or** spread `ic` across a PE line → `R_space` →
   **cascade stream** reduction (doc 02 §6). Same one-pragma flip as GEMM.

```python
s = sp.schedule(conv2d, target=sp.NPU)
s.pad(In, P); s.reduce(ax.ic, ax.kh, ax.kw)
s.place(px=ax.oc, py=ax.ow)          # weight/output-ish; ic stays temporal (R_time)
s.window(In, dims=(ax.oh, ax.ow), halo=(KH//2, KW//2)); s.linebuffer(In, along=ax.ow)
s.vectorize(ax.ic, 8); s.pipeline(ax.kw)
# flip to spatial channel reduction:  s.place(px=ax.oc, py=ax.ic)  → cascade over ic
```

---

## 3. Conv3D — dimensionality pressure on the *solver*, not the theory

```python
@sp.kernel
def conv3d(In: sp.f16[IC, ID, IH, IW], Wt: sp.f16[OC, IC, KD, KH, KW],
           Out: sp.f32[OC, OD, OH, OW]):
    for oc, od, oh, ow in sp.grid(OC, OD, OH, OW):
        acc = 0.0
        for ic, kd, kh, kw in sp.grid(IC, KD, KH, KW):
            acc += In[ic, od*S+kd-P, oh*S+kh-P, ow*S+kw-P] * Wt[oc, ic, kd, kh, kw]
        Out[oc, od, oh, ow] = acc
```

- **8-D domain**; **rank-4 reduction** `R = span{e_ic, e_kd, e_kh, e_kw}`.
- **The theory is untouched** — same `(σ,π) + REDUCE`. What changes is *scale*: 8 loop
  dims, only **2 physical PE axes + time**, so you can spatialize at most 2 and must tile
  + sequentialize the other 6.

**New wrinkles (all about the mapper, = 01-design-overview.md open problem #2):**
- **Combinatorial mapping choice** — "which 2 of 8 → `π`" is a large DSE; the `(σ,π)`
  solver (LP/ILP over coefficients) now matters more than any single elegant dataflow.
- **Tiling becomes mandatory** and multi-level (every axis tiled L3→L2→L1).
- **L1 capacity rider bites hard** (doc 02 §6): 3-D windows are large; the 64 KB hard cap
  (notes: exceeding L1 is a *compile* failure) is a *constraint that gates `(σ,π)`
  acceptance*, not a perf knob. The mapper must prove the working set fits *before* it
  blesses a schedule.

> **Takeaway:** conv3d is the honesty check — it shows the framework **generalizes by
> growing `(D, F_a, f)`**, and that the remaining hard problem is the *solver/heuristics*,
> exactly where we already pointed the open contribution.

---

## 4. CNN — the multi-layer story (graph-level schedule)

A CNN is a pipeline: `conv → relu → pool → … → fc → softmax`. Per-op `(σ,π)` is §2; the
**new layer is the graph**. (This is ARIES's "Challenge 1: multi-layer apps".)

```python
@sp.graph
def cnn(x):
    a = conv2d(x, W1); a = relu(a); a = maxpool(a)
    b = conv2d(a, W2); b = relu(b); b = maxpool(b)
    return fc(flatten(b), W3)        # + softmax (see escape below)

g = sp.schedule_graph(cnn, target=sp.NPU)
g.place_layer(conv1, cols=range(0,4))     # ← NEW: spatial partition layers across columns
g.place_layer(conv2, cols=range(4,8))
g.fuse(conv1, relu1, pool1)               # ← NEW: keep tiles on-chip (L2/L1), skip L3 round-trip
g.pipeline_layers()                       # ← NEW: layer-pipeline parallelism (macro-dataflow)
```

**New wrinkles:**
1. **Two-level schedule.** Graph level (`place_layer`, `fuse`, `pipeline_layers`) sits
   *above* the per-op `(σ,π)`. The graph schedule decides array partitioning and inter-
   layer buffering; each op still lowers via §2. This is the clean separation ARIES blurs.
2. **Layer fusion = inter-tile dataflow across ops.** Producer output tile → consumer
   input tile without evicting to L3 — on AIE, staged in the **memory tile (L2)** or
   forwarded via shared-L1/AXIS. The hyper-rectangle overlap analysis (doc 02 / ARIES
   §4.2) extends across the layer boundary.
3. **Pooling is *also* an A/C reduction.** `maxpool` = `REDUCE(max, window)`; `max` is
   associative+commutative → **the spatial-reduction machinery (doc 02 §5/R2) applies
   verbatim** (cascade a `max` instead of a `+`). Average-pool = `REDUCE(+)/n`. So pooling
   is in-scope with zero new theory — a nice unification.
4. **The one escape: softmax.** `exp(x)/Σexp(x)` has a **data-dependent denominator**
   (and numerically a running-max trick) → **not pure affine** → this is the
   expressiveness-cliff (01-design-overview.md open problem #2). Scope decision: the conv/pool/relu/fc
   trunk is fully in the affine model; the softmax head drops to the **Spatial-style
   explicit-dataflow escape hatch** (streaming reduce + elementwise). Be honest about the
   seam rather than overclaim (the ARIES lesson).

---

## 5. What the stories prove

| Story | New surface vocabulary | New *theory* needed? |
|---|---|---|
| Stencil (Jacobi) | `window/halo`, `linebuffer`, `time_tile` | **No** — uniform-dependence space-time |
| Conv2D | `pad`; reuses `window/linebuffer` + `reduce` | **No** — rank-3 `R`, stencil⊕matmul |
| Conv3D | (none new) | **No** — bigger `(D,F_a,f)`; stresses the *solver* |
| CNN | `place_layer`, `fuse`, `pipeline_layers` (graph) | **No** for trunk; **escape hatch** for softmax |

Two conclusions:

1. **The `(σ,π)+REDUCE` core is not matmul-shaped.** Every dense story is the *same*
   machinery with a different domain/reduction — stencils add windowing sugar, conv adds
   a higher-rank reduction that is literally stencil⊕matmul, conv3d just grows the domain,
   CNN adds a graph layer above. **No new legality theory** is required until softmax.
2. **The real work is exactly where we already aimed it:** the **solver** (which axes to
   spatialize under L1-capacity + routing) and **AIE routing/deadlock legality** for the
   spatial-reduction (now also `max`-cascades for pooling). The surface grows by a handful
   of honest clauses; the hard core does not move.

---

## Story references (supplement to `REFERENCES.md`)
- **Eyeriss / row-stationary**: Chen, Emer, Sze, "Eyeriss: A Spatial Architecture for
  Energy-Efficient Dataflow for CNNs," *ISCA* 2016.
- **Systolic conv/matmul via space-time (polyhedral)**: Cong & Wang, "PolySA," *ICCAD*
  2018; Wang, Guo, Cong, "AutoSA: A Polyhedral Compiler for High-Performance Systolic
  Arrays on FPGA," *FPGA* 2021. ← directly does conv/matmul space-time mapping; closest
  precedent for §2–§3.
- **Stencil time-tiling**: Bandishti, Pananilath, Bondhugula, "Tiling Stencil Computations
  to Maximize Parallelism," *SC* 2012 (diamond/overlapped tiling).
- **Stencil line-buffer / image pipelines**: Hegarty et al., "Darkroom," *SIGGRAPH* 2014;
  Halide (PLDI'13) stencil schedules.
