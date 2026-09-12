# MAESTRO — reading notes

> **Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows:
> A Data-Centric Approach Using MAESTRO**
> Kwon, Chatarasi, Pellauer, Parashar, Sarkar, Krishna — *MICRO 2019* (arXiv:1805.02566v6).
> Georgia Tech + NVIDIA. MAESTRO = *Modeling Accelerator Efficiency via Spatio-Temporal
> Reuse and Occupancy.*

*Per-paper deep-read notes (convention: `papers/*.md` = single-paper notes;
`../docs/notes.html` = AMD-NPU cross-paper synthesis hub). Read as grounding for the
cost-model gaps in [`../spatial-dsl/04-spatial-triton-dsl.md`](../spatial-dsl/04-spatial-triton-dsl.md)
§6 and [`../spatial-dsl/05-composition-and-fusion.md`](../spatial-dsl/05-composition-and-fusion.md) §4.*

---

## Why we're reading this (the hook)

Our `spatial-dsl` engine answers **what mappings are legal** (the `(σ,π)` space-time map +
reduction legality). It does **not** yet answer **what a legal mapping costs** — and several
open questions (doc 04 §6 multicast-vs-systolic-forward defaulting; doc 05 §4 fusion-regime
selection) are *cost-model* questions. MAESTRO is an analytical cost model for exactly DNN
dataflows. The thesis to test as we read: **MAESTRO is the quantitative complement to our
legality engine** — same reuse structure, costed instead of just gated.

Headline alignment (from §2.3, confirmed at overview): MAESTRO derives all reuse from
**two behaviors × {spatial, temporal}** — multicast (inputs) and reduction (outputs). That
2×2 is the same object as our **reuse trichotomy** (doc 04 §3), reached from the cost side.

---

## Structural map

| § | Title | Notes status |
|---|---|---|
| 1 | Introduction | ✅ done |
| 2 | Background — 2.1 Tensors · 2.2 Accelerators · 2.3 Data Reuse Taxonomy · 2.4 Dataflow Defn · 2.5 Existing Expressions | ✅ done |
| 3 | Describing Dataflows — 3.1 Data-Centric Representation · 3.2 Dataflow Playground · 3.3 Hardware Implications of Reuse · 3.4 Row-stationary Example | ✅ done |
| 4 | Quantitative Dataflow Analysis — 4.1 Preliminary Engines · 4.2 Performance · 4.3 Cost · 4.4 Complex/Multi-cluster · 4.5 Validation | _todo_ |
| 5 | Case Studies — 5.1 Dataflow Trade-offs · 5.2 HW Design-Parameters | _todo_ |
| 6 | Related Works | _todo_ |
| 7 | Discussion & Future Work + Conclusion | _todo_ |

---

## §1 Introduction

**Thesis.** *Dataflow* = (1) how you **schedule** DNN compute (loop transforms: ordering +
tiling) + (2) how you **map** compute across PEs. It dominates utilization and data reuse,
and — since **moving data costs more energy than computing on it** [Chen, Horowitz] —
optimizing dataflow is *the* central accelerator-design lever. It decides how data moves
between multipliers (L0), local buffers (L1), and the global hierarchy (L2+).

**The coupling that makes it hard.** Performance depends on three tightly-coupled axes:
(1) the DNN layer's type/dimensions, (2) the dataflow, (3) the HW resources + connectivity.
A dataflow exploiting input-channel parallelism dies on layers with few channels; a
dataflow needing more bandwidth than the NoC provides under-utilizes the array; enlarging
L1 to fix that costs area + energy. **Co-optimizing microarch × dataflow is the open
problem** — and worse, the literature uses the word "dataflow" inconsistently and no
proposal spans the space exhaustively enough to be a reference.

**Four contributions:**

| # | Contribution | One-liner |
|---|---|---|
| 1 | **Data-centric notation** | mappings + reuse are *first-class*, vs compute-centric loop-nests that *infer* reuse from loop order. Can be an IR extracted from a loop-nest **or** written directly. |
| 2 | **Structured reuse reasoning** | each directive → a specific algorithmic reuse → a HW capability that exploits it. Claims to cover the **complete** reuse space. |
| 3 | **MAESTRO cost model** | in: DNN model + per-layer dataflow (directives) + HW config. out: exec time, energy (compute+buffer+interconnect), NoC cost. **90–95% accurate vs RTL, 1029–4116× faster** (10 ms vs 7.2–28.8 hr). |
| 4 | **DSE demo** | Pareto-optimal params. NVDLA-like (KC-Partitioned) VGG16 CONV11: **2.16× power gap** between energy- vs throughput-optimal; energy-opt uses 10.6× SRAM + 80% the PEs → 65% better EDP at 62% throughput. |

**Figure 1 — the 7-dim tensor notation** (used throughout; worth memorizing):
`N` batch · `K` output-ch · `C` input-ch · `R,S` filter rows/cols · `Y,X` input rows/cols ·
`Y',X'` output rows/cols. Three tensors: `I[n][c][y][x]`, `W[k][c][r][s]`,
`O[n][k][y'][x']` (partial sum `P` adds `c,r,s`). **Coupled dimensions:** an index is
*coupled* to a tensor when changing it moves the position in that tensor's data space —
`c` couples `I`+`W`; `k` couples `W`+`O`. (This "coupling" is MAESTRO's version of our
access-map analysis: a dimension absent from a tensor's index set is a reuse direction for
that tensor — cf. doc 04 `L_a = ker M_a`.)

## §2 Background

### 2.1 Tensors in DNNs
7 dims, 3 tensors; CONV2D is the focus (>90% of CNN compute). The coupling structure is
what lets you **transform the loop nest to hold one tensor stationary** over a span of space
or time — cutting global/local buffer accesses *and* energy (unchanging local wires).

---

#### 🔍 Fig 1 deep dive — notation, the partial-sum convention, and the two loop nests

**The 7 dimensions** (upper-case = size, lower-case = index), split by what they index:

| dim | meaning | size in Fig-1 example |
|---|---|---|
| `N` | input batch | 2 |
| `K` | output channels | 4 |
| `C` | input channels | 6 |
| `R, S` | filter rows, cols | 3, 3 |
| `Y, X` | **input** rows, cols | 8, 8 |
| `Y', X'` | **output** rows, cols | 6, 6  (`Y' = Y−R+1 = 8−3+1`) |

**The four tensors and their index signatures (Fig 1a):**

```
Input Activation   I[n][c][y][x]
Filter Weight      W[k][c][r][s]
Partial Sum        P[n][k][c][y'][x'][r][s]      ← 7 indices (the full iteration space)
Output Activation  O[n][k][y'][x']               ← 4 indices
```

**Partial-sum convention (the subtle bit).** A *partial sum* is **one single MAC product**,
fully indexed by **all seven** loop variables:

```
P[n][k][c][y'][x'][r][s]  =  W[k][c][r][s] · I[n][c][y'+r][x'+s]
```

The output is that partial sum **reduced over `c, r, s`**:

```
O[n][k][y'][x']  =  Σ_c Σ_r Σ_s  P[n][k][c][y'][x'][r][s]
```

Compare the index sets — the difference *is* the reduction:

```
P :  n  k  c  y'  x'  r  s     (7 dims)
O :  n  k     y'  x'           (4 dims)
         └──────────┴── c, r, s  =  the RED "accumulated" dims in Fig 1a  =  summed away
```

So `P` is the tensor **before** reduction (every multiply named individually); `O` is
**after**. The *red "accumulated dimension"* annotation marks exactly the dims present in
`P` but absent from `O`. MAESTRO names the partial sum on purpose — reduction is one of its
two reuse behaviors (§2.3, output tensors → reduction), and you can't talk about *where* the
reduction happens (spatial adder-tree vs temporal accumulator) without a name for the
pre-reduction value.

> **= our `spatial-dsl` reduction space.** `c, r, s` are the reduction dimensions = the
> kernel of the output access map (`O` doesn't index them). `P` is the pre-reduction tensor,
> `O = REDUCE(P)` — MAESTRO's "partial sum" *is* doc-02's REDUCE operand. The dims that
> *survive* into `O` (`n, k, y', x'`) are the candidate **space/time** axes; the dims that
> *vanish* (`c, r, s`) are exactly what σ/π must schedule as a reduction (temporal
> accumulate or spatial cascade).

**Two loop nests, same conv — they differ in the loop anchor (Fig 1c vs 1d):**

```c
// (c) INPUT-centric — loop over input pixels (y,x), SCATTER to outputs
for n,k,c: for y in 0..8: for x in 0..8: for r in 0..3: for s in 0..3:
    O[k][y-r][x-s] += W[k][c][r][s] * I[c][y][x];     // y,x range over INPUT extent (8)

// (d) OUTPUT-centric — loop over output pixels (y',x'), GATHER from inputs
for n,k,c: for y' in 0..6: for x' in 0..6: for r in 0..3: for s in 0..3:
    O[k][y'][x'] += W[k][c][r][s] * I[c][y'+r][x'+s]; // y',x' range over OUTPUT extent (6)
```

Identical result; linked by the substitution **`y = y' + r`**. (c) scatters — one input
contributes to several outputs (`y−r`); (d) gathers — the standard "sliding correlation"
form everyone uses, and the one `P`/`O` notation matches (`P` carries `y',x'`, not `y,x`).
Both drop `n` in the body for brevity. **Point of showing both:** the *same algorithm* has
many loop-nest spellings, and a loop nest is **compute-centric** — reuse is implicit in
whichever spelling + loop order you pick. That motivates §2.5's data-centric notation, where
reuse is *not* hostage to how you happened to write the loop.

---

### 2.2 DNN Accelerators (the abstract HW model, Fig 2)
Hundreds of PEs, each = L1 scratchpad + MAC ALU; a shared L2 staging buffer; a NoC tying
them together. **A systolic array is just a special NoC** — a 2D array with unidirectional
East/South links. HW parameters MAESTRO sweeps: `#PEs`, `L1 size`, `ALU vector width`,
`ALU precision`, `NoC bandwidth/latency`, and crucially **the spatial/temporal multicast +
reduction implementation**. This abstract model is what makes one cost model cover TPU,
Eyeriss, NVDLA, Shidiannao, MAERI, etc.

### 2.3 Data Reuse Taxonomy ★ (the load-bearing section)
**All reuse = two behaviors × {spatial, temporal}.** Behaviors split by *tensor role*:

| | **Multicasting** (input tensors) | **Reduction** (output tensors) |
|---|---|---|
| **Spatial** | read once → replicate via **wires** → many PEs | accumulate partials from many PEs via **adder tree / reduce-and-forward** |
| **Temporal** | read once from remote → replicate in **local buffer** → many time-steps, one PE | accumulate over time in an **accumulation register/buffer** (e.g. TPU) |

> **This is the same object as our reuse trichotomy (doc 04 §3) — but factored more
> cleanly.** We classified *operand reuse directions* and got 3 cells (stationary /
> multicast / stream). MAESTRO factors by **{role} × {space,time}** and gets 4:
>
> | MAESTRO cell | our name (doc 04) | AIE mechanism |
> |---|---|---|
> | spatial multicast | **multicast** | AXIS broadcast / mem-tile fan-out |
> | temporal multicast | **stationary** (input held, reused over time) | local L1 |
> | spatial reduction | **stream / cascade** ("reduce-and-forward" = cascade!) | AIE cascade stream / adder tree |
> | temporal reduction | (our output-stationary accumulator — the K-loop) | accumulator register |
>
> The lesson for `spatial-dsl`: our trichotomy **silently merged "temporal reduction"
> into "stationary."** MAESTRO's 2×2 separates *input-reuse* (multicast) from
> *output-reduce* (reduction) as orthogonal to *space-vs-time*. That's a sharper
> decomposition — and the "reduce-and-forward" spatial-reduction is **literally the AIE
> cascade** we lean on for weight-stationary GEMM (doc 04 §5). Worth importing into doc 02.

### 2.4 Dataflow Definition & Example (Fig 3)
A dataflow = schedule (ordering + tiling transforms on the Fig-1 conv) + a partition of data
to PEs. Classic taxonomy names them by the **least-frequently-changed tensor**:
weight- / output- / input-stationary. Fig 3 (weight-stationary, 4 PEs): `W1` temporal-
multicast, `I1` spatial-multicast, `P3_1` reduced across space *and* time — so even one
"weight-stationary" dataflow uses **three of the four taxonomy cells at once**. (Confirms:
the stationary-name is a coarse label; the precise description is the per-tensor reuse
assignment.) **Chen et al. refinement:** dataflows that differ only in concrete tile bounds
are the *same dataflow* (a mobile vs datacenter chip share traversal order, differ in tile
size) — dataflow = equivalence class over bounds.

---

#### 🔍 Fig 3 deep dive — one weight-stationary schedule shows all three reuse types

**Setup:** 4 PEs, a **2×2 kernel** (`R=S=2`, weights `W0 W1 / W2 W3`), channels + batch
omitted. Input = a 3×5 grid `I0…I14`; the kernel slides to produce `O0 O1 O2 O3 / …`.

**Part (A) — what one output is** (defines the `P{output}_{tap}` naming):

```
O0  =  P0_0  +  P0_1  +  P0_2  +  P0_3
     = W0·I0 + W1·I1 + W2·I5 + W3·I6      // 2x2 kernel at top-left; I5 is the pixel below I0 (input is 5 wide)
```
First subscript = which output, second = which kernel tap / accumulation step. Concrete
instance of Fig-1a's `P[...][r][s]` with `r,s` enumerated as taps 0–3.

**Part (b) — the timeline** (PE = space ↑, cycle = time →). Each cell = `weight·input → partial sum`:

| | cycle 0 | cycle 1 | cycle 2 |
|---|---|---|---|
| **PE3** | `W1·I2 → P1_1` | | `W1·I4 → P3_1` |
| **PE2** | `W1·I1 → P0_1` | | `W1·I3 → P2_1` |
| **PE1** | `W0·I1 → P1_0` | `P1_0 + P1_1` | `W0·I3 → P3_0` |
| **PE0** | `W0·I0 → P0_0` | `P0_0 + P0_1` | `W0·I2 → P2_0` |

Fixed PE→weight binding: **PE0,PE1 hold `W0`; PE2,PE3 hold `W1`** (always). Pairs
`(PE0,PE2)` and `(PE1,PE3)` each build one output by combining tap0 + tap1.

**The three reuse types overlaid (the point of the figure):**

| label | reuse | in the figure | taxonomy cell |
|---|---|---|---|
| ❶ Temporal Reuse | **weight stationary** — `W0` stays in PE0, `W1` in PE3 across cycles 0→2 | red bar along the top | temporal multicast (inputs-role: weight) |
| ❷ Spatial Reuse (Multicasting) | **`I1` fed to PE1 *and* PE2 in the same cycle** (halo/overlap of adjacent windows) | red arrows → into PE1, PE2 | spatial multicast |
| ❸ Spatial Reuse (Reduction) | `P0_0`(PE0) + `P0_1`(PE2) → `O0`, combined across PEs *and* across cycles | diagonal red arrow | spatial + temporal reduction |

> **= our reuse trichotomy on a real schedule.** weights = *stationary* (held in PE L1);
> inputs = *multicast* (AXIS broadcast across the PEs that share the halo); partial sums =
> *cascade reduction* (reduce-and-forward across a PE pair, then accumulate over time). The
> `I1`-to-two-PEs overlap is the miniature answer to **doc-04 open-Q#4** (does conv's sliding
> window break clean multicast?): here it does *not* — it's still multicast, just to an
> **overlapping** PE set rather than a disjoint row/column.

### 2.5 Existing Expressions = Loop Nests = *compute-centric*
Loop-nest notation (imperative + explicit `parallel-for`) is **compute-centric**: data
movement is *implicit*, inferred from loop order + tiling + parallelism. Eyeriss v2 uses a
**22-dimensional loop nest**. MAESTRO's critique of the compute-centric / polyhedral route
(this is the paper's central methodological claim, and it cuts at our engine):

- doesn't *precisely* model reuse → hard to get throughput/energy accurately;
- heavyweight linear-algebra frameworks → impractical at real scale;
- chokes on **non-affine** subscripts (modulus in strided conv);
- can't analyze **explicit / multi-level parallelism** (which DNN dataflows always have);
- ignores **spatial reuse** (data reuse via wires/across-PEs — *not* cache spatial locality).

→ motivates a **data-centric IR** where data movement + organization are first-class, so the
cost model needs no heavyweight LA and runs fast.

> **Tension to hold for our design.** MAESTRO argues *against* the polyhedral/compute-centric
> approach **for cost modeling** — yet our `(σ,π)` engine is exactly an affine/polyhedral
> formulation. Two reconciling observations: (1) we use the affine maps for **legality**, not
> costing — MAESTRO's critique is about *cost estimation* speed/precision, a different job;
> (2) MAESTRO's "coupled dimensions" *is* access-map kernel analysis under another name, so
> the data-centric directives and our `(σ,π)` are **two surfaces over the same reuse facts** —
> exactly the "surface ⟂ engine" stance of doc 04. The open question this sharpens: should our
> engine *cost* via a MAESTRO-style closed-form occupancy model rather than anything polyhedral?

## §3 Describing Dataflows

The data-centric notation = **four directives**. Pedagogical vehicle is a **1D convolution**
`O[x'] += W[s]·I[x'+s]` (Fig 4; `X'=12`, `S=6`).

### 3.1 Data-Centric Representation — the four directives

A dataflow = (1) *schedule* over time (loop transforms → reuse) + (2) *map* across PEs
(parallelism). Captured by:

| Directive | Meaning | Parameters |
|---|---|---|
| **`SpatialMap(size, offset) α`** | distribute dim `α` **across PEs** (parallelism) | `size` = #indices of `α` per PE; `offset` = shift in start index between **consecutive PEs** |
| **`TemporalMap(size, offset) α`** | distribute dim `α` **across time-steps within a PE**; *same chunk on all PEs at a given step* | `size` = #indices per PE per step; `offset` = shift between **consecutive time-steps** |
| **Data Movement Order** | the **sequence** of the map directives = the order data mappings change over time (≈ loop order) | — |
| **`Cluster(size)`** (§3.2) | group PEs / sub-clusters into logical clusters of `size` → enables **multi-dim spatial** distribution | `size` = PEs per cluster |

**Two semantic subtleties that carry all the weight:**
- **`offset` vs `size`:** if `offset < size` → **indices overlap** across consecutive PEs (or
  steps). This is how you express **convolutional halo reuse** — the skewed input iteration
  space `x'+s`. `offset = size` → clean partition (no overlap).
- **`TemporalMap` ⇒ spatial multicast.** Because all PEs receive the *same* index chunk of a
  temporally-mapped dim at each step, that data can be **multicast across PEs** in that step.
  (`SpatialMap` gives each PE a *different* chunk ⇒ parallelism; if #PEs < dim size the map
  **folds over time**.)

Worked map for the 1D conv (Fig 4d): **`SpatialMap(2,2) X'`** (each PE owns 2 output cols) +
**`TemporalMap(3,3) S`** (3 weights per step, same on all PEs). That single directive pair =
one complete, unique dataflow — equivalent to the two red-boxed loops in the loop-nest form.

---

#### 🔍 Fig 4 deep dive — the Rosetta Stone (one dataflow, four notations)

Fig 4 shows **one output-stationary dataflow** for the 1D conv, expressed four equivalent ways.
The algorithm (a): `for x' in 0..12: for s in 0..6: O[x'] += W[s]·I[x'+s]`.

**(b) Loop-nest (compute-centric) — a 3-level tiling** of the `(x', s)` iteration space, one
level per memory tier. The factorization:
- `x' = 12` split as `x'2(2) · x'1(3) · x'0(2)` → `x' = 6·x'2 + 2·x'1 + x'0`
- `s  = 6`  split as `s2(1) · s1(2) · s0(3)`  → `s  = 6·s2 + 3·s1 + s0`

```c
for  x'2 in 0..2:        par_for s2 in 0..1:      // On-chip global buffer
  par_for x'1 in 0..3:   for     s1 in 0..2:      // PE L1   ◄── RED BOX = the map onto PEs
    for   x'0 in 0..2:   for     s0 in 0..3:      // PE L0 register
      O[x'] += W[s]·I[x'+s]
```
The **red box** = the two loops that place work on PEs: `par_for x'1 (3)` → **3 PEs**, `for s1
(2)` → time within a PE.

**(c) Data-centric (full) → (d) abbreviated.** The same dataflow as directives, one block per
memory tier; **gray = "omittable"** (upper = inferable global-buffer level; lower = intra-PE
L0, doesn't affect *inter-PE* reuse). The **red box is the only essential part**:

```
                                          (X': size at this level)
  [gray]  TemporalMap(6,6) X'  · TemporalMap(6,6) S · Cluster(3)    ← Off-chip / global
  [RED]   SpatialMap(2,2) X'   · TemporalMap(3,3) S                 ← PE L1  ⇒ (d) abbreviated
  [gray]  Cluster(1) · TemporalMap(1,1) X' · TemporalMap(1,1) S     ← Intra-PE / L0 register
```
So **(d) = `SpatialMap(2,2) X'` + `TemporalMap(3,3) S`** is the whole dataflow; everything else
is inferred. This is the payoff of the data-centric form: the 8-line loop nest collapses to 2
directives that name *exactly* the reuse-relevant decisions.

**(e) The resulting PE × time map** (3 PEs, each owns 2 output columns; weights advance in time):

| | PE0 | PE1 | PE2 |
|---|---|---|---|
| **t=0** | `x'={0,1}, s={0,1,2}` | `x'={2,3}, s={0,1,2}` | `x'={4,5}, s={0,1,2}` |
| **t=1** | `x'={0,1}, s={3,4,5}` | `x'={2,3}, s={3,4,5}` | `x'={4,5}, s={3,4,5}` |

Read the reuse straight off it:
- **`SpatialMap(2,2) X'`** → each PE keeps a **disjoint** pair of outputs (`offset=size=2`, no
  overlap) — **output-stationary**, partial sums accumulate locally.
- **`TemporalMap(3,3) S`** → the `s`-chunk is **identical across all 3 PEs** at each step ⇒
  **weights spatially multicast**; and it **advances over time** (`{0,1,2}→{3,4,5}`) ⇒ the
  output accumulates over `s` = **temporal reduction** of outputs.
- Why output-stationary and not weight-stationary? **Directive order**: `SpatialMap X'` (outer)
  before `TemporalMap S` (inner) ⇒ a PE sweeps *all* `s` for its fixed `x'` before it's done ⇒
  the *output* is the thing held put. Swap the order → weight-stationary (§3.2 lever #1).
- Note `X':6` at the PE level covers only `x'=0..5`; the outer `TemporalMap(6,6) X'` **folds**
  the array over the second half `x'=6..11` (3 PEs × 2 cols = 6 < 12).

**(f) Iteration-space view.** The `(X', S)` grid; each **dot = one partial sum** (a MAC). The
three colored **columns** = the PE that owns each `x'`; the **orange arrow ↑ along S** = time
(a PE marching through its `s` reductions); the two horizontal bands = `t=0` (`s=0,1,2`) and
`t=1` (`s=3,4,5`).

> **Rosetta Stone value for us:** (b)↔(d) is exactly the "surface ⟂ engine" claim — the
> compute-centric loop nest and the data-centric directives are two spellings of *one* mapping.
> Our `(σ,π)` engine is a *third* spelling of the same object: `SpatialMap X'` = `π` picks `x'`
> as the spatial axis; `TemporalMap S` = `s` is temporal (`s ∈ ker π`) and is the reduction
> dim; output-stationary = `R = span{e_s} ⊆ ker π` (doc-02 R1). MAESTRO's "omittable gray
> boxes" = our claim that only the `(σ,π)` on the reuse-carrying axes matters; the rest is
> inferred.

### 3.2 Dataflow Playground — six dataflows, the five levers (Fig 5)

Small edits to the base dataflow expose different reuse. The five transformations "that
capture all possible aspects of dataflows: scheduling, tiling, and mapping":

1. **Directive order → stationarity.** `SpatialMap X'` *then* `TemporalMap S` explores all `S`
   before advancing `X'` ⇒ partial sums (`X'`) reused over `S` ⇒ **output-stationary**.
   **Interchange** the two ⇒ weights (`S`) reused over `X'` ⇒ **weight-stationary**. → the
   informal "X-stationary" name is *not* a precise spec; the directive order is.
2. **Which dim is Spatial vs Temporal → the reuse type** (parallelism axis + multicast axis).
3. **Mapping `size` → temporal reuse depth.** `size=1` ⇒ full temporal reuse of the stationary
   tensor, none of the other; **increasing `size` ⇒ *partial* temporal reuse** — this is how
   you capture **convolutional input reuse** across steps (e.g. Fig 5E spatial-maps `S` with
   size>offset).
4. **`Cluster` → multi-dimensional spatial distribution.** Directives *above* a `Cluster` see
   **logical clusters**; directives *below* see **inside** a cluster. So one `SpatialMap` per
   level ⇒ two spatial axes at once. This is what expresses **real accelerators**: Eyeriss
   (spatially distributes `R` and `Y`), NVDLA (distributes `K` and `C`). Clusters also model
   coarse-grained PEs (SIMD lanes, GPU Tensor Cores).

---

#### 🔍 Fig 5 deep dive — six dataflows, two orthogonal levers

All six are the same 1D conv (`X'=12`, `S=6`); only the directives change. Read off the figure:

| ID | Mapping (top→inner) | Spatial axis | Temporal Reuse | Spatial Reuse | Informal name |
|---|---|---|---|---|---|
| **A** | `SpatialMap(1,1) X'` · `TemporalMap(1,1) S` | `X'` | **temporal reduction of outputs** (output-stat.) | multicast of weights | Output-Stationary |
| **B** | `TemporalMap(1,1) S` · `SpatialMap(1,1) X'` | `X'` | **temporal multicast of weights** (weight-stat.) | multicast of weights | Weight-Stationary |
| **C** | `TemporalMap(1,1) X'` · `SpatialMap(1,1) S` | `S` | temporal reduction of outputs | **spatial reduction of outputs** | Collaborative Output-Stationary |
| **D** | `SpatialMap(1,1) S` · `TemporalMap(1,1) X'` | `S` | temporal multicast of weights | **spatial reduction of outputs** | Collaborative Weight-Stationary |
| **E** | `SpatialMap(2,2) S` · `TemporalMap(1,1) X'` | `S` (size 2) | weight-stat. **+ partial temporal multicast of inputs** (halo: `X=3` reused by PE1 over t=0,1) | spatial reduction of outputs | Tiled Collaborative Weight-Stationary |
| **F** | `TemporalMap(3,3) S` · `SpatialMap(1,1) X'` · **`Cluster(3)`** · `SpatialMap(1,1) S` · `TemporalMap(1,1) X'` | `X'`×`S` (2 levels) | weight-stat. | spatial reduction of outputs | Clustered Tiled Collaborative Weight-Stationary |

**Two independent knobs generate the whole zoo:**

1. **Which dim is `SpatialMap`'d ⇒ multicast vs reduction.**
   - Spatial-map **`X'`** (a *non*-reduction, output-carrying dim) → PEs own **disjoint**
     outputs; the reduction dim `S` stays temporal → **weights spatially multicast**, no spatial
     reduction. (A, B)
   - Spatial-map **`S`** (*the reduction dim*) → different PEs make partial sums *for the same
     output* → **spatial reduction of outputs** ("**Collaborative**"). (C, D, E, F)
2. **Directive order (which map is inner) ⇒ stationarity.**
   - inner `TemporalMap S` / outer `S` held → **weight-stationary** (B, D, E, F).
   - the output `X'` held while `S` sweeps → **output-stationary** (A, C).

**The informal name decomposes into the knobs:**
`[Clustered]` = has a `Cluster` (F) · `[Tiled]` = spatial `size>1` capturing input halo reuse
(E) · `[Collaborative]` = spatial reduction, i.e. `S` is spatially mapped (C–F) ·
`Output-/Weight-Stationary` = the temporal-reuse knob.

> **This is our reduction-legality trichotomy, enumerated.** Spatial-mapping a **non-reduction**
> dim (A, B: `X'`) ⇒ `R = span{e_s} ⊆ ker π` ⇒ **`R_time` only** = temporal accumulate (doc-02
> **R1**, output-stationary). Spatial-mapping the **reduction** dim (C–F: `S`) ⇒ `R_space ≠ ∅`
> ⇒ **spatial reduction** = cascade / reduce-and-forward (doc-02 **R2**) — MAESTRO's
> "Collaborative." E's `SpatialMap(2,2) S` (`offset<size`) ⇒ overlap ⇒ **multicast to an
> *overlapping* PE set** (the doc-04 open-Q#4 halo case, again benign). F's `Cluster(3)` ⇒ a
> **2-axis `π`** onto a PE grid. So Fig 5 walks the entire `(σ,π)` + reduction design space of
> doc-02/04 on one tiny kernel — a ready-made test matrix for our lowering.

### 3.3 Hardware Implications of Reuse ★★ (the part that is *our* doc-04 lever)

**Table 1 — reuse follows from *coupling*.** For a spatially-mapped dim and the innermost
temporally-mapped dim, each tensor (F/I/O) gets an opportunity by this rule:

> A tensor is **spatially multicast** along a spatially-mapped dim `α` **iff `α` is *decoupled*
> from it** (the tensor doesn't index `α` ⇒ constant ⇒ broadcast). If `α` is a **reduction
> dim of the output** (`C`, `R`, `S`), spatially mapping it ⇒ **spatial reduction** of `O`.
> Temporally mapping a reduction dim ⇒ **temporal reduction**; temporally mapping a decoupled
> dim ⇒ **temporal multicast** (= stationary).

Example from the paper: spatially map `K` ⇒ input `I` (which has no `K`) is **broadcast** to
all PEs; innermost temporal `C` ⇒ `C` changes each step and is a reduction dim ⇒ **temporal
reduction** of outputs. Coupling cheat-sheet: `F=W[k,c,r,s]` (no `X/Y`), `I=I[c,y,x]` (no
`K`), `O=O[k,y',x']` (no `C,R,S` — those are its reduction dims).

> **This *is* our doc-04 §3 trichotomy, and MAESTRO's "coupling" is literally `ker M_a`.**
> "decoupled from tensor ⇒ multicast" = "operand doesn't index the spatial axis ⇒ multicast."
> Independent confirmation of the whole reuse-derivation move.

**Table 2 — four reuse categories × hardware (the multicast⇄forward lever, spelled out):**

| Reuse | Comm. type (by tensor role) | HW implementation choices |
|---|---|---|
| **Spatial** | **Multicast** (inputs) | **Fanout** (bus / tree)  **— or —**  **Store-and-Forward** (systolic array) |
| **Spatial** | **Reduction** (outputs) | **Fan-in / reduction tree**  **— or —**  **Reduce-and-Forward** (systolic array) |
| **Temporal** | **Multicast** (inputs) | **multiple reads from a (stationary) buffer** |
| **Temporal** | **Reduction** (outputs) | **multiple read-modify-write to a buffer** |

> **This is exactly the doc-04 §3 "multicast ⇄ systolic-forward lever" and open-Q#1.** Spatial
> multicast can be realized as **fanout (broadcast)** *or* **store-and-forward (systolic
> shift)** — the very choice we said needs a cost model. Same for spatial reduction:
> **reduction-tree** *or* **reduce-and-forward (cascade)**. MAESTRO gives the **taxonomy of
> choices**; §4 gives the **cost model that picks** — precisely the piece doc-04/05 are
> missing. AIE map: fanout = AXIS broadcast / mem-tile fan-out; store-and-fwd + reduce-and-fwd
> = **shared-L1 neighbor forward / cascade stream**; temporal multicast = L1 stationary;
> temporal reduction = accumulator RMW.

### 3.4 Extended Example: Row-Stationary (Fig 6)
Eyeriss's row-stationary dataflow on **6 PEs (2 clusters × 3 PEs)**, on the Fig-1 layer.
Reuse directions reproduce Eyeriss exactly: **weights reused horizontally**, **outputs
(partial-sum accumulation) vertically**, **inputs diagonally** (skewed replication across
clusters within a step). Subtlety: it's **weight-stationary at unit-time-step granularity but
row-stationary at coarse granularity** — showing the notation captures a *real, published*
dataflow and its precise reuse geometry. Motivates §4: we need a *fast, accurate* way to
*quantify* reuse for dataflows this complex.

## §4 Quantitative Dataflow Analysis

_todo_

## §5 Case Studies

_todo_

## §6 Related Works

_todo_

## §7 Discussion & Future Work

_todo_

---

## Cross-paper takeaways (for spatial-dsl)

_todo — filled as we read._
