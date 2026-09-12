# Composition & fusion — GEMM→ReLU, GEMM→GEMM in Spatial-Triton

*2026-06-19. How multi-op / multi-layer programs compose in the
[Spatial-Triton](04-spatial-triton-dsl.md) surface. Closes the loop with the CNN graph in
[doc 03 §4](03-user-stories-beyond-matmul.md) and open-question #4 in doc 04.*

Fusion is the whole point of a spatial array: keep intermediates on-chip, off DRAM. The
PE-tile unit makes the easy cases trivial and the hard case structured.

---

## 1. GEMM → ReLU — trivial epilogue fusion

ReLU is pointwise on the output tile. In output-stationary GEMM each PE *already holds*
its C-tile in L1 after the K-reduction, so ReLU runs in place, before store — zero data
movement, no new concept:

```python
@sp.kernel(grid=(PI, PJ))
def gemm_relu(A, B, C):
    pi, pj = sp.pe_id()
    c = sp.acc((TM, TN), stationary=True)
    for kk in sp.stream(0, K, TK):
        c += sp.dot(sp.load(A[pi*TM:(pi+1)*TM, kk:kk+TK]),
                    sp.load(B[kk:kk+TK, pj*TN:(pj+1)*TN]))
    c = sp.relu(c)                       # ← epilogue: pointwise, in-PE, free
    sp.store(C[pi*TM:(pi+1)*TM, pj*TN:(pj+1)*TN], c)
```

**Principle.** Any *pointwise* epilogue (ReLU, bias-add, scale, quantize) fuses for free
in the PE-tile model — it runs in the PE that owns the output tile.

> **The exception:** an epilogue that needs a *row/column reduction* (softmax, layernorm)
> is **not** pointwise — it needs cross-PE communication along a spatial line (a
> multicast/cascade of the partial max/sum). That is the doc-02 spatial-reduction
> machinery applied to the epilogue, and the first genuine escape toward the
> data-dependent tier (softmax, doc 03 §4).

---

## 2. GEMM → GEMM — the structurally interesting case

With `C1 = A@B1` (reduce K) and `C2 = C1@B2` (reduce J), one fact governs everything:

> **The producer's free dimension `J` is the consumer's reduction dimension.** This single
> coupling is what makes fused GEMM-chains hard — and the same shape recurs in attention
> (`QKᵀ → softmax → @V`: the score free-dim is softmax's reduction *and* the `@V`
> contraction). It dictates that the natural on-chip hand-off **streams along `J`**.

There are **three composition regimes**, and they are just the doc-04 **reuse trichotomy
(stationary / multicast / stream) applied to the *intermediate*** instead of an operand.

### (a) Inline fused kernel — flash-attention style, intermediate stays in L1

One kernel; the intermediate `C1` tile never leaves the PE. PE`(pi,pn)` owns `C2[i,n]` and
reduces over `J`; *inside* each `J`-step it produces one `C1` tile via an inner
K-reduction, ReLUs it, and consumes it immediately:

```python
@sp.kernel(grid=(PI, PN))
def mlp(A: f16[M,K], B1: f16[K,J], B2: f16[J,N], C2: f32[M,N]):   # C2 = relu(A@B1) @ B2
    pi, pn = sp.pe_id()
    c2 = sp.acc((TM, TN), stationary=True)              # final output, output-stationary
    for jj in sp.stream(0, J, TJ):                      # GEMM2 reduction over J
        c1 = sp.acc((TM, TJ))                           # transient C1 tile, lives in L1 only
        for kk in sp.stream(0, K, TK):                  # GEMM1 reduction over K
            a  = sp.load(A [pi*TM:(pi+1)*TM, kk:kk+TK]) # no pn   → multicast along the row
            b1 = sp.load(B1[kk:kk+TK, jj:jj+TJ])        # no pi,pn → broadcast to whole array
            c1 += sp.dot(a, b1)
        c1 = sp.relu(c1)                                # ReLU = GEMM1's pointwise epilogue
        b2 = sp.load(B2[jj:jj+TJ, pn*TN:(pn+1)*TN])     # no pi   → multicast along the column
        c2 += sp.dot(c1, b2)                            # feed straight into GEMM2
    sp.store(C2[pi*TM:(pi+1)*TM, pn*TN:(pn+1)*TN], c2)
```

`C1` **never touches L3** — the win. The cost is honest: `A[pi,kk]` is re-loaded for every
`jj`, trading A-reuse over `J` for keeping `C1` on-chip (unless A fits L1 and is marked
resident). Same recompute-vs-reuse tradeoff as Triton's fused kernels.

### (b) Graph composition — modular, intermediate parked on-chip

Two ordinary kernels; the compiler keeps the hand-off on-chip:

```python
@sp.graph
def mlp(A, B1, B2):
    h = gemm_relu(A, B1)            # [M, J]
    return gemm(h, B2)             # [M, N]

g = sp.map(mlp, target=sp.NPU)
g.stay(h, level="L2")              # h lives in the mem-tile, never spills to L3 (temporal fusion)
```

Run GEMM1 fully, keep `h` in L2, run GEMM2. Saves DRAM bandwidth; no overlap. `h` is a
**"stationary" intermediate** parked at L2.

### (c) Spatial pipeline — the option GPUs do *not* have

Place the two layers on **different regions of the array** and stream `h` between them:

```python
g.pipeline(gemm_relu, gemm, along="J")   # GEMM1 on some columns, GEMM2 on others; h streams along J
```

Because producer-free-`J` = consumer-reduction-`J`, GEMM1 emits `h` column-by-column and
those columns flow straight into GEMM2's accumulation — a **systolic composition of two
GEMMs**, overlapped in space, `h` forwarded core→core (AXIS / shared-L1), never to L3. A
GPU cannot do this (SMs do not form a spatial pipeline); the AIE array can.

---

## 3. The unifying point

Kernel composition is **not a new mechanism** — it is the reuse trichotomy on
*intermediates*:

| Regime | where `h` lives | trichotomy analog | when to pick |
|---|---|---|---|
| (a) inline fused | L1, same PE | stationary / transient | intermediate small, max on-chip locality, willing to recompute operands |
| (b) temporal fusion | L2 mem-tile | stationary (at L2) | intermediate too big for L1; just want to dodge the L3 round-trip |
| (c) spatial pipeline | streamed region→region | stream | enough cores to dedicate regions; want producer/consumer overlap |

So GEMM→GEMM reuses the **doc-02 `(σ,π)` engine at the graph level** — the two-level
schedule sketched for CNNs in doc 03 §4 (graph schedule above per-op `(σ,π)`). And
GEMM→ReLU→GEMM is just (a)/(b)/(c) with the ReLU folded into GEMM1's epilogue (§1).

The sequential-fallback property (doc 04 §2) survives composition: `@sp.graph` runs as
ordinary function calls, each kernel falls back to loops, so the whole fused program is
still its own reference oracle.

---

## 4. Open questions (think-together queue)
1. **Who picks the regime?** (a)/(b)/(c) is a cost-model decision (intermediate size vs L1
   cap, core budget, DRAM bandwidth). Default automatically, or make it an explicit
   `g.stay/g.pipeline` knob the user owns? Lean: explicit knob + a sensible default — same
   stance as the multicast/forward lever (doc 04 §6).
2. **Spatial-pipeline reduction coupling.** The (c) hand-off works cleanly when
   producer-free = consumer-reduction (GEMM chains, attention). What about a consumer
   whose reduction is a *different* axis (e.g. conv after conv, reducing channels)? Does
   the stream need reshaping/transpose in the mem-tile between regions?
3. **Back-pressure & deadlock.** (c) wires two regions with bounded FIFOs — the
   doc-02/§8 routing-and-deadlock-legality open problem now spans a producer/consumer
   boundary, not just one kernel. Bounded-FIFO (Kahn) proof must cover the pipeline.
4. **Softmax in the middle** (e.g. attention). A row-reduction epilogue (§1 exception)
   between two GEMMs forces a cross-PE reduction *inside* the fused loop — the flash
   pattern. Whether the inline-fused (a) form can express the running-max/running-sum
   rescale cleanly is the real test of the surface for transformers.
