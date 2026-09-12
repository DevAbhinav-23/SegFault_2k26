# LLD M8 — Kernels, fixtures, demo

*Phase 1, 2026-09-12. Owner: **Person C**. Reads on top of [`01-requirements.md`](01-requirements.md)
§3.7 (FR-K1…K5), [`02-hld.md`](02-hld.md) §7 (the three walk-throughs),
[`06-interfaces.md`](06-interfaces.md) §7.1 (the frozen surface) and §9 (fixture format),
[`04-test-plan.md`](04-test-plan.md) §4–§5, and [`05-work-breakdown.md`](05-work-breakdown.md)
§5–§6 (the demo and the Q&A drill). Citation convention as in `01-requirements.md` §0;
`PC §X` = `../hackathon/01-paradigm-comparison.md` r3.*

**The kernel sources and schedules in §3 are specification text, not implementation.** They are
the user-facing surface — the thing on the screen for the first ninety seconds of the pitch — and
they are given complete so that Person A writes M1's grammar against a fixed target at D0 and
nobody discovers at D4 that a kernel needs a construct the grammar rejects (R-08).

---

## 1. Purpose and FR IDs satisfied

| FR | What M8 owes it |
|---|---|
| **FR-K1** | `kernels/w1_gemm.py` — the GEMM source, the output-stationary schedule, the fixture, the golden, a passing oracle diff |
| **FR-K2** | the weight-stationary flip: **same kernel text**, four changed schedule lines and one dropped `tile`, `R_space = span{e_k0}`, a cascade chain, and `B` **resident for the whole run** (RULING 9) |
| **FR-K3** | `kernels/w2_jacobi.py` — Jacobi with halo exchange, plus the odd-`T` fixture the architect's D-4 override requires |
| **FR-K4** | `kernels/w3_sw.py` — the Smith-Waterman wavefront, and the reversed-skew rejection the pitch shows |
| **FR-K5** | W4 FFT is out of scope, and `test_K5_scope_documented` (M7 LLD §7.7) makes the documentation a gate |
| **FR-S3** | every source is inside the accepted grammar, **checked by hand here** and by a throwaway `ast.parse` script at D0 |
| **FR-S18** | every schedule is deletable: `kernels/*.py` separates the kernel from the schedule so "delete the schedule" is "delete a function" |
| **NFR-5** | each kernel ships a runnable script that prints `s.summary()` |
| **G5** | the honest-limits slide (§5) and the prepared answers (§6.3) |

**Non-responsibilities.** M8 does not change the library. If a kernel needs a library change it
is an FR change, not a kernel edit (HLD §2, M8). M8 does not own the mapping plans in §3's
walk-throughs — those are `02-hld.md` §7 and Person B's.

---

## 2. Public entry points

```
kernels/w1_gemm.py     gemm, PARAMS, schedule_os(target)  , schedule_ws(target), main()
kernels/w1_gemm_bf16.py gemm_bf16, PARAMS_LARGE, schedule_os(target), main()
kernels/w2_jacobi.py   jacobi, PARAMS, schedule(target), main()
kernels/w3_sw.py       sw, PARAMS, schedule(target), main()
kernels/rejections.py  bad_stationary(), bad_skew(), bad_capacity()   # the three demo rejections
demo/run_demo.py       the D7 script, driven by keystroke, no network, no device required
demo/honest_limits.md  the slide's text, asserted by test_trace_disclaimer and
                       test_K5_scope_documented
tests/fixtures/make_fixture.py   the generator named in every meta.json
```

`main()` for each kernel is NFR-5's runnable script: it builds the schedule, prints
`s.summary()`, and exits 0 with no toolchain installed (FR-S20 — the surface and the checker do
not import `air`).

Every `schedule_*` function takes `target` **positionally and explicitly**, never defaulting to
`"auto"` (D-13, `python/air/api/_trace.py:182` shells out to `xrt-smi`).

---

## 3. The three kernels — sources, schedules, and the grammar check

Shape parameters are module-level integers, captured by the annotations and by the loop bounds.
Every source below was checked by hand against FR-S3's six-clause grammar; the clause each
construct relies on is named in the table after it.

### 3.1 W1 — GEMM, output-stationary (FR-K1)

`kernels/w1_gemm.py`, verbatim:

```python
"""W1 — dense GEMM. The kernel is the specification: delete every s.* line
below and this file still runs in CPython and still computes C = A @ B."""

import spatial as sp

M, N, K = 64, 64, 64
TM, TN, TK = 32, 32, 16
PI, PJ = 2, 2
PK = K // TK                      # 4 — the flip's grid


@sp.kernel
def gemm(A: sp.f32[M, K], B: sp.f32[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]


def schedule_os(target):
    """Output-stationary: C stays in a PE, A and B multicast into it."""
    s = sp.schedule(gemm, target=target)
    ax = s.axes()
    s.grid(PI, PJ)
    s.tile(ax.i, TM); s.tile(ax.j, TN); s.tile(ax.k, TK)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")
    s.pipeline(ax.k0)
    return s


def schedule_ws(target):
    """Weight-stationary. THE KERNEL IS UNCHANGED. Four clauses differ from
    schedule_os -- grid, place, stationary, double_buffer -- and the tile clause
    loses its j term, so B is resident rather than merely spatially stationary.
    The reduction over k stops being temporal and becomes spatial, so it lowers
    to a cascade down the PE line."""
    s = sp.schedule(gemm, target=target)
    ax = s.axes()
    s.grid(PK)                                     # was  grid(PI, PJ)
    s.tile(ax.i, TM); s.tile(ax.k, TK)             # NO tile on j: TN = N = 64, so this PE's
                                                   # B[kchunk, :] tile is [TK, N] = [16, 64]
                                                   # and is constant for the whole run. With
                                                   # tile(ax.j, TN) it would move every j0
                                                   # trip and "weight-stationary" would be a
                                                   # word, not a fact (RULING 9).
    s.reduce(ax.k, op="+")
    s.place(px=ax.k0)                              # was  place(px=ax.i0, py=ax.j0)
    s.stationary("B")                              # was  stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A")                           # was  double_buffer("A", "B"). A is
                                                   # re-fetched on each of the two i0 trips,
                                                   # so its get-per-trip alloc is the one
                                                   # ping-pong candidate; double_buffer("B")
                                                   # would be a PINGPONG-SHAPE rejection,
                                                   # because a tile resident for the whole run
                                                   # has no streaming loop to be allocated
                                                   # inside (FR-S13, 03-lld-M3-checker.md
                                                   # §3.13 condition 2).
    return s


def main():
    print(schedule_os("npu1").summary())
```

| Construct | FR-S3 clause |
|---|---|
| `for i in range(M)` | 1 — `range` with a shape-parameter bound |
| `C[i, j] += A[i, k] * B[k, j]` | 2 (augmented assignment), 3 (affine subscripts), 4 (`*` on scalars) |
| `sp.f32[M, K]` | FR-S2 — a dtype descriptor subscripted by shape entries |

**Why the flip is 1-D `grid(PK)` and not `grid(PK, PJ)`.** Adopted as REVIEW-round1 RULING 4;
`04-test-plan.md` §4 and `02-hld.md` §7.1's flip paragraph now both carry it. The 2-D forms are
unusable, for two independent reasons:

1. **`place(px=ax.i0, py=ax.k0)` cannot satisfy `stationary("B")`.** `F_B = (k, j)`, so
   `ker M_B = span{e_i}`; with `i0` placed, `e_i0 ∉ ker Sπ`, so `ker M_B ⊄ ker Sπ` and FR-L3
   rejects it with `STATIONARITY`. That placement is in fact **the demo's second rejection**
   (§3.4, paired with `stationary("C")`), so using it for the flip as well would be showing the
   same two lines as both the success and the failure.
2. **A 2-D grid strip-mines the cascade axis on both targets.** `PHYSICAL_HERD` is
   `{"npu1": {1: (4,), 2: (1, 4)}, "npu2": {1: (8,), 2: (2, 4)}}`
   (`python/air/api/_trace.py:88-91`) and the resolution rule is *the largest divisor of each
   extent not exceeding the cap* — `_resolve_physical` ends
   `tuple(_largest_divisor_at_most(g, cap) for g, cap in zip(self.grid, default))`
   (`python/air/api/_trace.py:1347-1373`), with `repeats = g // p` at `:1297`. Logical `(4, 2)`:
   on `npu1` → physical `(1, 2)`, **repeats `(4, 1)`**; on `npu2` → physical `(2, 2)`, **repeats
   `(2, 1)`**. Either way the four cascade stages fold onto fewer physical rows, and `air.api`'s
   own text says a cascade *"is a physical link between neighbouring cores"*
   (`python/air/api/_trace.py:1771-1778`). FR-L6(b) — "mapped to a **contiguous line** of PEs" —
   is then false.
   Logical `(4,)` resolves to physical `(4,)` with repeats `(1,)` on **both** targets, so the
   chain is four real neighbouring cores and FR-L6(c) ("the herd is 1-D") holds outright.

*Documented alternative, if Person B wants to keep a `PJ` axis:* `grid(PJ, PK) = (2, 4)` with
`place(px=ax.j0, py=ax.k0)` resolves on `npu1` to physical `(1, 4)`, repeats `(2, 1)` — the
cascade axis lands on the cap-4 dimension as a real line of four, and `j0` is time-multiplexed
instead. It is strictly more plan to write for no demo gain, and it would also flip the chain
**direction**: a 2-D herd is one column of four rows and must descend, whereas the adopted 1-D
`grid(4)` is one row of four columns and must **ascend** — both measured on the pinned wheel
(REVIEW-round1 P-R3).

**The chain ascends.** `aie.cascade_flow(%tile_0_2, %tile_1_2)`, `(1,2)→(2,2)`, `(2,2)→(3,2)`,
`aircc` exit 0; the descending variant of the same 1-D module fails with `'aie.cascade_flow' op
source tile must be to the North or West of the destination tile`. `ChannelPlan.chain_direction`
carries it, and `test_M6_cascade_chain` asserts it.

**L1 for the flip**, computed the way FR-L9 computes it: `acc [32,64] f32 = 8192`,
`recv [32,64] f32 = 8192` (the cascade's receive tile, synthesised by M4 §6.2),
`a [32,16] f32 = 2048` **doubled by `double_buffer("A")` → 4096**, `b [16,64] f32 = 4096`
⇒ **24 576 B** of 65 536 — the figure `02-hld.md` §7 tabulates. It is twice the base schedule's
`12 288`, and that is the honest price of the flip: every per-PE tile now spans the whole `N`,
because `j` is untiled so that `B` is resident rather than merely spatially stationary
(RULING 9). The per-stage cascade payload is one `[32,64] f32` partial tile, `8 192 B`, sent
`PK-1 = 3` times **per `i0` trip**, and there are two trips. `double_buffer` names `A` and only
`A`: `A`'s slab moves with `i0`, so its alloc is a direct child of the streaming loop and the
pass can fire, while `double_buffer("B")` would be a `PINGPONG-SHAPE` rejection — declaring it
is exactly the "promising a pass will fire when it cannot" failure D-5 exists to prevent. `B` is
genuinely stationary under `place(px=ax.k0)` —
`ker M_B = span{e_i} ⊆ ker Sπ_u = span{e_i, e_j}` — so FR-L3 accepts the declared clause and
**B-O5 is closed**; the residency line then says how long it stays, which is the half of B-O5
the predicate never answered.

### 3.2 W1-large — the smoke and IR-facts shape

`kernels/w1_gemm_bf16.py` is the same body with `bf16` inputs, sized to VF §E.5's measured
probe so the ping-pong facts are asserted on comparable IR (D-11):

```python
M, N, K = 256, 256, 256
TM, TN, TK = 64, 64, 64
PI, PJ = 4, 4


@sp.kernel
def gemm_bf16(A: sp.bf16[M, K], B: sp.bf16[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
```

L1: `acc [64,64] f32 = 16 384` + `a [64,64] bf16 = 8 192` + `b = 8 192`, with `a`/`b` doubled ⇒
**49 152 B** of 65 536. In `f32` the same shape would be 81 920 B and would be rejected by
FR-L9 — which is why this fixture is `bf16`, and `04-test-plan.md` §4's larger-`W1` paragraph
now says `bf16` explicitly. This fixture has **no**
`expected.npz` and **no** `oracle.npz`: it exists only for levels S and I, so no `bf16` numerical
tolerance is ever needed (see the M6 LLD §3.6, Q-4).

Physical herd: logical `(4, 4)` on `npu1` → physical `(1, 4)`, repeats `(4, 1)`; on `npu2` →
physical `(2, 4)`, repeats `(2, 1)`.

### 3.3 W2 — Jacobi 5-point with halo exchange (FR-K3)

`kernels/w2_jacobi.py`, verbatim:

```python
"""W2 — 2-D 5-point Jacobi over T timesteps. The halo is never written by the
user: halo=1 is a width, and the ghost rows are derived from the access map."""

import spatial as sp

T = 4
H, W = 16, 16                     # H, W are the INTERIOR extents; U carries the halo rows
PI = 2
HS = H // PI                      # 8 interior rows per PE; PI*HS == H exactly


@sp.kernel
def jacobi(U: sp.f32[T + 1, H + 2, W]):
    for t in range(T):
        for i in range(1, H + 1):
            for j in range(1, W - 1):
                U[t + 1, i, j] = 0.2 * (U[t, i, j]
                                        + U[t, i - 1, j] + U[t, i + 1, j]
                                        + U[t, i, j - 1] + U[t, i, j + 1])


def schedule(target):
    s = sp.schedule(jacobi, target=target)
    ax = s.axes()
    s.grid(PI)
    s.tile(ax.i, HS)                                 # one row strip per PE
    s.place(px=ax.i0)
    s.sequential(ax.t)                               # t stays temporal
    s.window("U", dims=(ax.i, ax.j), halo=1)         # declare the +/-1 ghost region
    s.exchange("U", along=ax.i0, halo=1)             # put boundary rows, then get ghosts
    s.reside(U="L1")
    s.double_buffer("U")
    return s


def main():
    print(schedule("npu1").summary())
```

| Construct | FR-S3 clause |
|---|---|
| `range(1, H + 1)` | 1 — an affine bound over a shape parameter |
| `U[t + 1, i, j] = …` | 2, 3 — affine subscripts including `t + 1` and `i ± 1` |
| `0.2 * (…)` | 4 — a float literal and scalar `*`/`+` |

**The fixture, settled (REVIEW-round1 RULING 1).** `U` is one rank-3 parameter
`sp.f32[T + 1, H + 2, W]`; `H = W = 16` are the **interior** extents, so the `i` axis has extent
`H = 16`, `HS = 8` divides it, and the invariant `PI·HS == H` holds exactly. Rows `0` and `H + 1`
and columns `0` and `W - 1` are read-only Dirichlet boundary. Two strips of
`(HS + 2) × W = 10 × 16` f32 give `2 × 10 × 16 × 4` = **1 280 B** per core (`02-hld.md` §7), and
`04-test-plan.md` §4 carries the same row.

The earlier `H = W = 18` proposal fixed the divisibility problem (`4 ∤ 14`) and left `PI = 4`,
which is the real blocker: **`PI = 2` is forced by the measured 2-S2MM circuit budget.** At
`PI = 4` the first interior PE needs three circuit-switched inbound channels and `aircc` fails
with `'aie.connect' op … TileID(1, 2) targets same dst` (REVIEW-round1 P-R2). FR-M4's
interior-PE assertion therefore moves to the `PI = 4` **negative** fixture, which is checked and
**never emitted**: `03-lld-M4-mapping.md` §3.8's `DMA-CHANNELS` rejects it before `aircc` ever
sees it, and `test_P3_dma_inbound` is the test.

**The odd-`T` fixture (the architect's D-4 override).** D-4 as written made an odd trip count a
`SWAP-PARITY` rejection; the architect has overridden it — odd `T` is legal via peeling, and
FR-L14 now says so. M8 ships a **second W2 fixture, `w2_odd`, with `T = 5`**, whose `meta.json`
carries `"expect": "accept"`, and `test_W2_odd_T` asserts (a) the schedule checks clean, (b) the
emitted herd body contains one `air.sequential(0, 4, 2)` plus one peeled copy of the update, and
(c) the oracle diff is within `1e-5` (the update multiplies by `0.2`; `04-test-plan.md` §8
item 5). A **third** fixture, `w2_zero_t` with `T = 0`, is the `SWAP-PARITY` negative — the only
reachable one after the override, so without it `test_D3_catalogue_complete` fails (§4).

**`U` is read and written.** `air.api` marks any written tensor `is_output = True`
(`python/air/api/ops.py:310`), which removes it from the launch's input list
(`python/air/api/_compile.py:98-103`), so `CompiledKernel.__call__` would hand the device a
**zeroed** initial field and compute a plausible wrong answer. This is why the M6 LLD §3.5 drives
`XRTBackend.load(...)`'s invoker directly — it uploads and syncs back every buffer. Recorded
here because it is a property of *this kernel's shape*, not of the driver.

### 3.4 W3 — Smith-Waterman anti-diagonal wavefront (FR-K4)

`kernels/w3_sw.py`, verbatim:

```python
"""W3 — Smith-Waterman local alignment, anti-diagonal wavefront. The skew is
the one thing the user must state, and the one thing we check before codegen."""

import spatial as sp

MQ, NR = 32, 32
PJ = 4
CW = NR // PJ                     # 8 columns per PE
MATCH = 2                         # substitution score on a match
MISMATCH = -1                     # ... and on a mismatch
GAP = 1                           # linear gap penalty


@sp.kernel
def sw(q: sp.i32[MQ], r: sp.i32[NR], S: sp.i32[MQ + 1, NR + 1]):
    for i in range(1, MQ + 1):
        for j in range(1, NR + 1):
            sub = MATCH if q[i - 1] == r[j - 1] else MISMATCH
            S[i, j] = max(0, S[i - 1, j - 1] + sub,
                          S[i - 1, j] - GAP, S[i, j - 1] - GAP)


def schedule(target):
    s = sp.schedule(sw, target=target)
    ax = s.axes()
    s.grid(PJ)
    s.tile(ax.j, CW)                              # CW columns per PE
    s.place(px=ax.j0)                             # column block -> PE coordinate
    s.skew(time=(ax.i, ax.j0))                    # anti-diagonal wavefront
    s.forward("S", along=ax.j0, dir="W->E")       # left-edge value to the east neighbour
    s.reside(S="L1", q="L1", r="L1")
    return s


def main():
    print(schedule("npu1").summary())
```

| Construct | FR-S3 clause |
|---|---|
| `sub = …` | 6 — a local scalar bound by simple assignment and read in the same body |
| `MATCH if q[i-1] == r[j-1] else MISMATCH` | 8 — a value-level conditional: one `arith.select`, both arms evaluated, no branch |
| `MATCH`, `MISMATCH`, `GAP` | 7 — module-level `int`s resolved from `fn.__globals__` |
| `max(0, …, …, …)` | 5 — a call the grammar admits, with 4 scalar arguments |
| `S[i-1, j-1] + sub`, `- GAP` | 4 — scalar arithmetic and integer literals |
| `S[i - 1, j - 1]`, `q[i - 1]` | 3 — affine subscripts |

**Three properties of this source, all settled by REVIEW-round1 RULING 2:**

1. **The comparison is a value, not control flow.** `MATCH if q[i-1] == r[j-1] else MISMATCH` is
   FR-S3 item 8: the comparison and both arms are scalar expressions, so it becomes one
   `ExprNode.Select` and lowers to `arith.select` (`06-interfaces.md` §5.5). Both arms are
   evaluated and no PE diverges, which is why it is inside the affine subset while an `if`
   **statement** is not. The earlier arithmetic dodge `sub = 2 - 3*min(d*d, 1)` is deleted: it
   worked only for an alphabet encoded `0..3`, it was four tokens longer, and it is not what a
   reader of Smith-Waterman expects to see on the slide. A table lookup `Sub[q[i-1], r[j-1]]`
   remains impossible — that is a **data-dependent subscript**, which FR-S3 rejects by name.
2. **`MATCH`, `MISMATCH` and `GAP` are module-level `int`s**, resolved from `fn.__globals__`
   (FR-S3 item 7) exactly as `M`, `H` and `T` are in the other two kernels' bounds and
   annotations. This is A's D0 ruling, now written into the requirement; the fallback ("the
   kernel inlines `2`, `-1` and `1`") is retired.
3. **`q` and `r` are `sp.i32`, not `sp.i8`** — `i32` matches `04-test-plan.md` §4's "`i32`" row
   and removes every width question from the oracle.

**The L1 figure is 240 B**, and it counts `q` and `r`, which the first draft of `02-hld.md` §7.3
and `04-test-plan.md` §4 omitted. `q[i-1]` is indexed by the temporal axis only, so every PE
holds the whole vector: `MQ × 4 = 128 B` — and because `q` does not index `j` at all, M4
classifies it **multicast along px** and stages it on one broadcast channel (`QIn`). `r[j-1]` is
indexed by `j = CW·j0 + j1`, so each PE holds its own band: `CW × 4 = 32 B`. With
`prev`/`cur` at 36 B each and the two 1-element edge buffers, the per-core total is
`128 + 32 + 72 + 8` = **240 B** of 65 536 (`02-hld.md` §7). Every document now carries that
number.

Divisibility: `j`'s extent is `NR = 32` and `CW = 8` divides it (FR-S7); `i`'s extent is
`MQ = 32`, which is **even**, so the `prev`/`cur` swap's unroll-by-two has the even trip count
FR-L14 requires.

### 3.5 The three demo rejections

`kernels/rejections.py`. Each is a two-line edit of a working schedule, so the audience sees
what changed.

```python
def bad_skew(target):
    """L2-CAUSALITY. 'Time is just the row index' — the naive schedule, and the
    one that silently computes garbage in every other system."""
    s = sp.schedule(w3.sw, target=target)
    ax = s.axes()
    s.grid(PJ); s.tile(ax.j, CW); s.place(px=ax.j0)
    s.skew(time=(ax.i,))                          # was  skew(time=(ax.i, ax.j0))
    s.forward("S", along=ax.j0, dir="W->E")
    s.reside(S="L1", q="L1", r="L1")
    return s


def bad_stationary(target):
    """STATIONARITY. Place i and k, then claim C is stationary."""
    s = sp.schedule(w1.gemm, target=target)
    ax = s.axes()
    s.grid(PI, PK)
    s.tile(ax.i, TM); s.tile(ax.j, TN); s.tile(ax.k, TK)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.k0)                   # was  place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    return s


def bad_capacity(target):
    """L1-CAPACITY. Fits at 61 440 B — until double_buffer doubles A and B."""
    s = sp.schedule(w1_192.gemm, target=target)   # M=N=K=192, f32
    ax = s.axes()
    s.grid(2, 2)
    s.tile(ax.i, 96); s.tile(ax.j, 96); s.tile(ax.k, 32)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")                     # <- this line is what tips it over
    return s
```

**`bad_stationary` is arithmetically unambiguous and is the safe rejection.** `F_C = (i, j)`, so
`ker M_C = span{e_k}`; `Sπ = [e_i0; e_k0]` places `k0`, so `e_k0 ∉ ker Sπ` and the containment
`ker M_C ⊆ ker Sπ` fails. `STATIONARITY` fires with no interpretive latitude.

**`bad_capacity`'s numbers, worked**, because the message prints them:

| Buffer | Shape | Bytes | Doubled? |
|---|---|---|---|
| `acc` | `[96, 96] f32` | 36 864 | no |
| `a` | `[96, 32] f32` | 12 288 | **yes** |
| `b` | `[32, 96] f32` | 12 288 | **yes** |
| undoubled total | | **61 440** | ≤ 65 536 ✓ |
| **charged total (FR-L9)** | | **86 016** | > 65 536 ✗ by 20 480 |

This is the message `04-test-plan.md` §5 item 3 asks for — "fits until ping-pong doubles it".
FR-L9's earlier example (`TM=TN=TK=128` in `bf16`) did **not** have that property: it is 98 304 B
before any doubling, so it never demonstrated the doubling. **FR-L9's acceptance now names this
case** (REVIEW-round1 EDIT-41), so one fixture serves both `test_L9_capacity` and the slide:
`w1_l1_overflow`.

**`bad_skew` is a rejection, and the rule that makes it one is now a requirement.** The pitch's
headline is *"an illegal skew is rejected with the violating dependence vector"* (05-wbs §5
item 4), and it survives on **one** stated rule (FR-S17, REVIEW-round1 RULING 6):

> `skew(time=(a, b))` defines `σ`'s **leading row** as the sum of the named axes. The remaining
> `σ` rows are the default loop order restricted to axes that are **neither named in `skew` nor
> placed**.

Apply it to `skew(time=(ax.i,))` on W3: `i` is skewed, `j0` is **placed**, so neither is
appended and `Sσ = [e_i; e_j1]`. The tile-crossing representative of the cross-PE dependence is
`d = (0, 1, −7)` over `(i, j0, j1)`, and `Sσ·d = (0, −7)`, whose first non-zero entry is `−7`:
**rejected**, with the vector printed. Had `j0` been appended instead, `Sσ·d = (0, 1, −7)` would
be lexicographically positive and the schedule would be **accepted** — the demo's headline turns
on exactly that one exclusion, which is why it is now in FR-S17, FR-L2 and
`03-lld-M3-checker.md` §3.3 line 24 rather than in a single pseudocode line.

Note also that *reversing* the two terms (`skew(time=(ax.j0, ax.i))`) is **not** a rejection:
`skew` is a **sum**, so the reversal produces an identical `σ` and the schedule is accepted. The
demo must drop the term, not reverse it — 05-wbs §5 item 4 says so in those words. `bad_stationary`
remains the unambiguous backup rejection if the live demo needs one.

---

## 4. Fixtures

One directory per fixture under `tests/fixtures/`, in the frozen format of `06-interfaces.md`
§9. `expected.npz` is computed by an **independent numpy expression**, never by our oracle;
`oracle.npz` is the decorated kernel's output, committed so a CPython regression is a file diff
(§9 rule 1 and 4).

| Fixture | Params | dtype | Seed | `tol` | Levels | L1 per core (FR-L9 charge) |
|---|---|---|---|---|---|---|
| `w1` | `M=N=K=64, TM=TN=32, TK=16, PI=PJ=2` | `f32`, integer-valued in `[-8, 8)` | 0 | **0.0** | U G O I S D | 12 288 |
| `w1_flip` | `M=N=K=64, TM=32, TK=16, PK=4`, grid `(4,)`, **`j` untiled** (`TN = N = 64`) | same arrays as `w1` | 0 | **0.0** | G O I S | 24 576 |
| `w1_large` | `M=N=K=256, TM=TN=TK=64, PI=PJ=4` | `A,B: bf16`, `C: f32` | 0 | — | **I S only** | 49 152 |
| `w1_l1_overflow` | `M=N=K=192, TM=TN=96, TK=32, PI=PJ=2` | `f32` | — | — | **N only** (never emitted) | 86 016 → rejected |
| `w2` | `T=4, H=W=16, U[T+1,H+2,W], PI=2, HS=8` | `f32`, integer-valued in `[-8, 8)` | 1 | **1e-5** | U G O I S D | 1 280 |
| `w2_odd` | `T=5`, otherwise as `w2` | same | 1 | **1e-5** | G O S | 1 280 |
| `w2_zero_t` | `T=0`, otherwise as `w2` | — | — | — | **N only** (never emitted) | rejected `SWAP-PARITY` |
| `w2_pi4` | `PI=4, HS=4`, otherwise as `w2` | — | — | — | **N only** (never emitted) | rejected `DMA-CHANNELS` |
| `w3` | `MQ=NR=32, PJ=4, CW=8, MATCH=2, MISMATCH=-1, GAP=1` | `i32` | 2 | **0.0** | U G O I S D | 240 |

`w2_zero_t` and `w2_pi4` are negative fixtures: neither is ever emitted, and each exists because
a catalogue code would otherwise have no test. `T = 0` is the **only** reachable `SWAP-PARITY`
condition once FR-L14 accepts an odd `T` by peeling, and `PI = 4` is the `DMA-CHANNELS` case,
whose `aircc` failure was measured (REVIEW-round1 P-R2) so that the rejection message can quote
it. `w1_l1_overflow` is the third of the same kind, for `L1-CAPACITY`.

### 4.1 Array contents and how `expected.npz` is produced

```pseudo
# tests/fixtures/make_fixture.py — the generator named in every meta.json.
# Fixed seeds, numpy.random.default_rng only (HLD §5 rule 5).

W1 / W1-flip  (seed 0)
  rng = default_rng(0)
  A = rng.integers(-8, 8, (M, K)).astype(float32)     # integer-valued floats
  B = rng.integers(-8, 8, (K, N)).astype(float32)
  C = zeros((M, N), float32)                          # committed, so oracle and device start equal
  inputs.npz   = {A, B, C}
  expected.npz = {"C": (A.astype(float64) @ B.astype(float64)).astype(float32)}
  oracle.npz   = {"C": run(gemm, A, B, C.copy())}

W1-large      (seed 0)   inputs only; no expected, no oracle — levels I and S do not execute

W2 / W2-odd   (seed 1)
  rng = default_rng(1)
  U = zeros((T + 1, H + 2, W), float32)
  U[0, 1:H+1, 1:W-1] = rng.integers(-8, 8, (H, W-2))     # interior only
  # the Dirichlet boundary stays 0 on every plane, for every t, and is never written
  inputs.npz   = {U}
  expected.npz = {"U": jacobi_reference(U)}   # an explicit two-loop numpy Jacobi, NOT the kernel
  oracle.npz   = {"U": run(jacobi, U.copy())}

W3            (seed 2)
  rng = default_rng(2)
  q = rng.integers(0, 4, MQ).astype(int32)    # a 4-letter alphabet
  r = rng.integers(0, 4, NR).astype(int32)
  S = zeros((MQ + 1, NR + 1), int32)          # the DP base row and column
  inputs.npz   = {q, r, S}
  expected.npz = {"S": sw_reference(q, r)}    # a textbook two-loop DP written with a Python
                                              # `if` STATEMENT, which the kernel may not use
  oracle.npz   = {"S": run(sw, q, r, S.copy())}
```

Two properties this buys, both of which are the point of `06-interfaces.md` §9 rule 1:

* `sw_reference` scores with a Python `if` **statement** and an explicit two-branch body. The
  kernel scores with a value-level `MATCH if q[i-1] == r[j-1] else MISMATCH`, which the grammar
  admits as one `arith.select` (FR-S3 item 8) and which the statement form is not. They are
  different expressions of the same function, so the diff actually tests something. If the
  reference were a copy of the kernel, the diff would test nothing.
* `jacobi_reference` iterates `t` outermost with an explicit two-plane copy. VF §I measured that
  a PE-outermost order **does not** compute Jacobi (maxerr 0.203) while timestep-outermost does;
  `test_W2_oracle_is_timestep_outermost` (FR-K3) asserts the kernel's own outermost loop is `t`,
  so the surface cannot express the order VF §I found broken.

### 4.2 Why `tol = 0.0` survives the device on W1 and the flip

The W1 fixtures are `f32` arrays holding **integers**. Every product is an integer with
`|p| ≤ 64`; the sum over `K = 64` terms has `|C| ≤ 4096`. Integers of magnitude `< 2²⁴` are
exactly representable in `f32` and `f32` addition of such integers is exact, so the result is
**independent of summation order**. That is the property that matters, because the flip
reassociates the reduction across four PEs and a cascade: output-stationary and weight-stationary
must produce *bit-identical* `C`, and they do. `test_W1_flip` asserts exactly that —
`array_equal(C_os, C_ws)` — which is a stronger and cheaper check than either against numpy.

W2 multiplies by `0.2`, which is not a dyadic rational, so no choice of integer inputs makes it
exact; its tolerance is `1e-5` absolute with the bound worked in the M6 LLD §3.6.

### 4.3 Fixture sizing against the 3-minute budget (NFR-3)

Every oracle is a triple or quadruple Python loop, so the cost is trip count, not bytes:

| Workload | CPython trips | Measured budget |
|---|---|---|
| W1 | `64³ = 262 144` | ~0.3 s, run once per session and cached |
| W2 | `T·(H−2)·(W−2) = 4 · 16 · 14 = 896` | negligible |
| W2-odd | `5 · 16 · 14 = 1 120` | negligible |
| W3 | `MQ·NR = 1 024` | negligible |

W1 is the only one that costs anything, and the session-scoped fixture (M7 LLD §3.3) pays it
once. `w1_large` is never executed in CPython — `256³ = 16.7 M` trips would be minutes — which is
why it ships without `oracle.npz`.

---

## 5. The demo

### 5.1 Five minutes, with a running clock

The venue publishes *"Five minutes to pitch in front of the jury, then two to three minutes of
questions"* and **no scoring rubric** (PC §8, accessed 2026-09-12). `05-work-breakdown.md` §5's
script sums to **330 s**; the version below sums to **300 s** and says what is cut first.

| At | For | Beat | On screen | Off-device or on-device |
|---|---|---|---|---|
| 0:00 | 40 s | **The claim.** A plain Python loop nest and a separate list of schedule clauses. Delete the clauses and the program runs in CPython, and that run **is** the specification | `kernels/w1_gemm.py` split down the middle: `gemm` on the left, `schedule_os` on the right | neither — it is a text editor and a CPython run |
| 0:40 | 55 s | **W1.** Run `python -m kernels.w1_gemm`. The audience reads the dataflow off the summary: `C: stationary (derived)`, `A: multicast along py (derived)`, `B: multicast along px (derived)`. Say the sentence: *the dataflow is named, not emergent — that is the ARIES criticism answered out loud* | `s.summary()` | **off-device.** No toolchain needed for this beat at all (FR-S20) |
| 1:35 | 40 s | **The flip.** Change **four schedule lines** — `grid(PK)`, `place(px=ax.k0)`, `stationary("B")`, `double_buffer("A")` — and drop `tile(ax.j, TN)`, **with no edit to the kernel**. The summary now says `B: stationary (declared)`, `B: stationary (spatial), resident for the whole run`, `A: … re-fetched per i0`, `R_space = span{e_k0}`, `3 npu_cascade channels`. Say it: *the weights stay put; `A` streams; partial sums cascade* — and the residency line is the tool saying so | a side-by-side diff of `schedule_os` and `schedule_ws`, then the new summary | **off-device** |
| 2:15 | 85 s | **The differentiator — rejection before codegen.** Run `bad_skew`. The checker prints the code, the clause, the violating dependence vector and the fix, and **no IR exists**. Then the sentence that matters: *mlir-air ships a specification for exactly this checker (`docs/AIRCorrectnessChecker.md`, properties P1–P4) and has not implemented it — `mlir/lib/Analysis/` does not exist, and `air-opt` prints its one channel diagnostic while exiting 0. The idea is upstream and public; the implementation is ours.* Show the `exit=0` terminal line | the rejection message, then a two-line terminal capture of `air-opt … ; echo $?` | **off-device**, and the `exit=0` capture is a **recorded** transcript, not a live run |
| 3:40 | 55 s | **It lowers.** W2's halo exchange: two channel bundles, put-first, per-iteration balanced, no prologue and no seeded ghost. `aircc --device npu1 --output-format=none` exits 0 with no `error:` line. Then show the `unroll = 2 : i32` the ping-pong labeller wrote on W1's K loop — *our emitter produced the shape the upstream pass requires, and the pass says so in its own IR* | the W2 plan's four-site order, then the `aircc` exit line, then the `grep unroll` line | **off-device**, on any laptop. **If D5 produced a device result, this beat becomes: the same W1 compiled to `xclbin`, run on the NPU, `max_abs_err = 0.0, mismatched = 0 of 4096`** |
| 4:35 | 25 s | **Honest limits.** §5.2, read, not skipped | the slide | — |
| 5:00 | | end | | |

**What is cut, in order, if a beat overruns:** (1) the flip (40 s — it is `05-wbs` §4's stretch
item and G5 may already have cut it); (2) the `unroll = 2` half of beat five (20 s); (3) W2's
protocol detail in beat five, leaving only the `aircc` exit line. **Never cut:** beat four. PC
§6.5 is explicit that the rejection is the stronger half — *"the flip invites 'swap the transform
script'; the rejection does not, because no neighbour has one"*.

**What is never shown:** an `air-runner` Chrome trace. The M6 LLD §3.8 records the three measured
reasons — no NPU resource model ships in the wheel, the only model in the project is a
`testdevice` with a 32 KB L1, and `air-runner` prices `linalg` bodies while FR-E2 makes ours
scalar loops. One sentence on the honest-limits slide replaces it.

**Everything in the script runs off-device.** That is deliberate and planned for (R-09): the
demo's primary artifact is the surface, the rejection, the AIR text and
`aircc --output-format=none`, all of which were **measured** to work on a machine with no NPU, no
XRT and no Vitis (VF §G.3, §G.5). A device result upgrades one beat; its absence changes one
line on the slide.

### 5.2 The honest-limits slide, verbatim

`demo/honest_limits.md`. It is read aloud, not skipped (05-wbs §5 beat 6). Everything below is
either measured or cited.

> **One lowering path, two device flags.** `air-to-aie` → MLIR-AIE, with `npu1` (Phoenix, AIE2)
> and `npu2` (Strix, AIE2P). Versal values are accepted by the *pass* but not by `air.api`'s
> `resolve_target` (`python/air/api/_trace.py:164-183`), so we do not claim them.
>
> **Compute is scalar.** Per-PE bodies are native `air.sequential` loops with element
> load/store. An `air.extern` vectorised kernel would be faster — and would *disable* the
> ping-pong passes on any buffer it touches first, because an opaque callee is not a definite
> write (`AIRDependencyScheduleOpt.cpp:1620-1630`). Future work, with that cost stated.
>
> **No performance numbers.** Optimisation passes are out of scope. We measured nothing and
> claim nothing.
>
> **`air-runner` is a timing model, not a correctness oracle.** mlir-air ships one
> (`docs/AIRRunner.md:3`); it prices `linalg` bodies and ours are scalar loops, and no NPU
> resource model ships in the wheel — the only one in the project describes a `testdevice` with
> a 32 KB L1. So we do not show a trace, rather than showing one that means less than it looks
> like it does.
>
> **Semantics off-device are established structurally, not by executing the emitted IR.**
> `air.api` has no interpreter, and the CPU backend JITs the *lowered* module, which tests the
> compiler's output rather than the user's source. Three structural checks stand in: access-region
> reconstruction, compute-node replay, and write-domain coverage. **Only the device run closes
> that loop.** *(Adjust this line to what D5 actually achieved.)*
>
> **W4 FFT is out of scope**, and the reason is not a schedule limit: the stage-parameterised
> partner map `p ↦ p ⊕ 2ˢ` is affine in neither `p` nor `s`, so this surface cannot state it
> without an escape hatch.
>
> **The checker's idea is not ours; its implementation is.** AMD specified it in their own
> repository — `docs/AIRCorrectnessChecker.md`, properties P1 (channel balance), P2 (deadlock
> freedom), P3 (resource constraints), P4 (token-constraint consistency) — and did not build it.
>
> **Nearest neighbours, named, with the delta in one sentence each.**
> * **`amd/Triton-XDNA`** — Triton SPMD → MLIR-AIR for AIE2/AIE2P, and it already ships a
>   *schedule surface*: a per-kernel `transform_aie2.mlir` selected by
>   `AIR_TRANSFORM_TILING_SCRIPT`, with memory-space promotion to L2/L1 and herd width both
>   user-settable. **Our delta is not tiling or residency; it is that the intent is bound to the
>   program's own loops and tensors, and is checked — a transform script cannot move `stationary`
>   from `C` to `B`, because nothing in it names stationarity.**
> * **Dato** (Cornell, arXiv:2509.06794) — Python-embedded task graph with `Stream[T, N, P]`
>   linear types, rejecting deadlock and inconsistent put/get by abstract interpretation over
>   the CFG. **Its safety story is stronger than ours; our delta is only the target — AIR, so
>   async tokens, ping-pong and placement come free and `npu1`/`npu2` are one flag apart — plus
>   declared reuse intent, which its stream and layout types do not carry.**
> * **AIEHalide** (accepted at PACT 2026) — Halide → MLIR-AIE, with ignorable directives and
>   bounds-inferred halos already. **So neither ignorability nor derived halos is our delta. Ours
>   is four things: AIR as the target; a plain loop nest as the surface rather than a pipeline of
>   `Func`s; a named space-time vocabulary (`skew`, `stationary`, `exchange`, `stream` — all four
>   occur zero times in Halide's `Func.h`); and a pre-codegen legality rejection, where AIEHalide
>   is explicitly open-loop: a failing schedule is replaced by the next-best, and a feedback loop
>   is its own stated future work.**
> * **IRON/ObjectFIFO** and **ARIES** — the lower-level and the higher-level ends of the same
>   stack.
>
> **Out of scope**: multi-kernel fusion, autotuning, GPUs.

**One attribution rule, binding.** The claim *"our group's Halide-to-NPU compiler is at PACT
'26"* is a **user-supplied fact** (`proposals/spatial_dsl_project_proposal.md` L9), not something
the reviews file establishes — the rebuttal in `PACT/56.txt` is signed by a different person at
the same institution (PC §6.3's provenance note). **The pitch may make that claim only if the
user confirms the attribution.** If unconfirmed, AIEHalide is named as a neighbour and nothing
more. Recorded here so it cannot be decided by accident at 4:40 on D7.

### 5.3 The rejection demo, expected text

Each of the three rejections has a byte-for-byte golden message
(`tests/golden/reject.<code>.txt`, M7 LLD §3.2), because this is the text on the screen.

```
L2-CAUSALITY: dependence (0, 1) is not carried forward by the schedule
  in clause: skew(time=(ax.i,))
  at:        kernels/rejections.py:14
  because:   Ssigma = [e_i]; for the dependence d = (0, +1, -7) between
             S[i, j-1] and S[i, j] across the tile boundary, Ssigma . d = 0,
             and causality needs Ssigma . d >= 1 lexicographically. PE j0 and
             PE j0+1 would be scheduled at the same time step while one reads
             what the other has not yet written.
  fix:       skew(time=(ax.i, ax.j0)) — include the placed axis in the time map,
             which is what makes this a wavefront rather than a race.
```

```
STATIONARITY: operand C cannot be stationary under this placement
  in clause: stationary("C")
  at:        kernels/rejections.py:27
  because:   ker M_C = span{e_k} (C is indexed by (i, j), so moving along k
             leaves C's address unchanged), and ker Spi = span{e_i1, e_j0,
             e_j1, e_k1} because place(px=ax.i0, py=ax.k0) placed i0 and k0.
             span{e_k} is not contained in ker Spi: e_k0 is a placed direction,
             so consecutive k tiles live on different PEs and C would have to
             move between them.
  fix:       place(px=ax.i0, py=ax.j0) to keep C stationary, or stationary("B")
             to keep this placement and reduce across the PE line instead.
```

```
L1-CAPACITY: the per-core working set is 86 016 bytes against a 65 536 byte budget
  in clause: double_buffer("A", "B")
  at:        kernels/rejections.py:41
  because:   acc [96, 96] f32 = 36 864
             a   [96, 32] f32 = 12 288, doubled by double_buffer -> 24 576
             b   [32, 96] f32 = 12 288, doubled by double_buffer -> 24 576
             total 86 016 > 65 536. Without double_buffer the same tiles are
             61 440 and would fit; the figure that has to fit is the ping-ponged
             one, because the pipeline duplicates every buffer it ping-pongs.
  fix:       tile(ax.k, 16) instead of 32, which halves a and b to 12 288 doubled
             (total 61 440), or drop double_buffer("A", "B").
```

The `because` line of each is the line judges read (HLD §4.2), which is why `details` carries the
numbers and the renderer, not the check, formats them.

---

## 6. Worked examples, prepared answers, and the rehearsal

### 6.1 W1 from source to artifact, as the demo runs it

```
$ python -m kernels.w1_gemm                     # no toolchain needed
gemm  target=npu1  herd logical (2, 2)  physical (1, 2)  repeats (2, 1)
  C: stationary (derived)
  A: multicast along py (derived)   channel A2L1  size [2, 1]  broadcast_shape [2, 2]
  B: multicast along px (derived)   channel B2L1  size [1, 2]  broadcast_shape [2, 2]
  reduction  R = span{e_k}   R_time = span{e_k}   R_space = {}
  L1  acc 4096 + a 2048x2 + b 2048x2 = 12 288 of 65 536
$ python -m demo.emit w1 npu1 > /tmp/w1.npu1.air.mlir
$ aircc --device npu1 --output-format none --tmpdir /tmp/ap /tmp/w1.npu1.air.mlir ; echo $?
0
$ air-opt /tmp/w1.npu1.air.mlir \
    -pass-pipeline='builtin.module(air-dependency,air-label-scf-for-to-ping-pong{device=npu1})' \
    | grep -c 'unroll = 2'
1
```

The last command is the M6 LLD §3.7 pipeline, **measured** to report `unroll = 2` on a module of
this shape. The `1` is the number in `w1.base.npu1.ir_facts.json`.

### 6.2 The flip, as the demo runs it

```
$ diff <(sed -n '/def schedule_os/,/return s/p' kernels/w1_gemm.py) \
       <(sed -n '/def schedule_ws/,/return s/p' kernels/w1_gemm.py)
<     s.grid(PI, PJ)                         >     s.grid(PK)
<     s.tile(ax.i, TM); s.tile(ax.j, TN); s.tile(ax.k, TK)
                                            >     s.tile(ax.i, TM); s.tile(ax.k, TK)
<     s.place(px=ax.i0, py=ax.j0)            >     s.place(px=ax.k0)
<     s.stationary("C")                      >     s.stationary("B")
<     s.double_buffer("A", "B")              >     s.double_buffer("A")
$ python -c "from kernels import w1_gemm as w; print(w.schedule_ws('npu1').summary())"
gemm  target=npu1  herd logical (4,)  physical (4,)  repeats (1,)
  A: stationary (derived)
  B: stationary (declared)
  C: cascade reduction along px (derived)   CascadeK size=[3] type="npu_cascade" ascending
  A: stationary (spatial), re-fetched per i0
  B: stationary (spatial), resident for the whole run
  C: cascade along px, re-fetched per i0
  reduction (tiled axes)  R_time = span{e_k1}   R_space = span{e_k0}
  L1  acc 8192 + recv 8192 + a 2048x2 + b 4096 = 24 576 of 65 536
```

Four schedule lines change, one `tile` clause goes away, no kernel edit, and the reduction
changed category. **The sentence to say over this screen: the weights stay put; `A` streams;
partial sums cascade.** Three things the summary is saying that are worth pointing at:

* **`R_time` and `R_space` are rendered over the post-tiling axes**, which is why the line says
  `R_time = span{e_k1}` rather than `R_time = {}`. `LegalMapping.r_time` is `{}` in the untiled
  frame and that is correct — the *untiled* `k` reduction is entirely spatial — but printed bare
  it reads as "no local accumulation", which is false: each PE accumulates its own `TK`-slice
  before handing it up the chain. The summary says which basis it is showing
  (`03-lld-M4-mapping.md` §3.9), and the `LegalMapping` fields stay untiled.
* On a 1-D grid `A` is **stationary** rather than multicast, because with no second PE axis
  there is nothing to fan out along. That is the summary doing its job: the reader does not have
  to re-derive it from slice arithmetic (PC §1.4 item 2).
* **The residency lines are the ones that make "weight-stationary" a fact.** `stationary` is a
  *spatial* claim — none of the operand's reuse crosses a PE — and on its own it is compatible
  with re-fetching the tile every trip, which is what `tile(ax.j, 32)` used to do to `B`. With
  `j` untiled, `B: resident for the whole run` and `A: re-fetched per i0`, and those two lines
  are what FR-K2's acceptance asserts (RULING 9, `03-lld-M4-mapping.md` §3.9). They are also why
  `double_buffer` names `A` only: a tile resident for the whole run has no streaming loop to be
  allocated inside, so `double_buffer("B")` is a `PINGPONG-SHAPE` rejection.

### 6.3 Prepared answers (2–3 minutes of Q&A)

Each is one breath. The first four are `05-work-breakdown.md` §6's, sharpened with what has since
been measured.

**"Why not just use `air.api`?"**
> Two things it does not give. First, the program does not run as its own specification — an
> `air.api` kernel is a *trace*, not an executable nest, so there is no CPython oracle to diff
> against. Second, nothing in it says *why* a transfer is shaped the way it is: `broadcast_shape
> = [2, 2]` is a fan-out, not a reason. Everything else — channels, broadcast validation,
> ping-pong, scope checking, placement — is upstream and we use it **unchanged**. Our emitter
> targets `air.api` and inherits the whole `aircc` pipeline.

**"Doesn't AIR already check put/get balance?"**
> Its compute model *states* that a violation is a compile-time error. No pass implements it. We
> fed three violations through the real `air-opt`: an unmatched put printed an error **and exited
> 0**; a per-iteration imbalance and a `get`-before-`put` cycle across two herds were silent all
> the way through `air-to-aie`. AMD wrote the specification for the missing checker —
> `docs/AIRCorrectnessChecker.md` — and `mlir/lib/Analysis/` does not exist. The idea is theirs;
> the implementation is ours.

**"How is this not Triton-XDNA?"**
> Triton-XDNA is AMD's own Triton front end for the same AIR backend, and it already ships a
> schedule surface — a per-kernel transform script, selected by `AIR_TRANSFORM_TILING_SCRIPT`,
> that sets tile sizes and promotes to L2 and L1. So tiling and residency are **not** our delta,
> and we say so on the slide. Our delta is that the intent is bound to the program's own loops
> and tensors and is checked. Concretely: no transform script can move `stationary` from `C` to
> `B`, because nothing in it names stationarity — the cascade you just saw is a consequence of
> `(σ, π)`, not of a tile size.

**"How is this not Dato, or AIEHalide?"**
> Different answers. **Dato** has a *stronger* safety story than ours — typed streams and
> abstract interpretation that rejects deadlock and inconsistent put/get — and we do not claim
> otherwise; our difference is the target, MLIR-AIR rather than MLIR-AIE, so async tokens,
> the ping-pong passes and placement come free and the two NPU generations are one flag apart,
> plus a reuse vocabulary its stream types do not carry. **AIEHalide** already has ignorable
> directives and derived halos, so neither of those is our delta either; ours is a plain loop
> nest instead of a pipeline of `Func`s, a vocabulary Halide does not have — `skew`, `wavefront`
> and `stationar` occur zero times in `Func.h` — and a rejection *with a reason, at the surface*.
> AIEHalide is explicitly open-loop: a failing schedule is replaced by the next-best, and a
> feedback loop is its own stated future work.

**"What happens without a device?"**
> Everything you have seen. The surface, the checker, the mapper and the emitter need no
> toolchain at all; `aircc --output-format=none` and `--output-format=pdi` both compile through
> all twenty-two AIE stages with no XRT and no hardware — we measured that on a laptop. The only
> thing a device adds is the last container step, `xclbinutil`, which comes from XRT, and
> actually executing the kernel. *(Then, from D5's actual result:)* We **did** run W1 on silicon
> and the diff against the CPython oracle was exact — `0 mismatched of 4096`. **or** We have
> **not** run it on silicon, the end-to-end claim stops at `aircc`, and the slide says so.

**"Performance?"**
> We measured none and claim none. Optimisation passes are out of scope by design, and the
> compute bodies are scalar. mlir-air ships a timing simulator, `air-runner`; it is not a
> correctness oracle, it prices `linalg` bodies and ours are loops, and no NPU resource model
> ships with the wheel — so we are not showing you a trace either.

**"Is the halo protocol safe?"** *(the question a spatial-hardware person asks)*
> Yes, and it was checked four ways, one of them measured: every producer lock `air-to-aie`
> emits starts at ≥ 1, never 0; the lock allocator does that by construction; the ObjectFIFO path
> acquires an *empty* buffer rather than rendezvousing; and upstream ships a hardware-CI ring of
> exactly this protocol in `programming_examples/channel_examples/worker_to_worker/`.

### 6.4 The D7 rehearsal checklist

D7 is rehearsal only — the freeze is the **end of D6** (D-10), because the evaluation is listed
19–20 September and a freeze on the evaluation day is not a freeze.

**Before the first rehearsal (30 minutes):**

1. `sha256sum -c vendor/wheels/SHA256SUMS` on the demo machine, then reinstall from
   `--no-index --find-links vendor/wheels` into a **fresh** venv. The demo runs from that venv
   and nothing else. (R-13.)
2. `pytest` — green, under 180 s. Record the number.
3. `pytest -m "slow"` — green. This is the `aircc` evidence for beat five.
4. `pytest -ra` and **read the skip list aloud**. That list *is* the honest-limits slide
   (04-test-plan §8 item 10); any skip whose reason does not fit on the slide is a bug in the
   reason, not in the slide.
5. `git status --porcelain` — empty. A golden rewritten by a stray `--update-goldens` is caught
   here if CI did not catch it.
6. Capture the two terminal transcripts the pitch replays — the `air-opt … ; echo $?` exit-0
   capture, and the `aircc --output-format=none ; echo $?` capture — into `demo/captures/`.
   **They are replayed, not run live**: a live `aircc` takes minutes and beat five has 55 s.

**Rehearsal, three times (05-wbs: #1 on D6, #2 and #3 on D7):**

7. Run the full script with a visible clock. Record the elapsed time at each beat boundary
   against §5.1's table. Any beat over by more than 10 s: cut in §5.1's stated order, do not
   speed up.
8. Rehearse **beat four alone**, twice more than the others. It is the one that cannot be cut and
   the one that carries the sentence about `docs/AIRCorrectnessChecker.md`.
9. Read the honest-limits slide out loud, in full, every time. It is 25 s. A slide that is only
   ever skimmed in rehearsal gets skipped on the day.

**Q&A drill (05-wbs §6, plus §6.3 above):**

10. Two people ask, one answers; swap. Time each answer — **none may exceed 25 s**, because there
    are 2–3 minutes for all of them.
11. Drill the three answers that concede something: Dato's safety story is stronger; AIEHalide
    already has ignorability and derived halos; Triton-XDNA already ships a schedule surface.
    Conceding cleanly and then naming the delta is stronger than being caught not knowing, and
    the red team has already found all three (PC §6.3).
12. Fix the device answer to **what D5 actually produced**, in writing, before rehearsal #2.
    There are two versions of that answer and picking the wrong one on stage is unrecoverable.
13. Confirm or drop the PACT attribution (§5.2's binding rule). If the user has not confirmed by
    rehearsal #2, the sentence is deleted from the deck, not left in "just in case".

**Ten minutes before:**

14. Fresh terminal, `python -m kernels.w1_gemm` — proves the venv, the import path and the
    summary in one command, and it needs no toolchain, so it cannot fail for an environment
    reason that the rest of the demo would then also hit.
15. Font size, and the terminal width set so the `because:` block of a rejection does not wrap.
    Those four lines are the pitch.

---

## 7. Tests table

| Test id | Level | Mark | FR | Asserts | Owner |
|---|---|---|---|---|---|
| `test_W1_end_to_end` | G, O, S | — / `slow` | FR-K1 | oracle matches numpy exactly; module, summary and plan match goldens for both targets; `aircc --output-format=none` exits 0 with no `error:` line | C |
| `test_W1_flip` | G, S | — / `slow` | FR-K2 | plan has `R_space = span{e_k0}`, `3` cascade channels, `channel_type = "npu_cascade"` three times and **no** `broadcast_shape` on them; and `array_equal(C_os, C_ws)` — the two dataflows agree bit for bit (§4.2) | C |
| `test_W2_end_to_end` | G, O, S | — / `slow` | FR-K3 | oracle within `tol = 1e-5` of the independent two-loop reference; goldens; `aircc` clean | C |
| `test_W2_oracle_is_timestep_outermost` | U | — | FR-K3 | `KernelModel.axes[0].name == "t"` — the surface cannot express the PE-outermost order VF §I measured wrong | C |
| `test_W2_odd_T` | G, O, S | — / `slow` | FR-K3, D-4 override | `w2_odd` checks clean, emits `air.sequential(0, 4, 2)` plus one peeled body, and diffs within tolerance | C |
| `test_W3_end_to_end` | G, O, S | — / `slow` | FR-K4 | oracle exactly matches the textbook DP; goldens; `aircc` clean | C |
| `test_K5_scope_documented` | U | — | **FR-K5** | `demo/honest_limits.md` names W4 and the reason; `00-README.md` §3 has the out-of-scope row; `kernels/` exports no `w4`/`fft` name | C |
| `test_grammar_accepts` | U | — | FR-S3 | the four kernel sources parse, field by field against the stored `KernelModel`s | A |
| `test_demo_rejections` | N | — | FR-L2, FR-L3, FR-L9 | each of the three raises its code, and `str(e)` matches `tests/golden/reject.<code>.txt` byte for byte | C |
| `test_fixture_contract` | U | — | 06-interfaces §9 | the per-fixture contract of the M7 LLD §3.3, parametrised over all seven directories | C |
| `test_kernel_scripts_run` | U | — | NFR-5 | each `kernels/*.py` `main()` exits 0 and prints a non-empty summary, in a process where `air` is absent from `sys.modules` afterwards | C |
| `test_ignorability[W1,W2,W3]` | O | — | FR-S18 | the property form of the M7 LLD §3.4 | C |
| `test_sem_access_regions`, `test_sem_compute_nodes`, `test_sem_coverage` | O | — | D-9 | the three structural checks of 04-test-plan §3.4 | C |

---

## 8. Dependencies

### 8.1 On other people

| From | What | When | If it slips |
|---|---|---|---|
| **A** | the FR-S3 grammar decision on captured module-level constants (`GAP`) | D0 | the kernel inlines `1` |
| **A** | the `σ` semantics of `skew` (§3.5) | **D4** | `bad_stationary` becomes the headline rejection |
| **A** | the D-4 override's effect on FR-L14 / `test_L14_swap_parity` (§3.3) | D5 | `w2_odd` ships with `"expect": "reject"` instead, and the demo does not mention peeling |
| **B** | whether ping-pong fires on the flip's `i0` loop (`A` only — `j` is untiled, so there is no `j0` loop, and `B` is resident) | D6 | drop `double_buffer("A")` from `schedule_ws`; `PINGPONG-SHAPE` says so at check time, and L1 falls to `22 528` (`8192 + 8192 + 2048 + 4096`) |
| **B** | the W1-flip cascade plan on a 1-D herd | D6 | G5 cuts the flip; the pitch loses beat three |
| **C** (self, M6) | `artifact`, `ir_facts`, `has_device` | D2 | levels S and I skip with a printed reason |

### 8.2 Files in the clone

| Concern | `path:line` |
|---|---|
| physical herd caps and the strip-mining rule that forces the 1-D flip | `python/air/api/_trace.py:88-91`; `_resolve_physical` at `:1347-1373`; `repeats` at `:1297` |
| "a cascade is a physical link between neighbouring cores"; `at=` pinning | `python/air/api/_trace.py:1771-1778` |
| `npu_cascade` is implemented, and rejects `broadcast_shape` | `python/air/api/_channel.py:547`, `:170-177` |
| the worked head/middle/tail cascade | `programming_examples/cascade_reduction/cascade_reduction.py` |
| a written tensor becomes an output and drops out of the input list | `python/air/api/ops.py:310`; `python/air/api/_compile.py:98-103`, `:318-347` |
| `L1_BYTES = 65536` — the budget every fixture is sized against | `python/air/api/_trace.py:100` |
| an opaque callee disqualifies ping-pong (the honest-limits `air.extern` line) | `mlir/lib/Transform/AIRDependencyScheduleOpt.cpp:1620-1630` |
| the hardware-CI ring that answers the halo question | `programming_examples/channel_examples/worker_to_worker/run_makefile_peano.lit:4-9` |
| the unimplemented checker the pitch names | `docs/AIRCorrectnessChecker.md:15-20` |
| `air-opt` prints an error and exits 0 — beat four's terminal capture | `mlir/lib/Util/Dependency.cpp:2063-2066` |

### 8.3 Third-party

`numpy` only, for the fixture generator and the references. No plotting library, no deck
framework: the honest-limits slide is Markdown so that `test_K5_scope_documented` and
`test_trace_disclaimer` can grep it.

---

## 9. Implementation order, effort, definition of done

Effort figures are **estimates**, consistent with `05-work-breakdown.md` §7's 1.0 pd for M8.
Note that `05-work-breakdown.md` §2 assigns the three kernel sources to **Person A at D0** while
`00-README.md` §2 assigns M8 to C. The split that works: **A writes the sources at D0 against
FR-S3 and owns their grammar-legality; C owns the schedules, the fixtures, the demo and
everything after.** The sources in §3 are written here so A has a target rather than a brief.

| Step | Day | Est. | Output | Done when |
|---|---|---|---|---|
| 1. the four kernel sources + the throwaway `ast.parse` smoke script | D0 | 0.15 pd | `kernels/*.py` bodies | all four parse; every construct maps to an FR-S3 clause (§3's tables) |
| 2. `make_fixture.py` + `w1`, `w2`, `w3` + the three independent references | D1 | 0.3 pd | `tests/fixtures/w{1,2,3}/` | `test_fixture_contract` green; `expected` and `oracle` agree |
| 3. `w1_large`, `w2_odd`, `w1_l1_overflow` | D3 | 0.1 pd | — | the large one feeds `test_I_pingpong_labels` |
| 4. the schedules and `main()` for each kernel | D2 | 0.1 pd | `schedule_*` functions | `python -m kernels.w1_gemm` prints the summary with no toolchain |
| 5. `rejections.py` + the three golden messages | D4–D5 | 0.15 pd | `tests/golden/reject.*.txt` | `test_demo_rejections` green |
| 6. `demo/honest_limits.md` | D6 | 0.1 pd | the slide | `test_K5_scope_documented`, `test_trace_disclaimer` green |
| 7. `demo/run_demo.py`, the captures, rehearsal #1 | D6 | 0.1 pd | the deck and the script | the script runs end to end in ≤ 300 s on a fresh venv |

**Definition of done for M8:**

1. All four kernel sources parse under M1 and their `KernelModel`s match the stored fixtures.
2. The three workloads' oracle diffs pass at their stated tolerances (04-test-plan §8 item 5).
3. `test_W1_flip` passes, including `array_equal(C_os, C_ws)` — or G5 cut the flip and the deck,
   the slide and this document's §5.1 all say so in the same words.
4. The three rejection messages match their goldens byte for byte and each fits in the terminal
   without wrapping.
5. The demo script runs in ≤ 300 s, three times, on the frozen venv, with the clock recorded.
6. Every number on the honest-limits slide is either measured or carries a `path:line`.
7. The PACT attribution is confirmed in writing or the sentence is deleted (§5.2).

---

## 10. Open questions owned, and their resolutions

| # | Question | Verdict |
|---|---|---|
| **(new) Q-C9** | the W1-flip grid: `04-test-plan.md` §4 *said* `(4, 2)` and `02-hld.md` §7.1 *said* `place(px=ax.i0, py=ax.k0)`; both are now RULING 4's 1-D form | **RESOLVED — 1-D `grid(PK)` with `place(px=ax.k0)` (§3.1).** The HLD's placement cannot satisfy `stationary("B")` at all, and a 2-D grid strip-mines the cascade axis on **both** targets (`_trace.py:88-91`). 1-D resolves to physical `(4,)` with repeats `(1,)` on both. **Adopted as RULING 4**; `04-test-plan.md` §4, `02-hld.md` §7.1, FR-K2, M4 §6.2, M5 §6.2 and `test_M6_cascade_chain` all carry it, and the chain is **ascending** (measured P-R3) |
| **(new) Q-C10** | W2's `tile(ax.i, HS)` does not divide the interior extent at `H = 16` | **RESOLVED — RULING 1 (§3.3).** `U: sp.f32[T+1, H+2, W]` with `H = W = 16` **interior**, `PI = 2`, `HS = 8`: `PI·HS == H` exactly and the `i` extent is 16. `H = 18` is **not** adopted, because it keeps `PI = 4`, which `aie.connect` rejects (P-R2). L1 = **1 280 B** everywhere |
| **(new) Q-C11** | W3's `sub(q, r)` is a call the grammar forbids | **RESOLVED — RULING 2 (§3.4):** a value-level conditional `MATCH if q[i-1] == r[j-1] else MISMATCH`, admitted as FR-S3 item 8 and lowered to one `arith.select`. The arithmetic dodge `2 - 3*min(d*d, 1)` is deleted; a table lookup stays rejected as a data-dependent subscript |
| **(new) Q-C12** | W3's L1 figure omits `q` and `r` | **RESOLVED — 240 B, not 80 B (§3.4).** `02-hld.md` §7, `04-test-plan.md` §4 and `03-lld-M3-checker.md` §6.4 (232 B at M3's scope) all carry it |
| **(new) Q-C13** | FR-L9's `L1-CAPACITY` example does not demonstrate the doubling it exists to demonstrate | **RESOLVED — one fixture (§3.5).** FR-L9's acceptance now names `M=N=K=192, TM=TN=96, TK=32, f32` (61 440 undoubled, 86 016 charged, 20 480 over), so `w1_l1_overflow` serves both `test_L9_capacity` and the slide |
| **(new) Q-C14** | `w1_large` in `f32` exceeds L1 | **RESOLVED — `bf16` inputs, `f32` accumulator (§3.2)**, which is also VF §E.5's measured shape. 49 152 B of 65 536. No numerical tolerance is needed because the fixture is never executed |
| **(closed) Q-C15** | does `skew(time=…)` replace `σ` outright or prefix a default `σ`? | **CLOSED by RULING 6.** Neither, exactly: `skew` defines `σ`'s **leading row**, and the remaining rows are the loop order minus the **placed and skewed** axes. On W3 that gives `Sσ = [e_i; e_j1]` and `Sσ·(0,1,−7) = (0,−7)` — the headline rejection survives. Written into FR-S17, FR-L2 and M3 §3.3 |
| **(closed) Q-C16** | does the D-4 override make `w2_odd` accept-and-peel, retiring `test_L14_swap_parity`? | **CLOSED.** `T = 5` is accepted and peeled; `test_L14_swap_parity` keeps its name and asserts the `T = 0` rejection instead, for which M8 ships the `w2_zero_t` fixture (§4). Without it `SWAP-PARITY` is unreachable and `test_D3_catalogue_complete` fails |
| **(open) Q-C17** | is the PACT 2026 attribution confirmed by the user? | **Owner C, due rehearsal #2 (§5.2).** Unconfirmed ⇒ the sentence is deleted from the deck, not hedged |
