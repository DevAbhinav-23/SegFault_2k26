# High-level design — Spatial DSL

*Phase 1, 2026-09-12. Reads on top of [`01-requirements.md`](01-requirements.md); the frozen
contracts are in [`06-interfaces.md`](06-interfaces.md). Citation convention as in
`01-requirements.md` §0.*

---

## 1. Context

```
                 ┌───────────────────────────────────────────────────────────┐
   user  ─────▶  │  kernel.py    @sp.kernel  def gemm(A, B, C): <loop nest>   │
                 │  schedule.py  s = sp.schedule(gemm, target="npu1"); s.…()  │
                 └───────────────┬───────────────────────────┬───────────────┘
                                 │                           │
              (delete every s.*) │                           │ s.check() / s.mlir() / s.build()
                                 ▼                           ▼
                       ┌──────────────────┐        ┌──────────────────────────┐
                       │ CPython ORACLE   │        │  our compiler (M1…M5)    │
                       │ numpy arrays in, │        └────────────┬─────────────┘
                       │ arrays out       │                     │ AIR MLIR text
                       └────────┬─────────┘                     ▼
                                │                    ┌──────────────────────┐
                                │                    │ mlir-air (upstream)  │
                                │                    │ air.api → aircc →    │
                                │                    │ air-to-aie → mlir-aie│
                                │                    └──────────┬───────────┘
                                │                               │ .pdi / none / .xclbin
                                │                               ▼
                                │                      ┌────────────────────┐
                                └─── diff (M6) ───────▶│  XRT + NPU device  │
                                                       │  (optional)        │
                                                       └────────────────────┘
```

**Everything to the right of "our compiler" is upstream and unmodified.** The entry is the
box in the middle plus the oracle property on the left. The pipeline inside `aircc` is the
stock C++ one (`tools/aircc/aircc.cpp:905-995`, VF §D.9); we neither add nor reorder a pass.

---

## 2. Modules

| ID | Name | Owner | One-line responsibility |
|---|---|---|---|
| **M0** | `model` — shared contract | all three (frozen) | the dataclasses and error types every other module speaks |
| **M1** | Frontend / kernel capture | A | Python AST → `KernelModel`; owns the accepted grammar |
| **M2** | Schedule builder / schedule IR | A | clause API with argument validation → `ScheduleModel`; pure data |
| **M3** | Legality checker | A | `(KernelModel, ScheduleModel)` → `LegalMapping` or a structured rejection |
| **M4** | Mapping derivation / protocol synthesis | B | `LegalMapping` → `MappingPlan` (buffers, channels, protocols) + self-check |
| **M5** | AIR emitter | B | `MappingPlan` → `air.api` calls → AIR MLIR text |
| **M6** | Toolchain & runtime driver | C | install, `build`, `aircc`, XRT run, oracle diff, `air-runner` |
| **M7** | Tests & CI harness | C | pytest layout, fixtures, goldens, marks, CI |
| **M8** | Kernels, fixtures, demo | C | W1/W2/W3 sources + schedules, the pitch, the honest-limits slide |

### M0 — `model` (the shared contract)

*Responsible for*: the dataclass definitions, the error hierarchy, the `Diagnostic` schema, and
nothing else. Frozen dataclasses, no behaviour beyond validation of invariants in
`__post_init__`.
*Not responsible for*: any parsing, any checking, any emission. It imports nothing from M1–M8
and nothing from `air`.
*Why it is its own module*: three people write against it in parallel from D1; a field rename
mid-week costs two of the three a rebase (R-15).

### M1 — Frontend / kernel capture

*Responsible for*: `inspect.getsource` of the decorated function, `ast.parse`, validation
against the grammar of FR-S3, construction of the iteration domain, per-operand access maps
`F_a` (as integer matrices plus constant offsets), statement list, the reduction projection `f`
when an augmented assignment is present, and the uniform dependence vectors. Also: the callable
passthrough that makes the decorated function its own oracle (FR-S1).
*Not responsible for*: any notion of PEs, memory levels, channels or AIR. M1 never imports
`air`. It does not decide whether a schedule is legal; it only says what the kernel *is*.
*Key invariant*: `KernelModel` is a pure function of the source text.

### M2 — Schedule builder / schedule IR

*Responsible for*: the clause methods, their argument-domain validation (FR-S19), axis handles
including the handles `tile` creates, and the accumulation of clauses into a `ScheduleModel`
with a deterministic ordering.
*Not responsible for*: any legality reasoning — `place` records rows of `Sπ`, it does not check
them; `stationary` records an assertion, it does not verify it. No lowering (FR-S20): building a
schedule does not import `air`.

### M3 — Legality checker

*Responsible for*: `Sσ`/`Sπ` construction from the `ScheduleModel`, FR-L1…L14 as small integer
linear algebra over numpy, the L1 capacity estimate, and the production of either a
`LegalMapping` (which carries the maps, the reuse classification inputs, the reduction split and
the resolved physical herd shape) or a `LegalityError` carrying a `Diagnostic`.
*Not responsible for*: choosing `(σ,π)` — the user supplies both (PC §1.3). Not responsible for
channels, protocols or buffers; those are M4's. Not responsible for the *emitted* balance and
acyclicity — that self-check reads a `MappingPlan`, which does not exist yet at M3 time, and
lives in M4.
*Hot spot*: this module is the entry's differentiator (G2). Every check's message is judged.

### M4 — Mapping derivation / protocol synthesis

*Responsible for*: the reuse trichotomy per operand; the buffer plan (level, `air.api` scope
spelling, shape, dtype, bytes, nesting depth); the channel plan (name, `size`,
`broadcast_shape`, `channel_type`, ordered put/get sites with regions, scope and async flag);
the three protocols (halo put-first, wavefront forward with source/drain, cascade chain); and
the **self-check** of balance and acyclicity against AIR's own P1/P2 definitions before the plan
is handed on.
*Not responsible for*: any `air.api` call. M4 produces data; M5 translates it. This split is
what makes the protocols testable without a toolchain.
*Hot spot*: everything AIR does not check is checked here (R-06).

### M5 — AIR emitter

*Responsible for*: a **mechanical** translation of `MappingPlan` into `air.api` calls, in plan
order, plus `launch.build(target)` and `str(module)`. Owns the `air.api` integration and the
golden snapshots.
*Not responsible for*: any decision. If the emitter has to choose something, that choice belongs
in M4 and the plan is under-specified. This is the rule that keeps the goldens meaningful.

### M6 — Toolchain & runtime driver

*Responsible for*: the pinned install recipe and its verification; running `aircc` with the
right flags per output format; parsing `air-opt`/`aircc` **stderr** rather than exit codes
(FR-T5); the XRT run path; the oracle diff; `air-runner` invocation for a trace when asked.
*Not responsible for*: correctness claims. `air-runner` is a timing model, not an oracle
(VF §S7), and the driver's output must say so.

### M7 — Tests & CI harness

*Responsible for*: the pytest layout, marks (`slow`, `requires_device`, `requires_aircc`),
golden-file conventions and the single `--update-goldens` switch, the negative-test corpus, and
CI on CPU with the pinned wheel.
*Not responsible for*: writing the per-module unit tests — each owner writes their own; M7
provides the harness they plug into.

### M8 — Kernels, fixtures, demo

*Responsible for*: the three kernel sources and their schedules, the fixture generator with
fixed seeds, the runnable demo scripts that print the mapping summary, the pitch script, and the
honest-limits slide.
*Not responsible for*: the library. If a kernel needs a library change, it is an FR change.

---

## 3. Data flow

```
source text
   │  M1 (ast)                       ┌─ oracle path: the decorated function, called directly
   ▼                                 │
KernelModel ──────────────────┐      │
                              │      │
clause calls                  │      │
   │  M2                      │      │
   ▼                          │      │
ScheduleModel ────────────────┤      │
                              ▼      │
                         M3 legality │
                              │      │
                  LegalityError  or  LegalMapping
                                     │
                                     ▼  M4 mapping + protocol synthesis
                                MappingPlan ──▶ M4 self-check (balance P1′, acyclicity P2b)
                                     │                    │
                                     │              MappingError
                                     ▼  M5
                                 air.api calls ──▶ launch.build(target) ──▶ Module ──▶ str()
                                     │
                                     ▼  AIR MLIR text  (+ MappingSummary)
                                     │  M6
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
        aircc --output-format=none   pdi          xclbin (needs XRT)
                                                      │
                                                      ▼  XRT run
                                                  device result ──▶ diff vs oracle
```

Every arrow is a pure function except the three that shell out. Nothing in the left column
touches the network or the filesystem.

---

## 4. Error model

### 4.1 Hierarchy

| Class | Raised by | Meaning |
|---|---|---|
| `SpatialError` | — | base; carries a `Diagnostic`; never raised directly |
| `GrammarError` | M1 | the kernel body is outside the accepted subset (FR-S3) |
| `ClauseError` | M2 | a clause argument is outside its domain (FR-S19) |
| `LegalityError` | M3 | the `(σ,π)` map or a declared property fails a check (FR-L1…L14) |
| `MappingError` | M4 | the plan's self-check failed, or a protocol cannot be synthesised |
| `EmissionError` | M5 | `air.api` rejected a construct, or `module.operation.verify()` failed |
| `ToolchainError` | M6 | `aircc`/`air-opt`/XRT failed, or the wheel pin does not match |

Six leaf classes, one base. The specificity lives in `Diagnostic.code`, not in the class tree —
so a new check adds a code and a test, not a class (R-15).

### 4.2 Rendering

Every `SpatialError.__str__` renders the `Diagnostic` in a fixed four-part shape:

```
<code>: <one-line reason>
  in clause: <clause text as the user wrote it, or the source line>
  at:        <file>:<line>            (when a source location exists)
  because:   <the check, with the actual numbers>
  fix:       <one concrete clause edit>
```

The `because` line carries the identity that failed with its operands substituted — e.g.
`ker M_C = span{e_k}, ker Sπ = span{e_j}, and span{e_k} ⊄ span{e_j}`. This is the line judges
read; it is the reason FR-D1 requires `details` to be a mapping of the concrete numbers rather
than pre-formatted prose.

### 4.3 What is never raised to the user

A raw MLIR diagnostic, a numpy exception, a `KeyError` from an internal table, or a traceback
through `air.api`. M5 catches `air.api`'s own exceptions and re-raises them as `EmissionError`
with the original text in `details["air_api_message"]` (NFR-4). `air.api`'s messages are
generally good — `_compile.py` already annotates an L1 overflow in terms of tile size — so the
original text is preserved, not discarded.

---

## 5. Determinism rules

NFR-1 is met by five rules, each testable:

1. **No unordered iteration in any output path.** Every dict/set traversal that affects names,
   ordering or text is `sorted(...)` on an explicit key.
2. **Names are derived, not counted.** A channel's symbol name is
   `f"{operand}_{role}"` (e.g. `A2L1`, `ToNorth`, `CascadeK`), a buffer's name is
   `f"{operand}_{level}"`. Where an index is needed it is the plan-order index, which rule 1
   makes deterministic.
3. **Locations are unknown.** `air.api` builds inside `with Context(), Location.unknown():`
   (`_compile.py:127`), so no absolute path enters the emitted text and goldens are portable.
4. **Literals are formatted from the source token**, not from a float round-trip.
5. **No wall-clock, no `id()`, no `uuid`, no RNG** anywhere in M1–M5. The fixture generator
   (M8) uses `numpy.random.default_rng(seed)` with a fixed seed, and the seed is part of the
   fixture file name.

---

## 6. Hot spots

Ranked by the cost of getting them wrong, each with the fact that governs it.

| # | Hot spot | Governing fact | Consequence of getting it wrong |
|---|---|---|---|
| H-1 | The ping-pong loop shape in the emitter | `isPingPongCandidate`'s eight conditions — `AIRDependencyScheduleOpt.cpp:1604`, `:1620-1630`, `:1690`; `Transform/Passes.td:970-971`; measured in VF §E.5 | `double_buffer` silently does nothing; no diagnostic anywhere |
| H-2 | Put/get balance and acyclicity | `docs/AIRCorrectnessChecker.md:15-20` is a **plan**; `mlir/lib/Analysis/` does not exist; `air-opt` prints and exits 0 (VF §B.4, §B.6, §S11); `AIRToAIEPass.cpp:4484-4491` silently repairs L2 imbalance (VF §S4) | a wrong program compiles and hangs or produces garbage on device, with nothing upstream to say so |
| H-3 | Channel bundle indices | `ChannelPutOp::verify()` — *"channel bundle indices must not be temporal `scf.for` induction variables"* (`AIRDialect.cpp:3589-3592`) | the module fails to verify at `build()`; cheap to hit, cheap to fix, but only if the emitter knows the rule |
| H-4 | Broadcast channel shape | `ChannelOp::verify()` rank/multiple rules (`AIRDialect.cpp:3777-3806`) and `air.api`'s own validator (`_channel.py:120-145`) | a fan-out that is off by one destination; caught by upstream, but with an AIR-level message |
| H-5 | Per-core discrimination | a Python `if` on a herd coordinate raises because `bool()` on a `Condition` is refused (`_cond.py:41-45`); `ops.branch` is the construct (`_cond.py:18-28`) and `air-to-aie` folds it once the coordinate is literal (`_cond.py:49-54`) | the emitter picks one branch for every core, silently |
| H-6 | L1 budget with ping-pong | `L1_BYTES = 65536` (`_trace.py:100`); `_compile.py`'s `_annotate_l1_failure` — the figure that must fit is the ping-ponged one | a design that passes our check and fails in placement |
| H-7 | Exit codes | `air-opt` prints `error:` and exits 0 (VF §B.6, §S11), from `Dependency.cpp:2063-2066` calling `emitOpError` without `signalPassFailure()` | our driver reports success on a broken module |
| H-8 | `air-broadcast-detection` firing unasked | it walks `air::DmaMemcpyNdOp` only (`AIRDependencyScheduleOpt.cpp:3328`), needs an enclosing herd, an L1 endpoint and constant herd sizes (`:3338`, `:3386-3402`); it fired on the VF §E.5 GEMM unprompted | a multicast we did not plan appears (or does not), and our golden changes underneath us |
| H-9 | Segment scope for L3 endpoints | the **implemented** check requires an `air.launch` for any L3 endpoint and a segment only when the endpoint is inside a herd body; the module docstring is stricter — cite the code (PC §1.1 fact 4) | an avoidable `air.api` rejection, or an unnecessary segment |
| H-10 | Herd rank and strip-mining | 1-D/2-D only (`_trace.py:1288-1291`); physical caps `npu1 {1:(4,), 2:(1,4)}`, `npu2 {1:(8,), 2:(2,4)}` (`_trace.py:88-91`); `shape=` must divide exactly (`_trace.py:1355-1372`); the default is the largest divisor ≤ cap (`_resolve_physical`) | a grid that cannot be placed, discovered at `build()` rather than at `check()` |

---

## 7. The three kernels through the pipeline

This is the "no surprises" section: what each module produces, concretely, for each workload.
Fixture sizes are the CPU-oracle sizes from the test plan.

**The per-core L1 arithmetic, computed once.** Every other document quotes these four numbers and
none recomputes them. The budget is `L1_BYTES = 65536` (`_trace.py:100`); a `double_buffer`
operand is charged twice (FR-L9).

| Workload | Buffers (shape × dtype) | Arithmetic | Total |
|---|---|---|---|
| **W1** GEMM OS | `acc [32,32] f32`; `a [32,16] f32` ×2; `b [16,32] f32` ×2 | `4096 + 2·2048 + 2·2048` | **12 288 B** |
| **W1-flip** WS | `acc [32,64] f32`; `recv [32,64] f32`; `a [32,16] f32` **×2** (`double_buffer("A")`); `b [16,64] f32` | `8192 + 8192 + 2·2048 + 4096` | **24 576 B** |
| **W2** Jacobi | `cur`, `next`, each a strip of `HS` rows plus 2 ghost rows: `[HS+2, W] f32` = `[10, 16] f32` | `2 × (8+2) × 16 × 4` | **1 280 B** |
| **W3** SW | `q [MQ] i32` (whole, broadcast); `r [CW] i32` (this PE's slice); `prev`, `cur`, each `[CW+1] i32`; `edge_in`, `edge_out`, each `[1] i32` | `128 + 32 + 2·36 + 2·4` | **240 B** |

W2's two strips are the explicit `cur`/`next` pair the swap protocol uses (D-5); there is no
third staging strip, because `UIn` gets into `cur` and `UOut` puts out of `next`. W3's figure at
M3's scope is **232 B** — M3 does not yet know about `edge_in`/`edge_out`, which M4 adds
(`03-lld-M3-checker.md` §6.4); the **W1-flip's** figure at M3's scope is **16 384 B**, for the
same reason — `recv` is M4's (`03-lld-M3-checker.md` §6.2). The flip's tiles span the whole `N`
because `j` is untiled, which is what makes `B` resident for the whole run (§7.1, RULING 9), and
it is why the flip costs twice W1's L1. No fixture is anywhere near the 65 536 budget; the
only workload that exercises `L1-CAPACITY` is the deliberate `M=N=K=192` negative of FR-L9.

### 7.1 W1 — GEMM, output-stationary

**Source and schedule** (the surface text; the schedule lines are the only ones deleted for the
oracle):

```
@sp.kernel
def gemm(A: sp.f32[M, K], B: sp.f32[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]

s = sp.schedule(gemm, target="npu1"); ax = s.axes()
s.grid(PI, PJ)
s.tile(ax.i, TM); s.tile(ax.j, TN); s.tile(ax.k, TK)
s.reduce(ax.k, op="+")
s.place(px=ax.i0, py=ax.j0)
s.stationary("C")
s.reside(A="L1", B="L1", C="L1")
s.double_buffer("A", "B")
s.pipeline(ax.k0)
```

Fixture: `M=N=K=64`, `TM=TN=32`, `TK=16`, `PI=PJ=2`, `f32`.

**M1** → `KernelModel`: domain `[0,64)³` over `(i,j,k)`; `F_A=(i,k)`, `F_B=(k,j)`, `F_C=(i,j)`;
one statement, an augmented assignment, so `f : (i,j,k) ↦ (i,j)`; dependence vector `(0,0,1)`
on `C`.

**M2** → `ScheduleModel`: grid `(2,2)`; tiles producing `i0,i1,j0,j1,k0,k1`; `Sπ` rows
`[e_i0; e_j0]`; `σ` default `(i0, j0, k0, i1, j1, k1)`; `reduce(k, "+")`; `stationary("C")`;
`double_buffer{A, B}`.

**M3** → `LegalMapping`: `ker Sπ ⊇ span{e_k0}`; `R = ker Sf = span{e_k}`; `R_time = R`,
`R_space = {}`; `stationary(C)` holds since `ker M_C = span{e_k} ⊆ ker Sπ`; (L1) holds;
(L2) `Sσ·(0,0,1) = 1 ≥ 1`; physical herd on `npu1` = `(1,2)` with repeats `(2,1)`, on `npu2` =
`(2,2)` with repeats `(1,1)`; L1 = `acc 32·32·4 = 4096` + `2×(32·16·4)` + `2×(16·32·4)` =
**12 288 B** of 65 536.

**M4** → `MappingPlan`:

| item | value |
|---|---|
| herd | `[range(2), range(2)]`, name `gemm_herd`, 2-D |
| buffers | `acc [32,32] f32 h.private()` depth 0; `a [32,16] f32 h.private()` depth 1; `b [16,32] f32 h.private()` depth 1 |
| channel `A2L1` | `size=[2,1]`, `broadcast_shape=[2,2]` — A does not index `j`, so it multicasts along the PE **row** |
| channel `B2L1` | `size=[1,2]`, `broadcast_shape=[2,2]` — B does not index `i`, so it multicasts along the PE **column** |
| channel `C2L3` | `size=[2,2]`, no broadcast |
| put sites | `A2L1.put(A[pi*32:(pi+1)*32, kk:kk+16], indices=[pi,0])` at **segment** scope, under a **Python** loop over `pi ∈ {0,1}` (the bundle index must be a trace-time constant) wrapping an **`air.sequential(0,64,16)`** over `kk`; likewise for `B2L1` at `indices=[0,pj]`; `C2L3.put(acc, indices=[tx,ty])` once per PE at **herd** scope |
| get sites | `A2L1.get(a, indices=[tx,ty])` and `B2L1.get(b, indices=[tx,ty])`, once per iteration of the K loop, at **herd** scope; `C2L3.get(C[…], indices=[i,j])` for `i,j ∈ {0,1}` at **segment** scope after the herd |
| loop shape | `air.sequential(0, 64, 16)` over `kk` in the herd body, with the two `air.alloc`s as **direct children** and each first touched by its `get` |
| delivery summary | `C: stationary (derived)`, `A: multicast along py (derived)`, `B: multicast along px (derived)` |

Balance self-check, per key: `A2L1[0,0]` 4 puts; fan-out `{[0,0],[0,1]}` each gets 4 → balanced
under the broadcast rule (`06-interfaces.md` §5.6 invariant 1); same for `[1,0]`; `B2L1` mirrored;
`C2L3[tx,ty]` 1 put, 1 get. Acyclicity: the put→get graph is a two-level DAG
(segment → herd → segment); no SCC contains a channel edge.

**M5** → `air.api` calls in plan order: `air.tensor` ×3, `air.channel` ×3, `air.launch` →
`air.segment` → the A/B puts → `air.herd([range(2), range(2)])` with body `(tx, ty)` →
`air.alloc(acc)` → zeroing loops → `air.sequential(0,64,16)` → `air.alloc(a)`, `air.alloc(b)`,
two `get`s, three nested `air.sequential` loops with `acc[m,n] = acc[m,n] + a[m,t]*b[t,n]` →
after the herd, the four `C2L3` gets. Then `launch.build("npu1")`, `str(module)`.

**Design rule that falls out of this, and applies to all three kernels.** A loop whose index is
a **channel bundle index** must be a *Python* loop (a trace-time constant per put), because
`ChannelPutOp::verify` rejects a bundle index that is a temporal `scf.for` induction variable
(`AIRDialect.cpp:3589-3592`). A loop whose body **reuses an L1 buffer across trips** must be an
`air.sequential`, because a Python loop unrolls and "the objectFifo acquire/release pairs that
the pipeline would place inside the loop end up stranded between the unrolled copies, and the
kernel computes with stale operands" (`_loop.py:14-19`). Those two rules never conflict: the
bundle index is a *spatial* quantity, the reused buffer is driven by a *temporal* one. Herd
coordinates are neither — they are spatial and legal as bundle indices inside an
`air.sequential` body, which is what W2 and W3 rely on. Segment-scope `air.sequential` is legal
and idiomatic (`programming_examples/matrix_multiplication/bf16/run.py:161`).

**Open at D1 (Q-7)**: whether the segment-scope producer loop may precede the herd op (the
`broadcast` example's shape, generalised to several chunks) or whether the herd must sit inside
it (upstream's bf16 GEMM shape). The fallback is recorded with the question.

**Expected AIR shape** (from VF §E.5's measured probe on the same loop structure): a K `scf.for`
whose direct children are two `memref.alloc`s; after `air-label-scf-for-to-ping-pong` the loop
carries `unroll = 2 : i32`; after `air-ping-pong-transform` the step doubles and four async
tokens appear as `iter_args`.

**W1-flip (FR-K2)**: the same kernel text, `grid(PK)`, `place(px=ax.k0)`, `stationary("B")`,
`double_buffer("A")` — a **1-D** herd of `PK = 4` PEs, one coordinate `tx`. **`j` is not tiled**
(`TN = N = 64`), and that is what makes the flip *weight*-stationary rather than merely
spatially stationary (RULING 9): each PE's `B` tile is `B[kchunk, :]` = `[16, 64]`, constant
over the whole temporal sweep, so the weights are fetched once and stay put, while `A`'s
`[32,16]` tile changes with `i0` and is re-fetched on each of the `M/TM = 2` trips. `i` is tiled
`TM = 32` and `k` is tiled `TK = 16` over the `PK = 4` PEs. The residency line of the summary
prints exactly that (`03-lld-M4-mapping.md` §3.9), and it is the line the demo points at.
M3 then reports `R_space = span{e_k}` and demands the A/C tag (present); `ker M_B = span{e_i} ⊆ ker Sπ_u`, so the
declared `stationary("B")` also holds (this closes B-O5). M4 emits **`PK−1 = 3` cascade links on
one `npu_cascade` bundle** along the row, carrying one `[32, 64]` `f32` partial tile (`8 192 B`)
per link **per `i0` trip**, with the head at `tx == 0` skipping its `get` and the
tail at `tx == PK-1` putting `C[i0 tile, :]` to L3 once per `i0` — the shape of
`programming_examples/cascade_reduction/cascade_reduction.py`. **The chain ascends in `tx`**:
measured on the pinned wheel, a 1-D `grid(4)` herd compiles with
`aie.cascade_flow(%tile_0_2, %tile_1_2)`, `(1,2)→(2,2)`, `(2,2)→(3,2)` and `aircc` exit 0, while
the descending variant of the same module fails with `'aie.cascade_flow' op source tile must be
to the North or West of the destination tile` (REVIEW-round1 P-R3). A 2-D `(1,4)` herd is one
column of four rows and is the case that must descend; the direction is therefore a plan field
(`ChannelPlan.chain_direction`), never an emitter choice.

### 7.2 W2 — Jacobi 5-point, halo exchange, `T` timesteps

Fixture: `T=4`, `H=W=16`, `U: sp.f32[T+1, H+2, W]`, `PI=2`, `HS=8` (invariant `PI·HS == H`),
`f32`. The odd-`T` variant is `T=5`; `T=0` is the `SWAP-PARITY` negative; `PI=4` is the
`DMA-CHANNELS` negative (measured `aie.connect` failure, REVIEW-round1 P-R2).

**M1** → one rank-3 parameter `U`, written at `U[t+1, i, j]` and read at five places in plane
`t`; domain `[0,T) × [1,H+1) × [1,W-1)`, i.e. write planes `1..T`, rows `1..H`, cols `1..W-2`.
Rows `0`, `H+1` and columns `0`, `W-1` are read-only Dirichlet boundary and are never assigned.
`R = {}` (the 5-point sum is an unrolled window, not a reduction); uniform dependences
`(1,0,0)`, `(1,±1,0)`, `(1,0,±1)` from `U[t+1,…]` to `U[t,…]`.

**M3** → `Sπ = [e_i0]`; `t ∈ ker Sπ` so the strip is strip-stationary (derived, not declared);
(L2) holds because `σ`'s first row is `t`; halo footprint along `i` is 1 and along `j` is 1, so
`window(halo=1)` satisfies FR-L7; physical herd on `npu1` = `(2,)`, repeats `(1,)`; L1 =
two `(HS+2)×W` f32 strips = `2 × 10 × 16 × 4` = **1 280 B** (§7's table).

**M4** → plan:

| item | value |
|---|---|
| tensors | one L3 `BufferPlan`, `U [T+1, H+2, W] f32` — the kernel's single parameter, read **and** written, so it is both the staged input and the drain target (§5.6 invariant 6 is vacuous with one tensor) |
| herd | `[range(2)]`, 1-D, name `jacobi_herd`, coordinate `tx` |
| buffers | `cur [10,16] f32 h.private()`, `next [10,16] f32 h.private()`, both at depth 0 (outside the `t` loop) |
| channel `UIn` | `size=[PI] = [2]`; stages plane 0 into `cur` **once**, before the `t` loop: PE `tx` gets `U[0, tx·HS : tx·HS+HS+2, :]`, i.e. its `HS` rows plus both ghost rows |
| channel `UOut` | `size=[PI] = [2]`; one put per PE per timestep of the computed strip, region `U[t+1, tx·HS+1 : (tx+1)·HS+1, :]` — **every plane the kernel writes is drained**, which is what makes `test_sem_coverage` pass (`04-test-plan.md` §3.4) |
| channel `ToNorth` | `size=[PI-1] = [1]`; index `p` = "PE `p+1` sends its top boundary row to PE `p`" |
| channel `ToSouth` | `size=[PI-1] = [1]`; index `p` = "PE `p` sends its bottom boundary row to PE `p+1`" |
| per timestep, per PE, **in this order** | `ToNorth.put(cur[1:2, :], indices=[tx-1])` guarded `tx > 0`, async; `ToSouth.put(cur[HS:HS+1, :], indices=[tx])` guarded `tx < PI-1`, async; `ToNorth.get(cur[HS+1:HS+2, :], indices=[tx])` guarded `tx < PI-1`, **no token dependency on the puts**; `ToSouth.get(cur[0:1, :], indices=[tx-1])` guarded `tx > 0`, likewise; then the update `next[i,j] = 0.2 × (cur[i,j] + cur[i-1,j] + cur[i+1,j] + cur[i,j-1] + cur[i,j+1])`; then `UOut.put(next[1:HS+1, :], indices=[tx])` |
| loop shape | `air.sequential(0, T, 2)` over `t` inside the herd body, with the body emitted **twice** — once `cur → next`, once `next → cur`; the update is two nested `air.sequential` loops over `(i, j)`. The unroll by two is how the swap is realised, because `air.sequential` has **no loop-carried values** (`_loop.py:180`; VF §D.5) and a plain Python `t` loop would unroll the whole exchange and strand the acquire/release pairs (`_loop.py:14-19`). An **odd** `T` is legal: `floor(T/2)` pairs inside the loop and one **peeled** timestep after it (FR-L14). `T = 0` is `SWAP-PARITY` |
| per-core endpoints at `PI=2` | inbound = `UIn` + one ghost link = **2**; outbound = `UOut` + one boundary link = **2**. Both are exactly at the 2-S2MM / 2-MM2S circuit budget, which is why `PI ≥ 3` — whose interior PE needs three circuit-switched inbound — is rejected by `DMA-CHANNELS` before `aircc` fails on it |
| delivery summary | `U: stationary strip (derived); halo exchange along px, width 1 (declared)` |

Balance self-check, per channel index `k` on `ToSouth` (`size=[1]`, so `k = 0` only): put by PE 0
under `0 < PI-1`; get by PE 1 under `1 > 0`. One put, one get, each `T` times. `ToNorth`
mirrored. `UIn[tx]`: one put at segment scope, one get per PE. `UOut[tx]`: `T` puts by PE `tx`,
`T` gets at segment scope. Balanced per iteration and per branch. Because the grid extent is a
compile-time constant, the check enumerates the two coordinate values rather than reasoning
symbolically. Acyclicity: within a timestep the graph is `put(p)→get(p±1)` plus the drain edge;
there is no path from a get back to a put in the same iteration because the two puts precede
both gets in program order and carry no token into them. The loop-carried edge is a
`StructuralBack` edge and is excluded, exactly as the spec prescribes
(`docs/AIRCorrectnessChecker.md` §5.3).

**Why there is no prologue and no seeded ghost**: at `t = 0` each PE puts its *initial* boundary
row and gets the neighbour's *initial* boundary row, which is the correct value (PC §3.2). A
prologue put would give `T+1` puts against `T` gets and break per-iteration balance. `UIn` is
outside the `t` loop and is not part of that count.

**Why the surface's oracle is safe here**: the kernel's outermost loop is `t`, so the
un-annotated nest is already timestep-outermost — the failure VF §I measured (PE-outermost
mismatches, maxerr 0.203) is a property of a *per-PE* surface's fallback, not of this one.

**Tolerance**: the update multiplies by `0.2`, so `f32` rounding is not associative and the
device/oracle diff for W2 uses `tol = 1e-5`, not `0.0` (`04-test-plan.md` §8 item 5). W1,
W1-flip and W3 stay exact.

**Double buffering**: `double_buffer("U")` is asserted against the `cur`/`next` pair. The strips
are allocated **outside** the `t` loop, so they are not ping-pong candidates in
`isPingPongCandidate`'s sense; the double buffering here is the explicit `cur`/`next` pair, and
the checker's message says exactly that rather than promising the pass will fire. *(This is the
one place where `double_buffer` means "the emitter writes two buffers", not "the pass unrolls by
two"; recorded as decision D-5 in `00-README.md` §5.)*

### 7.3 W3 — Smith-Waterman anti-diagonal wavefront

Fixture: `MQ=32`, `NR=32`, `PJ=4`, `CW=8`, `MATCH=+2`, `MISMATCH=-1`, `GAP=1`, `i32` scores.

**M1** → three parameters: `q i32 [MQ]` (read), `r i32 [NR]` (read), `S i32 [MQ+1, NR+1]`
(written) — read-only params first, which is the order `MappingPlan.tensors` must keep
(`06-interfaces.md` §5.6 invariant 6). Domain `[1,MQ+1) × [1,NR+1)`; the body is

```text
sub = MATCH if q[i-1] == r[j-1] else MISMATCH
S[i, j] = max(0, S[i-1, j-1] + sub, S[i-1, j] - GAP, S[i, j-1] - GAP)
```

so `MATCH`, `MISMATCH` and `GAP` resolve from `fn.__globals__` as integer literals (FR-S3 item
7), `sub` forward-substitutes transitively (item 6), and the conditional is a value-level
`Select`, not control flow (item 8). Dependence vectors `(1,0)`, `(0,1)`, `(1,1)`, all RAW on
`S`; `max` is a 4-ary expression, **not** a reduction over an axis, so `R = {}` and FR-L4 records
an empty split. Row 0 and column 0 of `S` are zero-initialised read-only boundary.

**M3** → `Sπ = [e_j0]`; `σ`'s leading row is `i + j0` from `skew(time=(ax.i, ax.j0))`, and the
remaining rows are the loop order minus the placed and skewed axes, leaving `[e_j1]` — so
`Sσ = [e_i + e_j0; e_j1]` (FR-S17). (L2) checks `Sσ·(1,0) = (1,0) ≻ 0`,
`Sσ·(0,1) ≻ 0` (via `j0`'s coefficient after tiling) and `Sσ·(1,1) ≻ 0`. **The demo's headline
rejection** is the *dropped* term, `skew(time=(ax.i,))`: `Sσ` becomes `[e_i; e_j1]`, the
tile-crossing representative of `d = (0,1)` gives `Sσ·d = (0, −7) ⪯ 0`, and the checker prints
the violating vector. (Reversing the two terms is a no-op — `skew` is a sum.) Physical herd on
`npu1` = `(4,)`, repeats `(1,)`. L1 at M3's scope = `q` 128 B + `r` 32 B + `prev`/`cur` at
`[CW+1] i32` = 36 B each = **232 B**; M4 adds `edge_in`/`edge_out` for **240 B** (§7's table).

**M4** → plan:

| item | value |
|---|---|
| tensors | `q`, `r`, then `S` — **all read-only params before the written one**, or `_check_interface` raises `output tensors must be declared after all input tensors` (`_compile.py:226-240`, measured P-R4) |
| herd | `[range(4)]`, 1-D, name `sw_herd`, coordinate `tx` |
| buffers | `qb [32] i32`, `rb [8] i32`, `prev [9] i32`, `cur [9] i32`, `edge_in [1] i32`, `edge_out [1] i32`, all `h.private()` |
| channel `QIn` | `size=[1]`, `broadcast_shape=[PJ] = [4]` — `q` is read by every PE at every column, so it is a multicast of the whole vector into `qb`, staged once before the row loop |
| channel `RIn` | `size=[PJ] = [4]` — PE `tx` gets `r[tx·CW : (tx+1)·CW]` into `rb`, staged once before the row loop |
| channel `WestIn` | `size=[1]`, L3→L1 — the constant-zero column-0 boundary into PE 0, `MQ` puts at segment scope |
| channel `West` | `size=[PJ-1] = [3]`, L1→L1 — the PE-to-PE links |
| channel `EastOut` | `size=[1]`, L1→L3 — the drain from PE 3, `MQ` gets at segment scope |
| channel `SOut` | `size=[PJ] = [4]` — **one put per row per PE** of `cur[1:CW+1]` into `S[i, tx·CW+1 : (tx+1)·CW+1]`, so every row of the write domain reaches L3 |
| per row, per PE | `WestIn.get(edge_in, indices=[0])` when `tx == 0` else `West.get(edge_in, indices=[tx-1])`; the `CW` inner updates as one `air.sequential` loop, each a `StoreNode` whose expression is `MaxMin("maximum", (0, prev[j-1] + Select(…), prev[j] − GAP, cur[j-1] − GAP))`; `West.put(edge_out, indices=[tx])` when `tx < PJ-1` else `EastOut.put(edge_out, indices=[0])`; `SOut.put(cur[1:CW+1], indices=[tx])`; then the `prev`/`cur` role swap |
| loop shape | `air.sequential(1, MQ+1, 2)` over rows in the herd body, body emitted **twice** for the `prev`/`cur` swap (same reason as W2; an odd `MQ` is peeled, FR-L14); the bundle index is `tx`/`tx-1`, **never** the loop IV — required by `ChannelPutOp::verify` (H-3) |
| per-core endpoints | inbound = `QIn` + `RIn` + one west link = **3**; outbound = one east link (or `EastOut`) + `SOut` = **2**. The three inbound compile because L3→L1 gets lower to **packet** flows, which multiplex: measured, `aie.q7a.mlir` carries 3 `aie.packet_flow` with `air_A2L1_0`, `air_B2L1_0` and `air_B2L1_1` all allocated `shim_noc_tile_0_0, MM2S, 0`, and `aircc --device npu1 --output-format=none` exits 0 with zero `error:` lines (REVIEW-round1 P-R4). `DMA-CHANNELS` therefore counts only circuit-switched endpoints as errors and **warns** here |
| delivery summary | `q: multicast along px (derived); r: column band stationary (derived); S: column band stationary (derived), forward W->E along px, one scalar per row (declared)` |

Balance self-check: `WestIn[0]` → `MQ` puts (segment source), `MQ` gets (PE 0 under `tx == 0`).
`West[k]` for `k ∈ {0,1,2}` → `MQ` puts (PE `k` under `tx < PJ-1`), `MQ` gets (PE `k+1` under
`tx > 0`). `EastOut[0]` → `MQ` puts (PE 3 under `tx == PJ-1`), `MQ` gets (segment drain).
`QIn[0]` → one put, one get at each of the 4 fan-out indices (the broadcast rule). `RIn[tx]`,
`SOut[tx]` → one and `MQ` respectively per PE. The head and tail **are** guarded by
`ops.branch`; a single `PJ+1` bundle spanning L3 and core-to-core members is not an option,
because `AIRLoweringPass.cpp:798` rejects it with `failed to specialize channel bundle indices`
(measured), which is why FR-M5 specifies three homogeneous channels. Acyclicity: a left-to-right
chain, acyclic by construction; the wavefront is an emergent consequence of each PE blocking on
its `get`, and nothing in AIR expresses the skew explicitly (PC §3.3 W3·P1 (d)).

---

## 8. Build order and integration checkpoints

The rule: **every module is testable against a stub from D1.** Stubs are hand-written
`MappingPlan` / `LegalMapping` literals in the fixtures package, owned by whoever consumes them.

| Stage | What is real | What is stubbed | Checkpoint (the observable that closes it) |
|---|---|---|---|
| **D0** | `model` (M0) frozen at `CONTRACT_VERSION = 3`; toolchain installed; W1/W2/W3 sources written | everything else | `06-interfaces.md` (with every REVIEW-round1 edit and RULING 9's `MappingSummary.residency` already applied) signed by all three; `air-opt --version`, `aircc --help`, `import air` all exit 0 on three machines; the three sources parse with a throwaway `ast.parse` smoke script |
| **D1** | M1 + M2 (A); M5 skeleton driving a **hand-written** W1 plan (B); pytest harness + fixtures (C) | M3 passes everything; M4 returns the hand-written plan | B's emitter produces AIR text for the stub W1 plan and `launch.build("npu1")` succeeds |
| **D2** | M3 checks L1/L2/L3/L8; M4 W1 path (trichotomy, broadcast channels, buffer plan); M6 `aircc --output-format=none` | M4 protocols for W2/W3 | **W1 end to end from source**: `s.mlir()` → golden → `aircc --output-format=none` exits 0; `air-opt` labelling shows `unroll = 2` |
| **D3** | M4 self-check (balance + acyclicity); the negative corpus for L1/L2/L3/L7/L8/L9 | W2/W3 protocols | every negative test raises the right `code`; W1 oracle diff passes on CPU |
| **D4** | W3 protocol (source/drain, forward) end to end | W2 | W3 golden + `aircc` + oracle diff |
| **D5** | W2 protocol (halo put-first) end to end | — | W2 golden + `aircc` + oracle diff; device runs if hardware exists |
| **D6** | W1-flip cascade; remaining negative tests; mapping summary goldens | — | flip emits `PK−1 = 3` **ascending** cascade links on one bundle, `ir_facts.cascade_channels == 3`, and `aircc` accepts |
| **D7** | freeze; demo rehearsal | — | full suite green in under 3 minutes; pitch rehearsed twice |

**Integration contracts** (who hands what to whom):
A → B: `ScheduleModel` + `LegalMapping`, plus one hand-written example of each in the fixtures
package from D0. B → C: the AIR text file and the build metadata (target, output format).
A → C: the oracle callable (`kernel(*arrays)`). C → all: fixtures (inputs, expected outputs,
goldens) and the harness.

---

## 9. What upstream does that we must not duplicate

Recorded so nobody rebuilds it: multicast derivation from access maps
(`air-broadcast-detection`, though it only sees `air.dma_memcpy_nd` — VF §S9);
double-buffering construction (the four ping-pong passes, in the default pipeline —
VF §E.1); NumPy broadcast-shape validation (`ChannelOp::verify` — VF §A.4); FIFO ordering
repair (`air-enforce-channel-fifo-order`, unconditional in `aircc` — VF §B.3, §S8);
hierarchy-locality race detection (`air-verify-hierarchy-locality{strict=true}`, on by default
— VF §S8); L2 put/get rebalancing with dummy ops (`AIRToAIEPass.cpp:4484-4553` — VF §S4, and
note that this is a *repair*, not a diagnosis, which is why H-2 is ours); placement
(`air-place-herds`); vectorisation and VLIW scheduling (the AIE backend).
