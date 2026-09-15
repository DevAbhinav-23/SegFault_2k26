# Frozen interfaces — Spatial DSL

*Phase 1, 2026-09-12. **This document is frozen at D0.** A change requires all three owners to
sign it in `00-README.md` §4 and a version bump of `CONTRACT_VERSION` below.*

**This file is specification text, not code.** Field lists and signatures describe what must be
built; no implementation exists.

`CONTRACT_VERSION = 6`

*Version 6 (architect ruling, 2026-09-15) corrects §7.2's device half to what M6 implements.
`m6.run` gains `target` and `kernel_name` and an optional `workdir`; `m6.trace` gains `function`
and an optional `workdir`; `DiffReport` is named as the frozen dataclass it is, in
`spatial/model.py`. Forcing requirement: FR-T3 and FR-T4 — the frozen two-argument shapes cannot
construct an `XRTBackend` nor name the function for `air-runner -f` (C, `progress.md` §3). No
field of any §2-§5 record changes. Pending A, B and C signatures in `00-README.md` §4.*

*Version 5 (architect ruling, 2026-09-13) gives `Statement` the value it stores. `Statement.expr`
is the complete right-hand side written into `target`, after scalar forward substitution
(`03-lld-M1-frontend.md` §3.5), as an `ExprNode` tree over **kernel-level** operands — §2.4 says
what that means and §2.7 carries the invariant that every `Load` in it names a `Param` of the
kernel. Forcing requirement: FR-M8 and FR-E2, because M4 must emit the kernel's arithmetic as
`StoreNode`s and has no other source for W2's `0.2 × (five-term sum)` or W3's `max`/`Select`
recurrence — `KernelModel` as frozen at v4 records `kind`, `target`, `reads` and `op`, from which
only the accumulate form can be rebuilt (B-P19). Two clarifications ride with it and change no
field: §5.6's `declared` is `True` exactly when a clause names that operand's delivery (B-P17),
and §5.5 fixes the naming of a plan-synthesised loop axis (B-P18). Signed in `00-README.md` §4.*

*Version 4 (architect ruling, 2026-09-13) adds two things P0c's measurements proved missing.
`HerdPlan` joins the `PlanNode` union and `MappingPlan.segment_body` carries exactly one
`HerdPlan` node, at top level, equal to `MappingPlan.herd` — it marks where the emitter opens
`air.herd` and walks `herd_body`; forcing requirement: FR-E1's launch/segment/herd nesting
together with D-14, since without the marker M5 would have to infer the herd's position from
site kinds. `KernelModel.bindings` records the integer value of every shape parameter at
capture; forcing requirement: FR-M7, because `TENSOR_PLAN` must produce concrete L3 shapes and
M4 has no other source for them (`03-lld-M1-frontend.md` §10 Q-M1-2 already says the bindings are
recorded in the model). Signed in `00-README.md` §4.*

*Version 3 (architect RULING 9, pre-D0) adds `MappingSummary.residency` and the residency
lines it renders: the stationarity predicate is spatial, so a summary that says only
"stationary" does not say how long the operand actually stays in L1. Forcing requirement:
FR-M11 / FR-D2 as amended by RULING 9. Signed in `00-README.md` §4.*

*Version 2 (REVIEW-round1, pre-D0) rewrote `ComputeNode` as `StoreNode` + `ExprNode`, added
`BranchNode`, `LegalMapping.pi_u`/`ker_pi_u`, `ChannelSite.id`, `ChannelPlan.chain_direction`,
`MappingPlan.launch_name`/`segment_name`, the tensor-ordering invariant and two error codes.
Forcing requirement: REVIEW-round1 B-3, B-4, B-5, B-9, B-10, B-13. Signed in `00-README.md` §4.*

---

## 1. Scalar types and enumerations

| Name | Values | Notes |
|---|---|---|
| `Dtype` | `f32`, `f16`, `bf16`, `i32`, `i8` | carries `.bits`, `.numpy` (the numpy dtype), `.mlir` (the MLIR spelling `f32`/`f16`/`bf16`/`i32`/`i8`); `sizeof = bits // 8` |
| `Level` | `"L1"`, `"L2"`, `"L3"` | memory residence |
| `Scope` | `"herd.private"`, `"segment.private"`, `"segment.shared"`, `"segment.per_core"`, `"tensor"` | the `air.api` spelling; `"tensor"` means L3, which is **not** `alloc`-able (VF §D.2) |
| `Delivery` | `"STATIONARY"`, `"MULTICAST"`, `"FORWARD"`, `"CASCADE"`, `"NONE"` | the reuse trichotomy plus cascade (SD-04 §3) |
| `ReduceOp` | `"+"`, `"max"`, `"min"` | all associative and commutative (SD-02 §4; SD-03 §4) |
| `Pattern` | `"broadcast"`, `"forward"`, `"cascade"` | the `stream` clause's domain (SD-01 §3) |
| `Direction` | `"W->E"`, `"E->W"`, `"N->S"`, `"S->N"` | `forward`'s direction |
| `Target` | `"npu1"`, `"npu2"`, `"auto"` | exactly what `resolve_target` accepts (`_trace.py:172-177`) |
| `ChannelType` | `None`, `"npu_cascade"`, `"npu_dma_packet"` | `None` means the default `npu_dma_stream`; these are `air.api`'s implemented set (`_channel.py:547`) |
| `Stage` | `"grammar"`, `"clause"`, `"legality"`, `"mapping"`, `"emission"`, `"toolchain"` | which module raised |

An `Expr` is an affine expression: a frozen `(coeffs: Mapping[str, int], const: int)` where keys
are loop-variable or shape-parameter names. Equality and hashing are structural.

---

## 2. `model` — kernel-side contracts (produced by M1)

All dataclasses in this document are **frozen** (immutable), with tuples rather than lists, and
`__eq__` by value.

### 2.1 `Param`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `name` | `str` | the Python parameter name | a valid identifier; unique within a kernel |
| `dtype` | `Dtype` | element type from the annotation | — |
| `shape` | `tuple[int \| str, ...]` | per-dim extent: a positive int, a shape-parameter NAME, or — when the annotation entry is an expression that is not a bare NAME, e.g. `MQ + 1` — the int it evaluates to under `KernelModel.bindings` at capture | rank ≥ 1; every `str` entry appears in `KernelModel.shape_params` |
| `is_written` | `bool` | the body assigns to it | at least one param has `is_written` |

### 2.2 `Axis`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `name` | `str` | loop variable name, or `f"{parent}0"`/`f"{parent}1"` for a tile | unique |
| `lo`, `hi`, `step` | `Expr` | the `range` bounds | `step` is a positive constant |
| `extent` | `int \| None` | `(hi-lo)//step` when constant, else `None` | — |
| `parent` | `str \| None` | the axis this was strip-mined from | set only for tile handles |
| `depth` | `int` | nesting depth in the source nest (0 = outermost) | — |

### 2.3 `AccessMap`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `operand` | `str` | the `Param.name` accessed | exists in `KernelModel.params` |
| `matrix` | `tuple[tuple[int, ...], ...]` | linear part `M_a`, shape `(rank_a, n_axes)` | rank matches the param |
| `offsets` | `tuple[Expr, ...]` | the constant/parametric part, one per array dim | — |
| `is_write` | `bool` | write access rather than read | — |

### 2.4 `Statement`

| Field | Type | Meaning |
|---|---|---|
| `kind` | `"assign" \| "accumulate"` | `x[...] = e` vs `x[...] += e` (or `x[...] = max(x[...], e)`) |
| `target` | `AccessMap` | the written access |
| `reads` | `tuple[AccessMap, ...]` | every read access, in source order |
| `expr` | `ExprNode` | **the complete value stored into `target`**, after scalar forward substitution (`03-lld-M1-frontend.md` §3.5), over **kernel-level** operands: `Load(buffer_id=<a `Param.name`>, subscripts=<one `Expr` per array dim, over kernel axis names, shape-parameter names and constants>)`, `Const(value, text, dtype=<the target param's dtype>)` with `text` the source token (a module-level constant resolves to its integer's decimal form, `MISMATCH` → `"-1"`), `BinOp`, `Neg`, `MaxMin`, `Select` (§5.5's union). For `kind == "accumulate"` it is the **desugared** right-hand side: `C[i,j] += A[i,k]*B[k,j]` is `BinOp("+", Load("C",(i,j)), BinOp("*", Load("A",(i,k)), Load("B",(k,j))))` and `x = max(x, e)` is `MaxMin("maximum", (Load(x,…), e))`. Association follows Python's parser, so a sum of five terms is left-nested `((((a+b)+c)+d)+e)`. M4 rewrites each `Load` into the L1 buffer staging that operand and rebuilds no arithmetic of its own |
| `op` | `ReduceOp \| None` | for `accumulate`, the accumulation operator recovered from the source |
| `axes` | `tuple[str, ...]` | the enclosing loop axes, outermost first |
| `line` | `int` | source line, for diagnostics |

### 2.5 `Dependence`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `vector` | `tuple[int, ...]` | the uniform dependence vector `d`, one entry per axis | not all zero |
| `kind` | `"RAW" \| "WAR" \| "WAW"` | — | — |
| `operand` | `str` | which array carries it | — |

### 2.6 `ReductionSpec`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `target` | `str` | the accumulated operand | — |
| `projection` | `tuple[tuple[int, ...], ...]` | linear part `Sf` of `f : D → C-space` | — |
| `space` | `tuple[tuple[int, ...], ...]` | a basis of `R = ker Sf`, in a canonical (row-echelon) form | deterministic |
| `op` | `ReduceOp \| None` | `None` until `reduce(...)` tags it | — |

### 2.7 `KernelModel`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `name` | `str` | the function name | — |
| `source` | `str` | the exact source text captured | — |
| `params` | `tuple[Param, ...]` | in declaration order | — |
| `shape_params` | `tuple[str, ...]` | sorted names of symbolic extents | — |
| `bindings` | `tuple[tuple[str, int], ...]` | the integer value of every shape parameter at capture | one entry per `shape_params` name, sorted by name, values ≥ 1 — **except** a parameter bound to `0` to reach a §6.3 code (W2's `T = 0`, the only reachable `SWAP-PARITY` condition, `03-lld-M3-checker.md` §9). M0 therefore enforces the names and the order, and leaves the bound to M2/M3 (Q-M2-3) |
| `axes` | `tuple[Axis, ...]` | outermost first | names unique |
| `statements` | `tuple[Statement, ...]` | in source order | ≥ 1 |
| `dependences` | `tuple[Dependence, ...]` | sorted by `(operand, vector)` | — |
| `reduction` | `ReductionSpec \| None` | present iff a statement is an `accumulate` | — |

**Invariant (v5)**: every `Load` reachable in any statement's `expr` names a `Param` of this
kernel. It is the one reference check M0 enforces, because a `KernelModel` holds both sides of
it; every other reference stays M2/M3/M4's (Q-M2-3).

---

## 3. `model` — schedule-side contracts (produced by M2)

### 3.1 `AxisRef`

An opaque handle returned by `Schedule.axes()`. Carries `name: str` and the owning schedule's
identity. Comparing handles from different schedules raises `ClauseError(CLAUSE-UNKNOWN-AXIS)`.

### 3.2 `ScheduleModel`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `target` | `Target` | the device generation | in the `Target` set |
| `grid` | `tuple[int, ...] \| None` | logical PE grid | rank 1 or 2; every entry ≥ 1 |
| `tiles` | `tuple[tuple[str, int], ...]` | `(axis name, factor)` in call order | each factor ≥ 1 and divides the extent when constant |
| `place` | `tuple[str, ...]` | placed axis names, `px` first | `len(place) == len(grid)`; names distinct |
| `reductions` | `tuple[tuple[str, ReduceOp], ...]` | `(axis, op)` | axis appears in the kernel |
| `stationary` | `tuple[str, ...]` | operand names, sorted | each is a `Param.name` |
| `streams` | `tuple[StreamClause, ...]` | delivery overrides | one per operand at most |
| `residency` | `tuple[tuple[str, Level], ...]` | sorted by operand | one per operand at most |
| `double_buffer` | `tuple[str, ...]` | operand names, sorted | — |
| `pipeline` | `tuple[str, ...]` | axis names, in call order | hint only |
| `sequential` | `tuple[str, ...]` | axis names, sorted | disjoint from `place` |
| `windows` | `tuple[WindowClause, ...]` | — | one per operand at most |
| `exchanges` | `tuple[ExchangeClause, ...]` | — | one per operand at most |
| `skew` | `tuple[str, ...] \| None` | the `σ` terms in order | each name is an axis |

`StreamClause`: `(operand: str, pattern: Pattern, along: str, direction: Direction | None,
depth: int | None)`. `depth` is recorded and **never emitted** (FR-E4).
`WindowClause`: `(operand: str, dims: tuple[str, ...], halo: tuple[int, ...])`.
`ExchangeClause`: `(operand: str, along: str, halo: int)`.

**Invariant across the whole model**: `ScheduleModel` contains no object that is not hashable
and no reference to the kernel function. It is serialisable to JSON by field order.

---

## 4. `model` — legality contract (produced by M3)

### 4.1 `LegalMapping`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `kernel` | `KernelModel` | the input | — |
| `schedule` | `ScheduleModel` | the input | — |
| `axes` | `tuple[Axis, ...]` | the **post-tiling** axis list, outermost first | supersedes `kernel.axes` |
| `sigma` | `tuple[tuple[int, ...], ...]` | `Sσ`, rows = time dims | `rank(Sσ) + rank(Sπ) ≥ n` where the two kernels intersect trivially |
| `pi` | `tuple[tuple[int, ...], ...]` | `Sπ`, rows = PE coordinates | `len(pi) == len(grid)` |
| `ker_pi` | `tuple[tuple[int, ...], ...]` | a canonical basis of `ker Sπ` | deterministic |
| `pi_u` | `tuple[tuple[int, ...], ...]` | `Sπ` in the **untiled** frame: row `r` is `e_{root(place[r])}` | `len(pi_u) == len(pi)`; `len(kernel.axes)` columns |
| `ker_pi_u` | `tuple[tuple[int, ...], ...]` | a canonical basis of `ker Sπ_u` | deterministic |
| `r_time` | `tuple[tuple[int, ...], ...]` | basis of `R ∩ ker Sπ` | empty when there is no reduction |
| `r_space` | `tuple[tuple[int, ...], ...]` | basis of `R / R_time` | — |
| `stationary_ops` | `tuple[str, ...]` | operands proved stationary (declared **or** derived) | sorted |
| `physical_herd` | `tuple[int, ...]` | the resolved physical shape | each entry divides the corresponding grid entry and is ≤ the target cap |
| `repeats` | `tuple[int, ...]` | `grid // physical_herd` | — |
| `l1_bytes` | `int` | the per-core estimate **with `double_buffer` doubling applied** | ≤ 65536 |
| `halo_footprint` | `tuple[tuple[str, tuple[int, ...]], ...]` | per-operand derived footprint per windowed dim | — |

**Two frames.** `sigma`, `pi`, `ker_pi` are over the post-tiling axis order (`axes`, `Coord`);
`pi_u`, `ker_pi_u`, `r_time`, `r_space` and every `AccessMap.matrix` are over `kernel.axes`
(`UCoord`) — `03-lld-M3-checker.md` §3.1. M4 classifies delivery in `UCoord` only.

---

## 5. `model` — mapping contract (produced by M4)

### 5.1 `BufferPlan`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `name` | `str` | emitted buffer name | unique in the plan |
| `operand` | `str \| None` | the kernel operand it stages, if any | — |
| `level` | `Level` | — | `L3` ⟹ `scope == "tensor"` |
| `scope` | `Scope` | the `air.api` spelling | never `"herd.shared"` — that raises (`_trace.py:1405-1414`) |
| `shape` | `tuple[int, ...]` | concrete extents | all ≥ 1 |
| `dtype` | `Dtype` | — | — |
| `bytes` | `int` | `prod(shape) * dtype.sizeof` | — |
| `loop_depth` | `int` | nesting depth of the `air.alloc` inside the herd body (0 = herd body top level) | — |
| `ping_pong_candidate` | `bool` | true iff the plan asserts it satisfies `isPingPongCandidate` | true ⟹ `loop_depth ≥ 1` and the first site touching it is a `get` |

### 5.2 `ChannelSite`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `id` | `str` | a plan-unique site identifier, `f"{channel}.{kind}.{order}@{scope}"` | unique in the plan |
| `kind` | `"put" \| "get"` | — | — |
| `channel` | `str` | the channel's symbol name | exists in the plan |
| `indices` | `tuple[Expr, ...]` | the bundle index, one `Expr` per channel-`size` dim | **no `Expr` may reference a temporal loop IV** (`AIRDialect.cpp:3589-3592`) |
| `buffer` | `str` | the `BufferPlan.name` or L3 tensor name at this end | exists |
| `region` | `Region` | offsets / sizes / strides of the transferred slab | rank matches the buffer |
| `scope` | `"launch" \| "segment" \| "herd"` | where the op is emitted | an L3 endpoint inside a herd body ⟹ a segment exists (the *implemented* rule, PC §1.1 fact 4) |
| `guard` | `Guard \| None` | an `ops.branch` condition on herd coordinates | never a Python `if` (`_cond.py:41-45`) |
| `is_async` | `bool` | emitted in asynchronous form | — |
| `depends_on` | `tuple[str, ...]` | `ChannelSite.id` values this site takes a token from | **empty for the halo gets** (FR-M4) |
| `order` | `int` | position in the enclosing body | strictly increasing within a body |

`Region`: `(offsets: tuple[Expr, ...], sizes: tuple[int, ...], strides: tuple[int, ...])`.
`Guard`: `(coord: str, relation: "==" | "!=" | "<" | "<=" | ">" | ">=", value: Expr)`.

### 5.3 `ChannelPlan`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `name` | `str` | symbol name, e.g. `A2L1` | unique; a valid MLIR symbol |
| `size` | `tuple[int, ...]` | the bundle extents | rank ≥ 1 |
| `broadcast_shape` | `tuple[int, ...] \| None` | fan-out extents | if present: same rank as `size`, and `broadcast_shape[d] % size[d] == 0` (`_channel.py:137-145`) |
| `channel_type` | `ChannelType` | — | `"npu_cascade"` ⟹ `broadcast_shape is None` (`_channel.py:170-177`) |
| `chain_direction` | `"ascending" \| "descending" \| None` | the cascade chain's orientation in the carrying herd coordinate | non-`None` **iff** `channel_type == "npu_cascade"` |
| `dtype` | `Dtype` | payload element type | — |
| `sites` | `tuple[ChannelSite, ...]` | ordered | ≥ 1 put and ≥ 1 get |

### 5.4 `HerdPlan`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `name` | `str` | herd symbol name | — |
| `grid` | `tuple[int, ...]` | logical extents | rank 1 or 2 (`_trace.py:1288-1291`) |
| `shape` | `tuple[int, ...] \| None` | explicit physical shape, when pinned | each entry divides `grid` exactly (`_trace.py:1355-1372`) |
| `at` | `tuple[int, int] \| None` | `(column, row)` pinning | set only when a cascade chain needs a fixed line (`_trace.py:1771-1778`) |
| `coords` | `tuple[str, ...]` | body parameter names, e.g. `("tx", "ty")` | length == rank |

### 5.5 `LoopPlan`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `axis` | `str` | the axis it realises | — |
| `lo`, `hi`, `step` | `Expr` | bounds | constants where the ping-pong pattern requires static trip counts |
| `kind` | `"sequential" \| "unrolled"` | `air.sequential` vs a trace-time Python loop | `"unrolled"` only outside a herd body |
| `depth` | `int` | nesting depth | — |
| `body` | `tuple[PlanNode, ...]` | ordered children: allocs, sites, loops, compute | — |

`LoopPlan.axis` names the axis the loop realises, under one rule (v5, the ruling on B-P18):

* a **bundle-index** loop is `p<root>_bundle`, where `root` is the untiled parent of the placed
  axis that PE dimension carries — W1's `pi_bundle` and `pj_bundle`, the flip's `pk_bundle`;
* a **drain** loop is `<axis>_drain`, naming the axis whose trips it enumerates — W1's `i_drain`
  and `j_drain`, the flip's `i0_drain`, W3's `i_drain`;
* a **source** loop is `<axis>_source`, the same rule for the other direction — W3's `i_source`,
  the segment-scope loop that puts the forwarded operand's boundary column once per row.
  *(Added 2026-09-13 at P4, ruling **R-W3-3**: §6.1's protocol has no source loop, so the rule
  had no spelling for one. Both W3 row loops are `air.sequential` — a row index is not a bundle
  index, so `LOOP_KIND` takes its default — and only the PE loop around them is `unrolled`.)*
* a **compute or zeroing** nest is named by the **post-tiling axis it realises** — W1's `i1`,
  `j1`, `k1` (its zeroing nest `i1`, `j1`), the flip's `i1`, `j`, `k1`, W2's `i1`, `j`, W3's
  `j1` — never a positional `m`/`n`/`t`/`m0`/`n0`.

A `LoopPlan.axis` may therefore recur in different bodies (the zeroing nest and the compute nest
both realise `i1`); M5 rebinds the name in order as it walks, which is what it already does.
The first two forms match no `KernelModel.axes` entry; the third matches a `LegalMapping.axes`
entry for a tiled axis and a kernel axis for an untiled one.

`PlanNode` is the union `BufferPlan | ChannelSite | LoopPlan | StoreNode | BranchNode | HerdPlan`.

A `HerdPlan` node may appear only in `MappingPlan.segment_body`, exactly once and at top level,
and must equal `MappingPlan.herd`; it marks where the emitter opens the herd and walks
`herd_body`.

`StoreNode`: `(buffer_id: str, subscripts: tuple[Expr, ...], expr: ExprNode)` — one store of one
expression tree into one buffer. `ExprNode` is the frozen union

```text
ExprNode = Load(buffer_id: str, subscripts: tuple[Expr, ...])
         | Const(value: int | float, text: str, dtype: Dtype)
         | BinOp(op: "+" | "-" | "*" | "/", lhs: ExprNode, rhs: ExprNode)
         | Neg(operand: ExprNode)
         | MaxMin(op: "maximum" | "minimum", operands: tuple[ExprNode, ...])
         | Select(cmp_op: "==" | "!=" | "<" | "<=" | ">" | ">=",
                  lhs: ExprNode, rhs: ExprNode,
                  then: ExprNode, otherwise: ExprNode)
```

* `Const.text` is the source token, never a float round-trip (`02-hld.md` §5 rule 4).
* Every `buffer_id` names a `BufferPlan` in this plan, so M5 resolves it by lookup and never by
  derivation (D-14). The same union is also used *kernel-level* by `Statement.expr` (§2.4 at
  v5), where a `Load` names a `Param` instead; M4 rewrites each such `Load` into the buffer
  staging that operand, and only the rewritten form ever reaches a `StoreNode`.
* A `StoreNode` whose subscripts are fully integer emits one
  `memref.load`/`arith`/`memref.store` triple (`_value.py:590-593`, VF §D.11).
* **There is no whole-buffer form.** `b[:] = b[:] + r[:]` is expanded by M4 into an explicit
  `LoopPlan` over the tile carrying scalar `StoreNode`s (REVIEW-round1 RULING 3).
* Accumulator zeroing is an ordinary M4-produced `LoopPlan` of `StoreNode(Const 0)`; it carries
  no `statement_index`, because no kernel statement corresponds to it.
* `Select` is how a value-level conditional (`a if cmp else b`, `01-requirements.md` FR-D9)
  reaches the emitter — `arith.select`, an expression, never control flow.

`BranchNode`: `(predicate: Guard, then: tuple[PlanNode, ...], otherwise: tuple[PlanNode, ...])` —
emitted as `with air.ops.branch(predicate) as h: <then>` followed by `with h.otherwise(): <otherwise>`.
Conjunction is nesting; there is no `and` (`_cond.py:57-60`). `otherwise` may be empty, which
leaves an `scf.if` with an empty else that `RemoveEmptyElseBranch` deletes (`_cond.py:61-66`).
`BranchNode.then`/`.otherwise` are proper regions: they may contain sites, loops, stores and
further `BranchNode`s. `ChannelSite.guard` stays for the single-site case and is exactly
`BranchNode(predicate, (site,), ())`.

### 5.6 `MappingPlan`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `mapping` | `LegalMapping` | the input | — |
| `tensors` | `tuple[BufferPlan, ...]` | the L3 interface: one `BufferPlan` per `KernelModel.param`, `level="L3"`, `scope="tensor"` | every `level == "L3"`; every read-only param precedes every written param |
| `launch_name` | `str` | the `air.launch` symbol name | a valid MLIR symbol |
| `segment_name` | `str` | the `air.segment` symbol name | a valid MLIR symbol |
| `herd` | `HerdPlan` | — | — |
| `buffers` | `tuple[BufferPlan, ...]` | non-L3 buffers, in allocation order | — |
| `channels` | `tuple[ChannelPlan, ...]` | sorted by name | — |
| `segment_body` | `tuple[PlanNode, ...]` | nodes emitted at segment scope, before and after the herd | contains exactly one `HerdPlan` node, at top level, `== herd` |
| `herd_body` | `tuple[PlanNode, ...]` | nodes emitted inside the herd body | — |
| `delivery` | `tuple[tuple[str, Delivery, str \| None, bool], ...]` | `(operand, delivery, along PE axis, declared)`. `declared` is `True` **iff a clause names this operand's delivery** — `stationary(a)`, `stream(a, …)`, `forward(a, …)` or `exchange(a, …)`: a fact about the schedule, not about whether the derivation would have agreed (v5, the ruling on B-P17) | one per operand |
| `summary` | `MappingSummary` | see §5.7 | — |

**Plan invariants checked by M4's self-check before the plan leaves the module:**
1. **Balance (P1′)**: for every `(channel, index)` key, `put_count == get_count`, evaluated per
   loop body, per `BranchNode` arm, and with the cross-scope product rule. For a channel with
   `broadcast_shape`, a put at index `i` must be matched by exactly one get at **each** index of
   `i`'s fan-out set. *(Divergence from the upstream spec's literal P1, which keys on
   `(name, indices)` and would flag `air.api`'s own broadcast idiom — see `00-README.md` §5
   decision D-2.)*
2. **Acyclicity (P2b)**: no strongly connected component of (channel edges ∪ program-order
   edges, minus loop-carried back edges) contains a channel edge — the upstream spec's
   algorithm (`docs/AIRCorrectnessChecker.md` §5.2–5.3).
3. **Bundle indices**: no `ChannelSite.indices` entry references a temporal loop IV.
4. **Ping-pong shape**: every `BufferPlan` with `ping_pong_candidate` is a direct child of a
   `LoopPlan`, its first touching site is a `get`, it has exactly one `get` per iteration, and
   every enclosing `LoopPlan` has constant bounds.
5. **L1 budget**: `sum(b.bytes × (2 if ping_pong_candidate else 1))` over herd-private buffers
   ≤ 65536 (`_trace.py:100`).
6. **Tensor order**: `tensors` has one entry per `KernelModel.param`, in the order *all read-only
   params, then all written params* (each group keeping declaration order). Every `BufferPlan`
   the kernel only reads precedes every one it writes; `air.api`'s `_check_interface` raises
   `RuntimeError: output tensors must be declared after all input tensors` otherwise
   (`python/air/api/_compile.py:226-240`, measured REVIEW-round1 P-R4). `level == "L3"` ⟹
   `scope == "tensor"`, and `tensors` is the only place an L3 `BufferPlan` is created.
7. **Names**: `launch_name` and `segment_name` are set by M4 and emitted verbatim by M5, which
   derives no name.
8. **Herd position**: `segment_body` contains exactly one `HerdPlan` node, at top level — never
   nested inside a `LoopPlan` or a `BranchNode` arm, and never in `herd_body` — and it equals
   `herd`. It is the position at which M5 opens `air.herd` and walks `herd_body`; without it M5
   would have to infer the herd's position from site kinds, which D-14 forbids.

### 5.7 `MappingSummary`

| Field | Type | Meaning |
|---|---|---|
| `lines` | `tuple[str, ...]` | the rendered human-readable summary (FR-M11), one line per fact — the delivery block, then **one residency line per operand**, then the herd, reduction, buffer, L1 and channel lines |
| `residency` | `tuple[tuple[str, str], ...]` | one `(operand, duration)` pair per operand, sorted by operand: how long that operand's L1 tile stays put. `duration` is `"resident for the whole run"` or `"[resident across <axes>, ]re-fetched per <axis>"`, computed by `03-lld-M4-mapping.md` §3.9. The rendered line is `"<a>: <delivery>, <duration>"`, e.g. `B: stationary (spatial), resident for the whole run` |
| `herd_logical`, `herd_physical`, `repeats` | `tuple[int, ...]` | — |
| `reduction_split` | `tuple[str, str]` | rendered `R_time` / `R_space` bases |
| `l1_bytes`, `l1_budget` | `int` | — |
| `channels` | `tuple[tuple[str, tuple[int, ...], tuple[int, ...] \| None], ...]` | `(name, size, broadcast_shape)` |

### 5.8 `EmitResult`

| Field | Type | Meaning |
|---|---|---|
| `mlir` | `str` | the AIR module text |
| `target` | `Target` | the resolved target (never `"auto"` here) |
| `plan` | `MappingPlan` | the plan it came from |
| `summary` | `MappingSummary` | — |
| `l1_peak` | `int` | `LaunchContext._l1_peak` as reported by `air.api` after tracing |

---

## 6. Diagnostics

### 6.1 `Diagnostic`

| Field | Type | Meaning | Invariant |
|---|---|---|---|
| `code` | `str` | a stable identifier from §6.3 | uppercase, hyphenated; never reused |
| `stage` | `Stage` | which module raised | — |
| `clause` | `str \| None` | the clause implicated, rendered as the user would write it | present for every `clause`/`legality` diagnostic |
| `reason` | `str` | one line, no trailing period | non-empty |
| `fix` | `str` | one concrete edit | non-empty |
| `location` | `tuple[str, int] \| None` | `(file, line)` | present for every `grammar` diagnostic |
| `details` | `Mapping[str, object]` | the concrete numbers (matrices, byte counts, vectors, counts per channel key) | JSON-serialisable; deterministic key order |

### 6.2 Error classes

`SpatialError(Diagnostic)` is the base. Leaf classes: `GrammarError`, `ClauseError`,
`LegalityError`, `MappingError`, `EmissionError`, `ToolchainError`. No other exception type is
raised from a public entry point (NFR-4, NFR-7).

### 6.3 Error-code catalogue (FR-D3)

| Code | Stage | Raised when |
|---|---|---|
| `GRAMMAR-BAD-ANNOTATION` | grammar | a parameter annotation is not `sp.<dtype>[...]` |
| `GRAMMAR-UNSUPPORTED-STMT` | grammar | `if`/`while`/`with`/`try`/`return`/tuple-assign in the body |
| `GRAMMAR-UNSUPPORTED-EXPR` | grammar | comprehension, lambda, attribute access, `**`, comparison |
| `GRAMMAR-UNSUPPORTED-CALL` | grammar | a call other than `max`/`min` |
| `GRAMMAR-NONAFFINE-SUBSCRIPT` | grammar | a subscript not affine in axes, shape params and constants |
| `GRAMMAR-BAD-RANGE` | grammar | a `for` that is not `range(...)`, or non-affine bounds, or a non-positive step |
| `GRAMMAR-NONUNIFORM-DEP` | grammar | a written array is read at an affine but non-uniform distance (`S[i,j] = S[j,i]`) |
| `CLAUSE-UNKNOWN-AXIS` | clause | an axis handle not from this schedule |
| `CLAUSE-UNKNOWN-OPERAND` | clause | an operand name not among the kernel params |
| `CLAUSE-BAD-ENUM` | clause | `pattern`/`op`/`dir`/`level`/`target` outside its domain |
| `CLAUSE-BAD-VALUE` | clause | a non-positive factor, halo, grid extent or depth |
| `CLAUSE-RANK` | clause | `place` rank ≠ `grid` rank, or a `window` dim list of the wrong length |
| `CLAUSE-DUPLICATE` | clause | the same axis placed twice, or the same operand given two `stream`s |
| `CLAUSE-TILE-DIVIDES` | clause | the tile factor does not divide a constant extent |
| `CLAUSE-GRID-RANK` | clause | `grid` of rank > 2 |
| `L1-CONFLICT` | legality | `ker Sσ ∩ ker Sπ ≠ {0}` |
| `L2-CAUSALITY` | legality | `Sσ·d ⪰ 1` fails for some dependence `d` |
| `STATIONARITY` | legality | `ker M_a ⊄ ker Sπ` for a declared `stationary(a)` |
| `REDUCE-NOT-ACCUMULATED` | legality | `reduce(ax)` on an axis the body does not accumulate over |
| `RSPACE-NO-AC-OP` | legality | `R_space ≠ {}` with no A/C operator declared |
| `CASCADE-RANK` | legality | a cascade realisation with `rank(R_space) ≠ 1` or a non-contiguous PE line |
| `CASCADE-BROADCAST` | legality | a broadcast pattern requested on a cascade operand |
| `HALO-TOO-SMALL` | legality | declared halo < derived footprint |
| `PLACE-EXTENT` | legality | a placed axis's extent ≠ the corresponding grid extent |
| `PLACE-SEQUENTIAL-CONFLICT` | legality | an axis is both placed and `sequential` |
| `HERD-RANK` | legality | grid rank > 2 at check time |
| `HERD-PHYSICAL` | legality | no divisor of a grid extent fits the target's physical cap |
| `L1-CAPACITY` | legality | the doubled per-core working set exceeds 65 536 bytes |
| `PINGPONG-SHAPE` | legality | `double_buffer(x)` asserted but the planned loop shape cannot satisfy `isPingPongCandidate` |
| `SWAP-PARITY` | legality | a buffer-swap protocol needs an even trip count and the loop's is odd |
| `BALANCE` | mapping | the plan's P1′ self-check failed |
| `CHANNEL-CYCLE` | mapping | the plan's P2b self-check found an SCC containing a channel edge |
| `BUNDLE-INDEX-IS-IV` | mapping | a bundle index references a temporal loop IV |
| `PROTOCOL-UNSUPPORTED` | mapping | a declared protocol has no synthesis rule (e.g. a non-neighbour partner map, or `reside(x="L2")`) |
| `DMA-CHANNELS` | mapping | a core needs more **circuit-switched** inbound or outbound DMA channels than the target provides |
| `EMIT-AIR-API` | emission | `air.api` raised; `details["air_api_message"]` carries the original |
| `EMIT-VERIFY` | emission | `module.operation.verify()` failed |
| `TOOL-TARGET` | toolchain | a target outside `{npu1, npu2, auto}` |
| `TOOL-MISSING-XCLBINUTIL` | toolchain | `output_format="xclbin"` with no `xclbinutil` on `PATH` |
| `TOOL-AIRCC-FAILED` | toolchain | `aircc` exited non-zero |
| `TOOL-DIAGNOSTIC` | toolchain | a tool printed a line matching `error:` **regardless of exit status** (FR-T5) |
| `TOOL-VERSION-PIN` | toolchain | the installed `mlir_air` version differs from the pin |
| `TOOL-NO-DEVICE` | toolchain | a device run was requested with no `/dev/accel*` |

---

## 7. Public entry points

Signatures are specification text. `raises` lists the error classes a caller must expect.

### 7.1 Surface (`spatial` package)

```
sp.kernel(fn: Callable) -> Kernel
    raises GrammarError (at decoration time)

Kernel.__call__(*args: numpy.ndarray) -> None
    runs the original body in CPython; raises nothing of ours

Kernel.model -> KernelModel

sp.schedule(kernel: Kernel, target: Target = "npu1") -> Schedule
    raises ClauseError (bad target)

Schedule.axes() -> Axes                         # attribute access per axis name
Schedule.grid(*extents: int) -> Schedule
Schedule.tile(axis: AxisRef, factor: int) -> tuple[AxisRef, AxisRef]
Schedule.place(px: AxisRef, py: AxisRef | None = None) -> Schedule
Schedule.reduce(axis: AxisRef, op: ReduceOp = "+") -> Schedule
Schedule.stationary(operand: str) -> Schedule
Schedule.stream(operand: str, pattern: Pattern, along: AxisRef,
                depth: int | None = None) -> Schedule
Schedule.forward(operand: str, along: AxisRef, dir: Direction,
                 depth: int | None = None) -> Schedule
Schedule.reside(**levels: Level) -> Schedule
Schedule.double_buffer(*operands: str) -> Schedule
Schedule.pipeline(axis: AxisRef) -> Schedule
Schedule.sequential(axis: AxisRef) -> Schedule
Schedule.window(operand: str, dims: tuple[AxisRef, ...], halo: int | tuple[int, ...]) -> Schedule
Schedule.exchange(operand: str, along: AxisRef, halo: int) -> Schedule
Schedule.skew(time: tuple[AxisRef, ...]) -> Schedule
    every clause raises ClauseError; every clause returns self so calls chain

Schedule.model -> ScheduleModel
Schedule.check() -> LegalMapping                 raises LegalityError
Schedule.plan() -> MappingPlan                   raises LegalityError, MappingError
Schedule.summary() -> str                        raises LegalityError, MappingError
Schedule.mlir() -> str                           raises LegalityError, MappingError, EmissionError
Schedule.emit(path: str) -> str                  as mlir(), plus OSError
Schedule.build(target: Target | None = None) -> EmitResult
                                                 as mlir(), plus ToolchainError
```

`Schedule.build` does **not** run `aircc`. It traces through `air.api` and returns the module
text (`_compile.py:106-158` runs `module.operation.verify()` and nothing else, VF §D.9).

### 7.2 Module entry points (internal, but contractual between owners)

```
m1.capture(fn: Callable) -> KernelModel                          raises GrammarError
m3.check(kernel: KernelModel, schedule: ScheduleModel) -> LegalMapping
                                                                 raises LegalityError
m4.plan(mapping: LegalMapping) -> MappingPlan                    raises MappingError
m4.self_check(plan: MappingPlan) -> None                         raises MappingError
m5.emit(plan: MappingPlan, target: Target) -> EmitResult         raises EmissionError
m6.artifact(mlir_path: str, target: Target,
            output_format: "none" | "pdi" | "xclbin") -> str      raises ToolchainError
m6.run(artifact: str, inputs: Sequence[ndarray], target: Target, kernel_name: str,
       workdir: str | PathLike | None = None) -> list[ndarray]    raises ToolchainError
m6.diff(device: Sequence[ndarray], oracle: Sequence[ndarray],
        tol: float) -> DiffReport                                 raises nothing
m6.trace(mlir_path: str, model_json: str, function: str,
         workdir: str | PathLike | None = None) -> str            raises ToolchainError
m6.has_device() -> bool                                           raises nothing
```

`DiffReport` is a **frozen dataclass in `spatial/model.py`** (a `_Model`, validated like every
other §2-§5 record, not a bare tuple): `matched: bool`, `max_abs_err: float`,
`first_mismatch: tuple[int, ...] | None`, `count_mismatched: int`, `total: int` — every count
names its denominator.

*v6, 2026-09-15.* `m6.run`'s two-argument shape cannot construct an `XRTBackend` (it needs the
`target` to resolve the device and the `kernel_name` to name the entry point), and `m6.trace`'s
cannot name the function for `air-runner -f`; `workdir` is optional on both and keeps the tool's
scratch out of the repository (invariant I-5). Proposed by C, `progress.md` §3; bumped by the
architect 2026-09-15. `has_device` is listed because every `requires_device` skip predicate
calls it.

---

## 8. Golden-file convention

| Kind | Path | Comparison |
|---|---|---|
| AIR module text | `tests/golden/<workload>.<variant>.<target>.air.mlir` | **byte for byte** |
| Mapping summary | `tests/golden/<workload>.<variant>.<target>.summary.txt` | byte for byte |
| Mapping plan | `tests/golden/<workload>.<variant>.plan.json` | canonical JSON (sorted keys, 2-space indent, `\n` line ends), compared as parsed objects |
| Pass-inspection facts | `tests/golden/<workload>.<variant>.ir_facts.json` | a small object: `{"pingpong_unroll": 2, "broadcast_pattern_count": 0, "cascade_channels": 0, "lock_init_histogram": {...}}` — the facts we assert about post-pass IR, so the test does not depend on the whole IR text |

The `<variant>` vocabulary is closed: `base` (the workload's default schedule) and `flip` (W1's
re-placed variant). The eight goldens are therefore `w1.base.*`, `w1.flip.*`, `w2.base.*`,
`w3.base.*`, each for `npu1` and `npu2` where the kind is target-specific.

*(Erratum, 2026-09-13, phase P5: `odd` joins `base` and `flip` — `w2.odd.<target>.plan.json`,
W2's `T = 5` variant. It is a **plan** golden only, deliberately: the module differs from
`w2.base` by the peeled tail alone, which `test_E_peel_is_plan_driven` asserts on the text
directly, so a second module golden would freeze the same fact twice and diff twice. The
summary is target-specific for the same reason W1's is — `<workload>.<variant>.<target>
.summary.txt`, the B-P14 ruling.)*

Rules:
1. Goldens are regenerated only by `pytest --update-goldens`, which rewrites every golden and
   prints a diff summary. A PR that changes a golden must say why in its message.
2. Goldens are valid **only for the pinned wheel** (FR-T6). `test_T6_versions` runs first; if
   the pin does not match, every golden test is skipped with a clear reason rather than failing.
3. The AIR text is portable because `air.api` builds under `Location.unknown()`
   (`_compile.py:127`), so no absolute path appears in it.
4. `ir_facts.json` exists so that an upstream pass change shows up as one changed number rather
   than a 400-line diff — in particular the `air-broadcast-detection` count (R-04, H-8).

---

## 9. Fixture format

One directory per workload: `tests/fixtures/<workload>/`.

| File | Content |
|---|---|
| `meta.json` | `{"workload": "W1", "params": {"M": 64, "N": 64, "K": 64, "TM": 32, "TN": 32, "TK": 16, "PI": 2, "PJ": 2}, "dtype": "f32", "seed": 0, "tol": 0.0, "generator": "make_fixture.py"}` |
| `inputs.npz` | named arrays matching the kernel's read parameters |
| `expected.npz` | named arrays matching the kernel's written parameters, produced by **numpy**, not by our oracle |
| `oracle.npz` | the same, produced by calling the decorated kernel — committed so a CPython regression is visible as a file diff |

Rules:
1. `expected.npz` is computed by an independent numpy expression (`A @ B`, a two-loop Jacobi, a
   textbook DP), so the oracle is checked against something that is not itself.
2. `seed` is part of the fixture; `numpy.random.default_rng(seed)` only.
3. Integer-valued fixtures where possible, so `tol == 0.0` and the diff is exact.
4. Fixtures are sized to NFR-3's 3-minute budget; the sizes are fixed in `04-test-plan.md` §4.
