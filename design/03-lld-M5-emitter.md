# LLD — M5, AIR emitter

*Phase 1, 2026-09-12. Owner: **Person B**. Reads on top of [`02-hld.md`](02-hld.md) §2, §7,
[`06-interfaces.md`](06-interfaces.md) §5, §7.2, §8 and
[`03-lld-M4-mapping.md`](03-lld-M4-mapping.md), whose `MappingPlan` is M5's only input.
Findings N-1…N-10 in [`03-lld-B-open-questions.md`](03-lld-B-open-questions.md) are cited by
number. **No implementation exists.** Ordered call sequences below are `air.api` calls, listed in
`pseudo` fences; they are not our DSL.*

---

## 1. Purpose and requirements satisfied

M5 translates a `MappingPlan` into `air.api` calls, in plan order, then calls
`launch.build(target)` and `str(module)`. **It makes no decisions.** That is decision D-14, and it
is the rule that keeps the goldens meaningful and M5 a one-day task: *if the emitter would have to
choose something, the plan is under-specified and the choice moves to M4.*

The operational form of D-14, which §7's lint test enforces: **M5's source contains no branch on a
`LegalMapping` or `ScheduleModel` field, and no arithmetic on kernel sizes.** It branches only on
the *shape of the plan* — is this node a `LoopPlan` or a `ChannelSite`, is `guard` `None`, is
`broadcast_shape` `None` — which is dispatch, not decision.

| FR | What M5 does for it | Where |
|---|---|---|
| FR-E1 | `air.tensor` / `air.launch` / `air.segment` / `air.herd` nesting, coords as positional args | §3.2 rows 1-4 |
| FR-E2 | native compute: `air.sequential` nests with fully-integer subscripts on L1 | §3.6 |
| FR-E3 | ping-pong loop shape: alloc as a direct child, first touch a `get` | §3.5 |
| FR-E4 | never passes `buffer_resources` | §4, P-6 |
| FR-E5 | multicast as `air.channel(size=, broadcast_shape=)`, put at segment, get in the herd | §3.2 row 5 |
| FR-E6 | memory spaces: `h.private()` for L1, `air.tensor` for L3; never `<herd>.shared()` | §3.2 row 6 |
| FR-E7 | AIR text out via `LaunchContext.mlir()`; `s.emit(path)` writes it | §3.8 |
| FR-E8 | cascade channels with `channel_type="npu_cascade"`, no `broadcast_shape` | §3.2 row 5c |
| FR-E9 | `build()` runs `module.operation.verify()`; failure surfaces as `EmissionError` | §3.8, §5 |
| FR-E10 | byte-identical text across processes and `PYTHONHASHSEED` values | §3.3 |

---

## 2. Public entry point

One: `m5.emit(...)` from `06-interfaces.md` §7.2, returning an `EmitResult` (§5.8) and raising
`EmissionError` and nothing else. The surface's `Schedule.mlir()`, `.emit(path)` and `.build()`
(§7.1) all funnel through it.

`EmitResult.l1_peak` is read back from `LaunchContext._l1_peak` after tracing
(`_compile.py:74`, `:150`), which is `air.api`'s own figure and is therefore the one to quote when
it disagrees with the plan's — §5 says what happens then.

---

## 3. Internal design

### 3.1 Shape of the module

```pseudo
 1  function EMIT(plan, target):                              # m5.emit
 2      DECLARE_TENSORS(plan.tensors)                          # air.tensor, in plan order
 3      DECLARE_CHANNELS(plan.channels)                        # air.channel, sorted by name
 4      launch := OPEN(air.launch, name=plan.launch_name)         # verbatim; M5 derives no name
 5      segment := OPEN(air.segment, name=plan.segment_name)
 6      WALK(plan.segment_body, bindings={})                   # §3.4
 7      #   ... the <HERD> marker is the HerdPlan node in plan.segment_body (§5.6 invariant 8,
 7a     #   exactly one, at top level, == plan.herd), at which WALK does:
 8      #       herd := OPEN(air.herd, iterable, name=plan.herd.name, shape=plan.herd.shape,
 9      #                    at=plan.herd.at)
10      #       bind plan.herd.coords to the body's positional parameters
11      #       WALK(plan.herd_body, bindings={coords})
12      module := launch.build(target)                         # runs module.operation.verify()
13      return EmitResult(mlir=str(module), target=target, plan=plan,
14                        summary=plan.summary, l1_peak=launch._l1_peak)
```

The segment is opened unconditionally. The *implemented* rule (`_channel.py`, PC §1.1 fact 4,
H-9) requires an enclosing `air.launch` for any L3 endpoint and a segment only when the endpoint
is inside a herd body; every one of our four plans has L3 endpoints at segment scope, and every
upstream example that touches L3 puts them inside a segment with the stated reason — *"reaching L3
needs a shim DMA allocation, and outside a segment there is none to link to"*
(`worker_to_worker.py:29-30`, `broadcast/single_herd/broadcast.py:15-17`). Making it
unconditional removes a decision from M5, which is the point.

### 3.2 The translation table — one row per plan element

This is the whole of M5. Each row is mechanical: read the fields named, make the call named.

| # | Plan element (`06-interfaces.md` §) | `air.api` call | Notes |
|---|---|---|---|
| 1 | `BufferPlan` with `level == "L3"` in `plan.tensors` (§5.1) | `air.tensor(shape, dtype)` | declaration order is the tuple's order. M5 **asserts** §5.6 invariant 6 before emitting — every read-only tensor precedes every written one — and raises `EMIT-AIR-API` naming `_check_interface` if it does not (`_compile.py:226-240`). L3 is not `alloc`-able (VF §D.2) |
| 2 | `MappingPlan` root (§5.6) | `air.launch(name=kernel.name)` | no `grid=`: our launch is a single point |
| 3 | — (unconditional, §3.1) | `air.segment(name=f"{kernel.name}_seg")` | |
| 4 | `HerdPlan` (§5.4) | `air.herd([range(g) for g in grid], name=name, shape=shape, at=at)` | `at` is passed only when not `None`, and it is always `None` (M4 §3.6.3; Q-1 measured that no pinning is needed); body arity is `len(coords)` |
| 5a | `ChannelPlan` with `broadcast_shape is None`, `channel_type is None` (§5.3) | `air.channel(name, size=size)` | |
| 5b | `ChannelPlan` with `broadcast_shape` | `air.channel(name, size=size, broadcast_shape=broadcast_shape)` | validated by `_channel.py:120-145` |
| 5c | `ChannelPlan` with `channel_type` | `air.channel(name, size=size, channel_type=channel_type)` | `broadcast_shape` is `None` by M4 invariant I-5 |
| 6 | `BufferPlan` with `level == "L1"`, `scope == "herd.private"` (§5.1) | `air.alloc(shape, dtype, scope=<herd>.private())` | `<herd>` is the open herd handle. `"herd.shared"` never occurs (`_trace.py:1405-1414` raises) |
| 7 | `LoopPlan` with `kind == "sequential"` (§5.5) | `for iv in air.sequential(lo, hi, step):` | one `scf.for` (`_loop.py:89`) |
| 8 | `LoopPlan` with `kind == "unrolled"` | `for v in range(lo, hi, step):` | a trace-time Python loop; segment scope only (§5.5 invariant) |
| 9 | `ChannelSite(kind="put")` (§5.2) | `<chan>.put(<obj>, indices=[...], dependency=<tok or None>)` | `<obj>` is §3.7's slice expression |
| 10 | `ChannelSite(kind="get")` | `<chan>.get(<obj>, indices=[...], dependency=<tok or None>)` | `dependency` is `None` when `depends_on == ()` — which is the halo gets, FR-M4 |
| 11 | `BranchNode` (§5.5) | `with air.ops.branch(<predicate>) as h:` `WALK(node.then)` … `with h.otherwise():` `WALK(node.otherwise)` | M5 reads `node.predicate/.then/.otherwise` and nothing else; `h.otherwise()` is skipped when `otherwise == ()`. Never a Python `if` (`_cond.py:41-45`); no `and` — conjunction is nesting (`_cond.py:57-60`) |
| 11b | `ChannelSite.guard` not `None` (§5.2) | the same, with a single-site `then` | the degenerate `BranchNode(predicate, (site,), ())` |
| 12 | `StoreNode` (§5.5) | `<buffer>[<subs>] = EMIT_EXPR(node.expr)` over L1 buffers | one `ExprNode` walk, §3.6; a fully-integer subscript is rank 0 and emits `memref.load`/`store` with no loop of its own (`_value.py:590-593`; VF §D.11) |
| 13 | `MappingPlan` complete | `launch.build(target)` then `str(module)` | `build()` runs `module.operation.verify()` and no pass pipeline (`_compile.py:128-158`; VF §D.9) |

Nothing else. There is no row for `air.extern`, `link_with`, `ops.dot`, `ops.load`/`ops.store`,
`air.parallel`, `buffer_resources`, `pad_before`/`pad_after`, `dest=`, `packet_ids`,
`<segment>.shared()` or `<segment>.per_core()` — §8 lists each with the reason it is absent, and
`test_emitter_construct_closure` asserts the emitter's source names no `air.` attribute outside
this table.

### 3.3 Emission order and determinism (FR-E10, NFR-1)

Five rules, each testable, and together they are HLD §5's five determinism rules specialised to
M5:

1. **Plan order is emission order.** `plan.tensors`, `plan.segment_body` and `plan.herd_body` are
   ordered tuples; `ChannelSite.order` is strictly increasing within a scope; `LoopPlan.body` is
   ordered. M5 iterates them and never sorts, re-orders or re-groups. The single exception is
   channel *declaration*, where `plan.channels` is already sorted by name (M4 I-7) and M5 emits in
   that order so the module's leading `air.channel` lines are stable.
2. **No dict or set iteration reaches an output.** The only mapping M5 holds is the
   name → handle table (§3.4), and it is read by key, never iterated.
3. **Locations are unknown.** `air.api` builds inside `with Context(), Location.unknown():`
   (`_compile.py:127`), so no absolute path enters the text and goldens are portable across the
   three machines (`06-interfaces.md` §8 rule 3).
4. **Literals come from the plan's `Expr`, formatted by one function**, never from a float
   round-trip. `0.2` in W2's stencil is the source token, not `repr(1/5)`.
5. **No wall-clock, no `id()`, no `uuid`, no RNG** anywhere in M5.

`test_E10_byte_identical` runs two emissions in one process and one in a fresh process with a
different `PYTHONHASHSEED` and compares the three texts.

### 3.4 Binding: herd coordinates and Python-loop bundle indices

M5 keeps one environment mapping plan-space names to `air.api` values. It has exactly three kinds
of entry, and the distinction between the second and third is the whole of D-3.

```pseudo
 1  env := {}                                  # str -> air.api value (IndexExpr, Buffer, Channel, Tensor)
 2
 3  # (a) buffers, tensors, channels: bound once at their declaration
 4  env[bufferplan.name]  := the air.alloc handle
 5  env[tensor.name]      := the air.tensor handle
 6  env[channelplan.name] := the air.channel handle
 7
 8  # (b) herd coordinates: bound to the body's POSITIONAL parameters, in coords order
 9  #     air.api arity-checks this (VF §D.1), so a rank mismatch is caught at trace time
10  with air.herd(...) as h:
11      @h.body
12      def _(*params):
13          for name, p in zip(plan.herd.coords, params):  env[name] := p
14
15  # (c) loop indices:
16  #     kind == "sequential"  -> env[axis] := the air.sequential IV   (an IndexExpr)
17  #     kind == "unrolled"    -> env[axis] := a Python int, rebound each trip
```

**Why (c) splits.** A `ChannelSite.indices` entry that names an `"unrolled"` loop's axis resolves
to a **Python int**, so the bundle index is a trace-time constant — which is what
`ChannelPutOp::verify` demands (`AIRDialect.cpp:3586-3593`, *"channel bundle indices must not be
temporal `scf.for` induction variables"*). An entry that names a herd coordinate resolves to the
herd's positional parameter, which is spatial and legal. An entry that names a `"sequential"`
loop's axis would resolve to an `scf.for` IV and is **impossible by construction**: M4's
`BUNDLE-INDEX-IS-IV` check (M4 LLD §3.7.3) rejects such a plan before M5 sees it. M5 does not
re-check; it asserts, and §5 says what an assertion failure means.

Finding N-6 records that upstream's check is narrower than ours — it rejects an index that *is* an
IV, not one derived from one by an `affine.apply`, which is why `air.api`'s own strip-mine repeat
index passes. M5 benefits from the strictness and does not exploit the narrowness.

**Strip-mining is `air.api`'s business, not M5's** (finding N-7). Passing `shape=` smaller than
the logical grid makes `air.api` wrap the whole herd body in an `scf.for` over `repeats` and
rewrite the coordinate as `affine_map<()[s0,s1]->(s0*R + s1)>`. M5 passes `plan.herd.shape`
through and does nothing else; in particular it does **not** try to reconcile `BufferPlan.loop_depth`
(plan space) with the IR nesting depth (which is one deeper when `repeats > 1`).

### 3.5 Emitting `air.sequential` so `isPingPongCandidate` holds (FR-E3, H-1)

`isPingPongCandidate`'s conditions (`AIRDependencyScheduleOpt.cpp:1604`, `:1620-1630`, `:1690`;
`Transform/Passes.td:970-971`; VF §E.2) and the plan invariant that guarantees each:

| # | `isPingPongCandidate` requires | Guaranteed by |
|---|---|---|
| 1 | an `scf.for`, not already carrying `unroll` | translation-table row 7: `air.sequential` is one `scf.for` (`_loop.py:89`), and M5 sets no attributes |
| 2 | an `air.execute`-wrapped `memref.alloc` as a **direct child** of the loop body | `06-interfaces.md` §5.6 invariant 4, checked by M4 §3.7.3 line 8. M5 emits `LoopPlan.body` in order, so a `BufferPlan` at position 0 of the body **is** a direct child. M5 must not hoist, cache or reuse an alloc across trips |
| 3 | the alloc is dead on entry — its first access is a definite write | invariant 4, line 9: the first touching site is a `get`, and an `air.channel.get` counts as a definite write |
| 4 | no opaque callee touching a herd block argument | FR-E2 and §8: M5 emits no `func.call`; `StoreNode` is native `memref.load`/`store` |
| 5 | at most one `air.channel.get` per candidate alloc per iteration, static trip counts on intervening loops | invariant 4, lines 10-11 |
| 6 | L1 budget: herd body + duplicated allocs ≤ target local memory | invariant 5, checked by M4 §3.3 note 4 against `L1_BYTES = 65536` (`_trace.py:100`) |
| 7 | no `air.disable_ping_pong` attribute | M5 sets no attributes at all |
| 8 | `omit-memory-space` not excluding the alloc's space | a driver flag (M6); `aircc` defaults to `""` (`aircc.cpp:966-980`) |

So M5's obligation is exactly one negative: **emit the alloc where the plan puts it and nowhere
else.** The shape was measured to fire on our own W1 — `03-lld-B-open-questions.md` §1 evidence 4,
`{unroll = 2 : i32}` on the K loop with two `hoist_alloc = true` — and independently in VF §E.5.

`test_E3_alloc_is_direct_child` is a regex over the emitted text asserting the `memref.alloc`
lines sit at the indentation directly inside the K `scf.for`; `test_I_pingpong_labels`
(`04-test-plan.md` §3.2) asserts the post-pass fact.

### 3.6 Compute-node emission (FR-E2)

A `StoreNode` (`06-interfaces.md` §5.5) is one store of one `ExprNode` tree into one buffer; a
fully-integer subscript is a rank-0 access, which is what emits the
`memref.load`/`arith`/`memref.store` triple (`_value.py:590-593`, VF §D.11). M5 wraps them in the
`LoopPlan` nest the plan gives, and walks the tree structurally — **no case in this walk looks at
the kernel**:

```pseudo
 1  function EMIT_STORE(node, env):
 2      subs := [ EVAL_EXPR(s, env) for s in node.subscripts ]    # every entry an index expression
 3      buf  := env.buffers[node.buffer_id]                       # lookup, never derivation
 4      buf[subs] = EMIT_EXPR(node.expr, env)
 5      # A fully-integer subscript is a rank-0 access: no loop of the DSL's own,
 6      # just memref.load / arith / memref.store.  _emit.py:647-650.
 7
 8  function EMIT_EXPR(e, env):                                   # one branch per ExprNode case
 9      match e:
10        Load(b, subs)        -> env.buffers[b][[EVAL_EXPR(s, env) for s in subs]]
11        Const(v, text, dt)   -> the literal spelled `text`, never a float round-trip
12        BinOp(op, l, r)      -> EMIT_EXPR(l) <op> EMIT_EXPR(r)    # Python operator on air values
13        Neg(x)               -> - EMIT_EXPR(x)
14        MaxMin("maximum", a) -> fold air.ops.maximum over EMIT_EXPR(a_i), left to right
15        MaxMin("minimum", a) -> fold air.ops.minimum   (ops.py:709, finding N-9)
16        Select(cmp, l, r, t, f)
17                             -> air.ops.select(EMIT_EXPR(l) <cmp> EMIT_EXPR(r),
18                                               EMIT_EXPR(t), EMIT_EXPR(f))   # arith.select
```

The fold in lines 14-15 is why W3's 4-ary `max` needs no special case: `MaxMin("maximum", (a,b,c,d))`
becomes `maximum(a, maximum(b, maximum(c, d)))`, the element-level form measured in finding N-9.
`Select` is the only `ExprNode` that carries a comparison, and it is a value, not control flow
(`01-requirements.md` FR-D9).

The tree's own test is the shape to imitate — `python/test/api/partial_assign.py:85-92` writes
`dst[i, j, 0] = src[i + 3, j, 1] + 1` under two `air.sequential` loops, with FileCheck expecting
`scf.for` / `affine.apply` (`:79-84`), and accumulation into the same element works too
(`:120-127`). Our W1 probe produced exactly that shape:

```mlir
%6 = memref.load %alloc[%arg19, %arg20] : memref<32x32xf32, 2 : i32>
%7 = memref.load %alloc_27[%arg19, %arg21] : memref<32x16xf32, 2 : i32>
%8 = memref.load %alloc_28[%arg21, %arg20] : memref<16x32xf32, 2 : i32>
%9 = arith.mulf %7, %8 : f32
%10 = arith.addf %6, %9 : f32
memref.store %10, %alloc[%arg19, %arg20] : memref<32x32xf32, 2 : i32>
```

**Accumulator zeroing** is a `StoreNode` like any other, not a special case: M4 emits the zeroing
`LoopPlan` of `StoreNode(Const 0)` into the plan (`03-lld-M4-mapping.md` §6.1, herd-body lines 1-3)
and M5 translates it with row 12. It carries no `statement_index`, because no kernel statement
corresponds to it — which is precisely why `06-interfaces.md` v2 dropped that field. M5 does **not** call `ops.fill`. Two reasons, both D-14: `ops.fill` is a decision
about vectorisation the plan does not record, and a zeroing written as `StoreNode`s is what
`test_sem_compute_nodes` (`04-test-plan.md` §3.4) replays over numpy to close the oracle-to-plan
gap. The probe confirms the scalar form lowers: `memref.store %cst, %alloc[%arg18, %arg19]`.

**Whole-tile stores never reach M5.** The flip's `acc[:] = acc[:] + recv[:]` is expanded by M4
into an explicit two-deep `LoopPlan` over the tile carrying scalar `StoreNode`s
(`03-lld-M4-mapping.md` §6.2); M5 has no slice-assignment row and needs none.

### 3.7 Slice expressions and regions

A `ChannelSite` names a `buffer` and a `Region` (`06-interfaces.md` §5.2). M5 turns the pair into
the numpy-style subscript `air.api` expects; `air.api` then builds the strided memref region
itself (VF §D.3: *"regions are strided memref regions built from numpy-style subscripts"*).

```pseudo
 1  function SLICE(site, env):
 2      obj := env[site.buffer]
 3      if site.region is empty:           return obj                # the whole buffer
 4      subs := [ slice(EVAL_EXPR(off), EVAL_EXPR(off) + size) for (off, size) in zip(...) ]
 5      return obj[tuple(subs)]
```

M5 does not compute strides: they are row-major over the object's own shape and `air.api` derives
them. The plan's `Region.strides` exists so that M4's `test_sem_access_regions` can check the
index set and so the golden can be read; M5 **asserts** the plan's strides equal the row-major
ones and raises an internal-consistency error if not (§5).

### 3.8 `build`, text capture, and `ir_facts.json`

```pseudo
 1  module := launch.build(target)        # traces, replays into a func.func, verifies. VF §D.9
 2  text   := str(module)                 # == LaunchContext.mlir(), _compile.py:242
```

`build()` contains **no pass pipeline at all** (`grep -rn "PassManager" python/air/api/` → zero
hits, VF §D.9) and needs no device, no XRT and no `aircc` (A-6) — which is what makes the golden
tests runnable in CI on a stock CPU. `target` is always explicit, never `"auto"` (D-13):
`"auto"` shells out to `xrt-smi` (`_trace.py:182`) and a test must not spawn a subprocess.

**Golden convention** (`06-interfaces.md` §8): the text goes to
`tests/golden/<workload>.<variant>.<target>.air.mlir` and is compared **byte for byte**; the plan
goes to `*.plan.json` as canonical JSON; the summary to `*.summary.txt`. Four variants × two
targets = eight module goldens. They are regenerated only by `pytest --update-goldens`, and they
are valid only for the pinned wheel (FR-T6) — `test_T6_versions` runs first and a pin mismatch
skips every golden test with a reason rather than failing it.

**Two golden-text facts that were measured and will otherwise waste an afternoon:**

* `broadcast_shape` renders as `{broadcast_shape = [2 : index, 2 : index]}`, **not** `[2, 2]`
  (finding N-5). FR-M2's acceptance string as written never matches; the golden carries
  `air.channel @A2L1 [2, 1] {broadcast_shape = [2 : index, 2 : index]}`.
* A 1-D `air.herd` renders as a 2-D tile space with the second extent 1 —
  `air.herd @jacobi_herd tile (%a, %b) in (%x=%c4, %y=%c1)` (finding N-10) — while the body still
  takes **one** positional coordinate. Do not plan or expect a second.

**`ir_facts.json` extraction (D-8).** A second golden module would turn an upstream pass change
into a 400-line diff; a small fact file turns it into one changed number, which matters most for
the `air-broadcast-detection` count (R-04, H-8). M5 owns producing the fields; M6 owns running
`air-opt`.

| field | pipeline | extraction | expected for W1 |
|---|---|---|---|
| `pingpong_unroll` | VF §E.5's labelling prefix, ending `air-label-scf-for-to-ping-pong{device=npu1}` | the value of the single `unroll = N : i32` attribute | `2` |
| `hoist_alloc_count` | same | occurrences of `hoist_alloc = true` | `2` |
| `broadcast_pattern_count` | `air-dependency,air-broadcast-detection` | occurrences of `broadcast_pattern` | **`0`** — measured, finding N-8; confirms VF §S9 and D-4 |
| `cascade_channels` | `air-to-aie{device=npu1}` | occurrences of `aie.cascade_flow` | `0` for W1, **`3`** for W1-flip |
| `lock_init_histogram` | `air-to-aie{device=npu1}` | `aie.lock … {init = N}` counted by `N` | per VF §C.7's shape; **no producer lock may be `init = 0`** |

`cascade_channels` is also the honest replacement for FR-E8's `test_E8_cascade_text`, which
expects `channel_type = "npu_cascade"` three times: one bundle of `size=(PK-1,)` prints the
attribute once but lowers to three `aie.cascade_flow` ops, which is the fact that matters
(`03-lld-B-open-questions.md` §4).

---

## 4. Invariants, preconditions, postconditions

**Preconditions on `m5.emit(plan, target)`** — all are M4's postconditions, asserted and never
re-derived: the five plan invariants of `06-interfaces.md` §5.6, plus `target != "auto"` and
`target` in `{"npu1","npu2"}` (`_trace.py:172-177`).

**Postconditions**: `module.operation.verify()` succeeded (`build()` runs it,
`_compile.py:156-158`); `EmitResult.mlir` is non-empty; emitting the same plan twice gives
byte-identical text.

| # | Module invariant |
|---|---|
| P-1 | M5 contains **no branch on a `LegalMapping` or `ScheduleModel` field** and no arithmetic on kernel sizes (D-14; `test_emitter_makes_no_decisions`). Stronger, after `06-interfaces.md` v2: **M5 reads no field of `LegalMapping` or `KernelModel` at all** — names, buffers, expressions and the tensor order all reach it through `MappingPlan` |
| P-2 | M5 emits plan nodes in plan order and never re-orders, groups, hoists or deduplicates |
| P-3 | M5 never reconciles `BufferPlan.loop_depth` with IR nesting depth (finding N-7) |
| P-4 | Every `air.api` construct M5 names appears in §3.2's table |
| P-5 | M5 never sets an MLIR attribute (so conditions 1 and 7 of §3.5 hold trivially) |
| P-6 | M5 never passes `buffer_resources`, `packet_ids`, `pad_before` or `pad_after` (FR-E4; `air.api` raises, `_channel.py:553-570`) |
| P-7 | M5 passes `target` straight through to `launch.build(target=)` and never resolves `"auto"` itself |
| P-8 | Every guard is `air.ops.branch`; the substring `if tx ==` does not occur in M5's source (`test_M5_uses_branch`) |

---

## 5. Error paths

M5 raises `EmissionError` and nothing else (`06-interfaces.md` §6.2, NFR-7).

| Code | Raised when | `details` |
|---|---|---|
| `EMIT-AIR-API` | `air.api` raised — a channel-geometry rejection, an L1 overflow, an unsupported keyword, a herd rank or `shape=` divisibility failure | `air_api_message` carries the **original text verbatim** |
| `EMIT-VERIFY` | `module.operation.verify()` failed inside `build()` | the MLIR diagnostic, the kernel name, and the plan node most recently emitted |

**Preserve `air.api`'s own text, do not paraphrase it.** HLD §4.3 requires that no raw MLIR
diagnostic or Python traceback reach the top-level message, and that `air.api`'s message be kept
in `details["air_api_message"]` (NFR-4). `air.api`'s messages are generally better than anything
M5 could synthesise — `_compile.py`'s `_annotate_l1_failure` already explains an L1 overflow in
terms of tile size, and `_channel.py:127-131` spells out the fan-out idiom in its error. M5's own
`reason` names the plan node and the workload; the `fix` points at the clause M4's plan attributes
that node to.

**Internal-consistency failures that must never reach the user** (HLD §4.3). Everything M5 checks
about its input is something M4 guaranteed. A violation is our bug, not the user's, and the
message must say so rather than inventing a clause edit:

* a `ChannelSite.indices` entry resolving to an `scf.for` IV — M4's `BUNDLE-INDEX-IS-IV` should
  have caught it (§3.4);
* a `ping_pong_candidate` buffer that is not position-0 in a `LoopPlan.body` — invariant 4;
* a `Region` whose strides are not row-major over the object's shape (§3.7);
* a `BufferPlan.scope == "herd.shared"` — invariant I-6, and `air.api` would raise anyway
  (`_trace.py:1405-1414`);
* `EmitResult.l1_peak` disagreeing with the plan's L1 total by more than the ping-pong doubling.

Each is raised as an `EmissionError` whose `reason` names the plan field, the two values and the
document section that defines the agreement, and whose `fix` asks for a bug report. Nothing is
swallowed: `test_E9_verify_surfaced` feeds a deliberately broken plan (a segment coordinate used
inside an `IsolatedFromAbove` herd body) and asserts an `EmissionError`, not a bare MLIR
exception.

---

## 6. Worked examples — the exact ordered call sequence

Indentation is nesting. Every call's arguments come from the plan fields named in
`03-lld-M4-mapping.md` §6; nothing below is computed by M5.

### 6.1 W1 — GEMM output-stationary (`03-lld-M4-mapping.md` §6.1)

```pseudo
 1  A := air.tensor([64, 64], f32)
 2  B := air.tensor([64, 64], f32)
 3  C := air.tensor([64, 64], f32)
 4  cA := air.channel("A2L1", size=[2, 1], broadcast_shape=[2, 2])
 5  cB := air.channel("B2L1", size=[1, 2], broadcast_shape=[2, 2])
 6  cC := air.channel("C2L3", size=[2, 2])
 7  with air.launch(name=plan.launch_name) as launch:          # "gemm"; M5 derives no name
 8    @launch.body def _():
 9      with air.segment(name="gemm_seg") as seg:
10        @seg.body def _():
11          for pi in range(2):                        # PYTHON loop  (bundle index, D-3)
12            for kk in air.sequential(0, 64, 16):     # air.sequential at SEGMENT scope
13              cA.put(A[pi*32 : pi*32+32, kk : kk+16], indices=[pi, 0])
14          for pj in range(2):                        # PYTHON loop
15            for kk in air.sequential(0, 64, 16):
16              cB.put(B[kk : kk+16, pj*32 : pj*32+32], indices=[0, pj])
17          with air.herd([range(2), range(2)], name="gemm_herd", shape=(1, 2)) as h:
18            @h.body def _(tx, ty):                   # coords bound here (§3.4 b)
19              acc := air.alloc([32, 32], f32, scope=h.private())
20              for m in air.sequential(32):
21                for n in air.sequential(32):
22                  acc[m, n] = 0.0                    # StoreNode: memref.store of arith.constant
23              for kk in air.sequential(0, 64, 16):   # the PING-PONG CANDIDATE loop
24                a := air.alloc([32, 16], f32, scope=h.private())   # DIRECT child
25                b := air.alloc([16, 32], f32, scope=h.private())   # DIRECT child
26                cA.get(a, indices=[tx, ty])          # first touch of a  -> definite write
27                cB.get(b, indices=[tx, ty])          # first touch of b
28                for m in air.sequential(32):
29                  for n in air.sequential(32):
30                    for t in air.sequential(16):
31                      acc[m, n] = acc[m, n] + a[m, t] * b[t, n]
32              cC.put(acc, indices=[tx, ty])
33          for i in range(2):                         # PYTHON loop
34            for j in range(2):
35              cC.get(C[i*32 : i*32+32, j*32 : j*32+32], indices=[i, j])
36  module := launch.build("npu1");  text := str(module)
```

Verified end to end as an upstream-API probe (`$PROBE/q7_a.py`): builds and verifies, `aircc
--device npu1 --output-format=none` exits 0 with no `error:` line, and the labelling pipeline puts
`{unroll = 2 : i32}` on line 23's loop with `hoist_alloc = true` on lines 24 and 25.

**Golden head** (the first six lines of `w1.base.npu1.air.mlir`):

```mlir
module {
  air.channel @A2L1 [2, 1] {broadcast_shape = [2 : index, 2 : index]}
  air.channel @B2L1 [1, 2] {broadcast_shape = [2 : index, 2 : index]}
  air.channel @C2L3 [2, 2]
  func.func @gemm(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
```

L1 memrefs carry `2 : i32` as their memory space (`memref<32x16xf32, 2 : i32>`), which is
`test_E6_memory_spaces`.

### 6.2 W1-flip — cascade reduction (`03-lld-M4-mapping.md` §6.2)

`grid(PK=4)`, `place(px=ax.k0)` — a **1-D** herd, one coordinate `tx`, `TM=32`, `TK=16`, and
**`j` untiled** so `TN = N = 64` (RULING 9): `B`'s per-PE tile is `[16,64]` and is fetched once,
`A`'s `[32,16]` tile is fetched on each of the two `i0` trips.

```pseudo
 1  A := air.tensor([64, 64], f32);  B := air.tensor([64, 64], f32);  C := air.tensor([64, 64], f32)
 2  cA   := air.channel("A2L1", size=[4])
 3  cB   := air.channel("B2L1", size=[4])
 4  casc := air.channel("CascadeK", size=[3], channel_type="npu_cascade")     # NO broadcast_shape
 5  cC   := air.channel("C2L3", size=[1])
 6  with air.launch(name=plan.launch_name) as launch:                # "gemm_ws"
 7    @launch.body def _():
 8      with air.segment(name=plan.segment_name) as seg:             # "gemm_ws_seg"
 9        @seg.body def _():
10          for pk in range(4):                                      # PYTHON loop: bundle index
11            cB.put(B[pk*16 : pk*16+16, 0:64], indices=[pk])        # the WEIGHTS: once, hoisted
12          for pk in range(4):                                      # PYTHON loop: bundle index
13            for i0 in air.sequential(0, 64, 32):                   # A streams: 2 trips
14              cA.put(A[i0 : i0+32, pk*16 : pk*16+16], indices=[pk])
15          with air.herd([range(4)], name="gemm_ws_herd", shape=(4,)) as h:
16            @h.body def _(tx):                                     # ONE coordinate (finding N-10)
17              b    := air.alloc([16, 64], f32, scope=h.private())  # resident for the whole run
18              acc  := air.alloc([32, 64], f32, scope=h.private())
19              recv := air.alloc([32, 64], f32, scope=h.private())
20              cB.get(b, indices=[tx])                              # ONE get, outside every loop
21              for i0 in air.sequential(0, 64, 32):
22                a := air.alloc([32, 16], f32, scope=h.private())   # DIRECT child (ping-pong)
23                cA.get(a, indices=[tx])
24                <zeroing StoreNodes over acc>
25                <the m/n/t nest: acc[m,n] = acc[m,n] + a[m,t]*b[t,n]>   # n runs 0..64
26                with air.ops.branch(tx == 0) as head:      # ASCENDING chain (measured, P-R3)
27                  casc.put(acc, indices=[tx])
28                with head.otherwise():
29                  casc.get(recv, indices=[tx - 1])
30                  for m in air.sequential(0, 32):          # M4 expanded acc[:] += recv[:]
31                    for n in air.sequential(0, 64):
32                      acc[m, n] = acc[m, n] + recv[m, n]
33                  with air.ops.branch(tx == 3) as tail:
34                    cC.put(acc, indices=[0])
35                  with tail.otherwise():
36                    casc.put(acc, indices=[tx])
37          for i0 in range(0, 64, 32):                              # PYTHON loop: drain, unrolled
38            cC.get(C[i0 : i0+32, 0:64], indices=[0])
39  module := launch.build("npu1")
```

Lines 26-36 are the nesting `_cond.py:57-60` requires — there is no `and`, so conjunction is
nesting — and are the shape of `cascade_reduction.py:87-99`. **Measured (REVIEW-round1 P-R3)**:
on a 1-D `grid(4)` herd this **ascending** form gives `aircc` exit 0 and
`aie.cascade_flow(%tile_0_2, %tile_1_2)`, `(1,2)→(2,2)`, `(2,2)→(3,2)`; the descending form of
the same module fails with `'aie.cascade_flow' op source tile must be to the North or West of the
destination tile`. On a **2-D** `(1,4)` herd — the earlier `$PROBE/flip2.py` shape — the
opposite holds. That is why the direction is `ChannelPlan.chain_direction`, a plan field M5
reads, never an emitter choice: M5 has no rule that could pick it and D-14 forbids it trying.

Lines 17-20 against lines 21-23 are the flip's whole claim in IR: `b` is allocated and filled
**outside** the `i0` loop and never touched again, `a` is allocated **inside** it and re-filled
every trip. M5 emits that split because the plan's `BufferPlan.loop_depth` says so (`b` 0, `a` 1)
— not from any rule of its own — and the summary's residency lines are the human-readable form
of the same two numbers (`03-lld-M4-mapping.md` §3.9). `a` is also the only
`ping_pong_candidate`: it is a direct child of the streaming loop and its first touch is a `get`
(VF §E.2 items 2, 3, 5).

Lines 30-32 are the other half of D-14. The plan contains **no whole-tile store**: M4 expanded
`acc[:] = acc[:] + recv[:]` into this explicit `LoopPlan` nest of scalar `StoreNode`s
(`06-interfaces.md` §5.5), so M5 walks it with rows 7 and 12 and needs no slice-assignment rule.

### 6.3 W2 — Jacobi halo exchange (`03-lld-M4-mapping.md` §6.3)

`T = 4`, `H = W = 16`, `PI = 2`, `HS = 8`; `T` shown for both parities.

```pseudo
 1  U := air.tensor([5, 18, 16], f32)              # the KERNEL's one parameter: [T+1, H+2, W]
 2  uin  := air.channel("UIn",  size=[2])
 3  uout := air.channel("UOut", size=[2])
 4  tn   := air.channel("ToNorth", size=[1])        # PI-1 links  (open-Q §2)
 5  ts   := air.channel("ToSouth", size=[1])
 6  with air.launch(name=plan.launch_name) as launch:            # "jacobi"
 7    @launch.body def _():
 8      with air.segment(name=plan.segment_name) as seg:         # "jacobi_seg"
 9        @seg.body def _():
10          for p in range(2):                                   # PYTHON loop
11            uin.put(U[0, p*8 : p*8+10, 0:16], indices=[p])     # plane 0, strip + both ghosts
12          with air.herd([range(2)], name="jacobi_herd", shape=(2,)) as h:
13            @h.body def _(tx):                                 # ONE coordinate (finding N-10)
14              cur  := air.alloc([10, 16], f32, scope=h.private())  # OUTSIDE the t loop  (Q-6)
15              next := air.alloc([10, 16], f32, scope=h.private())
16              uin.get(cur, indices=[tx])
17              for t in air.sequential(0, T - (T % 2), 2):      # unroll by two
18                STEP(src=cur,  dst=next, t_off=0)              # expanded below
19                STEP(src=next, dst=cur,  t_off=1)
20              if T is odd:  STEP(src=cur, dst=next, t_off=0)   # PEELED tail, plan-space decision
21          for t in range(T):                                   # PYTHON loop over planes
22            for p in range(2):
23              uout.get(U[t+1, p*8+1 : p*8+9, 0:16], indices=[p])
24  module := launch.build("npu1")

STEP(src, dst, t_off) =
 s1  with air.ops.branch(tx > 0):      tn.put(src[1:2,  0:16], indices=[tx - 1])   # async
 s2  with air.ops.branch(tx < 1):      ts.put(src[8:9,  0:16], indices=[tx])       # async
 s3  with air.ops.branch(tx < 1):      tn.get(src[9:10, 0:16], indices=[tx])       # dependency=None
 s4  with air.ops.branch(tx > 0):      ts.get(src[0:1,  0:16], indices=[tx - 1])   # dependency=None
 s5  for i in air.sequential(1, 9):
 s6    for j in air.sequential(1, 15):
 s7      dst[i, j] = (src[i, j] + src[i-1, j] + src[i+1, j] + src[i, j-1] + src[i, j+1]) * 0.2
 s8  uout.put(dst[1:9, 0:16], indices=[tx])         # DRAIN, every timestep
```

Line 1 is the whole of B-1's fix: **one** tensor, the kernel's own parameter
`U: sp.f32[T+1, H+2, W]`, not a `U`/`Uout` pair of rank-2 tensors that appear nowhere in the
kernel. `MappingPlan.tensors` is the L3 interface and the L3 interface is `KernelModel.params`
(`06-interfaces.md` §5.6).

Line s7 is the **kernel's** update: five terms, `0.2 ×`, exactly
`03-lld-M1-frontend.md` §6.2's source. The earlier four-term `0.25 ×` form in this section was a
different program and is deleted.

Line s8 drains the computed strip **every timestep**, so lines 21-23 read back every plane the
kernel writes and the oracle can be compared over the whole write domain
(`04-test-plan.md` §3.4). A single drain after the loop would cover one plane out of `T`.

Line 20's `if` is **not** a branch in M5's source — it is the plan already containing or not
containing the peeled `STEP` nodes (`03-lld-M4-mapping.md` §3.6.1 line 8). M5 walks whatever is
in `plan.herd_body`, and with the per-`STEP` drain there is no `<live>` buffer to choose either.

s1-s4 are the FR-M4 order, `is_async=True` on both puts and `dependency=None` on both gets. The
probe shows what that lowers to:

```mlir
scf.if %3 {
  %18 = affine.apply #map1()[%arg10]
  air.channel.put  @ToNorth[%18] (%alloc[1, 0] [1, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
} else {
}
```

The empty `else` is expected and harmless — `_cond.py:61-66`: *"MLIR's `RemoveEmptyElseBranch`
canonicalization deletes it, and `canonicalize` runs ahead of everything in the AIR pipeline that
reads the region structure."*

**Per-core endpoints**: inbound `{UIn, one halo get}` = 2, outbound `{UOut, one halo put}` = 2 —
exactly the 2-S2MM / 2-MM2S budget at `PI = 2`. At `PI = 4` the interior PE needs three inbound
and `aircc` fails with `'aie.connect' op … TileID(1, 2) targets same dst` (P-R2); the checker
rejects it first (`03-lld-M4-mapping.md` §3.8).

### 6.4 W3 — wavefront (`03-lld-M4-mapping.md` §6.4)

```pseudo
 1  q := air.tensor([32], i32);  r := air.tensor([32], i32);  S := air.tensor([33, 33], i32)
 2  # READ-ONLY TENSORS FIRST: _check_interface raises otherwise (_compile.py:226-240, P-R4)
 3  qin  := air.channel("QIn", size=[1], broadcast_shape=[4])   # q, whole, to every PE
 4  rin  := air.channel("RIn", size=[4])                        # r, one slice per PE
 5  win  := air.channel("WestIn",  size=[1])        # L3 -> L1   (three channels: finding N-2)
 6  wmid := air.channel("West",    size=[3])        # L1 -> L1
 7  eout := air.channel("EastOut", size=[1])        # L1 -> L3
 8  sout := air.channel("SOut",    size=[4])
 9  with air.launch(name=plan.launch_name) as launch:            # "sw"
10    @launch.body def _():
11      with air.segment(name=plan.segment_name) as seg:         # "sw_seg"
12        @seg.body def _():
13          qin.put(q[0:32], indices=[0])                        # one put, PJ-way fan-out
14          for p in range(4):                                   # PYTHON loop
15            rin.put(r[p*8 : p*8+8], indices=[p])
16          for i in air.sequential(1, 33):                      # segment-scope SOURCE
17            win.put(ZERO_COL[i : i+1], indices=[0])            # S[i, 0] == 0, the boundary
18          with air.herd([range(4)], name="sw_herd", shape=(4,)) as h:
19            @h.body def _(tx):
20              qb   := air.alloc([32], i32, scope=h.private())
21              rb   := air.alloc([8],  i32, scope=h.private())
22              prev := air.alloc([9],  i32, scope=h.private())
23              cur  := air.alloc([9],  i32, scope=h.private())
24              ein  := air.alloc([1],  i32, scope=h.private())
25              eob  := air.alloc([1],  i32, scope=h.private())
26              qin.get(qb, indices=[tx]);  rin.get(rb, indices=[tx])
27              <zeroing StoreNodes over prev>                   # the S row-0 boundary
28              for i in air.sequential(1, 33, 2):               # unroll by two for the prev/cur swap
29                ROW(p=prev, c=cur,  i_off=0)
30                ROW(p=cur,  c=prev, i_off=1)
31          for i in air.sequential(1, 33):                      # segment-scope DRAIN
32            eout.get(SINK[i : i+1], indices=[0])
33          for i in range(1, 33):                               # PYTHON loop over rows
34            for p in range(4):
35              sout.get(S[i, p*8+1 : p*8+9], indices=[p])
36  module := launch.build("npu1")

ROW(p, c, i_off) =
 r1  with air.ops.branch(tx == 0) as hd:   win.get(ein, indices=[0])
 r2  with hd.otherwise():                  wmid.get(ein, indices=[tx - 1])
 r3  c[0] = ein[0]
 r4  for j in air.sequential(1, 9):
 r5    c[j] = air.ops.maximum(0,
 r6            air.ops.maximum(p[j-1] + air.ops.select(qb[i + i_off - 1] == rb[j-1], 2, -1),
 r7            air.ops.maximum(p[j] - 1, c[j-1] - 1)))
 r8  eob[0] = c[8]
 r9  with air.ops.branch(tx == 3) as tl:   eout.put(eob, indices=[0])
 r10 with tl.otherwise():                  wmid.put(eob, indices=[tx])
 r11 sout.put(c[1:9], indices=[tx])        # DRAIN, every row
```

Lines 1-2 are B-13's fix: the kernel's own three parameters, **reads before writes**. The measured
failure of the other order is verbatim `RuntimeError: output tensors must be declared after all
input tensors; the interface order is [('Zrow','in'),('Sink','out'),('Out','out'),('Q','in'),('Rr','in')]`
(P-R4) — and the earlier `Zrow`/`Sink`/`Out` triple in this section was exactly that program,
which is not W3's kernel at all.

Lines r5-r7 are the fold of `MaxMin("maximum", (0, …, …, …))` from §3.6, and line r6 is the
`Select` that makes this Smith-Waterman: the substitution score is `MATCH` or `MISMATCH`
depending on `qb[i-1] == rb[j-1]`, not the literal `+2` the earlier draft carried. `rb` is
indexed `j-1` because the buffer already holds this PE's slice of `r`. `GAP = 1` is a named
module constant resolved by the frontend (FR-S3 item 7), printed here as the literal it became.

Line r11 drains **every row**, so lines 33-35 read back the whole of `S`'s write domain.

Line 28's IV `i` is used only as a *row* index, never as a bundle index — the bundle index is `tx`
or `tx - 1` (H-3). Verified as a probe (`$PROBE/w3b.py`, with `q`/`r` staging added): `aircc
--device npu1 --output-format=none` exit 0, zero `error:` lines, three `aie.packet_flow` in the
lowered design (P-R4). The single-bundle form of HLD §7.3 (`West` of extent `PJ+1` carrying both
L3 ends) instead fails with `'airrt.dma_memcpy_nd' op failed to specialize channel bundle indices`
(`AIRLoweringPass.cpp:798`), which is finding N-2 and why lines 5-7 are three channels.

---

## 7. Unit tests

| id | input | expected |
|---|---|---|
| `test_E1_hierarchy` | W1 plan | text contains `air.launch`, `air.segment`, `air.herd` in that nesting; `module.operation.verify()` succeeds |
| `test_E1_herd_arity` | W2 plan (1-D herd) | the herd body takes one positional coordinate; the emitted tile space is 2-D with the second extent 1 (finding N-10) |
| `test_E2_native_body` | W1 | ≥ 3 nested `scf.for` inside the herd body; `memref.load` and `memref.store` present; **zero** occurrences of `func.call` and `link_with` |
| `test_E2_zero_is_compute_nodes` | W1 | the zeroing appears as `memref.store` of an `arith.constant`, and the emitter source never names `ops.fill` |
| `test_E3_alloc_is_direct_child` | W1 | regex: the two `memref.alloc` lines sit at the indentation directly inside the K `scf.for` |
| `test_E3_no_hoist` | W1 plan with `a` moved to plan depth 0 | the alloc is emitted **above** the K loop — M5 followed the plan, and M4's invariant 4 is what rejects such a plan |
| `test_no_buffer_resources_arg` | M5 source | the substring `buffer_resources` does not occur (FR-E4; `_channel.py:563` raises) |
| `test_E5_broadcast_emitted` | W1 | text contains `air.channel @A2L1 [2, 1] {broadcast_shape = [2 : index, 2 : index]}` (finding N-5) |
| `test_E6_memory_spaces` | W1 | every L1 memref carries `2 : i32`; no L2 memref (`1 : i32`) appears; no `<herd>.shared()` call in M5's source |
| `test_E7_text_roundtrip` | W1 | the written file re-parses through `air-opt` with exit 0 **and** no `error:` on stderr (FR-T5: exit code alone is not a verdict) |
| `test_E8_cascade_text` | W1-flip | text contains `air.channel @CascadeK [3] {channel_type = "npu_cascade"}` and **no** `broadcast_shape` on it; `ir_facts.cascade_channels == 3` |
| `test_E9_verify_surfaced` | a plan using a segment coordinate inside the `IsolatedFromAbove` herd body | `EmissionError` with `details["air_api_message"]`, not a bare MLIR exception |
| `test_E9_air_api_message_preserved` | a plan whose L1 total exceeds 65 536 | `EMIT-AIR-API`; `details["air_api_message"]` contains `air.api`'s own tile-size annotation verbatim |
| `test_E10_byte_identical` | W1, twice in one process and once in a fresh one with a different `PYTHONHASHSEED` | three identical texts |
| `test_M5_uses_branch` | W3 | text contains `scf.if`; M5's source contains no `if tx ==` construct (P-8) |
| `test_emitter_makes_no_decisions` | M5 source | no branch on a `LegalMapping` or `ScheduleModel` field; no arithmetic on kernel sizes (D-14, P-1) |
| `test_emitter_construct_closure` | M5 source | every `air.` name used appears in §3.2's translation table (P-4) |
| `test_E_target_passthrough` | W1 with `target="npu2"` | `build(target=)` receives `"npu2"` unchanged; `"auto"` is never resolved inside M5 (P-7, D-13) |
| `test_E_peel_is_plan_driven` | W2 plans with `T=4` and `T=5` | `T=5`'s text has the loop `step 2` to `4` plus one straight-line `STEP` after it, with `T` `UOut` puts in total; M5's source contains no parity test |
| `test_sequential_emits_scf_for` (FR-S14) | W1, W2, W3 | one `scf.for` per `LoopPlan(kind="sequential")`, in plan order, and **no** `scf.for` for a `LoopPlan(kind="unrolled")`; the count matches the plan's sequential-loop count exactly (`_loop.py:89`) |
| `test_pipeline_is_hint` (FR-S14) | two W1 schedules differing only in `pipeline(ax.k0)` | byte-identical emitted text |
| `test_E_select_emitted` | W3 | the text contains `arith.select` inside the `j` loop, and no `scf.if` inside it — the substitution score is a value, not control flow |
| `test_E_branch_node` | W1-flip, W3 | every `BranchNode` in the plan produces one `scf.if`, with `h.otherwise()` emitted only when `otherwise != ()`; nested `BranchNode`s nest |
| `test_E_tensor_order` | W3 | `air.tensor` declarations appear in `plan.tensors` order (`q`, `r`, `S`) and `build()` does not raise `_check_interface`'s `RuntimeError` |
| **Golden AIR-text tests** | | |
| `test_golden_w1[npu1|npu2]` | W1 | byte-for-byte against `tests/golden/w1.base.<target>.air.mlir` |
| `test_golden_w1_flip[npu1|npu2]` | W1-flip | byte-for-byte against `w1.flip.<target>.air.mlir` |
| `test_golden_w2[npu1|npu2]` | W2 | byte-for-byte against `w2.base.<target>.air.mlir` |
| `test_golden_w3[npu1|npu2]` | W3 | byte-for-byte against `w3.base.<target>.air.mlir` |
| `test_golden_skips_on_pin_mismatch` | a faked version | every golden test **skips with a reason**, never fails (`06-interfaces.md` §8 rule 2) |

Levels U and G (`04-test-plan.md` §1); all run on CPU with `import air` and no device (A-6).

---

## 8. Dependencies

**Consumed**: `MappingPlan` (`06-interfaces.md` §5.6) from M4, with B's own hand-written W1 plan
literal as the D0/D1 stub; the `model` dataclasses from M0; `air.api` from the pinned wheel
`mlir_air[aie] == 0.0.1.2026091204+ff95a9b` (FR-T6, verified in the probe environment).

**Every `air.api` construct M5 emits**, with its evidence:

| construct | VF § / `path:line` | used by |
|---|---|---|
| `air.tensor(shape, dtype)` | VF §D.2; `_trace.py:1803` | row 1 |
| `air.launch(name=)` | VF §D.1; `_compile.py:44` | row 2 |
| `air.segment(name=)` | VF §D.1; `_trace.py:1201` | row 3 |
| `air.herd(iterable, name=, shape=)` | VF §D.1; `_trace.py:1760`, rank cap `:1288-1291`, physical caps `:88-91`, `shape=` divisibility `:1363-1368` | row 4 |
| `air.herd(at=)` | VF §D.1; `_trace.py:1771-1778` | row 4, currently never passed |
| `air.alloc(shape, dtype, scope=<herd>.private())` | VF §D.2; `_trace.py:1845`; `<herd>.shared()` raises at `:1405-1414` | row 6 |
| `air.channel(name, size=)` | VF §D.4; `_channel.py:581` | row 5a |
| `air.channel(..., broadcast_shape=)` | VF §A.4, §D.4; `_channel.py:120-145`; `broadcast/single_herd/broadcast.py:44`; `broadcast_selective_capture.py:56-57` | row 5b |
| `air.channel(..., channel_type="npu_cascade")` | VF §D.4; `_channel.py:547` (`_IMPLEMENTED_TYPES`), `:170-177` (no `broadcast_shape`); `cascade_reduction.py:60-62` | row 5c |
| `Channel.put(obj, indices=, dependency=)` | VF §D.4; `_channel.py:511` | row 9 |
| `Channel.get(obj, indices=, dependency=)` | VF §D.4; `_channel.py:524` | row 10 |
| `air.sequential(start, stop, step)` | VF §D.5; `_loop.py:89`; no `iter_args` anywhere, `:180`; coordinate-derived bound refused, `:136-146` | row 7 |
| a Python `for` over `range` | VF §D.5, §D.10; `_loop.py:14-19` | row 8 |
| `air.ops.branch(c)` / `head.otherwise()` | VF §D.6; `_cond.py:18-28`, `:41-45`, `:49-54`, `:57-60`, empty-else note `:61-66` | row 11 |
| buffer element assignment / whole-tile `b[:] = …` | VF §D.11, §S1; `_value.py:590-593`; `_emit.py:647-650`, `:601`; `partial_assign.py:79-92`, `:120-127` | row 12 |
| `air.ops.maximum` at element level | finding N-9 (probe `$PROBE/w3b.py` builds and compiles) | row 12, W3 |
| `LaunchContext.build(target=)` | VF §D.9; `_compile.py:107`, `:128-158`; targets validated at `_trace.py:172-177` | row 13 |
| `LaunchContext.mlir()` / `str(module)` | `_compile.py:242` | row 13 |
| `LaunchContext._l1_peak` | `_compile.py:74`, `:150` | `EmitResult.l1_peak` |

**Constructs deliberately not emitted**, each with its reason, so nobody adds one later without
re-reading this row:

| construct | why not |
|---|---|
| `buffer_resources=` | `air.api` raises (`_channel.py:563`); the depth is realised anyway as the lock slot count the ping-pong unroll sets (VF §E.3, §E.5 — "8 locks with `init = 2`"). FR-E4 |
| `air.extern` / `link_with` / `func.call` | FR-E2; an opaque callee as a buffer's first toucher **disqualifies ping-pong** (VF §E.2 items 3-4); an extern kernel buys vectorisation, not expressiveness (`_extern.py:11-14`) |
| `ops.dot` | it lowers through `convert-linalg-to-loops` and comes out scalar anyway; `StoreNode` is the contract and is what `test_sem_compute_nodes` replays |
| `ops.load` / `ops.store` | they emit `air.dma_memcpy_nd`, which `air-broadcast-detection` **does** walk (`AIRDependencyScheduleOpt.cpp:3328`). Declaring `broadcast_shape` on an explicit channel bypasses the detector entirely (D-4, FR-E5, VF §E.4, §S9) and keeps R-04 pinned at a golden count of 0 |
| `ops.fill` | a vectorisation decision the plan does not record (§3.6) |
| `air.parallel` | no plan needs an `scf.forall` |
| `<segment>.shared()` / `<segment>.per_core()` | `03-lld-B-open-questions.md` §3 |
| `pad_before` / `pad_after` / `packet_ids` / `dest=` | unsupported on channels, or an unused knob (VF §D.4, §S6) |
| `target="auto"` | D-13: it shells out to `xrt-smi` (`_trace.py:182`) |

**Stubs consumed**: B's own hand-written W1 `MappingPlan` literal (D0), which is what makes M5's
D1 skeleton independent of M3 and M4 both. **Stubs produced**: the first AIR text file, handed to
Person C at D1 so M6's `aircc` wrapper and M7's golden helper have something real to run.

---

## 9. Implementation order, effort, definition of done

Effort is **estimated**. The WBS budgets M5 at 1.0 person-day (`05-work-breakdown.md` §7) and this
LLD does not change that: every shape M5 must produce has now been built by hand as an upstream
probe, so the emitter is transcription against a known-good target.

| Step | What | Day | Effort (est.) | Unblocks |
|---|---|---|---|---|
| 1 | Skeleton: rows 1-4, 5a, 6, 7, 8, 13 of §3.2 driving B's hand-written W1 plan; `build("npu1")` succeeds and `str(module)` is non-empty | D1 | 0.35 pd | C's `aircc` wrapper, M7's golden helper |
| 2 | Rows 5b, 9, 10, 12 — broadcast channels, sites, compute nodes; wire to M4's real W1 plan (gate **G2**) | D2 | 0.25 pd | the W1 golden |
| 3 | Row 11 — `ops.branch` guards, including nesting for conjunction (W3, gate **G3**) | D4 | 0.15 pd | the W3 golden |
| 4 | The W2 body: async puts, `dependency=None` gets, the peeled tail walked from the plan (gate **G4**) | D5 | 0.1 pd | the W2 golden |
| 5 | Row 5c — cascade (gate **G5**, stretch) | D6 am | 0.05 pd | the flip golden |
| 6 | §3.8 `ir_facts.json` field extraction, jointly with C's `air-opt` runner | D3, D6 | 0.1 pd | the IR-fact tests |
| | **total** | | **≈ 1.0 pd** | |

Steps 3, 4 and 5 are small precisely because M4 carries the decisions: M5 gains one `with`
statement, one keyword argument and one keyword argument respectively.

**Definition of done for M5.** All ten, checked at the D6 freeze:

1. FR-E1…FR-E10 each have their acceptance test from §7 passing.
2. All four variants emit text that `module.operation.verify()` accepts, for both `npu1` and
   `npu2`.
3. All eight module goldens match byte for byte, and a pin mismatch skips rather than fails.
4. All four variants pass `aircc --device <target> --output-format=none` with exit 0 **and** no
   `error:` line on stderr, for both targets (`04-test-plan.md` §8 item 4).
5. `ir_facts.json` for W1 records `pingpong_unroll == 2`, `hoist_alloc_count == 2` and
   `broadcast_pattern_count == 0`; for W1-flip, `cascade_channels == 3`.
6. `test_emitter_makes_no_decisions` and `test_emitter_construct_closure` both pass — D-14 is
   enforced mechanically, not by intention.
7. `test_E10_byte_identical` passes across two `PYTHONHASHSEED` values.
8. Every `EmissionError` carries `air.api`'s original text in `details["air_api_message"]`, and no
   raw MLIR diagnostic or traceback reaches a top-level message.
9. `buffer_resources`, `func.call`, `link_with` and `<herd>.shared()` do not occur in any emitted
   text or in M5's source.
10. Every construct in §8's first table is used and cited; every construct in the second table is
    absent and has its reason.

---

## 10. Open questions M5 owns

| # | Question | Due | Fallback |
|---|---|---|---|
| **B-O4** | Does everything proved on `npu1` hold for `--device npu2`? The caps are larger (`{1:(8,), 2:(2,4)}`, `_trace.py:88-91`) and the per-core DMA budget is not smaller, so `npu1` should be the binding case — but the eight goldens include four `npu2` modules and none has been built | D2 | Emit and `aircc` all four for `npu2` at D2 and record. If a shape fails, that target's goldens are marked `xfail` with the recorded reason and the demo quotes `npu1` only |
| **B-O8** | Is `str(module)` stable across a `--upgrade` *within* the pinned version? This is Q-3 (Person C's) seen from the emitter's side; it decides byte-for-byte versus FileCheck-style goldens | D1 (C answers) | Byte-for-byte for the pinned wheel, one `--update-goldens` switch, a CI guard on the pin — `06-interfaces.md` §8 rules 1-2 already say this |
| **B-O9** | Which `air-opt` pipeline prefix `ir_facts` extraction uses — VF §E.5's exact string, or the shorter `-air-dependency,-air-dma-to-channel,-air-label-scf-for-to-ping-pong`. This is Q-5 (Person C's); M5 only owns the field list | D2 (C answers) | VF §E.5's exact string, which is recorded as having worked and which our own W1 probe reproduced (`03-lld-B-open-questions.md` §1 evidence 4) |
| **B-O10** | Whether the emitted text should carry a leading comment banner naming the workload, variant and contract version. It would make a golden diff self-describing, but `air.api` gives no hook for a module-level comment and a post-hoc string prepend would break `air-opt` round-tripping (`test_E7_text_roundtrip`) | D6 | No banner. The golden's **filename** carries the workload, variant and target, which is `06-interfaces.md` §8's convention and is enough |
