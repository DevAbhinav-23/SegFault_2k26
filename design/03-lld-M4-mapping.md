# LLD — M4, mapping derivation and protocol synthesis

*Phase 1, 2026-09-12. Owner: **Person B**. Reads on top of [`02-hld.md`](02-hld.md) §2, §7 and
[`06-interfaces.md`](06-interfaces.md) §5. Every open question this module owned is resolved in
[`03-lld-B-open-questions.md`](03-lld-B-open-questions.md); findings N-1…N-10 there are load-bearing
here and are cited by number. **No implementation exists.** Algorithms are numbered pseudocode;
signatures are referenced by `06-interfaces.md` section, never restated field by field.*

---

## 1. Purpose and requirements satisfied

M4 turns a `LegalMapping` — which says *what the schedule is* — into a `MappingPlan` — which says
*exactly what IR to build*. It is the module where every decision is taken, because D-14 makes M5
a mechanical translator: **if M5 would have to choose something, the plan is under-specified and
the choice belongs here.**

| FR | What M4 does for it | Where |
|---|---|---|
| FR-M1 | reuse trichotomy per operand from `ker M_a` and `Sπ` | §3.2 |
| FR-M2 | multicast → `air.channel(size=, broadcast_shape=)` geometry | §3.2, §3.5 |
| FR-M3 | a `stream`/`forward` clause overrides the derivation, marked `declared` | §3.2 step 8 |
| FR-M4 | halo protocol: put north, put south, get north ghost, get south ghost; async puts, token-free gets; no prologue | §3.6.1 |
| FR-M5 | wavefront: one scalar get per row, one scalar put per row, constant source, drain | §3.6.2 |
| FR-M6 | cascade chain for `R_space`, head/middle/tail, `npu_cascade` | §3.6.3 |
| FR-M7 | buffer plan: name, level, scope spelling, shape, dtype, bytes, loop depth | §3.3 |
| FR-M8 | channel plan: name, size, broadcast, type, ordered sites with regions, scope, async | §3.4, §3.5 |
| FR-M9 | balance by construction **and** the P1′ self-check by concrete enumeration | §3.7.1 |
| FR-M10 | acyclicity by construction **and** the P2b SCC self-check | §3.7.2 |
| FR-M11 / FR-D2 | the human-readable `MappingSummary` | §3.9 |
| FR-M12 | determinism: every name derived, every traversal sorted | §4, rule I-7 |
| — (new) | **P3 resource check** on per-core DMA channels, forced by finding N-1 | §3.8 |

Non-responsibilities, restated so they are not drifted into: M4 never imports `air`; it never
formats MLIR; it never chooses `(σ,π)` (the user does, PC §1.3); it never re-checks legality
(that is M3's, and a second opinion is a second place to be wrong).

---

## 2. Public entry points

Exactly the two in `06-interfaces.md` §7.2: `m4.plan(...)` and `m4.self_check(...)`. `plan` calls
`self_check` on its own result before returning, so a `MappingPlan` that leaves M4 has always been
checked; `self_check` is public only so the negative tests can feed it hand-corrupted plan
literals (`04-test-plan.md` §2, M4 bullet 2). Both raise `MappingError` and nothing else (NFR-7).

Everything M4 produces is defined in `06-interfaces.md` §5: `BufferPlan` §5.1, `ChannelSite` and
`Region`/`Guard` §5.2, `ChannelPlan` §5.3, `HerdPlan` §5.4, `LoopPlan`/`PlanNode`/`StoreNode`
§5.5, `MappingPlan` and its five invariants §5.6, `MappingSummary` §5.7.

---

## 3. Internal design

### 3.1 Shape of the module

Ten passes over the `LegalMapping`, in this fixed order. Each is a pure function; each is
separately unit-testable; the order is a topological sort of what needs what.

```pseudo
 1  function PLAN(mapping):                                   # m4.plan
 2      herd     := RESOLVE_HERD(mapping)                      # §3.3 step 1
 2a     tensors  := TENSOR_PLAN(mapping)                       # §3.3: the L3 interface, ordered
 3      delivery := CLASSIFY(mapping)                          # §3.2, in UCoord
 4      buffers  := BUFFER_PLAN(mapping, delivery, herd)       # §3.3, L1 only
 5      channels := CHANNEL_PLAN(mapping, delivery, herd, buffers)   # §3.4-3.5
 6      loops    := LOOP_PLAN(mapping, buffers)                # §3.5
 7      bodies   := PROTOCOL(mapping, delivery, herd, buffers, channels, loops)  # §3.6
 7a     #   bodies.segment_body carries `herd` itself as its one HerdPlan node, at top level:
 7b     #   that node IS the herd's position (`06-interfaces.md` §5.6 invariant 8, v4)
 8      plan     := MappingPlan(tensors=tensors, herd=herd, ...,
 8a                             launch_name  = mapping.kernel.name,
 8b                             segment_name = f"{mapping.kernel.name}_seg",
 8c                             summary      = SUMMARY(...))    # §3.9
 9      SELF_CHECK(plan)                                       # §3.7, §3.8
10      return plan
```

`PROTOCOL` is a dispatch on which clauses are present, not on which workload it is: `exchange`
selects §3.6.1, `forward` selects §3.6.2, a non-empty `mapping.r_space` selects §3.6.3, and any
combination is legal because the three write into disjoint parts of the body.

**Name derivation (HLD §5 rule 2), fixed here so it is never invented twice.**

| thing | name |
|---|---|
| L1 staging buffer for operand `a` | `a.lower()` — `a`, `b`, `acc`, `u`, `v` |
| L3 tensor for operand `a` | `a` as the user wrote it — `A`, `B`, `C`, `U` |
| channel L3 → L1 for operand `a` | `f"{a}2L1"` |
| channel L1 → L3 for operand `a` | `f"{a}2L3"` |
| halo channels for `exchange(a, along=ax)` | `ToNorth`, `ToSouth` (px axis); `ToWest`, `ToEast` (py axis) |
| forward channels for `forward(a, along=ax, dir=)` | `WestIn`, `West`, `EastOut` for `"W->E"`, mirrored per `Direction` |
| cascade channel for `R_space` | `f"Cascade{axis.upper()}"` — `CascadeK` |
| herd | `f"{kernel.name}_herd"` |
| launch | `kernel.name` — recorded as `MappingPlan.launch_name` |
| segment | `f"{kernel.name}_seg"` — recorded as `MappingPlan.segment_name` |
| `ChannelSite.id` | `f"{channel}.{kind}.{order}@{scope}"`, which `depends_on` references |

The last three rows exist because M5 must not derive a name (D-14): every string it emits is a
field it reads (`06-interfaces.md` §5.6 invariant 7, §5.2).

### 3.2 Reuse trichotomy (FR-M1, FR-M2, FR-M3)

The maths is SD-04 §3: for operand `a` with linear access part `M_a`, its reuse space is
`L_a = ker M_a`, and a direction `r` is **stationary** if `r ∈ L_a` and `Sπ·r = 0`, **multicast**
if `r ∈ L_a` and `Sπ·r ≠ 0`, **stream** if `r ∉ L_a` and `Sπ·r ≠ 0`. What the plan needs is one
label per operand, and the operational form of that is much simpler than the general form,
because the only spatial directions that exist are the placed axes.

```pseudo
 1  function CLASSIFY(mapping) -> tuple[(operand, Delivery, along, declared)]:
 2      out := []
 3      for a in sorted(mapping.kernel.params, key=name):         # I-7 determinism
 4          M := the AccessMap matrix of a, over mapping.kernel.axes     # UCoord, not Coord
 5          if a is the reduction target and mapping.r_space != {}:
 6              out.append((a, CASCADE, PE axis carrying r_space, declared?))   # §3.6.3
 7              continue
 8          bcast := []
 9          for d in 0 .. len(mapping.schedule.place) - 1:
10              axis_d := mapping.schedule.place[d]               # a post-tiling name, e.g. "i0"
11              if M · e[root(axis_d)] == 0:                      # root = the UNTILED parent, "i"
12                  bcast.append(d)
13          if bcast != []:
14              out.append((a, MULTICAST, PE_AXIS_NAME[bcast[0]], False))
15          elif contains(mapping.ker_pi_u, kernel_basis(M)):     # both in UCoord
16              out.append((a, STATIONARY, None, False))
17          else:
18              out.append((a, FORWARD, PE axis of the non-reuse spatial direction, False))
19      # `declared` is a fact about the SCHEDULE: it is True iff a clause names this operand's
19a     # delivery — stationary(a), stream(a,...), forward(a,...) or exchange(a,...) — whatever
19b     # row the derivation produced above.  (Ruling on B-P17, `06-interfaces.md` §5.6 at v5.)
19c     out := [(a, kind, along, a in DECLARED_OPERANDS(mapping.schedule)) for (a, kind, along, _) in out]
20      # FR-M3: a stream()/forward() clause additionally REPLACES the derived row.  exchange()
20a     # does not: `Delivery` has no halo member, and §6.3 prints W2's derived `U: STATIONARY`
20b     # beside its declared halo sentence.
21      for clause in sorted(mapping.schedule.streams, key=operand):
21a         replace the row for clause.operand with (operand, PATTERN[clause], clause.along, True)
22      return tuple(out)
```

`DECLARED_OPERANDS` is `set(schedule.stationary) | {c.operand for c in schedule.streams} |
{c.operand for c in schedule.exchanges}`. It says a clause **named** the delivery, not that the
derivation needed the clause: W1's `C` is `declared` although `ker M_C ⊆ ker Sπ_u` holds anyway,
and the flip's `B` is `declared` for the same reason — one rule, both worked examples.

**Every classification here is computed in `UCoord`** — the untiled `kernel.axes` frame.
`mapping.pi` and `mapping.ker_pi` are over `Coord` (post-tiling) and are used **only** for herd
geometry; `mapping.pi_u` and `mapping.ker_pi_u` are the untiled frame's `Sπ` and `ker Sπ`, and
M3 supplies both as fields (`06-interfaces.md` §4.1, added in v2 — REVIEW-round1 B-5), so M4
never re-derives `root(place)` itself. `root(x)` is `Axis.parent` when `x` is a tile handle and
`x` otherwise; a placed `i0` means the PE coordinate is a coarsening of `i`, so its untiled row
is `e_i` (`03-lld-M3-checker.md` §3.1, §3.3 line 20). Mixing the frames is how the earlier draft
came to evaluate `M · e["i0"]` against a matrix that has no `i0` column at all.

`PE_AXIS_NAME` is `("px", "py")`. Line 11 is the whole content of "multicast is the
spatial-and-constant-access cell": **operand `a` multicasts along PE axis `d` exactly when the
kernel axis placed on `d` is one `a` does not index.** For W1 (`place(px=i0, py=j0)`): `A[i,k]`
does not index `j0` → multicast along `py`; `B[k,j]` does not index `i0` → multicast along `px`;
`C[i,j]` indexes both → line 15 applies and, in `UCoord`, `ker M_C = span{e_k} ⊆ ker Sπ_u` →
stationary.
This reproduces FR-M1's acceptance and SD-04 §5's worked derivation exactly.

Line 14 takes `bcast[0]` deliberately: an operand constant along **both** PE axes would fan out to
the whole herd, and `air.channel` expresses that as `size=[1,1]` with the full
`broadcast_shape` (`broadcast/single_herd/broadcast.py:44`). None of W1/W2/W3 needs it; if a
future kernel does, line 14 becomes `bcast` entire and §3.5's geometry already generalises. The
case is not built now (YAGNI) but the shape is left open.

**Channel geometry for a multicast operand** (FR-M2), derived once and used by §3.5:

```pseudo
 1  function MULTICAST_GEOMETRY(herd_grid, bcast_axis d):
 2      size            := list(herd_grid);  size[d] := 1
 3      broadcast_shape := list(herd_grid)
 4      put_indices(p)  := the herd coordinate on every axis except d, and 0 on d
 5      get_indices     := the full herd coordinate tuple
 6      return (size, broadcast_shape, put_indices, get_indices)
```

Invariant `broadcast_shape[x] % size[x] == 0` holds by construction (it is `g % g = 0` off the
broadcast axis and `g % 1 = 0` on it), which is the rule `_channel.py:137-145` enforces.

### 3.3 Buffer plan (FR-M7)

```pseudo
 1  function RESOLVE_HERD(mapping) -> HerdPlan:
 2      return HerdPlan(name    = f"{kernel.name}_herd",
 3                      grid    = mapping.schedule.grid,          # the LOGICAL grid
 4                      shape   = mapping.physical_herd,          # from M3, per _trace.py:88-91
 5                      at      = None,                           # never pinned: open-questions §4
 6                      coords  = ("tx",) if rank == 1 else ("tx","ty"))
 7
 8  function BUFFER_PLAN(mapping, delivery, herd) -> tuple[BufferPlan]:
 9      out := []
10      for a in sorted(operands):
11          lvl := mapping.schedule.residency.get(a, "L1")
11a         if lvl not in ("L1", "L3"):
11b             raise MappingError(PROTOCOL-UNSUPPORTED), clause = f'reside({a}="{lvl}")',
11c                 reason "no segment-private staging protocol is synthesised in this cut"
11d         if lvl != "L1": continue                              # L3 is handled by TENSOR_PLAN
12          shape := TILE_SHAPE(a, mapping)                       # the per-PE slab, §3.3 note 1
13          out.append(BufferPlan(name        = lower(a),
14                                operand     = a,
15                                level       = "L1",
16                                scope       = "herd.private",   # ALWAYS: open-questions §3
17                                shape       = shape,
18                                dtype       = a.dtype,
19                                bytes       = prod(shape) * dtype.sizeof,
20                                loop_depth  = DEPTH(a, delivery, mapping),
21                                ping_pong_candidate = PP(a, delivery, mapping)))
22      out += PROTOCOL_BUFFERS(mapping)                          # u/v pair, edge_in/edge_out, recv
23      return tuple(out)                                      # allocation order, 06-interfaces §5.6
24
24a function TENSOR_PLAN(mapping) -> tuple[BufferPlan]:           # MappingPlan.tensors, FR-M7
24b     reads  := [ p for p in mapping.kernel.params if not p.is_written ]   # declaration order
24c     writes := [ p for p in mapping.kernel.params if     p.is_written ]   # declaration order
24d     # ALL read-only params BEFORE ALL written params: air.api's _check_interface raises
24e     # "output tensors must be declared after all input tensors" otherwise
24f     # (python/air/api/_compile.py:226-240, measured REVIEW-round1 P-R4).
24g     return tuple( BufferPlan(name = p.name, operand = p.name, level = "L3",
24h                              scope = "tensor",                # L3 => "tensor", §5.1 invariant
24i                              shape = resolved(p.shape), dtype = p.dtype,
24j                              bytes = prod(shape) * dtype.sizeof,
24k                              loop_depth = 0, ping_pong_candidate = False)
24l                   for p in reads + writes )
24m     # This is the ONLY place an L3 BufferPlan is created; BUFFER_PLAN never makes one.
24n     # A param that is both read and written (W2's U) counts as WRITTEN and sorts last.
25
25  function DEPTH(a, delivery, mapping) -> int:
26      # 1 iff the operand is streamed once per trip of the innermost temporal loop,
27      # 0 iff it is resident for the whole herd body.
28      return 1 if delivery[a] in {MULTICAST, FORWARD} and a is re-fetched per trip else 0
29
30  function PP(a, delivery, mapping) -> bool:
31      return a in mapping.schedule.double_buffer
32         and DEPTH(a, ...) >= 1                                  # VF §E.2 item 2
33         and the first site touching the buffer is a `get`       # VF §E.2 item 3
34         and exactly one get per iteration                       # VF §E.2 item 5
35         and every enclosing LoopPlan has constant bounds        # VF §E.2 item 5
```

*Note 1 — `TILE_SHAPE`.* The per-PE slab is the image of the PE's iteration subdomain under
`M_a`, which for the affine, rectangular domains the grammar admits is a product of tile extents:
`A → [TM, TK]`, `B → [TK, TN]`, `C → [TM, TN]`, `U → [HS + 2·halo, W]`. It is computed from the
`AccessMap` and the tile factors, never hard-coded per workload.

*Note 2 — `scope` is always `"herd.private"`.* Resolved in open-questions §3: the only reason to
want `"segment.per_core"` is to split the L3 staging into its own herd, and that split either
mis-places the herds or hits the same routing limit. `"herd.shared"` is never emitted at all —
`_trace.py:1405-1414` raises.

*Note 3 — `loop_depth` is plan-space, not IR-space.* Finding N-7: when `repeats > 1`, `air.api`
wraps the entire herd body in an `scf.for` over the repeat index, so a plan-depth-0 buffer is at
IR depth 1. This does not disturb `isPingPongCandidate`, whose test is *"a direct child of the
candidate loop"* — the candidate loop is the streaming loop, and the alloc is still its direct
child. M5 must not try to reconcile the two numbers (see `03-lld-M5-emitter.md` §4, P-3).

*Note 5 — `reside(x="L2")`.* Line 11a is the rejection G-5 asked for: before this, an L2
residency fell through line 11's `continue` and was **silently ignored**, so FR-S12's
`test_reside_maps_to_scope` could not pass. L2 residency is future work (RULING 7); the clause
is accepted by M2 (it is in the `Level` enumeration) and rejected here, by name.

*Note 6 — `tensors` ordering is a hard upstream law, not a convention.* `_check_interface`
raises a bare `RuntimeError` listing the interface order it saw
(`[('Zrow','in'),('Sink','out'),('Out','out'),('Q','in'),('Rr','in')]` in the measured case), so
a violation surfaces as an `EMIT-AIR-API` at trace time with an unhelpful message. `TENSOR_PLAN`
makes it unreachable, and `06-interfaces.md` §5.6 invariant 6 makes M4's self-check assert it.
W3 is the case that bites: `q`, `r` are read, `S` is written, and any other order fails.

*Note 4 — the L1 budget (invariant 5, `06-interfaces.md` §5.6).* `sum(b.bytes × (2 if
ping_pong_candidate else 1)) ≤ 63488` (65 536 B of tile data memory less the 2 048 B core
stack `air-to-aie` reserves — **R-L1-2**, `03-lld-M3-checker.md` §3.10). M3 has already
charged this figure in
`LegalMapping.l1_bytes`; M4 recomputes it from the plan and raises if the two disagree — an
internal-consistency failure, §5.

**Erratum, 2026-09-15 — architect ruling R-L1-3.** The factor is `2 if (repeated or
ping_pong_candidate) else 1`, where `repeated = any(r > 1 for r in mapping.repeats)`: when a
repeat loop exists the ping-pong machinery labels **it** rather than the streaming loop and
unrolls it by 2 (`AIRDependencyScheduleOpt.cpp:1906-1908`), so every buffer the herd body
allocates exists twice on the core and the ping-pong pair is not doubled again. `l1_total` takes
the flag, `_check_l1`, `m4_selfcheck.l1_budget` and `SUMMARY`'s `L1:` line all pass it, so M4
and M3 charge the same arithmetic — W1 base is 16 384 B on npu1 (`repeats (2,1)`) and 12 288 B
on npu2. The per-buffer lines keep their `x2 (ping-pong)` annotation, which is a property of the
buffer; the repeat loop is a property of the herd and is already on the `herd:` line.
`03-lld-M3-checker.md` §3.10 carries the measured buffer counts.

**Erratum, 2026-09-15 — degenerate grids (B-P35).** Two defects the worked examples never
reached, both fixed:

1. **`ChannelSite.order` at a bundle of extent 1.** §3.1's table says `order` is the site's index
   in its enclosing body. `_bundle_nest` builds **no** loop for a dim of extent 1, so at
   `grid(1)` or `grid(1, 1)` the drain `get` has no wrapper and sits in the segment body itself,
   where its index is `len(fills) + 1 + position` and not 0. Counting it as 0 made M4's own
   `_check_orders` refuse every single-PE schedule.
2. **An outer tile axis of extent 1 is not temporal** (§3.9 line 3's `T`). `tile(ax.j, N)` at the
   full extent makes one tile whose loop runs once, so no operand's slab moves with it and it is
   the same schedule as leaving `j` untiled — which §3.9 already excludes. Counting it made
   `B[k,j]` "re-fetched per `j0`" while `A[i,k]` moved with `k0`, and §6.1's shape synthesises
   **one** streaming loop. That genuinely unsupported case (two staged operands, two axes with
   trips) is now a `PROTOCOL-UNSUPPORTED` naming both axes and the clause to edit, rather than an
   internal `NotImplementedError` (NFR-7).

### 3.4 Regions: access maps → offsets / sizes / strides (FR-M8)

Every `ChannelSite` carries a `Region` (`06-interfaces.md` §5.2). The L1 end is always the whole
buffer, which is the empty region `([], [], [])` — verified in the probe as
`(%alloc_27[] [] [])`. The L3 end is derived:

```pseudo
 1  function L3_REGION(a, access_map, pe_coord, loop_ivs) -> Region:
 2      offsets := for each array dim d:  access_map.offsets[d] evaluated with
 3                    the PE coordinate and any enclosing loop IVs left symbolic
 4      sizes   := the tile extent along array dim d
 5      strides := row-major over a.shape:  strides[-1] = 1,
 6                    strides[d] = product(a.shape[d+1:])
 7      assert len(offsets) == len(sizes) == len(strides) == rank(a)
 8      return Region(offsets, sizes, strides)
```

Worked: `A[pi*TM : (pi+1)*TM, kk : kk+TK]` on `A: [64,64] f32` gives
`offsets=(pi*32, kk)`, `sizes=(32,16)`, `strides=(64,1)` — which is exactly what the probe
emitted, `(%arg10[0, %0] [32, 16] [64, 1])`. A partial L1 region (the halo rows) is the same
computation against the buffer shape: `u[1:2, :]` on `[6,16]` gives
`offsets=(1,0)`, `sizes=(1,16)`, `strides=(16,1)`, and the probe emitted
`(%alloc[1, 0] [1, 16] [16, 1])`.

`test_sem_access_regions` (`04-test-plan.md` §3.4) reconstructs the index set from this triple and
compares it to the `AccessMap`'s image, which is the check that catches a wrong slice.

### 3.5 Channel plan and loop-kind assignment (FR-M8, D-3)

```pseudo
 1  function CHANNEL_PLAN(mapping, delivery, herd, buffers) -> tuple[ChannelPlan]:
 2      out := []
 3      for (a, kind, along, declared) in delivery:
 4          switch kind:
 5            MULTICAST:  (size, bshape, put_ix, get_ix) := MULTICAST_GEOMETRY(herd.grid, along)
 6                        out.append(ChannelPlan(f"{a}2L1", size, bshape, None, a.dtype, sites))
 7            STATIONARY: if a is written:                       # drain only
 8                        out.append(ChannelPlan(f"{a}2L3", herd.grid, None, None, dtype, sites))
 9                        if a is read:                          # fill only
10                        out.append(ChannelPlan(f"{a}2L1", herd.grid, None, None, dtype, sites))
11            FORWARD:    out += WAVEFRONT_CHANNELS(...)          # §3.6.2 — THREE channels
12            CASCADE:    out += CASCADE_CHANNELS(...)            # §3.6.3
13      out += HALO_CHANNELS(...)                                 # §3.6.1
14      return sorted(out, key=name)
```

**Loop kinds (D-3), the whole rule in four lines.**

```pseudo
 1  function LOOP_KIND(axis, body) -> "unrolled" | "sequential":
 2      if axis's value is used as a channel bundle index:  return "unrolled"   # Python loop
 3      if body reuses an L1 buffer across trips:           return "sequential" # air.sequential
 4      return "sequential"                                                     # the safe default
```

Line 2 is `ChannelPutOp::verify` (`AIRDialect.cpp:3586-3593`); line 3 is `_loop.py:14-19`. They
never conflict, because a bundle index is a *spatial* quantity and a reused buffer is driven by a
*temporal* one (HLD §7.1's design rule). `LoopPlan.kind == "unrolled"` is additionally forbidden
inside a herd body by `06-interfaces.md` §5.5's invariant, and M4 never produces one there:
inside a herd body the spatial quantity is the herd coordinate, which is not a loop index at all.

Finding N-6 narrows the upstream rule — the verifier only rejects an index that *is* an `scf.for`
IV, not one derived from it by an `affine.apply` — but the plan keeps the stricter form, because
the strict form is what makes `BUNDLE-INDEX-IS-IV` (§3.7.3) sound by construction and it costs
nothing.

### 3.6 Protocol synthesis

#### 3.6.1 Halo exchange (FR-M4) — W2

**Channels.** Four: two halo bundles at `size=[PI-1]`, one bundle index per *physical link*
(the convention resolved in open-questions §2, superseding HLD §7.2's `size=[4]`), plus the L3
staging channel and the L3 **drain**:

| channel | index `k` means | put by | put index | put guard | get by | get index | get guard |
|---|---|---|---|---|---|---|---|
| `ToNorth` | link PE `k+1` → PE `k` | PE `tx` | `tx-1` | `tx > 0` | PE `tx` | `tx` | `tx < PI-1` |
| `ToSouth` | link PE `k` → PE `k+1` | PE `tx` | `tx` | `tx < PI-1` | PE `tx` | `tx-1` | `tx > 0` |
| `UIn` | PE `k`'s initial strip | segment | `k`, Python loop | — | PE `tx` | `tx` | — |
| `UOut` | PE `k`'s computed strip | PE `tx` | `tx` | — | segment | `k`, Python loop | — |

Every halo index expression lands in `[0, PI-1)` on the branch where it is emitted.

**`UIn` stages plane 0 once**, before the `t` loop: PE `tx` gets `U[0, tx·HS : tx·HS+HS+2, :]`
into `cur` — its `HS` owned rows **plus both ghost rows**, so `t = 0`'s first update has its
boundary values without a prologue put. `next` is then seeded from `cur` by a `StoreNode` copy,
because the drain below writes the staged boundary back on **every** plane and the planes
alternate between the two strips (§6.3's erratum; the transfer count stays `UIn[k] = 1/1`).

**`UOut` drains every plane the kernel writes.** One put per PE per timestep, immediately after
the update, region `U[t+1, tx·HS+1 : (tx+1)·HS+1, :]`, `size=[PI]`. This is the edit G-10 asked
for and it is not cosmetic: with only a final-strip drain, `test_sem_coverage`
(`04-test-plan.md` §3.4) fails **by construction**, because the union of the written index sets
would be one plane and the kernel's write domain is `T` planes. The per-PE regions partition
rows `1..H` of each plane exactly, with no overlap, which is the other half of that test.

**`tensors` is one L3 buffer.** W2's kernel has a single parameter, `U`, which is both read and
written, so `MappingPlan.tensors` is `(U,)` and §3.3's `TENSOR_PLAN` ordering law is vacuous here.
There is no separate `Uout` tensor; a second rank-2 tensor would not be the kernel's L3 interface
and the oracle could not be compared against it.

**Site order, per PE per timestep, and it is not negotiable** (FR-M4, D6, VF §C's E1 SAFE
verdict): `PUT(north boundary) → PUT(south boundary) → GET(north ghost) → GET(south ghost)`,
both puts `is_async = True`, **both gets with `depends_on = ()`**. The gets must not take a token
from the puts: a token would serialise the exchange into a rendezvous and reintroduce exactly the
deadlock E1 was cleared of.

**No prologue, no seeded ghost.** At `t = 0` each PE puts its *initial* boundary row and gets its
neighbour's *initial* boundary row, which is the correct value (PC §3.2). A prologue put would
give `T+1` puts against `T` gets and break per-iteration balance, which AIR's own model requires
(PC §1.1 fact 2).

**The `u`/`v` swap, with D-4 overridden.** `air.sequential` has no `iter_args` anywhere in
`air.api` (`_loop.py:180`; VF §D.5), so the swap cannot be loop-carried, and a plain Python `t`
loop would unroll the whole exchange and strand the acquire/release pairs (`_loop.py:14-19`).
The swap is therefore realised by unrolling `t` by two. **The architect has overridden D-4: an odd
`T` is legal and is handled by peeling, not rejected.**

```pseudo
 1  function TIMESTEP_LOOP(T, u, v):
 2      full := T - (T mod 2)                       # the even part
 3      loop := LoopPlan(axis="t", lo=0, hi=full, step=2, kind="sequential", depth=0, body=[
 4                  *STEP(src=u, dst=v),            # 2 puts + 2 gets + the 5-point update
 5                  *STEP(src=v, dst=u)])           # 2 puts + 2 gets + the 5-point update
 6      nodes := [loop]
 7      if T mod 2 == 1:
 8          nodes += STEP(src=u, dst=v)             # PEELED: straight line, same depth as `loop`
 9          live := v                               # the drain reads v, not u
10      else:
11          live := u
12      return (nodes, live)
```

Three properties this shape has, each of which the self-check then re-derives independently:

* **every loop iteration is balanced**: 2 puts + 2 gets per `STEP`, 2 `STEP`s per trip, so 4 puts
  and 4 gets per channel-branch per trip;
* **the peeled tail is balanced on its own**: 2 puts + 2 gets, straight-line, outside any loop;
* **the totals come out right for either parity**: channel index `k` sees
  `2·(T div 2) + (T mod 2) = T` puts from PE `k+1` and `T` gets at PE `k`.

Line 9 is why the peel is an M4 decision and not an M5 one: **which buffer the drain reads depends
on `T`'s parity**, and under D-14 the emitter may not compute that. The plan records the drain
site against `live`.

`SWAP-PARITY` (`06-interfaces.md` §6.3) therefore no longer fires for an odd `T`. It is kept in
the catalogue for the case the peel cannot be built — a loop whose trip count is not a compile-time
constant, which the grammar does not currently admit — and §10 records that it must keep at least
one test that raises it or `test_D3_catalogue_complete` fails.

**Per-core DMA budget.** At `PI = 2` every PE is a boundary PE: inbound = `UIn` + one halo get
= **2**; outbound = `UOut` + one halo put = **2**. Both sit exactly at the AIE2 core tile's
2-S2MM / 2-MM2S budget. At `PI ≥ 3` the interior PE gains a second halo get and needs **three**
circuit-switched inbound channels — finding N-1, and measured: `aircc` fails with
`'aie.connect' op … TileID(1, 2) targets same dst` (REVIEW-round1 P-R2), and `aie.w2pi4.mlir`
has zero packet flows, so nothing multiplexes. §3.8 rejects it with `DMA-CHANNELS` naming the
2-S2MM budget; `PI = 2` is the fixture and `PI = 4` is the negative test.

#### 3.6.2 Wavefront forward (FR-M5) — W3

Finding N-2 forces the shape: **a channel bundle may not mix L3-attached and core-to-core
members**, so the single `size=[PJ+1]` bundle of HLD §7.3 is replaced by three homogeneous
channels. The *property* FR-M5 asks for — every channel index has equal puts and gets, closed at
both ends by a constant source and a drain rather than by an unbalanced boundary — is preserved
exactly, and there are still `PJ+1` indices in total.

| channel | `size` | endpoints | sites |
|---|---|---|---|
| `WestIn` | `[1]` | L3 → L1 | segment: `MQ` puts at `[0]`, inside `air.sequential(1, MQ+1)`; herd: `get(edge_in, [0])` guarded `tx == 0` |
| `West` | `[PJ-1]` | L1 → L1 | herd: `put(edge_out, [tx])` guarded `tx < PJ-1`; `get(edge_in, [tx-1])` guarded `tx > 0` |
| `EastOut` | `[1]` | L1 → L3 | herd: `put(edge_out, [0])` guarded `tx == PJ-1`; segment: `MQ` gets at `[0]` |
| `QIn` | `[1]`, `broadcast_shape=[PJ]` | L3 → L1 | segment: one put at `[0]`; herd: `get(qb, [tx])` once, before the row loop |
| `RIn` | `[PJ]` | L3 → L1 | segment: `PJ` puts at `[k]` under a Python loop; herd: `get(rb, [tx])` once, before the row loop |
| `SOut` | `[PJ]` | L1 → L3 | herd: `put(cur[1:CW+1], [tx])` **once per row**; segment: `MQ` gets at `[k]` per PE |

**`q` and `r` are staged, and this is what makes the kernel the kernel** (G-8). `q[i-1]` does not
index `j`, so §3.2 classifies it `MULTICAST along px` and §3.5's geometry gives `size=[1]`,
`broadcast_shape=[PJ]`: every PE holds the whole 32-element vector (128 B). `r[j-1]` indexes only
`j`, so PE `tx` takes the slice `r[tx·CW : (tx+1)·CW]` (32 B). Both are staged **once**, before
the row loop, because neither changes with `i`. The compute then reads them:
`Select("==", Load(qb, i-1), Load(rb, j-1-tx·CW), Const MATCH, Const MISMATCH)`. Without them the
plan computes a literal `+2` for every cell and is not Smith-Waterman at all.

**`SOut` drains every row** (G-9). One put per PE per row of `cur[1:CW+1]` into
`S[i, tx·CW+1 : (tx+1)·CW+1]`, so the union over PEs and rows is exactly the kernel's write
domain and `test_sem_coverage` passes. Draining only the last row would require restating FR-K4
to compare one row against the oracle, which is not what FR-K4 says.

**Measured**: with `q` and `r` staged — three logical inbound channels per core —
`aircc --device npu1 --output-format=none` exits 0 with zero `error:` lines
(REVIEW-round1 P-R4), because L3→L1 gets lower to `aie.packet_flow` and share one shim MM2S.
§3.8 therefore **warns** here rather than failing. `tensors` is `(q, r, S)` — reads before
writes, or `_check_interface` raises (§3.3 note 6; the measured message named exactly this
program).

The head and tail guards are `ops.branch`, never a Python `if` — `bool()` on a `Condition` raises
(`_cond.py:41-45`) and `air-to-aie` folds the branch once the coordinate is literal
(`_cond.py:49-54`). The two guards on a PE are mutually exclusive, which is what keeps §3.8's
inbound count at 1.

**Per row, per PE, in this order**: `GET(edge_in)` → `CW` scalar updates as one `air.sequential`
→ `PUT(edge_out)` → `PUT(SOut)` → the `prev`/`cur` role swap. Per-core endpoints: inbound
`QIn + RIn + (WestIn | West)` = 3 (two of them packet-capable); outbound
`(West | EastOut) + SOut` = 2. The bundle index is `tx` or `tx-1`, **never** the
row loop's IV (H-3). The `prev`/`cur` swap uses the same unroll-by-two-and-peel machinery as
§3.6.1, over the row axis, so an odd `MQ` is legal.

The wavefront itself is emergent: nothing in AIR expresses the skew, and the diagonal order is a
consequence of each PE blocking on its `get` (PC §3.3 W3·P1 (d)). M4 records that in the summary
rather than trying to emit it.

#### 3.6.3 Cascade for `R_space` (FR-M6) — W1-flip

M3 has already established `rank(R_space) == 1` and a contiguous PE line (`CASCADE-RANK`), and
that the operator is tagged A/C (`RSPACE-NO-AC-OP`). M4 builds the chain.

```pseudo
 1  function CASCADE_CHANNELS(mapping, herd) -> (ChannelPlan, chain_axis, direction):
 2      c := the PE axis on which the r_space basis vector is placed
 3      P := herd.grid[c]
 4      # Orientation is forced by aie.cascade_flow: "source tile must be to the
 5      # North or West of the destination tile". open-questions §4.
 6      if len(herd.grid) == 1:  direction := ASCENDING    # 1-D herd = a row of columns, W->E
 7      else:                    direction := DESCENDING   # 2-D herd on npu1 = (1,4): one column
 7a     # direction is recorded as ChannelPlan.chain_direction; M5 reads it and never derives it
 8      chan := ChannelPlan(name = f"Cascade{axis_name(c).upper()}",
 9                          size = (P - 1,),               # one index per LINK, no dead index
10                          broadcast_shape = None,        # forbidden: _channel.py:170-177
11                          channel_type = "npu_cascade",
12                          dtype = accumulator dtype, sites = ...)
13      return (chan, c, direction)
14
15  function CASCADE_SITES(chan, coord, P, direction):
16      # ASCENDING: head at 0, tail at P-1, link index = coord     (put) / coord-1 (get)
17      # DESCENDING: head at P-1, tail at 0, link index = coord-1  (put) / coord   (get)
18      head_at, tail_at, put_ix, get_ix := table above
19      return [ GUARD(coord == head_at): [ PUT(acc, put_ix) ],
20               OTHERWISE:               [ GET(recv, get_ix),
21                                          ACCUMULATE(acc := acc ⊕ recv),
22                                          GUARD(coord == tail_at): [ PUT_L3(acc) ],
23                                          OTHERWISE:               [ PUT(acc, put_ix) ] ] ]
```

**Measured on the pinned wheel, both directions, this session (REVIEW-round1 P-R3).** A 1-D
`grid(4)` herd places as four columns on one row and the chain **must ascend**: the ascending
module gives `aie.cascade_flow(%tile_0_2, %tile_1_2)`, `(1,2)→(2,2)`, `(2,2)→(3,2)` and `aircc`
exit 0, while the descending variant of the same module fails with `'aie.cascade_flow' op source
tile must be to the North or West of the destination tile`. A 2-D `(1,4)` herd is one column of
four rows and **must descend** — that is the earlier `$PROBE/q/flip2.py` observation, and it is
the opposite case, not a contradiction. Line 6-7's rule is therefore measured on both sides. The
direction is a `ChannelPlan.chain_direction` field (`06-interfaces.md` §5.3), **never an emitter
choice**: M5 has no rule that could recompute it and D-14 forbids it trying.

Lines 19-23 are the shape of `cascade_reduction.py:87-99`, with the nesting that `_cond.py:57-60`
requires (there is no `and`; conjunction is nesting). `at=` pinning is **not** planned — neither
upstream's example nor our own probe needed it, and pinning removes `air-place-herds`' freedom
(open-questions §4). `HerdPlan.at` stays `None`; it is the documented escape hatch if a device run
ever shows a bad placement (R-03).

### 3.7 The self-check

This is the module's reason for existing. `docs/AIRCorrectnessChecker.md:15-20` specifies P1
(channel balance), P2 (deadlock freedom), P3 (resource constraints) and P4 (token consistency);
`mlir/lib/Analysis/` does not exist (VF §B.4); `air-opt` prints its one diagnostic and **exits 0**
(VF §B.6, §S11); and `AIRToAIEPass.cpp:4484-4491` silently *repairs* L2 imbalance rather than
diagnosing it (VF §S4). Nothing upstream will catch what M4 gets wrong.

**Enumeration, not symbolic reasoning (D-12).** Grid extents are compile-time constants and at
most 8 per axis (`_trace.py:88-91`), so the check instantiates the herd body once per concrete
coordinate and evaluates every guard and index expression to an integer. Symbolic reasoning over
`ops.branch` guards is a research problem we do not need.

#### 3.7.1 P1′ balance (FR-M9)

```pseudo
 1  function BALANCE(plan):
 2      counts := defaultdict(int)                     # (channel, concrete index, "put"|"get") -> int
 3      for scope_body, coords in [(plan.segment_body, [()]),
 4                                 (plan.herd_body,    ENUMERATE(plan.herd.grid))]:
 5          for coord in coords:
 6              WALK(scope_body, coord, trips=1, counts)
 7      for chan in plan.channels:
 8          for ix in ALL_INDICES(chan.size):
 9              p := counts[(chan.name, ix, "put")]
10              if chan.broadcast_shape is None:
11                  g := counts[(chan.name, ix, "get")]
12                  if p != g: FAIL("BALANCE", chan, ix, p, g)
13              else:
14                  for f in FANOUT(ix, chan.size, chan.broadcast_shape):   # D-2
15                      g := counts[(chan.name, f, "get")]
16                      if p != g: FAIL("BALANCE", chan, ix, f, p, g)
17
18  function WALK(nodes, coord, trips, counts):
19      for n in nodes:
20          if n is ChannelSite:
21              if n.guard is not None and not EVAL(n.guard, coord): continue
22              counts[(n.channel, EVAL_INDICES(n.indices, coord), n.kind)] += trips
23          elif n is LoopPlan:
24              WALK(n.body, coord, trips * TRIP_COUNT(n), counts)
25          elif n is BranchNode:                                  # 06-interfaces.md §5.5
26              if EVAL(n.predicate, coord): WALK(n.then,      coord, trips, counts)
26a             else:                        WALK(n.otherwise, coord, trips, counts)
26b         # StoreNode contributes nothing to balance and is skipped
27
28  function FANOUT(ix, size, bshape):
29      # the set of destination indices one put at `ix` reaches
30      return { d in ALL_INDICES(bshape) : all(d[k] % size[k] == ix[k] for k in dims) }
```

Line 14 is decision **D-2**, and the divergence from the upstream spec's literal P1 is deliberate:
the spec keys on `(channel_name, indices)` and demands `put_count == get_count` per key, which
would reject `air.api`'s own fan-out idiom (one put at `[pi,0]`, `PJ` gets at `[pi,0..PJ-1]`). The
rule reduces to the literal one when `broadcast_shape is None`.

Three things the walk gets right that a naive count would not, each with a negative test in §7:

* **per iteration** — `trips` multiplies through nested `LoopPlan`s (line 24), so a put added
  inside a loop body is caught even though the totals could still match;
* **per branch** — a guarded site is counted only on the coordinates where its guard holds (line
  21), so guarding a put but not its get is caught;
* **cross-scope** — the segment body is walked once (`coords = [()]`) and the herd body once per
  coordinate, so a segment-scope put of `K/TK` against `PI·PJ` herd gets is compared with the
  right multiplicities.

Line 25 walks a **`BranchNode`**, not a per-site guard: three of our protocols (the cascade's
head/middle/tail, W3's head/tail, W2's boundary PEs) need a branch whose `otherwise` body carries
a get, a store **and** a nested branch, which `ChannelSite.guard` alone cannot express
(`06-interfaces.md` §5.5, REVIEW-round1 B-4). Because the two arms are exclusive, exactly one is
walked per coordinate, which is what keeps the per-branch rule sound.

Failure carries the per-key count table in `Diagnostic.details`, mirroring the spec's §4.4 error
shape.

#### 3.7.2 P2b acyclicity (FR-M10)

The spec's P2b verbatim (`docs/AIRCorrectnessChecker.md` §5.2-5.3): build the put→get graph plus
intra-body program-order edges, exclude loop-carried back edges, and reject any strongly connected
component containing a channel edge.

**Nodes** are `(site, coord)` pairs — a site in the herd body is one node per concrete herd
coordinate; a site in the segment body is one node.

**Channel edges**: for every put node and every get node whose `(channel, concrete index)` match
after fan-out expansion, a directed edge put → get.

*(Erratum, 2026-09-13, phase P5.)* That all-pairs rule over-approximates a channel, which is a
**FIFO**: the k-th get receives the k-th put and waits on that one, not on every put the index
will ever see. The over-approximation is harmless while one body holds one transfer per index,
and it stops being harmless once §3.6.1's `swap_loop` puts two timesteps in one body — W2's own
halo then reports a four-edge cycle
`put_n(phase 1, PE 1) → get_n(phase 0, PE 0) → put_s(phase 1, PE 0) → get_s(phase 0, PE 1)`
built entirely out of edges pairing a *later* put with an *earlier* get, which is a dependency
no execution has. Note that this paragraph's own argument below is made **within a timestep**,
and the unroll is what breaks that framing. So: when both sides have the same number of nodes
and each side is one coordinate's own program order, the k-th put is zipped to the k-th get —
the exact semantics, and strictly fewer edges; otherwise (several producers into one index, or a
body shape that splits one side and not the other, as W3's guarded `WestIn` does) the pairing is
not defined and the all-pairs rule stands. `m4_selfcheck._pairs` is the two-branch function, and
the halo reversed to `[GET, GET, PUT, PUT]` still raises `CHANNEL-CYCLE` under it, on both the
synthetic fixture and W2's real plan.

**Program-order edges — the exact rule**, because "program order" is ambiguous across a scope
boundary and getting it wrong either misses a cycle or invents one:

1. *Within one body region* (a scope body, a `LoopPlan.body`, a `BranchNode` arm), for the same
   `coord`, an edge from the node at `order = i` to the node at `order = i+1`. Only consecutive
   pairs; the rest follows by transitivity.
2. *Into and out of a nested region* — a `LoopPlan` or a `BranchNode` is contracted to a single
   node in its parent's order, with an edge from that node to the first node of the region's body
   and from the last node back out. The contracted node is **not** a channel edge endpoint, so it
   cannot itself make an SCC illegal.
3. *Between the two arms of the same `BranchNode`* — **no edge**, in either direction. The
   two branches never both execute for one coordinate, and joining them invents paths.
4. *Across a scope boundary* — the whole herd is contracted to one node `H` in the segment body's
   order. A segment site before `H` gets an edge to `H`; `H` gets an edge to each segment site
   after it. **No edge is drawn from a segment site directly to a herd-body site**: the launch and
   the herd are concurrent async regions, which the probe's token graph confirms (a segment-scope
   producer loop yields a token the `air.herd` does not consume — open-questions §1, evidence 3).
   Drawing one would falsely serialise producer and consumer and would make W1 look cyclic.
5. *Loop-carried back edges* — the edge from the last node of a `LoopPlan.body` to its first node
   at the next trip is **`StructuralBack` and is excluded**, exactly as the spec prescribes
   (§5.3). This is what makes W2 acyclic: within a timestep the graph is `put(p) → get(p-1)` and
   `put(p) → get(p+1)`, with no path from a get back to a put, and the only such path would be
   through the loop back edge.

```pseudo
 1  function ACYCLIC(plan):
 2      G := graph over (site, coord) nodes
 3      ADD_CHANNEL_EDGES(G, plan)              # put -> get, fan-out expanded
 4      ADD_PROGRAM_ORDER_EDGES(G, plan)        # rules 1-4 above; rule 5 omits the back edge
 5      for scc in TARJAN(G):
 6          if len(scc) > 1 and any edge inside scc is a channel edge:
 7              FAIL("CHANNEL-CYCLE", cycle = the edge list of scc)
```

Why the halo protocol is acyclic and its reversal is not, stated once so the negative test has a
reason: with the order `[PUT_n, PUT_s, GET_n, GET_s]` the only outgoing edges from a get are
program-order edges to later compute, so no path leads back to a put. Reverse it to
`[GET_n, GET_s, PUT_n, PUT_s]` and the cycle
`GET_n(p) → PUT_s(p) → GET_s(p+1) → PUT_n(p+1) → GET_n(p)` appears within one timestep. That is
`test_M10_cycle_rejected`.

#### 3.7.3 The two structural checks

```pseudo
 1  function BUNDLE_INDICES(plan):           # invariant 3, 06-interfaces 5.6
 2      for site in ALL_SITES(plan):
 3          for ix in site.indices:
 4              if ix references a temporal loop IV: FAIL("BUNDLE-INDEX-IS-IV", site, ix)
 5
 6  function PING_PONG_SHAPE(plan):          # invariant 4
 7      for b in plan.buffers where b.ping_pong_candidate:
 8          assert b is a direct child of a LoopPlan
 9          assert the first site touching b is a `get`
10          assert exactly one get of b per iteration
11          assert every enclosing LoopPlan has constant lo/hi/step
12          assert no StoreNode with an opaque callee touches b first
```

Lines 8-12 are `isPingPongCandidate`'s conditions 2, 3, 5 and 4 respectively
(`AIRDependencyScheduleOpt.cpp:1604`, `:1620-1630`, `:1690`; `Transform/Passes.td:970-971`),
measured to fire on this exact shape in VF §E.5 and re-measured on our own W1 in open-questions
§1 evidence 4. Condition 6 (the L1 budget) is invariant 5; conditions 1, 7 and 8 are properties of
what M5 emits and cannot be violated by a plan.

### 3.8 P3 — the per-core **circuit-switched** DMA-channel check (new, forced by finding N-1)

An AIE2 core tile has **two S2MM (inbound) and two MM2S (outbound)** DMA channels. Exceeding the
inbound budget is a hard `aie.connect` error raised deep inside mlir-aie's router, roughly twenty
seconds into `aircc`, naming a physical `TileID` the user has never heard of. Exceeding the
outbound budget is *silently* accommodated by merging two endpoints onto one MM2S (finding N-3),
which is a multicast on a circuit-switched fabric and is not established to be correct.

```pseudo
 1  DMA_IN_MAX  := 2
 2  DMA_OUT_MAX := 2
 3
 4  function CIRCUIT(site) -> bool:
 5      # A channel binds a circuit-switched DMA channel when BOTH ends are core tiles, or
 6      # when one end is L3 and the design cannot expect the flow to be packet-switched.
 7      # Measured: L3<->L1 gets on our shapes lower to aie.packet_flow and MULTIPLEX.
 8      return channel_of(site).is_core_to_core
 9          or (channel_of(site).touches_L3 and not channel_of(site).may_packet)
10
11  function DMA_CHANNELS(plan):
12      for coord in ENUMERATE(plan.herd.grid):
13          live := { s for s in herd sites if EVAL(s.guard, coord) }
14          for kind, budget, word in (("get", DMA_IN_MAX, "inbound"),
15                                     ("put", DMA_OUT_MAX, "outbound")):
16              hard := MAX_OVER_EXCLUSIVE_BRANCHES({ s.channel for s in live
17                                                    if s.kind == kind and CIRCUIT(s) })
18              all_ := MAX_OVER_EXCLUSIVE_BRANCHES({ s.channel for s in live
19                                                    if s.kind == kind })
20              if len(hard) > budget:  FAIL("DMA-CHANNELS", coord, word, hard)   # error
21              elif len(all_) > budget: WARN(coord, word, all_, "may lower to packet flows")
```

`MAX_OVER_EXCLUSIVE_BRANCHES` counts sites on mutually exclusive guard branches once, not twice —
that is what keeps W3's PE 0 at one inbound channel although it names two (`WestIn` under
`tx == 0`, `West` under `tx > 0`). It is sound because `air-to-aie` folds the branch away once the
coordinate is a literal (`_cond.py:49-54`), so only one of the two survives on any given tile.

**Counting logical channels would be unsound, and was.** Measured this session (REVIEW-round1
P-R4): W3 with `q` and `r` staged from L3 gives **three** distinct inbound channels per core
(`WestIn`/`West` on exclusive branches = 1, plus `QIn`, `RIn`) and
`aircc --device npu1 --output-format=none` **exits 0 with zero `error:` lines**. The reason is in
the lowered IR — W1's and W3's L3→L1 gets become **packet flows**, which multiplex: `aie.q7a.mlir`
carries 3 `aie.packet_flow` with `air_A2L1_0`, `air_B2L1_0` and `air_B2L1_1` all allocated
`shim_noc_tile_0_0, MM2S, 0`. W2 at `PI=4` has **zero** packet flows (every flow is a
circuit-switched `aie.flow`), which is exactly why its first interior PE ran out at three
(`aie.w2pi4.mlir`, P-R2). The budget binds on **circuit-switched** flows.

So: exceeding the budget counting circuit-switched core-to-core and non-packet L3 endpoints is a
`DMA-CHANNELS` **error**; exceeding it once packet-capable channels are counted is a **warning**
that names the uncertainty. A checker that rejects a program the toolchain accepts is worse for G2
than no checker at all — that is the whole argument for the split. The packet-vs-circuit rule is
**measured on four module shapes, not a documented upstream contract** (risk R-19, R-21), which is
why the warning stays a warning until a device run settles it.

The diagnostic names the PE coordinate, the channel list, the budget, and the clause to edit
(`grid`, or fewer staged operands) — the four-part shape of HLD §4.2. This converts the worst
failure mode in the design into one of our own messages, and it is the third of the four
properties AMD's own document specifies and does not implement.

**Erratum, 2026-09-15 — what P3 does *not* cover, and why it was not widened (B-P33).** An
architect ruling (R-P3-2) offered to extend `DMA-CHANNELS` to whatever resource makes
`air-to-aie` refuse `kernels/w1_gemm_bf16` on npu1. Measured, the refusal is **not** a DMA or
column resource. `AIRToAIESchedulingUtils.cpp:3892-3894` fails the launch-side
`air.channel.get @C2L3[0,0]`:

```cpp
if (f.MM2S_alloc.empty() || !f.MM2S_alloc[0].getDmaTile())
  return memcpyOpIf->emitOpError("failed to get MM2S tile for L3 allocation.");
```

`MM2S_alloc[0]` is empty because the **herd-side put's bundle index is still a loop induction
variable**. With `repeats == (4, 1)` the placed module carries
`%29 = affine.apply ()[%arg10, %arg14] -> (s0 * 4 + s1)` inside a surviving `scf.for %arg14`, so
`specializeChannelBundle` never resolves bundle position `[0,0]` to a producer. Upstream names
this cause in its own regression,
`mlir/test/Conversion/AIRToAIE/segment_id_remap_no_unroll.mlir:14-19`: *"channel operations using
them as indices fail to specialize, causing: 'air.channel.get' op failed to get MM2S tile for L3
allocation."*

The discriminator is `repeats`, not a channel count — measured over five grids at the same tiles
and the same 49 152 B of L1: `repeats[0] == 1` compiles (exit 0, both targets); `== 2` clears
`air-to-aie` (the ping-pong unroll factor 2 folds a trip-2 repeat loop away, leaving `s0*2` and
`s0*2+1`); `== 4` gives this error. Grid 2×4 and grid 4×2 both put **8** outbound `C2L3` flows on
the shim and land on opposite sides of the line, which is what rules the channel count out. So
`DMA-CHANNELS` was left alone: it names two S2MM and two MM2S per core tile, and widening it to
cover this would name a resource that is not the one that ran out. The honest condition is a
bundle-index one (`06-interfaces.md` §5.6 invariant 3's territory) and is **not** implemented,
because the unroll factor 2 is measured on this wheel and is not a documented contract (R-19,
R-21). `design/PROGRESS-B.md` B-P33 carries the full table.

**Erratum, 2026-09-15 — B-P33 is closed by R-HERD-1, and P3 is still not where it lives.** The
bundle-index condition the paragraph above declined to write **is** written now, but in M3 and
as a condition on `repeats`, not on the index: `_check_repeats` rejects `max(repeats) > 2` with
`HERD-PHYSICAL` (`03-lld-M3-checker.md` §3.11 erratum). P3 is unchanged, and the argument stands
— the resource that ran out was never a DMA channel.

**Erratum, 2026-09-15 — P3 gains two more counted rules, and the earlier "not modelled" note
is withdrawn (B-P36, B-P37).** Both refusals are outside the two-S2MM/two-MM2S budget this
section counts, and an earlier pass recorded them without a rule. Traced this pass, each has a
resource with a quotable capacity and a demand the *plan* can count, so each is now a
`DMA-CHANNELS` error raised by `m4_selfcheck.self_check` before anything is emitted.

* **B-P37 — the broadcast guard's column bound.** Not a shim resource at all. When
  `air-specialize-dma-broadcast` splits a bundled multicast it wraps each split get in an
  `affine.if`; the set pins the specialised herd coordinate and bounds **every other** one by
  `numCols - 1` (`mlir/lib/Transform/AIRMiscPasses.cpp:265-271`), and the `numCols` argument is
  `herd.getNumCols()` (`:347-349`) whichever dimension is being bounded. A herd with more rows
  than columns, split on the column axis, therefore admits only rows `0 .. cols-1`; the rows
  above fall into the chain's `else` arm, are served by the *last* split channel — a wrong-data
  lowering in its own right — and leave the *first* split channel one broadcast destination
  short, so `f.S2MM_alloc[i]` is never filled and
  `AIRToAIESchedulingUtils.cpp:3866-3868` reports *"'air.channel.put' op failed to get S2MM tile
  for L3 allocation"*. **Pinned by construction:** editing only `#set` in the placed IR of the
  `(2, 3)` module from `-s1 + 1 >= 0` to `-s1 + 2 >= 0` and re-running `air-to-aie` gives exit 0.
  Measured on npu2 with W1's clauses: physical `(2, 2)` exit 0, `(2, 3)` and `(2, 4)` this error,
  `(3, 2)` and `(4, 2)` — the same six and eight cores transposed, with the physical cap lifted —
  exit 0. `m4_selfcheck.broadcast_guard` is the rule; a repeat factor on the split axis exempts
  it, because the index is then an `affine.apply` rather than a bare herd id and upstream takes
  its `scf.if` arm (`AIRMiscPasses.cpp:380-422`), which bounds nothing. That exemption is why
  npu1, whose 2-D physical herd is one column, never reaches it.
* **B-P36 — switchbox master selects, and the count the earlier pass could not find.** The
  capacity was always quotable (`int numMselsPerArbiter = 4`, mlir-aie
  `AIECreatePathFindFlows.cpp`; `aie.amsel`'s `msel` confined to 0..3, `AIEOps.td`). What was
  missing was the demand, because the flip's 2-D variant puts **eight** flows into one column and
  compiles. Counted this pass, on the *routed* IR (`aie-opt
  --aie-create-pathfinder-flows=route-packet` over `air_project/npu.src.mlir`), the demand is the
  number of L3→L1 fill flows **when one of them multicasts down the column**: that flow's port
  set at the bottom tile is `{DMA, North}`, which only partially overlaps each point-to-point
  flow's `{DMA}` or `{North}`, and a partial overlap opens a fresh amsel on the *same* arbiter
  instead of sharing one. Measured, `aie.amsel` on `%tile_0_2`: physical `(1, 2)` 3 flows → 3,
  `(1, 3)` 4 → 4, `(2, 2)` 4 → exit 0, `(1, 4)` and `(2, 4)` 5 → *used up all its msels*; the
  flip at `(1, 4)`, whose eight flows are **every one single-destination**, → 2 amsels on two
  arbiters, exit 0. `m4_selfcheck.msels` is the rule and `fill_flows` the count, which reproduces
  the emitted `aie.packet_flow` count on all six shapes.

What neither rule models is stated in `m4_selfcheck.msels`'s docstring: that every fill flow of
a herd enters the same shim column. It is what all six shapes show, but the column a flow climbs
is the shim bin packing's choice (`AIRToAIESchedulingUtils.cpp:3825-3845`), not the plan's.
`design/PROGRESS-B.md` B-P36/B-P37 carry the verbatim errors and the full tables.

**`DMA-CHANNELS` is in the catalogue.** It landed in `06-interfaces.md` v2 before the D0 freeze
(REVIEW-round1 RULING 5), so the earlier `PROTOCOL-UNSUPPORTED` fallback is deleted and **B-O6 is
closed**. `PROTOCOL-UNSUPPORTED` now means only what it says: a declared protocol with no
synthesis rule, such as `reside(x="L2")` (§3.3 note 5).

### 3.9 `MappingSummary` (FR-M11, FR-D2)

The summary is what a judge reads in five minutes, and it exists because "the reader of the
program cannot tell output-stationary from weight-stationary without re-deriving it from slice
arithmetic" is the ARIES criticism this surface answers (PC §1.4 item 2).

```pseudo
 1  function SUMMARY(plan) -> MappingSummary:
 2      lines := []
 3      HOW := lambda kind, along: { STATIONARY: "stationary",
 4                                   MULTICAST:  f"multicast along {along}",
 5                                   FORWARD:    f"forward along {along}",
 6                                   CASCADE:    f"cascade along {along}" }[kind]
 7      for (a, kind, along, declared) in plan.delivery:            # already sorted
 8          lines.append(f"{a}: {HOW(kind, along)} ({'declared' if declared else 'derived'})")
 8a     for (a, kind, along, _) in plan.delivery:               # the residency block, RULING 9
 8b         what := "stationary (spatial)" if kind is STATIONARY else HOW(kind, along)
 8c         lines.append(f"{a}: {what}, {RESIDENCY(a, plan)}")
 9      lines.append(f"herd: logical {tuple} physical {tuple} repeats {tuple}")
10      lines.append(f"reduction (tiled axes): R_time = {basis}, R_space = {basis}")
11      for b in plan.buffers: lines.append(f"  {b.name} {b.shape} {b.dtype} = {b.bytes} B"
12                                          + ("  x2 (ping-pong)" if b.ping_pong_candidate else ""))
13      lines.append(f"L1: {l1_bytes} of {l1_budget} bytes")
14      for c in plan.channels: lines.append(f"  {c.name} size={c.size}"
15                                           + (f" broadcast_shape={c.broadcast_shape}" if ...)
16                                           + (f" type={c.channel_type}" if ...))
17      return MappingSummary(lines=tuple(lines), ...)
```

**Lines 8a-8c are the residency block (RULING 9).** FR-L3's stationarity predicate is *spatial*
— `ker M_a ⊆ ker Sπ`, "none of `a`'s reuse crosses a PE" — and is silent about time. An operand
can satisfy it and still be re-fetched on every trip, which is exactly what the W1-flip did
while `j` was tiled: `B` was declared stationary, legally, and its `[16,32]` tile moved with
`j0`. The summary therefore prints one **residency duration** per operand beside its delivery:

```pseudo
 1  function RESIDENCY(a, plan) -> str:            # how long a's L1 tile stays put
 2      T := [x for x in plan.sigma_axes                        # σ order, outermost first
 3              if IS_OUTER_TILE_AXIS(x) and e_x not in span(plan.pi)]   # temporal TILE axes
 4      moved := [x in T if M_a · e_root(x) != 0]               # T \ (ker M_a ∩ T), in σ order
 5      if moved is empty:   return "resident for the whole run"
 6      x0    := moved[0]                                       # outermost axis that moves the tile
 7      inner := [x in T strictly after x0 if x not in moved]
 8      return (f"resident across {', '.join(inner)}, " if inner else "") + f"re-fetched per {x0}"
```

Line 3's restriction to **outer tile axes** is what makes the answer true rather than merely
type-correct. Only the outer axis of a `tile(ax.x, T)` can move a tile: an *inner* axis and an
*untiled* axis both run **inside** the tile by construction of `TILE_SHAPE` (§3.3 note 1), and a
*placed* axis is spatial, not temporal. So on the flip `T = (i0,)`: `M_B · e_i = 0`, so `B` is
`resident for the whole run`, while `M_A · e_i ≠ 0`, so `A` is `re-fetched per i0` — and that
pair of lines is the demo's claim, checked rather than asserted. On W1, `T = (k0,)` and the
lines are `C: stationary (spatial), resident for the whole run` with `A`/`B`
`re-fetched per k0`. On W2 and W3 `T` is empty (`i0`/`j0` are placed; `t`, `i`, `j` are
untiled), so every operand prints `resident for the whole run` — true: W2's strip is swapped in
place across the whole `t` sweep.

The residency block is **additional**: the delivery block above it is unchanged, so FR-M11's
three literal W1 strings still appear verbatim and the golden gains lines rather than editing
the ones it has. `MappingSummary.residency` carries the same pairs as data
(`06-interfaces.md` §5.7, `CONTRACT_VERSION = 3`); `test_M11_residency_line` is the test.
Residency is also the exact form of ping-pong condition 2 (`03-lld-M3-checker.md` §3.13): an
operand resident for the whole run has no streaming loop to be allocated inside, so
`double_buffer` on it is a `PINGPONG-SHAPE` rejection.

**Line 10 renders the *tiled* split.** `LegalMapping.r_time` and `.r_space` are bases over
`UCoord` (`03-lld-M3-checker.md` §3.1), and on the flip that gives `R_time = {}`,
`R_space = span{e_k}` — but printed bare, `R_time = {}` reads as "no local accumulation", which
is false: each PE does accumulate its own `TK`-slice before handing it up the cascade. The
summary therefore re-expresses both bases over the post-tiling axes, so the flip prints
`R_time = span{e_k1}, R_space = span{e_k0}` (`03-lld-M8-kernels-demo.md` §6.2). The two objects
are **not** the same and the line says which it is showing: the `LegalMapping` fields stay
untiled and are what M4's own classification reads (§3.2); only the rendered text is tiled.

FR-M11's acceptance requires the literal strings `C: stationary (declared)`,
`A: multicast along py (derived)` and `B: multicast along px (derived)` — lines 4-8 produce
exactly those. *(Erratum, 2026-09-13, the ruling on B-P17: `C`'s row is **declared**, because
`stationary("C")` names its delivery; `A` and `B` are named by no clause. Two derived rows and
one declared still say the dataflow is named, not emergent.)* The summary is compared byte for byte against
`tests/golden/<workload>.<variant>.summary.txt` (`06-interfaces.md` §8).

---

## 4. Invariants, preconditions, postconditions

**Preconditions on `m4.plan(mapping)`** — all are M3's postconditions, and M4 asserts rather than
re-derives them (§5 explains why an assertion failure is a bug, not a user error):
`mapping.physical_herd` divides `mapping.schedule.grid` exactly and is within the target cap;
`len(mapping.pi) == len(mapping.schedule.grid)` and the rank is 1 or 2; `mapping.l1_bytes ≤ 63488`
(**R-L1-2**; `L1_BUDGET` still names the tile's 65 536, `L1_USABLE` is what binds);
`r_space` is empty or rank 1 with an A/C operator; every windowed operand's declared halo is at
least its derived footprint.

**Postconditions on the returned `MappingPlan`** — the five invariants of `06-interfaces.md` §5.6
(balance P1′, acyclicity P2b, bundle indices, ping-pong shape, L1 budget) plus the P3 check of
§3.8, each verified by §3.7/§3.8 before the plan is returned.

**Module invariants**, each with a test in §7:

| # | Invariant |
|---|---|
| I-1 | M4 imports nothing from `air` and nothing from M5. It is testable with no toolchain (A-6 is not even needed) |
| I-2 | `plan(m)` is a pure function of `m`. No clock, no `id()`, no RNG, no environment (HLD §5 rule 5) |
| I-3 | Every buffer name and channel name is derived by §3.1's table, never counted |
| I-4 | Every `ChannelPlan.sites` tuple has ≥ 1 put and ≥ 1 get, and `order` is strictly increasing within a scope |
| I-5 | `channel_type == "npu_cascade"` ⟹ `broadcast_shape is None` (`_channel.py:170-177`) |
| I-6 | No `BufferPlan.scope` is ever `"herd.shared"` (`_trace.py:1405-1414` raises) |
| I-7 | Every dict or set traversal that affects a name, an order or a text is `sorted(...)` on an explicit key |
| I-8 | `HerdPlan.at is None` for every plan M4 currently produces (open-questions §4) |

---

## 5. Error paths

M4 raises `MappingError` and nothing else (`06-interfaces.md` §6.2, NFR-7). Codes it owns:

| Code | Raised when | `details` carries |
|---|---|---|
| `BALANCE` | §3.7.1 found a key with `put_count != get_count` | the per-key count table, the channel, the index, and the site ids on both sides |
| `CHANNEL-CYCLE` | §3.7.2 found an SCC containing a channel edge | the ordered edge list of the cycle, with each site's channel, index and coordinate |
| `BUNDLE-INDEX-IS-IV` | §3.7.3 found a bundle index referencing a temporal IV | the site, the offending `Expr`, and the loop it came from |
| `PROTOCOL-UNSUPPORTED` | a declared protocol has no synthesis rule — a non-neighbour partner map, `rank(R_space) > 1` reaching M4, a `Pattern` with no builder, or `reside(x="L2")` (§3.3 note 5) | the clause, the operand, and what was asked for |
| `DMA-CHANNELS` | one of §3.8's three counted resources is exceeded: the per-core **circuit-switched** inbound/outbound budget, the broadcast guard's column bound (B-P37), or a switchbox arbiter's master selects (B-P36) | the demand, the capacity, and the resource: for the budget the PE coordinate, the channel names and which endpoints were counted as circuit-switched; for the guard the channel, its `size`/`broadcast_shape`, the split dimension and the rows left unserved; for the master selects the fill-flow count, the herd shape and which channel multicasts |

**What must never reach the user** (HLD §4.3). The self-check exists to catch *our* construction
bugs, and a self-check failure on a plan M4 itself built is an internal-consistency failure, not a
user error. The distinction is not cosmetic — it decides what the message says:

* A `BALANCE` or `CHANNEL-CYCLE` failure on a plan M4 built from an accepted schedule means the
  protocol builder is wrong. It is raised as a `MappingError` whose `reason` names the workload
  and the protocol and asks for a bug report — never a `fix` line that blames a clause the user
  wrote, because there is no clause edit that repairs our bug.
* The same codes on a plan handed to `m4.self_check` directly (the negative corpus) are *expected*
  and carry the count table for the test to assert.
* A precondition assertion from §4 failing means M3 and M4 disagree about the contract. It is
  raised with the two values and the document section that defines the agreement.
* A `KeyError` from an internal table, a numpy exception, or a traceback through the protocol
  builders never escapes: §3.1's `PLAN` wraps the nine passes and re-raises anything that is not
  already a `SpatialError` as a `MappingError` with the original in
  `details["internal_exception"]` (NFR-4, NFR-7).

`DMA-CHANNELS` is the one code here that *is* a genuine user error with a real fix, which is why
it has its own entry rather than living inside `PROTOCOL-UNSUPPORTED`. It entered the catalogue at
`CONTRACT_VERSION = 2` and is in the v3 contract signed at D0 (RULING 5), so **no fallback
remains and B-O6 is closed**. A §3.8 *warning* raises nothing at all: it is recorded in the
summary and printed, and the plan is returned.

---

## 6. Worked examples

Concrete plans, not sketches. These are the D0 hand-written fixtures (`05-work-breakdown.md` D0,
Person B) and the D2/D4/D5/D6 acceptance targets. Every channel name, size, index expression,
guard, scope and loop kind below is a literal field of the `MappingPlan`.

### 6.1 W1 — GEMM output-stationary

`M=N=K=64`, `TM=TN=32`, `TK=16`, `PI=PJ=2`, `f32`. Target `npu1`: `physical_herd=(1,2)`,
`repeats=(2,1)`.

**Delivery** (§3.2): `A: MULTICAST along py (derived)`, `B: MULTICAST along px (derived)`,
`C: STATIONARY (declared)` — *erratum, 2026-09-13: `stationary("C")` names `C`'s delivery, so
its row is `declared` (the ruling on B-P17)*.

**Herd**: `HerdPlan(name="gemm_herd", grid=(2,2), shape=(1,2), at=None, coords=("tx","ty"))`.

**Buffers** — all `scope="herd.private"`:

| name | operand | shape | bytes | `loop_depth` | `ping_pong_candidate` |
|---|---|---|---|---|---|
| `acc` | `C` | `(32,32)` | 4096 | 0 | False |
| `a` | `A` | `(32,16)` | 2048 | 1 | **True** |
| `b` | `B` | `(16,32)` | 2048 | 1 | **True** |

L1 = `4096 + 2·2048 + 2·2048 = 12 288` of 65 536, matching HLD §7.1 and `04-test-plan.md` §4.

**Channels**:

| name | `size` | `broadcast_shape` | `channel_type` |
|---|---|---|---|
| `A2L1` | `(2,1)` | `(2,2)` | `None` |
| `B2L1` | `(1,2)` | `(2,2)` | `None` |
| `C2L3` | `(2,2)` | `None` | `None` |

**Segment body** (`plan.segment_body`, in `order`):

```
0  LoopPlan(axis="pi_bundle", 0..2 step 1, kind="unrolled", depth=0, body=[     # Python loop: bundle index
1      LoopPlan(axis="k0", 0..64 step 16, kind="sequential", depth=1, body=[
2          ChannelSite(put, "A2L1", indices=(pi, 0), buffer="A", scope="segment",
                       region=Region(offsets=(pi*32, kk), sizes=(32,16), strides=(64,1)),
                       guard=None, is_async=False, depends_on=(), order=0) ])])
3  LoopPlan(axis="pj_bundle", 0..2 step 1, kind="unrolled", depth=0, body=[
4      LoopPlan(axis="k0", 0..64 step 16, kind="sequential", depth=1, body=[
5          ChannelSite(put, "B2L1", indices=(0, pj), buffer="B", scope="segment",
                       region=Region(offsets=(kk, pj*32), sizes=(16,32), strides=(64,1)), ...) ])])
6  <HERD>
7  LoopPlan(axis="i_drain", 0..2, kind="unrolled", body=[
8      LoopPlan(axis="j_drain", 0..2, kind="unrolled", body=[
9          ChannelSite(get, "C2L3", indices=(i, j), buffer="C", scope="segment",
                       region=Region(offsets=(i*32, j*32), sizes=(32,32), strides=(64,1)), ...) ])])
```

**Herd body** (`plan.herd_body`):

```
0  BufferPlan acc                                         # air.alloc, depth 0
1  LoopPlan(axis="i1", 0..32, "sequential", depth=1, body=[
2      LoopPlan(axis="j1", 0..32, "sequential", depth=2, body=[
3          StoreNode(zero acc[i1,j1]) ])])
4  LoopPlan(axis="k0", 0..64 step 16, kind="sequential", depth=0, body=[
5      BufferPlan a                                       # DIRECT child of the K loop  (H-1)
6      BufferPlan b                                       # DIRECT child of the K loop
7      ChannelSite(get, "A2L1", indices=(tx, ty), buffer="a", region=empty, scope="herd", order=2)
8      ChannelSite(get, "B2L1", indices=(tx, ty), buffer="b", region=empty, scope="herd", order=3)
9      LoopPlan i1, LoopPlan j1, LoopPlan k1
                        ->  StoreNode(acc[i1,j1] = acc[i1,j1] + a[i1,k1]*b[k1,j1]) ])
10 ChannelSite(put, "C2L3", indices=(tx, ty), buffer="acc", region=empty, scope="herd")
```

*Erratum, 2026-09-13 (the ruling on B-P18, `06-interfaces.md` §5.5 at v5): lines 1-3 and 9 are
named by the post-tiling axis each loop realises — `i1`, `j1`, `k1`, the zeroing nest reusing
`i1`/`j1` — where this section first printed `m0`, `n0`, `m`, `n`, `t`. Line 9's store is
`Statement.expr` (§2.4 at v5) with each `Load` rewritten into its L1 buffer; M4 rebuilds no
arithmetic (B-P19).*

Lines 5-8 are the ping-pong shape: the allocs are direct children of the K loop and each is first
touched by its `get`. Measured to fire — open-questions §1 evidence 4 gives `{unroll = 2 : i32}`
on this exact structure.

**Balance** (§3.7.1): `A2L1` put key `[0,0]` count 4 (`K/TK`); fan-out of `[0,0]` under
`size=(2,1)`, `broadcast_shape=(2,2)` is `{[0,0],[0,1]}`, and each has 4 gets → balanced. Same for
`[1,0]`. `B2L1` mirrored. `C2L3[tx,ty]`: 1 put, 1 get, for each of the four coordinates.
**Acyclicity**: segment → `H` → segment, a two-level DAG; no SCC contains a channel edge.
**DMA channels**: per PE, inbound `{A2L1, B2L1}` = 2, outbound `{C2L3}` = 1. Within budget.

**Summary** (the golden's first six lines — the delivery block, then the residency block of
§3.9; `T = (k0,)` is W1's only temporal tile axis):

```
A: multicast along py (derived)
B: multicast along px (derived)
C: stationary (declared)
A: multicast along py, re-fetched per k0
B: multicast along px, re-fetched per k0
C: stationary (spatial), resident for the whole run
```

### 6.2 W1-flip — weight-stationary, cascade reduction

Same kernel text. `grid(PK)` with `PK = 4`, `place(px=ax.k0)`, `stationary("B")`,
`tile(ax.i, TM=32)`, `tile(ax.k, TK=16)`, **no tile on `j`** so `TN = N = 64`, and
`double_buffer("A")` — a **1-D** herd, one coordinate `tx`. On `npu1` the 1-D cap is `(4,)`
(`_trace.py:88-91`), so `physical_herd=(4,)`, `repeats=(1,)`, and the four cascade-linked cores
are one **row** of columns — which is why the chain **ascends** (§3.6.3, measured
REVIEW-round1 P-R3).

**`j` is untiled because otherwise the flip is not weight-stationary** (RULING 9). With
`tile(ax.j, 32)` the declared `B` still satisfied the *spatial* predicate
`ker M_B ⊆ ker Sπ_u`, but its `[16,32]` tile moved with `j0`, so the weights were re-fetched on
every inner trip and were never resident. Leaving `j` whole makes each PE's tile
`B[kchunk, :] = [TK, N] = [16, 64]`, constant over the whole temporal sweep. The only temporal
tile axis left is `i0` (two trips, `M/TM = 2`), and `A` is the operand that moves along it —
hence `double_buffer("A")` and nothing else.

**Delivery**, all computed in `UCoord` (§3.2): `M_A = (i,k)`, `M_B = (k,j)`, `M_C = (i,j)`;
`Sπ_u = [e_k]`, so `ker Sπ_u = span{e_i, e_j}`.

* `A: STATIONARY (derived)` — `ker M_A = span{e_j} ⊆ ker Sπ_u`; each PE holds its own `[TM,TK]`
  slab for its own `k`-slice. **Residency `re-fetched per i0`** (§3.9): `M_A · e_i ≠ 0`, so the
  slab moves with the `i0` sweep. That is what makes it the one ping-pong candidate.
* `B: STATIONARY (declared)` — `ker M_B = span{e_i} ⊆ ker Sπ_u = span{e_i, e_j}`, so the
  containment **holds** and M3's `STATIONARITY` check accepts the declared clause. **This closes
  B-O5**: the *spatial* predicate is the right one, and the temporal half of the old question is
  answered separately by the residency line — **`resident for the whole run`**, because
  `M_B · e_i = 0` and no other temporal tile axis exists. The 2-D `place(px=ax.i0, py=ax.k0)`
  form, where `e_i0 ∉ ker Sπ` made the clause false, is abolished.
* `C: CASCADE along px (derived)` — `R_space = span{e_k}` (`03-lld-M3-checker.md` §6.2).

**Buffers**, all `herd.private`:

| name | operand | shape | bytes | `loop_depth` | `ping_pong_candidate` |
|---|---|---|---|---|---|
| `acc` | `C` | `(32,64)` | 8192 | 0 | False |
| `a` | `A` | `(32,16)` | 2048 | 1 | **True** |
| `b` | `B` | `(16,64)` | 4096 | 0 | False |
| `recv` | — (protocol) | `(32,64)` | 8192 | 0 | False |

L1 = `8192 + 2·2048 + 4096 + 8192` = **24 576** of 65 536 (`02-hld.md` §7). `recv` is the
cascade's receive tile and is synthesised here, which is why M3's own figure is `16 384` and the
plan's is `24 576`. `a` is the only doubled buffer: it is the direct child of the `i0` loop and
its first touch is a `get`, so `PP()` (§3.3) returns True. `PP("B")` would be False on
`DEPTH == 0` — and `double_buffer("B")` never reaches M4, because M3 rejects it with
`PINGPONG-SHAPE` at condition 2 (`03-lld-M3-checker.md` §3.13).

**Channels**:

| name | `size` | `broadcast_shape` | `channel_type` | `chain_direction` | sites |
|---|---|---|---|---|---|
| `A2L1` | `(4,)` | `None` | `None` | `None` | segment put at `(pk,)` under a Python bundle loop whose body is a 2-trip `i0` `air.sequential` — §6.1's nesting, with `i0` where W1 has `k0`; herd get at `(tx,)` inside the herd's `i0` loop |
| `B2L1` | `(4,)` | `None` | `None` | `None` | segment put at `(pk,)` ×4, **once**, before the herd; herd get at `(tx,)` once, at the top of the body |
| `CascadeK` | `(3,)` | `None` | **`"npu_cascade"`** | **`"ascending"`** | see below; one put and one get per link **per `i0` trip** |
| `C2L3` | `(1,)` | `None` | `None` | `None` | herd put at `(0,)` guarded `tx == PK-1`, once per `i0`; segment get ×2 under an **unrolled** drain loop (no L1 buffer is reused, D-3), region `((i0*32, 0), (32,64), (64,1))` |

`size=(PK-1,)=(3,)` is one index per **link**, with no dead index — R-A, and confirmed by the
three `aie.cascade_flow` ops the probe emits.

**Herd body**, in `order` — the shape is what the residency line claims:

```
0  BufferPlan b                                    # depth 0: fetched once, never re-fetched
1  ChannelSite(get, "B2L1", indices=(tx,), buffer="b", region=empty, scope="herd", order=0)
2  BufferPlan acc                                  # depth 0, [32,64]
3  BufferPlan recv                                 # depth 0, the cascade receive tile
4  LoopPlan(axis="i0", 0..64 step 32, kind="sequential", depth=0, body=[     # 2 trips
5      BufferPlan a                                # DIRECT child of the i0 loop  (ping-pong)
6      ChannelSite(get, "A2L1", indices=(tx,), buffer="a", region=empty, scope="herd", order=1)
7      <zeroing StoreNodes over acc[i1,j]>
8      LoopPlan i1 (0..32), LoopPlan j (0..64), LoopPlan k1 (0..16)
                        -> StoreNode(acc[i1,j] = acc[i1,j] + a[i1,k1]*b[k1,j])
9      <the cascade branch, below> ])
```

*Erratum, 2026-09-13 (B-P18): lines 7-8 are named by the post-tiling axis each loop realises —
`i1`, the **untiled** `j`, and `k1` — where this section first printed `m`, `n`, `t`.*

Lines 5-6 are the ping-pong shape (VF §E.2 items 2, 3, 5): the alloc is a direct child of the
streaming loop and its first touch is a `get`. Line 0 is deliberately **not**: hoisting `b`
above the loop is the whole point of the flip.

**Cascade sites**, ascending (head at `tx = 0`, tail at `tx = PK-1 = 3`), inside the `i0` loop:

```
Guard(tx == 0):        put CascadeK[tx]   <- acc
Otherwise:             get CascadeK[tx-1] -> recv
                       LoopPlan i1 (0..32) -> LoopPlan j (0..64)
                                          -> StoreNode(acc[i1,j] = acc[i1,j] + recv[i1,j])
                       Guard(tx == 3):    put C2L3[0] <- acc
                       Otherwise:         put CascadeK[tx] <- acc
```

The payload is one `[32,64]` `f32` partial tile — `8 192 B` per link per `i0` trip. Index ranges:
puts at `tx ∈ {0,1,2}` (only `tx ≤ 2` reaches a put), gets at `tx-1 ∈ {0,1,2}` (only `tx ≥ 1`
reaches a get). Balance per index `k`: 1 put (PE `k`) and 1 get (PE `k+1`) per `i0` trip, so
2 and 2 over the run; `A2L1[pk]` is 2 puts / 2 gets, `B2L1[pk]` is 1 / 1, `C2L3[0]` is 2 / 2
(the tail puts one `C[i0 tile, :]` per trip and the segment drains both). The lowered design
carries `aie.cascade_flow(%tile_0_2, %tile_1_2)`, `(1,2)→(2,2)`, `(2,2)→(3,2)` — three links,
**ascending**, `aircc` exit 0 (P-R3). The descending variant of the same module fails
`'aie.cascade_flow' op source tile must be to the North or West of the destination tile`.

**The accumulate is an explicit loop nest, not a whole-tile op.** `acc[:] = acc[:] + recv[:]` is
not a `StoreNode` — there is no whole-buffer form (`06-interfaces.md` §5.5). M4 expands it here,
into a two-deep `LoopPlan` over `(i1, j)` carrying one scalar `StoreNode` whose expression is
`BinOp("+", Load(acc,(i1,j)), Load(recv,(i1,j)))`. M5 then has nothing to decide (D-14). This is
the one arithmetic node M4 synthesises rather than reading off `Statement.expr`, because no
kernel statement corresponds to it — like the accumulator zeroing.

**The `i0` sweep** is a 2-trip `air.sequential` **inside** the herd body: with `k` placed and `j`
untiled, each PE walks the `M/TM = 2` row-blocks of `C` itself, each the full `[32, 64]` block,
accumulating its own `k`-slice into `acc` before the cascade. The `acc` zeroing is an M4-produced
`LoopPlan` of `StoreNode(Const 0.0)` at the top of each `i0` trip.

**Summary** (the delivery block, then the residency block of §3.9):

```
A: stationary (derived)
B: stationary (declared)
C: cascade along px (derived)
A: stationary (spatial), re-fetched per i0
B: stationary (spatial), resident for the whole run
C: cascade along px, re-fetched per i0
```

`M_C · e_i ≠ 0`, so §3.9 gives `C` the same duration as `A`. For a **written** operand the
phrase means *the tile it holds changes with that axis*: `acc` is allocated once, but it is
re-zeroed at the top of each `i0` trip and drained to a different `C[i0 tile, :]` region at the
end of it. Only `B` is resident for the whole run, and that is the flip's claim.

**DMA channels**: cascade puts and gets do **not** consume DMA channels (they lower to
`aie.put_cascade`/`aie.get_cascade`, `AIRToAIEPass.cpp:6955-7023`), so per PE inbound is
`{A2L1, B2L1}` = 2 and outbound is `{C2L3}` on `tx == PK-1` only = 1. *(Erratum, 2026-09-13,
ruling **R-F-4**: §3.8's P3 check therefore **skips** every `channel_type == "npu_cascade"`
site in both directions. Without that carve-out the chain is core-to-core, hence
circuit-switched by `CIRCUIT`, and PE 1 names three inbound channels against 2 S2MM — a
`DMA-CHANNELS` error on a design `aircc` compiles, measured exit 0 on both targets.)*

*Two errata on the node listings above, 2026-09-13.* (a) **Site orders.** `ChannelSite.order` is
the index in its **enclosing body** (`06-interfaces.md` §5.2 at v4, P0c reading 12), so the
`B2L1` get on herd-body line 1 carries `order=1`, not `0`, and the `A2L1` get inside the `i0`
loop carries `order=1` there. The four cascade sites are numbered from the outer `BranchNode`'s
own index in its body — `4`, `5`, `6`, `7` on the flip — which is what keeps every
`ChannelSite.id` distinct when one body holds two exclusive arms (phase P4's reading 3, extended
to a nest). (b) **The drain loop.** `C2L3`'s segment-scope drain is `air.sequential(0, M, TM)`
named `i0_drain`, **not** an unrolled loop: `LOOP_KIND` (§3.5) unrolls a loop only when its
variable is a channel bundle index, and the drain's index is the constant `0`. Its region is
`((i0_drain, 0), (32,64), (64,1))` in **element** units, which is the form `_tile_loop` produces
and the form `03-lld-M5-emitter.md` §6.2 line 38 prints. Ruling **R-F-1**.

### 6.3 W2 — Jacobi 5-point with halo exchange

`H=W=16`, `HS=8`, **`PI=2`** (finding N-1; the `PI=4` schedule becomes a `DMA-CHANNELS` negative
fixture), `T=4`, `f32`. `physical_herd=(2,)` on `npu1`, `repeats=(1,)`.

**Delivery**: `("U", STATIONARY, None, True)`, rendered `U: stationary (declared)`. *(Erratum, 2026-09-13, the ruling on **B-P22**: the summary's delivery block is the mechanical `f"{a}: {HOW} ({declared})"` line per operand and nothing else — protocol facts appear through the channel lines of §3.9 line 14. `exchange("U", …)` names `U`'s delivery, so under R2 — "`declared` is a fact about the schedule: a clause names this operand's delivery" — the row is `declared`; the earlier two-facts-in-one-sentence form predates the one-bit flag.)*

**Tensors**: `(U,)` — the kernel's one parameter, `[T+1, H+2, W] f32` = `[5, 18, 16]`, level
`L3`, scope `"tensor"`. Read **and** written, so §3.3's read-before-write ordering is vacuous.

**Buffers**: `cur (10,16) f32 herd.private depth 0`, `next (10,16) f32 herd.private depth 0`, both
`ping_pong_candidate = False` — they sit **outside** the `t` loop, so they are not
`isPingPongCandidate` candidates, and `double_buffer("U")` means D-5's second thing: the emitter
writes an explicit `cur`/`next` pair, and the checker's message says exactly that. L1 =
`2 · (HS+2) · W · 4 = 2 · 10 · 16 · 4 = 1280` of 65 536 (`02-hld.md` §7).

**Channels**: `UIn (2,)`, `UOut (2,)`, `ToNorth (1,)`, `ToSouth (1,)`. At `PI=2` there is one link
each way; the index tables of §3.6.1 still apply verbatim.

**Herd body**:

```
0  BufferPlan cur ; BufferPlan next
1  ChannelSite(get, "UIn", indices=(tx,), buffer="cur", region=empty, scope="herd")
1a LoopPlan i1 (0..HS+2) -> LoopPlan j (0..W) -> StoreNode(next[i1,j] = next <- cur)
2  LoopPlan(axis="t", 0..4 step 2, kind="sequential", depth=0, body=[
3      STEP(src=cur,  dst=next, t_off=0)     # orders 0..5 below
4      STEP(src=next, dst=cur,  t_off=1) ])
   # nothing after the loop: every plane was drained inside it

STEP(src, dst, t_off) =
  0  put  ToNorth[tx-1] <- src[1:2, :]      guard tx > 0,        is_async=True
  1  put  ToSouth[tx]   <- src[HS:HS+1, :]  guard tx < PI-1,     is_async=True
  2  get  ToNorth[tx]   -> src[HS+1:HS+2,:] guard tx < PI-1,     depends_on=()
  3  get  ToSouth[tx-1] -> src[0:1, :]      guard tx > 0,        depends_on=()
  4  LoopPlan i1 (0..HS) -> LoopPlan j (1..W-1) -> StoreNode(
         dst[i1+1,j] = 0.2*(src[i1+1,j]+src[i1,j]+src[i1+2,j]+src[i1+1,j-1]+src[i1+1,j+1]))
  5  put  UOut[tx] <- dst[1:HS+1, :]
         region on the L3 side = U[t + t_off + 1, tx*HS+1 : (tx+1)*HS+1, :]
```

*(Erratum, 2026-09-13, phase P5, two edits, both forced by the general machinery this section
is supposed to describe.)*

*Line 1a — the pair is **seeded**, not just `cur`.* The coverage paragraph below says of the
drain that "the value written back is the one that was staged in". That is a claim about every
plane, and the planes alternate between the two strips, so both have to carry the staged
read-only boundary; `next` is never a `get` target, so a `StoreNode` copy is what gives it those
values. Without it the boundary columns of every second plane — and the domain-edge ghost rows
of PE `0` and PE `PI-1`, which no neighbour ever fills — are whatever the alloc happened to
hold, and the Dirichlet boundary is not read-only at all. It is the same device the accumulator
zeroing and the cascade accumulate use: an arithmetic node M4 synthesises because no kernel
statement corresponds to it.

*Line 4 — the nest is `i1 ∈ [0, HS)` with the ghost offset in the subscript.* The loop is named
by the post-tiling axis it realises (B-P18), and `_compute_nest`'s rule for a tiled axis is
`lo = 0, hi = <tile extent>`; `l1_subscripts` then puts the halo back, giving `dst[i1+1, j]`.
The earlier `i (1..HS+1)` with `dst[i,j]` is the same set of rows written the other way round,
and it is the one form the general machinery cannot produce.

The update at order 4 is the **kernel's** five-term `0.2 ×` stencil
(`03-lld-M1-frontend.md` §6.2). The earlier four-term `0.25 ×` form in this section was a
different program from the one the oracle runs, and is deleted.

Order 5 is the drain, and it is **inside** the `t` loop: every plane the kernel writes reaches
L3, so `test_sem_coverage` compares the whole write domain rather than one plane. Because the
drain is per-`STEP`, there is no `<live>` question at all — the buffer parity is resolved inside
the unrolled body. **For `T = 5`** the plan is line 2 with `hi = 4`, followed at the same depth by
one straight-line `STEP(cur, next, t_off=0)` whose drain targets plane `5`.

**Balance**, by enumeration over `tx ∈ {0,1}` at `PI=2`: `ToNorth` index 0 — put by PE 1 (`tx>0`
holds), get by PE 0 (`tx<1` holds); 2 per trip × 2 trips = 4 = `T`. `ToSouth` index 0 mirrored.
`UIn[tx]`: 1 put (segment), 1 get. `UOut[tx]`: `T` puts (one per `STEP`), `T` gets at segment
scope under a Python loop over `t` and `tx`. **Acyclicity**: put-before-get within each `STEP`,
back edge excluded; the drain put has no get inside the herd, so it adds no cycle.
**DMA channels**: PE 0 inbound `{UIn, ToNorth}` = 2, outbound `{UOut, ToSouth}` = 2; PE 1
mirrored. Exactly at budget, with no MM2S sharing. At `PI ≥ 3` the interior PE would need
`{UIn, ToNorth, ToSouth}` = 3 circuit-switched inbound — §3.8 rejects it, and `aircc` is measured
to fail there (P-R2).

**Coverage**: the union over `(t, tx)` of the drained regions is
`{ U[t+1, tx·HS+1 : (tx+1)·HS+1, :] : t ∈ [0,T), tx ∈ [0,PI) }` = planes `1..T` × rows `1..H` ×
all `W` columns. The kernel writes planes `1..T` × rows `1..H` × cols `1..W-2`; the drained
region is a superset in the column direction only, because the strip is put whole. Columns `0`
and `W-1` are the read-only boundary and the value written back is the one that was staged in, so
the L3 image equals the oracle everywhere. `test_sem_coverage` compares against the **claimed**
drain domain and then that against the kernel's write domain (`04-test-plan.md` §3.4), and the
claim is this sentence.

*Fixture invariant **B-P25**, 2026-09-13 (ruling **R-F-5**).* That last clause holds only of an
input whose boundary is **time-invariant**. The plan carries plane 0's boundary forward — the
staged ghost rows at the domain edge and the seeded `next` are what every later plane reads at
rows `0`/`H+1` and columns `0`/`W-1` — while the kernel text reads plane `t`'s **own** boundary
out of the one rank-3 array. The two agree iff planes `1..T` of the input carry plane 0's
boundary rows and columns, which is what "read-only Dirichlet boundary" (`02-hld.md` §7.2) means
for a one-array kernel. Any fixture generator — `03-lld-M8-kernels-demo.md`'s `make_fixture.py`
included — must satisfy it, or the plan and the kernel text compute different things at the
boundary. `tests/integration/test_semantics.py`'s `w2_inputs` sets it explicitly.

### 6.4 W3 — Smith-Waterman anti-diagonal wavefront

`MQ=NR=32`, `PJ=4`, `CW=8`, `i32`. `physical_herd=(4,)`, `repeats=(1,)`.

**Delivery**: `S: forward along px (declared)`, `q: multicast along px (derived)`, `r: stationary (derived)` — three mechanical lines, in `MappingPlan.delivery` order (sorted by operand: `S`, `q`, `r`). *(Erratum, 2026-09-13, the ruling on **B-P22**: the delivery block carries one `f"{a}: {HOW} ({declared})"` line per operand and nothing else. `forward("S", …)` both names `S`'s delivery — R2, "`declared` is a fact about the schedule: a clause names this operand's delivery" — and replaces its derived row (FR-M3, §3.2 line 21a).)* *(Erratum, 2026-09-13, ruling **R-W3-2**, closing **B-P24**: `along` in a delivery row is always a PE axis, so §3.2 line 21a maps the clause's `ax.j0` through `mapping.schedule.place.index(...)` to `px`, and an `along` no `place()` carries is a `PROTOCOL-UNSUPPORTED` naming the clause.)*

**Tensors**: `(q, r, S)` — `q [32] i32` and `r [32] i32` are read-only and **must** precede the
written `S [33,33] i32` (§3.3 note 6; `_check_interface` raises otherwise, measured).

**Buffers**: `qb (32,) i32`, `rb (8,) i32`, `prev (9,) i32`, `cur (9,) i32`, `edge_in (1,) i32`,
`edge_out (1,) i32`, all `herd.private`, depth 0.
L1 = `128 + 32 + 2·36 + 2·4` = **240** bytes (`02-hld.md` §7).

**Channels** (§3.6.2, three of them — finding N-2):

| name | `size` | sites |
|---|---|---|
| `WestIn` | `(1,)` | segment: `LoopPlan(i, 1..MQ+1, "sequential")` → `put(S[i, 0:1], (0,))` — the zero column-0 boundary of `S` itself, not a synthetic `Zrow` tensor; herd: `get(edge_in, (0,))` guard `tx == 0` |
| `West` | `(3,)` | herd: `put(edge_out, (tx,))` guard `tx < 3`; `get(edge_in, (tx-1,))` guard `tx > 0` |
| `EastOut` | `(1,)` | herd: `put(edge_out, (0,))` guard `tx == 3`; segment: `LoopPlan(i_drain, 1..MQ+1, "sequential")` → `get(S[i, NR:NR+1], (0,))` — *(Erratum, 2026-09-13, ruling **R-W3-1**: the drain gets into the tail PE's own east edge column of `S`, not a synthetic `Sink` tensor; `MappingPlan.tensors` is exactly `(q, r, S)`. `S[i, NR]` is the same value the tail PE's `SOut` put also writes, and the shape is measured: `aircc --device npu1|npu2 --output-format=none` exits 0 with zero `error:` lines and `air-opt -pass-pipeline='builtin.module(air-verify-hierarchy-locality{strict=true})'` prints nothing.)* |
| `QIn` | `(1,)`, `broadcast_shape=(4,)` | segment: 1 put at `(0,)`; herd: `get(qb, (tx,))` once, before the row loop |
| `RIn` | `(4,)` | segment: 4 puts at `(k,)` under a Python loop; herd: `get(rb, (tx,))` once, before the row loop |
| `SOut` | `(4,)` | herd: `put(c[1:CW+1], (tx,))` **once per row**, L3 region `S[i, tx*CW+1 : (tx+1)*CW+1]`; segment: `MQ` gets per PE |

**Herd body**:

```
0  BufferPlan qb, rb, prev, cur, edge_in, edge_out
1  ChannelSite(get, "QIn", indices=(tx,), buffer="qb")     # q, whole, broadcast
2  ChannelSite(get, "RIn", indices=(tx,), buffer="rb")     # r[tx*CW : (tx+1)*CW]
3  LoopPlan(zero) -> StoreNode(prev[j] = Const 0)          # the S row-0 boundary
4  LoopPlan(axis="i", 1..MQ+1 step 2, kind="sequential", depth=0, body=[
5      ROW(p=prev, c=cur,  i_off=0)
6      ROW(p=cur,  c=prev, i_off=1) ])

ROW(p, c, i_off) =
  0  Guard(tx == 0):   get WestIn[0]  -> edge_in
     Otherwise:        get West[tx-1] -> edge_in
  1  StoreNode(c[0] = Load(edge_in, 0))
  2  LoopPlan(j, 1..CW+1, "sequential") -> StoreNode(
         c[j] = MaxMin("maximum", (Const 0,
                                   BinOp("+", Load(p, j-1),
                                         Select("==", Load(qb, i+i_off-1),
                                                      Load(rb, j-1),
                                                      Const MATCH, Const MISMATCH)),
                                   BinOp("-", Load(p, j), Const GAP),
                                   BinOp("-", Load(c, j-1), Const GAP))))
  3  StoreNode(edge_out[0] = Load(c, CW))
  4  Guard(tx == 3):   put EastOut[0] <- edge_out
     Otherwise:        put West[tx]   <- edge_out
  5  put SOut[tx] <- c[1:CW+1]        # L3 region S[i + i_off, tx*CW+1 : (tx+1)*CW+1]
```

Order 2 is the **kernel's** recurrence read from the staged `q` and `r`
(`03-lld-M1-frontend.md` §6.3): the substitution score is a `Select` over `qb[i-1] == rb[j-1]`,
not the literal `+2` an earlier draft carried, and `GAP` is a named constant rather than a
literal `1`. `rb` is indexed `j-1` because the buffer already holds this PE's slice, so the
`tx·CW` offset is in the staging region, not in the subscript.

Order 5 drains **every row**, so the union over `(i, tx)` is exactly rows `1..MQ` × cols
`1..NR` of `S` — the kernel's whole write domain, which is what `test_sem_coverage` needs.

The bundle index is `tx` or `tx-1`, **never** the row IV (H-3, `AIRDialect.cpp:3586-3593`). The
row loop uses the same unroll-by-two-and-peel as W2, so an odd `MQ` is legal; because the drain
is inside `ROW`, there is no `<live>` question.

**Balance**: `WestIn[0]` — `MQ` puts (segment loop), `MQ` gets (PE 0 only, under `tx == 0`).
`West[k]` for `k ∈ {0,1,2}` — `MQ` puts (PE `k`), `MQ` gets (PE `k+1`). `EastOut[0]` — `MQ` puts
(PE 3), `MQ` gets (segment loop). `SOut[tx]` — `MQ` puts, `MQ` gets. `QIn[0]` — 1 put, 1 get at
each of the 4 fan-out indices (the broadcast rule). `RIn[tx]` — 1 and 1. Five wavefront channel
indices, each with `put_count == get_count == MQ`, which is `test_M5_wavefront_balance` unchanged
in substance. **Acyclicity**: a left-to-right chain, acyclic by construction; the staging gets
precede the row loop and the drain puts leave the herd.
**DMA channels**: inbound `{QIn, RIn, (WestIn|West)}` = **3** (the two west gets are on exclusive
branches), outbound `{(West|EastOut), SOut}` = **2**. Three inbound is over the nominal 2-S2MM
budget and is a **warning**, not an error: `QIn` and `RIn` are L3→L1 and lower to packet flows,
which multiplex onto one shim MM2S, and the whole program is measured to compile — `aircc
--device npu1 --output-format=none`, exit 0, zero `error:` lines (REVIEW-round1 P-R4, §3.8).

---

## 7. Unit tests

`id → input → expected`. Every row is a pure-Python test needing no toolchain (I-1).

| id | input | expected |
|---|---|---|
| `test_M1_trichotomy` | W1 `LegalMapping` | `delivery == (("A",MULTICAST,"py",False),("B",MULTICAST,"px",False),("C",STATIONARY,None,**True**))` — `stationary("C")` names `C`'s delivery, so its row is `declared`; `A` and `B` are named by no clause (erratum 2026-09-13, the ruling on B-P17) |
| `test_M1_trichotomy_flip` | W1-flip `LegalMapping` | `delivery == (("A",STATIONARY,None,False),("B",STATIONARY,None,True),("C",CASCADE,"px",False))` — 1-D herd, `place(px=ax.k0)`, `j` untiled; `B` is `declared` and the containment holds in `UCoord` (§6.2, B-O5 closed) |
| `test_M2_broadcast_shape` | W1 | `ChannelPlan("A2L1", size=(2,1), broadcast_shape=(2,2))`; `B2L1` mirrored |
| `test_M2_broadcast_multiple` | hand plan, `size=(3,1)`, `broadcast_shape=(2,2)` | rejected before emission: `2 % 3 != 0` |
| `test_M3_stream_override` | W1 + `stream("A", pattern="forward", along=ax.j0)` | delivery row for `A` is `FORWARD` with `declared=True`; summary says `(declared)` |
| `test_M4_halo_protocol` | W2 | the per-`STEP` site list is exactly `[PUT(north), PUT(south), GET(north_ghost), GET(south_ghost), <update>, PUT(UOut)]`, `is_async=True` on both boundary puts, `depends_on == ()` on both ghost gets |
| `test_M4_balanced` (FR-M4) | W2 at `PI=2`, both boundary PEs | `self_check` reports `put_count == get_count` for **every** channel index — `ToNorth[0]`, `ToSouth[0]`, `UIn[0..1]`, `UOut[0..1]` — including the indices a guarded boundary PE never reaches, which must show `0 == 0` rather than being absent from the table |
| `test_M4_drains_every_plane` | W2 at `T=4` and `T=5` | the union of `UOut` L3 regions is planes `1..T` × rows `1..H`; `test_sem_coverage`'s claimed drain domain equals the kernel's write domain |
| `test_M4_stencil_is_five_point` | W2 | the update `StoreNode`'s expression is `BinOp("*", Const 0.2, <5-term sum>)`, and the five `Load`s are at offsets `(0,0), (−1,0), (1,0), (0,−1), (0,1)` |
| `test_M4_halo_indices` | W2 | `ToNorth.size == (PI-1,)`; put index `tx-1` guard `tx>0`; get index `tx` guard `tx<PI-1` |
| `test_M4_odd_T_peel` | W2 with `T=5` | the `t` `LoopPlan` has `hi=4, step=2`; one straight-line `STEP` follows it at the same depth and drains plane 5; balance holds with 5 puts and 5 gets per halo index |
| `test_M4_even_T_no_peel` | W2 with `T=4` | no peeled tail; 4 `UOut` puts per PE |
| `test_M5_wavefront_balance` | W3, `PJ=4` | 5 channel indices across `WestIn`/`West`/`EastOut`, each `put_count == get_count == MQ` |
| `test_M5_stages_q_and_r` | W3 | `QIn` is `size=(1,)` with `broadcast_shape=(4,)`; `RIn` is `size=(4,)`; the update's `Select` reads `qb` and `rb`, and no `Const 2` appears outside a `Select` arm |
| `test_M5_drains_every_row` | W3 | `SOut` carries `MQ` puts per PE and the union of its L3 regions is rows `1..MQ` × cols `1..NR` of `S` |
| `test_M4_tensor_order` | W3, W1, W2 | `plan.tensors` is `(q, r, S)` / `(A, B, C)` / `(U,)` — every read-only param before every written one (`06-interfaces.md` §5.6 invariant 6) |
| `test_M4_tensor_order_rejected` | a hand plan with `S` before `r` | `self_check` raises before emission, naming `_check_interface` and the offending pair |
| `test_M5_three_channels` | W3 | exactly three forward channels, none mixing an L3 and an L1 endpoint (finding N-2) |
| `test_M6_cascade_chain` | W1-flip, `grid=(4,)`, `PK=4` | one `ChannelPlan` with `channel_type="npu_cascade"`, `size=(3,)`, `broadcast_shape is None`, `chain_direction == "ascending"`; 4 guarded stages; the chain **ascends** in `tx`; the payload is one `[32,64]` `f32` tile per link per `i0` trip |
| `test_M6_cascade_orientation` | a 2-D `(1,4)` flip variant | `chain_direction == "descending"` when `len(grid) == 2`. Both directions are measured on the pinned wheel (REVIEW-round1 P-R3, `$PROBE/q/flip2.py`): 1-D must ascend, 2-D must descend, and the opposite of either fails `'aie.cascade_flow' op source tile must be to the North or West of the destination tile` |
| `test_M7_buffer_plan` | W1 | `acc` `herd.private` depth 0 not ping-pong; `a`,`b` `herd.private` depth 1 ping-pong |
| `test_M7_scope_never_shared` | all four plans | no `BufferPlan.scope == "herd.shared"`; no `"segment.per_core"` |
| `test_M8_channel_plan_complete` | W1, W1-flip, W2, W3 | every `ChannelPlan` field populated; every site's region rank matches its buffer rank |
| `test_M8_regions` | W1 | `A2L1` put region `((pi*32, kk), (32,16), (64,1))`; W2 `ToNorth` put region `((1,0),(1,W),(W,1))` |
| `test_M9_selfcheck_accepts` | the four real plans | `self_check` returns without raising |
| `test_M9_rejects_dropped_get` | W1 plan with one `C2L3` get removed | `BALANCE`, `details` has the per-key table showing `1 != 0` |
| `test_M9_rejects_extra_put_in_loop` | W1 plan with a second `A2L1` put inside the K loop | `BALANCE` citing the **per-iteration** rule (`4 != 8`) |
| `test_M9_rejects_guarded_put_only` | W2 plan with the guard removed from a get | `BALANCE` citing the **per-branch** rule, naming the coordinate |
| `test_M9_rejects_broadcast_underconsumed` | W1 plan with the `ty=1` `A2L1` get removed | `BALANCE` under the D-2 fan-out rule, naming the missing fan-out index |
| `test_M10_w2_acyclic` | W2 | no SCC contains a channel edge |
| `test_M10_cycle_rejected` | W2 plan with `[GET,GET,PUT,PUT]` order | `CHANNEL-CYCLE`, `details["cycle"]` lists the four sites of `GET_n(p)→PUT_s(p)→GET_s(p+1)→PUT_n(p+1)` |
| `test_M10_no_false_cross_scope_edge` | W1 | the segment producer loop and the herd are **not** joined by a program-order edge (rule 4); the plan is acyclic |
| `test_M10_branches_not_joined` | W3 | no program-order edge between the two guard regions of one branch (rule 3) |
| `test_M11_bundle_index_iv` | hand plan using the row IV as a bundle index | `BUNDLE-INDEX-IS-IV` naming the site and the loop |
| `test_M11_pingpong_shape` | W1 plan with `a` hoisted above the K loop but still `ping_pong_candidate` | invariant 4 fails before emission |
| `test_P3_dma_inbound` | W2 at `PI=4` | `DMA-CHANNELS` **error** naming PE 1, inbound `{UIn, ToNorth, ToSouth}`, budget 2, and the words `circuit-switched` and `2 S2MM` |
| `test_P3_dma_exclusive_branches` | W3 | PE 0's west inbound count is **1**, not 2 — the two gets are on exclusive branches |
| `test_P3_dma_packet_warns` | W3 | inbound is 3 and `self_check` **does not raise**: it records a warning naming `QIn`/`RIn` as packet-capable. A checker that rejected this would reject a program `aircc` accepts (P-R4, §3.8) |
| `test_tall_herd_is_refused_for_the_broadcast_guard_on_npu2` | W1's clauses at `grid(2, 4)` | `DMA-CHANNELS` naming `A2L1`, herd `(2, 4)`, split dim 0 and **2** rows unserved (B-P37) |
| `test_a_repeat_on_the_split_axis_takes_the_guard_out_of_range` | the same at `grid(2, 3)` on npu1 | physical `(1, 3)` with `repeats (2, 1)` plans — the exemption that keeps npu1 out of B-P37 |
| `test_four_row_column_is_refused_for_its_master_selects` | `grid(1, 4)`, both targets | `DMA-CHANNELS` with `flows == 5` against `budget == 4`, `multicast == ("A2L1",)` (B-P36) |
| `test_three_row_column_sits_on_the_master_select_cap` | `grid(1, 3)`, both targets | four flows, exactly on the cap, accepted — the control that makes B-P36 a count |
| `test_tall_herd_is_refused_by_aircc_and_the_square_one_is_not` (`slow`) | `grid(2, 4)` vs `grid(2, 2)`, npu2 | `aircc` prints *failed to get S2MM tile for L3 allocation* on the first and exits 0 on the second |
| `test_the_master_select_cap_is_where_aircc_stops` (`slow`) | `grid(1, 4)` vs `grid(1, 3)`, both targets | `aircc` prints *used up all its msels* on the first and exits 0 on the second |
| `test_M11_summary_golden` | W1 | summary matches the golden byte for byte and contains the three literal delivery lines |
| `test_M11_residency_line` | W1 and W1-flip | the residency block of §3.9, one line per operand: W1 prints `C: stationary (spatial), resident for the whole run` and `A: multicast along py, re-fetched per k0`; the **flip** prints `B: stationary (spatial), resident for the whole run` and `A: stationary (spatial), re-fetched per i0` — the two facts FR-K2's acceptance names. Also asserts `summary.residency == (("A","re-fetched per i0"),("B","resident for the whole run"),("C","re-fetched per i0"))` on the flip, and that a flip variant with `tile(ax.j, 32)` restored prints `B: … re-fetched per j0` — the bug RULING 9 exists to make visible |
| `test_M12_plan_stable` | W1 twice in one process, once in a fresh one with a different `PYTHONHASHSEED` | equal plans |
| `test_M4_l1_agrees_with_m3` | all four | the plan's recomputed L1 total equals `LegalMapping.l1_bytes` **plus** the protocol buffers M4 adds: W1 `12288 = 12288`, W1-flip `24576 = 16384 + recv 8192`, W2 `1280 = 1280`, W3 `240 = 232 + 8` (`02-hld.md` §7) |
| `test_M4_reside_l2` | W2 + `reside(U="L2")` | `PROTOCOL-UNSUPPORTED` whose `clause` is `reside(U="L2")` and whose `reason` names `reside` (FR-S12, §3.3 note 5) |

Negative tests key on `Diagnostic.code`, never on message prose (D-7).

---

## 8. Dependencies

**Consumed**: `LegalMapping` (`06-interfaces.md` §4.1) from M3, with A's hand-written literal as
the D0/D1 stub; the `model` dataclasses from M0; `numpy` for the integer linear algebra in §3.2.
Nothing else. M4 does **not** import `air` (I-1).

**`air.api` constructs the plan commits M5 to emitting.** M4 names none of them, but every plan
field exists to be translated into one, so the evidence belongs here too.

| construct | used for | evidence |
|---|---|---|
| `air.tensor(shape, dtype)` | `MappingPlan.tensors` | VF §D.2 (`_trace.py:1803`) |
| `air.launch(name=)` / `air.segment(name=)` | scope `"launch"`, `"segment"` | VF §D.1 (`_compile.py:44`, `_trace.py:1201`) |
| `air.herd(iterable, name=, shape=)` | `HerdPlan` | VF §D.1 (`_trace.py:1760`, `:1288-1291`, `:1363-1368`) |
| `air.herd(at=)` | `HerdPlan.at`, currently always `None` | VF §D.1 (`_trace.py:1771-1778`) |
| `air.alloc(scope=<herd>.private())` | `BufferPlan.scope == "herd.private"` | VF §D.2 (`_trace.py:1845`, `:1405-1414`) |
| `air.channel(name, size=)` | every `ChannelPlan` | VF §D.4 (`_channel.py:581`) |
| `air.channel(broadcast_shape=)` | multicast geometry | VF §A.4, §D.4 (`_channel.py:120-145`); `broadcast/single_herd/broadcast.py:44`; `broadcast_selective_capture.py:56-57` |
| `air.channel(channel_type="npu_cascade")` | `CascadeK` | VF §D.4 (`_channel.py:547`, `:170-177`); `cascade_reduction.py:60-62` |
| `Channel.put/get(obj, indices=, dependency=)` | every `ChannelSite` | VF §D.4 (`_channel.py:511`, `:524`) |
| `air.sequential(start, stop, step)` | `LoopPlan(kind="sequential")` | VF §D.5 (`_loop.py:89`, `:180`) |
| a Python `for` | `LoopPlan(kind="unrolled")`, segment scope only | VF §D.5 (`_loop.py:14-19`); `worker_to_worker.py:92-100` |
| `ops.branch(c)` / `head.otherwise()` | `Guard` | VF §D.6 (`_cond.py:18-28`, `:41-45`, `:49-54`, `:57-60`) |
| scalar `memref.load`/`store` under `air.sequential` | `StoreNode` | VF §D.11, §S1 (`_value.py:590-593`, `_emit.py:647-650`, `partial_assign.py:79-92`) |
| `ops.maximum` at element level | W3's 4-ary `max` | finding N-9 (probe `$PROBE/w3.py` builds) |

**Constructs deliberately not used**: `buffer_resources` (FR-E4, `air.api` raises,
`_channel.py:563`), `air.extern`/`link_with` (FR-E2, and an opaque callee as a buffer's first
toucher disqualifies ping-pong, VF §E.2 items 3-4), `ops.dot` (it lowers scalar anyway,
`_extern.py:11-14`, and `StoreNode` is the contract), `pad_before`/`pad_after` (unsupported on
channels, VF §D.4), `air.parallel` (no plan needs an `scf.forall`), `<segment>.shared()` and
`<segment>.per_core()` (open-questions §3).

**Stubs consumed**: A's hand-written `LegalMapping` literal for W1 from D0
(`05-work-breakdown.md` D0/D1), so M4 is unblocked if M3 slips. **Stubs produced**: B's
hand-written `MappingPlan` literal for W1, written at D0 against §6.1's table, which is what M5's
D1 skeleton drives and what the corrupted-plan negatives are derived from.

---

## 9. Implementation order, effort, definition of done

Effort is **estimated**, not measured. The WBS budgets M4 at 2.5 person-days on the critical path
(`05-work-breakdown.md` §7), and §3.8's new check adds to that.

| Step | What | Day | Effort (est.) | Unblocks |
|---|---|---|---|---|
| 1 | The hand-written W1 `MappingPlan` literal, validated against M0's invariants | D0 | 0.2 pd | M5's D1 skeleton |
| 2 | §3.2 `CLASSIFY` + §3.3 `BUFFER_PLAN` + §3.4 regions | D2 am | 0.4 pd | the W1 plan from source |
| 3 | §3.5 channel plan and loop kinds; W1 end to end (gate **G2**) | D2 pm | 0.4 pd | golden pipeline, `aircc` smoke |
| 4 | §3.7.1 balance + §3.7.2 acyclicity + §3.7.3 structural, with the six corrupted-plan negatives | D3 | 0.6 pd | gate G2's differentiator |
| 5 | §3.8 the DMA-channel check; the `PI=4` W2 negative fixture | D3 pm | 0.3 pd | keeps W2 out of a router crash |
| 6 | §3.6.2 wavefront, three channels + guards; W3 (gate **G3**) | D4 | 0.4 pd | W3 golden |
| 7 | §3.6.1 halo, put-first, unroll-by-two **and the odd-`T` peel** (gate **G4**) | D5 | 0.5 pd | W2 golden |
| 8 | §3.6.3 cascade, **ascending** chain on the 1-D `grid(PK)` herd; W1-flip (gate **G5**, stretch) | D6 am | 0.3 pd | the flip slide |
| 9 | §3.9 summary renderer against A's data-level renderer | D5-D6 | 0.2 pd | FR-M11 golden |
| | **total** | | **≈ 3.3 pd** | |

That is 0.8 pd over the WBS figure, all of it step 5 and the peel in step 7. Both are consequences
of findings this LLD's probes produced; the honest thing is to carry the number rather than hide
it. The slack comes from steps 6 and 8 being smaller than budgeted now that both shapes have been
proved to compile.

**Definition of done for M4.** All ten, and they are checked at the D6 freeze:

1. Every FR-M1…FR-M12 has its acceptance test from §7 passing.
2. All four plans (`W1`, `W1-flip`, `W2`, `W3`) pass `self_check` — balance, acyclicity, bundle
   indices, ping-pong shape, L1 budget, DMA channels — and their plan JSON matches its golden.
3. The six corrupted-plan negatives each raise the right code with the right `details`.
4. `test_M12_plan_stable` passes across two `PYTHONHASHSEED` values.
5. M4's module imports contain no `air` and no M5 (a lint test).
6. The W2 plan is correct for both an even and an odd `T`, with the peel exercised.
7. The cascade chain's orientation is correct for both a 1-D and a 2-D herd.
8. Every `Diagnostic` M4 raises has all four message parts populated (`test_D1_schema`).
9. The `MappingSummary` for all four variants matches its golden byte for byte.
10. Every construct in §8's table is either used and cited, or in the not-used list with its
    reason.

---

## 10. Open questions M4 owns

Q-1, Q-2, Q-6 and Q-7 are **closed** in `03-lld-B-open-questions.md` §1-§4. What remains:

| # | Question | Due | Fallback |
|---|---|---|---|
| **B-O3** *(closed)* | The architect's assent to the three restatements the probes force | — | **Closed**: all three are now in the requirement text — HLD §7.2's `size=[PI-1]` (EDIT-13), FR-M5's three homogeneous channels (EDIT-30), FR-M6/FR-E8's `PK−1` cascade **links** with `ir_facts.cascade_channels == 3` (EDIT-43) |
| **B-O5** *(closed)* | `stationary("B")` on W1-flip asserts *temporal residency*, not `ker M_B ⊆ ker Sπ` | — | **Closed by RULING 4, completed by RULING 9.** Under the adopted 1-D `grid(PK)`, `place(px=ax.k0)` form, `ker M_B = span{e_i} ⊆ ker Sπ_u = span{e_i, e_j}` in `UCoord`, so the containment reading holds as written and M3 accepts the declared clause; the 2-D `place(px=ax.i0, py=ax.k0)` form that made it false is abolished (§6.2). Temporal residency is not folded into the predicate — it is **reported** as the §3.9 residency line, and the fixture leaves `j` untiled so that `B` is in fact resident for the whole run |
| **B-O6** *(closed)* | `DMA-CHANNELS` needs adding to the frozen `06-interfaces.md` §6.3 | — | **Closed by RULING 5.** The code landed in `CONTRACT_VERSION = 2` **before** the D0 signature, together with `GRAMMAR-NONUNIFORM-DEP`; the catalogue is 43 codes and the `PROTOCOL-UNSUPPORTED` fallback is deleted (§3.8). `SWAP-PARITY` stays reachable through the `w2_zero_t` (`T = 0`) fixture (`03-lld-M8-kernels-demo.md` §4) |
| **B-O7** | Whether a `PI ≥ 3` halo is reachable at all on npu1 — the two candidates are `npu_dma_packet` channels (which multiplex several flows onto one DMA channel, which is why `llms/.../o_gemv_ffn_int4_fused` uses them) and a seeding wave that hands strips east along `ToSouth` before the `t` loop | after the freeze | Out of scope for the week. `PI = 2` is the fixture; `PI = 4` is a negative test; the honest-limits slide names the limit and the two candidates |
