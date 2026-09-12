# LLD M2 — Schedule builder / schedule IR

*Phase 1 low-level design, 2026-09-12. Owner: **Person A**. Reads on top of
[`01-requirements.md`](01-requirements.md) §3.1 (FR-S5…S20), [`02-hld.md`](02-hld.md) §2 (M2),
§5 and §7, and [`06-interfaces.md`](06-interfaces.md) §3 and §7 — **frozen**, and the single
source of truth for every field and signature named here. Field lists are cited, never
restated.*

---

## 1. Purpose and the FR IDs it satisfies

M2 is the clause API. It validates every clause argument at call time, accumulates the clauses
as **pure data**, and produces a `ScheduleModel` (`06-interfaces.md` §3.2) that is canonical,
hashable and JSON-serialisable. It performs **no legality reasoning** (`place` records the rows
of `Sπ`, it does not check them) and **no lowering** (building a schedule must not import `air`).

| FR | What M2 does for it |
|---|---|
| **FR-S5** | `sp.schedule(kernel, target=)`, `Schedule.axes()`, clause accumulation as pure data |
| **FR-S6** | `grid(PI[, PJ])` with the rank-2 cap |
| **FR-S7** | `tile(ax, F)` and the handles it creates (§3.2) |
| **FR-S8** | `place(px=, py=)` |
| **FR-S9** | `reduce(ax, op=)` — the argument domain half; the "axis must accumulate" half is M3's |
| **FR-S10** | `stationary(name)` |
| **FR-S11** | `stream(...)` / `forward(...)` and the delivery override record |
| **FR-S12** | `reside(**levels)` |
| **FR-S13** | `double_buffer(*operands)` — the *recording* half; the assertion is M3's (§3.6 of the M3 LLD) |
| **FR-S14** | `pipeline(ax)`, `sequential(ax)` |
| **FR-S15, S16** | `window(...)`, `exchange(...)` |
| **FR-S17** | `skew(time=(...))` |
| **FR-S18** | ignorability — M2's half: no clause touches the kernel, its module or any global |
| **FR-S19** | every clause validates its arguments and raises `ClauseError` with four parts |
| **FR-S20** | the schedule does not lower: no `air` import, no MLIR context, no checker run |
| **FR-D1** | every M2 rejection carries a full `Diagnostic` |
| **NFR-1, NFR-5** | canonical ordering (§3.5); docstrings on every clause |

---

## 2. Public entry points

Signatures are `06-interfaces.md` §7.1, verbatim and unchanged. One-line contracts:

| Name | Contract |
|---|---|
| `sp.schedule(kernel, target="npu1")` | build an empty `Schedule` bound to one `Kernel`; reject a `target` outside the `Target` set |
| `Schedule.axes()` | the live handle namespace: one attribute per post-tiling axis name, growing as `tile` is called (§3.2) |
| `grid`, `tile`, `place`, `reduce`, `stationary`, `stream`, `forward`, `reside`, `double_buffer`, `pipeline`, `sequential`, `window`, `exchange`, `skew` | each validates, records, and returns `self` so calls chain; each raises only `ClauseError` |
| `Schedule.model` | the canonicalised `ScheduleModel`; **rebuilt on each access** from the recorded clauses, so it is always in canonical form and never a mutable view |
| `Schedule.check()` / `.plan()` / `.summary()` / `.mlir()` / `.emit()` / `.build()` | the lowering entry points; M2 owns only the *dispatch* — they call M3, M4, M5, M6 in that order and are the **only** places `air` may be imported (FR-S20, FR-L13) |

`tile` is the one clause that does not return `self`: it returns the `(outer, inner)` handle pair
(§7.1). It also registers both handles on the namespace, so the W1 schedule text can write
`s.tile(ax.i, TM)` and then `s.place(px=ax.i0, ...)` without capturing the return value.

---

## 3. Internal design

### 3.1 State

A `Schedule` holds exactly: the `Kernel`, the resolved `target`, an `Axes` namespace object, and
one append-only list of `(clause_name, kwargs)` records in call order. Nothing else. The
`ScheduleModel` is a **projection** of that list (§3.5), computed on demand; there is no second
copy of the truth to drift.

### 3.2 The `AxisRef` model after `tile`

`AxisRef` (`06-interfaces.md` §3.1) is an opaque handle carrying a `name` and the owning
schedule's identity. M2 adds one internal structure the contract does not name: an **axis
table**, a mapping `name → (parent | None, factor | None, extent | None)` seeded from
`kernel.model.axes` and extended by `tile`.

```pseudo
TILE(ax, F):
 1  require ax is an AxisRef of THIS schedule          else CLAUSE-UNKNOWN-AXIS
 2  require F is an int and F >= 1                     else CLAUSE-BAD-VALUE
 3  e := extent of ax in the axis table
 4  if e is None:                                      # unresolvable bound, M1 §3.4 line 20
 5      raise CLAUSE-TILE-DIVIDES, reason "axis <ax> has no constant extent, so a tile factor
 6          cannot be checked"                         # FR-S7's "otherwise ... rejected"
 7  if e % F != 0:
 8      raise CLAUSE-TILE-DIVIDES with details {axis, extent: e, factor: F, divisors: divisors(e)}
 9  outer := ax.name + "0" ; inner := ax.name + "1"    # §2.2's naming rule
10  require neither name is already in the axis table  else CLAUSE-DUPLICATE
11  axis_table[outer] := (parent=ax.name, factor=F, extent=e // F)
12  axis_table[inner] := (parent=ax.name, factor=F, extent=F)
13  register handles ax.<outer>, ax.<inner> on the namespace
14  record ("tile", ax.name, F)
15  return (handle(outer), handle(inner))
```

Four rules fall out, and they are the whole of the model:

1. **`tile` adds handles; it never removes one.** `ax.i` stays valid after `tile(ax.i, TM)`. It
   has to: W1 calls `reduce(ax.k, op="+")` *after* `tile(ax.k, TK)`, and W2 calls
   `window(U, dims=(ax.i, ax.j), …)` after `tile(ax.i, HS)`.
2. **A parent handle means "the whole axis".** `reduce`, `window` and `stationary` are defined
   over original axes and take parents naturally.
3. **A parent handle passed where a *spatial* axis is wanted normalises to its outermost leaf.**
   `place(px=ax.i)` after `tile(ax.i, F)` records `"i0"`. This is unambiguous (the outer handle
   is the only one that indexes PEs) and costs no new error code; if the grid extent then
   disagrees with `i0`'s extent, M3 raises `PLACE-EXTENT` with both numbers, which is the
   message the user needed anyway.
4. **Re-tiling is allowed**: `tile(ax.i1, F2)` yields `i10`, `i11` by the same rule (the test
   plan asks for it, §2 M2).

`tiles` is recorded **in call order** (§3.2's invariant), because tiling a handle that an
earlier `tile` created is a genuine data dependence. Every other clause list is canonicalised
(§3.5).

### 3.3 Clause-by-clause argument validation (FR-S19)

Every clause runs the same three-step preamble — *resolve, domain-check, conflict-check* — and
then appends one record. The table is the specification; the shared helpers are
`AXIS(x)` (handle → name, else `CLAUSE-UNKNOWN-AXIS`), `OPERAND(s)` (str → `Param.name`, else
`CLAUSE-UNKNOWN-OPERAND` **listing the kernel's parameter names**, which FR-S10's acceptance
test asserts), `ENUM(v, domain)` (else `CLAUSE-BAD-ENUM` naming the domain) and
`POS(v)` (else `CLAUSE-BAD-VALUE`).

| Clause | Resolve | Domain | Conflict |
|---|---|---|---|
| `sp.schedule(k, target)` | — | `ENUM(target, Target)` → `CLAUSE-BAD-ENUM` naming `npu1, npu2, auto` | — |
| `grid(*extents)` | — | `1 <= len <= 2` else `CLAUSE-GRID-RANK` with the reason *"air.api supports 1-D and 2-D herd grids"* (`_trace.py:1288-1291`); each `POS` | a second `grid` → `CLAUSE-DUPLICATE` |
| `tile(ax, F)` | `AXIS` | `POS(F)`; §3.2 lines 4–8 | handle name collision → `CLAUSE-DUPLICATE` |
| `place(px, py=None)` | `AXIS` ×n, normalised per §3.2 rule 3 | `len(place) == len(grid)` else `CLAUSE-RANK` (details: both ranks); `grid` must exist else `CLAUSE-RANK` | same axis twice, or a second `place` → `CLAUSE-DUPLICATE` |
| `reduce(ax, op="+")` | `AXIS` | `ENUM(op, ReduceOp)` naming `+, max, min` | same axis twice → `CLAUSE-DUPLICATE` |
| `stationary(name)` | `OPERAND` | — | same operand twice → `CLAUSE-DUPLICATE` |
| `stream(name, pattern, along, depth=None)` | `OPERAND`, `AXIS(along)` | `ENUM(pattern, Pattern)`; `POS(depth)` if given; `along` **must be a placed axis** else `CLAUSE-BAD-VALUE` with the reason *"`along=` must name a placed axis; placed axes are …"* (FR-S11's `test_stream_along_must_be_placed`) | a second `stream`/`forward` on one operand → `CLAUSE-DUPLICATE` |
| `forward(name, along, dir, depth=None)` | as `stream` | `ENUM(dir, Direction)` naming the four spellings | as `stream` — they share one slot |
| `reside(**levels)` | `OPERAND` per key | `ENUM(level, Level)`; `level == "L3"` is accepted here and rejected by M3/M4 only if something tries to `alloc` it (FR-S12) | a second `reside` for one operand → `CLAUSE-DUPLICATE` |
| `double_buffer(*names)` | `OPERAND` ×n | — | duplicates within one call are collapsed; a repeat across calls is not an error |
| `pipeline(ax)` | `AXIS` | — | repeat → `CLAUSE-DUPLICATE` |
| `sequential(ax)` | `AXIS` | — | repeat → `CLAUSE-DUPLICATE`; overlap with `place` is **not** checked here (M3's `PLACE-SEQUENTIAL-CONFLICT`, so the model can be built in either clause order) |
| `window(name, dims, halo)` | `OPERAND`, `AXIS` per dim | `halo` is a non-negative `int`, or a tuple of non-negative ints with `len == len(dims)` else `CLAUSE-RANK`; negative → `CLAUSE-BAD-VALUE` | second `window` on an operand → `CLAUSE-DUPLICATE` |
| `exchange(name, along, halo)` | `OPERAND`, `AXIS(along)` | `halo >= 1` else `CLAUSE-BAD-VALUE`; `along` must be placed, as `stream` | second `exchange` on an operand → `CLAUSE-DUPLICATE` |
| `skew(time=(...))` | `AXIS` per term | `len(time) >= 1`; every term distinct else `CLAUSE-DUPLICATE` | a second `skew` → `CLAUSE-DUPLICATE` |

`place` and `stream`/`exchange` both need `grid`/`place` to already exist. Rather than force a
clause order on the user, the rule is: **a clause that depends on another clause's presence is
validated against whatever has been recorded so far, and re-validated when `model` is built**
(§3.5 line 9). `s.stream(...)` before `s.place(...)` therefore raises at `model` time, not at
call time, with the same code and message. This keeps FR-S19's "at call time" for every
*self-contained* check — which is all of the domain checks, i.e. the ones the acceptance tests
name — without making the surface order-dependent (§3.5).

### 3.4 What `place`, `skew` and `sequential` record — and who builds the matrices

M2 records **symbolic rows**, never integers:

* `place(px=a, py=b)` records the ordered axis-name tuple `("a", "b")`. That tuple *is* the row
  order of `Sπ`: row `r` is `e_{place[r]}` over the post-tiling coordinate order.
* `skew(time=(a, b))` records `("a", "b")`. That tuple *is* the first row of `Sσ`, as the **sum**
  `a + b` — FR-S17's wording, and HLD §7.3's `σ = i + j0`.
* `sequential(ax)` records an assertion about `ker Sπ`; it contributes **no row** to either
  matrix.
* `pipeline(ax)` records a hint and contributes nothing to either matrix (§3.7).
* The remaining σ rows come from the **default loop order**, defined in §3.6 below.

Turning those tuples into the integer matrices `Sσ`, `Sπ` — and computing `ker`, ranks, the
reduction split and everything else — is **M3's**, specified in
[`03-lld-M3-checker.md`](03-lld-M3-checker.md) §3.3. The split exists because HLD §2 assigns
"`Sσ`/`Sπ` construction" to M3 while M2 "records rows of `Sπ`, it does not check them"; keeping
the integers on one side of that line means there is exactly one place where a coordinate-order
bug can live.

### 3.5 Canonicalisation — determinism and ordering independence (NFR-1, FR-S5)

```pseudo
MODEL(schedule):                                   # runs on every `Schedule.model` access
 1  grid        := the single grid record, or None
 2  tiles       := the tile records, IN CALL ORDER                 # §3.2
 3  place       := the place record's axis names, px first         # argument order, not sorted
 4  skew        := the skew record's axis names, in argument order # argument order, not sorted
 5  reductions  := sorted by axis name
 6  stationary  := sorted
 7  streams     := sorted by operand
 8  residency   := sorted by operand
 9  double_buffer := sorted, de-duplicated
10  pipeline    := IN CALL ORDER                                   # hint; §3.2's field note
11  sequential  := sorted
12  windows, exchanges := sorted by operand
13  re-run every cross-clause check of §3.3's last column          # e.g. stream.along is placed
14  return ScheduleModel(...)                                      # frozen, hashable, JSON-able
```

**Ordering independence, stated precisely.** Two schedules that issue the same clause *set* in
different orders produce **identical** `ScheduleModel`s, with exactly two exceptions, both of
which are data dependences rather than free choices:

* `tiles` (line 2) — because `tile(ax.i1, 2)` is only expressible after `tile(ax.i, 4)`. Two
  `tile` calls on **unrelated** axes may be issued in either order and produce different
  `tiles` tuples; the test `test_M2_tile_order_is_observable` asserts this and the docstring
  says so, so nobody discovers it in a golden diff.
* `pipeline` (line 10) — a hint that nothing reads (§3.7), recorded in call order for
  faithfulness.

Everything else is sorted by an explicit key (HLD §5 rule 1). No `set` iteration, no `dict`
order, no `id()`, no timestamp, no RNG reaches any field (HLD §5 rule 5).

### 3.6 The post-tiling axis order and the default σ order

M2 defines both orders because both are functions of the *schedule* (which axes were tiled), and
M3 consumes them. They are stated here once and referenced everywhere.

**Coordinate order** — the column order of every matrix M3 builds over post-tiling axes:
take `kernel.model.axes` in source order and replace each tiled axis **in place** by its
`(outer, inner)` pair, recursively for nested tiles.

| Workload | Coordinate order | Extents |
|---|---|---|
| W1 | `(i0, i1, j0, j1, k0, k1)` | `(2, 32, 2, 32, 4, 16)` |
| W1-flip | `(i0, i1, j, k0, k1)` — **`j` is not tiled** (RULING 9) | `(2, 32, 64, 4, 16)` |
| W2 | `(t, i0, i1, j)` | `(4, 2, 8, 14)` |
| W3 | `(i, j0, j1)` | `(32, 4, 8)` |

Extents are the **iteration-domain** extents of the post-tiling axes, in exactly the order of
column 2, and they agree row-for-row with the worked examples in
[`03-lld-M3-checker.md`](03-lld-M3-checker.md) §6.1–§6.4. W2 is the row worth reading twice: its
order is `(t, i0, i1, j)` and the extents are `T = 4` written planes (`U` is declared
`sp.f32[T + 1, H + 2, W]`; plane 0 is read-only input), `PI = 2` row strips, `HS = 8` rows per
strip — `PI·HS == H == 16` — and `W − 2 = 14` interior columns, because `j` runs `1..W-2` and is
neither tiled nor placed.

**Default σ row order** — used when `skew` is absent: every **outer or untiled** handle in
source order, then every **inner** handle in source order.

| Workload | Default σ rows |
|---|---|
| W1 | `(i0, j0, k0, i1, j1, k1)` — exactly HLD §7.1's "σ default" |
| W1-flip | `(i0, j, k0, i1, k1)` — `j` is untiled, so it sorts with the outer handles |
| W2 | `(t, i0, j, i1)` — first term `t`, which is what HLD §7.2's causality argument needs |
| W3 | `(i, j0, j1)` — overridden by `skew`; see below |

**With `skew` present**: row 0 is the sum of the named axes; then one row per axis that is
**neither named in `skew` nor placed**, in default order. W3 therefore gets `σ = (i + j0, j1)` —
and that second row is what makes W3 satisfy (L1), since without it `e_{j1}` would sit in
`ker Sσ ∩ ker Sπ` (worked in [`03-lld-M3-checker.md`](03-lld-M3-checker.md) §6.4).

### 3.7 `pipeline` is inert

FR-S14 says `pipeline(ax)` is "a hint only", says it "shall place `ax` innermost in `σ`", and
then asserts `test_pipeline_is_hint` — *two schedules differing only in `pipeline` emit
byte-identical AIR text*. The three are not simultaneously satisfiable: moving an axis in `σ`
changes the loop order and therefore the text. HLD §7.1 settles it by printing W1's `σ` as
`(i0, j0, k0, i1, j1, k1)` **with `pipeline(ax.k0)` in force** — `k0` third, not innermost.

**Resolved here: `pipeline` is recorded in `ScheduleModel.pipeline` and read by nothing.** It
contributes no σ row, no reordering and no emission. Flagged in §10 (Q-M2-2) because it is a
wording defect in FR-S14, not a design choice.

### 3.8 FR-S20 — the schedule does not lower

Three mechanical guarantees:

1. The M2 module's import list contains no `air`, no `numpy` and no `mlir`. A lint test greps
   for them (`test_no_air_import_until_build`).
2. `air` is imported **inside** `Schedule.mlir()`/`.build()`, never at module scope, and never
   by `check()` — M3 is pure integer arithmetic and needs no toolchain.
3. `Schedule.check()`, `.plan()`, `.summary()`, `.mlir()`, `.build()` run **in that order** and
   short-circuit: `mlir()` calls `check()` first and returns before any `air` import if the
   check raises (FR-L13, whose `test_no_emission_on_illegal` spies on the emitter for zero
   calls).

Ignorability (FR-S18), M2's half: no clause writes to the `Kernel`, to `fn.__globals__`, to any
module attribute or to any argument. `Schedule` holds a reference to the `Kernel`; it never
mutates it. `test_schedule_is_pure_data` asserts the whole model equals a stored fixture after
every clause has been called.

---

## 4. Invariants, pre/postconditions

**Precondition** of every clause: the `Schedule` was built from a `Kernel` whose `model`
satisfies M1's postconditions ([`03-lld-M1-frontend.md`](03-lld-M1-frontend.md) §4).

**Postconditions of `Schedule.model`**, asserted once at the end of `MODEL`:

1. Every axis name mentioned by any field exists in the axis table (§3.2).
2. Every operand name mentioned by any field is a `Param.name` (`06-interfaces.md` §2.1).
3. The canonical orders of §3.5 hold field by field.
4. The model is **hashable** and contains no reference to the kernel function
   (`06-interfaces.md` §3.2's cross-model invariant), so it is JSON-serialisable by field order.
5. `MODEL` is idempotent: two accesses return equal models; an access does not mutate the
   schedule.
6. `len(place) == len(grid)` when both exist — enforced here and by FR-S8, **not** by M0's
   `__post_init__`; see §10 Q-M2-3.

**Invariant**: a clause either records exactly one item and returns, or raises `ClauseError` and
records nothing. There is no partially-applied clause.

---

## 5. Error paths

All M2 errors are `ClauseError(Diagnostic(...))` with `stage="clause"`, `clause` set to the
clause rendered as the user wrote it (e.g. `tile(ax.i, 3)`), `location` set from the caller's
frame when `inspect.stack()` gives one, and `details` carrying the numbers. The four parts of
HLD §4.2 are `reason` (what is wrong), `clause` (where), `because`/`details` (the actual values
and the accepted domain), and `fix` (one concrete clause edit).

| Code (`06-interfaces.md` §6.3) | Raised by | Message, four parts |
|---|---|---|
| `CLAUSE-UNKNOWN-AXIS` | `AXIS` | `axis handle 'q' is not an axis of this schedule` · `place(px=ax.q)` · details `{seen: "q", axes: ["i0","i1","j0","j1","k0","k1"]}` · `use one of the axes listed; tile handles are named i0/i1` |
| `CLAUSE-UNKNOWN-OPERAND` | `OPERAND` | `'D' is not a parameter of kernel gemm` · `stationary("D")` · details `{seen: "D", params: ["A","B","C"]}` · `name one of A, B, C` — FR-S10's acceptance asserts exactly this list |
| `CLAUSE-BAD-ENUM` | `ENUM` | `op='*' is not an accumulation operator` · `reduce(ax.k, op="*")` · details `{seen: "*", domain: ["+","max","min"]}` · `use op="+"; the operator must be associative and commutative (SD-02 §4)` |
| `CLAUSE-BAD-VALUE` | `POS`, `along` check | `tile factor 0 is not positive` / `along=ax.k0 is not a placed axis` · details `{placed: ["i0","j0"]}` · `place it first, or stream along px` |
| `CLAUSE-RANK` | `place`, `window` | `place has rank 1 but grid has rank 2` · details `{place_rank: 1, grid_rank: 2}` · `pass py=, or declare grid(PI)` |
| `CLAUSE-DUPLICATE` | every conflict column | `axis i0 is placed twice` · details `{axis: "i0", positions: ["px","py"]}` · `place two different axes` |
| `CLAUSE-TILE-DIVIDES` | `TILE` lines 5, 8 | `tile factor 3 does not divide the extent of axis i, which is 64` · `tile(ax.i, 3)` · details `{axis:"i", extent:64, factor:3, divisors:[1,2,4,8,16,32,64]}` · `use a factor from the divisor list, e.g. tile(ax.i, 32)` — FR-S7's acceptance asserts `tile`, `i`, `64`, `3` all appear |
| `CLAUSE-GRID-RANK` | `grid` | `grid of rank 3 is not supported` · details `{rank: 3}` · reason quotes *air.api supports 1-D and 2-D herd grids* (`_trace.py:1288-1291`) · `use grid(PI) or grid(PI, PJ)` |

No other exception leaves a clause (NFR-4, NFR-7). In particular a `TypeError` from passing a
`str` where an `AxisRef` is wanted is caught and re-raised as `CLAUSE-UNKNOWN-AXIS` with the
reason naming the type seen.

---

## 6. Worked examples — M2's exact output for W1, W1-flip, W2, W3

Field names and types are `06-interfaces.md` §3.2; only values appear here.

### 6.1 W1 — GEMM, output-stationary

```python
s = sp.schedule(gemm, target="npu1"); ax = s.axes()
s.grid(2, 2)
s.tile(ax.i, 32); s.tile(ax.j, 32); s.tile(ax.k, 16)
s.reduce(ax.k, op="+")
s.place(px=ax.i0, py=ax.j0)
s.stationary("C")
s.reside(A="L1", B="L1", C="L1")
s.double_buffer("A", "B")
s.pipeline(ax.k0)
```

| field | value |
|---|---|
| `target` | `"npu1"` |
| `grid` | `(2, 2)` |
| `tiles` | `(("i",32), ("j",32), ("k",16))` |
| `place` | `("i0", "j0")` |
| `reductions` | `(("k", "+"),)` |
| `stationary` | `("C",)` |
| `streams`, `windows`, `exchanges`, `sequential` | `()` |
| `residency` | `(("A","L1"), ("B","L1"), ("C","L1"))` |
| `double_buffer` | `("A", "B")` |
| `pipeline` | `("k0",)` |
| `skew` | `None` |

Derived, per §3.6: coordinate order `(i0,i1,j0,j1,k0,k1)`; σ rows `(i0,j0,k0,i1,j1,k1)`; π rows
`(i0, j0)`.

### 6.2 W1-flip — weight-stationary, cascade reduction

Same kernel text, **no edit** (FR-K2). `grid`, `place`, `stationary` and `double_buffer` change,
and the `tile(ax.j, …)` clause is **deleted** (RULING 9):

```python
s.grid(4)                       # PK = 4, a 1-D row herd
s.tile(ax.i, 32); s.tile(ax.k, 16)          # NO tile on j: TN = N = 64
s.reduce(ax.k, op="+")
s.place(px=ax.k0)
s.stationary("B")
s.reside(A="L1", B="L1", C="L1")
s.double_buffer("A")
```

`grid = (4,)`, `tiles = (("i",32), ("k",16))`, `place = ("k0",)`, `stationary = ("B",)`,
`double_buffer = ("A",)`; the rest as §6.1 with `pipeline = ()`.
Derived, per §3.6: coordinate order `(i0, i1, j, k0, k1)`, extents `(2, 32, 64, 4, 16)`;
σ rows `(i0, j, k0, i1, k1)`; π rows `(k0,)`.

**Why `j` is untiled.** With `tile(ax.j, 32)` the declared-stationary `B` would satisfy the
*spatial* predicate `ker M_B ⊆ ker Sπ` and still be re-fetched on every `j0` trip, because its
`[16,32]` tile moves with `j0`. Leaving `j` whole makes each PE's `B[kchunk, :]` tile constant
for the whole run — the weights genuinely stay put — and `A` becomes the one operand that
streams, which is why `double_buffer` names `A` and only `A`
(`03-lld-M3-checker.md` §3.13, `03-lld-M4-mapping.md` §3.9).

> **Finding, for the architect and Person C.** FR-K2's prose says
> `place(px=ax.i0, py=ax.k0)` **and** `stationary("B")`. Those two are inconsistent: with
> `π = (i, k)`, `ker Sπ = span{e_j}` and the operand whose reuse space it contains is **A**, not
> B (SD-02 §3.1's table, row 3 "row/input-stationary"; SD-02 §9's own flip says "with **A** now
> stationary"). `stationary("B")` needs `ker Sπ ⊇ ker M_B = span{e_i}`, i.e. `π = (k, j)` — which
> is SD-02 §3.1's weight-stationary row, is what the name "weight-stationary flip" means, and is
> what `04-test-plan.md` §4's fixture row (`PK=4, PJ=2`) places. FR-L6(c) then forces the herd to
> be 1-D or extent-1 on the other axis, so the published `grid (4, 2)` is itself rejected
> (`CASCADE-RANK`), and HLD §7.1's own words — *"3 `npu_cascade` channels along a **1-D row
> herd**"* — settle it: **`grid(4)`, `place(px=ax.k0)`**. That is the schedule used above and in
> [`03-lld-M3-checker.md`](03-lld-M3-checker.md) §6.2. `place(px=ax.i0, py=ax.k0)` survives as
> the **negative** fixture of FR-L3 (`STATIONARITY` on `stationary("C")`), which is where it is
> correct. **Adopted as REVIEW-round1 RULING 4**: FR-K2 now reads `grid(PK)`, `place(px=ax.k0)`,
> `stationary("B")`, and the containment holds in `UCoord` — `ker M_B = span{e_i} ⊆ ker Sπ_u =
> span{e_i, e_j}` — so B-O5 is closed. `04-test-plan.md` §4's fixture row and `02-hld.md` §7.1
> carry the same shape, and the chain **ascends** (measured P-R3).

### 6.3 W2 — Jacobi with halo exchange

```python
s = sp.schedule(jacobi, target="npu1"); ax = s.axes()
s.grid(2)
s.tile(ax.i, 8)
s.place(px=ax.i0)
s.sequential(ax.t)
s.window("U", dims=(ax.i, ax.j), halo=1)
s.exchange("U", along=ax.i0, halo=1)
s.reside(U="L1")
s.double_buffer("U")
```

Fixture, per **RULING 1**: `def jacobi(U: sp.f32[T + 1, H + 2, W])` with `H = W = 16`,
`PI = 2`, `HS = 8` (`PI·HS == H` exactly), `T = 4` — and the odd-`T` variant `T = 5`, which
changes no schedule field.

`grid = (2,)`; `tiles = (("i",8),)`; `place = ("i0",)`; `sequential = ("t",)`;
`windows = (("U", ("i","j"), (1,1)),)` — the scalar `halo=1` is broadcast to one entry per dim;
`exchanges = (("U", "i0", 1),)`; `residency = (("U","L1"),)`; `double_buffer = ("U",)`;
`reductions`, `stationary`, `streams`, `pipeline` empty; `skew = None`.
Derived, per §3.6: coordinate order `(t, i0, i1, j)`, extents `(4, 2, 8, 14)` — `T = 4` written
planes, `PI = 2` strips, `HS = 8` rows per strip, and `W − 2 = 14` interior columns;
σ rows `(t, i0, j, i1)`; π rows `(i0,)`.
`Sπ = [e_i0]` over that order, a single row, since `place` names one axis
([`03-lld-M3-checker.md`](03-lld-M3-checker.md) §6.3, which carries the per-core L1 figure of
**1 280 B**).

### 6.4 W3 — Smith-Waterman wavefront

```python
s = sp.schedule(sw, target="npu1"); ax = s.axes()
s.grid(4)
s.tile(ax.j, 8)
s.place(px=ax.j0)
s.skew(time=(ax.i, ax.j0))
s.forward("S", along=ax.j0, dir="W->E")
s.reside(S="L1")
```

`grid = (4,)`; `tiles = (("j",8),)`; `place = ("j0",)`;
`streams = (("S", "forward", "j0", "W->E", None),)` — `forward` is sugar for `stream` plus a
direction (FR-S11), so it occupies the single `streams` slot for `S`;
`residency = (("S","L1"),)`; `skew = ("i", "j0")`; everything else empty.
Derived: coordinate order `(i,j0,j1)`; σ rows `(i + j0, j1)`; π rows `(j0,)`.

---

## 7. Unit tests

| Test id | Input (concrete) | Expected |
|---|---|---|
| `test_schedule_is_pure_data` | every clause called on W1 | `model` equals the §6.1 fixture; no MLIR built; `sys.modules` has no `air` (FR-S5) |
| `test_no_air_import_until_build` | full schedule built in a subprocess | `air` absent from `sys.modules` afterwards (FR-S20) |
| `test_M2_models[w1, w1flip, w2, w3]` | §6.1–§6.4 sources | the four models, field by field |
| `test_M2_order_independent` | W1's clauses issued in 20 shuffled orders (respecting the `tile`→`place` data dependence) | all 20 models compare **equal** (FR-S5, NFR-1) |
| `test_M2_tile_order_is_observable` | `tile(ax.i,32); tile(ax.j,32)` vs the reverse | `tiles` differs; every other field equal; documented in the docstring (§3.5) |
| `test_tile_handles` | `tile(ax.i, 32)` then `tile(ax.i1, 8)` | handles `i0,i1,i10,i11` exist with extents `2,32,4,8`; both usable in later clauses |
| `test_M2_parent_handle_normalises` | `tile(ax.i,32)` then `place(px=ax.i)` | `place == ("i0",)` (§3.2 rule 3) |
| `test_pipeline_is_hint` | two W1 schedules differing only in `pipeline(ax.k0)` | equal `sigma` row order; identical emitted text (FR-S14, §3.7) |
| `test_skew_sets_sigma` | W3 | `model.skew == ("i","j0")`; derived σ rows `(i+j0, j1)` (FR-S17) |
| `test_stream_overrides_derivation` *(with M4)* | W1 + `stream("A", pattern="forward", along=ax.j0)` | `streams` records it; the plan's A-delivery is `FORWARD` where the default is `MULTICAST` (FR-S11, FR-M3) |
| `test_M2_model_is_json` | all four models | round-trips through canonical JSON unchanged; is hashable (§4 postcondition 4) |
| **property** `test_M2_model_idempotent` | 100 random clause sequences from a generator | `s.model == s.model`; no clause mutates the kernel; `PYTHONHASHSEED` does not change any field |

Negative corpus — `test_clause_errors[...]`, one case per row, 24 cases against FR-S19's floor
of 20. Each asserts the code, the clause text, the numbers in `details`, and a non-empty `fix`.

| # | Call | Code |
|---|---|---|
| 1 | `sp.schedule(gemm, target="xcvc1902")` | `CLAUSE-BAD-ENUM` |
| 2 | `grid(2,2,2)` | `CLAUSE-GRID-RANK` |
| 3 | `grid(0)` | `CLAUSE-BAD-VALUE` |
| 4 | `grid(2)` twice | `CLAUSE-DUPLICATE` |
| 5 | `tile(ax.i, 3)` on `M=64` | `CLAUSE-TILE-DIVIDES` |
| 6 | `tile(ax.i, 0)` | `CLAUSE-BAD-VALUE` |
| 7 | `tile(ax.q, 4)` | `CLAUSE-UNKNOWN-AXIS` |
| 8 | `tile` an axis with `extent is None` | `CLAUSE-TILE-DIVIDES` |
| 9 | `place(px=ax.i0)` with `grid(2,2)` | `CLAUSE-RANK` |
| 10 | `place(px=ax.i0, py=ax.i0)` | `CLAUSE-DUPLICATE` |
| 11 | `place(...)` with no `grid` | `CLAUSE-RANK` |
| 12 | `place` twice | `CLAUSE-DUPLICATE` |
| 13 | an `AxisRef` from another schedule | `CLAUSE-UNKNOWN-AXIS` |
| 14 | `reduce(ax.k, op="*")` | `CLAUSE-BAD-ENUM` |
| 15 | `reduce(ax.k)` twice | `CLAUSE-DUPLICATE` |
| 16 | `stationary("D")` | `CLAUSE-UNKNOWN-OPERAND` |
| 17 | `reside(D="L1")` | `CLAUSE-UNKNOWN-OPERAND` |
| 18 | `reside(A="L0")` | `CLAUSE-BAD-ENUM` |
| 19 | `reside(A="L1")` twice | `CLAUSE-DUPLICATE` |
| 20 | `stream("A", pattern="systolic", along=ax.j0)` | `CLAUSE-BAD-ENUM` |
| 21 | `stream("A", pattern="forward", along=ax.k0)` with `k0` unplaced | `CLAUSE-BAD-VALUE` |
| 22 | `stream("A", …)` then `forward("A", …)` | `CLAUSE-DUPLICATE` |
| 23 | `forward("S", along=ax.j0, dir="N->W")` | `CLAUSE-BAD-ENUM` |
| 24 | `window("U", dims=(ax.i, ax.j), halo=(1,))` | `CLAUSE-RANK` |
| 25 | `window("U", dims=(ax.i,), halo=-1)` | `CLAUSE-BAD-VALUE` |
| 26 | `exchange("U", along=ax.i0, halo=0)` | `CLAUSE-BAD-VALUE` |
| 27 | `skew(time=(ax.i, ax.i))` | `CLAUSE-DUPLICATE` |
| 28 | `double_buffer("D")` | `CLAUSE-UNKNOWN-OPERAND` |

---

## 8. Dependencies

**On other modules**: `model` (M0) for `ScheduleModel`, `StreamClause`, `WindowClause`,
`ExchangeClause`, `ClauseError`, `Diagnostic`; M1 for the `Kernel` it is constructed from
(`kernel.model.axes` seeds the axis table, `kernel.model.params` the operand table). M2 calls
M3/M4/M5/M6 only from the five lowering entry points of §2, by late import.

**On Python**: standard library only — `dataclasses`, `inspect` (caller frame for `location`),
`typing`. **No numpy, no `air`, no MLIR** at module scope (§3.8).

**Stubs M2 needs**: none. M2 is testable the hour M0 and M1 land. What M2 *provides* as a stub
is the `ScheduleModel` literal for W1 that B builds a hand-written `LegalMapping` against on D0
(HLD §8).

---

## 9. Implementation order, effort, definition of done

Order: (1) the axis table and `TILE` of §3.2, because every other clause resolves through it;
(2) the four shared validators `AXIS`/`OPERAND`/`ENUM`/`POS` plus the `ClauseError` renderer,
because they are 80 % of the negative corpus; (3) the fourteen clauses, which are then three
lines each; (4) `MODEL`'s canonicalisation; (5) the five lowering entry points as
thin dispatchers with the late import.

**Effort — estimate, not measured: 4 hours**, of which ~1.5 h is the 28-case negative corpus.
Lands on D1 alongside M1 (`05-work-breakdown.md` §2, D1).

**Definition of done.** The four models of §6 are produced field by field; all 28 negatives fire
with the right code and four message parts; `test_M2_order_independent`,
`test_schedule_is_pure_data`, `test_no_air_import_until_build` and `test_M2_model_is_json` pass;
every clause has a docstring naming its argument domain (NFR-5); and M3 consumes
`ScheduleModel` without needing a field that `06-interfaces.md` §3.2 does not list.

---

## 10. Open questions owned by M2

| # | Question | Resolution |
|---|---|---|
| **Q-M2-1** | `ScheduleModel`'s σ field is called `skew` (`06-interfaces.md` §3.2) but FR-S17's acceptance test asserts `ScheduleModel.sigma_terms == ("i","j0")`. Which name? | **Resolved from the frozen document**: the field is `skew`. `06-interfaces.md` is the contract (`00-README.md` §1 row 3). **Closed**: FR-S17's acceptance now asserts `ScheduleModel.skew` (REVIEW-round1 EDIT-37). No interface change. |
| **Q-M2-2** | FR-S14 says `pipeline(ax)` "shall place `ax` innermost in `σ`" *and* that two schedules differing only in `pipeline` emit byte-identical text. | **Resolved here** (§3.7): `pipeline` is inert — recorded, read by nothing. **Closed**: the "innermost in σ" phrase is deleted from FR-S14, which now says `pipeline` "shall be recorded and read by nothing" (REVIEW-round1 EDIT-40). No code or interface change. |
| **Q-M2-3** | Which of M0's `__post_init__` and M2/M3 enforces the *semantic* invariants listed in `06-interfaces.md` §3.2 (`len(place)==len(grid)`, `sequential` disjoint from `place`, tile factor divides)? | **Resolved here**: M0 enforces only **structural** invariants (types, arity, sortedness, hashability). The semantic cross-checks belong to M2 (clause time) and M3 (legality time) — because if M0 enforced them, `PLACE-SEQUENTIAL-CONFLICT`, `PLACE-EXTENT` and `CLAUSE-TILE-DIVIDES` would be unreachable and `test_D3_catalogue_complete` (FR-D3) would fail for lack of a test that raises them. **This is a note on how M0 is written, and M0 is co-owned — it must be said aloud at the D0 signature.** Due **D0**. |
| **Q-M2-4** | Should a clause that depends on another clause's presence (`stream.along` must be placed) fail at call time or at `model` time? | **Resolved here** (§3.3 last paragraph): both — validated against what is recorded so far, and re-validated in `MODEL`. Same code, same message, either way; the surface stays order-independent. No interface change. |
