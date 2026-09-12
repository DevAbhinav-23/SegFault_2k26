# LLD M3 — Legality checker

*Phase 1 low-level design, 2026-09-12. Owner: **Person A**. Reads on top of
[`01-requirements.md`](01-requirements.md) §3.2 (FR-L1…L14), [`02-hld.md`](02-hld.md) §2 (M3),
§4, §7, [`06-interfaces.md`](06-interfaces.md) §4, §6, §7 — **frozen** — and
[`../spatial-dsl/02-spacetime-ir-and-reduction-legality.md`](../spatial-dsl/02-spacetime-ir-and-reduction-legality.md),
whose §2–§6 this module implements. Cited as `SD-02 §n`. `VF §X` =
`../hackathon/VERIFIED-AIR-FACTS.md`; a bare `path:line` is mlir-air at `ff95a9b`.*

**This module is the entry's differentiator (G2).** Every message it prints is read by a judge.

---

## 1. Purpose and the FR IDs it satisfies

M3 takes `(KernelModel, ScheduleModel)` and returns a `LegalMapping`
(`06-interfaces.md` §4.1) or raises `LegalityError` carrying a `Diagnostic`. It builds `Sσ` and
`Sπ`, runs fourteen checks in a fixed order, and records the maps, the kernels, the reduction
split, the derived stationarity report, the resolved physical herd and the L1 estimate. It does
**not** choose `(σ, π)` — the user supplies both (PC §1.3) — and it never sees a channel, a
buffer plan or an AIR op.

| FR | Check |
|---|---|
| **FR-L1** | (L1) `ker Sσ ∩ ker Sπ = {0}` — §3.4 |
| **FR-L2** | (L2) `Sσ·d ⪰ 1` for every dependence — §3.5 |
| **FR-L3** | stationarity `ker M_a ⊆ ker Sπ`, declared and derived — §3.6 |
| **FR-L4** | reduction split `R = ker Sf`, `R_time`, `R_space` — §3.7 |
| **FR-L5** | A/C operator required when `R_space ≠ {}` — §3.7 |
| **FR-L6** | cascade rider (rank 1, contiguous line, herd shape) — §3.8 |
| **FR-L7** | halo ≥ dependence footprint — §3.9 |
| **FR-L8** | grid / place / tile consistency — §3.3 |
| **FR-L9** | L1 capacity with the doubled ping-pong figure — §3.10 |
| **FR-L10** | herd shape vs physical array — §3.11 |
| **FR-L11** | grid rank ≤ 2 — §3.3 |
| **FR-L12** | every rejection carries code, clause, reason-with-numbers, fix — §5 |
| **FR-L13** | check before emit — M2 §3.8 dispatches; M3 imports no `air` |
| **FR-L14** | swap parity, **as adjudicated 2026-09-12** — §3.12 |
| **FR-S9 (half)** | `REDUCE-NOT-ACCUMULATED` — §3.7 |
| **FR-S13 (half)** | `PINGPONG-SHAPE`, the `isPingPongCandidate` assertion — §3.13 |
| **FR-D1, FR-D3** | the legality half of the diagnostic schema and the code catalogue |
| **NFR-1, NFR-7** | deterministic output; no unhandled exception on any grammar-accepted input |

---

## 2. Public entry points

| Name | Contract | Signature |
|---|---|---|
| `m3.check(kernel, schedule)` | run §3.3–§3.13 in order; return a `LegalMapping` or raise the first `LegalityError` | `06-interfaces.md` §7.2 |
| `Schedule.check()` | the surface wrapper; identical contract | §7.1 |
| `m3.pingpong_mode(mapping, operand)` | a **pure helper, exported for M4**: `"PASS"`, `"PAIR"` or raises; the single definition of what `double_buffer` promises for an operand (§3.13) | new, internal-contractual |
| `m3.intlin` | the integer-linear-algebra helper (§3.2), also imported by M1 | new, internal-contractual |

`pingpong_mode` and `intlin` are *module entry points* in the sense of `06-interfaces.md` §7.2 —
internal, but contractual between owners. Neither adds a dataclass or a field, so neither needs
a `CONTRACT_VERSION` bump; both are named here so M4 does not re-derive them (§10 Q-M3-3).

---

## 3. Internal design

### 3.1 Internal data structures

Three, none of them shared:

* **`Coord`** — the post-tiling coordinate frame: the ordered axis-name tuple of
  [`03-lld-M2-schedule.md`](03-lld-M2-schedule.md) §3.6, plus each axis's extent, parent and tile
  factor. Every matrix built over post-tiling axes has `len(Coord)` columns.
* **`UCoord`** — the *untiled* frame: `kernel.axes` in source order. `M_a`, `Sf`, `R`, `R_time`,
  `R_space` and every `Dependence.vector` live here, because a tile split is an artifact of the
  schedule and the access maps are properties of the kernel.
* **`Box`** — a per-PE iteration sub-box: for each axis, a closed integer interval. Used by the
  footprint routine (§3.9) for both the halo check and the capacity estimate.

**The two-frame rule, stated once because it is the one thing that can silently go wrong.**

| Object | Frame | Columns |
|---|---|---|
| `LegalMapping.sigma`, `.pi`, `.ker_pi` | `Coord` | `len(mapping.axes)` |
| `LegalMapping.r_time`, `.r_space` | `UCoord` | `len(kernel.axes)` |
| `Dependence.vector` (from M1) | `UCoord`, **lifted** to `Coord` for (L2) only (§3.5) | — |
| `AccessMap.matrix` (from M1) | `UCoord` | `len(kernel.axes)` |

`06-interfaces.md` §4.1 fixes the *fields* but not the basis; this table is M3's resolution of
that gap and is what M4 must read (§10 Q-M3-1). `Sπ_u`, the untiled shadow of `Sπ`, is built in
§3.3 and is what stationarity and the reduction split use.

### 3.2 `intlin` — the integer linear algebra recipe

**Operations needed**, and nothing else: `rank(M)`, `kernel_basis(M)` (a basis of the integer
null space, canonical), `solve(M, b)` (integer particular solution or `None`), `intersect(A, B)`
(basis of the intersection of two subspaces), `complement(R, R_time)` (a basis of a complement),
`contains(A, B)` (is `span(B) ⊆ span(A)`), and matrix–vector / matrix–matrix products.

**Arithmetic choice: fraction-free (Bareiss) integer Gaussian elimination over Python `int`, not
`fractions.Fraction` and not `numpy.linalg`.** Three reasons, in order of weight:

1. `numpy.linalg.matrix_rank` and any SVD-based null space are **floating point with a
   tolerance**. A rank that depends on a tolerance is a rank that can differ between machines —
   which breaks NFR-1 outright, and breaks it in the worst possible way, silently, on a
   near-singular `Sσ`.
2. Bareiss keeps every intermediate an **exact integer** (each is a minor of the input), so
   there is no gcd blow-up and no rational normalisation pass. `Fraction` would also be exact,
   but its denominators carry no meaning for a lattice basis and every result would need
   re-clearing to primitive integers anyway.
3. **Python `int` is unbounded, and `int64` is not big enough.** Entries are bounded by the tile
   factors (≤ 256 in the shipped fixtures) and `n ≤ 8`, so a determinant can reach `256⁸ ≈
   1.8 × 10¹⁹` — past `int64`'s `9.2 × 10¹⁸`. Overflowing would be silent.

**numpy's role**, since NFR-2 pins it anyway: storage as `dtype=int64` arrays and the `@`
operator for matrix–vector and matrix–matrix products, where the magnitudes are tiny
(`|Sσ·d| ≤ n · max|coeff| · max|d|`, a few hundred). **No `numpy.linalg` call appears anywhere
in M3.** A lint test asserts that.

**Canonical form**, which is what makes every basis deterministic (NFR-1, and the reason
`06-interfaces.md` §2.6 and §4.1 say "canonical" four times):

```pseudo
KERNEL_BASIS(M):            # M is r x n over Z
 1  H := BAREISS_ROW_ECHELON(M)          # fraction-free; pivots left to right
 2  P := the pivot column indices of H ; F := the free column indices, ascending
 3  basis := []
 4  for each free column f in F:                       # one generator per free column
 5      v := zero vector of length n ; v[f] := 1
 6      back-substitute over the pivot rows of H to fill v[P], clearing denominators
 7      v := v / gcd(|v|)                              # primitive
 8      if the first nonzero entry of v is negative: v := -v     # sign-normalised
 9      basis.append(v)
10  return tuple(basis)                                # ordered by free column index, ascending
```

The output is unique given `M` and the column order. `rank(M) = len(P)`;
`intersect(A, B) = KERNEL_BASIS(stack(dual(A), dual(B)))` computed as the kernel of the stacked
constraint matrices; `contains(A, B)` is `rank(A) == rank(stack(A, B))`;
`complement(R, R_time)` greedily appends rows of `KERNEL_BASIS`-ordered `R` to `R_time` while the
rank strictly increases, and returns the appended rows — deterministic because both inputs are
canonical and the walk is in index order.

**Complexity.** `n ≤ 8`, matrices ≤ `8 × 8`, Bareiss is `O(n³)` with big-int entries bounded by
the minors: a few hundred arithmetic operations per call and at most a few dozen calls per
`check()`. The whole check is under a millisecond; none of it is worth optimising, and none of
it allocates beyond the inputs.

### 3.3 Building the frames, `Sπ`, `Sσ`, and the consistency checks (FR-L8, FR-L11)

```pseudo
BUILD(kernel, schedule):
 1  Coord  := post-tiling axis order (M2 LLD §3.6): source order, each tiled axis replaced
 2            in place by (outer, inner); extents from the tile factors
 3  UCoord := kernel.axes, source order
 4  # --- FR-L11 / HERD-RANK -------------------------------------------------
 5  if schedule.grid is None and schedule.place != ():
 6      raise LegalityError(PLACE-EXTENT), reason "place without grid"
 7      # M2 already rejects this at CLAUSE-RANK (M2 §3.4); this is the defence in depth
 8  if len(schedule.grid) > 2:                          raise HERD-RANK
 9  # --- FR-L8 (a)(b): rank and existence ------------------------------------
10  assert len(schedule.place) == len(schedule.grid)    # M2 §4 postcondition 6 guarantees it
11  assert every name in schedule.place is in Coord     # M2 §4 postcondition 1 guarantees it
12  # --- FR-L8 (c): extents --------------------------------------------------
13  for r, name in enumerate(schedule.place):
14      if extent(name) != schedule.grid[r]:            raise PLACE-EXTENT
15  # --- FR-L8 (d): placed and sequential ------------------------------------
16  for name in schedule.sequential:
17      if name in schedule.place or parent-of(name) in schedule.place:  raise PLACE-SEQUENTIAL-CONFLICT
18  # --- the matrices --------------------------------------------------------
19  pi   := rows [ e_name for name in schedule.place ]                    # over Coord
20  pi_u := rows [ e_{root(name)} for name in schedule.place ]            # over UCoord
21  if schedule.skew is not None:
22      row0  := sum of e_name for name in schedule.skew                  # FR-S17: a SUM
23      rest  := [ e_a for a in default-sigma-order(Coord)
24                 if a not in schedule.skew and a not in schedule.place ]
25      sigma := [row0] + rest
26  else:
27      sigma := [ e_a for a in default-sigma-order(Coord) ]              # every axis, M2 §3.6
28  ker_pi := KERNEL_BASIS(pi) ; ker_pi_u := KERNEL_BASIS(pi_u)
29  return Coord, UCoord, sigma, pi, pi_u, ker_pi, ker_pi_u
```

Line 20 is the definition of `Sπ_u`: placing a tile handle `i0` means the PE coordinate is a
coarsening of `i`, so in the untiled frame the row is `e_i`. That is what makes
`ker Sπ_u = span{e_k}` for W1 and reproduces SD-02 §3.1's table exactly.

**M3 raises only `legality`-stage codes.** Lines 5-7 and 11 previously raised `CLAUSE-RANK` and
`CLAUSE-UNKNOWN-AXIS` — both `stage = clause` in the catalogue — from a `legality`-stage check,
which `03-lld-M7-tests.md` §3.2's `assert_diagnostic` (`d.stage == CATALOGUE[d.code].stage`)
rejects, and which contradicts `02-hld.md` §4.1's assignment of `ClauseError` to **M2 only**.
Both checks now live in M2 (`03-lld-M2-schedule.md` §3.4 rows `place`, and §3.3 line 1), and what
remains here are an in-module `assert` and one `legality`-stage `PLACE-EXTENT`
(REVIEW-round1 B-12 / RULING 5). No `ClauseError` is constructed anywhere in M3.

**Line 24's exclusion of *placed* axes from the appended σ rows is what makes the demo's headline
rejection a rejection.** With `skew(time=(ax.i,))` on W3 — the *dropped* `j0` term — `j0` is
placed and therefore excluded, so `Sσ = [e_i; e_j1]` and the tile-crossing representative of
`d = (0, 1)` gives `Sσ·d = (0, −7) ⪯ 0`: **rejected** with `L2-CAUSALITY`. If `j0` were appended
instead, `Sσ·d = (0, 1, −7)` would be lexicographically positive and the schedule would be
**accepted** — and the demo would have no headline
(`03-lld-M8-kernels-demo.md` §3.5, REVIEW-round1 RULING 6). The rule is now written into FR-S17
and FR-L2 as well, so it is a requirement and not an implementation accident. It also gives W3
the second row `j1` it needs for (L1), without handing every skewed schedule a free full-rank σ
that would make (L1) vacuous.

**Property, worth stating because it tells you where the checks have teeth.** When `skew` is
absent, `Sσ` is a permutation matrix (line 27), so `ker Sσ = {0}` and (L1) holds
unconditionally; and `Sσ` is then a reordering of the tiled source nest, so (L2) fails only if
the *tiling* is illegal. **(L1) and (L2) are therefore checks on `skew` and on `tile`, not on
`place`** — which is why §7's negatives for both use `skew`, and why FR-L1's acceptance prose
("placing `i` and leaving `i` out of `σ`") needs the restatement given in §7.

### 3.4 (L1) — no structural conflict (FR-L1)

SD-02 §2 L1: `(σ,π)` injective on `D`, equivalently `ker Sσ ∩ ker Sπ = {0}`.

```pseudo
CHECK_L1(sigma, pi, Coord):
 1  K := intersect(KERNEL_BASIS(sigma), KERNEL_BASIS(pi))
 2  if K is empty: return                                      # rank 0 intersection
 3  v := K[0]                                                  # canonical, so deterministic
 4  raise L1-CONFLICT with
 5      reason  "iterations that differ by <render(v)> get the same time and the same PE"
 6      clause  the skew (or place) clause that produced the conflict
 7      details {conflict_basis: K, ker_sigma: ..., ker_pi: ..., axes: Coord}
 8      fix     "give <the axes with nonzero entries in v> a time of their own:
 9               add them to skew(time=...), or place one of them"
```

`06-interfaces.md` §4.1's invariant on `sigma` ("`rank(Sσ) + rank(Sπ) ≥ n` where the two kernels
intersect trivially") is exactly this condition, and is asserted as a postcondition.

### 3.5 (L2) — causality (FR-L2)

SD-02 §2 L2: for a uniform dependence `d`, `Sσ · d ⪰ 1`. With a multi-row `σ` that means
**lexicographically positive**: the first nonzero component is `≥ 1`. With a single-row `σ` it
reduces to the scalar inequality the brief writes.

`Dependence.vector` is in `UCoord`; `Sσ` is over `Coord`. Tiling is not linear (it is a floor
division), so a dependence does not lift to a single tiled vector — it lifts to **two
representatives per tiled axis**, and both must be causal:

```pseudo
LIFT(d, Coord):                       # d over UCoord -> a set of vectors over Coord
 1  reps := { the zero vector over Coord }
 2  for each untiled axis a with component delta:
 3      set component a := delta in every rep
 4  for each tiled axis a (factor F, handles a0/a1) with component delta:
 5      interior := (a0 = 0,          a1 = delta)                    # stays inside one tile
 6      boundary := (a0 = sign(delta), a1 = delta - sign(delta)*F)   # crosses a tile edge
 7      reps := { r + interior for r in reps } U { r + boundary for r in reps }
 8  return reps                                                       # <= 2^(#tiled axes)

CHECK_L2(sigma, deps, Coord):
 9  for each Dependence dep, sorted by (operand, vector):
10      for each r in LIFT(dep.vector, Coord), in canonical order:
11          t := sigma @ r
12          if not LEXPOS(t):                       # first nonzero entry >= 1
13              raise L2-CAUSALITY with
14                  reason  "dependence <dep.vector> on <dep.operand> is not causal:
15                           Sσ·d = <t>, whose first nonzero entry is <t[p]> <= 0"
16                  clause  the skew clause (or "the default loop order" when skew is absent)
17                  details {dependence: dep.vector, representative: r, sigma_d: t,
18                           sigma_rows: the row labels, operand: dep.operand}
19                  fix     "add <the axis whose row is p> to skew(time=...) before
20                           <the axis at the violating row>, or drop the tile on <a>"
21      # all representatives causal -> this dependence is discharged
```

Line 6 is why W1's `d = (0,0,1)` needs checking twice: the interior representative
`(0,0,0,0,0,1)` gives `Sσ·d = (0,0,0,0,0,1)`, and the boundary representative
`(0,0,0,0,1,−15)` gives `(0,0,1,0,0,−15)` — both lexicographically positive, both accepted. A
tiling that reorders a tile pair badly is caught by the boundary representative and by nothing
else; §7 has a synthetic case that exercises exactly that.

Bound: `≤ 8` dependences × `≤ 2³` representatives = 64 matrix–vector products of size `≤ 8`.

### 3.6 Stationarity, declared and derived (FR-L3)

SD-02 §3: **operand `a` is stationary ⟺ `L_a = ker M_a ⊆ ker Sπ`.** Computed in `UCoord`, with
`Sπ_u` from §3.3 line 20.

```pseudo
CHECK_STATIONARITY(kernel, schedule, pi_u):
 1  ker_pi_u := KERNEL_BASIS(pi_u)
 2  report := {}
 3  for each operand a in kernel.params, sorted by name:
 4      M_a := the access matrix of a       # every access shares it; M1 §3.1 R-c guarantees it
 5      L_a := KERNEL_BASIS(M_a)
 6      report[a] := contains(ker_pi_u, L_a)
 7  for each a in schedule.stationary:                      # the DECLARED ones
 8      if not report[a]:
 9          raise STATIONARITY with
10              reason  "ker M_<a> = <render(L_a)> is not contained in ker Sπ = <render(ker_pi_u)>"
11              clause  stationary("<a>")
12              details {operand: a, ker_M: L_a, ker_pi: ker_pi_u, placed: schedule.place,
13                       placed_roots: the untiled axes of pi_u}
14              fix     "place <the axes that would kill L_a>: place(px=ax.<x>, py=ax.<y>)"
15  return tuple(sorted(a for a in report if report[a]))     # LegalMapping.stationary_ops
```

Line 15 is the **derived stationarity report** `06-interfaces.md` §4.1 asks for: operands proved
stationary *declared or derived*. Two properties of it are worth writing down so they do not read
as bugs:

* An operand with `ker M_a = {0}` (a full-rank access, e.g. W2's `U`, W3's `S`) is vacuously
  stationary: it has no reuse direction at all, so none of its reuse crosses a PE. That is why
  HLD §7.2 says "U: stationary strip (derived)".
* "Stationary" means **every reuse direction stays on one PE**, not "the tile never changes".
  In the W1 flip, `A` is stationary because `ker M_A = span{e_j} ⊆ ker Sπ_u = span{e_i, e_j}`,
  even though the `A` tile is refetched as `i` sweeps — `A` is simply not reused along `i`.
  The predicate is *spatial* and says nothing about time, which is why the summary reports a
  separate **residency duration** per operand (RULING 9; algorithm in
  `03-lld-M4-mapping.md` §3.9, field in `06-interfaces.md` §5.7). M3 computes no residency
  field: the two facts are orthogonal and the checker's job is only the spatial one.

### 3.7 Reduction split and the A/C requirement (FR-L4, FR-L5, FR-S9)

SD-02 §5: `R = ker Sf`, `R_time = R ∩ ker Sπ`, `R_space = R / R_time`.

```pseudo
CHECK_REDUCTION(kernel, schedule, pi_u):
 1  if kernel.reduction is None:
 2      for (axis, op) in schedule.reductions:              # FR-S9's second half
 3          raise REDUCE-NOT-ACCUMULATED (the kernel has no accumulation at all)
 4      return r_time = (), r_space = ()
 5  Sf := kernel.reduction.projection      # = the linear part of the accumulate target's access
 6  R  := kernel.reduction.space           # = KERNEL_BASIS(Sf), computed by M1 §3.8
 7  for (axis, op) in schedule.reductions:
 8      if e_axis is not in span(R):
 9          raise REDUCE-NOT-ACCUMULATED with
10              reason  "axis <axis> is not a reduction axis: e_<axis> is not in
11                       R = ker Sf = <render(R)>"
12              details {axis, R, target: kernel.reduction.target}
13              fix     "reduce(ax.<the axis whose e is in R>, op=...)"
14  r_time  := intersect(R, KERNEL_BASIS(pi_u))
15  r_space := complement(R, r_time)                        # §3.2; canonical
16  if r_space is not empty:                                # FR-L5
17      op := the operator tagged by schedule.reductions for the axes spanning r_space,
18            else kernel.reduction.op
19      if op is None or op not in {"+", "max", "min"}:
20          raise RSPACE-NO-AC-OP with
21              reason  "R_space = <render(r_space)> is non-empty, so partial results are
22                       combined across PEs, and that is legal only for an associative and
23                       commutative operator; none is declared"
24              details {r_space, r_time, target, placed: schedule.place}
25              fix     'reduce(ax.k, op="+")'              # FR-L5's acceptance names this string
26  return r_time, r_space
```

Line 5 is SD-02 §4's whole point in one assignment: representing the reduction explicitly makes
`R` readable off the projection instead of recovered by dependence analysis. Line 19's A/C gate
is SD-02 §4's "without that tag the accumulation chain is a *rigid* sequential dependence" — the
tag is what buys reassociation, hence spatial reduction (R2).

### 3.8 Cascade rider (FR-L6)

Applies whenever `r_space ≠ {}` — because on AIE the only mechanism for a spatial combine is the
cascade (SD-02 §6), so a non-empty `R_space` *is* a cascade declaration whether or not the user
wrote `stream(pattern="cascade")`.

```pseudo
CHECK_CASCADE(r_space, pi, pi_u, schedule, mapping):
 1  if r_space is empty: return
 2  # (a) rank 1 ------------------------------------------------------------
 3  if len(r_space) != 1:
 4      raise CASCADE-RANK, reason "R_space has rank <n> but a cascade is a linear chain"
 5      fix "keep one reduction axis spatial and leave the rest temporal (untile or unplace one)"
 6  g := r_space[0]                               # the single generator, in UCoord
 7  # (b) one carrying PE axis, a contiguous line --------------------------
 8  carried := [ p for p in rows(pi_u) if (pi_u @ g)[p] != 0 ]
 9  if len(carried) != 1:
10      raise CASCADE-RANK, reason "the reduction direction crosses <len> PE axes"
11  p := carried[0]                               # a whole grid axis => contiguous by construction
12  # (c) herd 1-D, or extent 1 on the other axis ---------------------------
13  if len(schedule.grid) == 2 and schedule.grid[1 - p] != 1:
14      raise CASCADE-RANK with
15          reason  "a cascade needs a 1-D herd or extent 1 on the other axis;
16                   grid is <schedule.grid> with the chain along p<x|y>"
17          details {grid, carrier_axis: schedule.place[p], r_space: g}
18          fix     "grid(<grid[p]>) with place(px=ax.<carrier>)"
19  # (d) no broadcast on the cascade operand -------------------------------
20  tgt := kernel.reduction.target
21  if schedule.streams has an entry for tgt with pattern == "broadcast":
22      raise CASCADE-BROADCAST with
23          reason  'air.channel does not take broadcast_shape= with channel_type="npu_cascade"'
24          details {operand: tgt, pattern: "broadcast"}   # _channel.py:170-177
25          fix     'drop stream("<tgt>", pattern="broadcast", ...) — the cascade is the chain'
26  # (e) the chain must not be folded by strip-mining ----------------------
27  if mapping.repeats[p] != 1:
28      raise HERD-PHYSICAL with
29          reason  "the cascade chain of length <grid[p]> is folded onto <physical[p]> cores
30                   with repeat <repeats[p]>, which breaks the chain"
31          details {grid, physical_herd, repeats, cap: PHYSICAL_HERD[target]}
32          fix     "use grid(<physical[p]>) on this target, or target npu2"
33  record the carrying line; no at= pinning is needed (Q-1)         # FR-L6's last sentence
```

Line 11 is why contiguity needs no separate test: a placed axis is a whole grid axis, and the
herd's coordinates along one axis are contiguous by construction (`_trace.py:1286-1300`).
Line 23 quotes `air.api`'s own refusal verbatim (`_channel.py:170-177`) so the rejection happens
**before** `air.api` would raise it, which is FR-L6's acceptance criterion.
Lines 27–32 give `HERD-PHYSICAL` its only reachable condition — see §10 Q-M3-2.

### 3.9 Footprint: halo (FR-L7) and tile shape, one routine

The same computation answers "is the declared halo big enough" and "how many bytes does a PE
hold". It is exact, needs no uniformity assumption, and is three lines of integer arithmetic,
because the maximum of an affine form over a box is attained at a vertex and is computed
coefficient by coefficient.

```pseudo
PINNED_BOX(schedule, Coord, pe_coord):        # the iterations one PE runs in one temporal trip
 1  box := {}
 2  for each axis a in Coord:
 3      if a is placed:                     box[a] := [pe_coord[row(a)], pe_coord[row(a)]]
 4      elif a is an OUTER tile handle:     box[a] := [s, s]          # one trip of the stream
 5      elif a in schedule.sequential:      box[a] := [s, s]          # a temporal loop
 6      elif a in schedule.skew and a not placed: box[a] := [s, s]    # a temporal loop
 7      else:                               box[a] := [lo(a), hi(a)]  # full extent
 8  return box            # `s` is a symbolic single value; only the WIDTH of the box matters

IMAGE(access, box, dim):                      # min/max of one array index over the box
 9  lo := offset_const(access, dim) ; hi := lo
10  for each axis a in Coord:
11      c := coefficient of a in access, expressed over Coord      # tile split substituted
12      if c >= 0: lo += c*box[a].lo ; hi += c*box[a].hi
13      else:      lo += c*box[a].hi ; hi += c*box[a].lo
14  return (lo, hi)

FOOTPRINT(operand, box):                      # per array dim: (accessed range, owned range)
15  for each array dim d of operand:
16      acc_lo, acc_hi := min/max of IMAGE(r, box, d) over ALL accesses r to the operand
17      own_lo, own_hi := IMAGE(the WRITE access, box, d)      # or the first access if read-only
18      span[d]     := acc_hi - acc_lo + 1
19      overhang[d] := ( max(own_lo - acc_lo, 0), max(acc_hi - own_hi, 0) )
20  return span, overhang

CHECK_HALO(schedule, ...):                                          # FR-L7
21  for each WindowClause (operand, dims, halo) in schedule.windows:
22      span, overhang := FOOTPRINT(operand, PINNED_BOX(...))
23      for each declared dim position q, mapping to array dim d:
24          need := max(overhang[d])          # the derived footprint
25          if halo[q] < need:
26              raise HALO-TOO-SMALL with
27                  reason  "the derived footprint of <operand> along <dim> is <need>
28                           but the declared halo is <halo[q]>"
29                  clause  window("<operand>", dims=(...), halo=<halo>)
30                  details {operand, dim, derived: need, declared: halo[q],
31                           accessed: (acc_lo, acc_hi), owned: (own_lo, own_hi)}
32                  fix     "window(\"<operand>\", dims=..., halo=<need>)"
33  return tuple of (operand, per-dim derived footprint)            # LegalMapping.halo_footprint
```

Lines 4–6 are the pinning rule, and they are what makes the capacity estimate match the figures
in HLD §7 without any ad-hoc doubling: an axis that indexes a temporal *loop* contributes only
the span its accesses actually touch. W2's `t` is pinned, and `U[t+1,…]` / `U[t,…]` then give a
dim-0 span of **2** — which *is* the `u`/`v` pair. W3's `i` is pinned (it is in the skew and is
not placed), and `S[i,…]` / `S[i−1,…]` give a span of **2** — which *is* the `prev`/`cur` pair.
W1's `k0` is pinned as an outer tile handle, so `A`'s `k` span is `TK = 16`, not `K = 64`.

### 3.10 L1 capacity (FR-L9)

```pseudo
CHECK_L1(kernel, schedule, ...):
 1  total := 0 ; breakdown := []
 2  for each operand a with residency[a] == "L1", sorted by name:
 3      span, _ := FOOTPRINT(a, PINNED_BOX(...))
 4      bytes := product(span) * dtype(a).sizeof
 5      if a in schedule.double_buffer and pingpong_mode(a) == "PASS":
 6          bytes := bytes * 2                        # the ping-ponged figure, FR-L9
 7      total += bytes ; breakdown.append((a, span, dtype, bytes))
 8  if total > 65536:                                 # L1_BYTES, _trace.py:100
 9      raise L1-CAPACITY with
10          reason  "the per-core L1 working set is <total> bytes, over the 65536-byte budget"
11          clause  the tile / double_buffer clauses
12          details {total, budget: 65536, per_buffer: breakdown, doubled: [...]}
13          fix     "halve a tile factor (tile(ax.i, <TM/2>)) or drop double_buffer(\"<x>\")"
14  return total
```

Line 6 doubles only in **`PASS` mode**: in `PAIR` mode (§3.13) the pair is already in the span
(§3.9), and doubling again would charge four buffers for two. `L1_BYTES = 65536` is `air.api`'s
own trace-time budget (`_trace.py:100`); the figure that has to fit is the ping-ponged one, which
is what `_compile.py`'s `_annotate_l1_failure` docstring warns about and what
`exceedsL1Budget` (VF §E.2 item 6) itself applies.

**Scope, stated because it is an approximation.** M3's `l1_bytes` covers **operand tiles** — the
term the user controls with `tile`, `reside` and `double_buffer`. Staging scalars that only the
protocol needs (W3's `edge_in`/`edge_out`, HLD §7.3) are added by M4 and charged by
`06-interfaces.md` §5.6 invariant 5 against the same budget. M3's figure is therefore a **lower
bound** on M4's; for W3 they are 72 and 80 bytes. A design within a few bytes of 65536 would
pass M3 and fail M4 — acceptable, because M4's is the binding check and it runs before emission.

### 3.11 Physical herd resolution (FR-L10)

Mirrors `air.api`'s `_resolve_physical` exactly (`_trace.py:1347-1373`), with its table
(`_trace.py:88-91`): `npu1 = {1-D: (4,), 2-D: (1, 4)}`, `npu2 = {1-D: (8,), 2-D: (2, 4)}`.

```pseudo
RESOLVE_PHYSICAL(grid, target):
 1  caps := PHYSICAL_HERD[target][len(grid)]     # KeyError impossible: rank checked in §3.3
 2  physical := tuple( largest divisor of g that is <= cap  for g, cap in zip(grid, caps) )
 3  repeats  := tuple( g // p for g, p in zip(grid, physical) )
 4  return physical, repeats
```

Every extent has the divisor 1, so this never fails on its own — which is why `HERD-PHYSICAL`
fires only from §3.8 line 27, the cascade rider. `target == "auto"` is resolved to `npu2` here
with a recorded note, **never** by shelling out to `xrt-smi` (`_trace.py:182`; decision D-13 —
a check must not spawn a subprocess).

Table, asserted by `test_physical_herd_table`:

| grid | target | physical | repeats |
|---|---|---|---|
| `(2,2)` | npu1 | `(1,2)` | `(2,1)` |
| `(2,2)` | npu2 | `(2,2)` | `(1,1)` |
| `(2,)` | npu1 | `(2,)` | `(1,)` |
| `(2,)` | npu2 | `(2,)` | `(1,)` |
| `(4,)` | npu1 | `(4,)` | `(1,)` |
| `(4,)` | npu2 | `(4,)` | `(1,)` |
| `(8,)` | npu1 | `(4,)` | `(2,)` |
| `(3,5)` | npu1 | `(1,1)` | `(3,5)` |

The `(2,)` rows are W2's fixture at `PI = 2`; the last row is FR-L10's own acceptance case.

### 3.12 Swap parity (FR-L14, **as adjudicated 2026-09-12**)

Decision D-4 is **overridden**: an odd trip count is legal, and the mapper realises it by
peeling the final timestep after the unrolled-by-two loop. M3's check is therefore narrow.

```pseudo
CHECK_SWAP_PARITY(kernel, schedule, footprints):
 1  for each operand a whose FOOTPRINT span along a PINNED temporal axis is >= 2:
 2      # a >= 2 span along a pinned axis IS a swap pair (§3.9): W2's u/v, W3's prev/cur
 3      L := that temporal axis ; T := extent(L)
 4      if T is None or T < 1:
 5          raise SWAP-PARITY with
 6              reason  "the buffer swap on <a> is realised by unrolling <L> by two, which
 7                       needs a compile-time trip count of at least 1; <L> has <T>"
 8              clause  the clause that sets L (sequential(ax.t) / skew(time=...)) 
 9              details {operand: a, loop: L, trip_count: T, pairs: T // 2, peeled: T % 2}
10              fix     "give <L> a constant, positive extent (the fixture parameter <P>)"
11      # T >= 1 is legal. T odd is legal: floor(T/2) unrolled pairs plus one peeled body.
12      # M4 derives the peel from the trip count; no field is added to LegalMapping.
```

The rationale for the shape is unchanged and still binding: `air.sequential` emits `scf.for`
with `yield_([])` and there are **no `iter_args` anywhere in `air.api`** (`_loop.py:180`,
VF §D.5), so a swap cannot be loop-carried; and a plain Python loop unrolls and strands the
acquire/release pairs, computing "with stale operands" (`_loop.py:14-19`). What the adjudication
changes is only the residue: the odd timestep is **peeled**, not rejected.

### 3.13 `double_buffer` and the `isPingPongCandidate` assertion (FR-S13)

VF §E.2 lists eight conditions for `air-label-scf-for-to-ping-pong` to fire. M3 checks the ones
that are decided by the *schedule*; the plan-level ones are `06-interfaces.md` §5.6 invariant 4,
M4's; two are vacuous by construction and say why.

| VF §E.2 condition | Who checks it | How |
|---|---|---|
| 1. an `scf.for`, no existing `unroll` attribute | **vacuous** | we emit fresh IR and never set `unroll` |
| 2. an `air.execute`-wrapped `memref.alloc` as a **direct child** of the loop | **M3** (necessary half) + **M4** (invariant 4) | M3: the operand must have a *pinned temporal **tile** axis along which its access varies*, i.e. `∃ r ∈ ker Sπ_u` pinned with `M_a · r ≠ 0` **and carried by an outer tile axis**. Otherwise the tile is loop-invariant, is hoisted out of every loop, and no alloc is a direct child of anything. Equivalently: an operand whose residency is *the whole run* (`03-lld-M4-mapping.md` §3.9) fails this condition — which is why the W1-flip's resident `B` cannot be double-buffered while its streamed `A` can (RULING 9). An **untiled** axis never qualifies: its whole extent lives inside the tile, so the tile does not move along it |
| 3. the alloc is **dead on entry**: the first access is a definite write (a `channel.get` counts; an opaque callee does not) | **M3** | the operand must be **read-only** in the kernel (`Param.is_written == False`). A written operand's tile is live-in across the trip |
| 4. no opaque callee touching a herd block argument | **vacuous** | FR-E2 forbids `air.extern`, `func.call` and `link_with` outright |
| 5. at most one `air.channel.get` per alloc per iteration, static trip counts on intervening loops | **M3** (first half) + **M4** (second) | M3: the operand has exactly one access group inside one trip of the pinned axis; every pinned axis has a constant extent |
| 6. L1 budget: herd body + duplicated allocs ≤ target local memory | **M3** | §3.10, the same 65536 |
| 7. no `air.disable_ping_pong` | **vacuous** | never emitted |
| 8. `omit-memory-space` does not exclude the alloc's space | **vacuous** | the default pipeline, L1 allocs |

```pseudo
PINGPONG_MODE(mapping, a):                    # exported; the single definition (§2)
 1  if a is subject to an ExchangeClause or a WindowClause:
 2      return "PAIR"       # the strip lives across the temporal loop; the swap IS the pair.
 3                          # Decision D-5: the emitter writes an explicit u/v pair and the
 4                          # checker's message says so rather than promising the pass fires.
 5  if a is written by the kernel:                      # condition 3
 6      raise PINGPONG-SHAPE (condition 3: the tile is live on entry)
 7  r := a pinned temporal axis with M_a · r != 0       # condition 2
 8  if none exists:
 9      raise PINGPONG-SHAPE (condition 2: no streamed loop to allocate inside)
10  if a has more than one access group inside one trip of r:   # condition 5
11      raise PINGPONG-SHAPE (condition 5)
12  return "PASS"

CHECK_PINGPONG(schedule, ...):
13  for each a in schedule.double_buffer, sorted:
14      mode := PINGPONG_MODE(mapping, a)
15      record mode        # M4 re-reads it via m3.pingpong_mode; no LegalMapping field is added
```

The `PINGPONG-SHAPE` message names **which condition** failed, verbatim from VF §E.2, with the
`AIRDependencyScheduleOpt.cpp` line:

```
PINGPONG-SHAPE: double_buffer("C") cannot be realised by the ping-pong pass
  in clause: double_buffer("A", "B", "C")
  because:   condition 3 of isPingPongCandidate (AIRDependencyScheduleOpt.cpp:1620-1624)
             requires the buffer to be dead on entry to the loop body — its first access
             must be a definite write. C is accumulated into, so its first access is a read,
             and splitting it across the two parities would give each parity its own copy.
             Condition 2 also fails: ker M_C contains the streamed axis k, so C's tile does
             not change across the K loop and its alloc is hoisted out of it.
  fix:       drop "C" from double_buffer; C is the output accumulator and is stationary —
             it is held once per PE, not streamed.
```

In `PAIR` mode the message is not an error but the sentence D-5 requires, carried into the
mapping summary (FR-M11):

```
U: double_buffer realised as an explicit u/v pair, not by air-label-scf-for-to-ping-pong —
   the strip is allocated outside the t loop (HLD §7.2), so isPingPongCandidate's condition 2
   does not hold and the pass will not fire. The two buffers are emitted by hand.
```

Promising a pass will fire when it structurally cannot is exactly the silent-no-op failure R-05
is about; saying so is the point of the clause.

### 3.14 Check order

Fixed, because the *first* error is the one the user reads and it must be the most proximate one:

1. `BUILD` — rank, place/grid existence, extents, place∩sequential (§3.3) → `HERD-RANK`,
   `PLACE-EXTENT`, `PLACE-SEQUENTIAL-CONFLICT`
2. `RESOLVE_PHYSICAL` (§3.11)
3. (L1) (§3.4) → `L1-CONFLICT`
4. (L2) (§3.5) → `L2-CAUSALITY`
5. stationarity (§3.6) → `STATIONARITY`
6. reduction split + A/C (§3.7) → `REDUCE-NOT-ACCUMULATED`, `RSPACE-NO-AC-OP`
7. cascade rider (§3.8) → `CASCADE-RANK`, `CASCADE-BROADCAST`, `HERD-PHYSICAL`
8. halo (§3.9) → `HALO-TOO-SMALL`
9. ping-pong mode (§3.13) → `PINGPONG-SHAPE`
10. L1 capacity (§3.10) → `L1-CAPACITY`
11. swap parity (§3.12) → `SWAP-PARITY`

Capacity is late on purpose: a byte count is the least interesting thing to be told when the map
itself is illegal.

---

## 4. Invariants, pre/postconditions

**Preconditions**: `kernel` satisfies M1's postconditions
([`03-lld-M1-frontend.md`](03-lld-M1-frontend.md) §4); `schedule` satisfies M2's
([`03-lld-M2-schedule.md`](03-lld-M2-schedule.md) §4). M3 asserts both cheaply and treats a
violation as an internal error, not a user error.

**Postconditions of a returned `LegalMapping`** — every one is asserted before return, and each
is a thing M4 may rely on:

1. `len(pi) == len(schedule.grid)`; `pi` and `sigma` have `len(axes)` columns; `r_time` and
   `r_space` have `len(kernel.axes)` columns (§3.1's two-frame rule).
2. `intersect(ker(sigma), ker(pi))` is empty — (L1) holds.
3. `Sσ·d` is lexicographically positive for every dependence and every lift — (L2) holds.
4. `span(r_time) ⊕ span(r_space) == span(R)` and `r_time ⊆ ker Sπ_u`.
5. Every basis field (`ker_pi`, `r_time`, `r_space`) is in the canonical form of §3.2 — so two
   runs, in two processes, with different `PYTHONHASHSEED`, produce **equal** mappings (NFR-1).
6. `physical_herd[d]` divides `grid[d]` exactly and is ≤ the target cap;
   `repeats[d] == grid[d] // physical_herd[d]` (`_trace.py:1355-1372`).
7. `l1_bytes <= 65536`.
8. `stationary_ops` contains every declared `stationary` operand and is sorted.
9. `halo_footprint` has one entry per `WindowClause`, each ≤ the declared halo.

**Module invariants**: M3 imports no `air` and touches no filesystem, no network, no clock
(FR-L13, HLD §5). Every exit is a `LegalMapping` or a `SpatialError` (NFR-7) — the only internal
failure modes are in `intlin`, and a singular system there returns `None` rather than raising.

---

## 5. Error paths

All are `LegalityError(Diagnostic(...))`, `stage="legality"`, `clause` **always set** (§6.1 makes
it mandatory for this stage), rendered in HLD §4.2's four-part shape. `details` carries the
matrices and numbers as JSON-serialisable lists, never pre-formatted prose — that is what lets
the message be improved without breaking a test (FR-D3, decision D-7).

| Code | Raised in | `because` line carries |
|---|---|---|
| `L1-CONFLICT` | §3.4 | the conflict basis, `ker Sσ`, `ker Sπ`, the axis names |
| `L2-CAUSALITY` | §3.5 | the dependence, the lifted representative, `Sσ·d`, the violating row |
| `STATIONARITY` | §3.6 | `ker M_a`, `ker Sπ`, the placed axes |
| `REDUCE-NOT-ACCUMULATED` | §3.7 | the axis, `R = ker Sf`, the accumulation target |
| `RSPACE-NO-AC-OP` | §3.7 | `R_space`, `R_time`, the placed axes, the missing tag |
| `CASCADE-RANK` | §3.8 (a)(b)(c) | `rank(R_space)`, the carrying axis, the grid |
| `CASCADE-BROADCAST` | §3.8 (d) | the operand and `_channel.py:170-177`'s own sentence |
| `HALO-TOO-SMALL` | §3.9 | derived footprint, declared halo, accessed and owned ranges |
| `PLACE-EXTENT` | §3.3 | the axis, its extent, the grid extent, the position |
| `PLACE-SEQUENTIAL-CONFLICT` | §3.3 | the axis and both clauses |
| `HERD-RANK` | §3.3 | the rank and `_trace.py:1288-1291`'s sentence |
| `HERD-PHYSICAL` | §3.8 (e) | grid, physical, repeats, the cap table |
| `L1-CAPACITY` | §3.10 | per-buffer breakdown, which were doubled, the total, 65536 |
| `PINGPONG-SHAPE` | §3.13 | which numbered `isPingPongCandidate` condition, with its source line |
| `SWAP-PARITY` | §3.12 | the loop, its trip count, the clause that sets it |

Three get a hand-tuned message because they are the demo's set pieces
(`04-test-plan.md` §5). Two are printed in full above (§3.13) and below (§6.5); the third is
`L1-CAPACITY`, whose `details["per_buffer"]` is rendered as a table so the reader sees which
tile to halve.

---

## 6. Worked examples

Values only; the schema is `06-interfaces.md` §4.1. Coordinate and σ orders are
[`03-lld-M2-schedule.md`](03-lld-M2-schedule.md) §3.6.

### 6.1 W1 — GEMM output-stationary (`npu1`, `M=N=K=64`, `TM=TN=32`, `TK=16`, `PI=PJ=2`)

`axes = (i0, i1, j0, j1, k0, k1)`, extents `(2, 32, 2, 32, 4, 16)`; `UCoord = (i, j, k)`.

```
       i0 i1 j0 j1 k0 k1                     i0 i1 j0 j1 k0 k1
Sπ  = [ 1  0  0  0  0  0 ]          Sσ  = [   1  0  0  0  0  0 ]   row i0
      [ 0  0  1  0  0  0 ]                [   0  0  1  0  0  0 ]   row j0
                                          [   0  0  0  0  1  0 ]   row k0
ker Sπ = { e_i1, e_j1, e_k0, e_k1 }       [   0  1  0  0  0  0 ]   row i1
ker Sσ = { }            (permutation)     [   0  0  0  1  0  0 ]   row j1
                                          [   0  0  0  0  0  1 ]   row k1
```

Untiled frame: `Sπ_u = [e_i; e_j]`, `ker Sπ_u = span{e_k}`;
`M_A = ((1,0,0),(0,0,1))`, `ker M_A = span{e_j}`;
`M_B = ((0,0,1),(0,1,0))`, `ker M_B = span{e_i}`;
`M_C = ((1,0,0),(0,1,0))`, `ker M_C = span{e_k}`; `Sf = M_C`.

| check | result |
|---|---|
| (L1) | `ker Sσ = {0}` → intersection `{0}` ✓ |
| (L2) | `d = (0,0,1)`; interior lift `(0,0,0,0,0,1)` → `Sσ·d = (0,0,0,0,0,1)` ✓; boundary lift `(0,0,0,0,1,−15)` → `(0,0,1,0,0,−15)` ✓ |
| stationarity | `ker M_C = span{e_k} ⊆ span{e_k}` ✓ declared; derived: A ✗, B ✗ |
| reduction | `R = span{e_k}`; `R_time = span{e_k}`; `R_space = {}` — output-stationary, no network (SD-02 §5 R1) |
| cascade | not applicable (`R_space` empty) |
| physical | npu1 `(1,2)` repeats `(2,1)`; npu2 `(2,2)` repeats `(1,1)` |
| ping-pong | `A`, `B` → `PASS` (read-only; `M_A·e_k ≠ 0`, `M_B·e_k ≠ 0`; one access group per K trip) |
| L1 | `acc [32,32] f32 = 4096` + `A [32,16] ×2 = 4096` + `B [16,32] ×2 = 4096` = **12288** of 65536 |

`LegalMapping`: `ker_pi = ((0,1,0,0,0,0),(0,0,0,1,0,0),(0,0,0,0,1,0),(0,0,0,0,0,1))`;
`r_time = ((0,0,1),)`; `r_space = ()`; `stationary_ops = ("C",)`;
`halo_footprint = ()`; `l1_bytes = 12288`. Every figure matches HLD §7.1.

### 6.2 W1-flip — weight-stationary, cascade (`grid(4)`, `place(px=ax.k0)`, `stationary("B")`)

Same kernel text; `tile(ax.i, 32)`, `tile(ax.k, 16)` and **no tile on `j`** (RULING 9), so
`axes = (i0, i1, j, k0, k1)`, extents `(2, 32, 64, 4, 16)`; `UCoord = (i, j, k)`.
`Sπ = [e_k0]` over `Coord`; `Sπ_u = [e_k]`; `ker Sπ_u = span{e_i, e_j}`;
`σ = (i0, j, k0, i1, k1)` by §3.6's default rule (no `skew`).

| check | result |
|---|---|
| (L1) | `ker Sσ = {0}` ✓ |
| (L2) | `d = (0,0,1)`; interior lift `(0,0,0,0,1)` → `Sσ·d = (0,0,0,0,1)` ✓; boundary lift `(0,0,0,1,−15)` → `(0,0,1,0,−15)`, positive on the `k0` row ✓ |
| stationarity | declared `B`: `ker M_B = span{e_i} ⊆ span{e_i, e_j}` ✓. Derived report: `A` ✓ (`span{e_j} ⊆`), `B` ✓, `C` ✗ (`span{e_k} ⊄`) → `stationary_ops = ("A","B")` |
| reduction | `R = span{e_k}`; `R_time = span{e_k} ∩ span{e_i,e_j} = {}`; **`R_space = span{e_k}`** — SD-02 §9's "flip one pragma" |
| A/C | `reduce(ax.k, op="+")` present ✓ (without it: `RSPACE-NO-AC-OP`) |
| cascade | rank 1 ✓; one carrying PE axis (`px`) ✓; herd 1-D ✓; no broadcast on `C` ✓; `repeats = (1,)` ✓ → no `at=` pinning needed (Q-1 measured); `HerdPlan.at = None` |
| physical | npu1 `(4,)` repeats `(1,)`; npu2 `(4,)` repeats `(1,)` |
| ping-pong | `A` → `PASS`: read-only, its `[32,16]` tile varies along the pinned temporal **tile** axis `i0` (`M_A · e_i ≠ 0`), and there is one access group per `i0` trip. `B` is **not** declared and could not be — `M_B · e_i = 0` and `j` is untiled, so no temporal tile axis moves `B`'s tile and `double_buffer("B")` would raise `PINGPONG-SHAPE` (condition 2, §3.13) |
| L1 | `acc [32,64] = 8192` + `a [32,16] ×2 = 4096` (`double_buffer("A")`) + `b [16,64] = 4096` = **16 384** at M3's scope. The cascade needs an L1 `recv [32,64] = 8192` tile that M4 synthesises, so the plan's figure is `16 384 + 8 192 = ` **24 576** (`02-hld.md` §7) |

`r_time = ()`; `r_space = ((0,0,1),)`; `stationary_ops = ("A","B")`; `l1_bytes = 16384` at M3's
scope, `24 576` in the plan once `recv` exists (`03-lld-M4-mapping.md` §6.2). M4 turns `r_space`
into **`PK−1 = 3` cascade links on one `npu_cascade` bundle**, `chain_direction = "ascending"`,
with a head at `tx == 0` that skips its `get` and a tail at `tx == PK-1` that writes
`C[i0 tile, :]` to L3 once per `i0` trip (FR-M6) — HLD §7.1's flip paragraph.

**Two negatives live in this fixture's neighbourhood**, and both are in §7: `grid(4,2)` with
`place(px=ax.k0, py=ax.j0)` — which needs `tile(ax.j, 32)` put back so that a `j0` axis exists to
place — → `CASCADE-RANK` (c); `grid(8)` on `npu1` → `HERD-PHYSICAL`
(repeats 2 folds the chain). Note the herd is **1-D**: the cascade chain then ascends in `tx`,
measured (REVIEW-round1 P-R3); a 2-D `(1,4)` herd is the case that must descend, and
`ChannelPlan.chain_direction` carries which.

### 6.3 W2 — Jacobi (`npu1`, `T=4`, `H=W=16`, `U[T+1, H+2, W]`, `PI=2`, `HS=8`)

`axes = (t, i0, i1, j)`, extents `(4, 2, 8, 14)`; `UCoord = (t, i, j)`. The `i` extent is
`H = 16` (rows `1..H`) and `HS = 8` divides it exactly, so `PI·HS == H`; the `j` extent is
`W − 2 = 14` and `j` is neither tiled nor placed.

```
        t i0 i1  j                       t i0 i1  j
Sπ  = [ 0  1  0  0 ]          Sσ  =  [   1  0  0  0 ]   row t
                                     [   0  1  0  0 ]   row i0
ker Sπ = { e_t, e_i1, e_j }          [   0  0  0  1 ]   row j
ker Sσ = { }                         [   0  0  1  0 ]   row i1
```

`Sπ_u = [e_i]`, `ker Sπ_u = span{e_t, e_j}`; `M_U = I₃` for every access, `ker M_U = {0}`.

| check | result |
|---|---|
| consistency | `place = ("i0",)`, `grid = (2,)`, extent(`i0`) `= 16/8 = 2` ✓; `sequential("t")` not placed ✓ |
| (L1) | `ker Sσ = {0}` ✓ |
| (L2) | five dependences × two lifts each. `(1,0,0)→(1,0,0,0)→Sσ·d = (1,0,0,0)` ✓; `(1,1,0)` interior `(1,0,1,0)→(1,0,0,1)` ✓, boundary `(1,1,−7,0)→(1,1,0,−7)` ✓; `(1,−1,0)` interior `(1,0,−1,0)→(1,0,0,−1)` ✓, boundary `(1,−1,7,0)→(1,−1,0,7)` ✓; `(1,0,±1)→(1,0,0,±1)→(1,0,±1,0)` ✓. Every one leads with the `t` row, which is why (L2) is never the binding check for W2 |
| stationarity | none declared; derived `ker M_U = {0} ⊆ ker Sπ_u` ✓ vacuously → `stationary_ops = ("U",)`, rendered "stationary strip" |
| reduction | `kernel.reduction is None` → `R = R_time = R_space = {}` |
| halo | pinned box: `t` pinned (sequential), `i0` pinned (placed), `i1` full `[0,8)`, `j` full `[1,15)`. Along array dim 1: owned `[8p+1, 8p+8]`, accessed `[8p, 8p+9]` → overhang `(1, 1)` → derived **1**; along dim 2: owned `[1,14]`, accessed `[0,15]` → derived **1**. Declared `halo=1` ✓ (with `halo=0`: `HALO-TOO-SMALL`, derived 1 vs declared 0) |
| ping-pong | `U` has a `WindowClause` and an `ExchangeClause` → mode **`PAIR`**, message per §3.13 (decision D-5) |
| L1 | span `(2, 10, 16)` f32 → `2·10·16·4` = **1280**, not doubled again (`PAIR`). Dim 0 is the `{t, t+1}` plane pair, dim 1 is `HS + 2` ghost-padded rows, dim 2 is all `W = 16` columns because `j ± 1` reaches columns `0` and `W−1` |
| swap parity | the pinned-axis span on `t` is 2 → a swap pair; `T = 4 ≥ 1` ✓ (`T = 5` is **accepted**, two pairs plus one peeled body — the FR-L14 override; `T = 0` is the `SWAP-PARITY` negative) |
| physical | npu1 `(2,)` repeats `(1,)`; npu2 `(2,)` repeats `(1,)` |

`ker_pi = ((1,0,0,0),(0,0,1,0),(0,0,0,1))`; `r_time = r_space = ()`;
`stationary_ops = ("U",)`; `halo_footprint = (("U", (1,1)),)`; `l1_bytes = 1280`.

> **`PI = 2` is forced, not chosen.** `PI = 4` divides `H = 16` and passes every check in this
> module, and then fails inside `aircc` with `'aie.connect' op … TileID(1, 2) targets same dst`,
> because the first interior PE needs three circuit-switched inbound channels against a 2-S2MM
> budget (REVIEW-round1 P-R2). `03-lld-M4-mapping.md` §3.8's `DMA-CHANNELS` check is what turns
> that into a pre-codegen rejection, and `PI = 4` is a negative fixture. The earlier `H = W = 18`
> / `864` pair is retired: it fixed divisibility but kept `PI = 4`.

### 6.4 W3 — Smith-Waterman wavefront (`npu1`, `MQ=NR=32`, `PJ=4`, `CW=8`)

`axes = (i, j0, j1)`, extents `(32, 4, 8)`; `UCoord = (i, j)`. Parameters are `q i32 [32]`,
`r i32 [32]` (both read-only) and `S i32 [33, 33]` (written) — in that order.

```
         i j0 j1                        i j0 j1
Sπ  = [  0  1  0 ]           Sσ  =  [   1  1  0 ]   row (i + j0)   <- skew(time=(ax.i, ax.j0))
                                    [   0  0  1 ]   row j1         <- appended, §3.3 line 24
ker Sπ = { e_i, e_j1 }
ker Sσ = { (1, −1, 0) }
```

`Sπ_u = [e_j]`, `ker Sπ_u = span{e_i}`.

| check | result |
|---|---|
| (L1) | `ker Sσ = span{(1,−1,0)}`; a multiple of it lies in `ker Sπ` only if its `j0` entry is 0, i.e. only the zero vector → intersection `{0}` ✓. **Without the appended `j1` row**, `ker Sσ` would also contain `e_j1 ∈ ker Sπ` and (L1) would fail — this is why §3.3 line 24 exists |
| (L2) | `(1,0)→(1,0,0)→Sσ·d = (1,0)` ✓; `(0,1)` interior `(0,0,1)→(0,1)` ✓, boundary `(0,1,−7)→(1,−7)` ✓; `(1,1)` interior `(1,0,1)→(1,1)` ✓, boundary `(1,1,−7)→(2,−7)` ✓ |
| stationarity | derived, in `UCoord`: `ker M_S = {0} ⊆ span{e_i}` ✓; `M_r = (0,1)` so `ker M_r = span{e_i} ⊆ span{e_i}` ✓; `M_q = (1,0)` so `ker M_q = span{e_j} ⊄ span{e_i}` ✗ → `stationary_ops = ("S","r")`, and `q` is classified **multicast along px** by M4 (FR-M1) |
| reduction | `None` → all empty (the 4-ary `max` is a window, not a reduction over an axis — HLD §7.3) |
| halo | no `WindowClause`; `halo_footprint = ()`. The `j`-direction overhang of 1 still shows up in the tile span below |
| L1 | pinned box: `i` pinned (in `skew`, unplaced), `j0` pinned (placed), `j1` full `[0,8)`. `q` is read at `q[i-1]`, which does not index `j` at all, so every PE holds the **whole** vector: `32 · 4` = **128**. `r` is read at `r[j-1]`: owned `[8p, 8p+7]` → span 8 → `8 · 4` = **32**. `S` span along dim 0 = `{i−1, i}` → **2**; along dim 1: owned `[8p+1, 8p+8]`, accessed `[8p, 8p+8]` → span **9**; `2·9·4` = **72**. Total **232** (M4 adds the two staging scalars `edge_in`/`edge_out` → **240**; §3.10) |
| swap parity | pinned-axis span on `i` is 2 → a swap pair; `MQ = 32 ≥ 1` ✓ |
| physical | npu1 `(4,)` repeats `(1,)` |

`ker_pi = ((1,0,0),(0,0,1))`; `r_time = r_space = ()`; `stationary_ops = ("S","r")`;
`l1_bytes = 232`.

### 6.5 The demo's headline rejection — W3 with the skew dropped

`skew(time=(ax.i,))` instead of `skew(time=(ax.i, ax.j0))`. Then `Sσ = [[1,0,0],[0,0,1]]` (row
`i`, then the appended `j1`), and the boundary lift of `d = (0,1)` — the dependence that carries
a score from the last column of PE `p` to the first column of PE `p+1` — gives
`Sσ·d = (0, −7)`.

```
L2-CAUSALITY: dependence (0, 1) on S is not causal under this schedule
  in clause: skew(time=(ax.i,))
  at:        w3_sw.py:6
  because:   S[i, j] reads S[i, j-1], a distance d = (0, 1). Tiled by j -> (j0, j1) with
             CW = 8, the tile-crossing representative is (0, 1, -7), and
             Sσ·d = (0, -7): the first nonzero entry is -7, not >= 1. The value crosses
             from PE p to PE p+1 with no advance in time, so PE p+1 would have to consume
             it before PE p produced it.
  fix:       put the PE axis into time: skew(time=(ax.i, ax.j0)). That makes
             Sσ·d = (1, -7), which is lexicographically positive, and is the anti-diagonal
             wavefront.
```

FR-L2's acceptance names `skew(time=(ax.j0, ax.i))` "reversed" as the negative. It is **not** a
negative: FR-S17 defines `skew` as a **sum**, and `j0 + i = i + j0`, so the reversal produces an
identical `Sσ` and is accepted. The realisable negative is the dropped term above; see §10
Q-M3-4.

---

## 7. Unit tests

Positive, one per workload (`04-test-plan.md` §2 M3), asserting the **whole** `LegalMapping`
field by field against §6: `test_M3_w1`, `test_M3_w1flip`, `test_M3_w2`, `test_M3_w3` —
`sigma`, `pi`, `ker_pi`, `r_time`, `r_space`, `stationary_ops`, `physical_herd`, `repeats`,
`l1_bytes`, `halo_footprint`.

| Test id | Input (concrete) | Expected |
|---|---|---|
| `test_physical_herd_table` | the six rows of §3.11 | the tabulated physical/repeats |
| `test_l1_capacity_doubles` | W1 with and without `double_buffer("A","B")` | `12288` vs `8192`; the difference is exactly the two doubled tiles |
| `test_M3_pair_not_doubled` | W2 with and without `double_buffer("U")` | `1280` both times; the mode is `PAIR` and the summary carries the D-5 sentence |
| `test_M3_derived_stationarity` | W1, W1-flip, W2, W3 | `stationary_ops` = `("C",)`, `("A","B")`, `("U",)`, `("S","r")` — `q` is **not** stationary on W3, because `ker M_q = span{e_j} ⊄ ker Sπ_u = span{e_i}` (§6.4) |
| `test_M3_lift_both_reps` | a synthetic σ ordering `(i0, t, i1, j)` on W2 | rejected `L2-CAUSALITY` on the boundary representative `(1,−1,7,0)` → `Sσ·d = (−1,…)`; the **interior** representative alone would pass, so this test is what proves §3.5 line 6 earns its place |
| `test_M3_deterministic` | `check()` twice in one process and once in a fresh process with a different `PYTHONHASHSEED` | equal `LegalMapping`s, equal basis orders (NFR-1) |
| `test_M3_no_numpy_linalg` | source lint over the module | zero occurrences of `numpy.linalg` / `np.linalg` (§3.2) |
| `test_M3_no_air_import` | run the full checker in a subprocess | `air` absent from `sys.modules` (FR-L13) |
| `test_legality_error_schema` | every negative below | `code` in the catalogue; `clause`, `reason`, `fix` non-empty; `details` JSON-serialisable (FR-L12, FR-D1) |
| **property** `test_intlin_kernel` | 200 random integer matrices, `n ≤ 6`, entries in `[−4, 4]`, fixed seed | `M @ v == 0` for every basis vector `v`; `rank(M) + len(kernel_basis(M)) == n`; every basis vector primitive with a positive leading entry; the basis is stable under repeated calls |
| **property** `test_intlin_no_overflow` | matrices with entries up to 256, `n = 8` | results equal a `fractions.Fraction` reference implementation exactly (the reference is test-only) |

Negative corpus — one per legality code, **16 of the catalogue's 43** (`04-test-plan.md` §5):

| Test id | Input (concrete) | Expected |
|---|---|---|
| `test_L1_conflict` | W1 + `skew(time=(ax.i1, ax.j1))` | `L1-CONFLICT`; `details.conflict_basis == ((0,1,0,−1,0,0),)` i.e. `e_i1 − e_j1`, which is in `ker Sσ` and in `ker Sπ` |
| `test_L2_causality` | W3 + `skew(time=(ax.i,))` | `L2-CAUSALITY`; `details.dependence == (0,1)`, `representative == (0,1,−7)`, `sigma_d == (0,−7)` — the §6.5 message |
| `test_L3_stationarity` | W1 + `grid(2,4)`, `place(px=ax.i0, py=ax.k0)`, `stationary("C")` | `STATIONARITY`; message names `C`, `ker M_C = span{e_k}`, `ker Sπ = span{e_j}`, and the placed axes (FR-L3's acceptance verbatim) |
| `test_L4_split` | W1 and W1-flip | `r_time == ((0,0,1),) / r_space == ()` and `r_time == () / r_space == ((0,0,1),)` — FR-L4's acceptance |
| `test_reduce_axis_must_accumulate` | W1 + `reduce(ax.i, op="+")` | `REDUCE-NOT-ACCUMULATED`; `details.R == ((0,0,1),)` |
| `test_L5_ac_required` | W1-flip with `reduce` deleted | `RSPACE-NO-AC-OP`; `fix` contains the literal `reduce(ax.k, op="+")` |
| `test_L6_cascade_rank` | W1-flip + `tile(ax.j, 32)`, `grid(4,2)`, `place(px=ax.k0, py=ax.j0)` — `j` is re-tiled only so that a second placeable axis exists | `CASCADE-RANK` sub-clause (c); `details.grid == (4,2)` |
| `test_L6_cascade_no_broadcast` | W1-flip + `stream("C", pattern="broadcast", along=ax.k0)` | `CASCADE-BROADCAST`, raised **before** `air.api` would (`_channel.py:170-177`) |
| `test_L7_halo_too_small` | W2 + `window("U", dims=(ax.i, ax.j), halo=0)` | `HALO-TOO-SMALL`; `details.derived == 1`, `declared == 0` |
| `test_L8_place_extent` | W1 + `grid(2,4)` with `place(px=ax.i0, py=ax.j0)` | `PLACE-EXTENT`; `details == {axis:"j0", extent:2, grid:4, position:1}` |
| `test_L8_place_sequential` | W2 + `sequential(ax.i0)` | `PLACE-SEQUENTIAL-CONFLICT` naming `i0` and both clauses |
| `test_L11_rank` | a `ScheduleModel` built programmatically with `grid = (2,2,2)` | `HERD-RANK` quoting `_trace.py:1288-1291` |
| `test_L10_herd_physical` | W1-flip + `grid(8)` on `npu1` | `HERD-PHYSICAL`; `details == {grid:(8,), physical:(4,), repeats:(2,), cap:(4,)}` |
| `test_L9_capacity` | W1 with `M=N=K=192`, `TM=TN=96`, `TK=32`, `f32`, `double_buffer("A","B")` | `L1-CAPACITY`; `details.undoubled == 61440`, `total == 86016`, `budget == 65536`, `over == 20480`, per-buffer `[C 36864, A 12288×2, B 12288×2]` — FR-L9's acceptance. The old `TM=TN=TK=128` `bf16` case is `98304` B **before** doubling, so it never demonstrates the doubling |
| `test_S13_pingpong_shape` | W1 + `double_buffer("C")` | `PINGPONG-SHAPE`; `details.condition == 3` and the §3.13 message text |
| `test_L14_swap_parity` | W2 with `T = 5` / with `T = 0` | `T = 5` **accepted**, `details`-free, and the plan peels one body; `T = 0` (fixture `w2_zero_t`, `03-lld-M8-kernels-demo.md` §4) rejected `SWAP-PARITY` naming `t`, `0` and the fixture parameter (the adjudicated FR-L14). `T = 0` is the **only** reachable `SWAP-PARITY` condition after the D-4 override, so without this fixture `test_D3_catalogue_complete` fails |
| `test_M4_reside_l2` | W2 + `reside(U="L2")` | `PROTOCOL-UNSUPPORTED` whose `clause` is `reside`; L2 residency is out of scope for this cut (FR-S12, RULING 7). Raised by M4, listed here because it completes the legality-adjacent corpus |

---

## 8. Dependencies

**On other modules**: `model` (M0) for `LegalMapping`, `LegalityError`, `Diagnostic`; M1's
`KernelModel` (access matrices, dependence vectors, reduction projection); M2's `ScheduleModel`
and its two order definitions ([`03-lld-M2-schedule.md`](03-lld-M2-schedule.md) §3.6). M3 is
consumed by M4 (`LegalMapping`, plus `m3.pingpong_mode`) and by M1 (`m3.intlin`). M3 imports
**nothing** from M4, M5 or M6.

**On Python and numpy**:

| Operation | Where | Implementation |
|---|---|---|
| matrix–vector `Sσ · d`, `Sπ_u · g` | §3.5, §3.8 | numpy `int64` arrays and `@` — magnitudes in the hundreds, no overflow risk |
| matrix–matrix (stacking for `intersect`, `contains`) | §3.2 | numpy `int64` `@` and `vstack` |
| **rank** | §3.2 | `intlin`: Bareiss row echelon, count of pivots. **Not** `numpy.linalg.matrix_rank` (float SVD, tolerance-dependent, non-deterministic across machines — breaks NFR-1) |
| **integer null space / kernel basis** | §3.2 | `intlin.KERNEL_BASIS`: Bareiss echelon + back-substitution + primitive + sign normalisation. **Not** an SVD null space |
| **integer solve** `M·d = b` | M1 §3.7, via `intlin` | Bareiss forward elimination + back-substitution; returns `None` when no integral solution exists |
| subspace intersection / containment / complement | §3.2 | composed from the above |

Python standard library: `dataclasses`, `math.gcd`, `typing`. **No `fractions`** (it is used only
inside one test-only reference implementation), **no `sympy`**, **no `air`**, no filesystem, no
network, no clock.

**Stubs M3 needs**: none to be written. M3 is testable the moment M1 and M2 land, and its
`LegalMapping` for W1 is what B's hand-written literal (HLD §8, D0) is checked against on D2.

---

## 9. Implementation order, effort, definition of done

Order, which is also the order in which the checks become demonstrable:
(1) `intlin` and its property tests — everything else is wrong if it is;
(2) `BUILD` and the two frames (§3.3), with the W1/W2/W3 matrices as the first assertions;
(3) (L1) and (L2) with `LIFT` — the demo's headline lives here;
(4) stationarity and the derived report;
(5) the reduction split, the A/C gate and the cascade rider — this is what unlocks the W1 flip;
(6) `FOOTPRINT`, then halo, then capacity, then `pingpong_mode`, then swap parity;
(7) the message renderer and the three hand-tuned set pieces.

Steps 1–4 are D2's deliverable (`05-work-breakdown.md` §2 D2: "M3 checks L1, L2, stationarity,
grid/place/tile consistency, L1 capacity, herd physical shape"); 5–6 land on D4–D5 with W3 and
W2; 7 finishes on D6.

**Effort — estimate, not measured: 10 hours.** ~2 h `intlin` and its properties, ~2 h the frames
and the matrices, ~2 h (L1)+(L2)+`LIFT`, ~2 h reduction split + cascade rider, ~1 h footprint
and capacity, ~1 h the messages. A further ~3 h for the negative corpus sits on D3 and D6 in the
work breakdown, not here.

**Definition of done.** The four positive mappings of §6 are asserted field by field; all 16
negatives fire with the right code and all four message parts; `test_intlin_kernel` and
`test_intlin_no_overflow` pass; `test_M3_deterministic`, `test_M3_no_numpy_linalg` and
`test_M3_no_air_import` pass; every legality code in `06-interfaces.md` §6.3 has at least one
test that raises it (FR-D3); and M4 has consumed a `LegalMapping` for W1, W2, W3 and the flip
without asking for a field or a frame this document does not fix.

---

## 10. Open questions owned by M3

| # | Question | Resolution |
|---|---|---|
| **Q-M3-1** | In which frame do `LegalMapping.r_time` and `.r_space` live? `06-interfaces.md` §4.1 fixes the fields but not the basis, and `sigma`/`pi` are over post-tiling axes while `M_a` is not. | **Resolved here** (§3.1's two-frame rule): `sigma`, `pi`, `ker_pi` over `Coord` (post-tiling); `r_time`, `r_space` over `UCoord` (untiled, `kernel.axes` order), because `ker M_a` in the tiled frame contains artificial tile-collapse directions (`e_i0 − TM·e_i1`) that are not iteration-space directions at all — computing stationarity there would reject every correct schedule. **M4 must read the same convention** and now can: `06-interfaces.md` v2 carries `LegalMapping.pi_u` and `.ker_pi_u` plus the two-frame note, so M4 no longer re-derives `root(place)` (REVIEW-round1 B-5 / RULING 5). **Closed at the D0 signature.** |
| **Q-M3-2** | `HERD-PHYSICAL` says "no divisor of a grid extent fits the target's physical cap", but every extent has the divisor 1, so `_resolve_physical` never fails — the code is unreachable and `test_D3_catalogue_complete` (FR-D3) would have nothing to raise it. | **Resolved here** (§3.8 lines 27–32): the reachable condition is a **cascade chain folded by strip-mining** — `repeats[p] != 1` on the axis carrying `R_space` breaks the chain, and that is a genuine physical-resolution failure. `grid(8)` on `npu1` is the fixture. No interface change. |
| **Q-M3-3** | `double_buffer`'s two meanings (decision D-5) need to be recorded somewhere M4 can read, but `LegalMapping` is frozen with no field for it. | **Resolved here** (§2, §3.13): M3 exports the pure helper `m3.pingpong_mode(mapping, operand)` and M4 calls it. One definition, no duplicated predicate, no field, no contract bump. |
| **Q-M3-4** | FR-L2's acceptance names `skew(time=(ax.j0, ax.i))` "reversed" as the causality negative, but FR-S17 defines `skew` as a **sum**, so the reversal is a no-op. | **Resolved here** (§6.5): the realisable negative is `skew(time=(ax.i,))` — the dropped term — which fails on the tile-crossing representative of `d = (0,1)` with `Sσ·d = (0,−7)`. It is a **better** demo (it is the mistake a user actually makes). **Closed**: FR-L2's acceptance now names `skew(time=(ax.i,))` outright (REVIEW-round1 EDIT-38), and FR-S17 states the σ-construction rule that makes it a rejection (RULING 6). |
| **Q-M3-5** | FR-L1's acceptance describes the conflict as "placing `i` and leaving `i` out of `σ`", but with §3.3's σ construction every non-placed, non-skewed axis gets its own σ row, so that schedule is legal. | **Resolved here** (§3.3's property, §7): with no `skew`, `Sσ` is a permutation and (L1) holds unconditionally; a conflict requires a `skew` that names **two unplaced axes**, e.g. W1's `skew(time=(ax.i1, ax.j1))`, whose `ker Sσ ∩ ker Sπ = span{e_i1 − e_j1}`. That is the fixture. **Closed**: FR-L1's acceptance now names `skew(time=(ax.i1, ax.j1))` (REVIEW-round1 EDIT-39). |
