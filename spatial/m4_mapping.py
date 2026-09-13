"""M4 — mapping / protocol synthesis. Owner: Person B. LLD: design/03-lld-M4-mapping.md.

Entry point per design/06-interfaces.md §7.2. M4 turns a `LegalMapping` — what the schedule is —
into a `MappingPlan` — exactly what IR to build. Every decision is taken here, because D-14
makes M5 a mechanical translator: if M5 would have to choose something, the choice belongs here.

**Nothing here imports `air` and nothing imports `spatial.m5_emit`** (invariant I-1): M4 is
pure Python over the M0 dataclasses and is testable with no toolchain. The only arithmetic is
small exact integer linear algebra over `Fraction` (rank and null-space of matrices with at most
six columns), so `numpy` is not imported either.

**What this cut builds.** The ten passes of LLD §3.1 in their fixed order, general in the
machinery (`TILE_SHAPE` from the `AccessMap` and the tile factors, `CLASSIFY` in `UCoord`,
`MULTICAST_GEOMETRY`, `L3_REGION`, `LOOP_KIND`, `TENSOR_PLAN`, `RESIDENCY`, `SUMMARY`), with
**one protocol builder**: the multicast / stationary fill-compute-drain shape of LLD §6.1. The
halo (§3.6.1), wavefront (§3.6.2) and cascade (§3.6.3) builders raise `NotImplementedError`
naming the phase; §3.1's `PLAN` wrapper re-raises any non-`SpatialError` as a `MappingError`
carrying it in `details["internal_exception"]` (§5), so an input that needs one of them fails
loudly and legibly instead of silently producing a wrong plan.

**Determinism** (HLD §5, invariant I-7): every traversal that affects a name, an order or a
text is `sorted(...)` on an explicit key; no `id()`, no clock, no RNG.
"""

from __future__ import annotations

from dataclasses import fields, replace
from fractions import Fraction
from math import lcm, prod
from typing import Any, get_args

from spatial.m4_selfcheck import self_check
from spatial.model import (Axis, BufferPlan, ChannelPlan, ChannelSite, Const, Diagnostic, Dtype,
                           Expr, ExprNode, HerdPlan, LegalMapping, Load, LoopPlan, MappingError,
                           MappingPlan, MappingSummary, Param, Region, SpatialError, StoreNode)

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------

_EXPR_NODES: tuple[type, ...] = get_args(ExprNode)
"""The `ExprNode` union's members, for the one tree rewrite M4 does (`06-interfaces.md` §5.5)."""

PE_AXIS_NAME = ("px", "py")
"""The PE axis names, in `ScheduleModel.place` order (LLD §3.2)."""

HERD_COORDS = ("tx", "ty")
"""The herd body's coordinate parameter names, in rank order (LLD §3.3 step 1)."""

L1_BUDGET = 65536
"""`L1_BYTES` (`_trace.py:100`) — the budget `06-interfaces.md` §5.6 invariant 5 charges."""

EMPTY_REGION = Region((), (), ())
"""The L1 end of a whole-buffer transfer (LLD §3.4, measured as `(%alloc[] [] [])`)."""

RESIDENT = "resident for the whole run"
"""The `MappingSummary.residency` duration of an operand no temporal tile axis moves."""

ZERO = Expr()
"""The constant 0 — every plan loop's `lo` and the index of a size-1 channel dim."""

ONE = Expr((), 1)
"""The constant 1 — every unrolled and compute loop's `step`."""

_PHYSICAL_CAP = {("npu1", 1): (4,), ("npu1", 2): (1, 4),
                 ("npu2", 1): (8,), ("npu2", 2): (2, 4)}
"""The physical array caps per target and herd rank (`_trace.py:88-91`, 07-environment §5)."""

_CLAUSE = "plan()"
"""The surface call every mapping diagnostic without a user clause of its own points at."""

_LATER = "lands in P4/P5/P6: this cut builds the LLD §6.1 fill/compute/drain protocol only"
"""Every unbuilt path carries this phrase, so an unsupported input says which phase owns it."""

_BUG_FIX = ("this is a defect in the compiler, not in your program: please report it with the "
            "LegalMapping (spatial.model.to_json(mapping)) and this message")

_PATTERN = {"broadcast": "MULTICAST", "forward": "FORWARD", "cascade": "CASCADE"}
"""`Pattern` → `Delivery` for a declared `stream`/`forward` clause (LLD §3.2 line 21)."""


# --------------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------------


def _fail(reason: str, fix: str, *, clause: str = _CLAUSE, **details: Any) -> MappingError:
    """Build a `PROTOCOL-UNSUPPORTED` `MappingError` with all four message parts (§5)."""
    return MappingError(Diagnostic(code="PROTOCOL-UNSUPPORTED", stage="mapping", clause=clause,
                                   reason=reason, fix=fix, location=None, details=details))


def _internal(reason: str, **details: Any) -> MappingError:
    """A precondition M3 owed us, or a table we built ourselves, is wrong: our defect (§5)."""
    return _fail(reason, _BUG_FIX, internal_consistency=True, **details)


# --------------------------------------------------------------------------------------------
# Affine-expression and exact-integer-linear-algebra helpers
# --------------------------------------------------------------------------------------------


def _var(name: str, coeff: int = 1, const: int = 0) -> Expr:
    """The affine expression `coeff * name + const`."""
    return Expr({name: coeff}, const)


def _const(value: int) -> Expr:
    """The constant affine expression `value`."""
    return Expr((), value)


def _add(*terms: Expr) -> Expr:
    """The sum of affine expressions."""
    coeffs: dict[str, int] = {}
    const = 0
    for term in terms:
        for name, coeff in term.coeffs:
            coeffs[name] = coeffs.get(name, 0) + coeff
        const += term.const
    return Expr(dict(sorted(coeffs.items())), const)


def _scale(expr: Expr, factor: int) -> Expr:
    """`factor * expr`."""
    return Expr({name: coeff * factor for name, coeff in expr.coeffs}, expr.const * factor)


def _resolve(expr: Expr, bindings: dict[str, int]) -> Expr:
    """Substitute the shape-parameter bindings of `KernelModel.bindings` (v4, §2.7)."""
    coeffs: dict[str, int] = {}
    const = expr.const
    for name, coeff in expr.coeffs:
        if name in bindings:
            const += coeff * bindings[name]
        else:
            coeffs[name] = coeff
    return Expr(coeffs, const)


def _rref(rows: tuple[tuple[int, ...], ...],
          columns: int) -> tuple[list[list[Fraction]], list[int]]:
    """Reduced row echelon form over `Fraction` — exact, and the matrices here are tiny."""
    work = [[Fraction(value) for value in row] for row in rows]
    pivots: list[int] = []
    for column in range(columns):
        pivot = next((i for i in range(len(pivots), len(work)) if work[i][column] != 0), None)
        if pivot is None:
            continue
        rank = len(pivots)
        work[rank], work[pivot] = work[pivot], work[rank]
        lead = work[rank][column]
        work[rank] = [value / lead for value in work[rank]]
        for i in range(len(work)):
            if i != rank and work[i][column] != 0:
                factor = work[i][column]
                work[i] = [a - factor * b for a, b in zip(work[i], work[rank])]
        pivots.append(column)
    return work, pivots


def _rank(rows: tuple[tuple[int, ...], ...]) -> int:
    """The rank of an integer matrix."""
    return len(_rref(rows, len(rows[0]) if rows else 0)[1])


def _in_span(rows: tuple[tuple[int, ...], ...], vector: tuple[int, ...]) -> bool:
    """True when `vector` lies in the row space of `rows`."""
    if not any(vector):
        return True
    if not rows:
        return False
    return _rank(rows) == _rank(tuple(rows) + (vector,))


def kernel_basis(matrix: tuple[tuple[int, ...], ...], columns: int) -> tuple[tuple[int, ...], ...]:
    """A basis of `ker matrix` — one integer generator per free column, ascending.

    `03-lld-M3-checker.md` §3.2's `KERNEL_BASIS`, computed here so `CLASSIFY` can ask whether an
    operand's reuse space is contained in `ker Sπ_u` (LLD §3.2 line 15).
    """
    work, pivots = _rref(matrix, columns)
    basis = []
    for free in (c for c in range(columns) if c not in pivots):
        vector = [Fraction(0)] * columns
        vector[free] = Fraction(1)
        for i, column in enumerate(pivots):
            vector[column] = -work[i][free]
        scale = lcm(*(value.denominator for value in vector))
        basis.append(tuple(int(value * scale) for value in vector))
    return tuple(basis)


def _column(matrix: tuple[tuple[int, ...], ...], index: int) -> tuple[int, ...]:
    """Column `index` of `matrix`."""
    return tuple(row[index] for row in matrix)


# --------------------------------------------------------------------------------------------
# Model lookups
# --------------------------------------------------------------------------------------------


def _param(mapping: LegalMapping, operand: str) -> Param:
    for param in mapping.kernel.params:
        if param.name == operand:
            return param
    raise _internal(f"operand {operand!r} is not a kernel parameter", operand=operand)


def _axis(mapping: LegalMapping, name: str) -> Axis | None:
    """The post-tiling `Axis` called `name`, or `None`."""
    return next((axis for axis in mapping.axes if axis.name == name), None)


def _kernel_axis(mapping: LegalMapping, name: str) -> Axis:
    axis = next((a for a in mapping.kernel.axes if a.name == name), None)
    if axis is None:
        raise _internal(f"axis {name!r} is not a kernel axis", axis=name)
    return axis


def _root(mapping: LegalMapping, name: str) -> str:
    """`Axis.parent` when `name` is a tile handle, `name` otherwise (LLD §3.2 line 11)."""
    axis = _axis(mapping, name)
    return axis.parent if axis is not None and axis.parent is not None else name


def _ucol(mapping: LegalMapping, axis: str) -> int:
    """The column of `axis` in the untiled frame `UCoord` (`kernel.axes` order)."""
    for index, candidate in enumerate(mapping.kernel.axes):
        if candidate.name == axis:
            return index
    raise _internal(f"axis {axis!r} has no column in UCoord", axis=axis)


def _is_outer_tile(axis: Axis) -> bool:
    """True for the outer handle `f"{parent}0"` of a tiled axis (`03-lld-M2-schedule.md` §3.2)."""
    return axis.parent is not None and axis.name == f"{axis.parent}0"


def _tile_factor(mapping: LegalMapping, axis: str) -> int | None:
    """The inner extent of a tiled kernel axis, or `None` when the axis is not tiled."""
    inner = _axis(mapping, f"{axis}1")
    return None if inner is None else inner.extent


def _tile_extent(mapping: LegalMapping, axis: str) -> int:
    """The per-PE extent along kernel axis `axis`: its tile factor, or its whole extent."""
    factor = _tile_factor(mapping, axis)
    if factor is not None:
        return factor
    extent = _kernel_axis(mapping, axis).extent
    if extent is None:
        raise NotImplementedError(f"axis {axis!r} has no constant extent, so its tile shape is "
                                  f"not a compile-time constant; {_LATER}")
    return extent


def _access_matrix(mapping: LegalMapping, operand: str) -> tuple[tuple[int, ...], ...]:
    """`M_a` over `UCoord`, checked to be the same for every access to the operand."""
    seen = [access
            for statement in mapping.kernel.statements
            for access in (statement.target,) + statement.reads
            if access.operand == operand]
    if not seen:
        raise _internal(f"operand {operand!r} is never accessed", operand=operand)
    if any(access.matrix != seen[0].matrix for access in seen):
        raise NotImplementedError(
            f"operand {operand!r} is accessed through more than one linear map, so it has no "
            f"single reuse space; {_LATER}")
    return seen[0].matrix


def _sigma_axes(mapping: LegalMapping) -> tuple[Axis, ...]:
    """The post-tiling axes in σ order, outermost first (LLD §3.9 line 2).

    A σ row that selects one axis names it; a skewed row (W3's `i + j0`) names none and is
    skipped, because "the outer tile axis that moves the tile" is only defined for a row that
    is one axis.
    """
    out = []
    for row in mapping.sigma:
        support = [index for index, value in enumerate(row) if value]
        if len(support) == 1 and row[support[0]] == 1:
            out.append(mapping.axes[support[0]])
    return tuple(out)


# --------------------------------------------------------------------------------------------
# §4 Preconditions — M3's postconditions, asserted rather than re-derived
# --------------------------------------------------------------------------------------------


def preconditions(mapping: LegalMapping) -> None:
    """Assert LLD §4's preconditions. A failure is M3 and M4 disagreeing, not a user error."""
    schedule = mapping.schedule
    grid = schedule.grid or ()
    if len(mapping.physical_herd) != len(grid) or len(mapping.repeats) != len(grid):
        raise _internal(
            f"physical_herd {mapping.physical_herd} and repeats {mapping.repeats} must have the "
            f"rank of grid {grid} (design/06-interfaces.md §4.1)",
            physical_herd=list(mapping.physical_herd), grid=list(grid))
    for extent, physical, repeat in zip(grid, mapping.physical_herd, mapping.repeats):
        if physical < 1 or extent % physical or extent // physical != repeat:
            raise _internal(
                f"physical_herd {mapping.physical_herd} does not divide grid {grid} with "
                f"repeats {mapping.repeats} (design/06-interfaces.md §4.1)",
                physical_herd=list(mapping.physical_herd), grid=list(grid),
                repeats=list(mapping.repeats))
    cap = _PHYSICAL_CAP.get((schedule.target, len(grid)))
    if cap is not None and any(p > c for p, c in zip(mapping.physical_herd, cap)):
        raise _internal(
            f"physical_herd {mapping.physical_herd} exceeds the {schedule.target} cap {cap} "
            f"(python/air/api/_trace.py:88-91, design/07-environment.md §5)",
            physical_herd=list(mapping.physical_herd), cap=list(cap), target=schedule.target)
    if len(mapping.pi) != len(grid) or not 1 <= len(grid) <= 2:
        raise _internal(
            f"pi has {len(mapping.pi)} row(s) for a rank-{len(grid)} grid; the rank must be 1 "
            f"or 2 and must match (design/06-interfaces.md §4.1)",
            pi_rows=len(mapping.pi), grid=list(grid))
    if mapping.l1_bytes > L1_BUDGET:
        raise _internal(
            f"l1_bytes {mapping.l1_bytes} exceeds the {L1_BUDGET} byte budget; M3 raises "
            f"L1-CAPACITY before M4 sees it (design/06-interfaces.md §4.1)",
            l1_bytes=mapping.l1_bytes, l1_budget=L1_BUDGET)
    if mapping.r_space:
        if _rank(mapping.r_space) != 1:
            raise _internal(
                f"r_space has rank {_rank(mapping.r_space)}; a cascade realisation needs rank 1 "
                f"and M3 raises CASCADE-RANK otherwise (design/06-interfaces.md §4.1)",
                r_space=[list(row) for row in mapping.r_space])
        if not schedule.reductions and (mapping.kernel.reduction is None
                                        or mapping.kernel.reduction.op is None):
            raise _internal(
                "r_space is non-empty with no associative/commutative operator declared; M3 "
                "raises RSPACE-NO-AC-OP before M4 sees it (design/06-interfaces.md §4.1)",
                r_space=[list(row) for row in mapping.r_space])
    declared = dict(mapping.halo_footprint)
    for window in schedule.windows:
        derived = declared.get(window.operand)
        if derived is not None and any(h < d for h, d in zip(window.halo, derived)):
            raise _internal(
                f"the declared halo {window.halo} of {window.operand!r} is smaller than the "
                f"derived footprint {derived}; M3 raises HALO-TOO-SMALL "
                f"(design/06-interfaces.md §4.1)",
                operand=window.operand, declared=list(window.halo), derived=list(derived))


# --------------------------------------------------------------------------------------------
# §3.3 step 1 — RESOLVE_HERD
# --------------------------------------------------------------------------------------------


def resolve_herd(mapping: LegalMapping) -> HerdPlan:
    """The herd: the logical grid, M3's physical shape, never pinned (LLD §3.3 step 1)."""
    grid = mapping.schedule.grid or ()
    return HerdPlan(name=f"{mapping.kernel.name}_herd", grid=tuple(grid),
                    shape=mapping.physical_herd, at=None,
                    coords=HERD_COORDS[:len(grid)])


# --------------------------------------------------------------------------------------------
# §3.3 — TENSOR_PLAN
# --------------------------------------------------------------------------------------------


def tensor_plan(mapping: LegalMapping) -> tuple[BufferPlan, ...]:
    """The L3 interface: every read-only param, then every written one (LLD §3.3 note 6).

    `air.api`'s `_check_interface` raises "output tensors must be declared after all input
    tensors" (`_compile.py:226-240`) otherwise, so the order is an upstream law rather than a
    convention. Shapes are resolved through `KernelModel.bindings` (`06-interfaces.md` §2.7).
    """
    bindings = dict(mapping.kernel.bindings)
    params = mapping.kernel.params
    out = []
    for param in [p for p in params if not p.is_written] + [p for p in params if p.is_written]:
        shape = tuple(bindings[entry] if isinstance(entry, str) else entry
                      for entry in param.shape)
        out.append(BufferPlan(name=param.name, operand=param.name, level="L3", scope="tensor",
                              shape=shape, dtype=param.dtype,
                              bytes=prod(shape) * param.dtype.sizeof, loop_depth=0,
                              ping_pong_candidate=False))
    return tuple(out)


# --------------------------------------------------------------------------------------------
# §3.2 — CLASSIFY and MULTICAST_GEOMETRY
# --------------------------------------------------------------------------------------------


def multicast_geometry(grid: tuple[int, ...], axis: int) -> tuple[tuple[int, ...],
                                                                  tuple[int, ...]]:
    """`(size, broadcast_shape)` for an operand constant along PE axis `axis` (LLD §3.2).

    `size` is the grid with extent 1 on the broadcast axis and `broadcast_shape` is the whole
    grid, so `broadcast_shape[d] % size[d] == 0` holds by construction — the rule
    `_channel.py:137-145` enforces. The put indexes the herd coordinate on every axis except
    `axis` and 0 on it; the get indexes the full herd coordinate.
    """
    size = tuple(1 if d == axis else extent for d, extent in enumerate(grid))
    return size, tuple(grid)


def declared_operands(mapping: LegalMapping) -> frozenset[str]:
    """The operands whose delivery a clause names (LLD §3.2, architect ruling on **B-P17**).

    `stationary(a)`, `stream(a, ...)`, `forward(a, ...)` — sugar for `stream` — and
    `exchange(a, ...)` each name the delivery of `a`. `declared` is a fact about the schedule,
    not about whether the derivation would have agreed, so an operand a clause names is
    `declared` whichever row `CLASSIFY` derives for it.
    """
    schedule = mapping.schedule
    return (frozenset(schedule.stationary)
            | {clause.operand for clause in schedule.streams}
            | {clause.operand for clause in schedule.exchanges})


def classify(mapping: LegalMapping) -> tuple[tuple[str, str, str | None, bool], ...]:
    """The reuse trichotomy, one row per operand, computed in `UCoord` (LLD §3.2, FR-M1).

    A row is `(operand, Delivery, along PE axis, declared)`, with `declared` exactly
    `declared_operands` above.
    """
    kernel = mapping.kernel
    schedule = mapping.schedule
    columns = len(kernel.axes)
    target = kernel.reduction.target if kernel.reduction is not None else None
    named = declared_operands(mapping)
    rows: list[tuple[str, str, str | None, bool]] = []
    for param in sorted(kernel.params, key=lambda p: p.name):
        matrix = _access_matrix(mapping, param.name)
        declared = param.name in named
        if param.name == target and mapping.r_space:
            rows.append((param.name, "CASCADE", PE_AXIS_NAME[_cascade_axis(mapping)], declared))
            continue
        broadcast = [d for d in range(len(schedule.place))
                     if not any(_column(matrix, _ucol(mapping, _root(mapping, schedule.place[d]))))]
        if broadcast:
            rows.append((param.name, "MULTICAST", PE_AXIS_NAME[broadcast[0]], declared))
        elif all(_in_span(mapping.ker_pi_u, vector)
                 for vector in kernel_basis(matrix, columns)):
            rows.append((param.name, "STATIONARY", None, declared))
        else:
            rows.append((param.name, "FORWARD", PE_AXIS_NAME[_stream_axis(mapping, matrix)],
                         declared))
    # FR-M3: a stream()/forward() clause also *replaces* the derived row. An exchange() clause
    # does not: `Delivery` has no halo member, and §6.3 keeps W2's derived `U: STATIONARY`
    # beside its declared halo sentence (design/PROGRESS-B.md, phase P2).
    override = {clause.operand: clause for clause in schedule.streams}
    return tuple((row[0], _PATTERN[override[row[0]].pattern], override[row[0]].along, True)
                 if row[0] in override else row for row in rows)


def _cascade_axis(mapping: LegalMapping) -> int:
    """The PE dim carrying `r_space` (LLD §3.6.3 line 2)."""
    support = {index for row in mapping.r_space for index, value in enumerate(row) if value}
    for d, placed in enumerate(mapping.schedule.place):
        if _ucol(mapping, _root(mapping, placed)) in support:
            return d
    raise _internal("r_space is not carried by any placed axis; M3 raises CASCADE-RANK first "
                    "(design/06-interfaces.md §4.1)",
                    r_space=[list(row) for row in mapping.r_space],
                    place=list(mapping.schedule.place))


def _stream_axis(mapping: LegalMapping, matrix: tuple[tuple[int, ...], ...]) -> int:
    """The PE dim of the non-reuse spatial direction (LLD §3.2 line 18)."""
    for d, placed in enumerate(mapping.schedule.place):
        if any(_column(matrix, _ucol(mapping, _root(mapping, placed)))):
            return d
    raise _internal("no placed axis indexes this operand, so it cannot stream",
                    place=list(mapping.schedule.place))


# --------------------------------------------------------------------------------------------
# §3.3 — BUFFER_PLAN, TILE_SHAPE, DEPTH, PP
# --------------------------------------------------------------------------------------------


def tile_shape(mapping: LegalMapping, operand: str) -> tuple[int, ...]:
    """The per-PE slab of `operand`: the image of the PE's subdomain under `M_a` (§3.3 note 1).

    For the affine rectangular domains the grammar admits this is a product of tile extents, one
    per array dim — `A → [TM, TK]`, `B → [TK, TN]`, `C → [TM, TN]` — computed from the
    `AccessMap` and the tile factors, never hard-coded per workload.
    """
    matrix = _access_matrix(mapping, operand)
    out = []
    for index, row in enumerate(matrix):
        support = [mapping.kernel.axes[j].name for j, value in enumerate(row) if value]
        if not support:
            out.append(1)
            continue
        if len(support) > 1 or any(abs(value) != 1 for value in row if value):
            raise NotImplementedError(
                f"array dim {index} of {operand!r} is indexed by {support} with coefficients "
                f"{[v for v in row if v]}; only a single unit-coefficient axis per dim has a "
                f"rectangular tile shape; {_LATER}")
        out.append(_tile_extent(mapping, support[0]))
    return tuple(out)


def moved_axes(mapping: LegalMapping, operand: str) -> tuple[Axis, ...]:
    """The temporal outer tile axes that move `operand`'s L1 tile (LLD §3.9 lines 2-4).

    `T` is the outer tile axes in σ order that are not placed — only the outer axis of a
    `tile(ax.x, T)` can move a tile, since an inner axis and an untiled axis both run *inside*
    the tile by construction of `TILE_SHAPE`, and a placed axis is spatial, not temporal.
    """
    matrix = _access_matrix(mapping, operand)
    return tuple(axis for axis in temporal_axes(mapping)
                 if any(_column(matrix, _ucol(mapping, _root(mapping, axis.name)))))


def temporal_axes(mapping: LegalMapping) -> tuple[Axis, ...]:
    """`T` of LLD §3.9 line 3: the outer tile axes in σ order that are not placed."""
    columns = len(mapping.axes)
    out = []
    for axis in _sigma_axes(mapping):
        if not _is_outer_tile(axis):
            continue
        unit = tuple(1 if a.name == axis.name else 0 for a in mapping.axes)
        if len(unit) == columns and not _in_span(mapping.pi, unit):
            out.append(axis)
    return tuple(out)


def residency(mapping: LegalMapping, operand: str) -> str:
    """How long `operand`'s L1 tile stays put (LLD §3.9 lines 1-8, FR-M11, RULING 9)."""
    temporal = temporal_axes(mapping)
    moved = moved_axes(mapping, operand)
    if not moved:
        return RESIDENT
    first = moved[0]
    after = temporal[temporal.index(first) + 1:]
    inner = [axis.name for axis in after if axis not in moved]
    prefix = f"resident across {', '.join(inner)}, " if inner else ""
    return f"{prefix}re-fetched per {first.name}"


def buffer_name(mapping: LegalMapping, operand: str) -> str:
    """The L1 staging buffer's name (LLD §3.1's table): `acc` for the accumulator, else lower."""
    return "acc" if _is_accumulator_operand(mapping, operand) else operand.lower()


def _is_accumulator_operand(mapping: LegalMapping, operand: str) -> bool:
    reduction = mapping.kernel.reduction
    return reduction is not None and reduction.target == operand


def is_accumulator(mapping: LegalMapping, buffer: BufferPlan) -> bool:
    """True for the buffer staging the reduction target — the one the plan zeroes and drains."""
    return buffer.operand is not None and _is_accumulator_operand(mapping, buffer.operand)


def depth(mapping: LegalMapping, operand: str) -> int:
    """`DEPTH` (LLD §3.3): 1 when the buffer is re-filled per trip of the streaming loop.

    §3.3 line 28 writes the test as "`MULTICAST` or `FORWARD` **and** re-fetched per trip". The
    delivery half is dropped and the residency half kept, because §6.2's flip allocates its
    `STATIONARY` but re-fetched `a` inside the `i0` loop at `loop_depth 1`: what decides the
    depth is whether the buffer's *contents* change with the streaming loop, not how they are
    delivered. A written accumulator is allocated once and drained, so it is always 0.
    """
    if _param(mapping, operand).is_written:
        return 0
    return 1 if moved_axes(mapping, operand) else 0


def ping_pong(mapping: LegalMapping, operand: str) -> bool:
    """`PP` (LLD §3.3): `double_buffer` asked for it and the loop shape can satisfy it.

    VF §E.2's remaining conditions — the first site touching the buffer is a `get`, exactly one
    get per iteration, constant enclosing bounds — hold **by construction** of the fill protocol
    (§3.6/§6.1: the alloc is the direct child of the streaming loop and its `get` is the first
    site after the allocs), and P3's `self_check` re-derives them from the finished plan.
    """
    return operand in mapping.schedule.double_buffer and depth(mapping, operand) >= 1


def buffer_plan(mapping: LegalMapping,
                delivery: tuple[tuple[str, str, str | None, bool], ...],
                herd: HerdPlan) -> tuple[BufferPlan, ...]:
    """Every non-L3 buffer, in **allocation order** (`06-interfaces.md` §5.6, LLD §3.3).

    Allocation order is the order the herd body allocates them: the resident buffers first
    (`loop_depth 0`), then the streamed ones (`loop_depth 1`), each group in operand order.
    `plan()` asserts that the body it then builds allocates them in exactly this order.
    """
    residency_levels = dict(mapping.schedule.residency)
    out = []
    for operand, _kind, _along, _declared in delivery:
        level = residency_levels.get(operand, "L1")
        if level not in ("L1", "L3"):
            raise _fail(
                f"no segment-private staging protocol is synthesised in this cut, so "
                f"{operand!r} cannot reside in {level}",
                f'drop the clause or write reside({operand}="L1"): an L2 staging protocol is '
                f"future work (design/03-lld-M4-mapping.md §3.3 note 5)",
                clause=f'reside({operand}="{level}")', operand=operand, level=level)
        if level != "L1":
            continue                       # L3 is TENSOR_PLAN's, never BUFFER_PLAN's
        param = _param(mapping, operand)
        shape = tile_shape(mapping, operand)
        out.append(BufferPlan(name=buffer_name(mapping, operand), operand=operand, level="L1",
                              scope="herd.private", shape=shape, dtype=param.dtype,
                              bytes=prod(shape) * param.dtype.sizeof,
                              loop_depth=depth(mapping, operand),
                              ping_pong_candidate=ping_pong(mapping, operand)))
    return tuple(sorted(out, key=lambda b: (b.loop_depth, b.operand or b.name)))


# --------------------------------------------------------------------------------------------
# §3.4 — L3_REGION
# --------------------------------------------------------------------------------------------


def l3_region(mapping: LegalMapping, operand: str, origins: dict[str, Expr]) -> Region:
    """The L3 end of a transfer: offsets from the access map, tile sizes, row-major strides.

    `origins` gives, per **untiled** kernel axis, the element coordinate at which this slab
    starts along that axis — the PE coordinate times the tile factor for a placed axis, the
    streaming loop's induction variable for a temporal one (LLD §3.4 lines 2-3).
    """
    bindings = dict(mapping.kernel.bindings)
    matrix = _access_matrix(mapping, operand)
    access = next(a for statement in mapping.kernel.statements
                  for a in (statement.target,) + statement.reads if a.operand == operand)
    tensor = next(t for t in tensor_plan(mapping) if t.name == operand)
    offsets = []
    for index, row in enumerate(matrix):
        terms = [_scale(origins[mapping.kernel.axes[j].name], value)
                 for j, value in enumerate(row) if value]
        offsets.append(_add(_resolve(access.offsets[index], bindings), *terms))
    return Region(offsets=tuple(offsets), sizes=tile_shape(mapping, operand),
                  strides=_row_major(tensor.shape))


def _row_major(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The row-major strides of `shape` (LLD §3.4 lines 5-6)."""
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return tuple(strides)


# --------------------------------------------------------------------------------------------
# Loop names, and the one place each is derived (invariant I-3)
# --------------------------------------------------------------------------------------------


def bundle_name(mapping: LegalMapping, pe_dim: int) -> str:
    """The segment-scope bundle loop for PE dim `pe_dim`: `pi_bundle`, `pj_bundle`, `pk_bundle`.

    `06-interfaces.md` §5.5 at `CONTRACT_VERSION = 5`: a bundle-index loop is `p<root>_bundle`,
    where `root` is the untiled parent of the placed axis — the rule that closed **B-P18**.
    """
    return f"p{_root(mapping, mapping.schedule.place[pe_dim])}_bundle"


def drain_name(mapping: LegalMapping, pe_dim: int) -> str:
    """The segment-scope drain loop for PE dim `pe_dim`: `i_drain`, `j_drain` (LLD §6.1).

    §5.5's rule is `<axis>_drain`, naming the axis whose trips the loop enumerates. This cut's
    drain walks the PE grid, so the axis is the root of the placed axis; the flip's drain walks
    the temporal `i0` and is `i0_drain` (P6).
    """
    return f"{_root(mapping, mapping.schedule.place[pe_dim])}_drain"


def streaming_axis(mapping: LegalMapping, operand: str) -> str | None:
    """The outer tile axis the operand is re-fetched per, or `None` when it is resident."""
    moved = moved_axes(mapping, operand)
    return moved[0].name if moved else None


def compute_names(mapping: LegalMapping) -> dict[str, str]:
    """Kernel axis → the per-PE loop name that realises it (`06-interfaces.md` §5.5 at v5).

    A compute or zeroing nest is named by the **post-tiling axis it realises**: the inner tile
    handle `x1` where the axis is tiled, the axis itself where it is not — W1's `i1`, `j1`,
    `k1`, the flip's `i1`, `j`, `k1`. The zeroing nest realises the same axes as the compute
    nest, so it reuses the names; a `LoopPlan.axis` may recur in different bodies and M5 rebinds
    by name in order (the rule that closed **B-P18**).
    """
    return {axis: (f"{axis}1" if _tile_factor(mapping, axis) is not None else axis)
            for axis in _statement(mapping).axes}


def _statement(mapping: LegalMapping):
    statements = mapping.kernel.statements
    if len(statements) != 1:
        raise NotImplementedError(
            f"the kernel has {len(statements)} statements; this cut fuses none of them; "
            f"{_LATER}")
    return statements[0]


# --------------------------------------------------------------------------------------------
# §3.5 — CHANNEL_PLAN and LOOP_KIND
# --------------------------------------------------------------------------------------------


def loop_kind(axis: str, *, bundle_index: bool) -> str:
    """`LOOP_KIND` (LLD §3.5): a loop whose value is a channel bundle index is unrolled.

    Line 2 is `ChannelPutOp::verify` (`AIRDialect.cpp:3586-3593`) — a bundle index may not be an
    `scf.for` induction variable — and every other loop takes line 4's safe default,
    `air.sequential`.
    """
    return "unrolled" if bundle_index else "sequential"


def loop_plan(mapping: LegalMapping, buffers: tuple[BufferPlan, ...]) -> tuple[tuple[str, str],
                                                                              ...]:
    """`LOOP_KIND` applied to every loop axis the plan realises: `(axis, kind)`, sorted by axis.

    The bundle and drain loops carry a channel index and are unrolled; the streaming, compute
    and zeroing loops are `air.sequential`. `PROTOCOL` reads the kind from here and never
    decides it itself.
    """
    kinds: dict[str, str] = {}
    for pe_dim in range(len(mapping.schedule.place)):
        kinds[bundle_name(mapping, pe_dim)] = loop_kind(bundle_name(mapping, pe_dim),
                                                        bundle_index=True)
        kinds[drain_name(mapping, pe_dim)] = loop_kind(drain_name(mapping, pe_dim),
                                                       bundle_index=True)
    for buffer in buffers:
        axis = streaming_axis(mapping, buffer.operand) if buffer.operand else None
        if axis is not None:
            kinds[axis] = loop_kind(axis, bundle_index=False)
    for name in compute_names(mapping).values():
        kinds[name] = loop_kind(name, bundle_index=False)
    return tuple(sorted(kinds.items()))


def _kind(loops: tuple[tuple[str, str], ...], axis: str) -> str:
    for name, kind in loops:
        if name == axis:
            return kind
    raise _internal(f"loop axis {axis!r} has no kind in the loop plan", axis=axis)


def _fills_and_drains(mapping: LegalMapping,
                      delivery: tuple[tuple[str, str, str | None, bool], ...]
                      ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Which operands are filled from L3 and which are drained to it (LLD §3.5 lines 7-10).

    A written operand is drained; a read-only operand is filled. An accumulator is *not* filled:
    it is initialised by the zeroing `LoopPlan` of §6.1's herd body, not by a transfer.
    """
    fills, drains = [], []
    for operand, _kind_, _along, _declared in delivery:
        (drains if _param(mapping, operand).is_written else fills).append(operand)
    return tuple(fills), tuple(drains)


def channel_plan(mapping: LegalMapping,
                 delivery: tuple[tuple[str, str, str | None, bool], ...],
                 herd: HerdPlan, buffers: tuple[BufferPlan, ...]) -> tuple[ChannelPlan, ...]:
    """Every `air.channel` with its geometry and its ordered sites, sorted by name (FR-M8)."""
    for operand, kind, _along, _declared in delivery:
        if kind not in ("MULTICAST", "STATIONARY"):
            raise NotImplementedError(
                f"operand {operand!r} is delivered {kind}, whose protocol builder "
                f"(design/03-lld-M4-mapping.md §3.6) {_LATER}")
        if kind == "MULTICAST" and _param(mapping, operand).is_written:
            raise NotImplementedError(
                f"operand {operand!r} multicasts and is written, so it needs both a fan-out "
                f"fill and a drain; {_LATER}")
    sites = _sites(mapping, delivery, herd, buffers)
    fills, drains = _fills_and_drains(mapping, delivery)
    out = []
    for operand, kind, along, _declared in delivery:
        param = _param(mapping, operand)
        if kind == "MULTICAST":
            size, broadcast = multicast_geometry(herd.grid, PE_AXIS_NAME.index(along))
        else:
            size, broadcast = herd.grid, None
        if operand in fills:
            name = f"{operand}2L1"
            out.append(ChannelPlan(name=name, size=size, broadcast_shape=broadcast,
                                   channel_type=None, chain_direction=None, dtype=param.dtype,
                                   sites=(sites[(name, "put")], sites[(name, "get")])))
        if operand in drains:
            name = f"{operand}2L3"
            out.append(ChannelPlan(name=name, size=herd.grid, broadcast_shape=None,
                                   channel_type=None, chain_direction=None, dtype=param.dtype,
                                   sites=(sites[(name, "put")], sites[(name, "get")])))
    return tuple(sorted(out, key=lambda c: c.name))


def _site(channel: str, kind: str, order: int, scope: str, **rest: Any) -> ChannelSite:
    """One site, with LLD §3.1's `id` rule; this cut's sites are all plain and token-free."""
    return ChannelSite(id=f"{channel}.{kind}.{order}@{scope}", kind=kind, channel=channel,
                       scope=scope, guard=None, is_async=False, depends_on=(), order=order,
                       **rest)


def _sites(mapping: LegalMapping, delivery: tuple[tuple[str, str, str | None, bool], ...],
           herd: HerdPlan, buffers: tuple[BufferPlan, ...]) -> dict[tuple[str, str], ChannelSite]:
    """Every site of the fill/compute/drain protocol, keyed `(channel, kind)`.

    `ChannelSite.order` is the site's index in its **enclosing body**
    (`06-interfaces.md` §5.2 at `CONTRACT_VERSION = 4`), so the orders are counted here against
    the same body layout `PROTOCOL` then builds; `plan()` asserts the two agree.
    """
    fills, drains = _fills_and_drains(mapping, delivery)
    by_operand = {buffer.operand: buffer for buffer in buffers}
    streamed = [b for b in buffers if b.loop_depth >= 1]
    out: dict[tuple[str, str], ChannelSite] = {}
    herd_index = 0
    for buffer in buffers:
        if buffer.loop_depth >= 1:
            continue
        herd_index += 1                                       # the alloc
        if buffer.operand in fills:
            out[(f"{buffer.operand}2L1", "get")] = _site(
                f"{buffer.operand}2L1", "get", herd_index, "herd",
                indices=tuple(_var(coord) for coord in herd.coords), buffer=buffer.name,
                region=EMPTY_REGION)
            herd_index += 1
        if is_accumulator(mapping, buffer):
            herd_index += 1                                   # the zeroing nest
    if streamed:
        # the streaming loop's body is: every streamed alloc, then their gets, then the nest
        for position, buffer in enumerate(b for b in streamed if b.operand in fills):
            out[(f"{buffer.operand}2L1", "get")] = _site(
                f"{buffer.operand}2L1", "get", len(streamed) + position, "herd",
                indices=tuple(_var(coord) for coord in herd.coords), buffer=buffer.name,
                region=EMPTY_REGION)
        herd_index += 1                                       # the streaming loop
    for operand in drains:
        out[(f"{operand}2L3", "put")] = _site(
            f"{operand}2L3", "put", herd_index, "herd",
            indices=tuple(_var(coord) for coord in herd.coords),
            buffer=by_operand[operand].name, region=EMPTY_REGION)
        herd_index += 1
    for operand, kind, along, _declared in delivery:
        if operand in fills:
            size = (multicast_geometry(herd.grid, PE_AXIS_NAME.index(along))[0]
                    if kind == "MULTICAST" else herd.grid)
            out[(f"{operand}2L1", "put")] = _site(
                f"{operand}2L1", "put", 0, "segment",
                indices=tuple(_var(bundle_name(mapping, d)) if extent > 1 else ZERO
                              for d, extent in enumerate(size)),
                buffer=operand,
                region=l3_region(mapping, operand, _fill_origins(mapping, operand)))
        if operand in drains:
            out[(f"{operand}2L3", "get")] = _site(
                f"{operand}2L3", "get", 0, "segment",
                indices=tuple(_var(drain_name(mapping, d)) if extent > 1 else ZERO
                              for d, extent in enumerate(herd.grid)),
                buffer=operand,
                region=l3_region(mapping, operand, _drain_origins(mapping)))
    return out


def _origins(mapping: LegalMapping, pe_var) -> dict[str, Expr]:
    """Per untiled kernel axis, the element coordinate this slab starts at (LLD §3.4).

    `pe_var(pe_dim)` names the segment-scope loop standing in for PE coordinate `pe_dim`.
    """
    bindings = dict(mapping.kernel.bindings)
    placed = {_root(mapping, name): d for d, name in enumerate(mapping.schedule.place)}
    out: dict[str, Expr] = {}
    for axis in mapping.kernel.axes:
        factor = _tile_factor(mapping, axis.name)
        if axis.name in placed and factor is not None:
            out[axis.name] = _var(pe_var(placed[axis.name]), factor)
        elif axis.name in placed:
            out[axis.name] = _var(pe_var(placed[axis.name]))
        else:
            out[axis.name] = _resolve(axis.lo, bindings)
    return out


def _fill_origins(mapping: LegalMapping, operand: str) -> dict[str, Expr]:
    """`_origins` with the streaming axis of `operand` bound to its own loop variable.

    The streaming loop runs in element units (`air.sequential(0, 64, 16)` over `k0`), so the
    origin along that axis is the loop variable itself with coefficient 1 — §3.4's `kk`.
    """
    origins = _origins(mapping, lambda d: bundle_name(mapping, d))
    axis = streaming_axis(mapping, operand)
    if axis is not None:
        origins[_root(mapping, axis)] = _var(axis)
    return origins


def _drain_origins(mapping: LegalMapping) -> dict[str, Expr]:
    """`_origins` over the drain loops, which index the PE grid directly."""
    return _origins(mapping, lambda d: drain_name(mapping, d))


# --------------------------------------------------------------------------------------------
# §3.6 — PROTOCOL
# --------------------------------------------------------------------------------------------


def protocol(mapping: LegalMapping, delivery: tuple[tuple[str, str, str | None, bool], ...],
             herd: HerdPlan, buffers: tuple[BufferPlan, ...],
             channels: tuple[ChannelPlan, ...],
             loops: tuple[tuple[str, str], ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """`(segment_body, herd_body)` — the dispatch of LLD §3.1 line 7 / §3.6.

    The dispatch is on which clauses are present, not on which workload it is: `exchange`
    selects §3.6.1, `forward` §3.6.2 and a non-empty `r_space` §3.6.3. This cut builds none of
    those three, so each raises; what it does build is §6.1's fill / compute / drain shape.
    """
    if mapping.schedule.exchanges:
        raise NotImplementedError(
            f"exchange({mapping.schedule.exchanges[0].operand}, ...) selects the halo protocol "
            f"of design/03-lld-M4-mapping.md §3.6.1, which {_LATER}")
    if any(row[1] == "FORWARD" for row in delivery):
        raise NotImplementedError(
            f"a forward delivery selects the wavefront protocol of "
            f"design/03-lld-M4-mapping.md §3.6.2, which {_LATER}")
    if mapping.r_space:
        raise NotImplementedError(
            f"a non-empty r_space selects the cascade protocol of "
            f"design/03-lld-M4-mapping.md §3.6.3, which {_LATER}")
    sites = {(channel.name, site.kind): site for channel in channels for site in channel.sites}
    sizes = {channel.name: channel.size for channel in channels}
    fills, drains = _fills_and_drains(mapping, delivery)
    by_operand = {buffer.operand: buffer for buffer in buffers}
    names = compute_names(mapping)

    segment_body = tuple(
        _fill_loop(mapping, operand, sizes[f"{operand}2L1"], loops,
                   sites[(f"{operand}2L1", "put")]) for operand in fills)
    segment_body += (herd,)
    segment_body += tuple(
        _drain_loop(mapping, herd.grid, loops, sites[(f"{operand}2L3", "get")])
        for operand in drains)

    streamed = [b for b in buffers if b.loop_depth >= 1]
    axis = streaming_axis(mapping, streamed[0].operand) if streamed else None
    if any(streaming_axis(mapping, b.operand) != axis for b in streamed):
        raise NotImplementedError(
            f"the streamed buffers {[b.name for b in streamed]} are re-fetched per "
            f"different axes, so the herd body needs more than one streaming loop; {_LATER}")
    elements, slab = compute_frame(mapping, herd, axis, names)
    target = _statement(mapping).target
    _, stored = _in_l1(mapping, by_operand, elements, slab, target.operand,
                       kernel_subscripts(mapping, target))

    herd_body: list[Any] = []
    for buffer in buffers:
        if buffer.loop_depth >= 1:
            continue
        herd_body.append(buffer)
        if buffer.operand in fills:
            herd_body.append(sites[(f"{buffer.operand}2L1", "get")])
        if is_accumulator(mapping, buffer):
            herd_body.append(_zero_nest(mapping, buffer, names, loops, 0, stored))
    nest = _compute_nest(mapping, by_operand, names, loops, 1 if streamed else 0,
                         elements, slab, stored)
    if streamed:
        body = tuple(streamed)
        body += tuple(sites[(f"{b.operand}2L1", "get")] for b in streamed if b.operand in fills)
        herd_body.append(_tile_loop(mapping, axis, loops, 0, body + (nest,)))
    else:
        herd_body.append(nest)
    for operand in drains:
        herd_body.append(sites[(f"{operand}2L3", "put")])
    _check_orders(segment_body)
    _check_orders(tuple(herd_body))
    return segment_body, tuple(herd_body)


def _check_orders(nodes: tuple[Any, ...]) -> None:
    """Every `ChannelSite.order` is its index in the body holding it (`06-interfaces.md` §5.2)."""
    for index, node in enumerate(nodes):
        if isinstance(node, ChannelSite) and node.order != index:
            raise _internal(
                f"site {node.id!r} carries order {node.order} but sits at index {index} of its "
                f"body (design/06-interfaces.md §5.2)", site=node.id, order=node.order,
                index=index)
        if isinstance(node, LoopPlan):
            _check_orders(node.body)


def _tile_loop(mapping: LegalMapping, axis: str, loops: tuple[tuple[str, str], ...],
               depth_: int, body: tuple[Any, ...]) -> LoopPlan:
    """The streaming loop over an outer tile axis, in **element** units (LLD §6.1 line 1).

    `air.sequential(0, 64, 16)` over `k0`, not `range(4)`: the loop variable *is* the element
    offset the region uses, which is what §3.4's worked `offsets=(pi*32, kk)` shows.
    """
    bindings = dict(mapping.kernel.bindings)
    parent = _root(mapping, axis)
    kernel_axis = _kernel_axis(mapping, parent)
    factor = _tile_factor(mapping, parent)
    if factor is None:
        raise _internal(f"axis {axis!r} is not a tile handle", axis=axis)
    step = _resolve(kernel_axis.step, bindings)
    if not step.is_constant or step.const != 1:
        raise NotImplementedError(
            f"kernel axis {parent!r} has step {step}; a tiled axis with a non-unit step has no "
            f"element-unit streaming loop; {_LATER}")
    return LoopPlan(axis=axis, lo=_resolve(kernel_axis.lo, bindings),
                    hi=_resolve(kernel_axis.hi, bindings), step=_const(factor),
                    kind=_kind(loops, axis), depth=depth_, body=body)


def _bundle_nest(mapping: LegalMapping, size: tuple[int, ...], name,
                 loops: tuple[tuple[str, str], ...], depth_: int,
                 body: tuple[Any, ...]) -> tuple[Any, ...]:
    """Wrap `body` in one unrolled loop per bundle dim of extent > 1, outermost dim first."""
    dims = [d for d, extent in enumerate(size) if extent > 1]
    for position in reversed(range(len(dims))):
        axis = name(dims[position])
        body = (LoopPlan(axis=axis, lo=ZERO, hi=_const(size[dims[position]]), step=ONE,
                         kind=_kind(loops, axis), depth=depth_ + position, body=body),)
    return body


def _fill_loop(mapping: LegalMapping, operand: str, size: tuple[int, ...],
               loops: tuple[tuple[str, str], ...], put: ChannelSite) -> Any:
    """The segment-scope fill of one operand: bundle loops, the streaming loop, the put."""
    body: tuple[Any, ...] = (put,)
    axis = streaming_axis(mapping, operand)
    if axis is not None:
        body = (_tile_loop(mapping, axis, loops,
                           len([e for e in size if e > 1]), body),)
    return _bundle_nest(mapping, size, lambda d: bundle_name(mapping, d), loops, 0, body)[0]


def _drain_loop(mapping: LegalMapping, grid: tuple[int, ...],
                loops: tuple[tuple[str, str], ...], get: ChannelSite) -> Any:
    """The segment-scope drain of one operand: one unrolled loop per PE dim, then the get."""
    return _bundle_nest(mapping, grid, lambda d: drain_name(mapping, d), loops, 0, (get,))[0]


def kernel_subscripts(mapping: LegalMapping, access) -> tuple[Expr, ...]:
    """An `AccessMap` as one `Expr` per array dim, over kernel axis names — `M_a·x + c_a`.

    It is the form `Statement.expr`'s `Load`s already carry (`06-interfaces.md` §2.4 at v5), so
    the write access and the reads go through the same rewrite below.
    """
    return tuple(
        _add(access.offsets[index],
             *[_var(mapping.kernel.axes[j].name, value) for j, value in enumerate(row) if value])
        for index, row in enumerate(access.matrix))


def _substitute(expr: Expr, values: dict[str, Expr]) -> Expr:
    """`expr` with every name `values` covers replaced by its expression."""
    out = _const(expr.const)
    for name, coeff in expr.coeffs:
        value = values.get(name)
        out = _add(out, _scale(value, coeff) if value is not None else _var(name, coeff))
    return out


def l1_subscripts(subscripts: tuple[Expr, ...], elements: dict[str, Expr],
                  origin: tuple[Expr, ...], bindings: dict[str, int]) -> tuple[Expr, ...]:
    """One kernel-level access rewritten PE-relative, into its L1 staging buffer (FR-M8).

    Array dim `d` reads L3 at `subscripts[d]`, an `Expr` over kernel axis names, shape
    parameters and constants. Substituting `elements` — the L3 element coordinate the compute
    nest realises for each kernel axis — gives the index *this* PE's nest touches, and the
    staged slab starts at `origin[d]` in the same coordinates, so the buffer-local index is the
    difference: the placed and outer-tile contributions cancel exactly. W1's `A[i,k]` becomes
    `a[i1,k1]`; a ghost-padded or column-local staging region shifts the result by its own halo
    instead, which is what W2's `src[i1+1, j]` and W3's `p[j1]` will be (P4/P5).
    """
    if len(subscripts) != len(origin):
        raise NotImplementedError(
            f"the access has rank {len(subscripts)} and its staged slab rank {len(origin)}; a "
            f"buffer that stages fewer dims than the access indexes (a plane of a rank-3 "
            f"tensor, a row of a rank-2 one) needs the protocol's own staging map; {_LATER}")
    return tuple(_add(_substitute(_resolve(subscript, bindings), elements),
                      _scale(_resolve(origin[index], bindings), -1))
                 for index, subscript in enumerate(subscripts))


def compute_frame(mapping: LegalMapping, herd: HerdPlan, streaming: str | None,
                  names: dict[str, str]) -> tuple[dict[str, Expr], dict[str, Expr]]:
    """`(elements, slab)`, the two halves `l1_subscripts` needs, in herd-scope variables.

    `elements[x]` is the L3 element coordinate the nest realises for kernel axis `x`: the tile
    origin plus the inner loop variable where `x` is tiled, the loop variable itself where it is
    not (an untiled axis's loop runs over the kernel's own range). `slab` is the per-axis origin
    of this PE's staged slab, which `L3_REGION` turns into the per-dim origin of one operand.
    """
    slab = _origins(mapping, lambda d: herd.coords[d])
    if streaming is not None:
        slab[_root(mapping, streaming)] = _var(streaming)
    elements = {axis: (_add(slab[axis], _var(name))
                       if _tile_factor(mapping, axis) is not None else _var(name))
                for axis, name in names.items()}
    return elements, slab


def _in_l1(mapping: LegalMapping, by_operand: dict[str | None, BufferPlan],
           elements: dict[str, Expr], slab: dict[str, Expr], operand: str,
           subscripts: tuple[Expr, ...]) -> tuple[str, tuple[Expr, ...]]:
    """`(buffer name, PE-relative subscripts)` for one kernel-level access of `operand`."""
    buffer = by_operand.get(operand)
    if buffer is None:
        raise NotImplementedError(
            f"the statement reads {operand!r}, which this cut stages into no L1 buffer; its "
            f"protocol builder {_LATER}")
    origin = l3_region(mapping, operand, slab).offsets
    return buffer.name, l1_subscripts(subscripts, elements, origin,
                                      dict(mapping.kernel.bindings))


def _rewrite_loads(node: Any, rewrite) -> Any:
    """Rebuild an `ExprNode` tree with every `Load` replaced by `rewrite(load)` (§5.5)."""
    if isinstance(node, Load):
        return rewrite(node)
    changed = {}
    for field in fields(node):
        value = getattr(node, field.name)
        if isinstance(value, _EXPR_NODES):
            changed[field.name] = _rewrite_loads(value, rewrite)
        elif type(value) is tuple and all(isinstance(item, _EXPR_NODES) for item in value):
            changed[field.name] = tuple(_rewrite_loads(item, rewrite) for item in value)
    return replace(node, **changed) if changed else node


def _zero_nest(mapping: LegalMapping, accumulator: BufferPlan, names: dict[str, str],
               loops: tuple[tuple[str, str], ...], depth_: int,
               subscripts: tuple[Expr, ...]) -> LoopPlan:
    """The accumulator zeroing: an ordinary `LoopPlan` of `StoreNode(Const 0)` (§5.5, §6.1).

    It carries no `statement_index`, because no kernel statement corresponds to it. Its loops
    realise the same post-tiling axes as the compute nest and carry the same names (§5.5 at v5),
    and it writes the same element of the accumulator that the compute nest then accumulates
    into — hence the shared `subscripts`.
    """
    statement = _statement(mapping)
    zero = (Const(value=0.0, text="0.0", dtype=accumulator.dtype)
            if accumulator.dtype in (Dtype.f32, Dtype.f16, Dtype.bf16)
            else Const(value=0, text="0", dtype=accumulator.dtype))
    body: tuple[Any, ...] = (StoreNode(buffer_id=accumulator.name, subscripts=subscripts,
                                       expr=zero),)
    axes = [mapping.kernel.axes[j].name
            for row in statement.target.matrix for j, value in enumerate(row) if value]
    if len(axes) != len(accumulator.shape):
        raise NotImplementedError(
            f"the accumulator {accumulator.name!r} has {len(accumulator.shape)} dim(s) indexed "
            f"by {axes}; a zeroing nest needs exactly one axis per dim; {_LATER}")
    for position in reversed(range(len(axes))):
        axis = names[axes[position]]
        body = (LoopPlan(axis=axis, lo=ZERO, hi=_const(accumulator.shape[position]), step=ONE,
                         kind=_kind(loops, axis), depth=depth_ + position, body=body),)
    return body[0]


def _compute_nest(mapping: LegalMapping, by_operand: dict[str | None, BufferPlan],
                  names: dict[str, str], loops: tuple[tuple[str, str], ...], depth_: int,
                  elements: dict[str, Expr], slab: dict[str, Expr],
                  subscripts: tuple[Expr, ...]) -> LoopPlan:
    """The per-PE compute nest: one `air.sequential` per statement axis, over the tile.

    The store's value is the kernel's own expression tree (`Statement.expr`, §2.4 at v5) with
    every kernel-level `Load` rewritten into its L1 buffer by `l1_subscripts`. M4 rebuilds no
    arithmetic of its own — an accumulate arrives already desugared — which is what lets W2's
    `0.2 × (five-term sum)` and W3's `max`/`Select` reach M5 at all (**B-P19**).
    """
    statement = _statement(mapping)
    target = by_operand[statement.target.operand]
    body: tuple[Any, ...] = (StoreNode(
        buffer_id=target.name, subscripts=subscripts,
        expr=_rewrite_loads(statement.expr, lambda load: Load(
            *_in_l1(mapping, by_operand, elements, slab, load.buffer_id, load.subscripts)))),)
    bindings = dict(mapping.kernel.bindings)
    for position in reversed(range(len(statement.axes))):
        axis = statement.axes[position]
        name = names[axis]
        if _tile_factor(mapping, axis) is not None:
            lo, hi = ZERO, _const(_tile_extent(mapping, axis))
        else:
            kernel_axis = _kernel_axis(mapping, axis)
            lo, hi = _resolve(kernel_axis.lo, bindings), _resolve(kernel_axis.hi, bindings)
        body = (LoopPlan(axis=name, lo=lo, hi=hi,
                         step=_resolve(_kernel_axis(mapping, axis).step, bindings),
                         kind=_kind(loops, name), depth=depth_ + position, body=body),)
    return body[0]


# --------------------------------------------------------------------------------------------
# §3.9 — SUMMARY
# --------------------------------------------------------------------------------------------


def _how(kind: str, along: str | None) -> str:
    return {"STATIONARY": "stationary", "MULTICAST": f"multicast along {along}",
            "FORWARD": f"forward along {along}", "CASCADE": f"cascade along {along}",
            "NONE": "none"}[kind]


def reduction_split(mapping: LegalMapping) -> tuple[str, str]:
    """`R_time` / `R_space` rendered over the **post-tiling** axes (LLD §3.9 line 10).

    `LegalMapping.r_time`/`.r_space` are bases over `UCoord` and stay untiled — they are what
    `CLASSIFY` reads — so printing them bare says `R_time = {}` for the flip, which reads as "no
    local accumulation" and is false. The rendered line splits the whole reduction space over
    the tile handles instead: a post-tiling axis is spatial when it is placed and temporal
    otherwise.
    """
    support = {index for row in tuple(mapping.r_time) + tuple(mapping.r_space)
               for index, value in enumerate(row) if value}
    roots = {mapping.kernel.axes[index].name for index in support}
    time, space = [], []
    for index, axis in enumerate(mapping.axes):
        if _root(mapping, axis.name) not in roots:
            continue
        unit = tuple(1 if i == index else 0 for i in range(len(mapping.axes)))
        (space if _in_span(mapping.pi, unit) else time).append(f"e_{axis.name}")
    return (_span(time), _span(space))


def _span(names: list[str]) -> str:
    return "span{" + ", ".join(names) + "}" if names else "{}"


def summary(mapping: LegalMapping, delivery: tuple[tuple[str, str, str | None, bool], ...],
            herd: HerdPlan, buffers: tuple[BufferPlan, ...],
            channels: tuple[ChannelPlan, ...]) -> MappingSummary:
    """The human-readable summary and the facts behind each line (LLD §3.9, FR-M11, FR-D2)."""
    lines = [f"{operand}: {_how(kind, along)} ({'declared' if declared else 'derived'})"
             for operand, kind, along, declared in delivery]
    pairs = tuple(sorted((operand, residency(mapping, operand))
                         for operand, _kind, _along, _declared in delivery))
    durations = dict(pairs)
    for operand, kind, along, _declared in delivery:
        what = "stationary (spatial)" if kind == "STATIONARY" else _how(kind, along)
        lines.append(f"{operand}: {what}, {durations[operand]}")
    lines.append(f"herd: logical {herd.grid} physical {mapping.physical_herd} "
                 f"repeats {mapping.repeats}")
    time, space = reduction_split(mapping)
    lines.append(f"reduction (tiled axes): R_time = {time}, R_space = {space}")
    for buffer in buffers:
        lines.append(f"  {buffer.name} {buffer.shape} {buffer.dtype.mlir} = {buffer.bytes} B"
                     + ("  x2 (ping-pong)" if buffer.ping_pong_candidate else ""))
    total = l1_total(buffers)
    lines.append(f"L1: {total} of {L1_BUDGET} bytes")
    for channel in channels:
        lines.append(f"  {channel.name} size={channel.size}"
                     + (f" broadcast_shape={channel.broadcast_shape}"
                        if channel.broadcast_shape is not None else "")
                     + (f" type={channel.channel_type}"
                        if channel.channel_type is not None else ""))
    return MappingSummary(lines=tuple(lines), residency=pairs, herd_logical=herd.grid,
                          herd_physical=mapping.physical_herd, repeats=mapping.repeats,
                          reduction_split=(time, space), l1_bytes=total, l1_budget=L1_BUDGET,
                          channels=tuple((c.name, c.size, c.broadcast_shape) for c in channels))


def l1_total(buffers: tuple[BufferPlan, ...]) -> int:
    """`sum(bytes × (2 if ping_pong_candidate else 1))` (§5.6 invariant 5, LLD §3.3 note 4)."""
    return sum(b.bytes * (2 if b.ping_pong_candidate else 1)
               for b in buffers if b.level != "L3")


# --------------------------------------------------------------------------------------------
# §3.1 — PLAN
# --------------------------------------------------------------------------------------------


def plan(mapping: LegalMapping) -> MappingPlan:
    """Derive the reuse trichotomy, buffers, channels, loops. Raises MappingError.

    The ten passes of LLD §3.1 in their fixed order, then `m4.self_check` on the finished plan,
    so a `MappingPlan` that leaves M4 has always been checked. A `KeyError` from an internal
    table, or a traceback through a protocol builder, never escapes: anything that is not
    already a `SpatialError` is re-raised as a `MappingError` carrying the original in
    `details["internal_exception"]` (LLD §5, NFR-4, NFR-7).
    """
    try:
        preconditions(mapping)
        herd = resolve_herd(mapping)
        tensors = tensor_plan(mapping)
        delivery = classify(mapping)
        buffers = buffer_plan(mapping, delivery, herd)
        channels = channel_plan(mapping, delivery, herd, buffers)
        loops = loop_plan(mapping, buffers)
        segment_body, herd_body = protocol(mapping, delivery, herd, buffers, channels, loops)
        _check_l1(mapping, buffers)
        _check_allocation_order(buffers, herd_body)
        out = MappingPlan(mapping=mapping, tensors=tensors, launch_name=mapping.kernel.name,
                          segment_name=f"{mapping.kernel.name}_seg", herd=herd, buffers=buffers,
                          channels=channels, segment_body=segment_body, herd_body=herd_body,
                          delivery=delivery,
                          summary=summary(mapping, delivery, herd, buffers, channels))
        self_check(out)
    except SpatialError:
        raise
    except Exception as exc:                                  # noqa: BLE001 — NFR-7, LLD §5
        raise _fail(
            f"the mapping of {mapping.kernel.name!r} could not be planned: "
            f"{type(exc).__name__}: {exc}",
            _BUG_FIX, internal_exception=f"{type(exc).__name__}: {exc}",
            workload=mapping.kernel.name) from None
    return out


def _check_l1(mapping: LegalMapping, buffers: tuple[BufferPlan, ...]) -> None:
    """The staged buffers must charge exactly what M3 charged (LLD §3.3 note 4, §7)."""
    staged = l1_total(tuple(b for b in buffers if b.operand is not None))
    if staged != mapping.l1_bytes:
        raise _internal(
            f"the plan charges {staged} B for the staged buffers where LegalMapping.l1_bytes "
            f"says {mapping.l1_bytes} B (design/03-lld-M4-mapping.md §3.3 note 4)",
            plan_l1_bytes=staged, mapping_l1_bytes=mapping.l1_bytes)
    total = l1_total(buffers)
    if total > L1_BUDGET:
        raise _internal(
            f"the plan's L1 total {total} B exceeds the {L1_BUDGET} B budget "
            f"(design/06-interfaces.md §5.6 invariant 5)",
            plan_l1_bytes=total, l1_budget=L1_BUDGET)


def _check_allocation_order(buffers: tuple[BufferPlan, ...],
                            herd_body: tuple[Any, ...]) -> None:
    """`MappingPlan.buffers` is the order the herd body allocates them (§5.6)."""
    allocated = [node.name for node in _flatten(herd_body) if isinstance(node, BufferPlan)]
    if allocated != [buffer.name for buffer in buffers]:
        raise _internal(
            f"MappingPlan.buffers is {[b.name for b in buffers]} but the herd body allocates "
            f"{allocated}; §5.6 requires allocation order",
            buffers=[b.name for b in buffers], allocated=allocated)


def _flatten(nodes: tuple[Any, ...]):
    for node in nodes:
        yield node
        if isinstance(node, LoopPlan):
            yield from _flatten(node.body)
