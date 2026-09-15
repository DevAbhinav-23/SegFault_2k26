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
**four protocol builders**: the multicast / stationary fill-compute-drain shape of LLD §6.1;
the wavefront of §3.6.2 — one scalar get per row, one scalar put per row, three homogeneous
channels closed by a segment-scope source and drain; the halo of §3.6.1 — put north, put
south, get both ghosts, every plane drained; and the cascade of §3.6.3 — one `npu_cascade`
bundle of `PK-1` links, head / middle / tail by nested `BranchNode`, oriented by the herd rank.
Both swaps are the same unroll-by-two-and-peel (`swap_loop`), over the row axis for the
wavefront and over the timestep axis for the halo. A shape no builder covers raises
`NotImplementedError` naming the phase; §3.1's `PLAN` wrapper re-raises any non-`SpatialError`
as a `MappingError` carrying it in `details["internal_exception"]` (§5), so an input nothing
builds fails loudly and legibly instead of silently producing a wrong plan.

**Determinism** (HLD §5, invariant I-7): every traversal that affects a name, an order or a
text is `sorted(...)` on an explicit key; no `id()`, no clock, no RNG.
"""

from __future__ import annotations

from dataclasses import fields, replace
from fractions import Fraction
from math import lcm, prod
from typing import Any, Callable, NamedTuple, get_args

from spatial.m4_selfcheck import self_check, warnings
from spatial.model import (Axis, BinOp, BranchNode, BufferPlan, ChannelPlan, ChannelSite, Const,
                           Diagnostic, Dtype, Expr, ExprNode, Guard, HerdPlan, LegalMapping,
                           Load, LoopPlan, MappingError, MappingPlan, MappingSummary, MaxMin,
                           Param, Region, SpatialError, StoreNode)

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
"""One core tile's **data memory** — `air.api`'s `L1_BYTES` (`_trace.py:100`) and mlir-aie's
`AIE2TargetModel::getLocalMemorySize()` (`0x00010000`). This is the figure `MappingSummary`
reports as `l1_budget` and renders as *"L1: n of 65536 bytes"*; what a plan must actually fit
into is `L1_USABLE` below, which is smaller."""

L1_STACK_RESERVED = 2048
"""The per-core stack `air-to-aie` reserves below the first buffer — its `stack-size` option
default (mlir-air `mlir/include/air/Conversion/Passes.td:231-234`), written onto every
`aie.core` as `stack_size = 2048 : i32` and used by mlir-aie's `AIEAssignBuffers.cpp` as the
start address of buffer allocation."""

L1_USABLE = L1_BUDGET - L1_STACK_RESERVED
"""The binding budget of `06-interfaces.md` §5.6 invariant 5 — 63 488 B, architect ruling
**R-L1-2**, 2026-09-15."""

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

_LATER = ("is out of this cut: M4 builds the LLD §6.1 fill/compute/drain, §3.6.2 wavefront, "
          "§3.6.1 halo and §3.6.3 cascade protocols in the shapes design/04-test-plan.md §5 "
          "fixes, and raises by name for anything else")
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
    if mapping.l1_bytes > L1_USABLE:
        raise _internal(
            f"l1_bytes {mapping.l1_bytes} exceeds the {L1_USABLE} byte budget; M3 raises "
            f"L1-CAPACITY before M4 sees it (design/06-interfaces.md §4.1)",
            l1_bytes=mapping.l1_bytes, l1_budget=L1_USABLE)
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
    return tuple((row[0], _PATTERN[override[row[0]].pattern],
                  pe_axis(mapping, override[row[0]]), True)
                 if row[0] in override else row for row in rows)


def clause_text(clause) -> str:
    """A `StreamClause` as the user wrote it, for a diagnostic's `clause` field (HLD §4.2)."""
    if clause.pattern == "forward" and clause.direction is not None:
        return (f'forward("{clause.operand}", along=ax.{clause.along}, '
                f'dir="{clause.direction}")')
    return f'stream("{clause.operand}", pattern="{clause.pattern}", along=ax.{clause.along})'


def pe_axis(mapping: LegalMapping, clause) -> str:
    """A declared `along` mapped to the PE axis carrying it (the ruling on **B-P24**).

    `along` in a `MappingPlan.delivery` row is always `px`/`py`: the derived rows are built from
    `PE_AXIS_NAME` (LLD §3.2 lines 14, 18), so an overridden row spelling the *schedule* axis
    (`j0`) would make one column of the table mean two different things, and the summary would
    print `forward along j0` where §6.4 and `02-hld.md` §7.3 print `along px`. The clause axis
    must therefore be a placed axis, and naming an unplaced one is a `PROTOCOL-UNSUPPORTED`.
    """
    place = tuple(mapping.schedule.place)
    if clause.along not in place:
        raise _fail(
            f"{clause.along!r} is not a placed axis, so no PE axis carries the declared "
            f"delivery of {clause.operand!r}; placed here: {list(place)}",
            f"name a placed axis in the clause, or place {clause.along!r} with "
            f"place(px={clause.along})",
            clause=clause_text(clause), operand=clause.operand, along=clause.along,
            place=list(place))
    return PE_AXIS_NAME[place.index(clause.along)]


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
    """`T` of LLD §3.9 line 3: the outer tile axes in σ order that are not placed.

    An outer tile axis of **extent 1** is dropped: `tile(ax.j, N)` at `N == extent(j)` makes
    one tile, its loop runs once at `lo`, and no operand's slab moves with it — it is the same
    schedule as not tiling `j` at all, which §3.9 already excludes from `T` (an untiled axis
    runs *inside* the tile). Keeping it made `B[k,j]` "re-fetched per `j0`" on a one-tile `j`
    and, with a second moved axis, sent `PROTOCOL` looking for two streaming loops
    (defect D2, fixed 2026-09-15).
    """
    columns = len(mapping.axes)
    out = []
    for axis in _sigma_axes(mapping):
        if not _is_outer_tile(axis) or axis.extent == 1:
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
    """The L1 staging buffer's name (LLD §3.1's table): `acc` for the accumulator, else lower.

    A tensor and its L1 tile are two different `BufferPlan`s and `06-interfaces.md` §5.6 forbids
    one name covering both, so an already-lower-case parameter gets a `b` suffix: W1's `A` stages
    into `a`, W3's `q` into `qb` (LLD §6.4's buffer table).
    """
    if _is_accumulator_operand(mapping, operand):
        return "acc"
    lowered = operand.lower()
    return f"{lowered}b" if any(p.name == lowered for p in mapping.kernel.params) else lowered


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


def _level(mapping: LegalMapping, operand: str) -> str:
    """`reside(a=...)` for one operand, with L2 rejected by name (LLD §3.3 note 5, FR-S12)."""
    level = dict(mapping.schedule.residency).get(operand, "L1")
    if level not in ("L1", "L3"):
        raise _fail(
            f"no segment-private staging protocol is synthesised in this cut, so "
            f"{operand!r} cannot reside in {level}",
            f'drop the clause or write reside({operand}="L1"): an L2 staging protocol is '
            f"future work (design/03-lld-M4-mapping.md §3.3 note 5)",
            clause=f'reside({operand}="{level}")', operand=operand, level=level)
    return level


def buffer_plan(mapping: LegalMapping,
                delivery: tuple[tuple[str, str, str | None, bool], ...],
                herd: HerdPlan) -> tuple[BufferPlan, ...]:
    """Every non-L3 buffer, in **allocation order** (`06-interfaces.md` §5.6, LLD §3.3).

    Allocation order is the order the herd body allocates them: the resident buffers first
    (`loop_depth 0`), then the streamed ones (`loop_depth 1`), each group in operand order.
    `plan()` asserts that the body it then builds allocates them in exactly this order.
    """
    levels = {operand: _level(mapping, operand) for operand, *_rest in delivery}
    if _exchanged(mapping) is not None:
        return halo_buffers(mapping, delivery, herd, levels)
    if _forwarded(delivery) is not None:
        return wavefront_buffers(mapping, delivery, herd, levels)
    if mapping.r_space:
        return cascade_buffers(mapping, delivery, levels)
    return _staged_buffers(mapping, delivery, levels)


def _staged_buffers(mapping: LegalMapping,
                    delivery: tuple[tuple[str, str, str | None, bool], ...],
                    levels: dict[str, str]) -> tuple[BufferPlan, ...]:
    """One L1 tile per L1-resident operand, in allocation order (LLD §3.3 lines 10-21)."""
    out = []
    for operand, _kind, _along, _declared in delivery:
        if levels[operand] != "L1":
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


def source_name(axis: str) -> str:
    """The segment-scope source loop for the wavefront: `i_source` (R-W3-3, `06-interfaces` §5.5).

    §5.5's v5 rule names a loop by the axis whose trips it enumerates; `<axis>_source` is the
    one spelling the rule did not yet have, because §6.1 has no source loop at all. The addition
    is recorded in `design/PROGRESS-B.md`, phase P4.
    """
    return f"{axis}_source"


def row_drain_name(axis: str) -> str:
    """The segment-scope row drain: `i_drain`, the same `<axis>_drain` rule as §6.1's."""
    return f"{axis}_drain"


def row_axis(mapping: LegalMapping, operand: str) -> str:
    """The unplaced kernel axis `operand`'s rows run along — the wavefront's time (LLD §3.6.2)."""
    matrix = _access_matrix(mapping, operand)
    placed = {_root(mapping, name) for name in mapping.schedule.place}
    axes = [axis.name for axis in mapping.kernel.axes
            if axis.name not in placed and any(_column(matrix, _ucol(mapping, axis.name)))]
    if len(axes) != 1:
        raise NotImplementedError(
            f"the forwarded operand {operand!r} is indexed by {len(axes)} unplaced axes "
            f"{axes}; a wavefront needs exactly one, which is the row it advances along; "
            f"{_LATER}")
    return axes[0]


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


def loop_plan(mapping: LegalMapping,
              delivery: tuple[tuple[str, str, str | None, bool], ...],
              buffers: tuple[BufferPlan, ...]) -> tuple[tuple[str, str], ...]:
    """`LOOP_KIND` applied to every loop axis the plan realises: `(axis, kind)`, sorted by axis.

    The bundle loops carry a channel index and are unrolled; the streaming, compute and zeroing
    loops are `air.sequential`. `PROTOCOL` reads the kind from here and never decides it itself.

    The wavefront's source and drain loops walk **rows**, not the PE grid: the row index is not
    a bundle index, so `<row>_source` and `<row>_drain` are `air.sequential` and the PE grid is
    walked by the `p<root>_bundle` loop above (ruling **R-W3-3**; `03-lld-M5-emitter.md` §6.4
    lines 31-35 draw them as Python loops, which contradicts §3.5's `LOOP_KIND`). §6.1's
    `<root>_drain` is the PE-grid drain and does not exist on a wavefront plan.

    The halo's drain walks **planes** inside the PE bundle loop, for the same reason and by the
    same rule: `t` is not a bundle index, so `<time>_drain` is `air.sequential(lo, hi)` and only
    `p<root>_bundle` around it is unrolled (ruling **R-W2-1**; `03-lld-M5-emitter.md` §6.3
    lines 21-23 draw both as Python loops).
    """
    kinds: dict[str, str] = {}
    forwarded = _forwarded(delivery)
    exchanged = _exchanged(mapping)
    for pe_dim in range(len(mapping.schedule.place)):
        kinds[bundle_name(mapping, pe_dim)] = loop_kind(bundle_name(mapping, pe_dim),
                                                        bundle_index=True)
        if forwarded is None and exchanged is None:
            kinds[drain_name(mapping, pe_dim)] = loop_kind(drain_name(mapping, pe_dim),
                                                           bundle_index=True)
    if forwarded is not None:
        axis = row_axis(mapping, forwarded[0])
        for name in (source_name(axis), row_drain_name(axis)):
            kinds[name] = loop_kind(name, bundle_index=False)
    if exchanged is not None:
        for axis in mapping.schedule.sequential:
            kinds[row_drain_name(axis)] = loop_kind(row_drain_name(axis), bundle_index=False)
    if mapping.r_space:
        # The cascade's drain walks the temporal tile axis the accumulator moves with — the
        # flip's `i0`, not the PE grid — so it is `<axis>_drain` and `air.sequential`, exactly
        # as `drain_name`'s docstring anticipated (ruling **R-F-1**; `03-lld-M5-emitter.md`
        # §6.2 line 37 draws it as a Python loop, which `LOOP_KIND` contradicts).
        axis = streaming_axis(mapping, _statement(mapping).target.operand)
        if axis is not None:
            kinds[row_drain_name(axis)] = loop_kind(row_drain_name(axis), bundle_index=False)
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


def _forwarded(delivery: tuple[tuple[str, str, str | None, bool], ...]
               ) -> tuple[str, str, str | None, bool] | None:
    """The one `FORWARD` delivery row, or `None` — the wavefront dispatch of LLD §3.1 line 7."""
    rows = [row for row in delivery if row[1] == "FORWARD"]
    if len(rows) > 1:
        raise NotImplementedError(
            f"{len(rows)} operands are delivered FORWARD ({[row[0] for row in rows]}); this cut "
            f"synthesises one wavefront per plan; {_LATER}")
    return rows[0] if rows else None


def channel_plan(mapping: LegalMapping,
                 delivery: tuple[tuple[str, str, str | None, bool], ...],
                 herd: HerdPlan, buffers: tuple[BufferPlan, ...],
                 loops: tuple[tuple[str, str], ...]) -> tuple[ChannelPlan, ...]:
    """Every `air.channel` with its geometry and its ordered sites, sorted by name (FR-M8)."""
    if _exchanged(mapping) is not None:
        return halo(mapping, delivery, herd, buffers, loops)[0]
    if _forwarded(delivery) is not None:
        return wavefront(mapping, delivery, herd, buffers, loops)[0]
    if mapping.r_space:
        return cascade(mapping, delivery, herd, buffers, loops)[0]
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


def _site(channel: str, kind: str, order: int, scope: str, *, guard: Guard | None = None,
          is_async: bool = False, depends_on: tuple[str, ...] = (), **rest: Any) -> ChannelSite:
    """One site, with LLD §3.1's `id` rule.

    `guard`, `is_async` and `depends_on` default to the plain token-free form every fill /
    compute / drain and wavefront site takes; the halo's boundary puts pass `is_async=True` and
    its ghost gets keep `depends_on=()` deliberately (§3.6.1's site order, VF §C's E1 verdict).
    """
    return ChannelSite(id=f"{channel}.{kind}.{order}@{scope}", kind=kind, channel=channel,
                       scope=scope, guard=guard, is_async=is_async, depends_on=depends_on,
                       order=order, **rest)


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
    # The segment body is `PROTOCOL`'s: one fill per filled operand, then the herd, then one
    # drain per drained operand. A fill is wrapped in its bundle loops (one per bundle dim of
    # extent > 1) and its streaming loop, a drain in the PE-grid bundle loops; a site inside any
    # wrapper is the first node of that wrapper's body and carries order 0. When **no** wrapper
    # is built — `_bundle_nest` makes no loop for a dim of extent 1, so a `grid(1)` or `grid(1,
    # 1)` drain has none — the site sits in the segment body itself and its order is its index
    # there (the defect D1 fixed 2026-09-15: a single-PE schedule failed `_check_orders`).
    kinds = {operand: (kind, along) for operand, kind, along, _declared in delivery}
    index = 0
    for operand in fills:
        kind, along = kinds[operand]
        size = (multicast_geometry(herd.grid, PE_AXIS_NAME.index(along))[0]
                if kind == "MULTICAST" else herd.grid)
        wrapped = any(extent > 1 for extent in size) or streaming_axis(mapping, operand)
        out[(f"{operand}2L1", "put")] = _site(
            f"{operand}2L1", "put", 0 if wrapped else index, "segment",
            indices=tuple(_var(bundle_name(mapping, d)) if extent > 1 else ZERO
                          for d, extent in enumerate(size)),
            buffer=operand,
            region=l3_region(mapping, operand, _fill_origins(mapping, operand, size)))
        index += 1
    index += 1                                                # the herd itself
    for operand in drains:
        wrapped = any(extent > 1 for extent in herd.grid)
        out[(f"{operand}2L3", "get")] = _site(
            f"{operand}2L3", "get", 0 if wrapped else index, "segment",
            indices=tuple(_var(drain_name(mapping, d)) if extent > 1 else ZERO
                          for d, extent in enumerate(herd.grid)),
            buffer=operand,
            region=l3_region(mapping, operand, _drain_origins(mapping, herd.grid)))
        index += 1
    return out


def _origins(mapping: LegalMapping, pe_var, extents: tuple[int, ...] | None = None
             ) -> dict[str, Expr]:
    """Per untiled kernel axis, the element coordinate this slab starts at (LLD §3.4).

    `pe_var(pe_dim)` names the segment-scope loop standing in for PE coordinate `pe_dim`.

    The axis's own `lo` is part of every origin, placed or not: PE `p` owns
    `[lo + p·factor, lo + (p+1)·factor)`, not `[p·factor, …)`. W1's axes all start at 0, so the
    term was invisible until W2's `i` and W3's `i`/`j`, which start at 1 (the P2b caveat in
    `design/PROGRESS-B.md`, phase P2b — the one edit that phase asked P4 to make).

    `extents` is the bundle shape the caller's `_bundle_nest` walked, when there is one. That
    nest makes **no** loop for a dim of extent 1 (`_bundle_nest` skips it), so `pe_var(d)` names
    nothing there and the origin is the axis's `lo`: one PE owns the whole axis. It is invisible
    on W1/W2/W3 — every PE dim of theirs has extent > 1 — and bites on the flip's 2-D variant,
    whose `grid(1, 4)` places `i` on a dim of extent 1.
    """
    bindings = dict(mapping.kernel.bindings)
    placed = {_root(mapping, name): d for d, name in enumerate(mapping.schedule.place)}
    out: dict[str, Expr] = {}
    for axis in mapping.kernel.axes:
        lo = _resolve(axis.lo, bindings)
        pe_dim = placed.get(axis.name)
        if pe_dim is None or (extents is not None and extents[pe_dim] == 1):
            out[axis.name] = lo
            continue
        factor = _tile_factor(mapping, axis.name)
        out[axis.name] = _add(lo, _var(pe_var(pe_dim), 1 if factor is None else factor))
    return out


def _fill_origins(mapping: LegalMapping, operand: str,
                  extents: tuple[int, ...] | None = None) -> dict[str, Expr]:
    """`_origins` with the streaming axis of `operand` bound to its own loop variable.

    The streaming loop runs in element units (`air.sequential(0, 64, 16)` over `k0`), so the
    origin along that axis is the loop variable itself with coefficient 1 — §3.4's `kk`.
    """
    origins = _origins(mapping, lambda d: bundle_name(mapping, d), extents)
    axis = streaming_axis(mapping, operand)
    if axis is not None:
        origins[_root(mapping, axis)] = _var(axis)
    return origins


def _drain_origins(mapping: LegalMapping,
                   extents: tuple[int, ...] | None = None) -> dict[str, Expr]:
    """`_origins` over the drain loops, which index the PE grid directly."""
    return _origins(mapping, lambda d: drain_name(mapping, d), extents)


# --------------------------------------------------------------------------------------------
# §3.6 — PROTOCOL
# --------------------------------------------------------------------------------------------


def protocol(mapping: LegalMapping, delivery: tuple[tuple[str, str, str | None, bool], ...],
             herd: HerdPlan, buffers: tuple[BufferPlan, ...],
             channels: tuple[ChannelPlan, ...],
             loops: tuple[tuple[str, str], ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """`(segment_body, herd_body)` — the dispatch of LLD §3.1 line 7 / §3.6.

    The dispatch is on which clauses are present, not on which workload it is: `exchange`
    selects §3.6.1, `forward` §3.6.2 and a non-empty `r_space` §3.6.3. The cascade still
    raises; what this cut builds is §6.1's fill / compute / drain shape, §3.6.2's wavefront and
    §3.6.1's halo.
    """
    if _exchanged(mapping) is not None:
        _channels, segment_body, herd_body = halo(mapping, delivery, herd, buffers, loops)
        _check_orders(segment_body)
        _check_orders(herd_body)
        return segment_body, herd_body
    if _forwarded(delivery) is not None:
        _channels, segment_body, herd_body = wavefront(mapping, delivery, herd, buffers, loops)
        _check_orders(segment_body)
        _check_orders(herd_body)
        return segment_body, herd_body
    if mapping.r_space:
        _channels, segment_body, herd_body = cascade(mapping, delivery, herd, buffers, loops)
        _check_orders(segment_body)
        _check_orders(herd_body)
        return segment_body, herd_body
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
        # A user-facing limit, not an internal one (NFR-7, defect D2): the schedule left two
        # outer tile axes temporal and two operands moving with different ones, and this cut
        # builds one streaming loop per herd body (§6.1).
        moves = {b.operand: streaming_axis(mapping, b.operand) for b in streamed}
        raise _fail(
            "the staged operands move with different outer tile axes ("
            + ", ".join(f"{operand} per {name}" for operand, name in sorted(moves.items()))
            + "), so the herd body would need one streaming loop per axis and this cut "
              "synthesises one (design/03-lld-M4-mapping.md §6.1)",
            f"place one of {sorted({name for name in moves.values() if name})} on a PE axis, "
            f"or leave its kernel axis untiled, so every staged operand is re-fetched per the "
            f"same axis",
            clause="tile(...)/place(...)",
            streaming_axes={operand: name for operand, name in sorted(moves.items())})
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


def _zero(dtype: Dtype) -> Const:
    """The zero literal of `dtype`, carrying the token M5 prints (LLD §3.3 rule 4)."""
    if dtype in (Dtype.f32, Dtype.f16, Dtype.bf16):
        return Const(value=0.0, text="0.0", dtype=dtype)
    return Const(value=0, text="0", dtype=dtype)


def _zero_nest(mapping: LegalMapping, accumulator: BufferPlan, names: dict[str, str],
               loops: tuple[tuple[str, str], ...], depth_: int,
               subscripts: tuple[Expr, ...], expr: Any = None) -> LoopPlan:
    """The accumulator zeroing: an ordinary `LoopPlan` of `StoreNode(Const 0)` (§5.5, §6.1).

    It carries no `statement_index`, because no kernel statement corresponds to it. Its loops
    realise the same post-tiling axes as the compute nest and carry the same names (§5.5 at v5),
    and it writes the same element of the accumulator that the compute nest then accumulates
    into — hence the shared `subscripts`.

    `expr` overrides the stored value, which is what makes this the same nest the cascade's
    `acc[:] = acc[:] ⊕ recv[:]` needs (§3.6.3 line 21, §6.2): same shape, same names, same
    subscripts, one different leaf. It is the **only** other arithmetic node M4 synthesises
    rather than reading off `Statement.expr`, because no kernel statement corresponds to it.
    """
    statement = _statement(mapping)
    body: tuple[Any, ...] = (StoreNode(buffer_id=accumulator.name, subscripts=subscripts,
                                       expr=_zero(accumulator.dtype) if expr is None else expr),)
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
# §3.6.2 — the wavefront forward protocol (FR-M5) — W3
# --------------------------------------------------------------------------------------------

WEST_IN, WEST, EAST_OUT = "WestIn", "West", "EastOut"
"""FR-M5's three homogeneous channels: L3→L1 source, L1→L1 links, L1→L3 drain (finding N-2).

A single `size=[PJ+1]` bundle carrying both the L3 ends and the core-to-core links is what
`AIRLoweringPass.cpp:798` rejects with `failed to specialize channel bundle indices` (measured),
which is why the property FR-M5 asks for is spread across three channels instead.
"""

PREV, CUR, EDGE_IN, EDGE_OUT = "prev", "cur", "edge_in", "edge_out"
"""The protocol buffers §6.4 names: the row swap pair and the two edge scalars."""


def staging_channel_name(operand: str, suffix: str) -> str:
    """`QIn`, `RIn`, `SOut`, `UIn`, `UOut` — §6.3/§6.4's own tables, not §3.5's `{a}2L1`.

    §3.5's rule names W1's channels `A2L1`/`C2L3`; §6.4 and `02-hld.md` §7.3 name every one of
    the wavefront's six `<Name>In` / `<Name>Out`, and §3.6.1's halo table names `UIn`/`UOut` by
    the same rule — the family `WestIn`/`West`/`EastOut` already belongs to, and what the
    acceptance rows of FR-M4, FR-M5 and §7 name. The wavefront and the halo therefore keep
    §6.3/§6.4's spelling; the fill/compute/drain protocol keeps §3.5's. (Named
    `wavefront_channel_name` at P4, renamed at P5 when W2's halo became its second caller;
    recorded in `design/PROGRESS-B.md`.)
    """
    return f"{operand[:1].upper()}{operand[1:]}{suffix}"


class Band(NamedTuple):
    """The geometry of the forwarded operand's per-PE row band (LLD §3.6.2, §6.4)."""

    operand: str
    dtype: Dtype
    row_axis: str
    col_axis: str
    row_dim: int
    col_dim: int
    columns: int
    ghost: int
    width: int
    lo: int
    hi: int
    col_lo: int


def band_geometry(mapping: LegalMapping,
                  delivery: tuple[tuple[str, str, str | None, bool], ...],
                  herd: HerdPlan) -> Band:
    """The `Band` of the one `FORWARD` row — the whole of what §3.6.2 needs to be general.

    The forwarded operand is written, its rows advance along the unplaced axis and its columns
    along the placed one. The band is the PE's `CW` owned columns plus the **west overhang** its
    own accesses reach (`S[i, j-1]` → one ghost cell), and the row span is exactly 2 — the row
    being computed and the one before it — which is what makes the swap pair a pair.
    """
    operand = _forwarded(delivery)[0]
    param = _param(mapping, operand)
    clause = next((c for c in mapping.schedule.streams if c.operand == operand), None)
    if not param.is_written:
        raise _fail(
            f"{operand!r} is read-only, and the wavefront of "
            f"design/03-lld-M4-mapping.md §3.6.2 forwards the **written** operand's edge scalar "
            f"along the PE line; forwarding a read operand's tile has no synthesis rule in this "
            f"cut",
            f"drop the clause and let the derivation choose the delivery of {operand!r}, or "
            f"forward the operand the kernel writes",
            clause=_CLAUSE if clause is None else clause_text(clause), operand=operand)
    if _level(mapping, operand) != "L1":
        raise _fail(
            f"the forwarded operand {operand!r} does not reside in L1, so there is no band to "
            f"forward",
            f'write reside({operand}="L1")', operand=operand)
    if len(herd.grid) != 1:
        raise NotImplementedError(
            f"the herd is rank {len(herd.grid)}; a wavefront forwards along one PE line and "
            f"this cut synthesises the 1-D case only; {_LATER}")
    if herd.grid[0] < 2:
        raise NotImplementedError(
            f"the herd has {herd.grid[0]} PE(s); a wavefront needs at least one link, so "
            f"West would have extent 0; {_LATER}")
    matrix = _access_matrix(mapping, operand)
    col_axis = _root(mapping, mapping.schedule.place[0])
    column = _ucol(mapping, col_axis)
    col_dims = [d for d, row in enumerate(matrix) if row[column]]
    if len(matrix) != 2 or len(col_dims) != 1:
        raise NotImplementedError(
            f"{operand!r} has rank {len(matrix)} and is indexed by the placed axis "
            f"{col_axis!r} in {len(col_dims)} of its dims; a row band needs a rank-2 operand "
            f"with one row dim and one column dim; {_LATER}")
    col_dim = col_dims[0]
    row_dim = 1 - col_dim
    row = row_axis(mapping, operand)
    if matrix[row_dim] != tuple(1 if j == _ucol(mapping, row) else 0
                                for j in range(len(matrix[row_dim]))):
        raise NotImplementedError(
            f"dim {row_dim} of {operand!r} is not the bare axis {row!r}; a row band needs one "
            f"unit-coefficient row axis per operand; {_LATER}")
    bindings = dict(mapping.kernel.bindings)
    spans = []
    for dim in (0, 1):
        offsets = []
        for access in (a for statement in mapping.kernel.statements
                       for a in (statement.target,) + statement.reads if a.operand == operand):
            offset = _resolve(access.offsets[dim], bindings)
            if not offset.is_constant:
                raise NotImplementedError(
                    f"an access to {operand!r} has the non-constant offset {offset} in dim "
                    f"{dim}; a band's extent must be a compile-time constant; {_LATER}")
            offsets.append(offset.const)
        spans.append((min(offsets), max(offsets)))
    if spans[row_dim] != (-1, 0):
        raise NotImplementedError(
            f"{operand!r} is read at row offsets {spans[row_dim]}; the swap pair holds the row "
            f"being computed and the one before it, and nothing else; {_LATER}")
    if spans[col_dim][1] != 0 or spans[col_dim][0] > 0:
        raise NotImplementedError(
            f"{operand!r} is read at column offsets {spans[col_dim]}; a west-to-east wavefront "
            f"reaches west of its own band and never east of it; {_LATER}")
    columns = _tile_extent(mapping, col_axis)
    if columns * herd.grid[0] != _kernel_axis(mapping, col_axis).extent:
        raise NotImplementedError(
            f"{herd.grid[0]} PEs of {columns} columns do not cover the {col_axis!r} extent "
            f"{_kernel_axis(mapping, col_axis).extent} exactly; {_LATER}")
    axis = _kernel_axis(mapping, row)
    lo, hi, step = (_resolve(bound, bindings) for bound in (axis.lo, axis.hi, axis.step))
    if not (lo.is_constant and hi.is_constant and step.is_constant and step.const == 1):
        raise NotImplementedError(
            f"the row axis {row!r} runs {lo}..{hi} step {step}; a wavefront advances one row at "
            f"a time between compile-time constants; {_LATER}")
    return Band(operand=operand, dtype=param.dtype, row_axis=row, col_axis=col_axis,
                row_dim=row_dim, col_dim=col_dim, columns=columns, ghost=-spans[col_dim][0],
                width=columns - spans[col_dim][0], lo=lo.const, hi=hi.const,
                col_lo=_resolve(_kernel_axis(mapping, col_axis).lo, bindings).const)


def wavefront_buffers(mapping: LegalMapping,
                      delivery: tuple[tuple[str, str, str | None, bool], ...],
                      herd: HerdPlan, levels: dict[str, str]) -> tuple[BufferPlan, ...]:
    """The wavefront's L1 plan, in allocation order (LLD §6.4's buffer table).

    The staged read operands first — `q` whole, `r`'s own column slice — then the row swap pair
    and the two edge scalars. `prev`/`cur` carry `operand`, so `06-interfaces.md` §5.6
    invariant 5 charges them against `LegalMapping.l1_bytes`; the edges carry none, and are the
    8 bytes by which the plan's 240 exceeds M3's 232 (`02-hld.md` §7).
    """
    band = band_geometry(mapping, delivery, herd)
    out = []
    for operand, _how, _along, _declared in delivery:
        if operand == band.operand or levels[operand] != "L1":
            continue
        param = _param(mapping, operand)
        shape = tile_shape(mapping, operand)
        out.append(BufferPlan(name=buffer_name(mapping, operand), operand=operand, level="L1",
                              scope="herd.private", shape=shape, dtype=param.dtype,
                              bytes=prod(shape) * param.dtype.sizeof,
                              loop_depth=depth(mapping, operand),
                              ping_pong_candidate=ping_pong(mapping, operand)))
    for name, shape in ((PREV, (band.width,)), (CUR, (band.width,)),
                        (EDGE_IN, (1,)), (EDGE_OUT, (1,))):
        out.append(BufferPlan(name=name, operand=band.operand if name in (PREV, CUR) else None,
                              level="L1", scope="herd.private", shape=shape, dtype=band.dtype,
                              bytes=prod(shape) * band.dtype.sizeof, loop_depth=0,
                              ping_pong_candidate=False))
    return tuple(out)


def swap_loop(axis: str, lo: int, hi: int, kind: str, depth_: int,
              trip: Callable[[Expr, int, int], tuple[Any, ...]],
              base: int = 0) -> tuple[tuple[Any, ...], int]:
    """Unroll a two-phase temporal loop by two, peeling an odd trip (LLD §3.6.1 lines 1-11).

    `air.sequential` has no `iter_args` anywhere in `air.api` (`_loop.py:180`), so a `prev`/`cur`
    role swap cannot be loop-carried, and a plain Python loop would unroll the whole protocol and
    strand the acquire/release pairs (`_loop.py:14-19`). The loop is therefore emitted at
    `step=2` with **both** phases in its body, and an odd trip count is **peeled** rather than
    rejected — the architect's override of D-4. W3's row swap is the first user; W2's timestep
    swap (P5) is the second, which is why this is a function and not a branch of the builder.

    `trip(index, phase, order)` builds one trip's nodes, given the row index as an `Expr`, the
    phase (0 or 1) and the `ChannelSite.order` its first node carries; `base` is the index of the
    returned `LoopPlan` in its enclosing body, so a peeled trip's sites are ordered after it.
    Returns `(nodes, phase)` — the phase the next trip *would* have run, which is what says which
    buffer is live after the loop (line 9).
    """
    count = max(0, hi - lo)
    first = trip(_var(axis), 0, 0)
    loop = LoopPlan(axis=axis, lo=_const(lo), hi=_const(lo + count - count % 2), step=_const(2),
                    kind=kind, depth=depth_,
                    body=first + trip(_add(_var(axis), ONE), 1, len(first)))
    if count % 2 == 0:
        return (loop,), 0
    return (loop,) + trip(_const(lo + count - 1), 0, base + 1), 1


def _in_band(band: Band, elements: dict[str, Expr], origin: Expr,
             subscripts: tuple[Expr, ...], bindings: dict[str, int],
             previous: str, current: str) -> tuple[str, tuple[Expr, ...]]:
    """`(buffer, subscripts)` for one access to the forwarded operand (the P2b table, §6.4).

    The band is rank 1 where the access is rank 2, because the **row** offset does not index a
    buffer — it picks which of the two swapped buffers holds that row: `−1` is the previous row,
    `0` the one being computed. The column is the access's own column minus the band's L3 origin,
    so the PE's `tx·CW` cancels exactly, the way `l1_subscripts` cancels W1's tile origin.
    """
    row = _add(_substitute(_resolve(subscripts[band.row_dim], bindings), elements),
               _scale(elements[band.row_axis], -1))
    if not row.is_constant or row.const not in (-1, 0):
        raise NotImplementedError(
            f"an access to {band.operand!r} sits {row} rows from the one being computed; the "
            f"swap pair holds two rows; {_LATER}")
    column = _add(_substitute(_resolve(subscripts[band.col_dim], bindings), elements),
                  _scale(origin, -1))
    return (previous if row.const else current), (column,)


def wavefront(mapping: LegalMapping,
              delivery: tuple[tuple[str, str, str | None, bool], ...], herd: HerdPlan,
              buffers: tuple[BufferPlan, ...], loops: tuple[tuple[str, str], ...]
              ) -> tuple[tuple[ChannelPlan, ...], tuple[Any, ...], tuple[Any, ...]]:
    """`(channels, segment_body, herd_body)` — the wavefront of LLD §3.6.2 and §6.4.

    Six channels: the staged read operands through §3.5's ordinary `MULTICAST`/`STATIONARY`
    geometry, FR-M5's three homogeneous wavefront channels, and one per-PE drain so **every row**
    reaches L3 and `test_sem_coverage`'s (b) holds. The source puts the operand's own read-only
    boundary column and the drain gets its last written column, so the plan needs no synthetic
    `Zrow`/`Sink` tensor and `MappingPlan.tensors` stays the kernel's three parameters
    (ruling **R-W3-1**).
    """
    band = band_geometry(mapping, delivery, herd)
    tensor = next(t for t in tensor_plan(mapping) if t.name == band.operand)
    strides = _row_major(tensor.shape)
    coord, pes = herd.coords[0], herd.grid[0]
    bundle = bundle_name(mapping, 0)
    source, drain = source_name(band.row_axis), row_drain_name(band.row_axis)
    out_name = staging_channel_name(band.operand, "Out")
    names = compute_names(mapping)
    inner = names[band.col_axis]
    bindings = dict(mapping.kernel.bindings)
    staged = tuple(row for row in delivery if row[0] != band.operand)
    by_operand = {b.operand: b for b in buffers if b.operand not in (None, band.operand)}
    slab = _origins(mapping, lambda d: herd.coords[d])
    origin = _add(slab[band.col_axis], _const(-band.ghost))
    statement = _statement(mapping)

    def l3(row: Expr, column: Expr, width: int) -> Region:
        """One row-band region over the L3 tensor, in the operand's own dim order."""
        offsets: list[Any] = [None, None]
        sizes: list[int] = [0, 0]
        offsets[band.row_dim], sizes[band.row_dim] = row, 1
        offsets[band.col_dim], sizes[band.col_dim] = column, width
        return Region(offsets=tuple(offsets), sizes=tuple(sizes), strides=strides)

    def row_loop(axis: str, depth_: int, body: tuple[Any, ...]) -> LoopPlan:
        return LoopPlan(axis=axis, lo=_const(band.lo), hi=_const(band.hi), step=ONE,
                        kind=_kind(loops, axis), depth=depth_, body=body)

    def trip(index: Expr, phase: int, order: int) -> tuple[Any, ...]:
        """One row: get the west edge, compute the band, put the east edge, drain the row."""
        previous, current = (PREV, CUR) if phase == 0 else (CUR, PREV)
        elements = {axis: (_add(slab[axis], _var(name))
                           if _tile_factor(mapping, axis) is not None else _var(name))
                    for axis, name in names.items()}
        elements[band.row_axis] = index

        def rewrite(load: Load) -> Load:
            if load.buffer_id == band.operand:
                return Load(*_in_band(band, elements, origin, load.subscripts, bindings,
                                      previous, current))
            return Load(*_in_l1(mapping, by_operand, elements, slab, load.buffer_id,
                                load.subscripts))

        target, subscripts = _in_band(band, elements, origin,
                                      kernel_subscripts(mapping, statement.target), bindings,
                                      previous, current)
        return (
            BranchNode(
                predicate=Guard(coord=coord, relation="==", value=ZERO),
                then=(_site(WEST_IN, "get", order, "herd", indices=(ZERO,), buffer=EDGE_IN,
                            region=EMPTY_REGION),),
                otherwise=(_site(WEST, "get", order, "herd", indices=(_var(coord, 1, -1),),
                                 buffer=EDGE_IN, region=EMPTY_REGION),)),
            StoreNode(buffer_id=current, subscripts=(ZERO,), expr=Load(EDGE_IN, (ZERO,))),
            LoopPlan(axis=inner, lo=ZERO, hi=_const(band.columns), step=ONE,
                     kind=_kind(loops, inner), depth=1,
                     body=(StoreNode(buffer_id=target, subscripts=subscripts,
                                     expr=_rewrite_loads(statement.expr, rewrite)),)),
            StoreNode(buffer_id=EDGE_OUT, subscripts=(ZERO,),
                      expr=Load(current, (_const(band.width - 1),))),
            BranchNode(
                predicate=Guard(coord=coord, relation="==", value=_const(pes - 1)),
                then=(_site(EAST_OUT, "put", order + 4, "herd", indices=(ZERO,),
                            buffer=EDGE_OUT, region=EMPTY_REGION),),
                otherwise=(_site(WEST, "put", order + 4, "herd", indices=(_var(coord),),
                                 buffer=EDGE_OUT, region=EMPTY_REGION),)),
            _site(out_name, "put", order + 5, "herd", indices=(_var(coord),), buffer=current,
                  region=Region(offsets=(_const(band.ghost),), sizes=(band.columns,),
                                strides=(1,))),
        )

    geometry: dict[str, tuple[tuple[int, ...], tuple[int, ...] | None, Dtype]] = {}
    segment: list[Any] = []
    for operand, how, along, _declared in staged:                    # §3.5's CHANNEL_PLAN rows
        size, broadcast = (multicast_geometry(herd.grid, PE_AXIS_NAME.index(along))
                           if how == "MULTICAST" else (herd.grid, None))
        fill = staging_channel_name(operand, "In")
        geometry[fill] = (size, broadcast, _param(mapping, operand).dtype)
        top = all(extent == 1 for extent in size)
        put = _site(fill, "put", len(segment) if top else 0, "segment",
                    indices=tuple(_var(bundle_name(mapping, d)) if extent > 1 else ZERO
                                  for d, extent in enumerate(size)),
                    buffer=operand,
                    region=l3_region(mapping, operand, _fill_origins(mapping, operand, size)))
        segment.append(_bundle_nest(mapping, size, lambda d: bundle_name(mapping, d), loops, 0,
                                    (put,))[0])
    segment.append(row_loop(source, 0, (
        _site(WEST_IN, "put", 0, "segment", indices=(ZERO,), buffer=band.operand,
              region=l3(_var(source), _const(band.col_lo - band.ghost), 1)),)))
    segment.append(herd)
    segment.append(row_loop(drain, 0, (
        _site(EAST_OUT, "get", 0, "segment", indices=(ZERO,), buffer=band.operand,
              region=l3(_var(drain), _const(band.col_lo + band.columns * pes - 1), 1)),)))
    segment.append(_bundle_nest(
        mapping, herd.grid, lambda d: bundle_name(mapping, d), loops, 0,
        (row_loop(drain, 1, (
            _site(out_name, "get", 0, "segment", indices=(_var(bundle),), buffer=band.operand,
                  region=l3(_var(drain), _add(_const(band.col_lo), _var(bundle, band.columns)),
                            band.columns)),)),))[0])

    herd_body: list[Any] = list(buffers)
    for operand, _how, _along, _declared in staged:
        herd_body.append(_site(staging_channel_name(operand, "In"), "get", len(herd_body),
                               "herd",
                               indices=tuple(_var(name) for name in herd.coords),
                               buffer=by_operand[operand].name, region=EMPTY_REGION))
    herd_body.append(LoopPlan(                       # the row-0 boundary of the forwarded band
        axis=inner, lo=ZERO, hi=_const(band.width), step=ONE, kind=_kind(loops, inner), depth=0,
        body=(StoreNode(buffer_id=PREV, subscripts=(_var(inner),), expr=_zero(band.dtype)),)))
    rows, _phase = swap_loop(band.row_axis, band.lo, band.hi, _kind(loops, band.row_axis), 0,
                             trip, base=len(herd_body))
    herd_body += list(rows)

    geometry[WEST_IN] = ((1,), None, band.dtype)
    geometry[WEST] = ((pes - 1,), None, band.dtype)
    geometry[EAST_OUT] = ((1,), None, band.dtype)
    geometry[out_name] = ((pes,), None, band.dtype)
    by_channel: dict[str, list[ChannelSite]] = {}
    for node in list(_flatten(tuple(segment))) + list(_flatten(tuple(herd_body))):
        if isinstance(node, ChannelSite):
            by_channel.setdefault(node.channel, []).append(node)
    channels = tuple(sorted(
        (ChannelPlan(name=name, size=size, broadcast_shape=broadcast, channel_type=None,
                     chain_direction=None, dtype=dtype, sites=tuple(by_channel[name]))
         for name, (size, broadcast, dtype) in geometry.items()), key=lambda c: c.name))
    return channels, tuple(segment), tuple(herd_body)


# --------------------------------------------------------------------------------------------
# §3.6.1 — the halo exchange protocol (FR-M4) — W2
# --------------------------------------------------------------------------------------------

NEXT = "next"
"""The second half of §6.3's swap pair; `CUR` above is the first. D-5's *second* meaning:
`double_buffer` on an exchanged operand is this explicit pair, not `isPingPongCandidate`."""

HALO_CHANNELS = (("ToNorth", "ToSouth"), ("ToWest", "ToEast"))
"""§3.1's name table: the two halo bundles per PE axis, ascending-index direction first.

Index `k` of the first name means "the link PE `k+1` → PE `k`" and of the second "PE `k` →
PE `k+1`", which is §3.6.1's table read off the `px` row. Only the `px` pair is reachable in
this cut: a rank-2 herd would exchange along both axes at once, which `strip_geometry` rejects
by name rather than building half of it."""


class Strip(NamedTuple):
    """The geometry of an exchanged operand's per-PE strip (LLD §3.6.1, §6.3).

    The L1 strip is the operand's array **minus its time dim** — that dim's two-plane span is
    the `cur`/`next` pair, not a buffer axis — with every remaining dim widened by its declared
    halo. `dims` lists the array dims the strip keeps, in order, so `shape[k]` is the extent of
    array dim `dims[k]`.
    """

    operand: str
    dtype: Dtype
    time_axis: str
    time_dim: int
    plane: int
    exchange_axis: str
    exchange_dim: int
    halo: int
    owned: int
    dims: tuple[int, ...]
    axes: tuple[str, ...]
    halos: tuple[int, ...]
    shape: tuple[int, ...]
    lo: int
    hi: int
    pes: int
    north: str
    south: str


def _exchanged(mapping: LegalMapping):
    """The one `ExchangeClause`, or `None` — the halo dispatch of LLD §3.1 line 7."""
    clauses = mapping.schedule.exchanges
    if len(clauses) > 1:
        raise NotImplementedError(
            f"{len(clauses)} operands are exchanged "
            f"({[clause.operand for clause in clauses]}); this cut synthesises one halo per "
            f"plan; {_LATER}")
    return clauses[0] if clauses else None


def _dim_axes(mapping: LegalMapping, operand: str) -> tuple[str, ...]:
    """The one unit-coefficient kernel axis indexing each array dim of `operand`."""
    matrix = _access_matrix(mapping, operand)
    out = []
    for index, row in enumerate(matrix):
        support = [mapping.kernel.axes[j].name for j, value in enumerate(row) if value]
        if len(support) != 1 or row[_ucol(mapping, support[0])] != 1:
            raise NotImplementedError(
                f"array dim {index} of {operand!r} is indexed by {support} with coefficients "
                f"{[v for v in row if v]}; a halo strip needs one unit-coefficient axis per "
                f"dim; {_LATER}")
        out.append(support[0])
    return tuple(out)


def strip_geometry(mapping: LegalMapping,
                   delivery: tuple[tuple[str, str, str | None, bool], ...],
                   herd: HerdPlan) -> Strip:
    """The `Strip` of the one `exchange`d operand — all §3.6.1 needs to be general.

    Everything is derived: which array dim is the timestep (the `sequential` axis, whose write
    offset of `+1` against read offsets of `0` is what makes the swap a *pair*), which is the
    exchanged dim (the placed axis the clause names), the owned extent, and the per-dim halo
    from the `WindowClause`. `TILE_SHAPE` for the strip is `[owned + 2·halo]` per kept dim —
    W2's `(10, 16)` — with **no** W2 literal anywhere.
    """
    clause = _exchanged(mapping)
    operand = clause.operand
    param = _param(mapping, operand)
    text = f'exchange("{operand}", along=ax.{clause.along}, halo={clause.halo})'
    if not param.is_written:
        raise _fail(
            f"{operand!r} is read-only, and the halo protocol of "
            f"design/03-lld-M4-mapping.md §3.6.1 exchanges the boundary of the strip the "
            f"kernel **writes**; there is nothing to exchange on a read-only operand",
            f"drop the clause, or exchange the operand the kernel writes",
            clause=text, operand=operand)
    if _level(mapping, operand) != "L1":
        raise _fail(
            f"the exchanged operand {operand!r} does not reside in L1, so there is no strip to "
            f"exchange", f'write reside({operand}="L1")', clause=text, operand=operand)
    place = tuple(mapping.schedule.place)
    if clause.along not in place:
        raise _fail(
            f"{clause.along!r} is not a placed axis, so no PE line carries the halo of "
            f"{operand!r}; placed here: {list(place)}",
            f"name a placed axis in the clause, or place {clause.along!r} with "
            f"place(px={clause.along})",
            clause=text, operand=operand, along=clause.along, place=list(place))
    pe_dim = place.index(clause.along)
    if len(herd.grid) != 1:
        raise NotImplementedError(
            f"the herd is rank {len(herd.grid)}; a halo on a rank-2 herd exchanges along both "
            f"PE axes at once ({HALO_CHANNELS[0]} **and** {HALO_CHANNELS[1]}), and this cut "
            f"synthesises the 1-D case only; {_LATER}")
    if herd.grid[pe_dim] < 2:
        raise NotImplementedError(
            f"the herd has {herd.grid[pe_dim]} PE(s); a halo needs at least one link, so "
            f"{HALO_CHANNELS[pe_dim][0]} would have extent 0; {_LATER}")
    if _forwarded(delivery) is not None or mapping.r_space:
        raise NotImplementedError(
            f"the schedule combines an exchange with a wavefront or a cascade; §3.1 says the "
            f"three write into disjoint parts of the body, but this cut builds one protocol "
            f"per plan; {_LATER}")

    axes = _dim_axes(mapping, operand)
    exchange_axis = _root(mapping, clause.along)
    time = [d for d, axis in enumerate(axes) if axis in mapping.schedule.sequential]
    if len(time) != 1 or exchange_axis not in axes:
        raise NotImplementedError(
            f"{operand!r} is indexed by {list(axes)} with sequential axes "
            f"{list(mapping.schedule.sequential)}; a halo strip needs exactly one timestep dim "
            f"and one exchanged dim; {_LATER}")
    time_dim = time[0]
    time_axis = axes[time_dim]
    statement = _statement(mapping)
    bindings = dict(mapping.kernel.bindings)
    planes = {_resolve(access.offsets[time_dim], bindings)
              for access in statement.reads} | {_const(0)}
    plane = _resolve(statement.target.offsets[time_dim], bindings)
    if planes != {_const(0)} or not plane.is_constant or plane.const != 1:
        raise NotImplementedError(
            f"{operand!r} is written at timestep offset {plane} and read at "
            f"{sorted(p.const for p in planes if p.is_constant)}; the swap pair holds the plane "
            f"being computed and the one before it, and nothing else; {_LATER}")

    window = next((w for w in mapping.schedule.windows if w.operand == operand), None)
    halos = dict(zip(window.dims, window.halo)) if window is not None else {}
    if halos.get(exchange_axis, 0) < 1:
        raise _fail(
            f"{operand!r} is exchanged along {exchange_axis!r} but declares no halo there, so "
            f"there is no ghost row to fill",
            f'declare the window: window("{operand}", dims=(...), halo=1)',
            clause=text, operand=operand, along=clause.along,
            window=None if window is None else list(window.dims))
    axis = _kernel_axis(mapping, time_axis)
    lo, hi, step = (_resolve(bound, bindings) for bound in (axis.lo, axis.hi, axis.step))
    if not (lo.is_constant and hi.is_constant and step.is_constant and step.const == 1):
        raise NotImplementedError(
            f"the timestep axis {time_axis!r} runs {lo}..{hi} step {step}; a halo advances one "
            f"plane at a time between compile-time constants; {_LATER}")
    kept = tuple(d for d in range(len(axes)) if d != time_dim)
    widths = tuple(halos.get(axes[d], 0) for d in kept)
    north, south = HALO_CHANNELS[pe_dim]
    return Strip(operand=operand, dtype=param.dtype, time_axis=time_axis, time_dim=time_dim,
                 plane=plane.const, exchange_axis=exchange_axis,
                 exchange_dim=axes.index(exchange_axis), halo=halos[exchange_axis],
                 owned=_tile_extent(mapping, exchange_axis), dims=kept,
                 axes=tuple(axes[d] for d in kept), halos=widths,
                 shape=tuple(_tile_extent(mapping, axes[d]) + 2 * w
                             for d, w in zip(kept, widths)),
                 lo=lo.const, hi=hi.const, pes=herd.grid[pe_dim], north=north, south=south)


def halo_buffers(mapping: LegalMapping,
                 delivery: tuple[tuple[str, str, str | None, bool], ...],
                 herd: HerdPlan, levels: dict[str, str]) -> tuple[BufferPlan, ...]:
    """The halo's L1 plan: the `cur`/`next` swap pair, in allocation order (§6.3).

    Both carry `operand`, so `06-interfaces.md` §5.6 invariant 5 charges them against
    `LegalMapping.l1_bytes` — W2's `2 · 10 · 16 · 4 = 1280`, which M3 charged as the same
    two-plane span. Neither is a `ping_pong_candidate`: `DEPTH` is 0 for a written operand and
    the pair sits **outside** the timestep loop, so `isPingPongCandidate` has no candidate loop
    to find (R-W2-4, D-5's second meaning).
    """
    strip = strip_geometry(mapping, delivery, herd)
    out = []
    for name in (CUR, NEXT):
        out.append(BufferPlan(name=name, operand=strip.operand, level="L1",
                              scope="herd.private", shape=strip.shape, dtype=strip.dtype,
                              bytes=prod(strip.shape) * strip.dtype.sizeof,
                              loop_depth=depth(mapping, strip.operand),
                              ping_pong_candidate=ping_pong(mapping, strip.operand)))
    for operand, _how_, _along, _declared in delivery:
        if operand != strip.operand and levels[operand] == "L1":
            raise NotImplementedError(
                f"the schedule stages {operand!r} beside the exchanged {strip.operand!r}; this "
                f"cut's halo body stages the exchanged operand only; {_LATER}")
    return tuple(out)


def _in_strip(strip: Strip, elements: dict[str, Expr], origins: tuple[Expr, ...],
              subscripts: tuple[Expr, ...], bindings: dict[str, int],
              source: str, destination: str) -> tuple[str, tuple[Expr, ...]]:
    """`(buffer, subscripts)` for one access to the exchanged operand (§6.3's update).

    The strip is rank 2 where the access is rank 3, because the **timestep** offset does not
    index a buffer — it picks which of the two swapped buffers holds that plane: `0` is the one
    being read, `+1` the one being written. Every other dim is the access's own index minus the
    staged slab's origin, so the PE's `tx·HS` cancels exactly the way `l1_subscripts` cancels
    W1's tile origin, and the ghost row shifts the result by the halo.
    """
    plane = _add(_substitute(_resolve(subscripts[strip.time_dim], bindings), elements),
                 _scale(elements[strip.time_axis], -1))
    if not plane.is_constant or plane.const not in (0, strip.plane):
        raise NotImplementedError(
            f"an access to {strip.operand!r} sits {plane} planes from the one being read; the "
            f"swap pair holds two planes; {_LATER}")
    out = tuple(_add(_substitute(_resolve(subscripts[dim], bindings), elements),
                     _scale(origins[k], -1))
                for k, dim in enumerate(strip.dims))
    return (destination if plane.const else source), out


def halo(mapping: LegalMapping,
         delivery: tuple[tuple[str, str, str | None, bool], ...], herd: HerdPlan,
         buffers: tuple[BufferPlan, ...], loops: tuple[tuple[str, str], ...]
         ) -> tuple[tuple[ChannelPlan, ...], tuple[Any, ...], tuple[Any, ...]]:
    """`(channels, segment_body, herd_body)` — the halo exchange of LLD §3.6.1 and §6.3.

    Four channels: the two halo bundles at one index per **physical link**, the L3 staging fill
    and the L3 drain. The site order inside a timestep is **not negotiable** — put north, put
    south, get north ghost, get south ghost, both puts `is_async`, both gets `depends_on = ()` —
    because a token from a put into its own get would serialise the exchange into a rendezvous
    and reintroduce the deadlock VF §C's E1 verdict cleared it of.

    **The pair is seeded, not just `cur`.** §3.6.1 stages plane `lo` into `cur`; §6.3's coverage
    paragraph then says of the drain that "the value written back is the one that was staged
    in", which is a claim about *every* plane and so about *both* buffers — the drain puts the
    strip whole, so the read-only boundary columns and the domain-edge ghost rows of whichever
    buffer holds plane `t+1` are written back to L3. `next` is never a `get` target, so a
    `StoreNode` copy of the staged strip is what gives it those values; without it the boundary
    of every second plane is whatever the alloc happened to hold, and the Dirichlet boundary is
    not read-only at all. Recorded in `design/PROGRESS-B.md`, phase P5.
    """
    strip = strip_geometry(mapping, delivery, herd)
    tensor = next(t for t in tensor_plan(mapping) if t.name == strip.operand)
    strides = _row_major(tensor.shape)
    coord = herd.coords[0]
    bundle = bundle_name(mapping, 0)
    drain = row_drain_name(strip.time_axis)
    fill_name = staging_channel_name(strip.operand, "In")
    out_name = staging_channel_name(strip.operand, "Out")
    names = compute_names(mapping)
    bindings = dict(mapping.kernel.bindings)
    statement = _statement(mapping)
    slab = _origins(mapping, lambda d: herd.coords[d])
    origins = tuple(_add(slab[axis], _const(-width))
                    for axis, width in zip(strip.axes, strip.halos))
    local = _row_major(strip.shape)
    exchange_k = strip.dims.index(strip.exchange_dim)

    def band(start: int, size: int) -> Region:
        """One slab of the L1 strip: `size` rows at `start` along the exchanged dim."""
        offsets = [ZERO] * len(strip.shape)
        sizes = list(strip.shape)
        offsets[exchange_k], sizes[exchange_k] = _const(start), size
        return Region(offsets=tuple(offsets), sizes=tuple(sizes), strides=local)

    def l3(plane: Expr, ghost: bool) -> Region:
        """One strip of the L3 tensor at `plane`, for the PE the bundle loop is standing on.

        `ghost` is the fill: the staged slab reaches `halo` elements outside the owned box on
        **every** dim, which is what gives `t = lo`'s first update its boundary values with no
        prologue put. The drain is the same box narrowed to the owned extent along the exchanged
        dim only — the strip is put back whole, so the read-only columns travel with it (§6.3's
        coverage paragraph).
        """
        base = _origins(mapping, lambda _d: bundle)
        offsets: list[Any] = [ZERO] * len(tensor.shape)
        sizes = [1] * len(tensor.shape)
        offsets[strip.time_dim] = plane
        for k, dim in enumerate(strip.dims):
            owned = not ghost and k == exchange_k
            offsets[dim] = base[strip.axes[k]] if owned else _add(base[strip.axes[k]],
                                                                  _const(-strip.halos[k]))
            sizes[dim] = strip.owned if owned else strip.shape[k]
        return Region(offsets=tuple(offsets), sizes=tuple(sizes), strides=strides)

    def step(index: Expr, phase: int, order: int) -> tuple[Any, ...]:
        """One timestep: put both boundaries, get both ghosts, update, drain the new plane."""
        source, destination = (CUR, NEXT) if phase == 0 else (NEXT, CUR)
        elements = {axis: (_add(slab[axis], _var(name))
                           if _tile_factor(mapping, axis) is not None else _var(name))
                    for axis, name in names.items()}
        elements[strip.time_axis] = index

        def rewrite(load: Load) -> Load:
            return Load(*_in_strip(strip, elements, origins, load.subscripts, bindings,
                                   source, destination))

        target, subscripts = _in_strip(strip, elements, origins,
                                       kernel_subscripts(mapping, statement.target), bindings,
                                       source, destination)
        body: tuple[Any, ...] = (StoreNode(buffer_id=target, subscripts=subscripts,
                                           expr=_rewrite_loads(statement.expr, rewrite)),)
        nest = [axis for axis in statement.axes if axis != strip.time_axis]
        for position in reversed(range(len(nest))):
            axis, name = nest[position], names[nest[position]]
            if _tile_factor(mapping, axis) is not None:
                low, high = ZERO, _const(_tile_extent(mapping, axis))
            else:
                kernel_axis = _kernel_axis(mapping, axis)
                low, high = (_resolve(kernel_axis.lo, bindings),
                             _resolve(kernel_axis.hi, bindings))
            body = (LoopPlan(axis=name, lo=low, hi=high,
                             step=_resolve(_kernel_axis(mapping, axis).step, bindings),
                             kind=_kind(loops, name), depth=1 + position, body=body),)
        return (
            _site(strip.north, "put", order, "herd", indices=(_var(coord, 1, -1),),
                  buffer=source, region=band(strip.halo, strip.halo),
                  guard=Guard(coord=coord, relation=">", value=ZERO), is_async=True),
            _site(strip.south, "put", order + 1, "herd", indices=(_var(coord),),
                  buffer=source, region=band(strip.owned, strip.halo),
                  guard=Guard(coord=coord, relation="<", value=_const(strip.pes - 1)),
                  is_async=True),
            _site(strip.north, "get", order + 2, "herd", indices=(_var(coord),),
                  buffer=source, region=band(strip.owned + strip.halo, strip.halo),
                  guard=Guard(coord=coord, relation="<", value=_const(strip.pes - 1))),
            _site(strip.south, "get", order + 3, "herd", indices=(_var(coord, 1, -1),),
                  buffer=source, region=band(0, strip.halo),
                  guard=Guard(coord=coord, relation=">", value=ZERO)),
            body[0],
            _site(out_name, "put", order + 5, "herd", indices=(_var(coord),),
                  buffer=destination, region=band(strip.halo, strip.owned)),
        )

    segment: list[Any] = [_bundle_nest(
        mapping, herd.grid, lambda d: bundle_name(mapping, d), loops, 0,
        (_site(fill_name, "put", 0, "segment", indices=(_var(bundle),), buffer=strip.operand,
               region=l3(_const(strip.lo), True)),))[0]]
    segment.append(herd)
    segment.append(_bundle_nest(
        mapping, herd.grid, lambda d: bundle_name(mapping, d), loops, 0,
        (LoopPlan(axis=drain, lo=_const(strip.lo), hi=_const(strip.hi), step=ONE,
                  kind=_kind(loops, drain), depth=1,
                  body=(_site(out_name, "get", 0, "segment", indices=(_var(bundle),),
                              buffer=strip.operand,
                              region=l3(_var(drain, 1, strip.plane), False)),)),))[0])

    herd_body: list[Any] = list(buffers)
    herd_body.append(_site(fill_name, "get", len(herd_body), "herd", indices=(_var(coord),),
                           buffer=CUR, region=EMPTY_REGION))
    copy: tuple[Any, ...] = (StoreNode(buffer_id=NEXT,
                                       subscripts=tuple(_var(names[a]) for a in strip.axes),
                                       expr=Load(CUR, tuple(_var(names[a])
                                                            for a in strip.axes))),)
    for position in reversed(range(len(strip.axes))):
        name = names[strip.axes[position]]
        copy = (LoopPlan(axis=name, lo=ZERO, hi=_const(strip.shape[position]), step=ONE,
                         kind=_kind(loops, name), depth=position, body=copy),)
    herd_body.append(copy[0])
    steps, _phase = swap_loop(strip.time_axis, strip.lo, strip.hi,
                             _kind(loops, strip.time_axis), 0, step, base=len(herd_body))
    herd_body += list(steps)

    geometry = {fill_name: (herd.grid, strip.dtype), out_name: (herd.grid, strip.dtype),
                strip.north: ((strip.pes - 1,), strip.dtype),
                strip.south: ((strip.pes - 1,), strip.dtype)}
    by_channel: dict[str, list[ChannelSite]] = {}
    for node in list(_flatten(tuple(segment))) + list(_flatten(tuple(herd_body))):
        if isinstance(node, ChannelSite):
            by_channel.setdefault(node.channel, []).append(node)
    channels = tuple(sorted(
        (ChannelPlan(name=name, size=size, broadcast_shape=None, channel_type=None,
                     chain_direction=None, dtype=dtype, sites=tuple(by_channel[name]))
         for name, (size, dtype) in geometry.items()), key=lambda c: c.name))
    return channels, tuple(segment), tuple(herd_body)


# --------------------------------------------------------------------------------------------
# §3.6.3 — the cascade chain for a spatial reduction (FR-M6) — W1-flip
# --------------------------------------------------------------------------------------------

RECV = "recv"
"""§6.2's cascade receive tile: the one protocol buffer the chain adds beside the accumulator.

It carries no `operand`, so `06-interfaces.md` §5.6 invariant 5 charges it *outside*
`LegalMapping.l1_bytes` — which is why M3's figure for the flip is `16 384` and the plan's is
`24 576` (`02-hld.md` §7)."""

_REDUCE = {"+": "+", "*": "*", "max": "maximum", "min": "minimum", "maximum": "maximum",
           "minimum": "minimum"}
"""`ScheduleModel.reductions`' operator spelling → the `ExprNode` the accumulate builds."""


class Chain(NamedTuple):
    """The cascade's geometry: which PE line carries it, which way, and its two ends."""

    axis: int
    """The PE dim carrying the `r_space` basis vector (LLD §3.6.3 line 2)."""
    coord: str
    """The herd coordinate on that dim — `tx` on a 1-D herd, `ty` on the 2-D variant."""
    pes: int
    """`herd.grid[axis]`: `PK`, so the chain has `PK - 1` links."""
    direction: str
    """`"ascending"` | `"descending"` — `ChannelPlan.chain_direction`, never an emitter choice."""
    head: int
    """The coordinate that only puts: `0` ascending, `PK-1` descending."""
    tail: int
    """The coordinate that only drains to L3: `PK-1` ascending, `0` descending."""
    put_index: Expr
    """The link a PE puts into: `coord` ascending, `coord - 1` descending."""
    get_index: Expr
    """The link a PE gets from: `coord - 1` ascending, `coord` descending."""


def cascade_channel_name(mapping: LegalMapping, axis: int) -> str:
    """`f"Cascade{axis_name(c).upper()}"` (LLD §3.6.3 line 8) — the flip's `CascadeK`."""
    return f"Cascade{_root(mapping, mapping.schedule.place[axis]).upper()}"


def chain_geometry(mapping: LegalMapping, herd: HerdPlan) -> Chain:
    """The orientation table of LLD §3.6.3 lines 4-7 and 15-18, measured on both sides.

    **The direction is forced by `aie.cascade_flow`'s verifier** — *source tile must be to the
    North or West of the destination tile* — and both cases are measured on the pinned wheel
    (REVIEW-round1 P-R3, `03-lld-B-open-questions.md` §4):

    * a **1-D** `grid(4)` herd places as four columns on one row, so the chain must run west to
      east: `aie.cascade_flow(%tile_0_2, %tile_1_2)`, `(1,2)→(2,2)`, `(2,2)→(3,2)`, `aircc`
      exit 0 — and the descending form of the same module fails the verifier;
    * a **2-D** `(1,4)` herd is one column of four rows and must descend, which is the
      opposite case rather than a contradiction.

    It is recorded as `ChannelPlan.chain_direction` because M5 has no rule that could recompute
    it and D-14 forbids it trying.
    """
    axis = _cascade_axis(mapping)
    pes = herd.grid[axis]
    coord = herd.coords[axis]
    if pes < 2:
        raise NotImplementedError(
            f"the cascade axis {PE_AXIS_NAME[axis]} has extent {pes}; a chain needs at least "
            f"two PEs to have a link; {_LATER}")
    if len(herd.grid) == 1:
        return Chain(axis, coord, pes, "ascending", 0, pes - 1,
                     _var(coord), _var(coord, 1, -1))
    return Chain(axis, coord, pes, "descending", pes - 1, 0,
                 _var(coord, 1, -1), _var(coord))


def cascade_buffers(mapping: LegalMapping,
                    delivery: tuple[tuple[str, str, str | None, bool], ...],
                    levels: dict[str, str]) -> tuple[BufferPlan, ...]:
    """The staged tiles plus `recv`, in the order §6.2's herd body allocates them.

    `PROTOCOL_BUFFERS` (LLD §3.3 line 22) appends; §6.2's body allocates `b`, `acc`, `recv`,
    then `a` **inside** the `i0` loop, and `06-interfaces.md` §5.6 requires `MappingPlan.buffers`
    to be that order. `recv` therefore goes immediately after the accumulator it receives into,
    not at the end: `4096 + 8192 + 8192 + 2·2048 = 24 576` (`02-hld.md` §7).
    """
    out = list(_staged_buffers(mapping, delivery, levels))
    if any(buffer.name == RECV for buffer in out) or any(p.name == RECV
                                                         for p in mapping.kernel.params):
        raise _internal(f"the cascade's protocol buffer {RECV!r} collides with a staged buffer "
                        f"or a kernel parameter of the same name "
                        f"(design/06-interfaces.md §5.6)", buffer=RECV)
    accumulator = next((b for b in out if is_accumulator(mapping, b)), None)
    if accumulator is None:
        raise _internal("the cascade's reduction target is staged in no L1 buffer, so there is "
                        "nothing for the chain to accumulate into",
                        buffers=[b.name for b in out])
    recv = BufferPlan(name=RECV, operand=None, level="L1", scope="herd.private",
                      shape=accumulator.shape, dtype=accumulator.dtype, bytes=accumulator.bytes,
                      loop_depth=0, ping_pong_candidate=False)
    position = out.index(accumulator) + 1
    return tuple(out[:position] + [recv] + out[position:])


def _reduce(op: str, lhs: Any, rhs: Any) -> Any:
    """`acc ⊕ recv` for the declared reduction operator (LLD §3.6.3 line 21, ruling R-F-2).

    The operator comes from `ScheduleModel.reductions` and is never hard-coded: `+` and `*`
    build a `BinOp`, `max`/`min` a `MaxMin`.
    """
    spelling = _REDUCE.get(op)
    if spelling is None:
        raise NotImplementedError(
            f"the reduction operator {op!r} has no cascade accumulate form; {_LATER}")
    if spelling in ("+", "*"):
        return BinOp(op=spelling, lhs=lhs, rhs=rhs)
    return MaxMin(op=spelling, operands=(lhs, rhs))


def _reduction_op(mapping: LegalMapping, chain: Chain) -> str:
    """The operator declared for the axis the chain carries (`ScheduleModel.reductions`)."""
    axis = _root(mapping, mapping.schedule.place[chain.axis])
    declared = dict(mapping.schedule.reductions).get(axis)
    if declared is not None:
        return declared
    reduction = mapping.kernel.reduction
    if reduction is not None and reduction.op is not None:
        return reduction.op
    raise _internal(
        f"no associative/commutative operator is declared for the cascade axis {axis!r}; M3 "
        f"raises RSPACE-NO-AC-OP before M4 sees it (design/06-interfaces.md §4.1)", axis=axis)


def cascade(mapping: LegalMapping,
            delivery: tuple[tuple[str, str, str | None, bool], ...], herd: HerdPlan,
            buffers: tuple[BufferPlan, ...], loops: tuple[tuple[str, str], ...]
            ) -> tuple[tuple[ChannelPlan, ...], tuple[Any, ...], tuple[Any, ...]]:
    """`(channels, segment_body, herd_body)` — the cascade chain of LLD §3.6.3 and §6.2.

    The fills and the compute nest are §6.1's, unchanged; what the chain adds is one
    `npu_cascade` bundle of `PK-1` links and the head / middle / tail discrimination of lines
    19-23, built as **nested `BranchNode`s** because `_cond.py:57-60` has no `and` — conjunction
    is nesting. The drain is not the PE grid's: only the tail PE reaches L3, once per trip of
    the temporal tile axis the accumulator moves with (ruling **R-F-1**).

    **Cascade puts and gets consume no DMA channel** — they lower to
    `aie.put_cascade`/`aie.get_cascade` (`AIRToAIEPass.cpp:6955-7023`) — which is why
    `m4_selfcheck.dma_report` skips them and the flip's per-PE counts are 2 inbound and 1
    outbound (§6.2's DMA note, ruling **R-F-4**).
    """
    chain = chain_geometry(mapping, herd)
    bindings = dict(mapping.kernel.bindings)
    statement = _statement(mapping)
    target = statement.target.operand
    fills, drains = _fills_and_drains(mapping, delivery)
    if drains != (target,):
        raise NotImplementedError(
            f"the cascade drains {list(drains)} where the reduction target is {target!r}; this "
            f"cut chains one accumulator; {_LATER}")
    by_operand = {buffer.operand: buffer for buffer in buffers}
    accumulator = by_operand[target]
    names = compute_names(mapping)
    chan = cascade_channel_name(mapping, chain.axis)
    drain_channel = f"{target}2L3"
    drain_size = multicast_geometry(herd.grid, chain.axis)[0]
    drain_axis = streaming_axis(mapping, target)
    chain_root = _root(mapping, mapping.schedule.place[chain.axis])
    if any(_column(_access_matrix(mapping, target), _ucol(mapping, chain_root))):
        raise _internal(
            f"the accumulator {target!r} indexes {chain_root!r}, the axis the cascade reduces "
            f"over, so its tile is not constant along the chain",
            operand=target, axis=chain_root)

    geometry: dict[str, tuple[tuple[int, ...], tuple[int, ...] | None, Dtype, str | None,
                              str | None]] = {
        chan: ((chain.pes - 1,), None, accumulator.dtype, "npu_cascade", chain.direction),
        drain_channel: (drain_size, None, accumulator.dtype, None, None)}

    # ---- the segment body: the fills, the herd marker, the drain ----------------------------
    segment: list[Any] = []
    for operand, kind, along, _declared in sorted(
            (row for row in delivery if row[0] in fills),
            key=lambda row: (depth(mapping, row[0]), row[0])):
        # Resident fills before streamed ones: §6.2 hoists the weights above the `i0` sweep and
        # M5 §6.2 lines 10-14 print `B2L1` first. It is the rule `buffer_plan` already sorts by.
        size, broadcast = (multicast_geometry(herd.grid, PE_AXIS_NAME.index(along))
                           if kind == "MULTICAST" else (herd.grid, None))
        fill = f"{operand}2L1"
        geometry[fill] = (size, broadcast, _param(mapping, operand).dtype, None, None)
        top = (all(extent == 1 for extent in size)
               and streaming_axis(mapping, operand) is None)
        put = _site(fill, "put", len(segment) if top else 0, "segment",
                    indices=tuple(_var(bundle_name(mapping, d)) if extent > 1 else ZERO
                                  for d, extent in enumerate(size)),
                    buffer=operand,
                    region=l3_region(mapping, operand, _fill_origins(mapping, operand, size)))
        segment.append(_fill_loop(mapping, operand, size, loops, put))
    segment.append(herd)

    # `drain_size` is 1 on the chain dim, so `_origins` binds that axis to its own `lo` — which
    # is exactly right: the accumulator's tile is constant along the axis the chain reduces over
    # (asserted above), and the tail PE holds the whole of it.
    origins = _drain_origins(mapping, drain_size)
    if drain_axis is not None:
        origins[_root(mapping, drain_axis)] = _var(row_drain_name(drain_axis))
    drain_get = _site(drain_channel, "get", 0 if drain_axis is not None else len(segment),
                      "segment",
                      indices=tuple(_var(drain_name(mapping, d)) if extent > 1 else ZERO
                                    for d, extent in enumerate(drain_size)),
                      buffer=target, region=l3_region(mapping, target, origins))
    if drain_axis is None:
        segment.append(drain_get)
    else:
        parent = _root(mapping, drain_axis)
        kernel_axis = _kernel_axis(mapping, parent)
        segment.append(_bundle_nest(
            mapping, drain_size, lambda d: drain_name(mapping, d), loops, 0,
            (LoopPlan(axis=row_drain_name(drain_axis),
                      lo=_resolve(kernel_axis.lo, bindings),
                      hi=_resolve(kernel_axis.hi, bindings),
                      step=_const(_tile_factor(mapping, parent)),
                      kind=_kind(loops, row_drain_name(drain_axis)),
                      depth=len([e for e in drain_size if e > 1]), body=(drain_get,)),))[0])

    # ---- the herd body: the resident tiles, then the `i0` sweep -----------------------------
    herd_body: list[Any] = []
    streamed = [b for b in buffers if b.loop_depth >= 1]
    for buffer in buffers:
        if buffer.loop_depth >= 1:
            continue
        herd_body.append(buffer)
        if buffer.operand in fills:
            herd_body.append(_site(f"{buffer.operand}2L1", "get", len(herd_body), "herd",
                                   indices=tuple(_var(name) for name in herd.coords),
                                   buffer=buffer.name, region=EMPTY_REGION))

    axis = streaming_axis(mapping, streamed[0].operand) if streamed else None
    if any(streaming_axis(mapping, b.operand) != axis for b in streamed):
        raise NotImplementedError(
            f"the streamed buffers {[b.name for b in streamed]} are re-fetched per different "
            f"axes, so the herd body needs more than one streaming loop; {_LATER}")
    elements, slab = compute_frame(mapping, herd, axis, names)
    _, stored = _in_l1(mapping, by_operand, elements, slab, target,
                       kernel_subscripts(mapping, statement.target))

    nest_depth = 1 if streamed else 0
    base = 0 if streamed else len(herd_body)
    body: list[Any] = list(streamed)
    for buffer in streamed:
        if buffer.operand in fills:
            body.append(_site(f"{buffer.operand}2L1", "get", base + len(body), "herd",
                              indices=tuple(_var(name) for name in herd.coords),
                              buffer=buffer.name, region=EMPTY_REGION))
    body.append(_zero_nest(mapping, accumulator, names, loops, nest_depth, stored))
    body.append(_compute_nest(mapping, by_operand, names, loops, nest_depth, elements, slab,
                              stored))

    # Lines 19-23: the head puts; everyone else gets, accumulates, and either drains (the tail)
    # or puts on. The four sites are numbered from the branch's own index in its body, which is
    # what keeps every `ChannelSite.id` distinct when one body holds two arms (phase P4's
    # reading 3, extended to a nest).
    order = base + len(body)
    chain_put = _site(chan, "put", order, "herd", indices=(chain.put_index,),
                      buffer=accumulator.name, region=EMPTY_REGION)
    accumulate = _zero_nest(
        mapping, accumulator, names, loops, nest_depth, stored,
        _reduce(_reduction_op(mapping, chain), Load(accumulator.name, stored),
                Load(RECV, stored)))
    body.append(BranchNode(
        predicate=Guard(coord=chain.coord, relation="==", value=_const(chain.head)),
        then=(chain_put,),
        otherwise=(_site(chan, "get", order + 1, "herd", indices=(chain.get_index,),
                         buffer=RECV, region=EMPTY_REGION),
                   accumulate,
                   BranchNode(
                       predicate=Guard(coord=chain.coord, relation="==",
                                       value=_const(chain.tail)),
                       then=(_site(drain_channel, "put", order + 2, "herd",
                                   indices=tuple(ZERO if d == chain.axis else _var(name)
                                                 for d, name in enumerate(herd.coords)),
                                   buffer=accumulator.name, region=EMPTY_REGION),),
                       otherwise=(replace(chain_put, id=f"{chan}.put.{order + 3}@herd",
                                          order=order + 3),)))))
    if streamed:
        herd_body.append(_tile_loop(mapping, axis, loops, 0, tuple(body)))
    else:
        herd_body += body

    by_channel: dict[str, list[ChannelSite]] = {}
    for node in list(_flatten(tuple(segment))) + list(_flatten(tuple(herd_body))):
        if isinstance(node, ChannelSite):
            by_channel.setdefault(node.channel, []).append(node)
    channels = tuple(sorted(
        (ChannelPlan(name=name, size=size, broadcast_shape=broadcast, channel_type=kind,
                     chain_direction=direction, dtype=dtype, sites=tuple(by_channel[name]))
         for name, (size, broadcast, dtype, kind, direction) in geometry.items()),
        key=lambda c: c.name))
    return channels, tuple(segment), tuple(herd_body)


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
    # R-L1-3: the `x2 (ping-pong)` annotations above stay a property of the buffer; the repeat
    # loop's unroll is a property of the herd, already printed on the `herd:` line, and it is
    # the `L1:` total that has to agree with M3 and with the lowered IR.
    total = l1_total(buffers, repeated=any(r > 1 for r in mapping.repeats))
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


def l1_total(buffers: tuple[BufferPlan, ...], repeated: bool = False) -> int:
    """`sum(bytes × (2 if ping_pong_candidate else 1))` (§5.6 invariant 5, LLD §3.3 note 4).

    `repeated` is architect ruling **R-L1-3** (2026-09-15): when any repeat factor exceeds 1
    the whole herd body sits in the repeat loop, which the ping-pong machinery unrolls by 2
    (mlir-air `AIRDependencyScheduleOpt.cpp:1906-1908`), so **every** buffer is allocated
    twice — and the ping-pong pair is not doubled a second time. It is the same figure M3
    charges in `LegalMapping.l1_bytes` (`03-lld-M3-checker.md` §3.10).
    """
    return sum(b.bytes * (2 if repeated or b.ping_pong_candidate else 1)
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
        loops = loop_plan(mapping, delivery, buffers)
        channels = channel_plan(mapping, delivery, herd, buffers, loops)
        segment_body, herd_body = protocol(mapping, delivery, herd, buffers, channels, loops)
        _check_l1(mapping, buffers)
        _check_allocation_order(buffers, herd_body)
        out = MappingPlan(mapping=mapping, tensors=tensors, launch_name=mapping.kernel.name,
                          segment_name=f"{mapping.kernel.name}_seg", herd=herd, buffers=buffers,
                          channels=channels, segment_body=segment_body, herd_body=herd_body,
                          delivery=delivery,
                          summary=summary(mapping, delivery, herd, buffers, channels))
        self_check(out)
        # §3.8: "a DMA warning is recorded in the summary and printed". It is a property of the
        # finished plan — `warnings` enumerates every herd coordinate — so the summary is built
        # first and the lines appended here, rather than §3.1's pass order being changed to
        # compute the P3 report twice (recorded in design/PROGRESS-B.md, phase P4).
        out = _with_warnings(out)
    except SpatialError:
        raise
    except Exception as exc:                                  # noqa: BLE001 — NFR-7, LLD §5
        raise _fail(
            f"the mapping of {mapping.kernel.name!r} could not be planned: "
            f"{type(exc).__name__}: {exc}",
            _BUG_FIX, internal_exception=f"{type(exc).__name__}: {exc}",
            workload=mapping.kernel.name) from None
    return out


def _with_warnings(out: MappingPlan) -> MappingPlan:
    """Append one `warning:` line per P3 warning to the summary (LLD §3.8, §3.9)."""
    lines = tuple(f"warning: {text}" for text in warnings(out))
    if not lines:
        return out
    return replace(out, summary=replace(out.summary, lines=out.summary.lines + lines))


def _check_l1(mapping: LegalMapping, buffers: tuple[BufferPlan, ...]) -> None:
    """The staged buffers must charge exactly what M3 charged (LLD §3.3 note 4, §7).

    Both totals carry R-L1-3's repeat factor, so M4's staged figure and M3's `l1_bytes` are
    the same arithmetic on the same buffers.
    """
    repeated = any(r > 1 for r in mapping.repeats)
    staged = l1_total(tuple(b for b in buffers if b.operand is not None), repeated=repeated)
    if staged != mapping.l1_bytes:
        raise _internal(
            f"the plan charges {staged} B for the staged buffers where LegalMapping.l1_bytes "
            f"says {mapping.l1_bytes} B (design/03-lld-M4-mapping.md §3.3 note 4)",
            plan_l1_bytes=staged, mapping_l1_bytes=mapping.l1_bytes)
    total = l1_total(buffers, repeated=repeated)
    if total > L1_USABLE:
        raise _internal(
            f"the plan's L1 total {total} B exceeds the {L1_USABLE} B budget "
            f"(design/06-interfaces.md §5.6 invariant 5)",
            plan_l1_bytes=total, l1_budget=L1_USABLE)


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
    """Every node of a body, descending into loop bodies and both arms of a branch."""
    for node in nodes:
        yield node
        if isinstance(node, LoopPlan):
            yield from _flatten(node.body)
        elif isinstance(node, BranchNode):
            yield from _flatten(node.then)
            yield from _flatten(node.otherwise)
