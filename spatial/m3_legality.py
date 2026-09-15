"""M3 -- legality checker. Owner: Person A. LLD: design/03-lld-M3-checker.md.

**This module is the entry's differentiator (G2).** `check(kernel, schedule)` builds the
spatial map (Sigma, Pi) from the schedule, runs eleven checks in a fixed order (design doc
Sec 3.14), and returns a `LegalMapping` or raises the first `LegalityError`.

Two coordinate frames are used throughout (Sec 3.1): `Coord`, the post-tiling axis order (what
`sigma`/`pi` are matrices over), and `UCoord`, the kernel's own untiled axis order (what
`r_time`/`r_space`/access-map matrices/dependence vectors are over).
"""

from __future__ import annotations

import itertools
from typing import Any

from spatial import intlin
from spatial.model import Axis, Diagnostic, Expr, KernelModel, LegalMapping, LegalityError, ScheduleModel

_CLAUSE_UNKNOWN = "(schedule)"


def _fail(code: str, reason: str, fix: str, clause: str, **details: Any) -> LegalityError:
    return LegalityError(Diagnostic(code=code, stage="legality", clause=clause, reason=reason,
                                    fix=fix, location=None, details=details))


# ------------------------------------------------------------------------------------------
# Sec 3.3 -- frames: Coord, UCoord, the tile lattice, sigma, pi
# ------------------------------------------------------------------------------------------


class _Frames:
    def __init__(self, kernel: KernelModel, schedule: ScheduleModel) -> None:
        self.kernel = kernel
        self.schedule = schedule
        self.ucoord = tuple(a.name for a in kernel.axes)
        self.kernel_extent = {a.name: a.extent for a in kernel.axes}
        # created_from[child] = (parent, factor, is_outer)
        self.created_from: dict[str, tuple[str, int, bool]] = {}
        self.children: dict[str, tuple[str, str]] = {}
        for axis, factor in schedule.tiles:
            outer, inner = axis + "0", axis + "1"
            self.children[axis] = (outer, inner)
            self.created_from[outer] = (axis, factor, True)
            self.created_from[inner] = (axis, factor, False)

        def leaves(name: str) -> list[str]:
            if name in self.children:
                o, i = self.children[name]
                return leaves(o) + leaves(i)
            return [name]

        coord: list[str] = []
        for a in kernel.axes:
            coord += leaves(a.name)
        self.coord = tuple(coord)

        self._extent_memo: dict[str, int] = {}

    def extent(self, name: str) -> int:
        if name in self._extent_memo:
            return self._extent_memo[name]
        if name in self.created_from:
            parent, factor, is_outer = self.created_from[name]
            pe = self.extent(parent)
            val = (pe // factor) if is_outer else factor
        else:
            val = self.kernel_extent[name]
        self._extent_memo[name] = val
        return val

    def root(self, name: str) -> str:
        while name in self.created_from:
            name = self.created_from[name][0]
        return name

    def is_outer_or_untiled(self, name: str) -> bool:
        return name not in self.created_from or self.created_from[name][2]

    def default_sigma_order(self) -> tuple[str, ...]:
        outer_or_untiled = [c for c in self.coord if self.is_outer_or_untiled(c)]
        inner = [c for c in self.coord if not self.is_outer_or_untiled(c)]
        return tuple(outer_or_untiled + inner)

    def _row(self, name: str, frame: tuple[str, ...]) -> tuple[int, ...]:
        v = [0] * len(frame)
        v[frame.index(name)] = 1
        return tuple(v)

    def lift_row(self, ucoord_row: tuple[int, ...]) -> tuple[int, ...]:
        """A UCoord coefficient row -> the equivalent row over Coord (Sec 3.9's tile split:
        an untiled coeff `c` on axis `a` becomes `c*factor` on the outer handle and `c` on the
        inner handle, since `a = outer*factor + inner`)."""
        out = {name: 0 for name in self.coord}
        for name, c in zip(self.ucoord, ucoord_row):
            if c == 0:
                continue
            if name in self.children:
                outer, inner = self.children[name]
                factor = next(f for ax, f in self.schedule.tiles if ax == name)
                out[outer] += c * factor
                out[inner] += c
            else:
                out[name] += c
        return tuple(out[n] for n in self.coord)


def _build(kernel: KernelModel, schedule: ScheduleModel) -> tuple[_Frames, list, list, list]:
    f = _Frames(kernel, schedule)
    if schedule.grid is None:
        if schedule.place:
            raise _fail("PLACE-EXTENT", "place() was declared without grid()",
                       "declare grid(...) before place(...)", _CLAUSE_UNKNOWN)
        grid: tuple[int, ...] = ()
    else:
        grid = schedule.grid
    if len(grid) > 2:
        raise _fail("HERD-RANK", f"grid of rank {len(grid)} is not supported",
                   "use grid(PI) or grid(PI, PJ)", "grid(...)", rank=len(grid))
    if len(schedule.place) != len(grid):
        # M2 checks this at clause time (`CLAUSE-RANK`), so the surface cannot get here; a
        # `ScheduleModel` built by hand can, and NFR-7 says M3's entry point answers with a
        # diagnostic rather than an `AssertionError` (erratum, 2026-09-15).
        raise _fail("PLACE-EXTENT",
                   f"place has rank {len(schedule.place)} but grid has rank {len(grid)}",
                   "pass one placed axis per grid extent", "place(...)",
                   place=list(schedule.place), grid=list(grid))
    for r, name in enumerate(schedule.place):
        if f.extent(name) != grid[r]:
            raise _fail("PLACE-EXTENT",
                       f"axis {name} has extent {f.extent(name)} but grid position {r} is "
                       f"{grid[r]}", "match the placed axis's extent to the grid extent",
                       f"place(...)", axis=name, axis_extent=f.extent(name), grid_extent=grid[r])
    for name in schedule.sequential:
        root = f.root(name)
        if name in schedule.place or root in schedule.place:
            raise _fail("PLACE-SEQUENTIAL-CONFLICT",
                       f"axis {name} is both placed and sequential",
                       "an axis cannot be both spatial and temporal", f"sequential(ax.{name})",
                       axis=name)

    pi = [f._row(name, f.coord) for name in schedule.place]
    pi_u = [f._row(f.root(name), f.ucoord) for name in schedule.place]
    if schedule.skew is not None:
        row0 = [0] * len(f.coord)
        for name in schedule.skew:
            row0[f.coord.index(name)] += 1
        rest_names = [a for a in f.default_sigma_order()
                     if a not in schedule.skew and a not in schedule.place]
        sigma = [tuple(row0)] + [f._row(a, f.coord) for a in rest_names]
    else:
        sigma = [f._row(a, f.coord) for a in f.default_sigma_order()]
    return f, sigma, pi, pi_u


# ------------------------------------------------------------------------------------------
# Sec 3.4 (L1), Sec 3.5 (L2)
# ------------------------------------------------------------------------------------------


def _check_l1(sigma: tuple, pi: tuple) -> None:
    ker_s = intlin.kernel_basis(sigma)
    ker_p = intlin.kernel_basis(pi)
    conflict = intlin.intersect(ker_s, ker_p)
    if conflict:
        v = conflict[0]
        raise _fail("L1-CONFLICT",
                   f"iterations that differ by {v} get the same time and the same PE",
                   "give the conflicting axes a time of their own: add them to "
                   "skew(time=...), or place one of them", "skew(...)/place(...)",
                   conflict_basis=[list(x) for x in conflict], ker_sigma=[list(x) for x in ker_s],
                   ker_pi=[list(x) for x in ker_p])


def _lift(f: _Frames, vector: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Sec 3.5 LIFT: a UCoord dependence vector -> its <= 2^(#tiled axes) representatives
    over Coord (interior: stays inside one tile; boundary: crosses a tile edge)."""
    reps = [{}]
    for name, delta in zip(f.ucoord, vector):
        if delta == 0:
            continue
        if name in f.children:
            outer, inner = f.children[name]
            factor = next(fa for ax, fa in f.schedule.tiles if ax == name)
            sign = 1 if delta > 0 else -1
            interior = {outer: 0, inner: delta}
            boundary = {outer: sign, inner: delta - sign * factor}
            new_reps = []
            for r in reps:
                a = dict(r); a[outer] = a.get(outer, 0) + interior[outer]
                a[inner] = a.get(inner, 0) + interior[inner]
                b = dict(r); b[outer] = b.get(outer, 0) + boundary[outer]
                b[inner] = b.get(inner, 0) + boundary[inner]
                new_reps.append(a); new_reps.append(b)
            reps = new_reps
        else:
            for r in reps:
                r[name] = r.get(name, 0) + delta
    return [tuple(r.get(c, 0) for c in f.coord) for r in reps]


def _check_l2(f: _Frames, sigma: tuple) -> None:
    for dep in sorted(f.kernel.dependences, key=lambda d: (d.operand, d.vector)):
        for rep in _lift(f, dep.vector):
            t = intlin.matvec(sigma, rep)
            if not intlin.lexpos(t):
                raise _fail("L2-CAUSALITY",
                           f"dependence {dep.vector} on {dep.operand} is not causal: "
                           f"sigma.d = {t}, whose first nonzero entry is <= 0",
                           "add an axis to skew(time=...) before the violating one, or drop "
                           "the offending tile", "skew(...)", dependence=list(dep.vector),
                           representative=list(rep), sigma_d=list(t), operand=dep.operand)


# ------------------------------------------------------------------------------------------
# Sec 3.6 -- stationarity
# ------------------------------------------------------------------------------------------


def _operand_matrix(kernel: KernelModel, operand: str) -> tuple[tuple[int, ...], ...]:
    for st in kernel.statements:
        if st.target.operand == operand:
            return st.target.matrix
        for r in st.reads:
            if r.operand == operand:
                return r.matrix
    # Reachable from a user path: a parameter that the loop body never reads or writes has no
    # access matrix, so it has no reuse space to check. `STATIONARITY` is the closest code in
    # the frozen 43-entry catalogue (06-interfaces.md §6.3) -- it is §3.6's own code, and §3.6
    # is the only check that asks every parameter for its matrix. M4 treats the same shape as
    # its own defect (`m4_mapping._internal`, ruling B-P23) because by then M3 owed it.
    raise _fail("STATIONARITY",
               f"operand {operand} is declared in the kernel signature but is never read or "
               f"written, so it has no access matrix and no reuse space",
               f"remove {operand} from the kernel signature, or access it in the loop body",
               "(kernel signature)", operand=operand)


def _check_stationarity(kernel: KernelModel, schedule: ScheduleModel,
                        pi_u: tuple) -> tuple[str, ...]:
    ker_pi_u = intlin.kernel_basis(tuple(pi_u))
    report: dict[str, bool] = {}
    for p in sorted(kernel.params, key=lambda x: x.name):
        m_a = _operand_matrix(kernel, p.name)
        l_a = intlin.kernel_basis(m_a)
        report[p.name] = intlin.contains(ker_pi_u, l_a)
    for a in schedule.stationary:
        if not report.get(a, False):
            m_a = _operand_matrix(kernel, a)
            l_a = intlin.kernel_basis(m_a)
            raise _fail("STATIONARITY",
                       f"ker M_{a} = {l_a} is not contained in ker Spi = {ker_pi_u}",
                       f"place the axes that would kill the reuse direction", f"stationary"
                       f"({a!r})", operand=a, ker_M=[list(x) for x in l_a],
                       ker_pi=[list(x) for x in ker_pi_u], placed=list(schedule.place))
    return tuple(sorted(a for a, ok in report.items() if ok))


# ------------------------------------------------------------------------------------------
# Sec 3.7 -- reduction split and the A/C requirement
# ------------------------------------------------------------------------------------------


def _unit(f: _Frames, name: str) -> tuple[int, ...]:
    """The UCoord unit vector of `name`'s **root** axis.

    `R = ker Sf` is a UCoord space, so a tile handle has no column of its own there: it is the
    axis it was strip-mined from. Indexing `ucoord` with the handle itself raised a bare
    `ValueError` out of `tuple.index` for `reduce(ax.k0, ...)` -- a user path, so NFR-7 says
    that must be a diagnostic (erratum, 2026-09-15). `root` is the identity on an untiled name,
    so no accepted schedule changes.
    """
    v = [0] * len(f.ucoord)
    v[f.ucoord.index(f.root(name))] = 1
    return tuple(v)


def _check_reduction(f: _Frames, pi_u: tuple) -> tuple[tuple, tuple]:
    kernel, schedule = f.kernel, f.schedule
    if kernel.reduction is None:
        if schedule.reductions:
            axis, _ = schedule.reductions[0]
            raise _fail("REDUCE-NOT-ACCUMULATED",
                       f"axis {axis} is tagged as a reduction axis, but the kernel has no "
                       f"accumulation at all", "remove reduce(...), or use += in the kernel",
                       f"reduce(ax.{axis}, ...)", axis=axis)
        return (), ()
    r = kernel.reduction.space
    for axis, _op in schedule.reductions:
        e = _unit(f, axis)
        if not intlin.contains(r, (e,)):
            raise _fail("REDUCE-NOT-ACCUMULATED",
                       f"axis {axis} is not a reduction axis: e_{axis} is not in R = ker Sf "
                       f"= {r}", "reduce the axis whose unit vector is in R",
                       f"reduce(ax.{axis}, ...)", axis=axis, R=[list(x) for x in r],
                       target=kernel.reduction.target)
    ker_pi_u = intlin.kernel_basis(tuple(pi_u))
    r_time = intlin.intersect(r, ker_pi_u)
    r_space = intlin.complement(r, r_time)
    if r_space:
        op = None
        for axis, tagged_op in schedule.reductions:
            if intlin.contains(r_space, (_unit(f, axis),)):
                op = tagged_op
                break
        if op is None:
            op = kernel.reduction.op
        if op not in ("+", "max", "min"):
            raise _fail("RSPACE-NO-AC-OP",
                       f"R_space = {r_space} is non-empty, so partial results are combined "
                       f"across PEs, and that is legal only for an associative and "
                       f"commutative operator; none is declared",
                       'reduce(ax.k, op="+")', "reduce(...)",
                       r_space=[list(x) for x in r_space], r_time=[list(x) for x in r_time],
                       target=kernel.reduction.target, placed=list(schedule.place))
    return r_time, r_space


# ------------------------------------------------------------------------------------------
# Sec 3.8 -- cascade rider
# ------------------------------------------------------------------------------------------


def _check_cascade(f: _Frames, r_space: tuple, pi_u: tuple, physical: tuple,
                   repeats: tuple) -> None:
    if not r_space:
        return
    kernel, schedule = f.kernel, f.schedule
    if len(r_space) != 1:
        raise _fail("CASCADE-RANK", f"R_space has rank {len(r_space)} but a cascade is a "
                   "linear chain", "keep one reduction axis spatial and leave the rest "
                   "temporal", "reduce(...)", rank=len(r_space))
    g = r_space[0]
    carried = [p for p in range(len(pi_u)) if intlin.matvec((pi_u[p],), g)[0] != 0]
    if len(carried) != 1:
        raise _fail("CASCADE-RANK", f"the reduction direction crosses {len(carried)} PE axes",
                   "keep the cascade's carrier on exactly one placed axis", "place(...)",
                   carried=carried)
    p = carried[0]
    if len(schedule.grid) == 2 and schedule.grid[1 - p] != 1:
        raise _fail("CASCADE-RANK",
                   f"a cascade needs a 1-D herd or extent 1 on the other axis; grid is "
                   f"{schedule.grid} with the chain along position {p}",
                   f"grid({schedule.grid[p]}) with place(px=ax.{schedule.place[p]})",
                   "grid(...)", grid=list(schedule.grid), carrier_axis=schedule.place[p])
    target = kernel.reduction.target if kernel.reduction else None
    for s in schedule.streams:
        if s.operand == target and s.pattern == "broadcast":
            raise _fail("CASCADE-BROADCAST",
                       'air.channel does not take broadcast_shape= with '
                       'channel_type="npu_cascade"', f'drop stream("{target}", '
                       f'pattern="broadcast", ...) -- the cascade is the chain',
                       f"stream({target!r}, ...)", operand=target)
    if repeats[p] != 1:
        raise _fail("HERD-PHYSICAL",
                   f"the cascade chain of length {schedule.grid[p]} is folded onto "
                   f"{physical[p]} cores with repeat {repeats[p]}, which breaks the chain",
                   f"use grid({physical[p]}) on this target, or target npu2", "grid(...)",
                   grid=list(schedule.grid), physical_herd=list(physical),
                   repeats=list(repeats))


# ------------------------------------------------------------------------------------------
# Sec 3.9 -- footprint (halo and tile shape)
# ------------------------------------------------------------------------------------------


def _pinned(f: _Frames, skew_pins: bool = True) -> dict[str, bool]:
    """True for a Coord axis that is pinned to a single value within one PE's trip
    (placed, an outer tile handle, sequential, or a skewed-but-unplaced axis).

    `skew_pins=False` drops only the skewed-but-unplaced term. It is used by the L1
    footprint of a **read-only** operand -- architect ruling R-L9-1, 2026-09-15, written
    into design/03-lld-M3-checker.md Sec 3.10: a written operand under a skew is produced
    and forwarded step by step, so only the pinned-axis span is resident, but a read-only
    operand with no dependence on a placed axis is delivered whole by M4's multicast
    (02-hld.md Sec 7.3, FR-M1) -- a per-step stream of it is not a delivery M4 implements."""
    schedule = f.schedule
    pinned = {}
    skewed = set(schedule.skew or ())
    for c in f.coord:
        is_placed = c in schedule.place
        is_outer = c in f.created_from and f.created_from[c][2]
        is_sequential = c in schedule.sequential or f.root(c) in schedule.sequential
        is_skewed_unplaced = (skew_pins and (f.root(c) in skewed or c in skewed)
                              and not is_placed)
        pinned[c] = is_placed or is_outer or is_sequential or is_skewed_unplaced
    return pinned


def _footprint(f: _Frames, operand: str, pinned: dict[str, bool]) -> tuple[list[int], list[tuple[int, int]]]:
    kernel = f.kernel
    accesses = []
    write_access = None
    for st in kernel.statements:
        if st.target.operand == operand:
            accesses.append(st.target)
            write_access = st.target
        for r in st.reads:
            if r.operand == operand:
                accesses.append(r)
    if write_access is None:
        write_access = accesses[0]
    rank = len(write_access.matrix)
    span = []
    overhang = []
    for d in range(rank):
        acc_lo, acc_hi = None, None
        for acc in accesses:
            row = intlin.matvec((acc.matrix[d],), [1] * 0) if False else acc.matrix[d]
            lifted = f.lift_row(row)
            lo = hi = acc.offsets[d].const
            for c, coeff in zip(f.coord, lifted):
                if coeff == 0:
                    continue
                width_lo, width_hi = (0, 0) if pinned[c] else (0, f.extent(c) - 1)
                if coeff >= 0:
                    lo += coeff * width_lo; hi += coeff * width_hi
                else:
                    lo += coeff * width_hi; hi += coeff * width_lo
            acc_lo = lo if acc_lo is None else min(acc_lo, lo)
            acc_hi = hi if acc_hi is None else max(acc_hi, hi)
        w_row = f.lift_row(write_access.matrix[d])
        w_lo = w_hi = write_access.offsets[d].const
        for c, coeff in zip(f.coord, w_row):
            if coeff == 0:
                continue
            width_lo, width_hi = (0, 0) if pinned[c] else (0, f.extent(c) - 1)
            if coeff >= 0:
                w_lo += coeff * width_lo; w_hi += coeff * width_hi
            else:
                w_lo += coeff * width_hi; w_hi += coeff * width_lo
        span.append(acc_hi - acc_lo + 1)
        overhang.append((max(w_lo - acc_lo, 0), max(acc_hi - w_hi, 0)))
    return span, overhang


def _check_halo(f: _Frames) -> tuple[tuple, ...]:
    pinned = _pinned(f)
    out = []
    for w in f.schedule.windows:
        span, overhang = _footprint(f, w.operand, pinned)
        kernel_dims = [p.name for p in f.kernel.params if p.name == w.operand][0]
        param = next(p for p in f.kernel.params if p.name == w.operand)
        # map declared dim positions (axis names in w.dims) to array dims via the write access
        write_access = next(st.target for st in f.kernel.statements if st.target.operand == w.operand)
        derived = []
        for q, axis_name in enumerate(w.dims):
            # find the array dim whose matrix column for this axis is nonzero
            axis_idx = f.ucoord.index(axis_name)
            d = next((dd for dd in range(len(write_access.matrix))
                      if write_access.matrix[dd][axis_idx] != 0), q)
            need = max(overhang[d])
            derived.append(need)
            if w.halo[q] < need:
                raise _fail("HALO-TOO-SMALL",
                           f"the derived footprint of {w.operand} along {axis_name} is "
                           f"{need} but the declared halo is {w.halo[q]}",
                           f'window("{w.operand}", dims=..., halo={need})',
                           f"window({w.operand!r}, ...)", operand=w.operand, dim=axis_name,
                           derived=need, declared=w.halo[q])
        out.append((w.operand, tuple(derived)))
    return tuple(out)


# ------------------------------------------------------------------------------------------
# Sec 3.10 -- L1 capacity;  Sec 3.13 -- ping-pong mode
# ------------------------------------------------------------------------------------------

_DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i8": 1}


def pingpong_mode(f: _Frames, operand: str) -> str:
    """Exported for M4 (design/03-lld-M3-checker.md Sec 3.13)."""
    kernel, schedule = f.kernel, f.schedule
    has_exchange = any(e.operand == operand for e in schedule.exchanges)
    has_window = any(w.operand == operand for w in schedule.windows)
    if has_exchange or has_window:
        return "PAIR"
    param = next(p for p in kernel.params if p.name == operand)
    if param.is_written:
        raise _fail("PINGPONG-SHAPE",
                   f"double_buffer({operand!r}) cannot be realised by the ping-pong pass",
                   f"drop {operand!r} from double_buffer; it is written, so its tile is "
                   f"live on entry to the loop", f"double_buffer({operand!r})",
                   operand=operand, condition=3)
    m_a = _operand_matrix(kernel, operand)
    lifted_cols = [f.lift_row(row) for row in m_a]
    found = False
    for c_idx, c in enumerate(f.coord):
        is_outer = c in f.created_from and f.created_from[c][2]
        if not is_outer:
            continue
        if any(row[c_idx] != 0 for row in lifted_cols):
            found = True
            break
    if not found:
        raise _fail("PINGPONG-SHAPE",
                   f"double_buffer({operand!r}) cannot be realised by the ping-pong pass",
                   f"drop {operand!r} from double_buffer, or stream it along a tiled axis",
                   f"double_buffer({operand!r})", operand=operand, condition=2)
    return "PASS"


def _check_l1_capacity(f: _Frames) -> int:
    total = 0
    breakdown: list[tuple[str, tuple[int, ...], str, int]] = []
    doubled: list[str] = []
    pinned = _pinned(f)
    read_only_pinned = _pinned(f, skew_pins=False)
    explicit_l1 = {name for name, level in f.schedule.residency if level == "L1"}
    # Every operand actually referenced in the herd body needs an L1 buffer to compute
    # against, whether or not reside() names it explicitly (03-lld-M3-checker.md Sec 6.4's
    # worked total counts q/r alongside the explicitly-declared S).
    for param in sorted(f.kernel.params, key=lambda p: p.name):
        name = param.name
        # R-L9-1: a skew pins a written operand's footprint, never a read-only one.
        span, _ = _footprint(f, name, pinned if param.is_written else read_only_pinned)
        nbytes = 1
        for s in span:
            nbytes *= s
        nbytes *= _DTYPE_BYTES[param.dtype.value]
        if name in f.schedule.double_buffer:
            try:
                mode = pingpong_mode(f, name)
            except LegalityError:
                mode = None
            if mode == "PASS":
                nbytes *= 2
                doubled.append(name)
        total += nbytes
        breakdown.append((name, tuple(span), param.dtype.value, nbytes))
    if total > 65536:
        # FR-L9 and Sec 3.10 line 12: the message owes the per-buffer breakdown, in the
        # `(operand, span, dtype, bytes)` order of line 7, and the operands charged twice.
        # "the computed bytes, the budget, and the per-buffer breakdown" is FR-L9's own
        # acceptance; `total` alone does not say which tile to halve.
        raise _fail("L1-CAPACITY",
                   f"the per-core L1 working set is {total} bytes, over the 65536-byte budget",
                   "halve a tile factor, or drop a double_buffer(...) entry",
                   "tile(...)/double_buffer(...)", total=total, budget=65536,
                   per_buffer=[[n, list(s), d, b] for n, s, d, b in breakdown],
                   doubled=doubled)
    return total


# ------------------------------------------------------------------------------------------
# Sec 3.11 -- physical herd resolution
# ------------------------------------------------------------------------------------------

_PHYSICAL_HERD = {"npu1": {1: (4,), 2: (1, 4)}, "npu2": {1: (8,), 2: (2, 4)}}


def _resolve_physical(grid: tuple[int, ...], target: str) -> tuple[tuple, tuple]:
    t = target if target != "auto" else "npu2"
    caps = _PHYSICAL_HERD[t][len(grid)]
    physical = tuple(max(d for d in range(1, g + 1) if g % d == 0 and d <= cap)
                     for g, cap in zip(grid, caps))
    repeats = tuple(g // p for g, p in zip(grid, physical))
    return physical, repeats


# ------------------------------------------------------------------------------------------
# Sec 3.12 -- swap parity
# ------------------------------------------------------------------------------------------


def _check_swap_parity(f: _Frames) -> None:
    pinned = _pinned(f)
    for p in f.kernel.params:
        span, _ = _footprint(f, p.name, pinned)
        # A span >= 2 along a pinned temporal axis is a swap pair (u/v, prev/cur).
        if any(s >= 2 for s in span):
            for c in f.coord:
                is_sequential = c in f.schedule.sequential or f.root(c) in f.schedule.sequential
                is_skewed = (f.root(c) in (f.schedule.skew or ())) and c not in f.schedule.place
                if is_sequential or is_skewed:
                    t = f.extent(c)
                    if t is None or t < 1:
                        raise _fail("SWAP-PARITY",
                                   f"the buffer swap on {p.name} needs a compile-time trip "
                                   f"count of at least 1 on {c}", f"give {c} a constant, "
                                   f"positive extent", "sequential(...)/skew(...)",
                                   operand=p.name, loop=c, trip_count=t)


# ------------------------------------------------------------------------------------------
# Top level -- Sec 3.14 check order
# ------------------------------------------------------------------------------------------


def check(kernel: KernelModel, schedule: ScheduleModel) -> LegalMapping:
    """Verify (sigma, pi) before any IR exists. Raises `LegalityError`."""
    f, sigma, pi, pi_u = _build(kernel, schedule)
    physical, repeats = _resolve_physical(schedule.grid or (), schedule.target)
    _check_l1(tuple(sigma), tuple(pi))
    _check_l2(f, tuple(sigma))
    stationary_ops = _check_stationarity(kernel, schedule, tuple(pi_u))
    r_time, r_space = _check_reduction(f, tuple(pi_u))
    _check_cascade(f, r_space, tuple(pi_u), physical, repeats)
    halo_footprint = _check_halo(f)
    for name in schedule.double_buffer:
        pingpong_mode(f, name)
    l1_bytes = _check_l1_capacity(f)
    _check_swap_parity(f)

    kernel_axis_by_name = {a.name: a for a in kernel.axes}
    axes = []
    for i, c in enumerate(f.coord):
        if c in f.created_from:
            parent, _factor, _is_outer = f.created_from[c]
            axes.append(Axis(name=c, lo=Expr((), 0), hi=Expr((), f.extent(c)), step=Expr((), 1),
                             extent=f.extent(c), parent=parent, depth=i))
        else:
            k = kernel_axis_by_name[c]
            axes.append(Axis(name=c, lo=k.lo, hi=k.hi, step=k.step, extent=k.extent,
                             parent=None, depth=i))
    axes = tuple(axes)
    ker_pi = intlin.kernel_basis(tuple(pi))
    ker_pi_u = intlin.kernel_basis(tuple(pi_u))

    return LegalMapping(
        kernel=kernel, schedule=schedule, axes=axes, sigma=tuple(sigma), pi=tuple(pi),
        ker_pi=ker_pi, pi_u=tuple(pi_u), ker_pi_u=ker_pi_u, r_time=r_time, r_space=r_space,
        stationary_ops=stationary_ops, physical_herd=physical, repeats=repeats,
        l1_bytes=l1_bytes, halo_footprint=halo_footprint,
    )


__all__ = ["check", "pingpong_mode", "intlin"]
