"""M2 -- schedule builder / schedule IR. Owner: Person A. LLD: design/03-lld-M2-schedule.md.

`Schedule` is the clause API (`06-interfaces.md` §7.1): each clause validates its arguments,
records them as pure data, and returns `self` so calls chain (`tile` is the one exception --
it returns the `(outer, inner)` handle pair). `Schedule.model` re-canonicalises the recorded
clauses into a `ScheduleModel` on every access (§3.5): there is exactly one copy of the truth,
the append-only record list, and the model is always a fresh, canonical projection of it.

M2 performs no legality reasoning and does not lower: no `air`, no MLIR, at module scope.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from spatial.model import (
    ClauseError, Delivery, Diagnostic, Direction, ExchangeClause, Level, Pattern, ReduceOp,
    ScheduleModel, StreamClause, Target, WindowClause,
)

if TYPE_CHECKING:
    from spatial.m1_frontend import Kernel

_TARGETS = {"npu1", "npu2", "auto"}
_LEVELS = {"L1", "L2", "L3"}
_REDUCE_OPS = {"+", "max", "min"}
_PATTERNS = {"broadcast", "forward", "cascade"}
_DIRECTIONS = {"W->E", "E->W", "N->S", "S->N"}


def _caller_location() -> tuple[str, int] | None:
    # frame 0 = this function, 1 = the clause method, 2 = the user's call site.
    stack = inspect.stack()
    if len(stack) > 2:
        f = stack[2]
        return (f.filename, f.lineno)
    return None


def _fail(code: str, reason: str, fix: str, clause: str, **details: Any) -> ClauseError:
    return ClauseError(Diagnostic(code=code, stage="clause", clause=clause, reason=reason,
                                  fix=fix, location=_caller_location(), details=details))


class AxisHandle:
    """A resolved axis handle bound to one `Schedule`. Renders as `ax.<name>`."""

    __slots__ = ("name", "_owner")

    def __init__(self, name: str, owner: int) -> None:
        self.name = name
        self._owner = owner

    def __repr__(self) -> str:
        return f"ax.{self.name}"


class Axes:
    """The live handle namespace returned by `Schedule.axes()`; grows as `tile` is called."""

    def __init__(self, schedule: "Schedule") -> None:
        object.__setattr__(self, "_schedule", schedule)

    def __getattr__(self, name: str) -> AxisHandle:
        schedule = object.__getattribute__(self, "_schedule")
        if name not in schedule._axis_table:
            raise AttributeError(name)
        return AxisHandle(name, id(schedule))


class Schedule:
    """The clause API bound to one `Kernel`. Pure data; no `air` import at module scope."""

    def __init__(self, kernel: "Kernel", target: str) -> None:
        if target not in _TARGETS:
            raise _fail("CLAUSE-BAD-ENUM", f"target={target!r} is not a supported target",
                       "use target='npu1', target='npu2', or target='auto'",
                       f"sp.schedule(..., target={target!r})", seen=target,
                       domain=sorted(_TARGETS))
        self.kernel = kernel
        self.target = target
        self._axis_table: dict[str, dict[str, Any]] = {
            a.name: {"parent": None, "factor": None, "extent": a.extent}
            for a in kernel.model.axes
        }
        self._records: list[tuple[str, dict[str, Any]]] = []
        self._ax = Axes(self)

    # -- axis resolution --------------------------------------------------------------------

    def axes(self) -> Axes:
        """The live handle namespace: one attribute per post-tiling axis name."""
        return self._ax

    def _axis(self, ref: object, clause_text: str) -> str:
        if not isinstance(ref, AxisHandle):
            raise _fail("CLAUSE-UNKNOWN-AXIS",
                       f"expected an axis handle, got {type(ref).__name__}", "pass an axis "
                       "handle from Schedule.axes(), e.g. ax.i", clause_text,
                       seen=type(ref).__name__)
        if ref._owner != id(self) or ref.name not in self._axis_table:
            raise _fail("CLAUSE-UNKNOWN-AXIS",
                       f"axis handle '{ref.name}' is not an axis of this schedule",
                       "use one of the axes listed; tile handles are named i0/i1",
                       clause_text, seen=ref.name, axes=sorted(self._axis_table))
        return ref.name

    def _axis_outer_leaf(self, ref: object, clause_text: str) -> str:
        """§3.2 rule 3: a parent handle passed where a spatial axis is wanted normalises to
        its outermost leaf (the handle with the same root that has no parent-of-a-parent)."""
        name = self._axis(ref, clause_text)
        # Descend to the outermost ("0") child if this handle has been tiled.
        while True:
            outer = name + "0"
            if outer in self._axis_table and self._axis_table[outer]["parent"] == name:
                name = outer
            else:
                break
        return name

    def _operand(self, name: object, clause_text: str) -> str:
        params = {p.name for p in self.kernel.model.params}
        if name not in params:
            raise _fail("CLAUSE-UNKNOWN-OPERAND",
                       f"'{name}' is not a parameter of kernel {self.kernel.model.name}",
                       f"name one of {', '.join(sorted(params))}", clause_text,
                       seen=name, params=sorted(params))
        return name  # type: ignore[return-value]

    def _root(self, name: str) -> str:
        while self._axis_table[name]["parent"] is not None:
            name = self._axis_table[name]["parent"]
        return name

    # -- clauses ------------------------------------------------------------------------

    def grid(self, *extents: int) -> "Schedule":
        """Declare the herd's logical shape: `grid(PI)` or `grid(PI, PJ)`."""
        text = f"grid{extents!r}"
        if any(r[0] == "grid" for r in self._records):
            raise _fail("CLAUSE-DUPLICATE", "a second grid() call is not allowed",
                       "declare the grid once", text)
        if not (1 <= len(extents) <= 2):
            raise _fail("CLAUSE-GRID-RANK", f"grid of rank {len(extents)} is not supported",
                       "use grid(PI) or grid(PI, PJ)", text, rank=len(extents))
        for e in extents:
            if not (isinstance(e, int) and not isinstance(e, bool) and e >= 1):
                raise _fail("CLAUSE-BAD-VALUE", f"grid extent {e!r} is not a positive int",
                           "every grid extent must be a positive int", text, seen=e)
        self._records.append(("grid", {"extents": tuple(extents)}))
        return self

    def tile(self, ax: object, factor: int) -> tuple[AxisHandle, AxisHandle]:
        """Split `ax` into `(outer, inner)` with `outer * inner == extent`, `inner == factor`."""
        name = self._axis(ax, f"tile(ax.{getattr(ax, 'name', ax)}, {factor})")
        text = f"tile(ax.{name}, {factor})"
        if not (isinstance(factor, int) and not isinstance(factor, bool) and factor >= 1):
            raise _fail("CLAUSE-BAD-VALUE", f"tile factor {factor!r} is not positive",
                       "use a positive integer factor", text, seen=factor)
        extent = self._axis_table[name]["extent"]
        if extent is None:
            raise _fail("CLAUSE-TILE-DIVIDES",
                       f"axis {name} has no constant extent, so a tile factor cannot be checked",
                       "give the axis a constant, resolvable extent", text, axis=name)
        if extent % factor != 0:
            divisors = [d for d in range(1, extent + 1) if extent % d == 0]
            raise _fail("CLAUSE-TILE-DIVIDES",
                       f"tile factor {factor} does not divide the extent of axis {name}, "
                       f"which is {extent}",
                       f"use a factor from the divisor list, e.g. tile(ax.{name}, "
                       f"{divisors[len(divisors) // 2]})",
                       text, axis=name, extent=extent, factor=factor, divisors=divisors)
        outer, inner = name + "0", name + "1"
        if outer in self._axis_table or inner in self._axis_table:
            raise _fail("CLAUSE-DUPLICATE", f"axis {name} has already been tiled",
                       "tile each axis at most once per name", text, axis=name)
        self._axis_table[outer] = {"parent": name, "factor": factor, "extent": extent // factor}
        self._axis_table[inner] = {"parent": name, "factor": factor, "extent": factor}
        self._records.append(("tile", {"axis": name, "factor": factor, "outer": outer,
                                       "inner": inner}))
        return AxisHandle(outer, id(self)), AxisHandle(inner, id(self))

    def place(self, px: object, py: object | None = None) -> "Schedule":
        """Bind up to two spatial axes to the herd's physical PE coordinates."""
        text = "place(...)"
        if any(r[0] == "place" for r in self._records):
            raise _fail("CLAUSE-DUPLICATE", "a second place() call is not allowed",
                       "place the herd's axes once", text)
        names = [self._axis_outer_leaf(px, text)]
        if py is not None:
            names.append(self._axis_outer_leaf(py, text))
        if len(set(names)) != len(names):
            raise _fail("CLAUSE-DUPLICATE", f"axis {names[0]} is placed twice",
                       "place two different axes", text, axis=names[0], positions=["px", "py"])
        grid = next((r[1]["extents"] for r in self._records if r[0] == "grid"), None)
        if grid is None:
            raise _fail("CLAUSE-RANK", "place(...) requires grid(...) to be declared first",
                       "call grid(...) before place(...)", text)
        if len(names) != len(grid):
            raise _fail("CLAUSE-RANK", f"place has rank {len(names)} but grid has rank "
                       f"{len(grid)}", "pass py=, or declare a matching grid rank", text,
                       place_rank=len(names), grid_rank=len(grid))
        self._records.append(("place", {"names": tuple(names)}))
        return self

    def reduce(self, ax: object, op: str = "+") -> "Schedule":
        """Tag `ax` as a reduction axis with operator `op` (`+`, `max`, or `min`)."""
        name = self._axis(ax, "reduce(...)")
        text = f"reduce(ax.{name}, op={op!r})"
        if op not in _REDUCE_OPS:
            raise _fail("CLAUSE-BAD-ENUM", f"op={op!r} is not an accumulation operator",
                       'use op="+"; the operator must be associative and commutative '
                       "(SD-02 §4)", text, seen=op, domain=sorted(_REDUCE_OPS))
        if any(r[0] == "reduce" and r[1]["axis"] == name for r in self._records):
            raise _fail("CLAUSE-DUPLICATE", f"axis {name} is already a reduction axis",
                       "declare each reduction axis once", text, axis=name)
        self._records.append(("reduce", {"axis": name, "op": op}))
        return self

    def stationary(self, name: str) -> "Schedule":
        """Assert that `name` should be spatially stationary (checked by M3)."""
        op = self._operand(name, f"stationary({name!r})")
        text = f"stationary({name!r})"
        if any(r[0] == "stationary" and r[1]["operand"] == op for r in self._records):
            raise _fail("CLAUSE-DUPLICATE", f"'{op}' is already declared stationary",
                       "declare stationary() once per operand", text, operand=op)
        self._records.append(("stationary", {"operand": op}))
        return self

    def _stream_like(self, name: str, pattern: str, along: object, direction: str | None,
                     depth: int | None, clause_text: str) -> "Schedule":
        op = self._operand(name, clause_text)
        if pattern not in _PATTERNS:
            raise _fail("CLAUSE-BAD-ENUM", f"pattern={pattern!r} is not a stream pattern",
                       "use pattern in broadcast, forward, cascade", clause_text, seen=pattern,
                       domain=sorted(_PATTERNS))
        along_name = self._axis(along, clause_text)
        if depth is not None and not (isinstance(depth, int) and depth >= 1):
            raise _fail("CLAUSE-BAD-VALUE", f"depth={depth!r} is not a positive int",
                       "use a positive int depth, or omit it", clause_text, seen=depth)
        placed = next((r[1]["names"] for r in self._records if r[0] == "place"), ())
        if along_name not in placed:
            raise _fail("CLAUSE-BAD-VALUE", f"along={along_name} is not a placed axis",
                       "place it first, or stream along a placed axis", clause_text,
                       placed=list(placed))
        if any(r[0] == "stream" and r[1]["operand"] == op for r in self._records):
            raise _fail("CLAUSE-DUPLICATE", f"'{op}' already has a stream/forward delivery",
                       "declare at most one stream()/forward() per operand", clause_text,
                       operand=op)
        self._records.append(("stream", {"operand": op, "pattern": pattern, "along": along_name,
                                         "direction": direction, "depth": depth}))
        return self

    def stream(self, name: str, pattern: Pattern, along: object,
              depth: int | None = None) -> "Schedule":
        """Declare `name`'s delivery pattern along a placed axis."""
        return self._stream_like(name, pattern, along, None, depth,
                                 f"stream({name!r}, pattern={pattern!r}, ...)")

    def forward(self, name: str, along: object, dir: str,
               depth: int | None = None) -> "Schedule":
        """Sugar for `stream(..., pattern="forward")` plus a direction."""
        text = f"forward({name!r}, along=..., dir={dir!r})"
        if dir not in _DIRECTIONS:
            raise _fail("CLAUSE-BAD-ENUM", f"dir={dir!r} is not a direction",
                       "use one of W->E, E->W, N->S, S->N", text, seen=dir,
                       domain=sorted(_DIRECTIONS))
        return self._stream_like(name, "forward", along, dir, depth, text)

    def reside(self, **levels: str) -> "Schedule":
        """Declare a memory level per operand: `reside(A="L1", B="L2")`."""
        text = f"reside({levels!r})"
        for name, level in levels.items():
            op = self._operand(name, text)
            if level not in _LEVELS:
                raise _fail("CLAUSE-BAD-ENUM", f"level={level!r} is not a memory level",
                           "use one of L1, L2, L3", text, seen=level, domain=sorted(_LEVELS))
            if any(r[0] == "reside" and r[1]["operand"] == op for r in self._records):
                raise _fail("CLAUSE-DUPLICATE", f"'{op}' already has a residency",
                           "declare reside() once per operand", text, operand=op)
            self._records.append(("reside", {"operand": op, "level": level}))
        return self

    def double_buffer(self, *names: str) -> "Schedule":
        """Ask for `names` to be double-buffered (a ping-pong pair or an explicit swap)."""
        for name in names:
            op = self._operand(name, f"double_buffer{names!r}")
            self._records.append(("double_buffer", {"operand": op}))
        return self

    def pipeline(self, ax: object) -> "Schedule":
        """Hint that `ax` may be pipelined. Recorded; read by nothing (§3.7)."""
        name = self._axis(ax, "pipeline(...)")
        text = f"pipeline(ax.{name})"
        if any(r[0] == "pipeline" and r[1]["axis"] == name for r in self._records):
            raise _fail("CLAUSE-DUPLICATE", f"axis {name} is already pipelined",
                       "declare pipeline() once per axis", text, axis=name)
        self._records.append(("pipeline", {"axis": name}))
        return self

    def sequential(self, ax: object) -> "Schedule":
        """Declare `ax` a temporal (non-spatial) loop."""
        name = self._axis(ax, "sequential(...)")
        text = f"sequential(ax.{name})"
        if any(r[0] == "sequential" and r[1]["axis"] == name for r in self._records):
            raise _fail("CLAUSE-DUPLICATE", f"axis {name} is already sequential",
                       "declare sequential() once per axis", text, axis=name)
        self._records.append(("sequential", {"axis": name}))
        return self

    def window(self, name: str, dims: tuple[object, ...],
              halo: int | tuple[int, ...]) -> "Schedule":
        """Declare `name`'s halo window over `dims`."""
        op = self._operand(name, f"window({name!r}, ...)")
        text = f"window({name!r}, dims=..., halo={halo!r})"
        dim_names = tuple(self._axis(d, text) for d in dims)
        if isinstance(halo, tuple):
            if len(halo) != len(dim_names):
                raise _fail("CLAUSE-RANK", f"halo has {len(halo)} entries but dims has "
                           f"{len(dim_names)}", "give one halo entry per dim", text,
                           halo_len=len(halo), dims_len=len(dim_names))
            halo_tuple = halo
        else:
            halo_tuple = (halo,) * len(dim_names)
        for h in halo_tuple:
            if not (isinstance(h, int) and h >= 0):
                raise _fail("CLAUSE-BAD-VALUE", f"halo entry {h!r} is negative",
                           "use a non-negative halo", text, seen=h)
        if any(r[0] == "window" and r[1]["operand"] == op for r in self._records):
            raise _fail("CLAUSE-DUPLICATE", f"'{op}' already has a window", "declare window() "
                       "once per operand", text, operand=op)
        self._records.append(("window", {"operand": op, "dims": dim_names,
                                         "halo": halo_tuple}))
        return self

    def exchange(self, name: str, along: object, halo: int) -> "Schedule":
        """Declare a halo exchange of `name` along the placed axis `along`."""
        op = self._operand(name, f"exchange({name!r}, ...)")
        text = f"exchange({name!r}, along=..., halo={halo!r})"
        along_name = self._axis(along, text)
        if not (isinstance(halo, int) and halo >= 1):
            raise _fail("CLAUSE-BAD-VALUE", f"halo={halo!r} is not >= 1",
                       "use a positive halo width", text, seen=halo)
        placed = next((r[1]["names"] for r in self._records if r[0] == "place"), ())
        if along_name not in placed:
            raise _fail("CLAUSE-BAD-VALUE", f"along={along_name} is not a placed axis",
                       "place it first, or exchange along a placed axis", text,
                       placed=list(placed))
        if any(r[0] == "exchange" and r[1]["operand"] == op for r in self._records):
            raise _fail("CLAUSE-DUPLICATE", f"'{op}' already has an exchange",
                       "declare exchange() once per operand", text, operand=op)
        self._records.append(("exchange", {"operand": op, "along": along_name, "halo": halo}))
        return self

    def skew(self, time: tuple[object, ...]) -> "Schedule":
        """Set σ's leading row to the sum of `time`'s axes (FR-S17)."""
        text = f"skew(time={time!r})"
        if any(r[0] == "skew" for r in self._records):
            raise _fail("CLAUSE-DUPLICATE", "a second skew() call is not allowed",
                       "declare skew() once", text)
        if len(time) < 1:
            raise _fail("CLAUSE-BAD-VALUE", "skew(time=...) needs at least one axis",
                       "name at least one axis", text)
        names = [self._axis(a, text) for a in time]
        if len(set(names)) != len(names):
            raise _fail("CLAUSE-DUPLICATE", "skew(time=...) names the same axis twice",
                       "name each axis once", text, seen=names)
        self._records.append(("skew", {"names": tuple(names)}))
        return self

    # -- canonical model, §3.5 -----------------------------------------------------------

    @property
    def model(self) -> ScheduleModel:
        """The canonicalised `ScheduleModel`, rebuilt fresh on every access (§3.5)."""
        grid = next((r[1]["extents"] for r in self._records if r[0] == "grid"), None)
        place = next((r[1]["names"] for r in self._records if r[0] == "place"), ())
        tiles = tuple((r[1]["axis"], r[1]["factor"]) for r in self._records if r[0] == "tile")
        reductions = tuple(sorted(((r[1]["axis"], r[1]["op"]) for r in self._records
                                  if r[0] == "reduce")))
        stationary = tuple(sorted(r[1]["operand"] for r in self._records
                                 if r[0] == "stationary"))
        placed_now = place
        streams = []
        for r in self._records:
            if r[0] != "stream":
                continue
            if r[1]["along"] not in placed_now:
                raise _fail("CLAUSE-BAD-VALUE", f"along={r[1]['along']} is not a placed axis",
                           "place it first, or stream along a placed axis",
                           f"stream({r[1]['operand']!r}, ...)", placed=list(placed_now))
            streams.append(StreamClause(operand=r[1]["operand"], pattern=r[1]["pattern"],
                                        along=r[1]["along"], direction=r[1]["direction"],
                                        depth=r[1]["depth"]))
        streams = tuple(sorted(streams, key=lambda s: s.operand))
        residency = tuple(sorted(((r[1]["operand"], r[1]["level"]) for r in self._records
                                 if r[0] == "reside")))
        double_buffer = tuple(sorted({r[1]["operand"] for r in self._records
                                     if r[0] == "double_buffer"}))
        pipeline = tuple(r[1]["axis"] for r in self._records if r[0] == "pipeline")
        sequential = tuple(sorted({r[1]["axis"] for r in self._records
                                  if r[0] == "sequential"}))
        windows = []
        for r in self._records:
            if r[0] != "window":
                continue
            windows.append(WindowClause(operand=r[1]["operand"], dims=r[1]["dims"],
                                        halo=r[1]["halo"]))
        windows = tuple(sorted(windows, key=lambda w: w.operand))
        exchanges = []
        for r in self._records:
            if r[0] != "exchange":
                continue
            if r[1]["along"] not in placed_now:
                raise _fail("CLAUSE-BAD-VALUE", f"along={r[1]['along']} is not a placed axis",
                           "place it first, or exchange along a placed axis",
                           f"exchange({r[1]['operand']!r}, ...)", placed=list(placed_now))
            exchanges.append(ExchangeClause(operand=r[1]["operand"], along=r[1]["along"],
                                           halo=r[1]["halo"]))
        exchanges = tuple(sorted(exchanges, key=lambda e: e.operand))
        skew = next((r[1]["names"] for r in self._records if r[0] == "skew"), None)
        return ScheduleModel(
            target=self.target, grid=grid, tiles=tiles, place=place, reductions=reductions,
            stationary=stationary, streams=streams, residency=residency,
            double_buffer=double_buffer, pipeline=pipeline, sequential=sequential,
            windows=windows, exchanges=exchanges, skew=skew,
        )

    # -- lowering entry points (§7.1): the only place `air` may be imported --------------

    def check(self) -> Any:
        """Run M3's legality checker. Raises `LegalityError`."""
        from spatial import m3_legality
        return m3_legality.check(self.kernel.model, self.model)

    def plan(self) -> Any:
        """Run M3 then M4. Raises `LegalityError`, `MappingError`."""
        from spatial import m4_mapping, m4_selfcheck
        mapping = self.check()
        p = m4_mapping.plan(mapping)
        m4_selfcheck.self_check(p)
        return p

    def summary(self) -> str:
        """The mapping summary as text. Raises `LegalityError`, `MappingError`."""
        return "\n".join(self.plan().summary.lines)

    def mlir(self) -> str:
        """The AIR MLIR text. Raises `LegalityError`, `MappingError`, `EmissionError`."""
        from spatial import m5_emit
        p = self.plan()
        target = self.target if self.target != "auto" else "npu2"
        return m5_emit.emit(p, target).mlir

    def emit(self, path: str) -> str:
        """As `mlir()`, then write to `path`. Raises additionally `OSError`."""
        text = self.mlir()
        with open(path, "w") as f:
            f.write(text)
        return text

    def build(self, target: str | None = None) -> Any:
        """As `mlir()`, then run the toolchain. Raises additionally `ToolchainError`."""
        from spatial import m5_emit
        p = self.plan()
        t = target or (self.target if self.target != "auto" else "npu2")
        return m5_emit.emit(p, t)


def schedule(kernel: "Kernel", target: str = "npu1") -> Schedule:
    """Build an empty `Schedule` bound to one `Kernel` (design/06-interfaces.md §7.1)."""
    return Schedule(kernel, target)


__all__ = ["Schedule", "Axes", "AxisHandle", "schedule"]
