"""M0 — the frozen shared contract. Owner: all three. Spec: design/06-interfaces.md §1-§6.

Every dataclass here is `@dataclass(frozen=True)`, tuple-valued (never a list), equal by value
and hashable. Nothing in this module imports `air` (FR-S20) and nothing imports `numpy` at
module scope (`Dtype.numpy` resolves it lazily), so the contract is importable with no toolchain.

**What `__post_init__` enforces (Q-M2-3, design/03-lld-M2-schedule.md §10).** M0 enforces only
invariants that are *structural* and *local to the one object being constructed*. Every semantic
cross-check belongs to M2 (clause time), M3 (legality) or M4 (self-check), so that the error
codes of design/06-interfaces.md §6.3 stay reachable. In particular M0 does **not** check
`ScheduleModel` grid rank, `len(place) == len(grid)`, tile-divides, `sequential ∩ place`,
`l1_bytes <= 65536`, `physical_herd` divides the grid, put/get balance, channel acyclicity,
bundle-index-is-IV, the tensor read-before-write ordering, `LoopPlan.kind == "unrolled"`
placement, or any *reference* from one object to another (an operand naming a `Param`, a site
naming a channel or a buffer, a shape entry naming a `shape_param`).

Violations are programmer errors, not user diagnostics: they raise `TypeError` (wrong type) or
`ValueError` (wrong value), naming the class, the field and the offending value. NFR-4 / NFR-7
govern the public surface, which M0 is not.

Enforced invariants — one line each, with the design/06-interfaces.md section it comes from.
`tests/unit/test_m0_model.py` carries at least one case per id and asserts that the set of ids
tested equals the set listed here.

I01 every field has its declared type, and every sequence field is a `tuple`, never a
    `list` (§2-§6)
I02 every field with a closed domain holds a member of it - `Dtype`, `Level`, `Scope`,
    `Delivery`, `ReduceOp`, `Pattern`, `Direction`, `Target`, `ChannelType`, `Stage`, and the
    inline `kind` / `op` / `relation` / `cmp_op` literals; `BufferPlan.scope` therefore
    excludes `"herd.shared"` (§1, §5.1)
I03 `Expr.coeffs` has non-empty identifier keys, integer coefficients, no zero coefficient, and is
    strictly increasing by name (§1)
I04 `Param.name` is a valid Python identifier (§2.1)
I05 `Param.shape` has rank >= 1 (§2.1)
I06 every `Param.shape` entry is a positive int or a valid identifier (§2.1)
I07 `Axis.name` is non-empty (§2.2)
I08 `Axis.step` is a positive constant: no coefficients and `const >= 1` (§2.2)
I09 `Axis.depth` >= 0 (§2.2)
I10 `AccessMap.matrix` is rectangular and has exactly one row per `offsets` entry (§2.3)
I11 `Statement.op` is set if and only if `kind == "accumulate"` (§2.4)
I12 `Statement.line` >= 1 (§2.4)
I13 `Dependence.vector` is non-empty and not all zero (§2.5)
I14 `ReductionSpec.projection` and `ReductionSpec.space` are rectangular (§2.6)
I15 `KernelModel.params` names are unique (§2.1)
I16 `KernelModel` has at least one param with `is_written` (§2.1)
I17 `KernelModel.shape_params` is sorted with no duplicates (§2.7)
I18 `KernelModel.axes` names are unique (§2.7)
I19 `KernelModel.statements` has at least one entry (§2.7)
I20 `KernelModel.dependences` is sorted by `(operand, vector)` (§2.7)
I21 `KernelModel.reduction` is present if and only if a statement is an `accumulate` (§2.7)
I22 `AxisRef.name` is non-empty (§3.1)
I23 `WindowClause.dims` and `WindowClause.halo` have the same length (§3.2)
I24 every `ScheduleModel.grid` entry is >= 1 (§3.2)
I25 every `ScheduleModel.tiles` factor is >= 1 (§3.2)
I26 `ScheduleModel.place` names are distinct (§3.2)
I27 `ScheduleModel.stationary` is sorted with no duplicates (§3.2)
I28 `ScheduleModel.residency` is sorted by operand, at most one entry per operand (§3.2)
I29 `ScheduleModel.double_buffer` is sorted with no duplicates (§3.2)
I30 `ScheduleModel.sequential` is sorted with no duplicates (§3.2)
I31 `ScheduleModel.streams` has at most one entry per operand (§3.2)
I32 `ScheduleModel.windows` has at most one entry per operand (§3.2)
I33 `ScheduleModel.exchanges` has at most one entry per operand (§3.2)
I34 every `LegalMapping` matrix (`sigma`, `pi`, `ker_pi`, `pi_u`, `ker_pi_u`, `r_time`, `r_space`)
    is rectangular (§4.1)
I35 `LegalMapping.pi_u` has the same number of rows as `LegalMapping.pi` (§4.1)
I36 `LegalMapping.stationary_ops` is sorted with no duplicates (§4.1)
I37 `LegalMapping.l1_bytes` >= 0 (§4.1)
I38 every `BufferPlan.shape` entry is >= 1 (§5.1)
I39 `BufferPlan.bytes == prod(shape) * dtype.sizeof` (§5.1)
I40 `BufferPlan.level == "L3"` implies `scope == "tensor"` (§5.1)
I41 `BufferPlan.loop_depth` >= 0, and `ping_pong_candidate` implies `loop_depth >= 1` (§5.1)
I42 `Region.offsets`, `Region.sizes` and `Region.strides` have the same length (§5.2)
I43 `ChannelSite.id` is non-empty and `ChannelSite.order` >= 0 (§5.2)
I44 `ChannelPlan.name` is a valid MLIR symbol (§5.3)
I45 `ChannelPlan.size` has rank >= 1 and every entry >= 1 (§5.3)
I46 `ChannelPlan.broadcast_shape`, when present, has the rank of `size` and
    `broadcast_shape[d] % size[d] == 0` for every `d` (§5.3)
I47 `ChannelPlan.channel_type == "npu_cascade"` implies `broadcast_shape is None` (§5.3)
I48 `ChannelPlan.chain_direction is not None` if and only if `channel_type == "npu_cascade"` (§5.3)
I49 `HerdPlan.grid` has rank 1 or 2 and every entry >= 1 (§5.4)
I50 `HerdPlan.shape`, when present, has the rank of `grid` and each entry divides it exactly (§5.4)
I51 every `HerdPlan.at` entry is >= 0 (§5.4)
I52 `HerdPlan.coords` has one distinct identifier per `grid` entry (§5.4)
I53 `LoopPlan.step` is a positive constant and `LoopPlan.depth` >= 0 (§5.5)
I54 `MaxMin.operands` is non-empty (§5.5)
I55 `Const.text` is non-empty (§5.5)
I56 `MappingPlan.launch_name` and `MappingPlan.segment_name` are valid MLIR symbols (§5.6)
I57 every `MappingPlan.tensors` entry has `level == "L3"` (§5.6)
I58 a buffer name identifies one `BufferPlan` across the whole plan (§5.1, §5.6)
I59 `MappingPlan.channels` is sorted by name with no duplicate name (§5.3, §5.6)
I60 a `ChannelSite.id` identifies one site across the whole plan (§5.2)
I61 `MappingPlan.delivery` has at most one entry per operand (§5.6)
I62 `MappingSummary.residency` is sorted by operand, at most one entry per operand (§5.7)
I63 `MappingSummary.l1_bytes` and `MappingSummary.l1_budget` are >= 0 (§5.7)
I64 `EmitResult.target` is resolved, never `"auto"` (§5.8)
I65 `EmitResult.l1_peak` >= 0 (§5.8)
I66 `Diagnostic.code` is uppercase and hyphenated (§6.1)
I67 `Diagnostic.code` is in `CATALOGUE` (§6.1, §6.3)
I68 `Diagnostic.stage == CATALOGUE[code]` (§6.1, §6.3)
I69 `Diagnostic.reason` is non-empty and has no trailing period (§6.1)
I70 `Diagnostic.fix` is non-empty (§6.1)
I71 `Diagnostic.location` is present when `stage == "grammar"` (§6.1)
I72 `Diagnostic.clause` is present when `stage` is `"clause"` or `"legality"` (§6.1)
I73 `Diagnostic.details` is JSON-serialisable, with str keys, and is stored canonically (§6.1)
"""

from __future__ import annotations

import json
import re
import types
import typing
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from math import prod
from typing import Any, Literal, Union, get_args, get_origin

CONTRACT_VERSION = 3
"""The version of design/06-interfaces.md this module implements."""


# --------------------------------------------------------------------------------------------
# §1 Scalar types and enumerations
# --------------------------------------------------------------------------------------------

_BITS = {"f32": 32, "f16": 16, "bf16": 16, "i32": 32, "i8": 8}


class Dtype(Enum):
    """An element type: the MLIR spelling is the value (design/06-interfaces.md §1)."""

    f32 = "f32"
    f16 = "f16"
    bf16 = "bf16"
    i32 = "i32"
    i8 = "i8"

    @property
    def bits(self) -> int:
        """Width in bits."""
        return _BITS[self.value]

    @property
    def sizeof(self) -> int:
        """Width in bytes, `bits // 8`."""
        return self.bits // 8

    @property
    def mlir(self) -> str:
        """The MLIR spelling, e.g. `"bf16"`."""
        return self.value

    @property
    def numpy(self) -> Any:
        """The numpy (or `ml_dtypes`, for `bf16`) scalar type. Imported lazily."""
        if self is Dtype.bf16:
            import ml_dtypes

            return ml_dtypes.bfloat16
        import numpy

        return {"f32": numpy.float32, "f16": numpy.float16,
                "i32": numpy.int32, "i8": numpy.int8}[self.value]


Level = Literal["L1", "L2", "L3"]
"""Memory residence (design/06-interfaces.md §1)."""

Scope = Literal["herd.private", "segment.private", "segment.shared", "segment.per_core", "tensor"]
"""The `air.api` allocation scope spelling; `"tensor"` means L3 and is not `alloc`-able."""

Delivery = Literal["STATIONARY", "MULTICAST", "FORWARD", "CASCADE", "NONE"]
"""The reuse trichotomy plus cascade (SD-04 §3)."""

ReduceOp = Literal["+", "max", "min"]
"""An associative and commutative reduction operator."""

Pattern = Literal["broadcast", "forward", "cascade"]
"""The `stream` clause's domain (SD-01 §3)."""

Direction = Literal["W->E", "E->W", "N->S", "S->N"]
"""A `forward` clause's direction."""

Target = Literal["npu1", "npu2", "auto"]
"""Exactly what `resolve_target` accepts (`_trace.py:172-177`)."""

ChannelType = Literal[None, "npu_cascade", "npu_dma_packet"]
"""`None` is the default `npu_dma_stream` (`_channel.py:547`)."""

Stage = Literal["grammar", "clause", "legality", "mapping", "emission", "toolchain"]
"""Which module raised a diagnostic."""

LEVELS = frozenset(get_args(Level))
"""The allowed `Level` values."""
SCOPES = frozenset(get_args(Scope))
"""The allowed `Scope` values."""
DELIVERIES = frozenset(get_args(Delivery))
"""The allowed `Delivery` values."""
REDUCE_OPS = frozenset(get_args(ReduceOp))
"""The allowed `ReduceOp` values."""
PATTERNS = frozenset(get_args(Pattern))
"""The allowed `Pattern` values."""
DIRECTIONS = frozenset(get_args(Direction))
"""The allowed `Direction` values."""
TARGETS = frozenset(get_args(Target))
"""The allowed `Target` values."""
CHANNEL_TYPES = frozenset(get_args(ChannelType))
"""The allowed `ChannelType` values, including `None`."""
STAGES = frozenset(get_args(Stage))
"""The allowed `Stage` values."""

_MLIR_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_$.]*$")
_CODE = re.compile(r"^[A-Z0-9]+(-[A-Z0-9]+)*$")


# --------------------------------------------------------------------------------------------
# Field checking — the machinery behind I01 and I02
# --------------------------------------------------------------------------------------------


def _where(obj: Any, field: str) -> str:
    return f"{type(obj).__name__}.{field}"


def _err(obj: Any, field: str, why: str, value: Any) -> None:
    raise ValueError(f"{_where(obj, field)}: {why}; got {value!r}")


def _need(obj: Any, field: str, ok: bool, why: str, value: Any) -> None:
    if not ok:
        _err(obj, field, why, value)


def _check(value: Any, ann: Any, where: str) -> None:
    """Raise `TypeError`/`ValueError` unless `value` matches the annotation `ann`."""
    if ann is Any:
        return
    origin = get_origin(ann)
    if origin is Literal:
        allowed = get_args(ann)
        if not any(value is a or (type(value) is type(a) and value == a) for a in allowed):
            raise ValueError(f"{where}: {value!r} is not one of {allowed}")
        return
    if origin is Union or origin is types.UnionType:
        for arm in get_args(ann):
            try:
                _check(value, arm, where)
            except (TypeError, ValueError):
                continue
            return
        raise TypeError(f"{where}: {value!r} does not match {ann}")
    if origin is tuple:
        if type(value) is not tuple:
            raise TypeError(f"{where}: expected a tuple, got {type(value).__name__} {value!r}")
        args = get_args(ann)
        if len(args) == 2 and args[1] is Ellipsis:
            for i, item in enumerate(value):
                _check(item, args[0], f"{where}[{i}]")
            return
        if len(value) != len(args):
            raise ValueError(f"{where}: expected {len(args)} items, got {len(value)}: {value!r}")
        for i, (item, arm) in enumerate(zip(value, args)):
            _check(item, arm, f"{where}[{i}]")
        return
    if origin is not None and isinstance(origin, type) and issubclass(origin, Mapping):
        if not isinstance(value, Mapping):
            raise TypeError(f"{where}: expected a mapping, got {type(value).__name__}")
        for key in value:
            if type(key) is not str:
                raise TypeError(f"{where}: keys must be str, got {key!r}")
        return
    if ann is type(None):
        if value is not None:
            raise TypeError(f"{where}: expected None, got {value!r}")
        return
    if ann is bool or ann is int or ann is float:
        if type(value) is not ann:  # bool is a subclass of int; keep them apart
            raise TypeError(f"{where}: expected {ann.__name__}, got "
                            f"{type(value).__name__} {value!r}")
        return
    if isinstance(ann, type):
        if not isinstance(value, ann):
            raise TypeError(f"{where}: expected {ann.__name__}, got "
                            f"{type(value).__name__} {value!r}")
        return
    raise TypeError(f"{where}: unsupported annotation {ann!r}")  # pragma: no cover


@lru_cache(maxsize=None)
def _hints(cls: type) -> Mapping[str, Any]:
    return typing.get_type_hints(cls)


class _Model:
    """Base of every contract dataclass: checks field types, then the local invariants."""

    def __post_init__(self) -> None:
        hints = _hints(type(self))
        for field in fields(self):  # type: ignore[arg-type]
            _check(getattr(self, field.name), hints[field.name], _where(self, field.name))
        self._validate()

    def _validate(self) -> None:
        """Local invariants beyond the field types. Overridden where §2-§6 states any."""


def _sorted_unique(obj: Any, field: str, keys: tuple[Any, ...], what: str) -> None:
    if list(keys) != sorted(keys) or len(set(keys)) != len(keys):
        _err(obj, field, f"must be sorted by {what} with no duplicate", getattr(obj, field))


def _unique(obj: Any, field: str, keys: tuple[Any, ...], what: str) -> None:
    if len(set(keys)) != len(keys):
        _err(obj, field, f"must hold at most one entry per {what}", getattr(obj, field))


def _rect(obj: Any, field: str) -> None:
    rows = getattr(obj, field)
    if len({len(row) for row in rows}) > 1:
        _err(obj, field, "must be rectangular: every row the same length", rows)


def _identifier(obj: Any, field: str, value: str) -> None:
    _need(obj, field, value.isidentifier(), "must be a valid identifier", value)


# --------------------------------------------------------------------------------------------
# A JSON-shaped immutable mapping, so `Diagnostic.details` is hashable and canonical (I73)
# --------------------------------------------------------------------------------------------


class _FrozenDict(dict):
    """An immutable, hashable dict in sorted key order, holding only JSON-shaped values.

    It is a `dict` so that `json.dumps` and every `Mapping` consumer accept it unchanged.
    """

    def __init__(self, items: Mapping[str, Any]) -> None:
        super().__init__(sorted((k, _freeze_json(v)) for k, v in items.items()))

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self.items()))

    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("Diagnostic.details is immutable")

    __setitem__ = __delitem__ = __ior__ = _immutable
    update = setdefault = pop = popitem = clear = _immutable


def _freeze_json(value: Any) -> Any:
    """Canonicalise a JSON-serialisable value into an immutable, hashable form."""
    if isinstance(value, Mapping):
        for key in value:
            if type(key) is not str:
                raise TypeError(f"JSON object keys must be str, got {key!r}")
        return _FrozenDict(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(v) for v in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"{value!r} of type {type(value).__name__} is not JSON-serialisable")


# --------------------------------------------------------------------------------------------
# §1 Expr
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Expr(_Model):
    """An affine expression `sum(coeff * name) + const`. Equality and hashing are structural.

    `coeffs` may be given as a `Mapping[str, int]`; it is stored canonically as a tuple of
    `(name, coeff)` pairs, sorted by name, with zero coefficients dropped.
    """

    coeffs: tuple[tuple[str, int], ...] = ()
    const: int = 0

    def __post_init__(self) -> None:
        raw = self.coeffs
        if isinstance(raw, Mapping):
            raw = tuple(raw.items())
        if isinstance(raw, (tuple, list)):
            pairs = []
            for pair in raw:
                if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                    raise TypeError(f"Expr.coeffs: expected (name, coeff) pairs, got {pair!r}")
                if pair[1] != 0 or type(pair[1]) is not int:
                    pairs.append((pair[0], pair[1]))
            object.__setattr__(self, "coeffs", tuple(sorted(pairs, key=lambda p: str(p[0]))))
        super().__post_init__()

    def _validate(self) -> None:
        names = tuple(n for n, _ in self.coeffs)
        for name in names:
            _identifier(self, "coeffs", name)
        _sorted_unique(self, "coeffs", names, "name")
        for name, coeff in self.coeffs:
            _need(self, "coeffs", coeff != 0, f"coefficient of {name!r} must not be zero", coeff)

    @property
    def is_constant(self) -> bool:
        """True when the expression has no free names."""
        return not self.coeffs


# --------------------------------------------------------------------------------------------
# §2 Kernel-side contracts (produced by M1)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Param(_Model):
    """A kernel parameter: name, element type, per-dim extent, and whether the body writes it."""

    name: str
    dtype: Dtype
    shape: tuple[int | str, ...]
    is_written: bool

    def _validate(self) -> None:
        _identifier(self, "name", self.name)
        _need(self, "shape", len(self.shape) >= 1, "rank must be >= 1", self.shape)
        for entry in self.shape:
            if type(entry) is int:
                _need(self, "shape", entry >= 1, "an int extent must be >= 1", entry)
            else:
                _need(self, "shape", entry.isidentifier(),
                      "a symbolic extent must be a valid identifier", entry)


@dataclass(frozen=True)
class Axis(_Model):
    """A loop axis: its `range` bounds, its constant extent when known, and its nesting depth."""

    name: str
    lo: Expr
    hi: Expr
    step: Expr
    extent: int | None
    parent: str | None
    depth: int

    def _validate(self) -> None:
        _need(self, "name", bool(self.name), "must be non-empty", self.name)
        _need(self, "step", self.step.is_constant and self.step.const >= 1,
              "must be a positive constant", self.step)
        _need(self, "depth", self.depth >= 0, "must be >= 0", self.depth)


@dataclass(frozen=True)
class AccessMap(_Model):
    """One array access: the linear part `M_a`, the per-dim offsets, and read-vs-write."""

    operand: str
    matrix: tuple[tuple[int, ...], ...]
    offsets: tuple[Expr, ...]
    is_write: bool

    def _validate(self) -> None:
        _rect(self, "matrix")
        _need(self, "matrix", len(self.matrix) == len(self.offsets),
              f"must have one row per offset ({len(self.offsets)})", self.matrix)


@dataclass(frozen=True)
class Statement(_Model):
    """One kernel statement: the written access, its reads, and the enclosing axes."""

    kind: Literal["assign", "accumulate"]
    target: AccessMap
    reads: tuple[AccessMap, ...]
    op: ReduceOp | None
    axes: tuple[str, ...]
    line: int

    def _validate(self) -> None:
        _need(self, "op", (self.op is not None) == (self.kind == "accumulate"),
              "is set if and only if kind == 'accumulate'", self.op)
        _need(self, "line", self.line >= 1, "must be >= 1", self.line)


@dataclass(frozen=True)
class Dependence(_Model):
    """A uniform dependence: its distance vector, its kind, and the array that carries it."""

    vector: tuple[int, ...]
    kind: Literal["RAW", "WAR", "WAW"]
    operand: str

    def _validate(self) -> None:
        _need(self, "vector", len(self.vector) >= 1, "must be non-empty", self.vector)
        _need(self, "vector", any(self.vector), "must not be all zero", self.vector)


@dataclass(frozen=True)
class ReductionSpec(_Model):
    """The reduction's projection `Sf`, a canonical basis of `R = ker Sf`, and its operator."""

    target: str
    projection: tuple[tuple[int, ...], ...]
    space: tuple[tuple[int, ...], ...]
    op: ReduceOp | None

    def _validate(self) -> None:
        _rect(self, "projection")
        _rect(self, "space")


@dataclass(frozen=True)
class KernelModel(_Model):
    """The whole captured kernel: params, axes, statements, dependences and any reduction."""

    name: str
    source: str
    params: tuple[Param, ...]
    shape_params: tuple[str, ...]
    axes: tuple[Axis, ...]
    statements: tuple[Statement, ...]
    dependences: tuple[Dependence, ...]
    reduction: ReductionSpec | None

    def _validate(self) -> None:
        _unique(self, "params", tuple(p.name for p in self.params), "name")
        _need(self, "params", any(p.is_written for p in self.params),
              "at least one param must be written", tuple(p.name for p in self.params))
        _sorted_unique(self, "shape_params", self.shape_params, "name")
        _unique(self, "axes", tuple(a.name for a in self.axes), "name")
        _need(self, "statements", len(self.statements) >= 1, "must have at least one statement",
              self.statements)
        keys = tuple((d.operand, d.vector) for d in self.dependences)
        _need(self, "dependences", list(keys) == sorted(keys),
              "must be sorted by (operand, vector)", keys)
        accumulates = any(s.kind == "accumulate" for s in self.statements)
        _need(self, "reduction", (self.reduction is not None) == accumulates,
              "is present if and only if a statement is an accumulate", self.reduction)


# --------------------------------------------------------------------------------------------
# §3 Schedule-side contracts (produced by M2)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisRef(_Model):
    """An opaque axis handle: its name plus the identity of the schedule that issued it."""

    name: str
    owner: int

    def _validate(self) -> None:
        _need(self, "name", bool(self.name), "must be non-empty", self.name)


@dataclass(frozen=True)
class StreamClause(_Model):
    """A `stream`/`forward` delivery override. `depth` is recorded and never emitted (FR-E4)."""

    operand: str
    pattern: Pattern
    along: str
    direction: Direction | None
    depth: int | None


@dataclass(frozen=True)
class WindowClause(_Model):
    """A `window` clause: the windowed dims and the declared halo per dim."""

    operand: str
    dims: tuple[str, ...]
    halo: tuple[int, ...]

    def _validate(self) -> None:
        _need(self, "halo", len(self.halo) == len(self.dims),
              f"must have one entry per dim ({len(self.dims)})", self.halo)


@dataclass(frozen=True)
class ExchangeClause(_Model):
    """An `exchange` clause: the axis the halo is exchanged along and its width."""

    operand: str
    along: str
    halo: int


@dataclass(frozen=True)
class ScheduleModel(_Model):
    """Everything the clauses recorded. Pure data: no reference to the kernel function."""

    target: Target
    grid: tuple[int, ...] | None
    tiles: tuple[tuple[str, int], ...]
    place: tuple[str, ...]
    reductions: tuple[tuple[str, ReduceOp], ...]
    stationary: tuple[str, ...]
    streams: tuple[StreamClause, ...]
    residency: tuple[tuple[str, Level], ...]
    double_buffer: tuple[str, ...]
    pipeline: tuple[str, ...]
    sequential: tuple[str, ...]
    windows: tuple[WindowClause, ...]
    exchanges: tuple[ExchangeClause, ...]
    skew: tuple[str, ...] | None

    def _validate(self) -> None:
        for extent in self.grid or ():
            _need(self, "grid", extent >= 1, "every extent must be >= 1", self.grid)
        for _, factor in self.tiles:
            _need(self, "tiles", factor >= 1, "every factor must be >= 1", self.tiles)
        _unique(self, "place", self.place, "axis")
        _sorted_unique(self, "stationary", self.stationary, "operand")
        _sorted_unique(self, "residency", tuple(o for o, _ in self.residency), "operand")
        _sorted_unique(self, "double_buffer", self.double_buffer, "operand")
        _sorted_unique(self, "sequential", self.sequential, "axis")
        _unique(self, "streams", tuple(s.operand for s in self.streams), "operand")
        _unique(self, "windows", tuple(w.operand for w in self.windows), "operand")
        _unique(self, "exchanges", tuple(e.operand for e in self.exchanges), "operand")


# --------------------------------------------------------------------------------------------
# §4 Legality contract (produced by M3)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LegalMapping(_Model):
    """A schedule proved legal: the two frames, the reduction split and the resolved herd."""

    kernel: KernelModel
    schedule: ScheduleModel
    axes: tuple[Axis, ...]
    sigma: tuple[tuple[int, ...], ...]
    pi: tuple[tuple[int, ...], ...]
    ker_pi: tuple[tuple[int, ...], ...]
    pi_u: tuple[tuple[int, ...], ...]
    ker_pi_u: tuple[tuple[int, ...], ...]
    r_time: tuple[tuple[int, ...], ...]
    r_space: tuple[tuple[int, ...], ...]
    stationary_ops: tuple[str, ...]
    physical_herd: tuple[int, ...]
    repeats: tuple[int, ...]
    l1_bytes: int
    halo_footprint: tuple[tuple[str, tuple[int, ...]], ...]

    def _validate(self) -> None:
        for name in ("sigma", "pi", "ker_pi", "pi_u", "ker_pi_u", "r_time", "r_space"):
            _rect(self, name)
        _need(self, "pi_u", len(self.pi_u) == len(self.pi),
              f"must have the row count of pi ({len(self.pi)})", self.pi_u)
        _sorted_unique(self, "stationary_ops", self.stationary_ops, "operand")
        _need(self, "l1_bytes", self.l1_bytes >= 0, "must be >= 0", self.l1_bytes)


# --------------------------------------------------------------------------------------------
# §5 Mapping contract (produced by M4)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferPlan(_Model):
    """One buffer: where it lives, its shape and dtype, and whether it may ping-pong."""

    name: str
    operand: str | None
    level: Level
    scope: Scope
    shape: tuple[int, ...]
    dtype: Dtype
    bytes: int
    loop_depth: int
    ping_pong_candidate: bool

    def _validate(self) -> None:
        for extent in self.shape:
            _need(self, "shape", extent >= 1, "every extent must be >= 1", self.shape)
        expected = prod(self.shape) * self.dtype.sizeof
        _need(self, "bytes", self.bytes == expected,
              f"must be prod(shape) * dtype.sizeof == {expected}", self.bytes)
        if self.level == "L3":
            _need(self, "scope", self.scope == "tensor",
                  "level 'L3' requires scope 'tensor'", self.scope)
        _need(self, "loop_depth", self.loop_depth >= 0, "must be >= 0", self.loop_depth)
        if self.ping_pong_candidate:
            _need(self, "loop_depth", self.loop_depth >= 1,
                  "a ping-pong candidate must be allocated inside a loop", self.loop_depth)


@dataclass(frozen=True)
class Region(_Model):
    """The offsets / sizes / strides of one transferred slab."""

    offsets: tuple[Expr, ...]
    sizes: tuple[int, ...]
    strides: tuple[int, ...]

    def _validate(self) -> None:
        _need(self, "sizes", len(self.sizes) == len(self.offsets),
              f"must have the rank of offsets ({len(self.offsets)})", self.sizes)
        _need(self, "strides", len(self.strides) == len(self.offsets),
              f"must have the rank of offsets ({len(self.offsets)})", self.strides)


@dataclass(frozen=True)
class Guard(_Model):
    """An `ops.branch` condition on one herd coordinate."""

    coord: str
    relation: Literal["==", "!=", "<", "<=", ">", ">="]
    value: Expr


@dataclass(frozen=True)
class ChannelSite(_Model):
    """One `put` or `get`: its bundle index, its buffer end, its region and its ordering."""

    id: str
    kind: Literal["put", "get"]
    channel: str
    indices: tuple[Expr, ...]
    buffer: str
    region: Region
    scope: Literal["launch", "segment", "herd"]
    guard: Guard | None
    is_async: bool
    depends_on: tuple[str, ...]
    order: int

    def _validate(self) -> None:
        _need(self, "id", bool(self.id), "must be non-empty", self.id)
        _need(self, "order", self.order >= 0, "must be >= 0", self.order)


@dataclass(frozen=True)
class ChannelPlan(_Model):
    """One `air.channel`: its bundle extents, fan-out, type and the sites that use it."""

    name: str
    size: tuple[int, ...]
    broadcast_shape: tuple[int, ...] | None
    channel_type: ChannelType
    chain_direction: Literal["ascending", "descending"] | None
    dtype: Dtype
    sites: tuple[ChannelSite, ...]

    def _validate(self) -> None:
        _need(self, "name", bool(_MLIR_SYMBOL.match(self.name)),
              "must be a valid MLIR symbol", self.name)
        _need(self, "size", len(self.size) >= 1, "rank must be >= 1", self.size)
        for extent in self.size:
            _need(self, "size", extent >= 1, "every extent must be >= 1", self.size)
        if self.broadcast_shape is not None:
            _need(self, "broadcast_shape", len(self.broadcast_shape) == len(self.size),
                  f"must have the rank of size ({len(self.size)})", self.broadcast_shape)
            for d, (fan, one) in enumerate(zip(self.broadcast_shape, self.size)):
                _need(self, "broadcast_shape", fan % one == 0,
                      f"entry {d} must be a multiple of size[{d}] == {one}", self.broadcast_shape)
        if self.channel_type == "npu_cascade":
            _need(self, "broadcast_shape", self.broadcast_shape is None,
                  "a cascade channel cannot broadcast", self.broadcast_shape)
        _need(self, "chain_direction",
              (self.chain_direction is not None) == (self.channel_type == "npu_cascade"),
              "is set if and only if channel_type == 'npu_cascade'", self.chain_direction)


@dataclass(frozen=True)
class HerdPlan(_Model):
    """The herd: its logical grid, any pinned physical shape or placement, and its coords."""

    name: str
    grid: tuple[int, ...]
    shape: tuple[int, ...] | None
    at: tuple[int, int] | None
    coords: tuple[str, ...]

    def _validate(self) -> None:
        _need(self, "grid", 1 <= len(self.grid) <= 2, "rank must be 1 or 2", self.grid)
        for extent in self.grid:
            _need(self, "grid", extent >= 1, "every extent must be >= 1", self.grid)
        if self.shape is not None:
            _need(self, "shape", len(self.shape) == len(self.grid),
                  f"must have the rank of grid ({len(self.grid)})", self.shape)
            for d, (phys, logical) in enumerate(zip(self.shape, self.grid)):
                _need(self, "shape", phys >= 1 and logical % phys == 0,
                      f"entry {d} must divide grid[{d}] == {logical} exactly", self.shape)
        for coord in self.at or ():
            _need(self, "at", coord >= 0, "every coordinate must be >= 0", self.at)
        _need(self, "coords", len(self.coords) == len(self.grid),
              f"must have one name per grid entry ({len(self.grid)})", self.coords)
        _unique(self, "coords", self.coords, "name")
        for coord in self.coords:
            _identifier(self, "coords", coord)


@dataclass(frozen=True)
class Load(_Model):
    """A load of one element out of a planned buffer."""

    buffer_id: str
    subscripts: tuple[Expr, ...]


@dataclass(frozen=True)
class Const(_Model):
    """A literal, carrying the source token so no float round-trip reaches the text."""

    value: int | float
    text: str
    dtype: Dtype

    def _validate(self) -> None:
        _need(self, "text", bool(self.text), "must be non-empty", self.text)


@dataclass(frozen=True)
class BinOp(_Model):
    """A binary arithmetic node."""

    op: Literal["+", "-", "*", "/"]
    lhs: ExprNode
    rhs: ExprNode


@dataclass(frozen=True)
class Neg(_Model):
    """A unary negation node."""

    operand: ExprNode


@dataclass(frozen=True)
class MaxMin(_Model):
    """An n-ary `maximum`/`minimum` node."""

    op: Literal["maximum", "minimum"]
    operands: tuple[ExprNode, ...]

    def _validate(self) -> None:
        _need(self, "operands", len(self.operands) >= 1, "must be non-empty", self.operands)


@dataclass(frozen=True)
class Select(_Model):
    """A value-level conditional — `arith.select`, an expression, never control flow."""

    cmp_op: Literal["==", "!=", "<", "<=", ">", ">="]
    lhs: ExprNode
    rhs: ExprNode
    then: ExprNode
    otherwise: ExprNode


ExprNode = Union[Load, Const, BinOp, Neg, MaxMin, Select]
"""The frozen expression-tree union of design/06-interfaces.md §5.5."""


@dataclass(frozen=True)
class StoreNode(_Model):
    """One store of one expression tree into one buffer element."""

    buffer_id: str
    subscripts: tuple[Expr, ...]
    expr: ExprNode


@dataclass(frozen=True)
class LoopPlan(_Model):
    """One loop in the plan: `air.sequential`, or a trace-time Python loop when unrolled."""

    axis: str
    lo: Expr
    hi: Expr
    step: Expr
    kind: Literal["sequential", "unrolled"]
    depth: int
    body: tuple[PlanNode, ...]

    def _validate(self) -> None:
        _need(self, "step", self.step.is_constant and self.step.const >= 1,
              "must be a positive constant", self.step)
        _need(self, "depth", self.depth >= 0, "must be >= 0", self.depth)


@dataclass(frozen=True)
class BranchNode(_Model):
    """An `ops.branch` region pair. Conjunction is nesting; `otherwise` may be empty."""

    predicate: Guard
    then: tuple[PlanNode, ...]
    otherwise: tuple[PlanNode, ...]


PlanNode = Union[BufferPlan, ChannelSite, LoopPlan, StoreNode, BranchNode]
"""The union of everything a plan body may hold (design/06-interfaces.md §5.5)."""


@dataclass(frozen=True)
class MappingSummary(_Model):
    """The rendered human-readable summary (FR-M11) plus the facts behind each line."""

    lines: tuple[str, ...]
    residency: tuple[tuple[str, str], ...]
    herd_logical: tuple[int, ...]
    herd_physical: tuple[int, ...]
    repeats: tuple[int, ...]
    reduction_split: tuple[str, str]
    l1_bytes: int
    l1_budget: int
    channels: tuple[tuple[str, tuple[int, ...], tuple[int, ...] | None], ...]

    def _validate(self) -> None:
        _sorted_unique(self, "residency", tuple(o for o, _ in self.residency), "operand")
        _need(self, "l1_bytes", self.l1_bytes >= 0, "must be >= 0", self.l1_bytes)
        _need(self, "l1_budget", self.l1_budget >= 0, "must be >= 0", self.l1_budget)


def _walk(nodes: tuple[PlanNode, ...]) -> Iterator[PlanNode]:
    """Yield every node of a plan body, descending into loops and branches."""
    for node in nodes:
        yield node
        if isinstance(node, LoopPlan):
            yield from _walk(node.body)
        elif isinstance(node, BranchNode):
            yield from _walk(node.then)
            yield from _walk(node.otherwise)


@dataclass(frozen=True)
class MappingPlan(_Model):
    """The whole emission plan: every decision M5 replays, and nothing M5 decides."""

    mapping: LegalMapping
    tensors: tuple[BufferPlan, ...]
    launch_name: str
    segment_name: str
    herd: HerdPlan
    buffers: tuple[BufferPlan, ...]
    channels: tuple[ChannelPlan, ...]
    segment_body: tuple[PlanNode, ...]
    herd_body: tuple[PlanNode, ...]
    delivery: tuple[tuple[str, Delivery, str | None, bool], ...]
    summary: MappingSummary

    def _validate(self) -> None:
        for name in ("launch_name", "segment_name"):
            value = getattr(self, name)
            _need(self, name, bool(_MLIR_SYMBOL.match(value)),
                  "must be a valid MLIR symbol", value)
        for tensor in self.tensors:
            _need(self, "tensors", tensor.level == "L3",
                  f"{tensor.name!r} must have level 'L3'", tensor.level)
        _sorted_unique(self, "channels", tuple(c.name for c in self.channels), "name")
        _unique(self, "delivery", tuple(d[0] for d in self.delivery), "operand")
        nodes = tuple(_walk(self.segment_body)) + tuple(_walk(self.herd_body))
        seen_buffers: dict[str, BufferPlan] = {}
        for buffer in self.tensors + self.buffers + tuple(
                n for n in nodes if isinstance(n, BufferPlan)):
            if seen_buffers.setdefault(buffer.name, buffer) != buffer:
                _err(self, "buffers", f"buffer name {buffer.name!r} names two different plans",
                     buffer.name)
        seen_sites: dict[str, ChannelSite] = {}
        for site in tuple(s for c in self.channels for s in c.sites) + tuple(
                n for n in nodes if isinstance(n, ChannelSite)):
            if seen_sites.setdefault(site.id, site) != site:
                _err(self, "channels", f"site id {site.id!r} names two different sites", site.id)


@dataclass(frozen=True)
class EmitResult(_Model):
    """What `build()` returns: the AIR text, the resolved target, the plan and the summary."""

    mlir: str
    target: Target
    plan: MappingPlan
    summary: MappingSummary
    l1_peak: int

    def _validate(self) -> None:
        _need(self, "target", self.target != "auto", "must be resolved, never 'auto'",
              self.target)
        _need(self, "l1_peak", self.l1_peak >= 0, "must be >= 0", self.l1_peak)


# --------------------------------------------------------------------------------------------
# §6.3 The error-code catalogue — transcribed from design/06-interfaces.md §6.3
# --------------------------------------------------------------------------------------------

CATALOGUE: dict[str, Stage] = {
    "GRAMMAR-BAD-ANNOTATION": "grammar",
    "GRAMMAR-UNSUPPORTED-STMT": "grammar",
    "GRAMMAR-UNSUPPORTED-EXPR": "grammar",
    "GRAMMAR-UNSUPPORTED-CALL": "grammar",
    "GRAMMAR-NONAFFINE-SUBSCRIPT": "grammar",
    "GRAMMAR-BAD-RANGE": "grammar",
    "GRAMMAR-NONUNIFORM-DEP": "grammar",
    "CLAUSE-UNKNOWN-AXIS": "clause",
    "CLAUSE-UNKNOWN-OPERAND": "clause",
    "CLAUSE-BAD-ENUM": "clause",
    "CLAUSE-BAD-VALUE": "clause",
    "CLAUSE-RANK": "clause",
    "CLAUSE-DUPLICATE": "clause",
    "CLAUSE-TILE-DIVIDES": "clause",
    "CLAUSE-GRID-RANK": "clause",
    "L1-CONFLICT": "legality",
    "L2-CAUSALITY": "legality",
    "STATIONARITY": "legality",
    "REDUCE-NOT-ACCUMULATED": "legality",
    "RSPACE-NO-AC-OP": "legality",
    "CASCADE-RANK": "legality",
    "CASCADE-BROADCAST": "legality",
    "HALO-TOO-SMALL": "legality",
    "PLACE-EXTENT": "legality",
    "PLACE-SEQUENTIAL-CONFLICT": "legality",
    "HERD-RANK": "legality",
    "HERD-PHYSICAL": "legality",
    "L1-CAPACITY": "legality",
    "PINGPONG-SHAPE": "legality",
    "SWAP-PARITY": "legality",
    "BALANCE": "mapping",
    "CHANNEL-CYCLE": "mapping",
    "BUNDLE-INDEX-IS-IV": "mapping",
    "PROTOCOL-UNSUPPORTED": "mapping",
    "DMA-CHANNELS": "mapping",
    "EMIT-AIR-API": "emission",
    "EMIT-VERIFY": "emission",
    "TOOL-TARGET": "toolchain",
    "TOOL-MISSING-XCLBINUTIL": "toolchain",
    "TOOL-AIRCC-FAILED": "toolchain",
    "TOOL-DIAGNOSTIC": "toolchain",
    "TOOL-VERSION-PIN": "toolchain",
    "TOOL-NO-DEVICE": "toolchain",
}
"""Every error code of design/06-interfaces.md §6.3, mapped to the stage that raises it."""


# --------------------------------------------------------------------------------------------
# §6.1 / §6.2 Diagnostics and errors
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostic(_Model):
    """A structured rejection (FR-D1): the code, where it came from, why, and one fix."""

    code: str
    stage: Stage
    clause: str | None
    reason: str
    fix: str
    location: tuple[str, int] | None
    details: Mapping[str, object]

    def _validate(self) -> None:
        _need(self, "code", bool(_CODE.match(self.code)),
              "must be uppercase and hyphenated", self.code)
        _need(self, "code", self.code in CATALOGUE,
              "is not in design/06-interfaces.md §6.3", self.code)
        _need(self, "stage", self.stage == CATALOGUE[self.code],
              f"must be {CATALOGUE[self.code]!r} for code {self.code}", self.stage)
        _need(self, "reason", bool(self.reason), "must be non-empty", self.reason)
        _need(self, "reason", not self.reason.endswith("."),
              "is one line with no trailing period", self.reason)
        _need(self, "fix", bool(self.fix), "must be non-empty", self.fix)
        if self.stage == "grammar":
            _need(self, "location", self.location is not None,
                  "is required for a grammar diagnostic", self.location)
        if self.stage in ("clause", "legality"):
            _need(self, "clause", self.clause is not None,
                  f"is required for a {self.stage} diagnostic", self.clause)
        object.__setattr__(self, "details", _freeze_json(self.details))

    def render(self) -> str:
        """Render the four-part message of design/02-hld.md §4.2."""
        out = [f"{self.code}: {self.reason}"]
        if self.clause is not None:
            out.append(f"  in clause: {self.clause}")
        if self.location is not None:
            out.append(f"  at:        {self.location[0]}:{self.location[1]}")
        if self.details:
            facts = ", ".join(f"{k}={self.details[k]!r}" for k in sorted(self.details))
            out.append(f"  because:   {facts}")
        out.append(f"  fix:       {self.fix}")
        return "\n".join(out)


class SpatialError(Exception):
    """Base of every user-facing rejection; carries a `Diagnostic`. Never raised directly."""

    def __init__(self, diagnostic: Diagnostic) -> None:
        if not isinstance(diagnostic, Diagnostic):
            raise TypeError(f"{type(self).__name__}: expected a Diagnostic, got "
                            f"{type(diagnostic).__name__}")
        super().__init__(diagnostic)
        self.diagnostic = diagnostic

    def __str__(self) -> str:
        return self.diagnostic.render()


class GrammarError(SpatialError):
    """M1: the kernel body is outside the accepted subset (FR-S3)."""


class ClauseError(SpatialError):
    """M2: a clause argument is outside its domain (FR-S19)."""


class LegalityError(SpatialError):
    """M3: the `(σ,π)` map or a declared property fails a check (FR-L1…L14)."""


class MappingError(SpatialError):
    """M4: the plan's self-check failed, or a protocol cannot be synthesised."""


class EmissionError(SpatialError):
    """M5: `air.api` rejected a construct, or `module.operation.verify()` failed."""


class ToolchainError(SpatialError):
    """M6: `aircc`/`air-opt`/XRT failed, or the wheel pin does not match."""


# --------------------------------------------------------------------------------------------
# §8 Canonical JSON
# --------------------------------------------------------------------------------------------

_TAGGED: tuple[type, ...] = (Expr, Load, Const, BinOp, Neg, MaxMin, Select, StoreNode,
                             BranchNode, BufferPlan, ChannelSite, LoopPlan)
_TAG_REGISTRY = {cls.__name__: cls for cls in _TAGGED}


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Dtype):
        return obj.value
    if isinstance(obj, _Model):
        out: dict[str, Any] = {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
        if type(obj) in _TAGGED:
            out["__type__"] = type(obj).__name__
        return out
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if obj is None or type(obj) in (str, int, float, bool):
        return obj
    raise TypeError(f"{obj!r} of type {type(obj).__name__} is not JSON-serialisable")


def _from_jsonable(value: Any, ann: Any) -> Any:
    origin = get_origin(ann)
    # A mapping field carries opaque user data, so it never dispatches on a "__type__" key.
    if origin is not None and isinstance(origin, type) and issubclass(origin, Mapping):
        return _freeze_json(value)
    if isinstance(value, dict) and "__type__" in value:
        tag = value["__type__"]
        if tag not in _TAG_REGISTRY:
            raise ValueError(f"unknown __type__ {tag!r}; expected one of "
                             f"{sorted(_TAG_REGISTRY)}")
        return _build(_TAG_REGISTRY[tag], value)
    if origin is Union or origin is types.UnionType:
        if value is None:
            return None
        arms = [a for a in get_args(ann) if a is not type(None)]
        return _from_jsonable(value, arms[0]) if len(arms) == 1 else value
    if origin is tuple:
        args = get_args(ann)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_from_jsonable(v, args[0]) for v in value)
        return tuple(_from_jsonable(v, a) for v, a in zip(value, args))
    if ann is Dtype:
        return Dtype(value)
    if isinstance(ann, type) and is_dataclass(ann):
        return _build(ann, value)
    return value


def _build(cls: type, data: Mapping[str, Any]) -> Any:
    hints = _hints(cls)
    missing = [f.name for f in fields(cls) if f.name not in data]  # type: ignore[arg-type]
    if missing:
        raise ValueError(f"{cls.__name__}: the JSON object has no {', '.join(missing)}")
    return cls(**{f.name: _from_jsonable(data[f.name], hints[f.name])
                  for f in fields(cls)})  # type: ignore[arg-type]


def to_json(obj: Any) -> str:
    """Serialise any model object to the canonical JSON of design/06-interfaces.md §8."""
    return json.dumps(_to_jsonable(obj), sort_keys=True, indent=2) + "\n"


def from_json(text: str, cls: type) -> Any:
    """Rebuild a `cls` from `to_json` output; `from_json(to_json(x), type(x)) == x`."""
    return _from_jsonable(json.loads(text), cls)
