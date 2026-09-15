"""Spatial DSL for NPUs, lowering to MLIR-AIR.

A `@sp.kernel` plain-Python loop nest plus an ignorable `sp.schedule`: delete the schedule and
the program still runs in CPython and *is* the specification. See design/00-README.md §6.

As of **2026-09-15** (the final pass over the integration union): every module of
design/06-interfaces.md §7.2 is
built at `CONTRACT_VERSION = 7` -- `model` (M0), `m1_frontend`, `m2_schedule`, `m3_legality`,
`m4_mapping`, `m4_selfcheck`, `m5_emit`, `m6_tools` (both halves; the device half is untested
on hardware), and the Tenstorrent second emitter `m5tt_emit` / `m6tt_run`. `kernels/w1_gemm`,
`w2_jacobi` and `w3_sw` drive the surface end to end and reproduce every committed golden. The
surface of design/06-interfaces.md §7.1 is re-exported below.
"""

from spatial.model import CONTRACT_VERSION
from spatial.m1_frontend import Kernel, kernel
from spatial.m2_schedule import Schedule, schedule

__all__ = ["CONTRACT_VERSION", "Kernel", "kernel", "Schedule", "schedule",
          "f32", "f16", "bf16", "i32", "i8"]


class _ShapeSpec:
    """The inert object `sp.<dtype>[<dims>]` evaluates to at function-definition time.

    FR-S2: kernel parameter annotations are inert at run time -- M1 reads the *AST* of the
    annotation, never this object or `fn.__annotations__` (`03-lld-M1-frontend.md` §3.2). Its
    only job is to make `def f(A: sp.f32[M, K]): ...` evaluate without error under normal
    (non-`from __future__ import annotations`) semantics.
    """

    __slots__ = ("dtype_name", "dims")

    def __init__(self, dtype_name: str, dims: tuple) -> None:
        self.dtype_name = dtype_name
        self.dims = dims

    def __repr__(self) -> str:
        return f"sp.{self.dtype_name}[{', '.join(str(d) for d in self.dims)}]"


class _DtypeMarker:
    """`sp.f32`, `sp.f16`, etc. -- subscript to get an inert `_ShapeSpec` (FR-S2)."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getitem__(self, item: object) -> _ShapeSpec:
        dims = item if isinstance(item, tuple) else (item,)
        return _ShapeSpec(self._name, dims)

    def __repr__(self) -> str:
        return f"sp.{self._name}"


f32 = _DtypeMarker("f32")
f16 = _DtypeMarker("f16")
bf16 = _DtypeMarker("bf16")
i32 = _DtypeMarker("i32")
i8 = _DtypeMarker("i8")
