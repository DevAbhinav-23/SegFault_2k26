"""Hand-written LegalMapping literals. Owner: Person A (design/03-lld-M7-tests.md §3.8).

Stub written by B at P0c to unblock M4/M5; **Person A owns this package** and replaces each
literal with `m3.check`'s own output at D2.

Every number in `w1_legal`, `w1flip_legal`, `w2_legal` and `w3_legal` is transcribed from
`design/03-lld-M1-frontend.md` §6 (the kernel), `design/03-lld-M2-schedule.md` §6 (the schedule)
and `design/03-lld-M3-checker.md` §6 (the mapping). Nothing here is derived by reasoning where
one of those documents prints the number; where a document prints a basis as `span{...}` the
canonical row-echelon tuple it implies is written out and the comment says so
(`03-lld-M3-checker.md` §3.2's `KERNEL_BASIS`).

This module holds only what all four literals share.
"""

from __future__ import annotations

from spatial.model import Axis, Expr

PHYSICAL_HERD: dict[str, dict[int, tuple[int, ...]]] = {
    "npu1": {1: (4,), 2: (1, 4)},
    "npu2": {1: (8,), 2: (2, 4)},
}
"""The physical array caps (`design/07-environment.md` §5, `_trace.py:88-91`)."""

L1_BUDGET = 65536
"""`L1_BYTES` (`_trace.py:100`), for reference; M3's `l1_bytes` is charged against it."""

ZERO = Expr()
"""The constant 0."""

ONE = Expr((), 1)
"""The constant 1, and every axis's `step` in all four fixtures."""


def const(value: int) -> Expr:
    """The constant affine expression `value`."""
    return Expr((), value)


def lin(name: str, coeff: int = 1, offset: int = 0) -> Expr:
    """The affine expression `coeff * name + offset`."""
    return Expr({name: coeff}, offset)


def tile_axis(name: str, parent: str, extent: int, depth: int) -> Axis:
    """A tile-derived post-tiling `Axis`.

    `name` and `parent` are `03-lld-M2-schedule.md` §3.2's naming rule and `extent` is its
    table; `lo`/`hi`/`step` and `depth` are **B's reading**, because `03-lld-M3-checker.md` §3.3
    fixes neither: a tile handle runs `range(0, extent)` with step 1, and `depth` is its index
    in the post-tiling coordinate order, which is the nesting depth of the tiled nest
    (recorded in `design/PROGRESS-B.md`, phase P0c).
    """
    return Axis(name=name, lo=ZERO, hi=const(extent), step=ONE,
                extent=extent, parent=parent, depth=depth)


def resolve_physical(grid: tuple[int, ...], target: str) -> tuple[tuple[int, ...],
                                                                  tuple[int, ...]]:
    """`(physical_herd, repeats)`: the largest divisor of each extent within the target cap.

    `03-lld-M3-checker.md` §3.11, mirroring `_resolve_physical` (`_trace.py:1347-1373`).
    """
    caps = PHYSICAL_HERD[target][len(grid)]
    physical = tuple(max(d for d in range(1, extent + 1) if extent % d == 0 and d <= cap)
                     for extent, cap in zip(grid, caps))
    return physical, tuple(g // p for g, p in zip(grid, physical))
