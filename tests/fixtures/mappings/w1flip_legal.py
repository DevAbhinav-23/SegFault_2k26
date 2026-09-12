"""W1-flip — weight-stationary, cascade. The `LegalMapping` of `03-lld-M3-checker.md` §6.2.

The **same kernel text** as W1 (FR-K2, no source edit), the RULING 9 schedule shape:
`grid(4)`, `place(px=ax.k0)`, `stationary("B")`, `double_buffer("A")`, and **no tile on `j`**
(`TN = N = 64`), which is what makes `B` resident for the whole run.
Schedule: `03-lld-M2-schedule.md` §6.2.

Stub written by B at P0c to unblock M4/M5; **Person A owns this file**.
"""

from __future__ import annotations

from spatial.model import Axis, LegalMapping, ScheduleModel

from tests.fixtures.mappings import ONE, ZERO, lin, resolve_physical, tile_axis
from tests.fixtures.mappings.w1_legal import SOURCE, kernel

__all__ = ["SOURCE", "kernel", "schedule", "legal"]


def schedule(target: str = "npu1") -> ScheduleModel:
    """The flip's `ScheduleModel`, field for field from `03-lld-M2-schedule.md` §6.2."""
    return ScheduleModel(
        target=target,
        grid=(4,),
        tiles=(("i", 32), ("k", 16)),          # NO tile on j: TN = N = 64 (RULING 9)
        place=("k0",),
        reductions=(("k", "+"),),
        stationary=("B",),
        streams=(),
        residency=(("A", "L1"), ("B", "L1"), ("C", "L1")),
        double_buffer=("A",),
        pipeline=(),
        sequential=(),
        windows=(),
        exchanges=(),
        skew=None,
    )


def legal(target: str = "npu1") -> LegalMapping:
    """The flip's `LegalMapping`, field for field from `03-lld-M3-checker.md` §6.2."""
    physical, repeats = resolve_physical((4,), target)
    return LegalMapping(
        kernel=kernel(),
        schedule=schedule(target),
        # Coord = (i0, i1, j, k0, k1), extents (2, 32, 64, 4, 16); j is untiled and keeps the
        # kernel's own lo/hi/step.
        axes=(
            tile_axis("i0", "i", 2, 0),
            tile_axis("i1", "i", 32, 1),
            Axis(name="j", lo=ZERO, hi=lin("N"), step=ONE, extent=64, parent=None, depth=2),
            tile_axis("k0", "k", 4, 3),
            tile_axis("k1", "k", 16, 4),
        ),
        #        i0 i1  j k0 k1        sigma rows, in order: i0, j, k0, i1, k1
        sigma=((1, 0, 0, 0, 0),
               (0, 0, 1, 0, 0),
               (0, 0, 0, 1, 0),
               (0, 1, 0, 0, 0),
               (0, 0, 0, 0, 1)),
        #     i0 i1  j k0 k1           Sπ = [e_k0] over Coord
        pi=((0, 0, 0, 1, 0),),
        # §6.2 does not print ker Sπ over Coord; this is KERNEL_BASIS(pi) (§3.2), one
        # generator per free column of `pi` in ascending column order: i0, i1, j, k1.
        ker_pi=((1, 0, 0, 0, 0),
                (0, 1, 0, 0, 0),
                (0, 0, 1, 0, 0),
                (0, 0, 0, 0, 1)),
        # UCoord = (i, j, k); Sπ_u = [e_k]
        pi_u=((0, 0, 1),),
        # ker Sπ_u = span{e_i, e_j}, canonical basis
        ker_pi_u=((1, 0, 0),
                  (0, 1, 0)),
        r_time=(),
        r_space=((0, 0, 1),),                  # R_space = span{e_k} — the cascade
        stationary_ops=("A", "B"),
        physical_herd=physical,
        repeats=repeats,
        l1_bytes=16384,   # acc 8192 + a 2·2048 + b 4096, at M3's scope: `recv` is M4's
        halo_footprint=(),
    )
