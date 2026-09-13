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


def schedule(target: str = "npu1", *, grid2d: bool = False,
             tile_j: int | None = None) -> ScheduleModel:
    """The flip's `ScheduleModel`, field for field from `03-lld-M2-schedule.md` §6.2.

    Two variants exist **only as test inputs**, and neither is a demo artifact:

    * `grid2d=True` — `tile(ax.i, 64)` so `i0` has extent 1, `grid(1, 4)` and
      `place(px=ax.i0, py=ax.k0)`: the 2-D herd whose chain must **descend** in its second
      coordinate (`03-lld-M4-mapping.md` §3.6.3, measured on `$PROBE/q/flip2.py`). It carries
      **no** `double_buffer`: with `i` placed there is no temporal tile axis at all, so `A` is
      resident for the whole run and `03-lld-M3-checker.md` §3.13 condition 2 rejects
      `double_buffer("A")` with `PINGPONG-SHAPE`.
    * `tile_j=32` — RULING 9's regression, the shape the flip had *before* `j` was left whole.
      `B`'s `[16,32]` tile then moves with `j0` and `residency` says `re-fetched per j0`, which
      is the bug the residency line exists to make visible. Used at `residency()` level only.
    """
    tiles = (("i", 64 if grid2d else 32), ("k", 16))
    if tile_j is not None:
        tiles = tiles[:1] + (("j", tile_j),) + tiles[1:]
    return ScheduleModel(
        target=target,
        grid=(1, 4) if grid2d else (4,),
        tiles=tiles,                           # NO tile on j: TN = N = 64 (RULING 9)
        place=("i0", "k0") if grid2d else ("k0",),
        reductions=(("k", "+"),),
        stationary=("B",),
        streams=(),
        residency=(("A", "L1"), ("B", "L1"), ("C", "L1")),
        double_buffer=() if grid2d else ("A",),
        pipeline=(),
        sequential=(),
        windows=(),
        exchanges=(),
        skew=None,
    )


def _axes(grid2d: bool, tile_j: int | None) -> tuple[Axis, ...]:
    """`Coord` for the variant: the tile handles in post-tiling order, `j` whole unless tiled."""
    out = [tile_axis("i0", "i", 1 if grid2d else 2, 0),
           tile_axis("i1", "i", 64 if grid2d else 32, 1)]
    if tile_j is None:
        out.append(Axis(name="j", lo=ZERO, hi=lin("N"), step=ONE, extent=64, parent=None,
                        depth=2))
    else:
        out += [tile_axis("j0", "j", 64 // tile_j, 2), tile_axis("j1", "j", tile_j, 3)]
    out += [tile_axis("k0", "k", 4, len(out)), tile_axis("k1", "k", 16, len(out) + 1)]
    return tuple(out)


def legal(target: str = "npu1", *, grid2d: bool = False,
          tile_j: int | None = None) -> LegalMapping:
    """The flip's `LegalMapping`, field for field from `03-lld-M3-checker.md` §6.2.

    The two variants of `schedule` above carry their `LegalMapping` fields with them: `pi` gains
    a row, `ker_pi` loses one generator per placed column, and `l1_bytes` is recomputed at M3's
    scope — the operand-staging subset, without M4's `recv` (`02-hld.md` §7).
    """
    grid = (1, 4) if grid2d else (4,)
    physical, repeats = resolve_physical(grid, target)
    if grid2d:
        # Coord = (i0, i1, j, k0, k1); Sπ = [e_i0; e_k0]; l1_bytes = acc [64,64] + a [64,16]
        # + b [16,64] = 16384 + 4096 + 4096: `a` is NOT doubled, because with `i` placed no
        # temporal tile axis moves it (`03-lld-M4-mapping.md` §3.9) and the schedule declares
        # no `double_buffer`.
        return LegalMapping(
            kernel=kernel(),
            schedule=schedule(target, grid2d=True),
            axes=_axes(True, None),
            sigma=((1, 0, 0, 0, 0),            # i0, j, k0, i1, k1 — the 1-D variant's order
                   (0, 0, 1, 0, 0),
                   (0, 0, 0, 1, 0),
                   (0, 1, 0, 0, 0),
                   (0, 0, 0, 0, 1)),
            pi=((1, 0, 0, 0, 0),               # Sπ = [e_i0; e_k0] over Coord
                (0, 0, 0, 1, 0)),
            ker_pi=((0, 1, 0, 0, 0),           # KERNEL_BASIS(pi): the free columns i1, j, k1
                    (0, 0, 1, 0, 0),
                    (0, 0, 0, 0, 1)),
            pi_u=((1, 0, 0),                   # UCoord = (i, j, k); Sπ_u = [e_i; e_k]
                  (0, 0, 1)),
            ker_pi_u=((0, 1, 0),),             # ker Sπ_u = span{e_j}
            r_time=(),
            r_space=((0, 0, 1),),
            stationary_ops=("A", "B"),
            physical_herd=physical,
            repeats=repeats,
            l1_bytes=24576,
            halo_footprint=(),
        )
    axes = _axes(False, tile_j)
    order = ("i0", "j0", "k0", "i1", "j1", "k1") if tile_j else ("i0", "j", "k0", "i1", "k1")
    names = [axis.name for axis in axes]
    unit = {name: tuple(1 if k == index else 0 for k in range(len(names)))
            for index, name in enumerate(names)}
    tn = tile_j or 64
    return LegalMapping(
        kernel=kernel(),
        schedule=schedule(target, tile_j=tile_j),
        # Coord = (i0, i1, j, k0, k1), extents (2, 32, 64, 4, 16) — j untiled, keeping the
        # kernel's own lo/hi/step; with `tile_j` it is (i0, i1, j0, j1, k0, k1) instead.
        axes=axes,
        # sigma rows, in order: i0, j, k0, i1, k1 (the tiled variant interleaves j0/j1 as W1)
        sigma=tuple(unit[name] for name in order),
        pi=(unit["k0"],),                      # Sπ = [e_k0] over Coord
        # §6.2 does not print ker Sπ over Coord; this is KERNEL_BASIS(pi) (§3.2), one generator
        # per free column of `pi` in ascending column order: i0, i1, j, k1.
        ker_pi=tuple(unit[name] for name in names if name != "k0"),
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
        # acc [32,TN] + a [32,16] x2 + b [16,TN], at M3's scope: `recv` is M4's (02-hld §7).
        l1_bytes=32 * tn * 4 + 2 * 32 * 16 * 4 + 16 * tn * 4,
        halo_footprint=(),
    )
