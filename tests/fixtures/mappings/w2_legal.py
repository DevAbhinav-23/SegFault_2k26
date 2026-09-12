"""W2 — Jacobi 5-point with halo exchange. The `LegalMapping` of `03-lld-M3-checker.md` §6.3.

Kernel: `03-lld-M1-frontend.md` §6.2 (RULING 1: one rank-3 `U`, `H = W = 16`, `PI·HS == H`).
Schedule: `03-lld-M2-schedule.md` §6.3.

`legal()` is parametrised because M4's tests need two variants of it and the schedule fields do
not change between them (`03-lld-M2-schedule.md` §6.3):

* `T` — the number of written planes. `T = 4` is the fixture; `T = 5` is the odd-`T` peel of
  FR-L14; `T = 0` is the `SWAP-PARITY` negative.
* `PI` — the grid extent. `PI = 2` is the fixture; `PI = 4` is the `DMA-CHANNELS` negative
  (measured `aie.connect` failure, REVIEW-round1 P-R2).

`legal("npu1", T=4, PI=2)` reproduces `03-lld-M3-checker.md` §6.3 exactly.

Stub written by B at P0c to unblock M4/M5; **Person A owns this file**.
"""

from __future__ import annotations

from spatial.model import (AccessMap, Axis, Dependence, Dtype, ExchangeClause, KernelModel,
                           LegalMapping, Param, ScheduleModel, Statement, WindowClause)

from tests.fixtures.mappings import ONE, ZERO, const, lin, resolve_physical, tile_axis

H = W = 16
"""The **interior** extents (RULING 1). `U` carries the halo, so its rank-1 extent is `H + 2`."""

SOURCE = '''\
T = {T}; H = W = 16                 # H, W are the INTERIOR extents; U carries the halo

@sp.kernel
def jacobi(U: sp.f32[T + 1, H + 2, W]):
    for t in range(0, T):
        for i in range(1, H + 1):
            for j in range(1, W - 1):
                U[t + 1, i, j] = 0.2 * (U[t, i, j] + U[t, i - 1, j] + U[t, i + 1, j]
                                        + U[t, i, j - 1] + U[t, i, j + 1])
'''
"""The kernel text of `03-lld-M1-frontend.md` §6.2, verbatim at `T = 4`. The store is line 8.

`{T}` is the one substitution: the module-level `T` is an input of the model as well as of the
source (`03-lld-M1-frontend.md` §3.4 line 18), so the recorded text must agree with the extent.
"""

_I3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
"""`M_U` — every access to `U` shares it (§6.2)."""


def kernel(T: int = 4) -> KernelModel:
    """W2's `KernelModel`, field for field from `03-lld-M1-frontend.md` §6.2."""
    return KernelModel(
        name="jacobi",
        source=SOURCE.format(T=T),
        # `T + 1` and `H + 2` are not shape-parameter NAMEs, so they are resolved to ints;
        # the bare NAME `W` stays symbolic. See design/PROGRESS-B.md, phase P0c.
        params=(Param(name="U", dtype=Dtype.f32, shape=(T + 1, H + 2, "W"), is_written=True),),
        shape_params=("H", "T", "W"),
        # t(0, T, 1, extent T), i(1, H+1, 1, extent 16), j(1, W-1, 1, extent 14)
        axes=(
            Axis(name="t", lo=ZERO, hi=lin("T"), step=ONE, extent=T, parent=None, depth=0),
            Axis(name="i", lo=const(1), hi=lin("H", 1, 1), step=ONE, extent=H,
                 parent=None, depth=1),
            Axis(name="j", lo=const(1), hi=lin("W", 1, -1), step=ONE, extent=W - 2,
                 parent=None, depth=2),
        ),
        statements=(
            Statement(
                kind="assign",
                target=AccessMap(operand="U", matrix=_I3,
                                 offsets=(const(1), ZERO, ZERO), is_write=True),
                # the five reads in source order: (0,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)
                reads=(
                    AccessMap("U", _I3, (ZERO, ZERO, ZERO), False),
                    AccessMap("U", _I3, (ZERO, const(-1), ZERO), False),
                    AccessMap("U", _I3, (ZERO, const(1), ZERO), False),
                    AccessMap("U", _I3, (ZERO, ZERO, const(-1)), False),
                    AccessMap("U", _I3, (ZERO, ZERO, const(1)), False),
                ),
                op=None,
                axes=("t", "i", "j"),
                line=8,
            ),
        ),
        # the five vectors of §6.2, in the (operand, vector) order M0 enforces
        dependences=(
            Dependence((1, -1, 0), "RAW", "U"),
            Dependence((1, 0, -1), "RAW", "U"),
            Dependence((1, 0, 0), "RAW", "U"),
            Dependence((1, 0, 1), "RAW", "U"),
            Dependence((1, 1, 0), "RAW", "U"),
        ),
        reduction=None,                        # the 5-point sum is a window, not a reduction
    )


def schedule(target: str = "npu1", PI: int = 2) -> ScheduleModel:
    """W2's `ScheduleModel`, field for field from `03-lld-M2-schedule.md` §6.3."""
    return ScheduleModel(
        target=target,
        grid=(PI,),
        tiles=(("i", H // PI),),               # HS, and PI·HS == H exactly
        place=("i0",),
        reductions=(),
        stationary=(),
        streams=(),
        residency=(("U", "L1"),),
        double_buffer=("U",),
        pipeline=(),
        sequential=("t",),
        # the scalar halo=1 is broadcast to one entry per dim
        windows=(WindowClause(operand="U", dims=("i", "j"), halo=(1, 1)),),
        exchanges=(ExchangeClause(operand="U", along="i0", halo=1),),
        skew=None,
    )


def legal(target: str = "npu1", T: int = 4, PI: int = 2) -> LegalMapping:
    """W2's `LegalMapping`, field for field from `03-lld-M3-checker.md` §6.3."""
    HS = H // PI
    physical, repeats = resolve_physical((PI,), target)
    return LegalMapping(
        kernel=kernel(T),
        schedule=schedule(target, PI),
        # Coord = (t, i0, i1, j), extents (T, PI, HS, 14); t and j are untiled and keep the
        # kernel's own lo/hi/step.
        axes=(
            Axis(name="t", lo=ZERO, hi=lin("T"), step=ONE, extent=T, parent=None, depth=0),
            tile_axis("i0", "i", PI, 1),
            tile_axis("i1", "i", HS, 2),
            Axis(name="j", lo=const(1), hi=lin("W", 1, -1), step=ONE, extent=W - 2,
                 parent=None, depth=3),
        ),
        #         t i0 i1  j           sigma rows, in order: t, i0, j, i1
        sigma=((1, 0, 0, 0),
               (0, 1, 0, 0),
               (0, 0, 0, 1),
               (0, 0, 1, 0)),
        #      t i0 i1  j              Sπ = [e_i0]
        pi=((0, 1, 0, 0),),
        # ker Sπ = { e_t, e_i1, e_j }, printed as the canonical basis in §6.3
        ker_pi=((1, 0, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1)),
        # UCoord = (t, i, j); Sπ_u = [e_i]
        pi_u=((0, 1, 0),),
        # ker Sπ_u = span{e_t, e_j}, canonical basis
        ker_pi_u=((1, 0, 0),
                  (0, 0, 1)),
        r_time=(),
        r_space=(),
        stationary_ops=("U",),                 # derived: ker M_U = {0}, vacuously contained
        physical_herd=physical,
        repeats=repeats,
        # span (2, HS+2, W) f32 — the {t, t+1} plane pair, HS ghost-padded rows, all W columns.
        # Not doubled again: `double_buffer("U")` is PAIR mode (§3.13, decision D-5).
        # At PI = 2 this is 2·10·16·4 = 1280, §6.3's figure.
        l1_bytes=2 * (HS + 2) * W * 4,
        halo_footprint=(("U", (1, 1)),),
    )


__all__ = ["H", "W", "SOURCE", "kernel", "schedule", "legal"]
