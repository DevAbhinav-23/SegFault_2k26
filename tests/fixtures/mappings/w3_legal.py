"""W3 — Smith-Waterman wavefront. The `LegalMapping` of `03-lld-M3-checker.md` §6.4.

Kernel: `03-lld-M1-frontend.md` §6.3. Schedule: `03-lld-M2-schedule.md` §6.4.
Fixture: `MQ = NR = 32`, `PJ = 4`, `CW = 8`, `i32` scores.

Stub written by B at P0c to unblock M4/M5; **Person A owns this file**.
"""

from __future__ import annotations

from spatial.model import (AccessMap, Axis, Dependence, Dtype, KernelModel, LegalMapping, Param,
                           ScheduleModel, Statement, StreamClause)

from tests.fixtures.mappings import ONE, ZERO, const, lin, resolve_physical, tile_axis

MQ = NR = 32
"""The two sequence lengths. `S` is `[MQ + 1, NR + 1]`: row 0 and column 0 are the boundary."""

SOURCE = '''\
MQ = NR = 32; MATCH = 2; MISMATCH = -1; GAP = 1

@sp.kernel
def sw(q: sp.i32[MQ], r: sp.i32[NR], S: sp.i32[MQ + 1, NR + 1]):
    for i in range(1, MQ + 1):
        for j in range(1, NR + 1):
            sub = MATCH if q[i - 1] == r[j - 1] else MISMATCH
            S[i, j] = max(0, S[i - 1, j - 1] + sub,
                          S[i - 1, j] - GAP, S[i, j - 1] - GAP)
'''
"""The kernel text of `03-lld-M1-frontend.md` §6.3, verbatim. The store is line 8."""

_I2 = ((1, 0), (0, 1))
"""`M_S` — every access to `S` shares it (§6.3)."""


def kernel() -> KernelModel:
    """W3's `KernelModel`, field for field from `03-lld-M1-frontend.md` §6.3."""
    return KernelModel(
        name="sw",
        source=SOURCE,
        # read-only params FIRST, which MappingPlan.tensors must preserve (§5.6 invariant 6).
        # `MQ + 1` is not a shape-parameter NAME, so it is resolved to 33; the bare NAMEs
        # `MQ` and `NR` stay symbolic. See design/PROGRESS-B.md, phase P0c.
        params=(
            Param(name="q", dtype=Dtype.i32, shape=("MQ",), is_written=False),
            Param(name="r", dtype=Dtype.i32, shape=("NR",), is_written=False),
            Param(name="S", dtype=Dtype.i32, shape=(MQ + 1, NR + 1), is_written=True),
        ),
        shape_params=("MQ", "NR"),
        bindings=(("MQ", MQ), ("NR", NR)),              # `MQ = NR = 32`, SOURCE line 1
        # i(1, MQ+1, 1, extent 32), j(1, NR+1, 1, extent 32)
        axes=(
            Axis(name="i", lo=const(1), hi=lin("MQ", 1, 1), step=ONE, extent=MQ,
                 parent=None, depth=0),
            Axis(name="j", lo=const(1), hi=lin("NR", 1, 1), step=ONE, extent=NR,
                 parent=None, depth=1),
        ),
        statements=(
            Statement(
                kind="assign",                 # NOT accumulate: S[i,j] is not a `max` argument
                target=AccessMap("S", _I2, (ZERO, ZERO), True),
                # source order of the expression **after** `sub` is forward-substituted
                # (§3.5 line 15): S[i-1,j-1], q[i-1], r[j-1], S[i-1,j], S[i,j-1].
                reads=(
                    AccessMap("S", _I2, (const(-1), const(-1)), False),
                    AccessMap("q", ((1, 0),), (const(-1),), False),
                    AccessMap("r", ((0, 1),), (const(-1),), False),
                    AccessMap("S", _I2, (const(-1), ZERO), False),
                    AccessMap("S", _I2, (ZERO, const(-1)), False),
                ),
                op=None,
                axes=("i", "j"),
                line=8,
            ),
        ),
        dependences=(
            Dependence((0, 1), "RAW", "S"),
            Dependence((1, 0), "RAW", "S"),
            Dependence((1, 1), "RAW", "S"),
        ),
        reduction=None,                        # the 4-ary max is a window, not a reduction
    )


def schedule(target: str = "npu1") -> ScheduleModel:
    """W3's `ScheduleModel`, field for field from `03-lld-M2-schedule.md` §6.4."""
    return ScheduleModel(
        target=target,
        grid=(4,),
        tiles=(("j", 8),),                     # CW = 8
        place=("j0",),
        reductions=(),
        stationary=(),
        # forward() is sugar for stream() plus a direction, so it fills the single S slot
        streams=(StreamClause(operand="S", pattern="forward", along="j0",
                              direction="W->E", depth=None),),
        residency=(("S", "L1"),),
        double_buffer=(),
        pipeline=(),
        sequential=(),
        windows=(),
        exchanges=(),
        skew=("i", "j0"),
    )


def legal(target: str = "npu1") -> LegalMapping:
    """W3's `LegalMapping`, field for field from `03-lld-M3-checker.md` §6.4."""
    physical, repeats = resolve_physical((4,), target)
    return LegalMapping(
        kernel=kernel(),
        schedule=schedule(target),
        # Coord = (i, j0, j1), extents (32, 4, 8); i is untiled and keeps the kernel's bounds.
        axes=(
            Axis(name="i", lo=const(1), hi=lin("MQ", 1, 1), step=ONE, extent=MQ,
                 parent=None, depth=0),
            tile_axis("j0", "j", 4, 1),
            tile_axis("j1", "j", 8, 2),
        ),
        #         i j0 j1             row (i + j0) from skew, then the appended row j1
        sigma=((1, 1, 0),
               (0, 0, 1)),
        #      i j0 j1                Sπ = [e_j0]
        pi=((0, 1, 0),),
        # ker Sπ = { e_i, e_j1 }, printed as the canonical basis in §6.4
        ker_pi=((1, 0, 0),
                (0, 0, 1)),
        # UCoord = (i, j); Sπ_u = [e_j]
        pi_u=((0, 1),),
        # ker Sπ_u = span{e_i}, canonical basis
        ker_pi_u=((1, 0),),
        r_time=(),
        r_space=(),
        stationary_ops=("S", "r"),             # q is NOT stationary: ker M_q = span{e_j} ⊄
        physical_herd=physical,
        repeats=repeats,
        l1_bytes=232,     # q 128 + r 32 + S 72, at M3's scope: edge_in/edge_out are M4's
        halo_footprint=(),
    )


__all__ = ["MQ", "NR", "SOURCE", "kernel", "schedule", "legal"]
