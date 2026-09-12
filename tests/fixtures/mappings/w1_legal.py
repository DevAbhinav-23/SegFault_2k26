"""W1 — GEMM, output-stationary. The `LegalMapping` of `03-lld-M3-checker.md` §6.1.

Kernel: `03-lld-M1-frontend.md` §6.1. Schedule: `03-lld-M2-schedule.md` §6.1.
Fixture: `M=N=K=64`, `TM=TN=32`, `TK=16`, `PI=PJ=2`, `f32`.

Stub written by B at P0c to unblock M4/M5; **Person A owns this file**.
"""

from __future__ import annotations

from spatial.model import (AccessMap, Axis, Dependence, Dtype, KernelModel, LegalMapping, Param,
                           ReductionSpec, ScheduleModel, Statement)

from tests.fixtures.mappings import ONE, ZERO, lin, resolve_physical, tile_axis

SOURCE = '''\
M = N = K = 64

@sp.kernel
def gemm(A: sp.f32[M, K], B: sp.f32[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
'''
"""The kernel text of `03-lld-M1-frontend.md` §6.1, verbatim. The store is line 8."""


def kernel() -> KernelModel:
    """W1's `KernelModel`, field for field from `03-lld-M1-frontend.md` §6.1."""
    return KernelModel(
        name="gemm",
        source=SOURCE,
        params=(
            Param(name="A", dtype=Dtype.f32, shape=("M", "K"), is_written=False),
            Param(name="B", dtype=Dtype.f32, shape=("K", "N"), is_written=False),
            Param(name="C", dtype=Dtype.f32, shape=("M", "N"), is_written=True),
        ),
        shape_params=("K", "M", "N"),
        # i(0, M, 1, extent 64, depth 0), j(...64..., 1), k(...64..., 2)
        axes=(
            Axis(name="i", lo=ZERO, hi=lin("M"), step=ONE, extent=64, parent=None, depth=0),
            Axis(name="j", lo=ZERO, hi=lin("N"), step=ONE, extent=64, parent=None, depth=1),
            Axis(name="k", lo=ZERO, hi=lin("K"), step=ONE, extent=64, parent=None, depth=2),
        ),
        statements=(
            Statement(
                kind="accumulate",
                # matrix rows = array dims, columns = (i, j, k); offsets all 0
                target=AccessMap(operand="C", matrix=((1, 0, 0), (0, 1, 0)),
                                 offsets=(ZERO, ZERO), is_write=True),
                reads=(
                    AccessMap(operand="A", matrix=((1, 0, 0), (0, 0, 1)),
                              offsets=(ZERO, ZERO), is_write=False),
                    AccessMap(operand="B", matrix=((0, 0, 1), (0, 1, 0)),
                              offsets=(ZERO, ZERO), is_write=False),
                ),
                op="+",
                axes=("i", "j", "k"),
                line=8,
            ),
        ),
        dependences=(Dependence(vector=(0, 0, 1), kind="RAW", operand="C"),),
        # R = ker Sf = span{e_k}
        reduction=ReductionSpec(target="C", projection=((1, 0, 0), (0, 1, 0)),
                                space=((0, 0, 1),), op=None),
    )


def schedule(target: str = "npu1") -> ScheduleModel:
    """W1's `ScheduleModel`, field for field from `03-lld-M2-schedule.md` §6.1."""
    return ScheduleModel(
        target=target,
        grid=(2, 2),
        tiles=(("i", 32), ("j", 32), ("k", 16)),
        place=("i0", "j0"),
        reductions=(("k", "+"),),
        stationary=("C",),
        streams=(),
        residency=(("A", "L1"), ("B", "L1"), ("C", "L1")),
        double_buffer=("A", "B"),
        pipeline=("k0",),
        sequential=(),
        windows=(),
        exchanges=(),
        skew=None,
    )


def legal(target: str = "npu1") -> LegalMapping:
    """W1's `LegalMapping`, field for field from `03-lld-M3-checker.md` §6.1."""
    physical, repeats = resolve_physical((2, 2), target)
    return LegalMapping(
        kernel=kernel(),
        schedule=schedule(target),
        # Coord = (i0, i1, j0, j1, k0, k1), extents (2, 32, 2, 32, 4, 16)
        axes=(
            tile_axis("i0", "i", 2, 0),
            tile_axis("i1", "i", 32, 1),
            tile_axis("j0", "j", 2, 2),
            tile_axis("j1", "j", 32, 3),
            tile_axis("k0", "k", 4, 4),
            tile_axis("k1", "k", 16, 5),
        ),
        #        i0 i1 j0 j1 k0 k1     sigma rows, in order: i0, j0, k0, i1, j1, k1
        sigma=((1, 0, 0, 0, 0, 0),
               (0, 0, 1, 0, 0, 0),
               (0, 0, 0, 0, 1, 0),
               (0, 1, 0, 0, 0, 0),
               (0, 0, 0, 1, 0, 0),
               (0, 0, 0, 0, 0, 1)),
        #     i0 i1 j0 j1 k0 k1        pi rows: i0, j0
        pi=((1, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0)),
        # ker Sπ = { e_i1, e_j1, e_k0, e_k1 }, printed as the canonical basis in §6.1
        ker_pi=((0, 1, 0, 0, 0, 0),
                (0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 1, 0),
                (0, 0, 0, 0, 0, 1)),
        # UCoord = (i, j, k); Sπ_u = [e_i; e_j]
        pi_u=((1, 0, 0),
              (0, 1, 0)),
        # ker Sπ_u = span{e_k}, canonical basis
        ker_pi_u=((0, 0, 1),),
        r_time=((0, 0, 1),),
        r_space=(),
        stationary_ops=("C",),
        physical_herd=physical,
        repeats=repeats,
        l1_bytes=12288,          # acc 4096 + A 2·2048 + B 2·2048, §6.1's L1 row
        halo_footprint=(),
    )


__all__ = ["SOURCE", "kernel", "schedule", "legal"]
