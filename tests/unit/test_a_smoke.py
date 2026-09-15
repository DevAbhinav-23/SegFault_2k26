"""A compact smoke suite for M1 (frontend), M2 (schedule), M3 (legality).

Not the full negative corpus from the LLDs (design/03-lld-M1-frontend.md Sec 7,
design/03-lld-M2-schedule.md Sec 7) -- this is a focused subset: the three kernels' models
match the hand-written fixtures exactly, the full M1->M2->M3 pipeline runs end to end on all
three, and one representative negative case per module proves the rejection paths work.

Run with:  pytest tests/unit/test_a_smoke.py -v
"""

from __future__ import annotations

import pytest

import spatial as sp
from spatial.model import ClauseError, GrammarError
from tests.fixtures.mappings import w1_legal, w2_legal, w3_legal

# ------------------------------------------------------------------------------------------
# The three kernels, defined here so inspect.getsource() has real source to read.
# ------------------------------------------------------------------------------------------

M = N = K = 64


@sp.kernel
def gemm(A: sp.f32[M, K], B: sp.f32[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]


T = 4; H = W = 16


@sp.kernel
def jacobi(U: sp.f32[T + 1, H + 2, W]):
    for t in range(0, T):
        for i in range(1, H + 1):
            for j in range(1, W - 1):
                U[t + 1, i, j] = 0.2 * (U[t, i, j] + U[t, i - 1, j] + U[t, i + 1, j]
                                        + U[t, i, j - 1] + U[t, i, j + 1])


MQ = NR = 32; MATCH = 2; MISMATCH = -1; GAP = 1


@sp.kernel
def sw(q: sp.i32[MQ], r: sp.i32[NR], S: sp.i32[MQ + 1, NR + 1]):
    for i in range(1, MQ + 1):
        for j in range(1, NR + 1):
            sub = MATCH if q[i - 1] == r[j - 1] else MISMATCH
            S[i, j] = max(0, S[i - 1, j - 1] + sub,
                          S[i - 1, j] - GAP, S[i, j - 1] - GAP)


# ------------------------------------------------------------------------------------------
# M1 -- kernel model matches the hand-written fixture, field for field
# ------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-S2", "FR-S3")
def test_w1_kernel_model():
    e = w1_legal.kernel()
    assert gemm.model.params == e.params
    assert gemm.model.axes == e.axes
    assert gemm.model.dependences == e.dependences
    assert gemm.model.reduction == e.reduction


@pytest.mark.fr("FR-S2", "FR-S3")
def test_w2_kernel_model():
    e = w2_legal.kernel(T=4)
    assert jacobi.model.params == e.params
    assert jacobi.model.axes == e.axes
    assert jacobi.model.dependences == e.dependences


@pytest.mark.fr("FR-S2", "FR-S3")
def test_w3_kernel_model():
    e = w3_legal.kernel()
    assert sw.model.params == e.params
    assert sw.model.axes == e.axes
    assert sw.model.reduction is None  # a max-recurrence, not an accumulation


# ------------------------------------------------------------------------------------------
# M1 -- one representative grammar rejection (an `if` statement is not in the accepted subset)
# ------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-S3")
def test_grammar_rejects_if_statement():
    def bad(A: sp.f32[64, 64], B: sp.f32[64, 64]):
        for i in range(64):
            if i > 0:
                B[i, 0] = A[i, 0]

    with pytest.raises(GrammarError) as exc:
        sp.kernel(bad)
    assert exc.value.diagnostic.code == "GRAMMAR-UNSUPPORTED-STMT"


# ------------------------------------------------------------------------------------------
# M2 -- one representative clause rejection (a tile factor that doesn't divide the extent)
# ------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-S7")
def test_clause_rejects_bad_tile_factor():
    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    with pytest.raises(ClauseError) as exc:
        s.tile(ax.i, 3)  # 64 is not divisible by 3
    assert exc.value.diagnostic.code == "CLAUSE-TILE-DIVIDES"


# ------------------------------------------------------------------------------------------
# M1 -> M2 -> M3, end to end, real API calls (not fixture comparisons) -- all three kernels
# ------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-L3", "FR-L9")
def test_w1_pipeline_end_to_end():
    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    s.grid(2, 2)
    s.tile(ax.i, 32); s.tile(ax.j, 32); s.tile(ax.k, 16)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")
    mapping = s.check()
    assert mapping.l1_bytes == 12288
    assert mapping.stationary_ops == ("C",)


@pytest.mark.fr("FR-L7", "FR-L9")
def test_w2_pipeline_end_to_end():
    s = sp.schedule(jacobi, target="npu1")
    ax = s.axes()
    s.grid(2)
    s.tile(ax.i, 8)
    s.place(px=ax.i0)
    s.sequential(ax.t)
    s.window("U", dims=(ax.i, ax.j), halo=1)
    s.exchange("U", along=ax.i0, halo=1)
    s.reside(U="L1")
    s.double_buffer("U")
    mapping = s.check()
    assert mapping.l1_bytes == 2 * (8 + 2) * 16 * 4  # the {t, t+1} plane pair, 1280 bytes
    assert mapping.halo_footprint == (("U", (1, 1)),)


@pytest.mark.fr("FR-S8")
def test_w3_pipeline_end_to_end():
    s = sp.schedule(sw, target="npu1")
    ax = s.axes()
    s.grid(4)
    s.tile(ax.j, 8)
    s.place(px=ax.j0)
    s.skew(time=(ax.i, ax.j0))
    s.forward("S", along=ax.j0, dir="W->E")
    s.reside(S="L1")
    mapping = s.check()  # must not raise -- the headline-rejection schedule is skew(ax.i) alone
    assert mapping.r_space == ()  # no reduction in this kernel
    assert mapping.pi == ((0, 1, 0),)


@pytest.mark.fr("FR-L2")
def test_w3_headline_rejection():
    """The demo's set-piece: dropping ax.j0 from skew makes the schedule illegal (L2)."""
    s = sp.schedule(sw, target="npu1")
    ax = s.axes()
    s.grid(4)
    s.tile(ax.j, 8)
    s.place(px=ax.j0)
    s.skew(time=(ax.i,))  # the dropped term
    s.forward("S", along=ax.j0, dir="W->E")
    s.reside(S="L1")
    from spatial.model import LegalityError
    with pytest.raises(LegalityError) as exc:
        s.check()
    assert exc.value.diagnostic.code == "L2-CAUSALITY"


# ------------------------------------------------------------------------------------------
# W1-flip -- the cascade path (untested until now: R_space != {}, CHECK_CASCADE fires)
# ------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-L3", "FR-L4", "FR-L9")
def test_w1flip_pipeline_end_to_end():
    """Weight-stationary flip: same kernel text as W1, B resident, cascade reduction along k."""
    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    s.grid(4)
    s.tile(ax.i, 32); s.tile(ax.k, 16)  # no tile on j -- TN = N = 64 (RULING 9)
    s.reduce(ax.k, op="+")
    s.place(px=ax.k0)
    s.stationary("B")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A")
    mapping = s.check()
    # A is ALSO derived-stationary here (ker M_A = span{e_j} subset ker Spi_u = span{e_i,e_j}),
    # confirmed against tests/fixtures/mappings/w1flip_legal.py's own stationary_ops=("A","B").
    assert mapping.stationary_ops == ("A", "B")
    assert mapping.l1_bytes == 32 * 64 * 4 + 2 * 32 * 16 * 4 + 16 * 64 * 4
    assert mapping.r_space != ()  # this IS the cascade case -- CHECK_CASCADE actually ran


@pytest.mark.fr("FR-L3")
def test_w1flip_stationarity_rejection():
    """FR-L3's negative fixture: place(px=i0, py=k0) + stationary("C") should fail --
    C's reuse direction is NOT contained in ker(pi) under this placement.
    Grid must match i0's extent (2) and k0's extent (4) -- grid(2, 4), not (2, 2)."""
    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    s.grid(2, 4)
    s.tile(ax.i, 32); s.tile(ax.j, 32); s.tile(ax.k, 16)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.k0)
    s.stationary("C")
    from spatial.model import LegalityError
    with pytest.raises(LegalityError) as exc:
        s.check()
    assert exc.value.diagnostic.code == "STATIONARITY"
