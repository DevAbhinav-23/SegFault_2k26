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
from spatial.model import ClauseError, GrammarError, KernelModel
from tests.fixtures.mappings import w1_legal, w2_legal, w3_legal

# ------------------------------------------------------------------------------------------
# The three kernels, defined here so inspect.getsource() has real source to read.
# ------------------------------------------------------------------------------------------

M = N = K = 64

M8 = N8 = K8 = 8
"""`gemm8`'s shape. Small on purpose: `test_kernel_call_is_oracle` runs the kernel in CPython,
one Python iteration per point, and 8^3 is 512 of them against 64^3's 262 144 (NFR-3)."""


def _gemm8(A: sp.f32[M8, K8], B: sp.f32[K8, N8], C: sp.f32[M8, N8]):
    for i in range(M8):
        for j in range(N8):
            for k in range(K8):
                C[i, j] += A[i, k] * B[k, j]


gemm8 = sp.kernel(_gemm8)
"""The decorator applied by hand, so `test_kernel_call_is_oracle` can hold the *undecorated*
function object and assert the decorated one calls exactly it (FR-S1)."""


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


# ------------------------------------------------------------------------------------------
# FR-S1, FR-S5, FR-S17 -- the surface properties the negative corpus cannot reach
# ------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-S1")
def test_kernel_call_is_oracle():
    """FR-S1: the decorated kernel (a) **is** the CPython oracle and (b) exposes its model.

    (a) is asserted twice over: the decorated object calls exactly the function object that was
    decorated -- `Kernel.__call__` is `fn.__call__` and nothing else -- and running it on an
    integer-valued `f32` fixture reproduces `A @ B` **exactly**, which is the acceptance
    `01-requirements.md` §3.1 writes for this requirement.
    """
    import numpy as np

    assert gemm8.fn is _gemm8, "the decorator must not wrap or rebuild the function"
    assert isinstance(gemm8.model, KernelModel) and gemm8.model.name == "_gemm8"

    rng = np.random.default_rng(20260915)
    a = rng.integers(-8, 8, size=(M8, K8)).astype(np.float32)
    b = rng.integers(-8, 8, size=(K8, N8)).astype(np.float32)
    c = np.zeros((M8, N8), dtype=np.float32)
    gemm8(a, b, c)
    assert np.array_equal(c, a @ b), "the decorated kernel is not the oracle"

    # ...and the schedule is ignorable: the same call after a full schedule is byte-identical
    s = sp.schedule(gemm8, target="npu1")
    ax = s.axes()
    s.grid(2, 2)
    s.tile(ax.i, 4); s.tile(ax.j, 4); s.tile(ax.k, 4)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    again = np.zeros((M8, N8), dtype=np.float32)
    gemm8(a, b, again)
    assert again.tobytes() == c.tobytes()


@pytest.mark.fr("FR-S5")
def test_schedule_is_pure_data():
    """FR-S5: `axes()` is one handle per loop variable, and the clauses are pure data.

    "Pure data" is asserted as the three properties that make it so: the model is a fresh,
    equal projection on every access (§3.5 -- there is one copy of the truth, the record list),
    it survives the canonical JSON round-trip of `06-interfaces.md` §8, and it is hashable.
    `air` is not asserted absent here because this process has imported it for the golden
    tests; that half is FR-S20's `test_no_air_import_until_build`, which uses a fresh process.
    """
    from spatial.model import ScheduleModel, from_json, to_json

    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    assert [ax.i.name, ax.j.name, ax.k.name] == ["i", "j", "k"]
    with pytest.raises(AttributeError):
        ax.q                                   # no handle for an axis the kernel has not got
    s.tile(ax.i, 32)
    assert ax.i0.name == "i0", "axes() is live: a tile handle joins the namespace"

    s.grid(2, 2)
    s.tile(ax.j, 32); s.tile(ax.k, 16)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")
    s.pipeline(ax.k0)

    model = s.model
    assert isinstance(model, ScheduleModel)
    assert model == s.model and model is not s.model     # a fresh, equal projection each time
    assert hash(model) == hash(s.model)
    assert from_json(to_json(model), ScheduleModel) == model
    assert (model.grid, model.place, model.stationary) == ((2, 2), ("i0", "j0"), ("C",))
    assert model.tiles == (("i", 32), ("j", 32), ("k", 16))

    for target in ("npu1", "npu2", "auto"):
        assert sp.schedule(gemm, target=target).model.target == target


@pytest.mark.fr("FR-S17")
def test_skew_sets_sigma():
    """FR-S17: `skew(time=...)` defines σ's **leading row** as the sum of the named axes.

    The remaining rows are the default loop order restricted to axes that are neither skewed
    nor placed (RULING 6), which is what makes W3's dropped-term schedule a rejection rather
    than an acceptance: with `j0` excluded, `Sσ = [e_i + e_j0; e_j1]`.
    """
    s = sp.schedule(sw, target="npu1")
    ax = s.axes()
    s.grid(4)
    s.tile(ax.j, 8)
    s.place(px=ax.j0)
    s.skew(time=(ax.i, ax.j0))
    s.forward("S", along=ax.j0, dir="W->E")
    s.reside(S="L1")

    assert s.model.skew == ("i", "j0")
    mapping = s.check()
    assert mapping.sigma == ((1, 1, 0), (0, 0, 1))       # e_i + e_j0, then j1 alone

    # `skew` is a sum, so naming the same two axes in the other order is a no-op
    other = sp.schedule(sw, target="npu1")
    ax = other.axes()
    other.grid(4)
    other.tile(ax.j, 8)
    other.place(px=ax.j0)
    other.skew(time=(ax.j0, ax.i))
    other.forward("S", along=ax.j0, dir="W->E")
    other.reside(S="L1")
    assert other.check().sigma == mapping.sigma
