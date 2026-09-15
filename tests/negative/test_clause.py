"""Level N — the clause corpus (M2). Owner: Person A. Spec: `03-lld-M2-schedule.md` §7.

Every case goes through the **public surface**: `sp.schedule(...)` and the clause methods. The
eight `raises_*` functions are the eight `CLAUSE-*` codes of `06-interfaces.md` §6.3, one each;
the `corpus_*` functions are the rest of §7's 28-row table, which is what FR-S19's floor of 20
cases asks for.

A clause rejection is raised at **call time**, before any legality reasoning, which is the
property FR-S19 exists to pin: a schedule that fails late fails in the middle of emission,
where the message is about MLIR rather than about the user's clause.
"""

from __future__ import annotations

import pytest

import spatial as sp
from spatial.model import ClauseError
from tests.negative import _corpus

M = N = K = 64
T = 4
H = W = 16


@sp.kernel
def gemm(A: sp.f32[M, K], B: sp.f32[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]


@sp.kernel
def jacobi(U: sp.f32[T + 1, H + 2, W]):
    for t in range(0, T):
        for i in range(1, H + 1):
            for j in range(1, W - 1):
                U[t + 1, i, j] = 0.2 * (U[t, i, j] + U[t, i - 1, j] + U[t, i + 1, j]
                                        + U[t, i, j - 1] + U[t, i, j + 1])


# A kernel whose extents are a shape parameter bound to nothing, so `Axis.extent is None`.
# `from __future__ import annotations` is what makes it definable — the annotation is never
# evaluated, and M1 reads its AST. It is the only way to reach `tile`'s "no constant extent"
# arm, which is §7 row 8. (A docstring would be a statement the kernel grammar rejects.)
@sp.kernel
def unbounded(A: sp.f32[P, P], C: sp.f32[P, P]):
    for i in range(P):
        for j in range(P):
            C[i, j] = A[i, j]


def _w1() -> sp.Schedule:
    """A W1 schedule tiled and placed, ready for a clause that needs a placed axis."""
    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    s.grid(2, 2)
    s.tile(ax.i, 32)
    s.tile(ax.j, 32)
    s.tile(ax.k, 16)
    s.place(px=ax.i0, py=ax.j0)
    return s


# --------------------------------------------------------------------------------------------
# The eight codes, one canonical case each
# --------------------------------------------------------------------------------------------


def raises_clause_bad_enum():
    """§7 row 1: a target outside `{npu1, npu2, auto}`."""
    sp.schedule(gemm, target="xcvc1902")


def raises_clause_bad_value():
    """§7 row 3: a grid extent that is not a positive int."""
    sp.schedule(gemm, target="npu1").grid(0)


def raises_clause_duplicate():
    """§7 row 4: a second `grid()`."""
    s = sp.schedule(gemm, target="npu1")
    s.grid(2, 2)
    s.grid(4)


def raises_clause_grid_rank():
    """§7 row 2: `air.herd` is 1-D or 2-D only (FR-S6, `_trace.py:1288-1291`)."""
    sp.schedule(gemm, target="npu1").grid(2, 2, 2)


def raises_clause_rank():
    """§7 row 9: `place` of rank 1 against a grid of rank 2."""
    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    s.grid(2, 2)
    s.tile(ax.i, 32)
    s.place(px=ax.i0)


def raises_clause_tile_divides():
    """§7 row 5: 3 does not divide 64 (FR-S7)."""
    s = sp.schedule(gemm, target="npu1")
    s.tile(s.axes().i, 3)


def raises_clause_unknown_axis():
    """§7 row 7's neighbour: a plain string where an axis handle belongs."""
    sp.schedule(gemm, target="npu1").tile("i", 32)


def raises_clause_unknown_operand():
    """§7 row 16: `stationary("D")` on a kernel whose parameters are `A`, `B`, `C` (FR-S10)."""
    sp.schedule(gemm, target="npu1").stationary("D")


# --------------------------------------------------------------------------------------------
# The rest of §7's table
# --------------------------------------------------------------------------------------------


def corpus_clause_unknown_axis__foreign_handle():
    """§7 row 13: a handle taken from another schedule of the same kernel."""
    other = sp.schedule(gemm, target="npu1")
    sp.schedule(gemm, target="npu1").tile(other.axes().i, 32)


def corpus_clause_unknown_axis__unknown_name():
    """§7 row 7: an axis the kernel does not have — `axes()` refuses to hand one out."""
    s = sp.schedule(gemm, target="npu1")
    with pytest.raises(AttributeError):
        s.axes().q
    # ...and the same name reached through a live handle of another axis is still unknown
    handle = s.axes().i
    handle.name = "q"
    s.tile(handle, 4)


def corpus_clause_bad_value__tile_factor_zero():
    """§7 row 6."""
    s = sp.schedule(gemm, target="npu1")
    s.tile(s.axes().i, 0)


def corpus_clause_tile_divides__no_constant_extent():
    """§7 row 8: an axis whose extent no binding resolves cannot be checked."""
    s = sp.schedule(unbounded, target="npu1")
    s.tile(s.axes().i, 4)


def corpus_clause_rank__place_without_grid():
    """§7 row 11: `place(...)` before `grid(...)`."""
    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    s.tile(ax.i, 32)
    s.place(px=ax.i0)


def corpus_clause_duplicate__place_twice():
    """§7 row 12."""
    s = _w1()
    s.place(px=s.axes().k0, py=s.axes().j0)


def corpus_clause_duplicate__place_one_axis_twice():
    """§7 row 10: the same axis in both positions."""
    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    s.grid(2, 2)
    s.tile(ax.i, 32)
    s.place(px=ax.i0, py=ax.i0)


def corpus_clause_duplicate__reduce_twice():
    """§7 row 15."""
    s = _w1()
    s.reduce(s.axes().k, op="+")
    s.reduce(s.axes().k, op="+")


def corpus_clause_duplicate__stationary_twice():
    """One `stationary()` per operand."""
    s = _w1()
    s.stationary("C")
    s.stationary("C")


def corpus_clause_duplicate__reside_twice():
    """§7 row 19."""
    s = _w1()
    s.reside(A="L1")
    s.reside(A="L1")


def corpus_clause_duplicate__stream_then_forward():
    """§7 row 22: `forward` is sugar for `stream`, so the second delivery is a duplicate."""
    s = _w1()
    s.stream("A", pattern="broadcast", along=s.axes().j0)
    s.forward("A", along=s.axes().j0, dir="W->E")


def corpus_clause_duplicate__skew_names_an_axis_twice():
    """§7 row 27."""
    s = _w1()
    s.skew(time=(s.axes().i1, s.axes().i1))


def corpus_clause_bad_enum__reduce_op():
    """§7 row 14: the operator must be associative and commutative (FR-S9, SD-02 §4)."""
    s = _w1()
    s.reduce(s.axes().k, op="*")


def corpus_clause_bad_enum__stream_pattern():
    """§7 row 20."""
    s = _w1()
    s.stream("A", pattern="systolic", along=s.axes().j0)


def corpus_clause_bad_enum__forward_direction():
    """§7 row 23: `dir` is one of the four compass moves (FR-S11)."""
    s = _w1()
    s.forward("A", along=s.axes().j0, dir="N->W")


def corpus_clause_bad_enum__reside_level():
    """§7 row 18."""
    s = _w1()
    s.reside(A="L0")


def corpus_clause_bad_value__stream_along_unplaced():
    """§7 row 21: `along` must be a **placed** axis (FR-S11)."""
    s = _w1()
    s.stream("A", pattern="forward", along=s.axes().k0)


def corpus_clause_bad_value__stream_depth():
    """`depth` is a positive int hint or nothing at all."""
    s = _w1()
    s.stream("A", pattern="broadcast", along=s.axes().j0, depth=0)


def corpus_clause_bad_value__window_negative_halo():
    """§7 row 25: a halo is a width, so it is not negative (FR-S15)."""
    s = sp.schedule(jacobi, target="npu1")
    s.window("U", dims=(s.axes().i,), halo=-1)


def corpus_clause_bad_value__exchange_halo_zero():
    """§7 row 26: an exchange of nothing is not an exchange (FR-S16)."""
    s = sp.schedule(jacobi, target="npu1")
    ax = s.axes()
    s.grid(2)
    s.tile(ax.i, 8)
    s.place(px=ax.i0)
    s.exchange("U", along=ax.i0, halo=0)


def corpus_clause_bad_value__exchange_along_unplaced():
    """An exchange runs between neighbours along a **placed** axis (FR-S16)."""
    s = sp.schedule(jacobi, target="npu1")
    ax = s.axes()
    s.grid(2)
    s.tile(ax.i, 8)
    s.place(px=ax.i0)
    s.exchange("U", along=ax.j, halo=1)


def corpus_clause_rank__window_halo_arity():
    """§7 row 24: one halo entry per dim (FR-S15)."""
    s = sp.schedule(jacobi, target="npu1")
    ax = s.axes()
    s.window("U", dims=(ax.i, ax.j), halo=(1,))


def corpus_clause_unknown_operand__reside():
    """§7 row 17."""
    sp.schedule(gemm, target="npu1").reside(D="L1")


def corpus_clause_unknown_operand__double_buffer():
    """§7 row 28."""
    sp.schedule(gemm, target="npu1").double_buffer("D")


# --------------------------------------------------------------------------------------------
# The assertions
# --------------------------------------------------------------------------------------------


def _module():
    import sys
    return sys.modules[__name__]


@pytest.mark.fr("FR-S19", "FR-D3")
def test_clause_codes():
    """Every corpus function raises the `ClauseError` code its name spells."""
    _corpus.assert_codes(_module(), ClauseError)


@pytest.mark.fr("FR-S19")
def test_clause_errors_name_clause_argument_value_and_domain():
    """FR-S19's four parts, over all thirty-odd cases.

    `clause` is the clause text as the user wrote it, `reason` carries the offending value,
    `fix` names the accepted domain, and `details` carries the numbers — asserted for every
    case rather than for one, because the requirement is about every clause.
    """
    cases = _corpus.collect(_module())
    assert len(cases) >= 20, f"FR-S19 asks for at least 20 cases, found {len(cases)}"
    for name, exc in cases:
        diagnostic = exc.diagnostic
        assert diagnostic.clause, f"{name} names no clause"
        assert diagnostic.reason.strip(), f"{name} has no reason"
        assert len(diagnostic.fix.split()) >= 3, f"{name}'s fix is not a sentence"
        assert diagnostic.location is not None, f"{name} has no location"
        assert not diagnostic.location[0].startswith("/"), diagnostic.location
    # the enum cases carry the domain itself, which is what "accepted domain" means
    by_name = dict(cases)
    assert sorted(by_name["raises_clause_bad_enum"].diagnostic.details["domain"]) == [
        "auto", "npu1", "npu2"]


@pytest.mark.fr("FR-S6")
def test_grid_rank3_rejected():
    """FR-S6: rank > 2 is a clause-time rejection naming `grid` and the 2-D cap."""
    with pytest.raises(ClauseError) as excinfo:
        raises_clause_grid_rank()
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "CLAUSE-GRID-RANK"
    assert diagnostic.details["rank"] == 3
    assert "grid" in diagnostic.clause
    assert "grid(PI)" in diagnostic.fix and "grid(PI, PJ)" in diagnostic.fix
    # ...and rank 1 and rank 2 are both accepted
    assert sp.schedule(gemm, target="npu1").grid(4).model.grid == (4,)
    assert sp.schedule(gemm, target="npu1").grid(2, 2).model.grid == (2, 2)


@pytest.mark.fr("FR-S9")
def test_reduce_op_domain():
    """FR-S9: `op` is one of the three A/C operators, and the message lists all three."""
    with pytest.raises(ClauseError) as excinfo:
        corpus_clause_bad_enum__reduce_op()
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "CLAUSE-BAD-ENUM"
    assert sorted(diagnostic.details["domain"]) == ["+", "max", "min"]
    assert diagnostic.details["seen"] == "*"
    # the default is "+", and each of the three is recorded as given
    s = _w1()
    assert s.reduce(s.axes().k).model.reductions == (("k", "+"),)
    for op in ("+", "max", "min"):
        other = _w1()
        assert other.reduce(other.axes().k, op=op).model.reductions == (("k", op),)


@pytest.mark.fr("FR-S10")
def test_stationary_unknown_operand():
    """FR-S10: the message lists the kernel's parameter names — `A, B, C`."""
    with pytest.raises(ClauseError) as excinfo:
        raises_clause_unknown_operand()
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "CLAUSE-UNKNOWN-OPERAND"
    assert diagnostic.details["params"] == ("A", "B", "C")
    assert "A, B, C" in diagnostic.fix
    # ...and a known operand is recorded, unsolved, for FR-L3 to check
    assert _w1().stationary("C").model.stationary == ("C",)


@pytest.mark.fr("FR-S11")
def test_stream_and_forward_domains():
    """FR-S11: the three patterns, the four directions, and `along` must be placed."""
    with pytest.raises(ClauseError) as excinfo:
        corpus_clause_bad_enum__stream_pattern()
    assert sorted(excinfo.value.diagnostic.details["domain"]) == [
        "broadcast", "cascade", "forward"]

    with pytest.raises(ClauseError) as excinfo:
        corpus_clause_bad_enum__forward_direction()
    assert sorted(excinfo.value.diagnostic.details["domain"]) == [
        "E->W", "N->S", "S->N", "W->E"]

    with pytest.raises(ClauseError) as excinfo:
        corpus_clause_bad_value__stream_along_unplaced()
    assert excinfo.value.diagnostic.details["placed"] == ("i0", "j0")

    # `forward` is sugar for `stream(pattern="forward")` plus a direction, and `depth` is
    # recorded and emitted by nothing (FR-E4)
    s = _w1()
    clause = s.forward("A", along=s.axes().j0, dir="W->E", depth=2).model.streams[0]
    assert (clause.operand, clause.pattern, clause.along) == ("A", "forward", "j0")
    assert (clause.direction, clause.depth) == ("W->E", 2)


@pytest.mark.fr("FR-S15")
def test_window_halo_domain():
    """FR-S15: `halo` is a non-negative int or one entry per dim."""
    with pytest.raises(ClauseError) as excinfo:
        corpus_clause_rank__window_halo_arity()
    assert (excinfo.value.diagnostic.details["halo_len"],
            excinfo.value.diagnostic.details["dims_len"]) == (1, 2)

    with pytest.raises(ClauseError) as excinfo:
        corpus_clause_bad_value__window_negative_halo()
    assert excinfo.value.diagnostic.details["seen"] == -1

    # a scalar halo broadcasts over the dims; a tuple is taken entry by entry
    s = sp.schedule(jacobi, target="npu1")
    ax = s.axes()
    assert s.window("U", dims=(ax.i, ax.j), halo=1).model.windows[0].halo == (1, 1)
    other = sp.schedule(jacobi, target="npu1")
    ax = other.axes()
    assert other.window("U", dims=(ax.i, ax.j), halo=(1, 2)).model.windows[0].halo == (1, 2)


@pytest.mark.fr("FR-S16")
def test_exchange_declares_a_swap_along_a_placed_axis():
    """FR-S16: `halo >= 1`, `along` is placed, and the clause is recorded as pure data."""
    with pytest.raises(ClauseError) as excinfo:
        corpus_clause_bad_value__exchange_halo_zero()
    assert excinfo.value.diagnostic.details["seen"] == 0

    with pytest.raises(ClauseError) as excinfo:
        corpus_clause_bad_value__exchange_along_unplaced()
    assert excinfo.value.diagnostic.details["placed"] == ("i0",)

    s = sp.schedule(jacobi, target="npu1")
    ax = s.axes()
    s.grid(2)
    s.tile(ax.i, 8)
    s.place(px=ax.i0)
    clause = s.exchange("U", along=ax.i0, halo=1).model.exchanges[0]
    assert (clause.operand, clause.along, clause.halo) == ("U", "i0", 1)
