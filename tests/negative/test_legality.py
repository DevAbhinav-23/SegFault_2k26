"""Level N — the legality corpus (M3). Owner: Person A. Spec: `03-lld-M3-checker.md` §7.

Every case goes through the **public surface** — `sp.schedule(...)` then `.check()` — with one
exception that says why in its own docstring: `HERD-RANK` is unreachable from the clauses,
because `grid()` rejects rank 3 at clause time, so its case builds a `ScheduleModel`
programmatically and calls M3's entry point `m3_legality.check(kernel, schedule)`. FR-L11 asks
for exactly that: *"reject a grid of rank > 2 at legality time even if the clause was
constructed programmatically"*.

The fifteen `raises_*` functions are the fifteen `legality` codes of `06-interfaces.md` §6.3,
one each. Three of them are the demo's set pieces and are built by `kernels/rejections.py`, so
the corpus and the pitch reject the same schedules with the same messages.
"""

from __future__ import annotations

import pytest

import spatial as sp
from kernels import rejections
from spatial import m3_legality
from spatial.model import LegalityError, ScheduleModel
from tests.negative import _corpus

M = N = K = 64
T = 4
H = W = 16
T0 = 0
MQ = NR = 32
MATCH = 2
MISMATCH = -1
GAP = 1


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


# The `w2_zero_t` shape (`03-lld-M8-kernels-demo.md` §4): `T = 0` is the only reachable
# `SWAP-PARITY` condition after the D-4 override made an odd trip count accept-and-peel.
@sp.kernel
def jacobi_zero_t(U: sp.f32[T0 + 1, H + 2, W]):
    for t in range(0, T0):
        for i in range(1, H + 1):
            for j in range(1, W - 1):
                U[t + 1, i, j] = 0.2 * (U[t, i, j] + U[t, i - 1, j] + U[t, i + 1, j]
                                        + U[t, i, j - 1] + U[t, i, j + 1])


# A kernel with a parameter the body never touches: `D` has no access matrix, so §3.6 has no
# reuse space to check (HANDOFF, Person A item 7; architect ruling 2026-09-15).
@sp.kernel
def unused_operand(A: sp.f32[M, N], D: sp.f32[M, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            C[i, j] = A[i, j]


def _w1(target: str = "npu1") -> sp.Schedule:
    """W1 output-stationary, tiled and placed — the schedule every W1 negative edits."""
    s = sp.schedule(gemm, target=target)
    ax = s.axes()
    s.grid(2, 2)
    s.tile(ax.i, 32)
    s.tile(ax.j, 32)
    s.tile(ax.k, 16)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    return s


def _flip(grid: int = 4, tk: int = 16, target: str = "npu1") -> sp.Schedule:
    """W1 weight-stationary — `j` untiled, the cascade case (`R_space = span{e_k}`)."""
    s = sp.schedule(gemm, target=target)
    ax = s.axes()
    s.grid(grid)
    s.tile(ax.i, 32)
    s.tile(ax.k, tk)
    s.place(px=ax.k0)
    return s


def _w2() -> sp.Schedule:
    """W2 with the halo declared — the schedule the stencil negatives edit."""
    s = sp.schedule(jacobi, target="npu1")
    ax = s.axes()
    s.grid(2)
    s.tile(ax.i, 8)
    s.place(px=ax.i0)
    s.sequential(ax.t)
    s.reside(U="L1")
    return s


# --------------------------------------------------------------------------------------------
# The fifteen codes, one canonical case each
# --------------------------------------------------------------------------------------------


def raises_l1_conflict():
    """§7 `test_L1_conflict`: `ker Sσ ∩ ker Sπ = span{e_i1 − e_j1} ≠ {0}` (FR-L1)."""
    s = _w1()
    ax = s.axes()
    s.skew(time=(ax.i1, ax.j1))
    s.check()


def raises_l2_causality():
    """The demo's headline rejection: W3 with the **dropped** `j0` term (FR-L2)."""
    rejections.bad_skew("npu1")


def raises_stationarity():
    """The demo's second set piece: `place(px=i0, py=k0)` with `stationary("C")` (FR-L3)."""
    rejections.bad_stationary("npu1")


def raises_reduce_not_accumulated():
    """§7 `test_reduce_axis_must_accumulate`: `e_i ∉ R = ker Sf` (FR-S9)."""
    s = _w1()
    s.reduce(s.axes().i, op="+")
    s.check()


def raises_rspace_no_ac_op():
    """§7 `test_L5_ac_required`: the flip with `reduce` deleted (FR-L5)."""
    _flip().check()


def raises_cascade_rank():
    """§7 `test_L6_cascade_rank` sub-clause (c): a cascade needs a line, not a plane.

    At **npu2**: the `(4, 2)` grid folds onto its `(2, 4)` herd with `repeats (2, 1)`, which
    R-HERD-1 accepts, so the cascade rank is what answers. On npu1 the same grid repeats four
    times and `HERD-PHYSICAL` (R-HERD-1) answers first — also correct, but not this clause."""
    s = sp.schedule(gemm, target="npu2")
    ax = s.axes()
    s.grid(4, 2)
    s.tile(ax.i, 32)
    s.tile(ax.j, 32)
    s.tile(ax.k, 16)
    s.reduce(ax.k, op="+")
    s.place(px=ax.k0, py=ax.j0)
    s.check()


def raises_cascade_broadcast():
    """§7 `test_L6_cascade_no_broadcast`: `air.channel` refuses `broadcast_shape=` with
    `channel_type="npu_cascade"` (`_channel.py:170-177`), and M3 says so first."""
    s = _flip()
    s.reduce(s.axes().k, op="+")
    s.stream("C", pattern="broadcast", along=s.axes().k0)
    s.check()


def raises_halo_too_small():
    """§7 `test_L7_halo_too_small`: the derived footprint is 1, the declared halo 0 (FR-L7)."""
    s = _w2()
    ax = s.axes()
    s.window("U", dims=(ax.i, ax.j), halo=0)
    s.exchange("U", along=ax.i0, halo=1)
    s.check()


def raises_place_extent():
    """§7 `test_L8_place_extent`: `j0` has extent 2 against a grid extent of 4 (FR-L8 c)."""
    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    s.grid(2, 4)
    s.tile(ax.i, 32)
    s.tile(ax.j, 32)
    s.tile(ax.k, 16)
    s.place(px=ax.i0, py=ax.j0)
    s.check()


def raises_place_sequential_conflict():
    """§7 `test_L8_place_sequential`: an axis cannot be spatial and temporal (FR-L8 d)."""
    s = _w2()
    s.sequential(s.axes().i0)
    s.check()


def raises_herd_rank():
    """§7 `test_L11_rank` (FR-L11): a rank-3 grid, built programmatically.

    `grid(2, 2, 2)` never gets past M2 — that is `CLAUSE-GRID-RANK`, and it is in
    `test_clause.py`. FR-L11 asks for the check to hold *even when the clause was constructed
    programmatically*, which is what this does: M3's own entry point, `check(kernel, schedule)`,
    against a `ScheduleModel` no clause could have built.
    """
    schedule = ScheduleModel(
        target="npu1", grid=(2, 2, 2), tiles=(), place=(), reductions=(), stationary=(),
        streams=(), residency=(), double_buffer=(), pipeline=(), sequential=(), windows=(),
        exchanges=(), skew=None)
    m3_legality.check(gemm.model, schedule)


def raises_herd_physical():
    """§7 `test_L10_herd_physical`: a chain of 8 folded onto npu1's 4 cores (FR-L10)."""
    s = _flip(grid=8, tk=8)
    s.reduce(s.axes().k, op="+")
    s.check()


def raises_l1_capacity():
    """The demo's third set piece: 86 016 B of a 65 536 B budget, once ping-pong doubles
    `A` and `B` (FR-L9).

    Read at **npu2**, where the 2×2 grid is the whole herd: `repeats (1, 1)`, so ping-pong is
    the only doubling and the figure is FR-L9's own. On npu1 the same schedule charges
    122 880 under R-L1-3 (`kernels/rejections.py::bad_capacity`)."""
    rejections.bad_capacity("npu2")


def raises_pingpong_shape():
    """§7 `test_S13_pingpong_shape`: `double_buffer("C")` on the written accumulator."""
    s = _w1()
    s.double_buffer("C")
    s.check()


def raises_swap_parity():
    """§7 `test_L14_swap_parity`: `T = 0` has no timestep to swap into (FR-L14)."""
    s = sp.schedule(jacobi_zero_t, target="npu1")
    ax = s.axes()
    s.grid(2)
    s.tile(ax.i, 8)
    s.place(px=ax.i0)
    s.sequential(ax.t)
    s.window("U", dims=(ax.i, ax.j), halo=1)
    s.exchange("U", along=ax.i0, halo=1)
    s.reside(U="L1")
    s.check()


# --------------------------------------------------------------------------------------------
# Further rows of §7's table
# --------------------------------------------------------------------------------------------


def corpus_stationarity__unused_parameter():
    """A kernel parameter the body never reads or writes (HANDOFF, Person A item 7).

    `_operand_matrix` has no access matrix to return, and `STATIONARITY` is the closest code
    in the frozen 43-entry catalogue — §3.6's own code, and §3.6 is the only check that asks
    every parameter for its matrix. The architect accepted that on 2026-09-15; the right home
    is an M1 grammar rejection, which needs a new code and therefore the next contract round.
    Until then the message says what is wrong in plain words.
    """
    s = sp.schedule(unused_operand, target="npu1")
    ax = s.axes()
    s.grid(2)
    s.tile(ax.i, 32)
    s.place(px=ax.i0)
    s.check()


def corpus_place_extent__rank_mismatch():
    """A `ScheduleModel` whose `place` and `grid` ranks disagree, built by hand.

    M2 rejects this at clause time (`CLAUSE-RANK`), so the surface cannot reach it; M3's entry
    point used to answer a hand-built one with a bare `assert`, which NFR-7 forbids.
    """
    schedule = ScheduleModel(
        target="npu1", grid=(2, 2), tiles=(("i", 32),), place=("i0",), reductions=(),
        stationary=(), streams=(), residency=(), double_buffer=(), pipeline=(), sequential=(),
        windows=(), exchanges=(), skew=None)
    m3_legality.check(gemm.model, schedule)


def corpus_reduce_not_accumulated__tile_handle():
    """`reduce(ax.i0, ...)`: a tile handle has no column in `R = ker Sf`, which is a UCoord
    space, so it is checked as its root axis. This used to raise a bare `ValueError` out of
    `tuple.index` — an unhandled exception on a user path, which NFR-7 forbids."""
    s = _w1()
    s.reduce(s.axes().i0, op="+")
    s.check()


def corpus_pingpong_shape__no_streaming_loop():
    """§3.13 condition 2: an operand indexed by no outer tile handle has no loop to be
    allocated inside, so the ping-pong pass cannot fire on it."""
    s = sp.schedule(gemm, target="npu1")
    ax = s.axes()
    # `j` tiled by 8 gives a placeable `j0` of extent 8 — a grid npu1's 4-core row repeats
    # only twice, so **R-HERD-1** does not answer first — and `A[i, k]` is still indexed by
    # no outer tile handle, which is the condition under test.
    s.tile(ax.j, 8)
    s.grid(8)
    s.place(px=ax.j0)
    s.double_buffer("A")
    s.check()


# --------------------------------------------------------------------------------------------
# The assertions
# --------------------------------------------------------------------------------------------


def _module():
    import sys
    return sys.modules[__name__]


def _diagnostic(name: str):
    return dict(_corpus.collect(_module()))[name].diagnostic


@pytest.mark.fr("FR-L12", "FR-D3")
def test_legality_codes():
    """Every corpus function raises the `LegalityError` code its name spells."""
    _corpus.assert_codes(_module(), LegalityError)


@pytest.mark.fr("FR-L1")
def test_L1_conflict():
    """FR-L1: `ker Sσ ∩ ker Sπ = {0}`, or the schedule asks one PE for two things at once.

    The offending direction is `e_i1 − e_j1`: both tiles' inner axes are summed into σ's
    leading row, so a step along `i1` and back along `j1` is the same time, and neither is
    placed, so it is the same PE.
    """
    diagnostic = _diagnostic("raises_l1_conflict")
    assert diagnostic.code == "L1-CONFLICT"
    assert diagnostic.details["conflict_basis"] == ((0, 1, 0, -1, 0, 0),)   # e_i1 − e_j1
    assert diagnostic.details["conflict_basis"][0] in diagnostic.details["ker_sigma"]
    assert "skew" in diagnostic.clause and "place" in diagnostic.clause
    # ...and the schedule that only *places* i is legal: §3.3 line 27 gives every unplaced,
    # unskewed axis a σ row of its own, so ker Sσ is trivial and nothing conflicts.
    # 8 192 B of tiles (acc 4096 + a 2048 + b 2048, no double_buffer), charged **twice** on
    # npu1 — the 2×2 grid folds onto a (1, 2) herd and R-L1-3 doubles every buffer — and once
    # on npu2, whose herd takes the grid whole.
    assert _w1().check().l1_bytes == 16384
    assert _w1("npu2").check().l1_bytes == 8192


@pytest.mark.fr("FR-L5")
def test_L5_ac_required():
    """FR-L5: `R_space ≠ {}` without a declared A/C operator is rejected, naming the clause."""
    diagnostic = _diagnostic("raises_rspace_no_ac_op")
    assert diagnostic.code == "RSPACE-NO-AC-OP"
    assert diagnostic.details["r_space"] == ((0, 0, 1),)      # span{e_k}, across the PE line
    assert diagnostic.details["r_time"] == ()
    assert 'reduce(ax.k, op="+")' in diagnostic.fix
    # ...and with the clause restored the same schedule is legal, with the split recorded
    s = _flip()
    s.reduce(s.axes().k, op="+")
    mapping = s.check()
    assert (mapping.r_space, mapping.r_time) == (((0, 0, 1),), ())


@pytest.mark.fr("FR-L6")
def test_L6_cascade_rank_and_broadcast():
    """FR-L6: a cascade is a line of PEs (a, c), and it cannot also be a broadcast."""
    rank = _diagnostic("raises_cascade_rank")
    assert rank.code == "CASCADE-RANK"
    assert rank.details["grid"] == (4, 2)
    assert rank.details["carrier_axis"] == "k0"

    broadcast = _diagnostic("raises_cascade_broadcast")
    assert broadcast.code == "CASCADE-BROADCAST"
    assert broadcast.details["operand"] == "C"
    assert "npu_cascade" in broadcast.reason and "broadcast_shape" in broadcast.reason

    # (b): the 1-D flip is the accepted shape, and it records no `at=` pinning to check
    s = _flip()
    s.reduce(s.axes().k, op="+")
    assert s.check().physical_herd == (4,)


@pytest.mark.fr("FR-L8")
def test_L8_consistency():
    """FR-L8's four sub-clauses, each with its own code.

    (a) and (b) are clause-time — a rank mismatch and an axis that does not exist cannot be
    recorded at all — so they are in `tests/negative/test_clause.py`; what is asserted here is
    that they are rejected, and that (c) and (d) are the legality half.
    """
    from spatial.model import ClauseError

    s = sp.schedule(gemm, target="npu1")            # (a) rank(place) != rank(grid)
    ax = s.axes()
    s.grid(2, 2)
    s.tile(ax.i, 32)
    with pytest.raises(ClauseError) as rank:
        s.place(px=ax.i0)
    assert rank.value.diagnostic.code == "CLAUSE-RANK"

    with pytest.raises(AttributeError):             # (b) a placed axis must exist
        sp.schedule(gemm, target="npu1").axes().q

    extent = _diagnostic("raises_place_extent")     # (c) extent == grid extent
    assert extent.code == "PLACE-EXTENT"
    assert (extent.details["axis"], extent.details["axis_extent"],
            extent.details["grid_extent"]) == ("j0", 2, 4)

    both = _diagnostic("raises_place_sequential_conflict")   # (d) not placed and sequential
    assert both.code == "PLACE-SEQUENTIAL-CONFLICT"
    assert both.details["axis"] == "i0"
    assert "sequential" in both.clause


@pytest.mark.fr("FR-L9")
def test_L9_capacity_breakdown():
    """FR-L9: the message prints the computed bytes, the budget **and** the per-buffer
    breakdown — `86 016 = 36 864 (C) + 2 × 12 288 (A) + 2 × 12 288 (B)`.

    The breakdown is what says *which* tile to halve; `total` alone does not.
    """
    diagnostic = _diagnostic("raises_l1_capacity")
    assert diagnostic.code == "L1-CAPACITY"
    assert (diagnostic.details["total"], diagnostic.details["budget"]) == (86016, 63488)
    # R-L1-2: the budget is the tile's 65 536 B less the 2 048 B core stack air-to-aie reserves.
    assert (diagnostic.details["tile_bytes"],
            diagnostic.details["stack_reserved"]) == (65536, 2048)
    assert diagnostic.details["doubled"] == ("A", "B")
    per_buffer = {row[0]: row for row in diagnostic.details["per_buffer"]}
    assert per_buffer["C"] == ("C", (96, 96), "f32", 36864)
    assert per_buffer["A"] == ("A", (96, 32), "f32", 2 * 12288)
    assert per_buffer["B"] == ("B", (32, 96), "f32", 2 * 12288)
    assert sum(row[3] for row in diagnostic.details["per_buffer"]) == 86016
    # the undoubled set fits: 36 864 + 12 288 + 12 288 = 61 440
    assert sum(row[3] for row in diagnostic.details["per_buffer"]
               if row[0] not in diagnostic.details["doubled"]) == 36864


@pytest.mark.fr("FR-L10")
def test_L10_physical():
    """FR-L10: the resolved physical shape is the largest divisor of each extent under the cap.

    `_resolve_physical` is asserted directly for the table (`03-lld-M3-checker.md` §3.11) and
    through `check()` for what actually reaches the `LegalMapping`; the rejection half is
    `HERD-PHYSICAL`, which fires when folding would break a cascade chain.
    """
    resolve = m3_legality._resolve_physical
    assert resolve((3, 5), "npu1") == ((1, 1), (3, 5))
    assert resolve((4,), "npu1") == ((4,), (1,))
    assert resolve((2, 2), "npu1") == ((1, 2), (2, 1))
    assert resolve((8,), "npu2") == ((8,), (1,))
    assert resolve((8,), "npu1") == ((4,), (2,))

    mapping = _w1().check()
    assert (mapping.physical_herd, mapping.repeats) == ((1, 2), (2, 1))

    folded = _diagnostic("raises_herd_physical")
    assert folded.code == "HERD-PHYSICAL"
    assert (folded.details["grid"], folded.details["physical_herd"],
            folded.details["repeats"]) == ((8,), (4,), (2,))


@pytest.mark.fr("FR-L11")
def test_L11_rank():
    """FR-L11: a rank-3 grid is rejected at legality time, however it was constructed."""
    diagnostic = _diagnostic("raises_herd_rank")
    assert diagnostic.code == "HERD-RANK"
    assert diagnostic.details["rank"] == 3
    assert diagnostic.clause == "grid(...)"


@pytest.mark.fr("FR-L13")
def test_no_emission_on_illegal(monkeypatch):
    """FR-L13: `s.mlir()` runs the check first and emits nothing when it fails.

    The spy replaces `m5_emit.emit` itself — `Schedule.mlir` imports the module and calls the
    name, so a call would be counted. Zero calls is the assertion; the differentiator is that
    the rejection happens *before* codegen, not inside it.
    """
    from types import SimpleNamespace

    from spatial import m5_emit

    calls = []

    def spy(plan, target):
        calls.append(target)
        return SimpleNamespace(mlir="", target=target)

    monkeypatch.setattr(m5_emit, "emit", spy)

    s = _w1()
    s.skew(time=(s.axes().i1, s.axes().j1))
    with pytest.raises(LegalityError) as excinfo:
        s.mlir()
    assert excinfo.value.diagnostic.code == "L1-CONFLICT"
    assert calls == [], "an illegal schedule reached the emitter"

    # the same spy proves the positive direction: a legal schedule does reach it
    _w1().mlir()
    assert len(calls) == 1
