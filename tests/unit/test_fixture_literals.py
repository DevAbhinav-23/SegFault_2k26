"""The stub fixtures, checked against the documents rather than against themselves.

Every expected value below is re-quoted from `design/03-lld-M3-checker.md` §6.1-§6.4 (the
mapping), `design/03-lld-M1-frontend.md` §6 (the kernel), `design/03-lld-M2-schedule.md` §3.6
(the post-tiling order) and `design/03-lld-M4-mapping.md` §6.1 (the W1 plan) — never read off
the literal, so a typo in the literal fails here instead of propagating into M4 and M5.

Written by B at P0c together with the fixtures it guards.
"""

from __future__ import annotations

import pytest

from spatial.model import LegalMapping, MappingPlan, from_json, to_json
from tests.fixtures.mappings import w1_legal, w1flip_legal, w2_legal, w3_legal
from tests.fixtures.plans import w1_plan

MAPPINGS = {"w1": w1_legal, "w1flip": w1flip_legal, "w2": w2_legal, "w3": w3_legal}
TARGETS = ("npu1", "npu2")

# --------------------------------------------------------------------------------------------
# The numbers, re-quoted from 03-lld-M3-checker.md §6
# --------------------------------------------------------------------------------------------

EXPECTED = {
    # §6.1: axes = (i0, i1, j0, j1, k0, k1), extents (2, 32, 2, 32, 4, 16); UCoord = (i, j, k)
    "w1": dict(
        axes=("i0", "i1", "j0", "j1", "k0", "k1"),
        extents=(2, 32, 2, 32, 4, 16),
        sigma=((1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0),
               (0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1)),
        pi=((1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)),
        ker_pi=((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)),
        pi_u=((1, 0, 0), (0, 1, 0)),                    # Sπ_u = [e_i; e_j]
        ker_pi_u=((0, 0, 1),),                          # span{e_k}
        r_time=((0, 0, 1),),
        r_space=(),
        stationary_ops=("C",),
        l1_bytes=12288,
        halo_footprint=(),
        herd={"npu1": ((1, 2), (2, 1)), "npu2": ((2, 2), (1, 1))},
        params=("A", "B", "C"),
        dependences=((((0, 0, 1)), "RAW", "C"),),
    ),
    # §6.2: axes = (i0, i1, j, k0, k1), extents (2, 32, 64, 4, 16); Sπ = [e_k0]
    "w1flip": dict(
        axes=("i0", "i1", "j", "k0", "k1"),
        extents=(2, 32, 64, 4, 16),
        sigma=((1, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0),
               (0, 1, 0, 0, 0), (0, 0, 0, 0, 1)),
        pi=((0, 0, 0, 1, 0),),
        ker_pi=((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 1)),
        pi_u=((0, 0, 1),),                              # Sπ_u = [e_k]
        ker_pi_u=((1, 0, 0), (0, 1, 0)),                # span{e_i, e_j}
        r_time=(),
        r_space=((0, 0, 1),),                           # span{e_k}
        stationary_ops=("A", "B"),
        l1_bytes=16384,                                 # at M3's scope; `recv` is M4's
        halo_footprint=(),
        herd={"npu1": ((4,), (1,)), "npu2": ((4,), (1,))},
        params=("A", "B", "C"),
        dependences=((((0, 0, 1)), "RAW", "C"),),
    ),
    # §6.3: axes = (t, i0, i1, j), extents (4, 2, 8, 14); UCoord = (t, i, j)
    "w2": dict(
        axes=("t", "i0", "i1", "j"),
        extents=(4, 2, 8, 14),
        sigma=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0)),
        pi=((0, 1, 0, 0),),
        ker_pi=((1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        pi_u=((0, 1, 0),),                              # Sπ_u = [e_i]
        ker_pi_u=((1, 0, 0), (0, 0, 1)),                # span{e_t, e_j}
        r_time=(),
        r_space=(),
        stationary_ops=("U",),
        l1_bytes=1280,
        halo_footprint=(("U", (1, 1)),),
        herd={"npu1": ((2,), (1,)), "npu2": ((2,), (1,))},
        params=("U",),
        dependences=(((1, -1, 0), "RAW", "U"), ((1, 0, -1), "RAW", "U"),
                     ((1, 0, 0), "RAW", "U"), ((1, 0, 1), "RAW", "U"),
                     ((1, 1, 0), "RAW", "U")),
    ),
    # §6.4: axes = (i, j0, j1), extents (32, 4, 8); UCoord = (i, j)
    "w3": dict(
        axes=("i", "j0", "j1"),
        extents=(32, 4, 8),
        sigma=((1, 1, 0), (0, 0, 1)),                   # row (i + j0), then the appended j1
        pi=((0, 1, 0),),
        ker_pi=((1, 0, 0), (0, 0, 1)),
        pi_u=((0, 1),),                                 # Sπ_u = [e_j]
        ker_pi_u=((1, 0),),                             # span{e_i}
        r_time=(),
        r_space=(),
        stationary_ops=("S", "r"),                      # q is NOT stationary
        l1_bytes=232,                                   # M4 adds edge_in/edge_out for 240
        halo_footprint=(),
        herd={"npu1": ((4,), (1,)), "npu2": ((4,), (1,))},
        params=("q", "r", "S"),                         # read-only first (§5.6 invariant 6)
        dependences=(((0, 1), "RAW", "S"), ((1, 0), "RAW", "S"), ((1, 1), "RAW", "S")),
    ),
}


def _offsets(access):
    """Every offset in the three kernels is a constant; compare the constants."""
    assert all(o.is_constant for o in access.offsets), access
    return tuple(o.const for o in access.offsets)


@pytest.fixture(params=sorted(MAPPINGS), ids=sorted(MAPPINGS))
def workload(request):
    return request.param


@pytest.mark.parametrize("target", TARGETS)
def test_mapping_fields(workload, target):
    """Every `LegalMapping` field, against the numbers of 03-lld-M3-checker.md §6."""
    want = EXPECTED[workload]
    got = MAPPINGS[workload].legal(target=target)

    assert tuple(a.name for a in got.axes) == want["axes"]
    assert tuple(a.extent for a in got.axes) == want["extents"]
    for name in ("sigma", "pi", "ker_pi", "pi_u", "ker_pi_u", "r_time", "r_space",
                 "stationary_ops", "l1_bytes", "halo_footprint"):
        assert getattr(got, name) == want[name], name

    physical, repeats = want["herd"][target]
    assert got.physical_herd == physical
    assert got.repeats == repeats
    assert got.schedule.target == target
    assert got.schedule.grid is not None
    assert tuple(g // p for g, p in zip(got.schedule.grid, physical)) == repeats

    assert tuple(p.name for p in got.kernel.params) == want["params"]
    assert tuple((d.vector, d.kind, d.operand) for d in got.kernel.dependences) \
        == want["dependences"]


def test_two_frames(workload):
    """The two-frame rule of 03-lld-M3-checker.md §3.1, asserted on every literal."""
    got = MAPPINGS[workload].legal()
    n_coord, n_ucoord = len(got.axes), len(got.kernel.axes)
    for name in ("sigma", "pi", "ker_pi"):
        assert all(len(row) == n_coord for row in getattr(got, name)), name
    for name in ("pi_u", "ker_pi_u", "r_time", "r_space"):
        assert all(len(row) == n_ucoord for row in getattr(got, name)), name
    for statement in got.kernel.statements:
        for access in (statement.target, *statement.reads):
            assert all(len(row) == n_ucoord for row in access.matrix), access
    assert len(got.pi) == len(got.pi_u) == len(got.schedule.grid)


def test_tiled_axes_carry_their_parent(workload):
    """`03-lld-M2-schedule.md` §3.2: a tile handle is `<parent>0` / `<parent>1`."""
    for axis in MAPPINGS[workload].legal().axes:
        if axis.parent is None:
            continue
        assert axis.name in (axis.parent + "0", axis.parent + "1")
        assert axis.step.is_constant and axis.step.const == 1


@pytest.mark.parametrize("target", TARGETS)
def test_mapping_json_round_trip(workload, target):
    """Every literal survives `06-interfaces.md` §8's canonical JSON unchanged."""
    got = MAPPINGS[workload].legal(target=target)
    assert from_json(to_json(got), LegalMapping) == got


# --------------------------------------------------------------------------------------------
# Access maps — 03-lld-M1-frontend.md §6, one test per workload
# --------------------------------------------------------------------------------------------


def test_w1_accesses():
    """§6.1's table: columns are `(i, j, k)`, every offset 0."""
    statement, = w1_legal.kernel().statements
    assert (statement.kind, statement.op, statement.line) == ("accumulate", "+", 8)
    assert statement.axes == ("i", "j", "k")
    assert statement.target.operand == "C"
    assert statement.target.matrix == ((1, 0, 0), (0, 1, 0))        # C[i,j]
    assert _offsets(statement.target) == (0, 0)
    assert tuple(r.operand for r in statement.reads) == ("A", "B")
    assert statement.reads[0].matrix == ((1, 0, 0), (0, 0, 1))      # A[i,k]
    assert statement.reads[1].matrix == ((0, 0, 1), (0, 1, 0))      # B[k,j]
    assert all(_offsets(r) == (0, 0) for r in statement.reads)
    reduction = w1_legal.kernel().reduction
    assert reduction is not None
    assert (reduction.target, reduction.projection, reduction.space, reduction.op) \
        == ("C", ((1, 0, 0), (0, 1, 0)), ((0, 0, 1),), None)        # R = ker Sf = span{e_k}


def test_w1_flip_shares_the_kernel_text():
    """FR-K2: the flip is a schedule edit, not a source edit."""
    assert w1flip_legal.kernel() == w1_legal.kernel()
    assert w1flip_legal.schedule().tiles == (("i", 32), ("k", 16))   # no tile on j (RULING 9)
    assert w1flip_legal.schedule().place == ("k0",)
    assert w1flip_legal.schedule().stationary == ("B",)
    assert w1flip_legal.schedule().double_buffer == ("A",)


def test_w2_accesses():
    """§6.2: every access to `U` has `matrix = I3`; the write offset is `(1,0,0)`."""
    statement, = w2_legal.kernel().statements
    assert (statement.kind, statement.op, statement.line) == ("assign", None, 8)
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert statement.target.matrix == identity
    assert _offsets(statement.target) == (1, 0, 0)
    assert all(r.matrix == identity and r.operand == "U" for r in statement.reads)
    assert tuple(_offsets(r) for r in statement.reads) \
        == ((0, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    assert w2_legal.kernel().reduction is None


def test_w3_accesses():
    """§6.3: `S` is assigned, not accumulated, so `reduction is None` and `R = {}`."""
    statement, = w3_legal.kernel().statements
    assert (statement.kind, statement.op, statement.line) == ("assign", None, 8)
    assert statement.target.matrix == ((1, 0), (0, 1))
    assert _offsets(statement.target) == (0, 0)
    by_operand = {}
    for read in statement.reads:
        by_operand.setdefault(read.operand, []).append((read.matrix, _offsets(read)))
    assert by_operand["S"] == [(((1, 0), (0, 1)), (-1, -1)),
                               (((1, 0), (0, 1)), (-1, 0)),
                               (((1, 0), (0, 1)), (0, -1))]
    assert by_operand["q"] == [(((1, 0),), (-1,))]
    assert by_operand["r"] == [(((0, 1),), (-1,))]
    assert w3_legal.kernel().reduction is None


# --------------------------------------------------------------------------------------------
# The W2 parametrisation — M4 needs T = 5 and PI = 4
# --------------------------------------------------------------------------------------------


def test_w2_defaults_reproduce_the_document():
    """`legal("npu1", T=4, PI=2)` is `03-lld-M3-checker.md` §6.3 exactly."""
    assert w2_legal.legal("npu1", T=4, PI=2) == w2_legal.legal("npu1")


@pytest.mark.parametrize("T,PI,extents,l1,grid", [
    (4, 2, (4, 2, 8, 14), 1280, (2,)),      # the fixture
    (5, 2, (5, 2, 8, 14), 1280, (2,)),      # the odd-T peel, FR-L14
    (0, 2, (0, 2, 8, 14), 1280, (2,)),      # the SWAP-PARITY negative
    (4, 4, (4, 4, 4, 14), 768, (4,)),       # the DMA-CHANNELS negative, PI = 4
])
def test_w2_parametrisation(T, PI, extents, l1, grid):
    """`T` moves the `t` extent and the plane count; `PI` moves the grid and `HS = 16 // PI`."""
    got = w2_legal.legal("npu1", T=T, PI=PI)
    assert tuple(a.extent for a in got.axes) == extents
    assert got.l1_bytes == l1
    assert got.schedule.grid == grid
    assert got.schedule.tiles == (("i", 16 // PI),)             # PI·HS == H == 16
    assert got.kernel.params[0].shape == (T + 1, 18, "W")       # U carries the halo
    assert f"T = {T};" in got.kernel.source


# --------------------------------------------------------------------------------------------
# The W1 MappingPlan literal — 03-lld-M4-mapping.md §6.1
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_w1_plan_constructs_and_round_trips(target):
    """M0's invariants accept the plan, and it survives canonical JSON."""
    got = w1_plan.plan(target)
    assert from_json(to_json(got), MappingPlan) == got
    assert (got.launch_name, got.segment_name) == ("gemm", "gemm_seg")
    assert got.herd.name == "gemm_herd"
    assert got.herd.grid == (2, 2)
    assert got.herd.shape == got.mapping.physical_herd
    assert got.herd.at is None
    assert got.herd.coords == ("tx", "ty")
    assert tuple(t.name for t in got.tensors) == ("A", "B", "C")
    assert all(t.level == "L3" and t.scope == "tensor" for t in got.tensors)
    assert got.delivery == (("A", "MULTICAST", "py", False),
                            ("B", "MULTICAST", "px", False),
                            ("C", "STATIONARY", None, False))


def test_w1_plan_summary_first_six_lines():
    """The six lines of `03-lld-M4-mapping.md` §6.1, verbatim."""
    assert w1_plan.plan().summary.lines[:6] == (
        "A: multicast along py (derived)",
        "B: multicast along px (derived)",
        "C: stationary (derived)",
        "A: multicast along py, re-fetched per k0",
        "B: multicast along px, re-fetched per k0",
        "C: stationary (spatial), resident for the whole run",
    )


def test_w1_plan_channel_table():
    """§6.1's channel table, and `_channel.py:137-145`'s `broadcast_shape % size == 0`."""
    got = w1_plan.plan()
    assert tuple((c.name, c.size, c.broadcast_shape, c.channel_type) for c in got.channels) == (
        ("A2L1", (2, 1), (2, 2), None),
        ("B2L1", (1, 2), (2, 2), None),
        ("C2L3", (2, 2), None, None),
    )
    assert got.summary.channels == (("A2L1", (2, 1), (2, 2)),
                                    ("B2L1", (1, 2), (2, 2)),
                                    ("C2L3", (2, 2), None))
    for channel in got.channels:
        if channel.broadcast_shape is not None:
            assert all(fan % one == 0
                       for fan, one in zip(channel.broadcast_shape, channel.size))


def test_w1_plan_buffers_and_l1():
    """§6.1's buffer table: `acc` 4096 + `a` 2·2048 + `b` 2·2048 = 12 288 of 65 536."""
    got = w1_plan.plan()
    assert tuple((b.name, b.operand, b.shape, b.bytes, b.loop_depth, b.ping_pong_candidate)
                 for b in got.buffers) == (
        ("acc", "C", (32, 32), 4096, 0, False),
        ("a", "A", (32, 16), 2048, 1, True),
        ("b", "B", (16, 32), 2048, 1, True),
    )
    charged = sum(b.bytes * (2 if b.ping_pong_candidate else 1) for b in got.buffers)
    assert charged == got.mapping.l1_bytes == got.summary.l1_bytes == 12288
    assert got.summary.l1_budget == 65536


def test_w1_plan_regions():
    """§3.4: the L1 end is the empty region; the L3 end carries offsets/sizes/strides."""
    got = w1_plan.plan()
    sites = {s.id: s for c in got.channels for s in c.sites}

    a_put = sites["A2L1.put.0@segment"]
    assert a_put.buffer == "A" and a_put.scope == "segment"
    assert a_put.region.sizes == (32, 16) and a_put.region.strides == (64, 1)
    assert [(dict(o.coeffs), o.const) for o in a_put.region.offsets] \
        == [({"pi_bundle": 32}, 0), ({"k0": 1}, 0)]

    for site_id in ("A2L1.get.2@herd", "B2L1.get.3@herd", "C2L3.put.3@herd"):
        assert sites[site_id].region == w1_plan.EMPTY, site_id

    assert all(s.is_async is False and s.depends_on == () and s.guard is None
               for s in sites.values())
