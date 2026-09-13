"""Level U — mapping and protocol synthesis. Spec: design/03-lld-M4-mapping.md §7.

The rows of M4 §7 this phase can honour: everything the W1 fill/compute/drain path, the W3
wavefront, the general machinery (`CLASSIFY`, `TENSOR_PLAN`, `TILE_SHAPE`, `L3_REGION`,
`RESIDENCY`, `SUMMARY`) and the `classify()`-level half of the flip can carry. The halo and
cascade rows wait on P5/P6, and `test_M4_unbuilt_protocols_fail_legibly` is what keeps their
absence loud rather than silent.

Written by B at P2 together with `spatial/m4_mapping.py`; extended at P4 with §3.6.2.
"""

from __future__ import annotations

import ast
import importlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from spatial import m4_mapping as m4, m4_selfcheck as selfcheck
from spatial.model import (BinOp, ChannelPlan, Const, Dtype, Expr, Load, LoopPlan, MappingError,
                           Select, StoreNode, StreamClause, to_json)
from tests.fixtures.mappings import w1_legal, w1flip_legal, w2_legal, w3_legal
from tests.fixtures.plans import w1_plan
from tests.helpers import determinism, plan_interp
from tests.helpers.diagnostics import assert_diagnostic
from tests.helpers.golden import assert_golden

TARGETS = ("npu1", "npu2")

SOURCES = {name: Path(module.__file__).read_text(encoding="utf-8")
           for name, module in (("m4_mapping", m4),
                                ("m4_selfcheck", importlib.import_module("spatial.m4_selfcheck")))}
"""M4's own source, for the import lint (invariant I-1, FR-S20)."""


WORKLOADS = {"w1": w1_legal, "w3": w3_legal}
"""The workloads whose whole plan M4 builds in this cut (the flip and W2 land at P5/P6)."""


def plan_json(target: str = "npu1", workload: str = "w1") -> str:
    """Module-level and picklable, so `determinism.in_fresh_process` can call it (FR-M12)."""
    return to_json(m4.plan(WORKLOADS[workload].legal(target)))


def imports_after_importing_m4() -> tuple[bool, bool]:
    """`(air imported, m5 imported)` after importing M4 in a fresh process (I-1, FR-S20)."""
    importlib.import_module("spatial.m4_mapping")
    importlib.import_module("spatial.m4_selfcheck")
    return ("air" in sys.modules, "spatial.m5_emit" in sys.modules)


# --------------------------------------------------------------------------------------------
# FR-M1, FR-M2, FR-M3 — the reuse trichotomy, its geometry and its override
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-M1")
def test_M1_trichotomy():
    """W1: `A` multicasts along `py` and `B` along `px`, both derived; `C` is stationary.

    `C`'s row is **declared**: `stationary("C")` names its delivery, and `declared` says a
    clause named it, not that the derivation needed the clause (the ruling on **B-P17**). `A`
    and `B` are named by no clause — `double_buffer` is not a delivery clause — so they stay
    derived, and the summary's point survives: two derived rows and one declared.
    """
    assert m4.classify(w1_legal.legal()) == (("A", "MULTICAST", "py", False),
                                             ("B", "MULTICAST", "px", False),
                                             ("C", "STATIONARY", None, True))
    assert m4.plan(w1_legal.legal()).delivery == w1_plan.DELIVERY
    assert m4.declared_operands(w1_legal.legal()) == {"C"}
    # an exchange() clause names its operand's delivery too, and a stream() clause does both
    assert m4.declared_operands(w2_legal.legal()) == {"U"}
    assert m4.declared_operands(w3_legal.legal()) == {"S"}


@pytest.mark.fr("FR-M1")
def test_M1_trichotomy_flip():
    """The flip, at `classify()` level: 1-D herd, `place(px=ax.k0)`, `j` untiled (§6.2).

    `B` is `declared` — `stationary("B")` names it and it is not the reduction target — and the
    containment `ker M_B = span{e_i} ⊆ ker Sπ_u = span{e_i, e_j}` holds in `UCoord`, which is
    what closed B-O5. `C` cascades along `px` because `r_space = span{e_k}` is carried by the
    placed `k0`. The full-plan half waits on the §3.6.3 builder (P6).
    """
    assert m4.classify(w1flip_legal.legal()) == (("A", "STATIONARY", None, False),
                                                 ("B", "STATIONARY", None, True),
                                                 ("C", "CASCADE", "px", False))


@pytest.mark.fr("FR-M2")
def test_M2_broadcast_shape():
    """`A2L1` is `size=(2,1)`, `broadcast_shape=(2,2)`; `B2L1` is the mirror (§3.2, §3.5)."""
    channels = {c.name: c for c in m4.plan(w1_legal.legal()).channels}
    assert (channels["A2L1"].size, channels["A2L1"].broadcast_shape) == ((2, 1), (2, 2))
    assert (channels["B2L1"].size, channels["B2L1"].broadcast_shape) == ((1, 2), (2, 2))
    assert channels["C2L3"].broadcast_shape is None
    # the geometry itself, away from W1's square grid
    assert m4.multicast_geometry((3, 4), 0) == ((1, 4), (3, 4))
    assert m4.multicast_geometry((3, 4), 1) == ((3, 1), (3, 4))


@pytest.mark.fr("FR-M2")
def test_M2_broadcast_multiple():
    """`broadcast_shape[d] % size[d] == 0` is rejected at construction (`_channel.py:137-145`).

    M0 enforces it (invariant I46), so a plan carrying `size=(3,1)` with
    `broadcast_shape=(2,2)` cannot be built at all, let alone emitted.
    """
    with pytest.raises(ValueError) as excinfo:
        ChannelPlan(name="Bad", size=(3, 1), broadcast_shape=(2, 2), channel_type=None,
                    chain_direction=None, dtype=Dtype.f32, sites=())
    assert "broadcast_shape" in str(excinfo.value)


@pytest.mark.fr("FR-M3")
def test_M3_stream_override():
    """A `stream()` clause replaces the derived row and marks it `declared` (§3.2 lines 19-21).

    Two halves. At `classify()` level the W1 schedule plus
    `stream("A", pattern="forward", along=ax.j0)` makes `A` a `FORWARD`, and its `along` is the
    **PE axis** `py` that carries `j0`, not the schedule axis the clause names — the ruling on
    **B-P24**, so one column of `MappingPlan.delivery` does not mean two different things.

    At plan level it is a `PROTOCOL-UNSUPPORTED` (ruling **R-W3-4**): the clause asks for tile
    forwarding of a *read* operand along a PE axis, and §3.6.2's wavefront forwards the written
    operand's edge scalar. There is no synthesis rule for the other shape in this cut, so the
    diagnostic names the clause rather than producing a plan that is not what was asked for.
    """
    mapping = w1_legal.legal()
    clause = StreamClause(operand="A", pattern="forward", along="j0", direction=None, depth=None)
    streamed = replace(mapping, schedule=replace(mapping.schedule, streams=(clause,)))
    rows = m4.classify(streamed)
    assert rows[0] == ("A", "FORWARD", "py", True)
    assert rows[1:] == (("B", "MULTICAST", "px", False), ("C", "STATIONARY", None, True))

    with pytest.raises(MappingError) as excinfo:
        m4.plan(streamed)
    assert_diagnostic(excinfo, code="PROTOCOL-UNSUPPORTED",
                      clause='stream("A", pattern="forward", along=ax.j0)',
                      mentions=("read-only", "§3.6.2"), details_keys=("operand",))
    # ...and an `along` that no `place()` carries is named, rather than silently indexed
    unplaced = replace(mapping, schedule=replace(
        mapping.schedule, streams=(replace(clause, along="k0"),)))
    with pytest.raises(MappingError) as excinfo:
        m4.classify(unplaced)
    assert_diagnostic(excinfo, code="PROTOCOL-UNSUPPORTED",
                      clause='stream("A", pattern="forward", along=ax.k0)',
                      mentions=("placed",), details_keys=("along", "place"))


# --------------------------------------------------------------------------------------------
# FR-M7 — the buffer plan and the L3 interface
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-M7")
def test_M7_buffer_plan():
    """`acc` is depth 0 and not ping-pong; `a` and `b` are depth 1 and are (§6.1's table)."""
    buffers = {b.name: b for b in m4.plan(w1_legal.legal()).buffers}
    assert [b.name for b in m4.plan(w1_legal.legal()).buffers] == ["acc", "a", "b"]
    assert (buffers["acc"].shape, buffers["acc"].loop_depth,
            buffers["acc"].ping_pong_candidate) == ((32, 32), 0, False)
    assert (buffers["a"].shape, buffers["a"].loop_depth,
            buffers["a"].ping_pong_candidate) == ((32, 16), 1, True)
    assert (buffers["b"].shape, buffers["b"].loop_depth,
            buffers["b"].ping_pong_candidate) == ((16, 32), 1, True)
    assert all(b.scope == "herd.private" and b.level == "L1" for b in buffers.values())
    # the shapes are computed from the AccessMap and the tile factors, not tabulated
    assert (m4.tile_shape(w1_legal.legal(), "A"), m4.tile_shape(w1_legal.legal(), "B"),
            m4.tile_shape(w1_legal.legal(), "C")) == ((32, 16), (16, 32), (32, 32))
    assert m4.tile_shape(w1flip_legal.legal(), "B") == (16, 64)   # j untiled on the flip


@pytest.mark.fr("FR-M7")
def test_M7_scope_never_shared():
    """No plan scope is `"herd.shared"` (it raises, `_trace.py:1405-1414`) or per-core."""
    plan = m4.plan(w1_legal.legal())
    scopes = {b.scope for b in plan.buffers + plan.tensors}
    assert scopes == {"herd.private", "tensor"}


@pytest.mark.fr("FR-M7")
@pytest.mark.parametrize(("mapping", "expected"), [
    (w1_legal.legal(), ("A", "B", "C")),
    (w1flip_legal.legal(), ("A", "B", "C")),
    (w2_legal.legal(), ("U",)),
    (w3_legal.legal(), ("q", "r", "S")),
], ids=["w1", "flip", "w2", "w3"])
def test_M4_tensor_order(mapping, expected):
    """Every read-only param before every written one (§5.6 invariant 6, §3.3 note 6).

    `_check_interface` raises "output tensors must be declared after all input tensors"
    otherwise (`_compile.py:226-240`), so W3's `(q, r, S)` is the case that bites.
    """
    tensors = m4.tensor_plan(mapping)
    assert tuple(t.name for t in tensors) == expected
    if mapping.kernel.name in ("gemm", "sw") and not mapping.r_space:
        # ...and at plan level, which is the order `air.tensor` is declared in (§5.6 inv. 6)
        assert tuple(t.name for t in m4.plan(mapping).tensors) == expected
    assert all(t.level == "L3" and t.scope == "tensor" and t.operand == t.name for t in tensors)
    written = [t.name for t in tensors if any(p.name == t.name and p.is_written
                                              for p in mapping.kernel.params)]
    assert [t.name for t in tensors][-len(written):] == written
    # shapes are resolved through KernelModel.bindings (§2.7, CONTRACT_VERSION 4)
    assert all(all(isinstance(extent, int) for extent in t.shape) for t in tensors)




# --------------------------------------------------------------------------------------------
# FR-M8 — the channel plan and its regions
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-M8")
def test_M8_channel_plan_complete():
    """Every field populated, every site's region rank matching its buffer rank (W1)."""
    plan = m4.plan(w1_legal.legal())
    assert [c.name for c in plan.channels] == ["A2L1", "B2L1", "C2L3"]
    ranks = {b.name: len(b.shape) for b in plan.buffers + plan.tensors}
    for channel in plan.channels:
        assert channel.size and channel.dtype is Dtype.f32
        assert channel.channel_type is None and channel.chain_direction is None
        kinds = [site.kind for site in channel.sites]
        assert kinds.count("put") >= 1 and kinds.count("get") >= 1
        for site in channel.sites:
            assert len(site.indices) == len(channel.size)
            assert site.id == f"{channel.name}.{site.kind}.{site.order}@{site.scope}"
            if site.region.offsets:                       # the L3 end carries a real region
                assert len(site.region.offsets) == ranks[site.buffer]
            assert site.buffer in ranks


@pytest.mark.fr("FR-M8")
def test_M8_regions():
    """`A2L1`'s put region is `((pi_bundle*32, k0), (32,16), (64,1))` — §3.4's worked example."""
    sites = {(c.name, s.kind): s for c in m4.plan(w1_legal.legal()).channels for s in c.sites}
    put = sites[("A2L1", "put")]
    assert put.region == w1_plan.A_PUT.region
    assert put.region.offsets[0].coeffs == (("pi_bundle", 32),)
    assert put.region.offsets[1].coeffs == (("k0", 1),)
    assert (put.region.sizes, put.region.strides) == ((32, 16), (64, 1))
    # the L1 end of a whole-buffer transfer is the empty region
    assert sites[("A2L1", "get")].region == m4.EMPTY_REGION
    assert sites[("C2L3", "put")].region == m4.EMPTY_REGION
    assert sites[("C2L3", "get")].region.strides == (64, 1)


@pytest.mark.fr("FR-M8")
def test_M4_loop_axis_names():
    """`06-interfaces.md` §5.5 at v5: bundle, drain and compute names (the **B-P18** ruling).

    A bundle-index loop is `p<root>_bundle`, a drain loop `<axis>_drain`, and a compute or
    zeroing nest is named by the post-tiling axis it realises — never positional `m`/`n`/`t`.
    The zeroing nest and the compute nest both realise `i1`/`j1`, so the name recurs in two
    bodies, which M5 rebinds by name in order.
    """
    mapping = w1_legal.legal()
    assert (m4.bundle_name(mapping, 0), m4.bundle_name(mapping, 1)) == ("pi_bundle", "pj_bundle")
    assert (m4.drain_name(mapping, 0), m4.drain_name(mapping, 1)) == ("i_drain", "j_drain")
    assert m4.compute_names(mapping) == {"i": "i1", "j": "j1", "k": "k1"}
    # the flip's `j` is untiled, so the nest realises `j` itself; its bundle root is `k`
    assert m4.compute_names(w1flip_legal.legal()) == {"i": "i1", "j": "j", "k": "k1"}
    assert m4.bundle_name(w1flip_legal.legal(), 0) == "pk_bundle"
    plan = m4.plan(mapping)
    axes = [node.axis for node in _loops(plan.segment_body) + _loops(plan.herd_body)]
    assert axes == ["pi_bundle", "k0", "pj_bundle", "k0", "i_drain", "j_drain",
                    "i1", "j1", "k0", "i1", "j1", "k1"]


def _loops(nodes):
    """Every `LoopPlan` of a body, outermost first."""
    out = []
    for node in nodes:
        if isinstance(node, LoopPlan):
            out += [node] + _loops(node.body)
    return out


@pytest.mark.fr("FR-M8")
def test_M4_compute_node_is_the_kernel_expression():
    """The store's value is `Statement.expr`, rewritten load by load into L1 (**B-P19**).

    W1's desugared accumulate `C[i,j] + A[i,k]*B[k,j]` becomes
    `acc[i1,j1] + a[i1,k1]*b[k1,j1]`: M4 rebuilds no arithmetic of its own, so the shape of the
    tree — and W2's `0.2 ×` sum and W3's `max`/`Select`, once their protocols land — is the
    kernel's, which is the only source there is for either (§2.4 at CONTRACT_VERSION 5).
    """
    plan = m4.plan(w1_legal.legal())
    i1, j1, k1 = Expr({"i1": 1}), Expr({"j1": 1}), Expr({"k1": 1})
    assert _loops(plan.herd_body)[-1].body[0] == StoreNode(
        buffer_id="acc", subscripts=(i1, j1),
        expr=BinOp("+", Load("acc", (i1, j1)),
                   BinOp("*", Load("a", (i1, k1)), Load("b", (k1, j1)))))
    # the zeroing nest writes the element the compute nest then accumulates into
    assert _loops(plan.herd_body)[1].body[0] == StoreNode(
        buffer_id="acc", subscripts=(i1, j1), expr=Const(0.0, "0.0", Dtype.f32))


@pytest.mark.fr("FR-M8")
def test_M4_l1_subscripts_rule():
    """`l1_subscripts` is the whole PE-relative rewrite, as a pure function (§3.4, FR-M8).

    Dim `d`'s index is the kernel subscript with each axis replaced by the element coordinate
    the nest realises, minus the staged slab's own origin along `d`. W1's placed `tx·TM` and
    streamed `k0` cancel exactly; a ghost-padded row band (W2) and a whole-width column band
    (W3) leave their own offset behind, which is the P4/P5 case this rule is general for.
    """
    elements = {"i": Expr({"tx": 8, "i1": 1}), "j": Expr({"j": 1})}
    tile, ghost, whole = (Expr({"tx": 8}),), (Expr({"tx": 8}, -1),), (Expr(),)
    assert m4.l1_subscripts((Expr({"i": 1}),), elements, tile, {}) == (Expr({"i1": 1}),)
    assert m4.l1_subscripts((Expr({"i": 1}),), elements, ghost, {}) == (Expr({"i1": 1}, 1),)
    assert m4.l1_subscripts((Expr({"i": 1}, -1),), elements, ghost, {}) == (Expr({"i1": 1}),)
    assert m4.l1_subscripts((Expr({"j": 1}, 1),), elements, whole, {}) == (Expr({"j": 1}, 1),)
    # a shape parameter in the subscript resolves through KernelModel.bindings first
    assert m4.l1_subscripts((Expr({"W": 1}),), elements, whole, {"W": 16}) == (Expr((), 16),)
    with pytest.raises(NotImplementedError, match="stages fewer dims"):
        m4.l1_subscripts((Expr({"i": 1}), Expr({"j": 1})), elements, whole, {})


# --------------------------------------------------------------------------------------------
# FR-M5 — the wavefront forward protocol (W3, §3.6.2 and §6.4)
# --------------------------------------------------------------------------------------------


def w3_plan(MQ: int = 32, target: str = "npu1"):
    """The derived W3 plan, at the fixture's `MQ` or an odd variant."""
    return m4.plan(w3_legal.legal(target, MQ=MQ))


def _sites(plan, channel: str):
    return [s for c in plan.channels if c.name == channel for s in c.sites]


@pytest.mark.fr("FR-M5")
@pytest.mark.parametrize("target", TARGETS)
def test_M5_wavefront_balance(target):
    """`WestIn[0]`, `West[0..2]`, `EastOut[0]`: five indices, each `MQ` puts and `MQ` gets.

    FR-M5's acceptance verbatim. The source and the drain are what close the chain at both ends
    — without them `West[0]`'s gets at PE 0 and `West[2]`'s puts at PE 3 would be unmatched, and
    the whole wavefront would be an unbalanced boundary (ruling **R-W3-1** keeps both).
    """
    plan = w3_plan(target=target)
    rows = {(row["channel"], tuple(row["index"])): (row["puts"], row["gets"])
            for row in selfcheck.balance_table(plan)
            if row["channel"] in ("WestIn", "West", "EastOut")}
    assert rows == {("WestIn", (0,)): (32, 32), ("West", (0,)): (32, 32),
                    ("West", (1,)): (32, 32), ("West", (2,)): (32, 32),
                    ("EastOut", (0,)): (32, 32)}
    assert len(rows) == w3_legal.NR // 8 + 1                       # PJ + 1 indices in total


@pytest.mark.fr("FR-M5", "FR-K4")
def test_M5_stages_q_and_r():
    """`q` multicasts whole, `r` is one column slice per PE, and the score is a `Select` (G-8).

    Without the staging the plan computes a literal `+2` for every cell and is not
    Smith-Waterman at all, so the test also asserts that no `Const 2` appears outside a `Select`
    arm.
    """
    plan = w3_plan()
    channels = {c.name: c for c in plan.channels}
    assert (channels["QIn"].size, channels["QIn"].broadcast_shape) == ((1,), (4,))
    assert (channels["RIn"].size, channels["RIn"].broadcast_shape) == ((4,), None)
    # one segment put for `q`, PJ for `r`; one herd get each, before the row loop
    assert [s.kind for s in channels["QIn"].sites] == ["put", "get"]
    assert _sites(plan, "RIn")[0].region.sizes == (8,)
    buffers = {b.name: b for b in plan.buffers}
    assert (buffers["qb"].shape, buffers["rb"].shape) == ((32,), (8,))
    store = plan.herd_body[9].body[2].body[0]
    select = store.expr.operands[1].rhs
    assert isinstance(select, Select) and select.cmp_op == "=="
    assert (select.lhs, select.rhs) == (Load("qb", (Expr({"i": 1}, -1),)),
                                        Load("rb", (Expr({"j1": 1}),)))
    assert (select.then, select.otherwise) == (Const(2, "2", Dtype.i32),
                                               Const(-1, "-1", Dtype.i32))
    outside = [node for node in _tree(store.expr) if isinstance(node, Const) and node.value == 2]
    assert outside == [select.then], "a literal +2 outside the Select is not Smith-Waterman"


def _tree(node):
    """Every node of an `ExprNode` tree, parents before children."""
    from dataclasses import fields as _fields

    out = [node]
    for field in _fields(node):
        value = getattr(node, field.name)
        for item in value if type(value) is tuple else (value,):
            if hasattr(item, "__dataclass_fields__") and not isinstance(item, Expr):
                out += _tree(item)
    return out


@pytest.mark.fr("FR-M5", "FR-K4")
def test_M5_drains_every_row():
    """`SOut` carries `MQ` puts per PE and its L3 regions are rows `1..MQ` × cols `1..NR` (G-9).

    Draining only the last row would leave `test_sem_coverage`'s (b) — "the claimed drain domain
    equals the kernel's write domain" — false by construction.
    """
    from itertools import product

    plan = w3_plan()
    rows = [row for row in selfcheck.balance_table(plan) if row["channel"] == "SOut"]
    assert [(row["puts"], row["gets"]) for row in rows] == [(32, 32)] * 4
    get = next(s for s in _sites(plan, "SOut") if s.kind == "get")
    drained = [plan_interp.region_indices(get.region, {"i_drain": i, "pj_bundle": p})
               for i, p in product(range(1, 33), range(4))]
    union = set().union(*drained)
    assert sum(len(part) for part in drained) == len(union) == 32 * 32
    assert union == set(product(range(1, 33), range(1, 33)))


@pytest.mark.fr("FR-M5")
def test_M5_three_channels():
    """Three homogeneous forward channels; none mixes an L3 and an L1 endpoint (finding N-2).

    A single `size=[PJ+1]` bundle carrying both is measured to fail
    `'airrt.dma_memcpy_nd' op failed to specialize channel bundle indices`
    (`AIRLoweringPass.cpp:798`), which is the whole reason FR-M5 names three.
    """
    plan = w3_plan()
    names = [c.name for c in plan.channels]
    assert names == ["EastOut", "QIn", "RIn", "SOut", "West", "WestIn"]
    forward = {c.name: c for c in plan.channels if c.name in ("WestIn", "West", "EastOut")}
    assert len(forward) == 3
    assert (forward["WestIn"].size, forward["West"].size,
            forward["EastOut"].size) == ((1,), (3,), (1,))
    for channel in plan.channels:
        core = selfcheck.is_core_to_core(channel, plan)
        l3 = selfcheck.l3_direction(channel, plan) is not None
        assert core != l3, f"{channel.name} mixes an L3 endpoint with a core-to-core one"
    assert selfcheck.is_core_to_core(forward["West"], plan)
    assert selfcheck.l3_direction(forward["WestIn"], plan) == "in"
    assert selfcheck.l3_direction(forward["EastOut"], plan) == "out"


@pytest.mark.fr("FR-M5")
def test_M4_swap_loop_peel():
    """An odd row count is **peeled**, not rejected (the override of D-4; §3.6.1 lines 7-9).

    The same `swap_loop` machinery W2's timestep swap will reuse at P5, exercised here on the
    row axis: 31 rows give a `1..31 step 2` loop of 15 two-row trips plus one straight-line trip
    at the same depth, and every wavefront index still sees 31 puts against 31 gets.
    """
    even, odd = w3_plan(MQ=32), w3_plan(MQ=31)
    assert (even.herd_body[9].hi, even.herd_body[9].step) == (Expr((), 33), Expr((), 2))
    assert len(even.herd_body) == 10, "an even row count needs no peel"
    assert (odd.herd_body[9].hi, odd.herd_body[9].step) == (Expr((), 31), Expr((), 2))
    assert len(odd.herd_body) == 16, "the peeled trip is six nodes at the loop's own depth"
    peeled = odd.herd_body[10:]
    assert [type(node).__name__ for node in peeled] == [
        "BranchNode", "StoreNode", "LoopPlan", "StoreNode", "BranchNode", "ChannelSite"]
    # the peeled row's index is the literal last row, because no loop binds it
    assert peeled[2].body[0].expr.operands[1].rhs.lhs == Load("qb", (Expr((), 30),))
    for row in selfcheck.balance_table(odd):
        if row["channel"] in ("WestIn", "West", "EastOut", "SOut"):
            assert (row["puts"], row["gets"]) == (31, 31), row
    # the pure function on its own, away from any workload
    trips = []
    nodes, phase = m4.swap_loop("t", 0, 5, "sequential", 0,
                                lambda index, ph, order: trips.append((index, ph, order)) or ())
    assert phase == 1 and len(nodes) == 1
    assert trips == [(Expr({"t": 1}), 0, 0),            # phase 0 of a trip runs row `t`
                     (Expr({"t": 1}, 1), 1, 0),         # phase 1 runs `t + 1`
                     (Expr((), 4), 0, 1)]               # the peel runs the literal last row
    assert (nodes[0].lo, nodes[0].hi, nodes[0].step) == (Expr((), 0), Expr((), 4), Expr((), 2))


@pytest.mark.fr("FR-M8")
def test_M4_wavefront_l1_subscripts():
    """The P2b table, checked on the real plan: the band is rank 1 where the access is rank 2.

    `_origins` gained the axis's own `lo` this phase (the caveat P2b recorded), which is what
    makes `8·tx` cancel when `j` starts at 1 rather than 0. The row offset picks the buffer —
    `−1` is `prev`, `0` is `cur` — and the column is the access's column minus the band origin.
    """
    plan = w3_plan()
    store = plan.herd_body[9].body[2].body[0]
    j1, i = Expr({"j1": 1}), Expr({"i": 1})
    assert (store.buffer_id, store.subscripts) == ("cur", (Expr({"j1": 1}, 1),))
    assert store.expr.operands[1].lhs == Load("prev", (j1,))           # S[i-1, j-1]
    assert store.expr.operands[2].lhs == Load("prev", (Expr({"j1": 1}, 1),))   # S[i-1, j]
    assert store.expr.operands[3].lhs == Load("cur", (j1,))            # S[i,   j-1]
    select = store.expr.operands[1].rhs
    assert select.lhs == Load("qb", (Expr({"i": 1}, -1),))             # q[i-1]
    assert select.rhs == Load("rb", (j1,))                             # r[j-1]
    # and the band geometry those follow from
    band = m4.band_geometry(w3_legal.legal(), plan.delivery, plan.herd)
    assert (band.row_axis, band.col_axis, band.row_dim, band.col_dim) == ("i", "j", 0, 1)
    assert (band.columns, band.ghost, band.width, band.lo, band.hi, band.col_lo) == (
        8, 1, 9, 1, 33, 1)
    assert i == Expr({"i": 1})                                          # the frame's row symbol


@pytest.mark.fr("FR-M7")
@pytest.mark.parametrize(("name", "mapping", "expected"), [
    ("w1", w1_legal.legal(), 12288),
    ("w3", w3_legal.legal(), 240),
], ids=["w1", "w3"])
def test_M4_l1_agrees_with_m3(name, mapping, expected):
    """The plan's L1 total is `LegalMapping.l1_bytes` plus the protocol buffers M4 adds (§7).

    W1 is `12288 == 12288` — it synthesises no protocol buffer. W3 is `240 == 232 + 8`:
    `edge_in` and `edge_out` are M4's, carry no `operand`, and are the only allowed difference
    from what M3 charged (`02-hld.md` §7, §3.3 note 4).
    """
    plan = m4.plan(mapping)
    staged = [b for b in plan.buffers if b.operand is not None]
    assert m4.l1_total(tuple(staged)) == mapping.l1_bytes
    assert m4.l1_total(plan.buffers) == expected == plan.summary.l1_bytes
    assert plan.summary.l1_budget == 65536
    extra = sum(b.bytes for b in plan.buffers if b.operand is None)
    assert expected == mapping.l1_bytes + extra
    assert (mapping.l1_bytes, extra) == {"w1": (12288, 0), "w3": (232, 8)}[name]


# --------------------------------------------------------------------------------------------
# FR-M11 — the summary and the residency block
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("workload", sorted(WORKLOADS), ids=sorted(WORKLOADS))
@pytest.mark.fr("FR-M11")
def test_M11_summary_golden(workload, target):
    """The summary matches its golden byte for byte, per target (B-P14's ruling).

    The herd line carries the *physical* shape and the repeats, which differ between npu1 and
    npu2, so one target-less file cannot hold both: the path is
    `<workload>.<variant>.<target>.summary.txt`.
    """
    summary = m4.plan(WORKLOADS[workload].legal(target)).summary
    assert_golden(f"{workload}.base.{target}.summary.txt", "\n".join(summary.lines) + "\n",
                  kind="text")
    expected = {"w1": ("C: stationary (declared)", "A: multicast along py (derived)",
                       "B: multicast along px (derived)"),
                "w3": ("S: forward along px (declared)", "q: multicast along px (derived)",
                       "r: stationary (derived)")}[workload]
    for line in expected:
        assert line in summary.lines


@pytest.mark.fr("FR-M11")
def test_M11_residency_line():
    """W1's residency block: `C` is resident for the whole run, `A`/`B` are re-fetched per `k0`.

    `T = (k0,)` is W1's only temporal tile axis — `i0` and `j0` are placed, `i1`/`j1`/`k1` run
    inside the tile — and `M_C · e_k = 0` while `M_A · e_k ≠ 0` (§3.9 lines 1-8).
    """
    summary = m4.plan(w1_legal.legal()).summary
    assert "C: stationary (spatial), resident for the whole run" in summary.lines
    assert "A: multicast along py, re-fetched per k0" in summary.lines
    assert summary.residency == w1_plan.RESIDENCY
    assert [a.name for a in m4.temporal_axes(w1_legal.legal())] == ["k0"]
    assert m4.residency(w1_legal.legal(), "C") == "resident for the whole run"
    # the flip's claim, at the residency level: only B stays put (RULING 9, FR-K2)
    flip = w1flip_legal.legal()
    assert m4.residency(flip, "B") == "resident for the whole run"
    assert m4.residency(flip, "A") == "re-fetched per i0"


# --------------------------------------------------------------------------------------------
# FR-M12, FR-S20 — determinism and the import discipline
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("workload", sorted(WORKLOADS), ids=sorted(WORKLOADS))
@pytest.mark.fr("FR-M12")
def test_M4_plan_stable(workload):
    """The same plan twice in one process, and once in a fresh one under another seed."""
    legal = WORKLOADS[workload].legal
    assert m4.plan(legal()) == m4.plan(legal())
    assert plan_json("npu1", workload) == determinism.in_fresh_process(
        plan_json, "npu1", workload)


@pytest.mark.fr("FR-S20")
def test_M4_no_air_no_m5_import():
    """M4 imports nothing from `air` and nothing from M5 (invariant I-1).

    The lint reads the import statements out of the AST rather than grepping the text, so a
    docstring naming `spatial.m5_emit` — which this module's does, to say it must not import it
    — is not a false positive; and a fresh process proves the module graph agrees.
    """
    for name, source in SOURCES.items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            for module in imported:
                assert module != "air" and not module.startswith("air."), f"{name}: {module}"
                assert "m5_emit" not in module, f"{name}: {module}"
    assert determinism.in_fresh_process(imports_after_importing_m4) == (False, False)


# --------------------------------------------------------------------------------------------
# The acceptance: the derived plan is the literal, and the unbuilt protocols say so
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_M4_plan_equals_literal(target):
    """`m4.plan(w1_legal.legal(t)) == w1_plan.plan(t)`, field for field (M7 §3.8)."""
    derived, literal = m4.plan(w1_legal.legal(target)), w1_plan.plan(target)
    if derived != literal:                                 # name the first differing path
        import dataclasses
        pytest.fail(_first_difference(dataclasses.asdict(derived), dataclasses.asdict(literal)))
    assert to_json(derived) == to_json(literal)


def _first_difference(left, right, path="plan"):
    """The first differing field path between two `asdict`ed plans, for a readable failure."""
    if type(left) is not type(right):
        return f"{path}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f"{path}.{key}: present on one side only"
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return ""
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}\n  derived={left}\n  literal={right}"
        for index, (one, other) in enumerate(zip(left, right)):
            found = _first_difference(one, other, f"{path}[{index}]")
            if found:
                return found
        return ""
    return "" if left == right else f"{path}: derived {left!r} != literal {right!r}"


@pytest.mark.fr("FR-S12")
def test_M4_reside_l2():
    """`reside(U="L2")` is `PROTOCOL-UNSUPPORTED`, by name (§3.3 note 5, RULING 7)."""
    mapping = w2_legal.legal()
    bad = replace(mapping, schedule=replace(mapping.schedule, residency=(("U", "L2"),)))
    with pytest.raises(MappingError) as excinfo:
        m4.plan(bad)
    assert_diagnostic(excinfo, code="PROTOCOL-UNSUPPORTED", clause='reside(U="L2")',
                      mentions=("reside", "L2"), details_keys=("operand", "level"))


@pytest.mark.parametrize(("mapping", "phrase"), [
    (w1flip_legal.legal(), "§3.6"),
    (w2_legal.legal(), "§3.6.1"),
], ids=["flip-cascade", "w2-halo"])
def test_M4_unbuilt_protocols_fail_legibly(mapping, phrase):
    """A protocol this cut does not build fails as a `MappingError` naming the phase (§5).

    §3.1's `PLAN` wrapper re-raises anything that is not already a `SpatialError` with the
    original in `details["internal_exception"]`, so an unbuilt path cannot silently produce a
    wrong plan.
    """
    with pytest.raises(MappingError) as excinfo:
        m4.plan(mapping)
    assert_diagnostic(excinfo, code="PROTOCOL-UNSUPPORTED", clause="plan()",
                      mentions=("lands in P4/P5/P6", phrase),
                      details_keys=("internal_exception", "workload"))
    assert "NotImplementedError" in excinfo.value.diagnostic.details["internal_exception"]
