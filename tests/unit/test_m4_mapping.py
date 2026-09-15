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
from spatial.model import (BinOp, ChannelPlan, Const, Dtype, Expr, Guard, Load, LoopPlan,
                           MappingError, Select, StoreNode, StreamClause, to_json)
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


WORKLOADS = {"w1": w1_legal, "flip": w1flip_legal, "w2": w2_legal, "w3": w3_legal}
"""Every workload whose whole plan M4 builds — all four since P6's cascade builder."""

GOLDEN = {"w1": "w1.base", "flip": "w1.flip", "w2": "w2.base", "w3": "w3.base"}
"""`<workload>.<variant>` per fixture (`06-interfaces.md` §8): the flip is W1's `flip` variant."""


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
    placed `k0`. The plan-level half is the same tuple, now that §3.6.3's builder exists.
    """
    expected = (("A", "STATIONARY", None, False),
                ("B", "STATIONARY", None, True),
                ("C", "CASCADE", "px", False))
    assert m4.classify(w1flip_legal.legal()) == expected
    plan = m4.plan(w1flip_legal.legal())
    assert plan.delivery == expected
    # ...and the delivery block of the summary says it in words (§3.9 lines 3-8)
    assert plan.summary.lines[:3] == ("A: stationary (derived)", "B: stationary (declared)",
                                      "C: cascade along px (derived)")


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
@pytest.mark.parametrize("workload", sorted(WORKLOADS), ids=sorted(WORKLOADS))
def test_M7_scope_never_shared(workload):
    """No plan scope is `"herd.shared"` (it raises, `_trace.py:1405-1414`) or per-core."""
    plan = m4.plan(WORKLOADS[workload].legal())
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
    """`A2L1`'s put region is `((pi_bundle*32, k0), (32,16), (64,1))` — §3.4's worked example.

    §7's row also names W2's `ToNorth` put: `((1,0),(1,W),(W,1))` on the `(HS+2, W)` strip — the
    L1 half of §3.4's "a partial L1 region (the halo rows) is the same computation against the
    buffer shape".
    """
    halo = m4.plan(w2_legal.legal())
    north = next(s for c in halo.channels if c.name == "ToNorth" for s in c.sites
                 if s.kind == "put")
    assert (north.region.offsets, north.region.sizes, north.region.strides) == (
        (Expr((), 1), Expr((), 0)), (1, 16), (16, 1))
    assert north.buffer == "cur" and north.region.sizes[1] == 16

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
    ("w1", w1_legal.legal(), 16384),
    ("w2", w2_legal.legal(), 1280),
    ("w3", w3_legal.legal(), 240),
], ids=["w1", "w2", "w3"])
def test_M4_l1_agrees_with_m3(name, mapping, expected):
    """The plan's L1 total is `LegalMapping.l1_bytes` plus the protocol buffers M4 adds (§7).

    W1 is `16384 == 16384` — it synthesises no protocol buffer, and npu1 (the literal's default
    target) folds its 2×2 grid onto a (1, 2) herd, so **R-L1-3** charges each of the three
    buffers twice: 2·(4096 + 2048 + 2048). W2 is `1280 == 1280`: the halo
    adds **no** buffer of its own, and `double_buffer("U")` is the `cur`/`next` pair itself, not
    a doubling on top of it (R-W2-4, D-5's second meaning). W3 is `240 == 232 + 8`: `edge_in`
    and `edge_out` are M4's, carry no `operand`, and are the only allowed difference from what
    M3 charged (`02-hld.md` §7, §3.3 note 4).
    """
    plan = m4.plan(mapping)
    repeated = any(r > 1 for r in mapping.repeats)
    staged = [b for b in plan.buffers if b.operand is not None]
    assert m4.l1_total(tuple(staged), repeated=repeated) == mapping.l1_bytes
    assert m4.l1_total(plan.buffers, repeated=repeated) == expected == plan.summary.l1_bytes
    assert plan.summary.l1_budget == 65536
    extra = sum(b.bytes * (2 if repeated else 1)
                for b in plan.buffers if b.operand is None)
    assert expected == mapping.l1_bytes + extra
    assert (mapping.l1_bytes, extra) == {"w1": (16384, 0), "w2": (1280, 0),
                                         "w3": (232, 8)}[name]
    if name == "w1":                    # the same three buffers, on a herd that takes the grid
        assert m4.plan(w1_legal.legal("npu2")).summary.l1_bytes == 12288
    if name == "w2":                                    # R-W2-4: DEPTH 0, so PP() is False
        assert m4.depth(mapping, "U") == 0 and not m4.ping_pong(mapping, "U")
        assert "U" in mapping.schedule.double_buffer
        assert [(b.name, b.shape, b.ping_pong_candidate) for b in plan.buffers] == [
            ("cur", (10, 16), False), ("next", (10, 16), False)]


# --------------------------------------------------------------------------------------------
# FR-M4 — the halo exchange protocol (§3.6.1, §6.3) — W2
# --------------------------------------------------------------------------------------------

HS, WIDTH = 8, 16
"""W2's per-PE owned rows and full staged width at the fixture's `PI = 2` (`03-lld-M3` §6.3)."""


def w2_plan(T: int = 4, PI: int = 2, target: str = "npu1"):
    """The derived W2 plan, at the fixture's `T`/`PI` or a variant."""
    return m4.plan(w2_legal.legal(target, T=T, PI=PI))


def _steps(plan) -> list[tuple]:
    """Every `STEP` of a W2 plan as a six-node tuple, in execution order.

    §6.3's herd body is `cur, next, UIn.get, <seed copy>, t-loop[STEP, STEP][, peeled STEP]`,
    so the loop is at index 4 and anything after it is the odd-`T` peel.
    """
    loop = plan.herd_body[4]
    out = [loop.body[0:6], loop.body[6:12]]
    if len(plan.herd_body) > 5:
        out.append(tuple(plan.herd_body[5:]))
    return out


@pytest.mark.fr("FR-M4")
def test_M4_halo_protocol():
    """The per-`STEP` site list, in order, with its flags — §3.6.1's "not negotiable" order.

    `PUT(north) → PUT(south) → GET(north ghost) → GET(south ghost) → <update> → PUT(UOut)`,
    both boundary puts `is_async=True`, **both** ghost gets with `depends_on == ()`: a token
    from a put into its own get would serialise the exchange into a rendezvous and reintroduce
    the deadlock VF §C's E1 verdict cleared it of (and `test_I_w2_no_put_get_token_edge`
    measures that the lowering agrees).
    """
    plan = w2_plan()
    assert len(plan.herd_body) == 5, [type(n).__name__ for n in plan.herd_body]
    for phase, step in enumerate(_steps(plan)):
        source, destination = ("cur", "next") if phase == 0 else ("next", "cur")
        assert [(s.kind, s.channel) for s in step if hasattr(s, "channel")] == [
            ("put", "ToNorth"), ("put", "ToSouth"), ("get", "ToNorth"), ("get", "ToSouth"),
            ("put", "UOut")]
        assert type(step[4]).__name__ == "LoopPlan", "the update nest sits at order 4"
        assert [s.order for s in step[:4]] == [step[0].order + k for k in range(4)]
        assert step[5].order == step[0].order + 5
        assert [s.is_async for s in step[:4]] == [True, True, False, False]
        assert all(s.depends_on == () for s in step[:4] + (step[5],))
        assert [s.buffer for s in step[:4]] == [source] * 4
        assert step[5].buffer == destination, "the drain reads the plane just computed"
        assert all(s.scope == "herd" for s in step[:4] + (step[5],))


@pytest.mark.fr("FR-M4")
def test_M4_halo_indices():
    """§3.6.1's index table verbatim: one bundle index per **physical link**, guards included."""
    plan = w2_plan()
    channels = {c.name: c for c in plan.channels}
    assert [c.name for c in plan.channels] == ["ToNorth", "ToSouth", "UIn", "UOut"]
    assert channels["ToNorth"].size == channels["ToSouth"].size == (2 - 1,)
    assert channels["UIn"].size == channels["UOut"].size == (2,)
    assert all(c.broadcast_shape is None and c.channel_type is None for c in plan.channels)
    north, south = _steps(plan)[0][0], _steps(plan)[0][1]
    ghost_n, ghost_s = _steps(plan)[0][2], _steps(plan)[0][3]
    assert (north.indices, north.guard) == ((Expr({"tx": 1}, -1),),
                                            Guard("tx", ">", Expr((), 0)))
    assert (ghost_n.indices, ghost_n.guard) == ((Expr({"tx": 1}),),
                                                Guard("tx", "<", Expr((), 1)))
    assert (south.indices, south.guard) == ((Expr({"tx": 1}),), Guard("tx", "<", Expr((), 1)))
    assert (ghost_s.indices, ghost_s.guard) == ((Expr({"tx": 1}, -1),),
                                                Guard("tx", ">", Expr((), 0)))
    # the four L1 regions: top owned row out, bottom owned row out, south ghost in, north ghost in
    assert [s.region.offsets[0].const for s in (north, south, ghost_n, ghost_s)] == [
        1, HS, HS + 1, 0]
    assert all(s.region.sizes == (1, WIDTH) for s in (north, south, ghost_n, ghost_s))
    # and the geometry is derived, not tabulated: PI = 4 gives 3 links and 6 owned rows
    strip = m4.strip_geometry(w2_legal.legal("npu1", T=4, PI=4), plan.delivery,
                              m4.resolve_herd(w2_legal.legal("npu1", T=4, PI=4)))
    assert (strip.pes, strip.owned, strip.halo, strip.shape) == (4, 4, 1, (6, 16))


@pytest.mark.fr("FR-M4", "FR-K3")
def test_M4_stencil_is_five_point():
    """The update is the **kernel's** five-term `0.2 ×` stencil, in source association.

    `BinOp("*", Const 0.2, <5-term sum>)` with the five `Load`s at `(0,0), (−1,0), (1,0),
    (0,−1), (0,1)` of the written element — M4 rebuilds no arithmetic, it rewrites
    `Statement.expr`'s loads into the strip (**B-P19**).
    """
    nest = _steps(w2_plan())[0][4]
    assert (nest.axis, nest.lo, nest.hi) == ("i1", Expr((), 0), Expr((), HS))
    assert (nest.body[0].axis, nest.body[0].lo, nest.body[0].hi) == ("j", Expr((), 1),
                                                                    Expr((), WIDTH - 1))
    store = nest.body[0].body[0]
    assert isinstance(store, StoreNode) and store.buffer_id == "next"
    assert store.subscripts == (Expr({"i1": 1}, 1), Expr({"j": 1}))
    assert isinstance(store.expr, BinOp) and store.expr.op == "*"
    assert store.expr.lhs == Const(value=0.2, text="0.2", dtype=Dtype.f32)
    loads, node = [], store.expr.rhs               # the sum is left-nested: ((((a+b)+c)+d)+e)
    while isinstance(node, BinOp):
        assert node.op == "+"
        loads.append(node.rhs)
        node = node.lhs
    loads.append(node)
    loads.reverse()
    assert len(loads) == 5 and all(load.buffer_id == "cur" for load in loads)
    write = store.subscripts
    assert [tuple(l.subscripts[d].const - write[d].const for d in (0, 1)) for l in loads] == [
        (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


@pytest.mark.fr("FR-M4", "FR-K3")
@pytest.mark.parametrize("T", [4, 5])
def test_M4_drains_every_plane(T):
    """`UOut` drains **every** plane the kernel writes: planes `1..T` × rows `1..H` (§6.3).

    Draining only the final strip would make `test_sem_coverage` fail by construction — the
    union of the written index sets would be one plane against the kernel's `T` (G-10). The
    drained box is a superset of the write domain in the **column** direction only, because the
    strip is put back whole; §6.3's coverage paragraph is that sentence.
    """
    plan = w2_plan(T=T)
    get = next(s for c in plan.channels if c.name == "UOut" for s in c.sites
               if s.scope == "segment")
    drained = [plan_interp.region_indices(get.region, {"t_drain": t, "pi_bundle": p})
               for t in range(T) for p in range(2)]
    union = set().union(*drained)
    assert sum(len(part) for part in drained) == len(union) == T * 16 * WIDTH   # no overlap
    assert {point[0] for point in union} == set(range(1, T + 1))
    assert {point[1] for point in union} == set(range(1, 16 + 1))
    assert {point[2] for point in union} == set(range(WIDTH))                  # the whole width
    # one *transfer* per timestep per PE: two sites inside a two-phase trip, plus the peel
    puts = [s for c in plan.channels if c.name == "UOut" for s in c.sites if s.scope == "herd"]
    assert len(puts) == 2 + T % 2
    assert all(row["puts"] == row["gets"] == T for row in selfcheck.balance_table(plan)
               if row["channel"] == "UOut")


@pytest.mark.fr("FR-M4", "FR-L14")
def test_M4_odd_T_peel():
    """`T = 5`: the loop runs `0..4 step 2` and one straight-line `STEP` follows it (R-W2-2).

    The peel is an M4 decision, not an M5 one: `air.sequential` has no `iter_args`
    (`_loop.py:180`), so the swap cannot be loop-carried and the parity has to be resolved in
    plan space. Because the drain is **inside** each `STEP` there is no `<live>` buffer to
    choose — the peeled `STEP(cur → next)` drains plane `T` itself.
    """
    plan = w2_plan(T=5)
    loop = plan.herd_body[4]
    assert (loop.axis, loop.lo, loop.hi, loop.step) == ("t", Expr((), 0), Expr((), 4),
                                                        Expr((), 2))
    assert loop.kind == "sequential" and loop.depth == 0
    peel = plan.herd_body[5:]
    assert [type(node).__name__ for node in peel] == [
        "ChannelSite", "ChannelSite", "ChannelSite", "ChannelSite", "LoopPlan", "ChannelSite"]
    assert [node.order for node in peel if hasattr(node, "order")] == [5, 6, 7, 8, 10]
    # the peeled STEP is built by the same `trip` as an in-loop one, so its update nest keeps
    # the in-loop `depth = 1` although it now sits at body top level. `depth` is recorded and
    # never emitted (`06-interfaces.md` §5.5), and W3's peel has carried the same since P4.
    assert peel[4].depth == 1
    assert peel[0].buffer == "cur" and peel[5].buffer == "next", "the peel is a phase-0 STEP"
    for row in selfcheck.balance_table(plan):
        assert (row["puts"], row["gets"]) == ((1, 1) if row["channel"] == "UIn" else (5, 5)), row


@pytest.mark.fr("FR-M4")
def test_M4_even_T_no_peel():
    """`T = 4`: nothing follows the loop, and each PE makes exactly four `UOut` puts."""
    plan = w2_plan(T=4)
    assert len(plan.herd_body) == 5, "no peeled tail"
    assert (plan.herd_body[4].hi, plan.herd_body[4].step) == (Expr((), 4), Expr((), 2))
    puts = [s for c in plan.channels if c.name == "UOut" for s in c.sites if s.scope == "herd"]
    assert len(puts) == 2, "two sites, two trips: four transfers per PE"
    for row in selfcheck.balance_table(plan):
        assert (row["puts"], row["gets"]) == ((1, 1) if row["channel"] == "UIn" else (4, 4)), row


@pytest.mark.fr("FR-M4")
def test_M4_halo_seeds_both_strips():
    """`next` is seeded from `cur`, because §6.3's drain writes the staged boundary back.

    §3.6.1 stages plane `lo` into `cur` alone. §6.3's coverage paragraph then says of the drain
    that *"the value written back is the one that was staged in"* — a claim about **every**
    plane, and the planes alternate between the two strips, so both have to carry the staged
    read-only boundary. `next` is never a `get` target, so a `StoreNode` copy is what gives it
    those values; without it every second plane's boundary columns are whatever the alloc held.
    Recorded in `design/PROGRESS-B.md`, phase P5.
    """
    plan = w2_plan()
    copy = plan.herd_body[3]
    assert (copy.axis, copy.lo, copy.hi) == ("i1", Expr((), 0), Expr((), HS + 2))
    inner = copy.body[0]
    assert (inner.axis, inner.lo, inner.hi) == ("j", Expr((), 0), Expr((), WIDTH))
    assert inner.body[0] == StoreNode(buffer_id="next",
                                      subscripts=(Expr({"i1": 1}), Expr({"j": 1})),
                                      expr=Load("cur", (Expr({"i1": 1}), Expr({"j": 1}))))
    assert plan.herd_body[2].channel == "UIn" and plan.herd_body[2].buffer == "cur"
    assert plan.herd_body[2].region == m4.EMPTY_REGION, "the whole strip, ghosts included"


# --------------------------------------------------------------------------------------------
# FR-M6 — §3.6.3, the cascade chain (P6)
# --------------------------------------------------------------------------------------------


def _sites_of(plan, channel):
    """Every site of one channel, in plan order."""
    return next(c for c in plan.channels if c.name == channel).sites


@pytest.mark.fr("FR-M6", "FR-K2")
@pytest.mark.parametrize("target", TARGETS)
def test_M6_cascade_chain(target):
    """One `npu_cascade` bundle of `PK-1` links, ascending, 4 guarded stages (§6.2, §7).

    The payload is one `[32,64]` `f32` partial tile per link per `i0` trip — the whole
    accumulator, which is what `EMPTY_REGION` means (§3.4) — and the chain ascends in `tx`,
    which is the direction `aie.cascade_flow`'s verifier accepts on a 1-D herd (P-R3).
    """
    plan = m4.plan(w1flip_legal.legal(target))
    chan = next(c for c in plan.channels if c.channel_type is not None)
    assert (chan.name, chan.size, chan.broadcast_shape, chan.channel_type,
            chan.chain_direction) == ("CascadeK", (3,), None, "npu_cascade", "ascending")
    assert chan.dtype is Dtype.f32
    assert [c.name for c in plan.channels] == ["A2L1", "B2L1", "C2L3", "CascadeK"]

    # the four stages of §3.6.3 lines 19-23, as nested BranchNodes: conjunction is nesting
    loop = plan.herd_body[-1]
    assert (loop.axis, loop.kind, loop.lo.const, loop.hi.const, loop.step.const) == \
        ("i0", "sequential", 0, 64, 32)
    outer = loop.body[-1]
    assert outer.predicate == Guard(coord="tx", relation="==", value=Expr((), 0))
    assert [n.id for n in outer.then] == ["CascadeK.put.4@herd"]
    get, accumulate, inner = outer.otherwise
    assert (get.id, get.kind, get.buffer) == ("CascadeK.get.5@herd", "get", "recv")
    assert get.indices == (Expr({"tx": 1}, -1),)                 # the link below this PE
    assert outer.then[0].indices == (Expr({"tx": 1}),)           # the link above it
    assert inner.predicate == Guard(coord="tx", relation="==", value=Expr((), 3))
    assert [n.id for n in inner.then] == ["C2L3.put.6@herd"]
    assert [n.id for n in inner.otherwise] == ["CascadeK.put.7@herd"]
    assert {s.region for s in _sites_of(plan, "CascadeK")} == {m4.EMPTY_REGION}
    # the payload: one whole [32,64] f32 tile, 8192 B per link per trip
    recv = next(b for b in plan.buffers if b.name == m4.RECV)
    assert (recv.shape, recv.bytes, recv.operand, recv.loop_depth) == ((32, 64), 8192, None, 0)

    # ...and the accumulate M4 synthesises for it (R-F-2): the reduction operator comes from
    # the schedule, and the nest is explicit because there is no whole-buffer StoreNode.
    assert w1flip_legal.schedule().reductions == (("k", "+"),)
    i1, j = accumulate, accumulate.body[0]
    assert (i1.axis, i1.hi.const, j.axis, j.hi.const) == ("i1", 32, "j", 64)
    subs = (Expr({"i1": 1}), Expr({"j": 1}))
    assert j.body[0] == StoreNode(buffer_id="acc", subscripts=subs,
                                  expr=BinOp(op="+", lhs=Load("acc", subs),
                                             rhs=Load("recv", subs)))
    assert m4._reduce("max", Load("acc", subs), Load("recv", subs)).op == "maximum"


@pytest.mark.fr("FR-M6")
def test_M6_cascade_orientation():
    """A 2-D `(1,4)` herd descends; the 1-D `grid(4)` ascends (§3.6.3 lines 4-7).

    Both directions are measured on the pinned wheel: a 1-D herd is a row of columns and the
    chain runs west to east, a 2-D `(1,4)` herd is one column of rows and the chain must run
    from the higher row to the lower one, and the opposite of either fails
    `'aie.cascade_flow' op source tile must be to the North or West of the destination tile`
    (REVIEW-round1 P-R3, `03-lld-B-open-questions.md` §4). The sites mirror exactly: head and
    tail swap, and so do the put and get index expressions.
    """
    plan = m4.plan(w1flip_legal.legal(grid2d=True))
    chan = next(c for c in plan.channels if c.channel_type is not None)
    assert (chan.name, chan.size, chan.chain_direction) == ("CascadeK", (3,), "descending")
    assert plan.herd.grid == (1, 4) and plan.herd.coords == ("tx", "ty")
    assert m4.chain_geometry(w1flip_legal.legal(grid2d=True), plan.herd).axis == 1
    outer = plan.herd_body[-1]
    assert outer.predicate == Guard(coord="ty", relation="==", value=Expr((), 3))  # head at 3
    assert outer.then[0].indices == (Expr({"ty": 1}, -1),)       # put at coord - 1
    get, _accumulate, inner = outer.otherwise
    assert get.indices == (Expr({"ty": 1}),)                     # get at coord
    assert inner.predicate == Guard(coord="ty", relation="==", value=Expr((), 0))  # tail at 0
    assert inner.otherwise[0].indices == (Expr({"ty": 1}, -1),)
    # `C` still cascades, now along the second PE axis
    assert plan.delivery[2] == ("C", "CASCADE", "py", False)


@pytest.mark.fr("FR-M6", "FR-M7")
def test_M7_buffer_plan_flip():
    """§6.2's buffer table: `a` is the one ping-pong candidate, `b` is hoisted at depth 0.

    Allocation order is what `06-interfaces.md` §5.6 requires and what the herd body does:
    `b`, `acc`, `recv` outside the `i0` sweep, `a` inside it. Line 0 is deliberately **not**
    the ping-pong shape — hoisting `b` above the loop is the whole point of the flip.
    """
    plan = m4.plan(w1flip_legal.legal())
    assert [(b.name, b.shape, b.bytes, b.loop_depth, b.ping_pong_candidate)
            for b in plan.buffers] == [("b", (16, 64), 4096, 0, False),
                                       ("acc", (32, 64), 8192, 0, False),
                                       ("recv", (32, 64), 8192, 0, False),
                                       ("a", (32, 16), 2048, 1, True)]
    assert [b.operand for b in plan.buffers] == ["B", "C", None, "A"]
    assert all(b.scope == "herd.private" and b.level == "L1" for b in plan.buffers)
    # the herd body allocates them in exactly that order, `a` as a direct child of the loop
    assert [n.name for n in plan.herd_body if isinstance(n, type(plan.buffers[0]))] == \
        ["b", "acc", "recv"]
    assert plan.herd_body[-1].body[0].name == "a"
    assert plan.summary.l1_bytes == 24576 == 4096 + 8192 + 8192 + 2 * 2048


@pytest.mark.fr("FR-M6", "FR-M8")
def test_M6_cascade_bodies():
    """The fills, the drain and the herd body of §6.2, node by node (R-F-1).

    `B2L1` is one unrolled `pk_bundle` loop of four puts, **once**, hoisted above everything;
    `A2L1` is the same bundle loop around a 2-trip `air.sequential` over `i0`; the drain is
    `i0_drain`, **sequential** — `i0` is not a channel bundle index, so `LOOP_KIND` gives it
    `air.sequential` and `03-lld-M5-emitter.md` §6.2 line 37's Python loop is the slip.
    """
    plan = m4.plan(w1flip_legal.legal())
    fill_b, fill_a, herd, drain = plan.segment_body
    assert (fill_b.axis, fill_b.kind, fill_b.hi.const) == ("pk_bundle", "unrolled", 4)
    assert fill_b.body[0].channel == "B2L1" and len(fill_b.body) == 1
    assert fill_b.body[0].region.offsets == (Expr({"pk_bundle": 16}), Expr(()))
    assert fill_b.body[0].region.sizes == (16, 64)
    assert (fill_a.axis, fill_a.kind) == ("pk_bundle", "unrolled")
    assert (fill_a.body[0].axis, fill_a.body[0].kind, fill_a.body[0].step.const) == \
        ("i0", "sequential", 32)
    assert fill_a.body[0].body[0].region.offsets == (Expr({"i0": 1}), Expr({"pk_bundle": 16}))
    assert herd is plan.herd
    assert (drain.axis, drain.kind, drain.lo.const, drain.hi.const, drain.step.const) == \
        ("i0_drain", "sequential", 0, 64, 32)
    assert drain.body[0].channel == "C2L3" and drain.body[0].indices == (Expr(()),)
    assert drain.body[0].region.offsets == (Expr({"i0_drain": 1}), Expr(()))
    assert drain.body[0].region.sizes == (32, 64)
    # the herd body: `b` and its get, `acc`, `recv`, then the `i0` sweep
    assert plan.herd_body[1].channel == "B2L1" and plan.herd_body[1].order == 1
    assert plan.herd_body[1].indices == (Expr({"tx": 1}),)
    inner = plan.herd_body[-1].body
    assert inner[1].channel == "A2L1" and inner[1].order == 1
    zero, compute = inner[2], inner[3]
    assert (zero.axis, zero.body[0].axis) == ("i1", "j")
    assert zero.body[0].body[0].expr == Const(value=0.0, text="0.0", dtype=Dtype.f32)
    assert [compute.axis, compute.body[0].axis, compute.body[0].body[0].axis] == \
        ["i1", "j", "k1"]
    # the compute store is the kernel's own tree, rewritten into L1 (§6.2 line 8)
    store = compute.body[0].body[0].body[0]
    assert store == StoreNode(
        buffer_id="acc", subscripts=(Expr({"i1": 1}), Expr({"j": 1})),
        expr=BinOp(op="+", lhs=Load("acc", (Expr({"i1": 1}), Expr({"j": 1}))),
                   rhs=BinOp(op="*", lhs=Load("a", (Expr({"i1": 1}), Expr({"k1": 1}))),
                             rhs=Load("b", (Expr({"k1": 1}), Expr({"j": 1}))))))


# --------------------------------------------------------------------------------------------
# FR-M11 — the summary and the residency block
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("workload", sorted(WORKLOADS), ids=sorted(WORKLOADS))
@pytest.mark.fr("FR-M11", "FR-D2")
def test_M11_summary_golden(workload, target):
    """The summary matches its golden byte for byte, per target (B-P14's ruling).

    The herd line carries the *physical* shape and the repeats, which differ between npu1 and
    npu2, so one target-less file cannot hold both: the path is
    `<workload>.<variant>.<target>.summary.txt`.

    **FR-D2** rides along: its whole *Shall* clause is "see FR-M11", so this is the one test
    that accepts it, and `03-lld-M7-tests.md` §6.3 names this pair as one of the three that
    legitimately carry two ids. The marker was added at P7 by `test_traceability`.
    """
    summary = m4.plan(WORKLOADS[workload].legal(target)).summary
    assert_golden(f"{GOLDEN[workload]}.{target}.summary.txt", "\n".join(summary.lines) + "\n",
                  kind="text")
    expected = {"w1": ("C: stationary (declared)", "A: multicast along py (derived)",
                       "B: multicast along px (derived)"),
                # the four lines `05-work-breakdown.md` §5 step 3 reads off the screen, verbatim
                "flip": ("B: stationary (declared)",
                         "B: stationary (spatial), resident for the whole run",
                         "A: stationary (spatial), re-fetched per i0",
                         "reduction (tiled axes): R_time = span{e_k1}, R_space = span{e_k0}",
                         "  CascadeK size=(3,) type=npu_cascade"),
                "w2": ("U: stationary (declared)",
                       "U: stationary (spatial), resident for the whole run"),
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
    assert [a.name for a in m4.temporal_axes(flip)] == ["i0"]
    # ...and in the summary the demo points at (`05-work-breakdown.md` §5 step 3)
    flip_summary = m4.plan(flip).summary
    for line in ("A: stationary (spatial), re-fetched per i0",
                 "B: stationary (spatial), resident for the whole run",
                 "C: cascade along px, re-fetched per i0"):
        assert line in flip_summary.lines
    assert flip_summary.residency == (("A", "re-fetched per i0"),
                                      ("B", "resident for the whole run"),
                                      ("C", "re-fetched per i0"))
    # RULING 9's regression: restore `tile(ax.j, 32)` and `B`'s tile moves with `j0` again —
    # legally stationary in space, and re-fetched on every inner trip. That is the bug the
    # residency line exists to make visible, and why the flip leaves `j` whole.
    tiled = w1flip_legal.legal(tile_j=32)
    assert m4.residency(tiled, "B") == "re-fetched per j0"
    assert m4.tile_shape(tiled, "B") == (16, 32)


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


def _two_statements(mapping):
    """The same kernel with its one statement twice — a shape no protocol builder covers."""
    kernel = mapping.kernel
    return replace(mapping, kernel=replace(kernel, statements=kernel.statements * 2))


@pytest.mark.parametrize(("mapping", "phrase"), [
    (_two_statements(w1_legal.legal()), "fuses none of them"),
    (_two_statements(w1flip_legal.legal()), "fuses none of them"),
], ids=["w1-two-statements", "flip-two-statements"])
def test_M4_unbuilt_protocols_fail_legibly(mapping, phrase):
    """A shape M4 does not build fails as a `MappingError` naming what is out of the cut (§5).

    §3.1's `PLAN` wrapper re-raises anything that is not already a `SpatialError` with the
    original in `details["internal_exception"]`, so an unbuilt path cannot silently produce a
    wrong plan. All four protocol builders exist since P6, so the vehicle is a kernel shape
    rather than a protocol: two fused statements, which `_statement` refuses by name.
    """
    with pytest.raises(MappingError) as excinfo:
        m4.plan(mapping)
    assert_diagnostic(excinfo, code="PROTOCOL-UNSUPPORTED", clause="plan()",
                      mentions=("is out of this cut", phrase),
                      details_keys=("internal_exception", "workload"))
    assert "NotImplementedError" in excinfo.value.diagnostic.details["internal_exception"]
