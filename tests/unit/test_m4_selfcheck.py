"""Level U — the plan self-check. Spec: `design/03-lld-M4-mapping.md` §3.7, §3.8 and §7.

The `test_M9_*`, `test_M10_*`, `test_M11_*`, `test_M4_*` and `test_P3_*` rows of M4 §7 that this
phase can honour: everything driven by W1's, W3's and W2's real plans, by the six hand-corrupted
W1 literals of `tests/fixtures/corrupt/` and by the two synthetic hand plans built there on
`w2_legal`, and — since P6 — by the flip's own plan. Every `test_M9_*`/`test_P3_*` row of §7
now has a vehicle.

The synthetic halo plans are **kept** beside the real ones they now supersede: they are the same
shape with no protocol builder behind them, so a `BALANCE`, `CHANNEL-CYCLE` or `DMA-CHANNELS`
that fires on both says the checker caught it and not the builder.

Written by B at P3 together with `spatial/m4_selfcheck.py`; extended at P5 with W2's own plan.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from spatial import m4_mapping as m4
from spatial import m4_selfcheck as sc
from spatial.model import BranchNode, ChannelSite, Dtype, LoopPlan, MappingError
from tests.fixtures.corrupt import (branch_gets, broadcast_underconsumed, dma_packet_warn,
                                    dma_three_inbound, dropped_get, extra_put_in_loop, halo,
                                    halo_guard_dropped, halo_reversed, iv_bundle_index,
                                    pingpong_hoisted, tensor_order)
from tests.fixtures.mappings import w1_legal, w1flip_legal, w2_legal, w3_legal
from tests.fixtures.plans import w1_plan
from tests.helpers.diagnostics import assert_diagnostic

TARGETS = ("npu1", "npu2")


def raises(plan) -> MappingError:
    """`self_check(plan)` must raise; hand the caller the `pytest` `ExceptionInfo`."""
    with pytest.raises(MappingError) as excinfo:
        sc.self_check(plan)
    return excinfo


# --------------------------------------------------------------------------------------------
# FR-M9 — P1' balance (§3.7.1)
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-M9", "FR-M10")
@pytest.mark.parametrize("target", TARGETS)
def test_M9_selfcheck_accepts(target):
    """W1's real plan passes every check, on both targets, and records no warning (§7)."""
    plan = m4.plan(w1_legal.legal(target))
    assert sc.self_check(plan) is None
    assert sc.warnings(plan) == ()
    # the synthetic hand plans the negatives are cut from are themselves legal
    assert sc.self_check(halo.plan(2)) is None
    assert sc.self_check(branch_gets.plan()) is None


@pytest.mark.fr("FR-M9", "FR-M10", "FR-M5")
@pytest.mark.parametrize("target", TARGETS)
def test_M9_selfcheck_accepts_w3(target):
    """W3's real plan passes every check, with **exactly one** warning per PE and no error.

    The wavefront is the first plan with guarded branches, a peelable swap loop, a source and a
    drain on the same L3 tensor, and a per-core inbound count over the nominal budget — so it is
    the case each of §3.7's three checks and §3.8's split was written for.
    """
    plan = m4.plan(w3_legal.legal(target))
    assert sc.self_check(plan) is None
    messages = sc.warnings(plan)
    assert len(messages) == plan.herd.grid[0]                 # one per PE, inbound only
    assert all("packet" in message and "2 S2MM" in message for message in messages)
    assert all(len(row.hard) <= row.budget for row in sc.dma_report(plan))
    # and the warning reaches the summary, which is what §3.8 asks for
    assert [line for line in plan.summary.lines if line.startswith("warning: ")] == [
        f"warning: {message}" for message in messages]


def w2_plan(T: int = 4, PI: int = 2, target: str = "npu1"):
    """The derived W2 plan, at the fixture's `T`/`PI` or a variant."""
    return m4.plan(w2_legal.legal(target, T=T, PI=PI))


def _flat(nodes):
    """Every node of a plan body, descending into loops and both arms of a branch."""
    for node in nodes:
        yield node
        if isinstance(node, LoopPlan):
            yield from _flat(node.body)
        elif isinstance(node, BranchNode):
            yield from _flat(node.then)
            yield from _flat(node.otherwise)


def w2_with_body(plan, herd_body):
    """`plan` with a rewritten herd body, its `channels` re-pointed at the sites it now holds.

    `06-interfaces.md` §5.6 forbids one site id naming two different sites, so a corruption made
    with `dataclasses.replace` on a body node has to be reflected in `ChannelPlan.sites` too.
    """
    sites = {node.id: node for body in (plan.segment_body, herd_body)
             for node in _flat(body) if isinstance(node, ChannelSite)}
    return replace(plan, herd_body=herd_body,
                   channels=tuple(replace(channel,
                                          sites=tuple(sites[s.id] for s in channel.sites))
                                  for channel in plan.channels))


def w2_reversed(plan):
    """The real W2 plan with each `STEP` emitted `[GET, GET, PUT, PUT]` — §3.7.2's cycle.

    Built from the plan M4 derives, not from a hand-written stand-in: the site ids, buffers and
    regions are the real protocol's and only the **order** changes, which is precisely the one
    property §3.6.1 calls not negotiable.
    """
    loop = plan.herd_body[4]
    body = list(loop.body)
    for base in (0, 6):
        body[base:base + 4] = [body[base + 2], body[base + 3], body[base], body[base + 1]]
    return replace(plan, herd_body=plan.herd_body[:4] + (replace(loop, body=tuple(body)),))


@pytest.mark.fr("FR-M9", "FR-M10", "FR-M4")
@pytest.mark.parametrize("target", TARGETS)
def test_M9_selfcheck_accepts_w2(target):
    """W2's real plan passes every check, on both targets, and records **no** warning (§7).

    `PI = 2` is exactly at the 2-S2MM / 2-MM2S budget with every flow circuit-switched, so
    unlike W3 there is nothing for §3.8 to warn about — and `aircc` is measured to agree, six
    `aie.flow`s and zero `aie.packet_flow`s (`design/PROGRESS-B.md`, phase P5).
    """
    plan = w2_plan(target=target)
    assert sc.self_check(plan) is None
    assert sc.warnings(plan) == ()
    assert [line for line in plan.summary.lines if line.startswith("warning: ")] == []
    report = {(row.coord, row.kind): row for row in sc.dma_report(plan)}
    assert report[((0,), "get")].hard == ("ToNorth", "UIn")
    assert report[((0,), "put")].hard == ("ToSouth", "UOut")
    assert report[((1,), "get")].hard == ("ToSouth", "UIn")
    assert report[((1,), "put")].hard == ("ToNorth", "UOut")
    assert all(len(row.hard) == len(row.all_) == row.budget for row in report.values())
    assert sc.self_check(w2_plan(T=5, target=target)) is None


@pytest.mark.fr("FR-M9", "FR-M4")
@pytest.mark.parametrize("T", [4, 5])
def test_M4_balanced(T):
    """Every channel index of W2's real plan, both boundary PEs, `put_count == get_count`.

    The table carries a row per index of every channel rather than only the ones something
    reached, because "absent" and "balanced" are the two readings a reader must not have to
    tell apart (§7). At `PI = 2` the halo bundles are `size=(1,)` and both PEs reach their one
    index, so the zero rows this fixture could show are the ones `test_M9_count_table_has_
    every_index` shows on W1's fan-out instead.
    """
    plan = w2_plan(T=T)
    rows = {(row["channel"], tuple(row["index"])): (row["puts"], row["gets"])
            for row in sc.balance_table(plan)}
    assert rows == {("ToNorth", (0,)): (T, T), ("ToSouth", (0,)): (T, T),
                    ("UIn", (0,)): (1, 1), ("UIn", (1,)): (1, 1),
                    ("UOut", (0,)): (T, T), ("UOut", (1,)): (T, T)}
    assert sc.balance(plan) is None


@pytest.mark.fr("FR-M9")
def test_M9_count_table_has_every_index():
    """The per-key table carries **every** index of every channel, including the zero rows.

    `test_M4_balanced` (§7) requires the indices a guarded boundary PE never reaches to appear
    as `0 == 0` rather than to be absent; W1's `A2L1` is the case available now — its put space
    is `(2,1)` and its fan-out space `(2,2)`, so `[0,1]` and `[1,1]` are put-free by
    construction and must still be rows.
    """
    table = sc.balance_table(w1_plan.plan("npu1"))
    rows = {(row["channel"], tuple(row["index"])): (row["puts"], row["gets"]) for row in table}
    assert len(rows) == len(table) == 12                      # 4 + 4 + 4, no index missing
    assert rows[("A2L1", (0, 0))] == (4, 4) and rows[("A2L1", (0, 1))] == (0, 4)
    assert all(rows[("C2L3", index)] == (1, 1)
               for index in ((0, 0), (0, 1), (1, 0), (1, 1)))


@pytest.mark.fr("FR-M9")
def test_M9_rejects_dropped_get():
    """A drained index with no get: `BALANCE`, `1 != 0`, with the count table in `details`."""
    excinfo = raises(dropped_get.plan())
    assert_diagnostic(excinfo, code="BALANCE", clause="plan()", mentions=("C2L3", 1, 0),
                      details_keys=("rule", "counts", "put_sites", "get_sites", "channel"))
    details = excinfo.value.diagnostic.details
    assert (details["put_count"], details["get_count"]) == (1, 0)
    assert details["put_sites"][0]["site"] == "C2L3.put.3@herd"
    assert len(details["counts"]) == 12


@pytest.mark.fr("FR-M9")
def test_M9_rejects_extra_put_in_loop():
    """A second put inside the K loop: the **per-iteration** rule, `4 != 8` (§3.7.1)."""
    excinfo = raises(extra_put_in_loop.plan())
    assert_diagnostic(excinfo, code="BALANCE", clause="plan()", mentions=("A2L1", 4, 8),
                      details_keys=("rule",))
    details = excinfo.value.diagnostic.details
    assert details["rule"] == "per-iteration"
    assert (details["put_count"], details["get_count"]) == (8, 4)
    assert [site["site"] for site in details["put_sites"]] == ["A2L1.put.0@segment",
                                                              "A2L1.put.1@segment"]


@pytest.mark.fr("FR-M9", "FR-M2")
def test_M9_rejects_broadcast_underconsumed():
    """Half a fan-out set unconsumed: the **fan-out** rule (D-2), naming the missing index."""
    excinfo = raises(broadcast_underconsumed.plan())
    assert_diagnostic(excinfo, code="BALANCE", clause="plan()", mentions=("A2L1", [0, 1]),
                      details_keys=("rule", "fanout_index"))
    details = excinfo.value.diagnostic.details
    assert details["rule"] == "fan-out"
    assert (tuple(details["index"]), tuple(details["fanout_index"])) == ((0, 0), (0, 1))
    assert (details["put_count"], details["get_count"]) == (4, 0)


@pytest.mark.fr("FR-M9", "FR-M4")
def test_M9_rejects_guarded_put_only():
    """A halo get with its guard dropped: the **per-branch** rule, naming the coordinate."""
    excinfo = raises(halo_guard_dropped.plan())
    assert_diagnostic(excinfo, code="BALANCE", clause="plan()", mentions=("ToNorth", 0, 1),
                      details_keys=("rule", "get_sites"))
    details = excinfo.value.diagnostic.details
    assert details["rule"] == "per-branch"
    assert (details["put_count"], details["get_count"]) == (0, 1)
    assert [tuple(site["coord"]) for site in details["get_sites"]] == [(1,)]  # PE 1

    # ...and on W2's **real** plan, which now supersedes the synthetic stand-in: the guard is
    # what keeps `tx` inside `[0, PI-1)`, so without it PE 1 gets at `ToNorth[1]`, an index of a
    # `size=(1,)` bundle that no put can reach.
    plan = w2_plan()
    loop = plan.herd_body[4]
    body = list(loop.body)
    for position in (2, 8):                                   # the north ghost get, both phases
        body[position] = replace(body[position], guard=None)
    excinfo = raises(w2_with_body(plan, plan.herd_body[:4]
                                  + (replace(loop, body=tuple(body)),)))
    assert_diagnostic(excinfo, code="BALANCE", clause="plan()", mentions=("ToNorth", 0, 4),
                      details_keys=("rule", "get_sites"))
    details = excinfo.value.diagnostic.details
    assert details["rule"] == "per-branch"
    assert (tuple(details["index"]), details["put_count"], details["get_count"]) == ((1,), 0, 4)
    assert {tuple(site["coord"]) for site in details["get_sites"]} == {(1,)}


# --------------------------------------------------------------------------------------------
# FR-M10 — P2b acyclicity (§3.7.2)
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-M10", "FR-M4")
def test_M10_cycle_rejected():
    """The halo emitted `[GET, GET, PUT, PUT]` closes a four-site cycle inside one timestep."""
    excinfo = raises(halo_reversed.plan())
    assert_diagnostic(excinfo, code="CHANNEL-CYCLE", clause="plan()",
                      mentions=("ToNorth", "ToSouth"), details_keys=("cycle", "channels"))
    cycle = excinfo.value.diagnostic.details["cycle"]
    assert len(cycle) == 4
    assert sum(1 for edge in cycle if edge["kind"] == "channel") == 2
    assert {edge["from"]["site"] for edge in cycle} == {
        "ToNorth.get.2@herd", "ToSouth.put.5@herd", "ToSouth.get.3@herd", "ToNorth.put.4@herd"}
    for edge in cycle:                                # every end names where it is
        assert set(edge["from"]) == {"site", "channel", "kind", "index", "coord"}

    # ...and on W2's **real** plan: the same reversal of the same four sites, built with
    # `dataclasses.replace` on the protocol M4 derives rather than on a hand-written stand-in.
    excinfo = raises(w2_reversed(w2_plan()))
    assert_diagnostic(excinfo, code="CHANNEL-CYCLE", clause="plan()",
                      mentions=("ToNorth", "ToSouth"), details_keys=("cycle", "channels"))
    cycle = excinfo.value.diagnostic.details["cycle"]
    assert list(excinfo.value.diagnostic.details["channels"]) == ["ToNorth", "ToSouth"]
    assert len(cycle) >= 4 and len(cycle) % 2 == 0
    assert sum(1 for edge in cycle if edge["kind"] == "channel") == len(cycle) // 2
    assert {edge["from"]["channel"] for edge in cycle} == {"ToNorth", "ToSouth"}
    # the order M4 actually builds is the one that is acyclic — that is `test_M10_w2_acyclic`
    assert sc.acyclicity(w2_plan()) is None


@pytest.mark.fr("FR-M10", "FR-M4")
@pytest.mark.parametrize("T", [4, 5])
def test_M10_w2_acyclic(T):
    """No SCC of W2's real plan contains a channel edge, for either parity of `T` (§7).

    With `[PUT_n, PUT_s, GET_n, GET_s]` the only outgoing edges of a get are program-order edges
    to later compute, and the loop back edge is `StructuralBack` and excluded. The put→get edges
    themselves are FIFO-paired (`m4_selfcheck._pairs`): pairing a *later* put with an *earlier*
    get is a dependency no execution has, and it is what the unroll-by-two would otherwise
    manufacture out of a protocol §3.7.2 itself calls acyclic.
    """
    plan = w2_plan(T=T)
    nodes, edges = sc.graph(plan)
    assert sc.acyclicity(plan) is None
    channel_edges = [e for e in edges if e.kind == "channel"]
    assert channel_edges, "a halo with no channel edge would make this vacuous"
    components = [c for c in sc._tarjan(nodes, edges) if len(c) > 1]
    assert components == [], components
    # every halo put is paired with exactly one get, and it is the one at the same phase
    pairs = [(e.src.site, e.dst.site) for e in channel_edges
             if e.src.site.startswith(("ToNorth", "ToSouth"))]
    assert len(pairs) == 2 * (2 + T % 2)
    assert all(len({src for src, _ in pairs if _ == dst}) == 1 for _, dst in pairs)


@pytest.mark.fr("FR-M10")
def test_M10_no_false_cross_scope_edge():
    """Rule 4: the segment producer and the herd are **not** joined by a program-order edge.

    The launch and the herd are concurrent async regions (open-questions §1, evidence 3), so the
    whole herd is contracted to one node `H` in the segment's order and no edge runs from a
    segment site to a herd-body site. Drawing one would serialise producer and consumer and make
    W1 look cyclic.
    """
    plan = w1_plan.plan("npu1")
    nodes, edges = sc.graph(plan)
    assert sc.HERD in nodes
    scope = {site.id: site.scope for channel in plan.channels for site in channel.sites}
    crossing = [edge for edge in edges if edge.kind == "program"
                and scope.get(edge.src.site) == "segment" and scope.get(edge.dst.site) == "herd"]
    assert crossing == [], crossing
    assert any(edge.dst == sc.HERD for edge in edges)         # ... but the marker is joined
    assert any(edge.src == sc.HERD for edge in edges)
    assert any(edge.kind == "channel" and scope.get(edge.src.site) == "segment"
               and scope.get(edge.dst.site) == "herd" for edge in edges)


@pytest.mark.fr("FR-M10")
def test_M10_branches_not_joined():
    """Rule 3: no program-order edge between the two arms of one `BranchNode`.

    Enumerating coordinates is what makes it hold: exactly one arm is live at any coordinate, so
    the two gets never appear in the same body order. The assertion is the contrapositive — the
    two sites are in the graph, at different coordinates, with no edge between them.
    """
    nodes, edges = sc.graph(branch_gets.plan())
    arms = {branch_gets.WEST_IN_GET.id, branch_gets.WEST_GET.id}
    assert {node.site for node in nodes} >= arms
    assert not [edge for edge in edges if edge.src.site in arms and edge.dst.site in arms]
    assert {node.coord for node in nodes if node.site == branch_gets.WEST_IN_GET.id} == {(0,)}
    assert {node.coord for node in nodes if node.site == branch_gets.WEST_GET.id} == {(1,)}

    # ...and the same on W3's real plan, whose four branch arms are the shape the rule is for
    nodes, edges = sc.graph(m4.plan(w3_legal.legal()))
    heads = {node.site for node in nodes if node.site.startswith("WestIn.get")}
    bodies = {node.site for node in nodes if node.site.startswith("West.get")}
    tails = {node.site for node in nodes if node.site.startswith("EastOut.put")}
    middles = {node.site for node in nodes if node.site.startswith("West.put")}
    assert heads and bodies and tails and middles
    for one, other in ((heads, bodies), (tails, middles)):
        assert not [edge for edge in edges
                    if edge.src.site in one and edge.dst.site in other
                    and edge.src.coord == edge.dst.coord]
    assert {node.coord for node in nodes if node.site in heads} == {(0,)}
    assert {node.coord for node in nodes if node.site in bodies} == {(1,), (2,), (3,)}
    assert {node.coord for node in nodes if node.site in tails} == {(3,)}


# --------------------------------------------------------------------------------------------
# The structural checks (§3.7.3, `06-interfaces.md` §5.6 invariants 3-8)
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-M8")
def test_M11_bundle_index_iv():
    """A bundle index that is the K loop's IV: `BUNDLE-INDEX-IS-IV`, naming site and loop."""
    excinfo = raises(iv_bundle_index.plan())
    assert_diagnostic(excinfo, code="BUNDLE-INDEX-IS-IV", clause="plan()",
                      mentions=("k0", "A2L1.get.2@herd"),
                      details_keys=("site", "loop", "expr", "channel"))
    assert excinfo.value.diagnostic.details["loop"] == "k0"
    # and the structural pass finds it on its own, without going through the balance walk
    with pytest.raises(MappingError) as direct:
        sc.bundle_indices(iv_bundle_index.plan())
    assert direct.value.diagnostic.code == "BUNDLE-INDEX-IS-IV"


@pytest.mark.fr("FR-M7")
def test_M11_pingpong_shape():
    """`a` hoisted above the K loop but still a candidate fails invariant 4, as an internal error.

    Nothing a user wrote can cause it, so it is `PROTOCOL-UNSUPPORTED` with
    `internal_consistency` and a `fix` that asks for a bug report — the ruling on **B-P23**.
    """
    excinfo = raises(pingpong_hoisted.plan())
    assert_diagnostic(excinfo, code="PROTOCOL-UNSUPPORTED", clause="plan()",
                      mentions=("a", "direct child", "gemm"),
                      details_keys=("internal_consistency", "invariant", "workload", "buffer"))
    diagnostic = excinfo.value.diagnostic
    assert (diagnostic.details["invariant"], diagnostic.details["internal_consistency"]) == (
        4, True)
    assert diagnostic.reason.startswith("internal:")
    assert "report" in diagnostic.fix


@pytest.mark.fr("FR-M7")
def test_M4_tensor_order_rejected():
    """A written tensor before a read-only one fails invariant 6, naming `_check_interface`."""
    excinfo = raises(tensor_order.plan())
    assert_diagnostic(excinfo, code="PROTOCOL-UNSUPPORTED", clause="plan()",
                      mentions=("_check_interface", "C", "A"),
                      details_keys=("internal_consistency", "invariant", "pair"))
    assert excinfo.value.diagnostic.details["invariant"] == 6


@pytest.mark.fr("FR-M7")
def test_M4_l1_agrees_with_m3():
    """Invariant 5: W1 charges `16384` on npu1, and an inflated buffer is an internal error.

    16 384 and not §6.1's 12 288 because npu1 repeats the 2×2 grid onto a (1, 2) herd and
    R-L1-3 charges every buffer twice; the self-check reads the same `repeats` M3 did.

    **Substitution, recorded:** `BufferPlan.bytes` cannot be inflated on its own — M0's `I39`
    ties it to `prod(shape) * dtype.sizeof` — so the nearest constructible corruption widens
    `acc` to `(32, 64)`, which is what an off-by-one `TILE_SHAPE` would produce.
    """
    plan = w1_plan.plan("npu1")
    assert sc.l1_total(plan.buffers, repeated=True) == 16384 == plan.mapping.l1_bytes
    assert sc.l1_total(w1_plan.plan("npu2").buffers) == 12288
    assert sc.l1_budget(plan) is None
    fat = replace(w1_plan.ACC, shape=(32, 64), bytes=32 * 64 * 4)
    swapped = replace(plan, buffers=(fat,) + plan.buffers[1:],
                      herd_body=(fat,) + plan.herd_body[1:])
    excinfo = raises(swapped)
    assert_diagnostic(excinfo, code="PROTOCOL-UNSUPPORTED", clause="plan()",
                      mentions=(24576, 16384), details_keys=("plan_l1_bytes",
                                                             "mapping_l1_bytes", "invariant"))
    assert excinfo.value.diagnostic.details["invariant"] == 5


@pytest.mark.fr("FR-M7", "FR-M6")
@pytest.mark.parametrize("target", TARGETS)
def test_M4_l1_agrees_with_m3_flip(target):
    """The flip charges `24576 == 16384 + 8192`: M3's staging subset plus M4's `recv` (§7).

    `recv` carries no `operand`, which is exactly what `06-interfaces.md` §5.6 invariant 5 uses
    to split the two figures — M3 charges what the schedule stages, M4 adds what the protocol
    needs (`02-hld.md` §7).
    """
    plan = m4.plan(w1flip_legal.legal(target))
    staged = sc.l1_total(tuple(b for b in plan.buffers if b.operand is not None))
    assert (sc.l1_total(plan.buffers), staged, plan.mapping.l1_bytes) == (24576, 16384, 16384)
    assert sc.l1_total(plan.buffers) == staged + 8192
    assert sc.l1_budget(plan) is None


@pytest.mark.fr("FR-M9", "FR-M10", "FR-M6")
@pytest.mark.parametrize("target", TARGETS)
def test_M9_selfcheck_accepts_flip(target):
    """The flip's real plan passes every check on both targets, with **no** warning (§7).

    The balance rows are §6.2's own arithmetic: one put and one get per cascade link per `i0`
    trip, `A2L1` refilled on both trips, `B2L1` filled once, and the tail PE's two drained
    row-blocks matching the drain loop's two trips.
    """
    plan = m4.plan(w1flip_legal.legal(target))
    assert sc.self_check(plan) is None
    assert sc.warnings(plan) == ()
    assert [line for line in plan.summary.lines if line.startswith("warning: ")] == []
    rows = {(row["channel"], tuple(row["index"])): (row["puts"], row["gets"])
            for row in sc.balance_table(plan)}
    assert rows == {("A2L1", (0,)): (2, 2), ("A2L1", (1,)): (2, 2),
                    ("A2L1", (2,)): (2, 2), ("A2L1", (3,)): (2, 2),
                    ("B2L1", (0,)): (1, 1), ("B2L1", (1,)): (1, 1),
                    ("B2L1", (2,)): (1, 1), ("B2L1", (3,)): (1, 1),
                    ("C2L3", (0,)): (2, 2),
                    ("CascadeK", (0,)): (2, 2), ("CascadeK", (1,)): (2, 2),
                    ("CascadeK", (2,)): (2, 2)}
    assert sc.acyclicity(plan) is None
    # the 2-D descending variant is checked too: it is a test fixture, not a demo artifact
    assert sc.self_check(m4.plan(w1flip_legal.legal(target, grid2d=True))) is None


@pytest.mark.fr("FR-M6")
def test_P3_cascade_not_counted():
    """A cascade op binds **no** DMA channel, in either direction (§6.2's note, ruling R-F-4).

    `aie.put_cascade` / `aie.get_cascade` (`AIRToAIEPass.cpp:6955-7023`) is a dedicated
    core-to-core wire, not a tile DMA. Counting the chain would make PE 1 name three inbound
    channels against 2 S2MM and reject a design `aircc` compiles (P-R3) — so the flip's per-PE
    counts are inbound `{A2L1, B2L1}` = 2 and outbound `{C2L3}` on the tail PE only = 1.
    """
    plan = m4.plan(w1flip_legal.legal())
    assert next(c for c in plan.channels if c.name == "CascadeK").channel_type == sc.CASCADE
    report = {(row.coord, row.kind): row for row in sc.dma_report(plan)}
    for tx in range(4):
        assert report[((tx,), "get")].hard == report[((tx,), "get")].all_ == ("A2L1", "B2L1")
        expected = ("C2L3",) if tx == 3 else ()
        assert report[((tx,), "put")].hard == report[((tx,), "put")].all_ == expected
    assert all(len(row.hard) <= row.budget for row in report.values())
    assert sc.dma_channels(plan) is None and sc.warnings(plan) == ()
    # ...and the chain *is* core-to-core, so without the carve-out it would have been counted
    chain = next(c for c in plan.channels if c.name == "CascadeK")
    assert sc.is_core_to_core(chain, plan) and sc.circuit(chain, plan)


@pytest.mark.fr("FR-M7")
def test_M4_names_and_marker():
    """Invariants 7 and 8 are asserted even though M0 enforces them (`self_check` is public).

    Neither has a *corrupt* fixture, and that is the finding: M0 refuses to construct a
    `MappingPlan` with an empty `launch_name` (`I56`) or with the `HerdPlan` marker missing,
    duplicated or nested (`I75`-`I77`), so the only reachable spellings are the checks below.
    The assertion is made anyway because `self_check` is public and cheap to run.
    """
    plan = w1_plan.plan("npu1")
    assert sc.names_and_marker(plan) is None
    with pytest.raises(ValueError, match="launch_name"):
        replace(plan, launch_name="")
    with pytest.raises(ValueError, match="HerdPlan"):
        replace(plan, segment_body=tuple(n for n in plan.segment_body if n is not plan.herd))


# --------------------------------------------------------------------------------------------
# P3 — the per-core circuit-switched DMA budget (§3.8)
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-M4")
def test_P3_dma_inbound():
    """Three circuit-switched inbound channels on one PE is an error naming the physical budget.

    §7's row is W2 at `PI = 4`, and it is now the **real** plan: `m4.plan` runs `self_check` on
    its own result, so `w2_legal.legal(target, T=4, PI=4)` never leaves M4. The interior PE `(1,)`
    names `UIn` plus both ghost gets, all three circuit-switched (per-column shim pressure is 1,
    so `UIn` is not upgraded to a packet flow) — the shape `aircc` is measured to fail on with
    `'aie.connect' op … TileID(1, 2) targets same dst` (REVIEW-round1 P-R2), twenty seconds in
    and naming a physical tile the user has never heard of. The checker rejects it first.
    """
    with pytest.raises(MappingError) as excinfo:
        m4.plan(w2_legal.legal("npu1", T=4, PI=4))
    assert_diagnostic(excinfo, code="DMA-CHANNELS", clause="grid(4)",
                      mentions=("circuit-switched", "2 S2MM", "UIn", "ToNorth", "ToSouth", 2),
                      details_keys=("coord", "direction", "channels", "budget", "count"))
    details = excinfo.value.diagnostic.details
    assert (tuple(details["coord"]), details["direction"]) == ((1,), "inbound")
    assert (sorted(details["channels"]), details["budget"], details["count"]) == (
        ["ToNorth", "ToSouth", "UIn"], 2, 3)
    assert sorted(details["circuit_switched"]) == sorted(details["all_channels"])
    assert m4.plan(w2_legal.legal("npu1", T=4, PI=2)) is not None      # PI = 2 is accepted

    # the synthetic stand-in, kept: it is the same rule with no protocol builder behind it
    excinfo = raises(dma_three_inbound.plan(PI=4))
    assert_diagnostic(excinfo, code="DMA-CHANNELS", clause="grid(4)",
                      mentions=("circuit-switched", "2 S2MM", "UIn", "ToNorth", "ToSouth", 2),
                      details_keys=("coord", "direction", "channels", "budget", "count"))
    details = excinfo.value.diagnostic.details
    assert (tuple(details["coord"]), details["direction"]) == ((1,), "inbound")
    assert (sorted(details["channels"]), details["budget"]) == (["ToNorth", "ToSouth", "UIn"], 2)
    # the 3-PE variant is the same rule one PE earlier, and both targets agree
    assert raises(dma_three_inbound.plan()).value.diagnostic.code == "DMA-CHANNELS"


@pytest.mark.fr("FR-M4")
def test_P3_dma_exclusive_branches():
    """Two gets on mutually exclusive arms count **once** (`MAX_OVER_EXCLUSIVE_BRANCHES`)."""
    report = {(row.coord, row.kind): row for row in sc.dma_report(branch_gets.plan())}
    assert report[((0,), "get")].all_ == ("WestIn",)          # the head arm only
    assert report[((1,), "get")].all_ == ("West",)            # the body arm only
    assert all(len(row.hard) <= row.budget for row in report.values())
    assert sc.self_check(branch_gets.plan()) is None

    # W3's real plan: PE 0 names WestIn and never West, so its west inbound count is 1, not 2
    plan = m4.plan(w3_legal.legal())
    report = {(row.coord, row.kind): row for row in sc.dma_report(plan)}
    west = {coord: [name for name in report[((coord,), "get")].all_
                    if name in ("WestIn", "West")] for coord in range(4)}
    assert west == {0: ["WestIn"], 1: ["West"], 2: ["West"], 3: ["West"]}
    assert report[((0,), "get")].all_ == ("QIn", "RIn", "WestIn")
    assert report[((0,), "put")].all_ == ("SOut", "West")


@pytest.mark.fr("FR-M4")
def test_P3_dma_packet_warns():
    """Over budget by named channels, within it by circuit-switched ones: a warning, no raise."""
    plan = dma_packet_warn.plan()
    assert sc.self_check(plan) is None
    messages = sc.warnings(plan)
    assert len(messages) == 2                                 # one per PE, inbound only
    for message in messages:
        assert "packet flows" in message and "not contractual" in message
        assert all(name in message for name in ("QIn", "RIn", "UIn"))
    assert sc.warnings(w1_plan.plan("npu1")) == ()

    # W3's real plan is the measured case the split exists for: three inbound channels per core,
    # and `aircc --device npu1 --output-format=none` exits 0 with zero `error:` lines (P-R4).
    plan = m4.plan(w3_legal.legal())
    assert sc.self_check(plan) is None
    messages = sc.warnings(plan)
    assert len(messages) == 4                                 # one per PE, inbound only
    for coord, message in enumerate(messages):
        assert message.startswith(f"PE [{coord}] names 3 inbound channels")
        assert "QIn" in message and "RIn" in message and "packet flows" in message
        assert "2 S2MM" in message


def test_may_packet_evidence():
    """`may_packet` reproduces every row of the evidence table in its own docstring.

    The shim pressure of each probe is recomputed from a plan with the same channel geometry, so
    a change to the predicate that broke one of the five measured lowerings fails here. The
    numbers are the ones `air-dma-to-channel` itself printed: *per-column pressure 3 exceeds shim
    DMA limit of 2* for `q7a` and for `w3c`, and no warning at all for `w2pi2`, `w2pi4`, `w3b`.
    """
    w1 = w1_plan.plan("npu1")                                 # q7a: A2L1 + B2L1, pressure 3
    assert sc.shim_pressure(w1, "in") == 3 and sc.shim_pressure(w1, "out") == 1
    assert [sc.may_packet(c, w1) for c in w1.channels] == [True, True, False]
    assert [sc.circuit(c, w1) for c in w1.channels] == [False, False, True]

    w2 = halo.plan(2, staging=("UIn",))                       # w2pi2 / w2pi4: UIn alone
    assert sc.shim_pressure(w2, "in") == 1
    assert not any(sc.may_packet(c, w2) for c in w2.channels)
    assert all(sc.circuit(c, w2) for c in w2.channels)
    assert sc.is_core_to_core(next(c for c in w2.channels if c.name == "ToNorth"), w2)

    w3b = branch_gets.plan()                                  # w3b: WestIn alone, circuit
    assert sc.shim_pressure(w3b, "in") == 1
    assert not any(sc.may_packet(c, w3b) for c in w3b.channels)

    w3c = dma_packet_warn.plan()                              # w3c: three L3 inputs, packet
    assert sc.shim_pressure(w3c, "in") == 3
    assert {c.name for c in w3c.channels if sc.may_packet(c, w3c)} == {"UIn", "QIn", "RIn"}
    assert {c.name for c in w3c.channels if sc.circuit(c, w3c)} == {"ToNorth", "ToSouth"}

    # the discriminating pair: WestIn is circuit alone and packet beside QIn/RIn, which no
    # predicate over one channel's own shape can express
    assert sc.shim_pressure(halo.plan(2, staging=("UIn",)), "in") != sc.shim_pressure(w3c, "in")


# --------------------------------------------------------------------------------------------
# I-1 — the module is testable with no toolchain
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-S20")
def test_selfcheck_imports_no_air():
    """`spatial.m4_selfcheck` imports neither `air` nor M5 nor M4 (invariant I-1)."""
    source = Path(sc.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            continue
        for module in imported:
            assert module != "air" and not module.startswith("air.")
            assert "m5_emit" not in module and "m4_mapping" not in module
    assert Dtype.f32.sizeof == 4                              # M0 is the only spatial import
