"""M4 — plan self-check (P1' balance, P2b acyclicity, structural, P3 DMA). Owner: Person B.

Spec: `design/03-lld-M4-mapping.md` §3.7 (the three checks) and §3.8 (the DMA-channel budget),
which implement `mlir-air/docs/AIRCorrectnessChecker.md` §4 (P1), §5.2-§5.3 (P2b) and §6.1-§6.2
(P3) over a `MappingPlan` instead of over MLIR. Entry point per `design/06-interfaces.md` §7.2;
the eight plan invariants it enforces are §5.6's.

**Why this module is the entry's differentiator.** `AIRCorrectnessChecker.md` specifies P1-P4;
`mlir/lib/Analysis/` does not exist; `air-opt` prints its one channel diagnostic and exits 0;
and `AIRToAIEPass.cpp:4484-4491` silently *repairs* L2 imbalance rather than diagnosing it.
Nothing upstream catches what M4 gets wrong, so M4 checks itself, by enumeration rather than by
symbolic reasoning (decision D-12): grid extents are compile-time constants of at most 8 per
axis, so every guard and every bundle index is evaluated to an integer.

**Nothing here imports `air`** (invariant I-1), and nothing imports `spatial.m4_mapping` — which
imports this module. Every check is a module-level pure function with its own unit test.

**What `self_check` raises.** `MappingError` and nothing else (NFR-7). `BALANCE`,
`CHANNEL-CYCLE`, `BUNDLE-INDEX-IS-IV` and `DMA-CHANNELS` are the user-facing codes of
`06-interfaces.md` §6.3; a failure that can only be M4's own construction bug — a ping-pong
shape, an L1 total disagreeing with M3, a tensor order, a misplaced `HerdPlan` — is raised as
`PROTOCOL-UNSUPPORTED` with `details["internal_consistency"] = True`, `details["invariant"]`
naming the §5.6 number, a `reason` beginning `internal:` and a `fix` that asks for a bug report
rather than blaming a clause the user wrote (HLD §4.3). That spelling is the ruling on **B-P23**
and the mirror of M5's B-P13; the catalogue stays at 43 codes.

**Two readings the LLD's pseudocode leaves open** (recorded in `design/PROGRESS-B.md`, P3):

1. §3.7.1 line 24 multiplies `trips` through every `LoopPlan`. That is right for an
   `air.sequential` loop, whose body is emitted once and run `n` times, but wrong for an
   `"unrolled"` loop, which is a trace-time Python loop: its variable *is* a bundle index, and
   §6.1's own balance statement ("`A2L1` put key `[0,0]` count 4") only comes out if the
   unrolled loop is **enumerated** and its variable bound. So: sequential loops multiply
   `trips`, unrolled loops enumerate.
2. §3.7.2 rule 5 excludes the loop back edge. It is applied to unrolled loops too — trip `n`'s
   last node gets no edge to trip `n+1`'s first — which is conservative (fewer edges can only
   hide a cycle, never invent one) and keeps one rule for both loop kinds.
"""

from __future__ import annotations

from itertools import product
from math import prod
from typing import Any, Iterator, NamedTuple

from spatial.model import (BranchNode, BufferPlan, ChannelPlan, ChannelSite, Diagnostic, Expr,
                           Guard, HerdPlan, Load, LoopPlan, MappingError, MappingPlan, StoreNode)

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------

DMA_IN_MAX = 2
"""S2MM channels on an AIE2 core tile (LLD §3.8 line 1) — the inbound budget."""

DMA_OUT_MAX = 2
"""MM2S channels on an AIE2 core tile (LLD §3.8 line 2) — the outbound budget."""

SHIM_DMA_CHANNELS_PER_COL = 2
"""`air-dma-to-channel`'s `shim-dma-channels-per-col`, default 2 (`Passes.td:1808-1812`)."""

MSELS_PER_ARBITER = 4
"""Master selects one switchbox arbiter can hand out — mlir-aie's `int numMselsPerArbiter = 4`
(`lib/Dialect/AIE/Transforms/AIECreatePathFindFlows.cpp`, pinned wheel `10767b5`), and the same
number as `aie.amsel`'s `msel` bound, `ConfinedAttr<AIEI8Attr, [IntMinValue<0>, IntMaxValue<3>]>`
(`include/aie/Dialect/AIE/IR/AIEOps.td`, re-read 2026-09-15). Exhausting it is
*"'aie.tile' op tile op arbiter 0 has used up all its msels"* (**B-P36**)."""

L1_BUDGET = 65536
"""One core tile's **data memory** — `air.api`'s `L1_BYTES` (`_trace.py:100`) and mlir-aie's
`AIE2TargetModel::getLocalMemorySize()` (`0x00010000`), reported by `MappingSummary.l1_budget`.
What a plan must fit into is `L1_USABLE`."""

L1_STACK_RESERVED = 2048
"""The per-core stack `air-to-aie` reserves below the first buffer — the `stack-size` option
default of mlir-air's `-air-to-aie` (`mlir/include/air/Conversion/Passes.td:231-234`)."""

L1_USABLE = L1_BUDGET - L1_STACK_RESERVED
"""`06-interfaces.md` §5.6 invariant 5's budget, 63 488 B — architect ruling **R-L1-2**."""

CASCADE = "npu_cascade"
"""A cascade channel lowers to `aie.put_cascade` / `aie.get_cascade`
(`AIRToAIEPass.cpp:6955-7023`) — a dedicated core-to-core wire, **not** a tile DMA — so its
sites are invisible to the P3 budget in **both** directions (`03-lld-M4-mapping.md` §6.2's DMA
note, ruling R-F-4). Counting them would reject the flip: PE 1 would name three inbound
channels (`A2L1`, `B2L1`, `CascadeK`) against 2 S2MM, and `aircc` compiles it (P-R3)."""

_CLAUSE = "plan()"
"""The surface call a mapping diagnostic with no user clause of its own points at."""

_BUG_FIX = ("this is a defect in the compiler, not in your program: please report it with the "
            "plan (spatial.model.to_json(plan)) and this message")

_RELATION = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b,
             "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
             ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}
"""`Guard.relation` → the predicate it names (`06-interfaces.md` §5.2)."""


# --------------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------------


def _error(code: str, reason: str, fix: str, *, clause: str = _CLAUSE,
           **details: Any) -> MappingError:
    """One `MappingError` with all four message parts populated (`06-interfaces.md` §6.1)."""
    return MappingError(Diagnostic(code=code, stage="mapping", clause=clause, reason=reason,
                                   fix=fix, location=None, details=details))


def _internal(plan: MappingPlan, invariant: int, reason: str, **details: Any) -> MappingError:
    """A self-check failure that can only be M4's own construction bug (B-P23, HLD §4.3).

    `06-interfaces.md` §6.3 gives the mapping stage five codes and M0 enforces the set (I67),
    so an internal-consistency failure is spelled `PROTOCOL-UNSUPPORTED` with
    `details["internal_consistency"]`, exactly as M5 spells its own (B-P13).
    """
    return _error("PROTOCOL-UNSUPPORTED",
                  f"internal: {plan.mapping.kernel.name!r} violates plan invariant "
                  f"{invariant} (design/06-interfaces.md §5.6) — {reason}",
                  _BUG_FIX, internal_consistency=True, invariant=invariant,
                  workload=plan.mapping.kernel.name, **details)


# --------------------------------------------------------------------------------------------
# Enumeration helpers (D-12: concrete coordinates, never symbolic reasoning)
# --------------------------------------------------------------------------------------------


def coordinates(grid: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Every concrete herd coordinate, in lexicographic order (LLD §3.7.1 line 4)."""
    return tuple(product(*(range(extent) for extent in grid)))


def all_indices(size: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Every concrete bundle index of a channel of this `size` (LLD §3.7.1 line 8)."""
    return coordinates(size)


def fanout(index: tuple[int, ...], size: tuple[int, ...],
           broadcast_shape: tuple[int, ...] | None) -> tuple[tuple[int, ...], ...]:
    """The destination indices one put at `index` reaches — decision **D-2**, LLD §3.7.1 line 28.

    A put at `i` on a channel with `broadcast_shape` reaches every `d` of the fan-out shape with
    `d[k] % size[k] == i[k]`. With no `broadcast_shape` the set is `{i}`, which is the upstream
    spec's literal P1 key.
    """
    if broadcast_shape is None:
        return (index,)
    return tuple(d for d in all_indices(broadcast_shape)
                 if all(d[k] % size[k] == index[k] for k in range(len(size))))


def _constant(expr: Expr, what: str, plan: MappingPlan, invariant: int) -> int:
    """An `Expr` that must be a compile-time constant (LLD §3.7.1: trip counts, §3.7.3 line 11)."""
    if not expr.is_constant:
        raise _internal(plan, invariant, f"{what} is {expr!r}, which is not a constant: the "
                                         f"self-check enumerates and cannot evaluate it")
    return expr.const


def trip_count(loop: LoopPlan, plan: MappingPlan) -> int:
    """`ceil((hi - lo) / step)` — the number of times a loop body runs, never negative."""
    lo = _constant(loop.lo, f"loop {loop.axis!r} lower bound", plan, 1)
    hi = _constant(loop.hi, f"loop {loop.axis!r} upper bound", plan, 1)
    step = _constant(loop.step, f"loop {loop.axis!r} step", plan, 1)
    return max(0, -(-(hi - lo) // step))


def _values(loop: LoopPlan, plan: MappingPlan) -> range:
    """The values an **unrolled** (trace-time Python) loop binds its variable to."""
    return range(_constant(loop.lo, f"loop {loop.axis!r} lower bound", plan, 1),
                 _constant(loop.hi, f"loop {loop.axis!r} upper bound", plan, 1),
                 _constant(loop.step, f"loop {loop.axis!r} step", plan, 1))


def _eval(expr: Expr, env: dict[str, int], plan: MappingPlan, what: str) -> int:
    """`expr` over bound herd coordinates and unrolled loop variables, as an integer."""
    total = expr.const
    for name, coeff in expr.coeffs:
        if name not in env:
            raise _internal(plan, 3, f"{what} names {name!r}, which is bound by nothing in "
                                     f"scope (bound here: {sorted(env)})", name=name)
        total += coeff * env[name]
    return total


def _holds(guard: Guard, env: dict[str, int], plan: MappingPlan, where: str) -> bool:
    """Whether an `ops.branch` condition holds at this concrete coordinate (LLD §3.7.1 line 21)."""
    if guard.coord not in env:
        raise _internal(plan, 8, f"the guard on {where} tests {guard.coord!r}, which is not a "
                                 f"coordinate in scope (bound here: {sorted(env)})",
                        coord=guard.coord)
    return _RELATION[guard.relation](env[guard.coord], _eval(guard.value, env, plan, "a guard"))


def _index(site: ChannelSite, env: dict[str, int], loops: tuple[tuple[str, str], ...],
           plan: MappingPlan) -> tuple[int, ...]:
    """One site's concrete bundle index, or `BUNDLE-INDEX-IS-IV` if it names a temporal IV.

    §3.7.3's structural check is *reached through here* as well as run on its own, because a
    bundle index that is an `air.sequential` induction variable has no concrete value and the
    balance walk would otherwise fail with an internal error on a real user-visible defect.
    """
    out = []
    for expr in site.indices:
        for name, _coeff in expr.coeffs:
            if name in env:
                continue
            temporal = [axis for axis, kind in loops if axis == name and kind == "sequential"]
            if temporal:
                raise bundle_index_error(site, expr, temporal[0])
        out.append(_eval(expr, env, plan, f"the bundle index of site {site.id!r}"))
    return tuple(out)


def bundle_index_error(site: ChannelSite, expr: Expr, loop: str) -> MappingError:
    """`BUNDLE-INDEX-IS-IV` (`ChannelPutOp::verify`, `AIRDialect.cpp:3586-3593`), LLD §3.7.3."""
    return _error("BUNDLE-INDEX-IS-IV",
                  f"the bundle index of {site.kind} site {site.id!r} on channel "
                  f"{site.channel!r} references {loop!r}, the induction variable of a temporal "
                  f"air.sequential loop",
                  f"index the bundle by a herd coordinate or a trace-time constant, and move "
                  f"the {loop!r} loop inside the channel op",
                  site=site.id, channel=site.channel, loop=loop,
                  expr=_render(expr), indices=[_render(e) for e in site.indices])


def _render(expr: Expr) -> str:
    """An `Expr` as the source-like text a diagnostic prints."""
    terms = [f"{coeff}*{name}" if coeff != 1 else name for name, coeff in expr.coeffs]
    if expr.const or not terms:
        terms.append(str(expr.const))
    return " + ".join(terms)


# --------------------------------------------------------------------------------------------
# The occurrence walk — every live (site, coordinate) pair, with its multiplicity
# --------------------------------------------------------------------------------------------


class Occurrence(NamedTuple):
    """One live channel op: which site, at which coordinate, with which index, how often."""

    site: ChannelSite
    index: tuple[int, ...]
    coord: tuple[int, ...]
    trips: int
    guarded: bool
    in_loop: bool


def _walk(nodes: tuple[Any, ...], env: dict[str, int], trips: int,
          loops: tuple[tuple[str, str], ...], guarded: bool, in_loop: bool,
          coord: tuple[int, ...], plan: MappingPlan, out: list[Occurrence]) -> None:
    """`WALK` of LLD §3.7.1 lines 18-26, with reading 1 of the module docstring applied."""
    for node in nodes:
        if isinstance(node, ChannelSite):
            if node.guard is not None and not _holds(node.guard, env, plan, f"site {node.id!r}"):
                continue
            out.append(Occurrence(node, _index(node, env, loops, plan), coord, trips,
                                  guarded or node.guard is not None, in_loop))
        elif isinstance(node, LoopPlan):
            inner = loops + ((node.axis, node.kind),)
            if node.kind == "unrolled":
                for value in _values(node, plan):
                    _walk(node.body, {**env, node.axis: value}, trips, inner, guarded,
                          in_loop, coord, plan, out)
            else:
                count = trip_count(node, plan)
                _walk(node.body, env, trips * count, inner, guarded, in_loop or count > 1,
                      coord, plan, out)
        elif isinstance(node, BranchNode):
            taken = (node.then if _holds(node.predicate, env, plan, "a BranchNode")
                     else node.otherwise)
            _walk(taken, env, trips, loops, True, in_loop, coord, plan, out)
        # BufferPlan, StoreNode and HerdPlan contribute nothing to balance (LLD §3.7.1 line 26)


def occurrences(plan: MappingPlan, *, herd_only: bool = False) -> tuple[Occurrence, ...]:
    """Every live channel op of the plan: the segment body once, the herd body per coordinate.

    The segment body is walked with `coords = [()]` and the herd body once per concrete herd
    coordinate, which is the cross-scope rule of LLD §3.7.1: a segment-scope put of `K/TK` is
    compared against `PI·PJ` herd gets with the right multiplicities.
    """
    out: list[Occurrence] = []
    if not herd_only:
        _walk(plan.segment_body, {}, 1, (), False, False, (), plan, out)
    for coord in coordinates(plan.herd.grid):
        _walk(plan.herd_body, dict(zip(plan.herd.coords, coord)), 1, (), False, False, coord,
              plan, out)
    return tuple(out)


# --------------------------------------------------------------------------------------------
# §3.7.1 — P1' balance (FR-M9); upstream P1 is AIRCorrectnessChecker.md §4
# --------------------------------------------------------------------------------------------


def balance_table(plan: MappingPlan) -> tuple[dict[str, Any], ...]:
    """The per-key count table: **every** index of every channel, including the `0 == 0` rows.

    The zero rows are not padding: `test_M4_balanced` (LLD §7) requires the indices a guarded
    boundary PE never reaches to appear as `0 == 0` rather than to be absent, because "absent"
    and "balanced" are the two readings a reader must not have to distinguish.
    """
    counts, _sites = _counts(plan)
    rows = []
    for channel in plan.channels:
        keys = set(all_indices(channel.broadcast_shape or channel.size))
        keys |= {index for name, index, _kind in counts if name == channel.name}
        for index in sorted(keys):
            rows.append({"channel": channel.name, "index": list(index),
                         "puts": counts.get((channel.name, index, "put"), 0),
                         "gets": counts.get((channel.name, index, "get"), 0)})
    return tuple(rows)


def _counts(plan: MappingPlan) -> tuple[dict[tuple[str, tuple[int, ...], str], int],
                                        dict[tuple[str, tuple[int, ...], str], list[Occurrence]]]:
    """`(counts, sites)` keyed `(channel, concrete index, kind)` (LLD §3.7.1 line 2)."""
    counts: dict[tuple[str, tuple[int, ...], str], int] = {}
    sites: dict[tuple[str, tuple[int, ...], str], list[Occurrence]] = {}
    for occurrence in occurrences(plan):
        key = (occurrence.site.channel, occurrence.index, occurrence.site.kind)
        counts[key] = counts.get(key, 0) + occurrence.trips
        sites.setdefault(key, []).append(occurrence)
    return counts, sites


def _where(occurrences_: list[Occurrence]) -> list[dict[str, Any]]:
    """The site ids and coordinates on one side of an imbalance, for `Diagnostic.details`."""
    return [{"site": o.site.id, "coord": list(o.coord), "scope": o.site.scope, "trips": o.trips}
            for o in sorted(occurrences_, key=lambda o: (o.site.id, o.coord))]


def _rule(channel: ChannelPlan, index: tuple[int, ...], counts: dict, sides: list[Occurrence],
          plan: MappingPlan) -> str:
    """Which of the four rules of LLD §3.7.1 caught this imbalance.

    `fan-out` first, and only when the fan-out set is *asymmetric* — some destination index of
    this put balances and another does not — because that is the failure D-2's rule exists to
    catch; a put that is wrong for every destination is not a broadcast defect. Then
    `per-branch` (some site of the channel is guarded), then `per-iteration` (some contributing
    site sits inside a sequential loop that runs more than once), then `total`.
    """
    if channel.broadcast_shape is not None:
        puts = counts.get((channel.name, index, "put"), 0)
        gets = [counts.get((channel.name, d, "get"), 0)
                for d in fanout(index, channel.size, channel.broadcast_shape)]
        if any(g == puts for g in gets) and any(g != puts for g in gets):
            return "fan-out"
    if any(site.guard is not None for site in channel.sites) or any(o.guarded for o in sides):
        return "per-branch"
    if any(o.in_loop for o in sides):
        return "per-iteration"
    return "total"


def _fail_balance(plan: MappingPlan, channel: ChannelPlan, index: tuple[int, ...],
                  destination: tuple[int, ...], puts: int, gets: int, counts: dict,
                  sites: dict) -> MappingError:
    """`BALANCE` with the full per-key table, the rule that caught it, and both site lists."""
    put_sites = sites.get((channel.name, index, "put"), [])
    get_sites = sites.get((channel.name, destination, "get"), [])
    rule = _rule(channel, index, counts, put_sites + get_sites, plan)
    fanned = destination != index
    return _error(
        "BALANCE",
        f"channel {channel.name!r} index {list(index)} has {puts} put(s) against {gets} "
        f"get(s)" + (f" at fan-out index {list(destination)}" if fanned else "")
        + f": the {rule} rule of the P1' balance check",
        f"every channel index needs one get per put on every execution path — add the missing "
        f"{'get' if puts > gets else 'put'} for {channel.name}"
        f"{list(destination) if fanned else list(index)}, or drop the extra one",
        rule=rule, channel=channel.name, index=list(index), fanout_index=list(destination),
        put_count=puts, get_count=gets, put_sites=_where(put_sites), get_sites=_where(get_sites),
        counts=list(balance_table(plan)))


def balance(plan: MappingPlan) -> None:
    """P1' — put/get balance per channel index, per iteration, per branch, across scopes.

    LLD §3.7.1, implementing `AIRCorrectnessChecker.md` §4 with decision **D-2**: a put on a
    broadcast channel must be matched by one get at **each** index of its fan-out set, which is
    the rule that keeps `air.api`'s own broadcast idiom legal (the upstream spec's literal P1
    keys on `(name, indices)` and would reject it).
    """
    counts, sites = _counts(plan)
    for channel in sorted(plan.channels, key=lambda c: c.name):
        for index in all_indices(channel.size):
            puts = counts.get((channel.name, index, "put"), 0)
            for destination in fanout(index, channel.size, channel.broadcast_shape):
                gets = counts.get((channel.name, destination, "get"), 0)
                if puts != gets:
                    raise _fail_balance(plan, channel, index, destination, puts, gets, counts,
                                        sites)
        # A key outside the bundle's own index space is an imbalance by definition: nothing on
        # the other side can ever match it. It is how a dropped guard surfaces, because the
        # guard is what kept `tx - 1` and `tx + 1` inside `[0, size)`.
        space = {"put": set(all_indices(channel.size)),
                 "get": set(all_indices(channel.broadcast_shape or channel.size))}
        for (name, index, kind), count in sorted(counts.items()):
            if name != channel.name or not count or index in space[kind]:
                continue
            raise _fail_balance(plan, channel, index, index,
                                counts.get((name, index, "put"), 0),
                                counts.get((name, index, "get"), 0), counts, sites)


# --------------------------------------------------------------------------------------------
# §3.7.2 — P2b acyclicity (FR-M10); upstream P2b is AIRCorrectnessChecker.md §5.2-§5.3
# --------------------------------------------------------------------------------------------


class Node(NamedTuple):
    """One node of the P2b graph: a site at one concrete coordinate and bundle index."""

    site: str
    coord: tuple[int, ...]
    index: tuple[int, ...]


class Edge(NamedTuple):
    """One directed edge, tagged with whether it is a channel edge (the SCC test keys on it)."""

    src: Node
    dst: Node
    kind: str


class _Graph:
    """The nodes, edges and per-(channel, index) endpoint lists built by `graph`."""

    def __init__(self) -> None:
        self.nodes: dict[Node, ChannelSite | None] = {}
        self.edges: list[Edge] = []
        self.ends: dict[tuple[str, tuple[int, ...], str], list[Node]] = {}

    def snapshot(self) -> tuple[tuple[Node, ...], tuple[Edge, ...]]:
        return tuple(sorted(self.nodes)), tuple(sorted(self.edges))

    def add(self, occurrence: Occurrence) -> Node:
        node = Node(occurrence.site.id, occurrence.coord, occurrence.index)
        self.nodes[node] = occurrence.site
        self.ends.setdefault(
            (occurrence.site.channel, occurrence.index, occurrence.site.kind), []).append(node)
        return node

    def edge(self, src: Node, dst: Node, kind: str) -> None:
        if src != dst and Edge(src, dst, kind) not in self.edges:
            self.edges.append(Edge(src, dst, kind))


HERD = Node("<herd>", (), ())
"""Rule 4's contracted node `H`: the whole herd, in the segment body's order. It is a channel
edge endpoint for nothing, so it can never make an SCC illegal."""


def graph(plan: MappingPlan) -> tuple[tuple[Node, ...], tuple[Edge, ...]]:
    """The P2b graph: channel edges plus program-order edges, minus loop-carried back edges.

    Program-order edges follow LLD §3.7.2's five rules **exactly**:

    1. consecutive nodes within one body region, at the same coordinate;
    2. a `LoopPlan` or `BranchNode` contracted to its region — the edge runs from the previous
       sibling's exits to the region's entries and from its exits to the next sibling;
    3. **no** edge between the two arms of a `BranchNode` — enumerating coordinates makes this
       hold by construction, since exactly one arm is live at any coordinate;
    4. the whole herd contracted to one node `H` in the segment body's order, and **no** edge
       from a segment site to a herd-body site: the launch and the herd are concurrent async
       regions, and drawing one would make W1 look cyclic;
    5. the loop back edge is `StructuralBack` and is **excluded** (`AIRCorrectnessChecker.md`
       §5.3) — which is what makes the halo acyclic within a timestep.

    Channel edges follow `_pairs` below, which is the **one** place this module is more precise
    than §3.7.2's literal text; the reason is there.
    """
    return _build(plan).snapshot()


def _pairs(puts: list[Node], gets: list[Node]) -> Iterator[tuple[Node, Node]]:
    """Which put feeds which get on one `(channel, concrete index)` — FIFO order where it holds.

    §3.7.2 writes the channel edge as *"for every put node and every get node whose (channel,
    concrete index) match, a directed edge put → get"*. That all-pairs rule over-approximates a
    channel, which is a **FIFO**: the k-th get receives the k-th put and waits on that one, not
    on every put the index will ever see. The over-approximation is harmless while one body
    holds one transfer per index, and it is not harmless once `swap_loop` unrolls a two-phase
    protocol: W2's own halo then reports a four-edge cycle
    `put_n(phase 1, PE 1) → get_n(phase 0, PE 0) → put_s(phase 1, PE 0) → get_s(phase 0, PE 1)`
    built entirely out of edges that pair a *later* put with an *earlier* get — a dependency no
    execution has. §3.7.2's own worked argument ("no path leads back to a put") is made
    **within a timestep**, and the unroll is what puts two timesteps in one body.

    So: when both sides have the same number of nodes and each side is one coordinate's own
    program order, the k-th put is zipped to the k-th get — the exact semantics, and strictly
    fewer edges. Otherwise (several producers into one index, or a body shape that splits one
    side and not the other, as W3's guarded `WestIn` does) the pairing is not defined and the
    all-pairs rule stands. Recorded in `design/PROGRESS-B.md`, phase P5.
    """
    if (len(puts) == len(gets) and len({node.coord for node in puts}) <= 1
            and len({node.coord for node in gets}) <= 1):
        return iter(zip(puts, gets))
    return ((put, get) for put in puts for get in gets)


def _build(plan: MappingPlan) -> _Graph:
    """`graph`'s working form, keeping the site behind each node for the diagnostic."""
    out = _Graph()
    out.nodes[HERD] = None
    for coord in coordinates(plan.herd.grid):
        _region(plan.herd_body, dict(zip(plan.herd.coords, coord)), (), coord, plan, out)
    _region(plan.segment_body, {}, (), (), plan, out, herd=True)
    for (channel, index, kind), nodes in sorted(out.ends.items()):
        if kind != "put":
            continue
        plan_channel = next(c for c in plan.channels if c.name == channel)
        for destination in fanout(index, plan_channel.size, plan_channel.broadcast_shape):
            for put, get in _pairs(nodes, out.ends.get((channel, destination, "get"), [])):
                out.edge(put, get, "channel")
    return out


def _region(nodes: tuple[Any, ...], env: dict[str, int], loops: tuple[tuple[str, str], ...],
            coord: tuple[int, ...], plan: MappingPlan, out: _Graph,
            herd: bool = False) -> tuple[list[Node], list[Node]]:
    """`(entry nodes, exit nodes)` of one body region, appending its program-order edges.

    Rule 2's contracted node is the region's own first and last nodes, which carries the same
    edges by transitivity and keeps every node a real site — so a contracted region can never
    be a channel-edge endpoint, as the rule requires.
    """
    first: list[Node] = []
    previous: list[Node] = []
    for node in nodes:
        entries, exits = _child(node, env, loops, coord, plan, out, herd)
        if not entries:
            continue
        for src in previous:
            for dst in entries:
                out.edge(src, dst, "program")
        first = first or entries
        previous = exits
    return first, previous


def _child(node: Any, env: dict[str, int], loops: tuple[tuple[str, str], ...],
           coord: tuple[int, ...], plan: MappingPlan, out: _Graph,
           herd: bool) -> tuple[list[Node], list[Node]]:
    """One child of a body: a site, a contracted region, the herd marker, or nothing."""
    if isinstance(node, ChannelSite):
        if node.guard is not None and not _holds(node.guard, env, plan, f"site {node.id!r}"):
            return [], []
        occurrence = Occurrence(node, _index(node, env, loops, plan), coord, 1,
                                node.guard is not None, False)
        graph_node = out.add(occurrence)
        return [graph_node], [graph_node]
    if isinstance(node, LoopPlan):
        inner = loops + ((node.axis, node.kind),)
        if node.kind != "unrolled":
            return _region(node.body, env, inner, coord, plan, out, herd)
        entries: list[Node] = []
        exits: list[Node] = []
        for value in _values(node, plan):                 # rule 5: no edge between two trips
            trip_entries, trip_exits = _region(node.body, {**env, node.axis: value}, inner,
                                               coord, plan, out, herd)
            entries += trip_entries
            exits += trip_exits
        return entries, exits
    if isinstance(node, BranchNode):
        taken = node.then if _holds(node.predicate, env, plan, "a BranchNode") else node.otherwise
        return _region(taken, env, loops, coord, plan, out, herd)
    if isinstance(node, HerdPlan) and herd:
        return [HERD], [HERD]                             # rule 4: the herd is one node `H`
    return [], []


def _tarjan(nodes: tuple[Node, ...], edges: tuple[Edge, ...]) -> tuple[tuple[Node, ...], ...]:
    """Tarjan's SCCs, iterative so a deep plan cannot overflow the stack (spec §5.1)."""
    successors: dict[Node, list[Node]] = {node: [] for node in nodes}
    for edge in edges:
        successors[edge.src].append(edge.dst)
    index: dict[Node, int] = {}
    low: dict[Node, int] = {}
    on_stack: set[Node] = set()
    stack: list[Node] = []
    components: list[tuple[Node, ...]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(successors[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(successors[child])))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    popped = stack.pop()
                    on_stack.discard(popped)
                    component.append(popped)
                    if popped == node:
                        break
                components.append(tuple(sorted(component)))
    return tuple(components)


def _cycle(component: set[Node], edges: tuple[Edge, ...]) -> list[Edge]:
    """One concrete cycle through a channel edge of this SCC, as an ordered edge list."""
    inside = [e for e in edges if e.src in component and e.dst in component]
    start = min((e for e in inside if e.kind == "channel"), key=lambda e: (e.src, e.dst))
    successors: dict[Node, list[Edge]] = {}
    for edge in sorted(inside):
        successors.setdefault(edge.src, []).append(edge)
    previous: dict[Node, Edge] = {}
    frontier = [start.dst]
    while frontier and start.src not in previous:
        node = frontier.pop(0)
        for edge in successors.get(node, ()):
            if edge.dst not in previous and edge.dst != start.dst:
                previous[edge.dst] = edge
                frontier.append(edge.dst)
    path: list[Edge] = []
    node = start.src
    while node in previous and node != start.dst:
        path.append(previous[node])
        node = previous[node].src
    return [start] + list(reversed(path))


def _end(node: Node, site: ChannelSite | None) -> dict[str, Any]:
    """One end of a cycle edge: the site id, its channel, its index and its coordinate."""
    return {"site": node.site, "channel": site.channel if site else None,
            "kind": site.kind if site else None,
            "index": list(node.index), "coord": list(node.coord)}


def acyclicity(plan: MappingPlan) -> None:
    """P2b — no SCC of (channel edges ∪ program order − back edges) contains a channel edge.

    LLD §3.7.2, implementing `AIRCorrectnessChecker.md` §5.2-§5.3. The halo's
    `[PUT, PUT, GET, GET]` order is acyclic because every outgoing edge of a get is a
    program-order edge to later compute; reversing it closes the loop
    `GET_n(p) → PUT_s(p) → GET_s(p+1) → PUT_n(p+1) → GET_n(p)` inside one timestep.
    """
    built = _build(plan)
    nodes, edges = built.snapshot()
    sites = built.nodes
    for component in _tarjan(nodes, edges):
        if len(component) < 2:
            continue
        members = set(component)
        if not any(e.kind == "channel" and e.src in members and e.dst in members for e in edges):
            continue
        cycle = _cycle(members, edges)
        channels = sorted({sites[e.src].channel for e in cycle if e.kind == "channel"})
        raise _error(
            "CHANNEL-CYCLE",
            f"the channel graph has a blocking cycle of {len(cycle)} edge(s) through "
            f"{', '.join(channels)}: every put in it waits on a get that waits back on it",
            "reorder the protocol so every put of a cycle precedes the get it feeds — the halo "
            "order is put north, put south, get north ghost, get south ghost "
            "(design/03-lld-M4-mapping.md §3.6.1)",
            cycle=[{"from": _end(e.src, sites[e.src]), "to": _end(e.dst, sites[e.dst]),
                    "kind": e.kind} for e in cycle],
            channels=channels)


# --------------------------------------------------------------------------------------------
# §3.7.3 — the structural checks (`06-interfaces.md` §5.6 invariants 3-8)
# --------------------------------------------------------------------------------------------


def _sites_with_loops(nodes: tuple[Any, ...], loops: tuple[tuple[str, str], ...] = ()
                      ) -> Iterator[tuple[ChannelSite, tuple[tuple[str, str], ...]]]:
    """Every site of a body with the loop axes enclosing it, regardless of any guard."""
    for node in nodes:
        if isinstance(node, ChannelSite):
            yield node, loops
        elif isinstance(node, LoopPlan):
            yield from _sites_with_loops(node.body, loops + ((node.axis, node.kind),))
        elif isinstance(node, BranchNode):
            yield from _sites_with_loops(node.then, loops)
            yield from _sites_with_loops(node.otherwise, loops)


def bundle_indices(plan: MappingPlan) -> None:
    """Invariant 3 — no `ChannelSite.indices` entry references a temporal loop IV (LLD §3.7.3).

    `ChannelPutOp::verify` rejects a bundle index that *is* an `scf.for` induction variable
    (`AIRDialect.cpp:3586-3593`); finding N-6 narrows the upstream rule to that literal form,
    and the plan keeps the stricter one because it costs nothing and makes the check sound by
    construction.
    """
    for body in (plan.segment_body, plan.herd_body):
        for site, loops in _sites_with_loops(body):
            temporal = {axis for axis, kind in loops if kind == "sequential"}
            for expr in site.indices:
                for name, _coeff in expr.coeffs:
                    if name in temporal:
                        raise bundle_index_error(site, expr, name)


def _allocations(nodes: tuple[Any, ...], parent: LoopPlan | None = None,
                 loops: tuple[LoopPlan, ...] = ()
                 ) -> Iterator[tuple[BufferPlan, LoopPlan | None, tuple[LoopPlan, ...],
                                     tuple[Any, ...]]]:
    """Every `air.alloc` with its direct parent loop, its enclosing loops and its body."""
    for node in nodes:
        if isinstance(node, BufferPlan):
            yield node, parent, loops, nodes
        elif isinstance(node, LoopPlan):
            yield from _allocations(node.body, node, loops + (node,))
        elif isinstance(node, BranchNode):
            yield from _allocations(node.then, None, loops)
            yield from _allocations(node.otherwise, None, loops)


def _touches(nodes: tuple[Any, ...], buffer: str) -> Iterator[Any]:
    """Every node that reads or writes one buffer, in program order."""
    for node in nodes:
        if isinstance(node, ChannelSite) and node.buffer == buffer:
            yield node
        elif isinstance(node, StoreNode):
            if node.buffer_id == buffer or any(
                    isinstance(item, Load) and item.buffer_id == buffer
                    for item in _tree(node.expr)):
                yield node
        elif isinstance(node, LoopPlan):
            yield from _touches(node.body, buffer)
        elif isinstance(node, BranchNode):
            yield from _touches(node.then, buffer)
            yield from _touches(node.otherwise, buffer)


def _tree(node: Any) -> Iterator[Any]:
    """Every node of an `ExprNode` tree, parents before children."""
    yield node
    for name in ("lhs", "rhs", "operand", "then", "otherwise"):
        child = getattr(node, name, None)
        if child is not None and not isinstance(child, tuple):
            yield from _tree(child)
    for child in getattr(node, "operands", ()):
        yield from _tree(child)


def ping_pong_shape(plan: MappingPlan) -> None:
    """Invariant 4 — every ping-pong candidate has `isPingPongCandidate`'s shape (LLD §3.7.3).

    Conditions 2, 3, 5 and 4 of `AIRDependencyScheduleOpt.cpp:1604`, `:1620-1630`, `:1690`:
    a direct child of the candidate loop, first touched by a `get`, exactly one get per
    iteration, and constant enclosing bounds. A violation is M4's own construction bug (B-P23).
    """
    seen = set()
    for buffer, parent, loops, body in _allocations(plan.herd_body):
        seen.add(buffer.name)
        if not buffer.ping_pong_candidate:
            continue
        if parent is None:
            raise _internal(plan, 4, f"buffer {buffer.name!r} is a ping-pong candidate but its "
                                     f"alloc is not the direct child of a loop",
                            buffer=buffer.name)
        for loop in loops:
            for bound, name in ((loop.lo, "lo"), (loop.hi, "hi"), (loop.step, "step")):
                if not bound.is_constant:
                    raise _internal(plan, 4, f"buffer {buffer.name!r} is a ping-pong candidate "
                                             f"inside loop {loop.axis!r}, whose {name} is not "
                                             f"constant", buffer=buffer.name, loop=loop.axis)
        touching = list(_touches(parent.body, buffer.name))
        first = touching[0] if touching else None
        if not isinstance(first, ChannelSite) or first.kind != "get":
            raise _internal(plan, 4, f"buffer {buffer.name!r} is a ping-pong candidate but the "
                                     f"first site touching it is "
                                     f"{'nothing' if first is None else _describe(first)}, not a "
                                     f"get", buffer=buffer.name)
        gets = [node for node in touching
                if isinstance(node, ChannelSite) and node.kind == "get"]
        if len(gets) != 1:
            raise _internal(plan, 4, f"buffer {buffer.name!r} is a ping-pong candidate with "
                                     f"{len(gets)} gets per iteration of {parent.axis!r}, not 1",
                            buffer=buffer.name, gets=[g.id for g in gets])
    for buffer in plan.buffers:
        if buffer.ping_pong_candidate and buffer.name not in seen:
            raise _internal(plan, 4, f"buffer {buffer.name!r} is a ping-pong candidate but the "
                                     f"herd body never allocates it", buffer=buffer.name)


def _describe(node: Any) -> str:
    """A node named the way a diagnostic names it."""
    if isinstance(node, ChannelSite):
        return f"the {node.kind} site {node.id!r}"
    if isinstance(node, StoreNode):
        return f"a store into {node.buffer_id!r}"
    return type(node).__name__


def l1_total(buffers: tuple[BufferPlan, ...], repeated: bool = False) -> int:
    """`sum(bytes × (2 if ping_pong_candidate else 1))` over non-L3 buffers (§5.6 invariant 5).

    `repeated` is R-L1-3: with any repeat factor above 1 the repeat loop around the herd body
    is unrolled by 2 (mlir-air `AIRDependencyScheduleOpt.cpp:1906-1908`) and every buffer is
    allocated twice. Same arithmetic as `m4_mapping.l1_total` and `m3_legality`'s.
    """
    return sum(b.bytes * (2 if repeated or b.ping_pong_candidate else 1)
               for b in buffers if b.level != "L3")


def l1_budget(plan: MappingPlan) -> None:
    """Invariant 5 — the L1 working set fits, and the staged part is exactly what M3 charged.

    The staging subset is the buffers with an `operand`, so the protocol buffers M4 adds itself
    (`recv`, `edge_in`/`edge_out`) are the only allowed difference from `LegalMapping.l1_bytes`
    (LLD §3.3 note 4).
    """
    repeated = any(r > 1 for r in plan.mapping.repeats)
    total = l1_total(plan.buffers, repeated=repeated)
    if total > L1_USABLE:
        raise _internal(plan, 5, f"the plan's L1 total {total} B exceeds the {L1_USABLE} B "
                                 f"budget ({L1_BUDGET} B of tile memory less the "
                                 f"{L1_STACK_RESERVED} B core stack)",
                        plan_l1_bytes=total, l1_budget=L1_USABLE)
    staged = l1_total(tuple(b for b in plan.buffers if b.operand is not None),
                      repeated=repeated)
    if staged != plan.mapping.l1_bytes:
        raise _internal(plan, 5, f"the plan charges {staged} B for the operand-staging buffers "
                                 f"where LegalMapping.l1_bytes says {plan.mapping.l1_bytes} B",
                        plan_l1_bytes=staged, mapping_l1_bytes=plan.mapping.l1_bytes)


def tensor_order(plan: MappingPlan) -> None:
    """Invariant 6 — every read-only tensor precedes every written one (LLD §3.3 note 6).

    `air.api`'s `_check_interface` raises a bare `RuntimeError`, *output tensors must be
    declared after all input tensors* (`python/air/api/_compile.py:226-240`, measured
    REVIEW-round1 P-R4), so a violation would otherwise surface as an `EMIT-AIR-API` at trace
    time with an unhelpful message. W3's `(q, r, S)` is the case that bites.
    """
    written = {p.name for p in plan.mapping.kernel.params if p.is_written}
    names = [t.name for t in plan.tensors]
    if sorted(names) != sorted(p.name for p in plan.mapping.kernel.params):
        raise _internal(plan, 6, f"MappingPlan.tensors is {names}, which is not one entry per "
                                 f"kernel parameter "
                                 f"{[p.name for p in plan.mapping.kernel.params]}",
                        tensors=names)
    for position, name in enumerate(names):
        if name in written and any(later not in written for later in names[position + 1:]):
            after = [later for later in names[position + 1:] if later not in written]
            raise _internal(plan, 6, f"MappingPlan.tensors puts the written tensor {name!r} "
                                     f"before the read-only {after[0]!r}, which air.api's "
                                     f"_check_interface rejects with 'output tensors must be "
                                     f"declared after all input tensors'",
                            tensors=names, written=sorted(written), pair=[name, after[0]])


def names_and_marker(plan: MappingPlan) -> None:
    """Invariants 7 and 8 — the emitted names are set, and the herd marker is where it belongs.

    M0 already enforces most of this (`I56`, `I75`-`I77`), so the assertion is cheap; it is made
    anyway because `self_check` is public and a hand-built plan literal reaches it directly.
    """
    for field in ("launch_name", "segment_name"):
        if not getattr(plan, field):
            raise _internal(plan, 7, f"MappingPlan.{field} is empty: M5 emits it verbatim and "
                                     f"derives no name", field=field)
    marks = [node for node in plan.segment_body if isinstance(node, HerdPlan)]
    if len(marks) != 1 or marks[0] != plan.herd:
        raise _internal(plan, 8, f"MappingPlan.segment_body holds {len(marks)} HerdPlan "
                                 f"marker(s) at top level, not exactly one equal to "
                                 f"MappingPlan.herd", markers=len(marks))
    if any(isinstance(node, HerdPlan) for body in (plan.herd_body,)
           for node in _flatten(body)):
        raise _internal(plan, 8, "MappingPlan.herd_body holds a HerdPlan marker: the marker "
                                 "belongs to segment_body")


def _flatten(nodes: tuple[Any, ...]) -> Iterator[Any]:
    """Every node of a body, descending into loops and branch arms."""
    for node in nodes:
        yield node
        if isinstance(node, LoopPlan):
            yield from _flatten(node.body)
        elif isinstance(node, BranchNode):
            yield from _flatten(node.then)
            yield from _flatten(node.otherwise)


def structural(plan: MappingPlan) -> None:
    """The four structural checks of LLD §3.7.3 and `06-interfaces.md` §5.6 invariants 3-8."""
    bundle_indices(plan)
    ping_pong_shape(plan)
    l1_budget(plan)
    tensor_order(plan)
    names_and_marker(plan)


# --------------------------------------------------------------------------------------------
# §3.8 — P3, the per-core circuit-switched DMA-channel budget
# --------------------------------------------------------------------------------------------


def _endpoints(channel: ChannelPlan, plan: MappingPlan) -> tuple[set[str], set[str]]:
    """`(kinds at herd scope, kinds at an L3 endpoint)` for one channel."""
    tensors = {t.name for t in plan.tensors}
    herd = {s.kind for s in channel.sites if s.scope == "herd"}
    l3 = {s.kind for s in channel.sites if s.scope in ("segment", "launch") and s.buffer in
          tensors}
    return herd, l3


def is_core_to_core(channel: ChannelPlan, plan: MappingPlan) -> bool:
    """True when both ends are core tiles — a flow that always binds a circuit-switched DMA."""
    herd, l3 = _endpoints(channel, plan)
    return not l3 and {"put", "get"} <= herd


def l3_direction(channel: ChannelPlan, plan: MappingPlan) -> str | None:
    """`"in"` for an L3→L1 channel, `"out"` for L1→L3, `None` when it touches no L3 tensor.

    The directions are `AIRDmaToChannel.cpp:1658-1672`'s: input is a herd-side get with a
    launch-side L3 put, output a herd-side put with a launch-side L3 get.
    """
    herd, l3 = _endpoints(channel, plan)
    if "get" in herd and "put" in l3:
        return "in"
    if "put" in herd and "get" in l3:
        return "out"
    return None


def shim_pressure(plan: MappingPlan, direction: str) -> int:
    """The per-shim-column DMA pressure of one direction — `AIRDmaToChannel.cpp:1694-1709`.

    Non-broadcast channels all compete for one column and count one each. A broadcast channel is
    split by `air-specialize-dma-broadcast` into one channel per bundle index, each carrying the
    per-member fan-out `broadcast_shape[d] // size[d]`; `K` such members spanning `C` columns
    cost `ceil(K / C)`. The sum is what the pass compares against `shim-dma-channels-per-col`.
    """
    pressure = 0
    spans: dict[int, int] = {}
    for channel in plan.channels:
        if l3_direction(channel, plan) != direction:
            continue
        if channel.broadcast_shape is None:
            pressure += 1
        else:
            span = max(1, channel.broadcast_shape[0] // channel.size[0])
            spans[span] = spans.get(span, 0) + prod(channel.size)
    for span, members in sorted(spans.items()):
        pressure += -(-members // span)
    return pressure


def may_packet(channel: ChannelPlan, plan: MappingPlan) -> bool:
    """Whether this channel's flows may be packet-switched, and so may share a DMA channel.

    **This is the `CIRCUIT` predicate of LLD §3.8 lines 4-9, which the LLD does not define
    operationally.** It is the transcription of `air-dma-to-channel`'s own auto-upgrade rule
    (`mlir/lib/Transform/AIRDmaToChannel.cpp:1598-1740`, wheel `0.0.1.2026091204+ff95a9b`): per
    segment and per direction, if the per-column shim pressure exceeds `shim-dma-channels-per-col`
    (default 2, `Passes.td:1808-1812`) then **every** L3-attached channel of that direction is
    upgraded to `channel_type = "npu_dma_packet"` and multiplexes; otherwise they stay
    circuit-switched `aie.flow`s. Core-to-core channels are never upgraded.

    **Measured, not contractual** (risks R-19, R-21). The rule is a pass option's default on a
    pinned wheel, not a documented contract, so §3.8 makes exceeding the budget on
    packet-capable channels a *warning* and only a circuit-switched overflow an error. Evidence,
    regenerated this session — five probes through `aircc --device npu1 --output-format=none`,
    one at a time, then `aie.*.mlir` tabulated per channel (`design/PROGRESS-B.md`, phase P3):

    | probe | L3 in-channels (pressure) | lowering of each | measured |
    |---|---|---|---|
    | `q/q7_a.py` (W1) | `A2L1` bcast, `B2L1` bcast → **3** after broadcast specialisation | packet | 3 `aie.packet_flow`, all on `shim_noc_tile_0_0, MM2S, 0` |
    | `q/pi2/w2_pi2.py` (W2, PI=2) | `UIn` → **1** | circuit | 0 packet flows; `aie.flow(shim…, tile_x_2)` |
    | `q/q2_w2.py` (W2, PI=4) | `UIn` → **1** | circuit | 0 packet flows; `aircc` fails, `TileID(1,2) targets same dst` |
    | `q/w3b.py` (W3, no `q`/`r`) | `WestIn` → **1** | circuit | 0 packet flows |
    | `review/w3c.py` (W3, `q`/`r` staged) | `WestIn`, `QIn`, `RIn` → **3** | packet | 9 `aie.packet_flow`; the pass itself printed *auto-upgrading 3 input channels to dma_packet (per-column pressure 3 exceeds shim DMA limit of 2)* |

    Every row is reproduced, including the discriminating pair: `WestIn` alone is circuit in
    `w3b` and packet in `w3c`, which no predicate over the channel's own shape can express — the
    pressure is a property of the whole direction, which is why the signature takes the plan.
    The three candidate discriminators the brief listed are each refuted by a row: an L3 endpoint
    with `broadcast_shape` (`RIn`/`WestIn` are plain and packet), an L3 endpoint whose puts sit
    under a temporal segment-scope loop (`WestIn` is, and is circuit in `w3b`), and the count of
    distinct L3 endpoints per shim (`q7a`'s pressure is 3 from two channels).

    **Not measured**: the outbound half at pressure > 2 — no probe has three L1→L3 channels — and
    any shim-column assignment other than the `same_column` default the pass assumes.
    """
    direction = l3_direction(channel, plan)
    return (direction is not None
            and shim_pressure(plan, direction) > SHIM_DMA_CHANNELS_PER_COL)


def circuit(channel: ChannelPlan, plan: MappingPlan) -> bool:
    """`CIRCUIT` (LLD §3.8 lines 4-9): the channel binds a circuit-switched DMA channel."""
    return is_core_to_core(channel, plan) or (l3_direction(channel, plan) is not None
                                              and not may_packet(channel, plan))


class Pressure(NamedTuple):
    """One `(coordinate, direction)` row of the P3 report: which channels, and how they lower."""

    coord: tuple[int, ...]
    kind: str
    word: str
    budget: int
    hard: tuple[str, ...]
    all_: tuple[str, ...]


def dma_report(plan: MappingPlan) -> tuple[Pressure, ...]:
    """Per PE coordinate and direction, the circuit-switched and the total channel sets.

    `MAX_OVER_EXCLUSIVE_BRANCHES` (LLD §3.8 line 16) is what enumeration gives for free: at one
    concrete coordinate only one arm of a `BranchNode` is live and a contradictory guard is
    false, so W3's PE 0 counts `WestIn | West` once rather than twice. It is sound because
    `air-to-aie` folds the branch away once the coordinate is a literal (`_cond.py:49-54`).
    """
    by_name = {channel.name: channel for channel in plan.channels}
    live: dict[tuple[int, ...], list[Occurrence]] = {
        coord: [] for coord in coordinates(plan.herd.grid)}
    for occurrence in occurrences(plan, herd_only=True):
        if by_name[occurrence.site.channel].channel_type == CASCADE:
            continue          # a cascade op binds no DMA channel at all — see `CASCADE` above
        live[occurrence.coord].append(occurrence)
    out = []
    for coord in coordinates(plan.herd.grid):
        for kind, budget, word in (("get", DMA_IN_MAX, "inbound"),
                                   ("put", DMA_OUT_MAX, "outbound")):
            named = {o.site.channel for o in live[coord] if o.site.kind == kind}
            hard = {name for name in named if circuit(by_name[name], plan)}
            out.append(Pressure(coord, kind, word, budget, tuple(sorted(hard)),
                                tuple(sorted(named))))
    return tuple(out)


def _grid_clause(plan: MappingPlan) -> str:
    """The `grid(...)` clause as the user wrote it — the edit the diagnostic points at."""
    return f"grid({', '.join(str(extent) for extent in plan.mapping.schedule.grid)})"


def dma_channels(plan: MappingPlan) -> None:
    """P3 — the per-core **circuit-switched** DMA budget (LLD §3.8, finding N-1).

    An AIE2 core tile has two S2MM and two MM2S DMA channels. Exceeding the inbound budget with
    circuit-switched flows is an `aie.connect` error raised twenty seconds into `aircc`, naming a
    physical `TileID` the user has never heard of (measured: W2 at `PI = 4`, REVIEW-round1 P-R2);
    exceeding the outbound budget is *silently* accommodated by merging two endpoints onto one
    MM2S (finding N-3), which is a multicast on a circuit-switched fabric and is not established
    to be correct. Exceeding it once packet-capable channels are counted is a **warning**
    (`warnings`), because a checker that rejects a program the toolchain accepts is worse for G2
    than no checker at all.
    """
    for row in dma_report(plan):
        if len(row.hard) <= row.budget:
            continue
        physical = "2 S2MM" if row.kind == "get" else "2 MM2S"
        raise _error(
            "DMA-CHANNELS",
            f"PE {list(row.coord)} needs {len(row.hard)} circuit-switched {row.word} DMA "
            f"channels ({', '.join(row.hard)}) but an AIE2 core tile has {physical} — a budget "
            f"of {row.budget}",
            f"shrink the herd so fewer neighbours meet at one PE, or stage fewer operands from "
            f"L3: {_grid_clause(plan)} puts {len(row.hard)} {row.word} flows on PE "
            f"{list(row.coord)}",
            clause=_grid_clause(plan), coord=list(row.coord), direction=row.word,
            channels=list(row.hard), count=len(row.hard), budget=row.budget,
            circuit_switched=list(row.hard), all_channels=list(row.all_))


def specialize_dim(channel: ChannelPlan) -> int | None:
    """`findSpecializeDim` (mlir-air `AIRMiscPasses.cpp:165-177`): the bundle dim upstream splits.

    `SpecializeChannelBroadcastPattern` (`AIRMiscPasses.cpp:128-524`) matches a channel whose
    `broadcast_shape` is present and not all ones, and splits it on the **first** bundle
    dimension whose extent exceeds 1. `None` means the pattern does not fire.
    """
    if channel.broadcast_shape is None or all(fan == 1 for fan in channel.broadcast_shape):
        return None
    return next((d for d, extent in enumerate(channel.size) if extent > 1), None)


def _herd_dim(index: Expr, coords: tuple[str, ...]) -> int | None:
    """The herd dimension this bundle index **is**, or `None` for anything else.

    Upstream's guarded dispatch (`AIRMiscPasses.cpp:321-328`) fires only when the get's index at
    the specialised dimension is a herd id itself (`idxVal == herdIds[d]`). A constant index
    takes case 1 (a direct rewrite) and anything else takes case 3's `scf.if`; neither builds
    the affine set below, so neither can carry its bound.
    """
    if index.const or len(index.coeffs) != 1:
        return None
    name, coeff = index.coeffs[0]
    return coords.index(name) if coeff == 1 and name in coords else None


def broadcast_guard(plan: MappingPlan) -> None:
    """P3 — the column bound upstream puts on a specialised broadcast's guard (**B-P37**).

    When `air-specialize-dma-broadcast` splits a bundled multicast it wraps each split get in an
    `affine.if` whose set pins the specialised herd coordinate and leaves the others *bounded*:

    ```cpp
    exprs.push_back(getAffineSymbolExpr(d, ctx));                 // 0 <= s_d
    exprs.push_back(numCols - 1 - getAffineSymbolExpr(d, ctx));   // s_d <= numCols - 1
    ```

    — `AIRMiscPasses.cpp:265-271`, and the `numCols` argument is `herd.getNumCols()` at
    `:347-349`. The bound is the herd's **column** count whichever dimension `d` is, so a herd
    with more rows than columns, split on the column axis, admits only rows `0 .. cols-1`: every
    row above falls into the chain's `else` arm and is served by the *last* split channel. The
    first split channel is then short of broadcast destinations, its `S2MM_alloc[i]` is never
    filled, and `air-to-aie` fails on the L3 put with *"'air.channel.put' op failed to get S2MM
    tile for L3 allocation"* (`AIRToAIESchedulingUtils.cpp:3866-3868`).

    Measured on this wheel (W1's clauses, `repeats (1, 1)`, `l1_bytes` 12 288, npu2): physical
    `(2, 2)` exit 0; `(2, 3)` and `(2, 4)` this error; `(3, 2)` and `(4, 2)` — the same six and
    eight cores, transposed — exit 0. Editing only the emitted affine set of the `(2, 3)` module
    from `-s1 + 1 >= 0` to `-s1 + 2 >= 0` and re-running `air-to-aie` gives exit 0, which is what
    pins the cause to this bound and not to a shim resource.

    **A repeat factor on the split axis takes it out of range, which is why npu1 is untouched.**
    The guarded dispatch only fires when the get's index *is* the herd id; at `repeats[dim] > 1`
    `air.api` folds the repeat loop into it (`affine.apply ()[%arg13, %arg17] -> (s0 * n + s1)`,
    `run_strip_mined`, `_trace.py:1675`) and upstream takes its `scf.if` arm instead
    (`AIRMiscPasses.cpp:380-422`), which bounds nothing. `air.api`'s 2-D physical herd on npu1 is
    one column wide (`_trace.py:88-91`), so every multi-column logical grid arrives there with a
    repeat on exactly this axis — measured, W1's clauses at `grid(2, 2)` and `grid(2, 3)` compile
    on npu1 at physical `(1, 2)` and `(1, 3)`, both `repeats (2, 1)`.
    """
    physical = plan.mapping.physical_herd
    if len(physical) != 2 or physical[1] <= physical[0]:
        return
    cols, rows = physical
    for channel in plan.channels:
        dim = specialize_dim(channel)
        # A bundle dimension only names a herd dimension when the two ranks agree, which is
        # what makes `repeats[dim]` and `_herd_dim` below mean anything.
        if dim is None or len(channel.size) != len(physical) or plan.mapping.repeats[dim] != 1:
            continue
        for site in channel.sites:
            if site.kind != "get" or site.scope != "herd" or len(site.indices) <= dim:
                continue
            if _herd_dim(site.indices[dim], plan.herd.coords) != 0:
                continue
            raise _error(
                "DMA-CHANNELS",
                f"the physical herd is {cols} column(s) by {rows} row(s), and channel "
                f"{channel.name!r} (size {list(channel.size)}, broadcast_shape "
                f"{list(channel.broadcast_shape or ())}) is a multicast that "
                f"air-specialize-dma-broadcast splits on the column axis (bundle dim {dim}, "
                f"mlir-air AIRMiscPasses.cpp:165-177). The affine guard it builds around each "
                f"split bounds the *row* coordinate by the column count — "
                f"'numCols - 1 - s_d >= 0' at AIRMiscPasses.cpp:265-271, the argument being "
                f"herd.getNumCols() at :347-349 — so it admits rows 0..{cols - 1} where the "
                f"herd has {rows}: {rows - cols} row(s) fall into the else arm, are served by "
                f"the last split channel, and leave the first split channel's broadcast "
                f"destination unallocated. air-to-aie then fails on the L3 put with "
                f"\"'air.channel.put' op failed to get S2MM tile for L3 allocation\" "
                f"(AIRToAIESchedulingUtils.cpp:3866-3868)",
                f"give the herd at least as many columns as rows: exchange the two place(px=, "
                f"py=) axes together with the two {_grid_clause(plan)} extents, or lower the "
                f"second extent to at most the first. Measured on this wheel, the same clauses "
                f"at physical (2, 2) compile on both generations where ({cols}, {rows}) does "
                f"not",
                clause=_grid_clause(plan), channel=channel.name,
                channel_size=list(channel.size),
                broadcast_shape=list(channel.broadcast_shape or ()), specialize_dim=dim,
                herd_physical=list(physical), rows=rows, columns=cols,
                rows_admitted=cols, rows_unserved=rows - cols, site=site.id)


def fans_down_the_column(channel: ChannelPlan) -> bool:
    """The channel multicasts along the herd's **row** axis — more rows than bundle positions."""
    fan = channel.broadcast_shape
    return fan is not None and len(fan) == 2 and fan[1] > channel.size[1]


def fill_flows(plan: MappingPlan) -> int:
    """The `aie.packet_flow` count of the L3→L1 fill — one per **specialisable** bundle position.

    A bundle index runs over `size[d]` positions, but only a dimension whose repeat factor is 1
    is split: at `repeats[d] > 1` the index is an `affine.apply` of the herd id and the repeat
    variable, `specializeChannelBundle` cannot resolve it, and the whole bundle stays one flow
    (the same condition `broadcast_guard` reads, `AIRMiscPasses.cpp:321-328`).

    Verified against the emitted `aie.packet_flow` count, W1's clauses, every shape measured
    this pass: physical `(1, 2)` **3**, `(1, 3)` **4** (reached both as `grid(1, 3)` and as
    `grid(2, 3)` with `repeats (2, 1)`), `(1, 4)` **5**, `(2, 4)` from `grid(4, 4)` **5**, and
    W1-flip 2-D at `(1, 4)` **8**.
    """
    repeats = plan.mapping.repeats
    return sum(prod(extent if repeat == 1 else 1
                    for extent, repeat in zip(channel.size, repeats))
               for channel in plan.channels if l3_direction(channel, plan) == "in")


def msels(plan: MappingPlan) -> None:
    """P3 — the switchbox master selects the fill spends on a stacked herd (**B-P36**).

    Every L3→L1 packet flow serving a column passes that column's bottom core tile, and each one
    costs a master select on that tile's arbiter 0 **when one of them is a multicast down the
    column**: the multicast's port set there is `{DMA, North}`, which only partially overlaps the
    `{DMA}` or `{North}` of each point-to-point flow, and a partial overlap makes upstream open a
    fresh amsel on the *same* arbiter rather than share one. `MSELS_PER_ARBITER` is the cap.

    Measured this pass, W1's clauses, `aie.amsel` counted on `%tile_0_2` of the routed IR
    (`aie-opt --aie-create-pathfinder-flows=route-packet` over `air_project/npu.src.mlir`), on
    **both** generations unless noted:

    | physical herd | fill flows | multicast | amsels on arbiter 0 | `aircc` |
    |---|---|---|---|---|
    | `(1, 2)` | 3 | yes | `<0>(0..2)` = 3 | exit 0 |
    | `(1, 3)` | 4 | yes | `<0>(0..3)` = 4 | exit 0 |
    | `(1, 4)` | 5 | yes | — | *used up all its msels* |
    | `(2, 2)` (npu2) | 4 | yes | — | exit 0 |
    | `(2, 4)` (npu2, from `grid(4, 4)`) | 5 | yes | — | *used up all its msels* |
    | `(1, 4)` W1-flip 2-D | 8 | **no** | `<0>(0)`, `<1>(0)` = 2 | exit 0 |

    `(1, 3)` and `(2, 2)` sit exactly on the cap and are accepted; five flows are refused at
    both `(1, 4)` and `(2, 4)`, and `(1, 4)` is refused whether it is reached as `grid(1, 4)` or
    as `grid(2, 4)` folded with `repeats (2, 1)`.

    **The flip is the reason the count is gated on the multicast, and the reason this rule was
    not written a pass earlier.** Its eight flows are every one of them single-destination
    (`aie.packet_flow` with one `packet_dest`), so their port sets match exactly and they share
    two amsels — one for `{DMA}`, one for `{North}` — on two different arbiters. A rule counting
    flows alone would have rejected it, and §3.8's standing argument is that rejecting a program
    the toolchain accepts is worse than no checker.

    **What is assumed, not proved.** That every fill flow of a herd enters the same shim column
    and so meets at one bottom tile. It is what all six shapes above show — every
    `aie.packet_source` is `%shim_noc_tile_0_0`, including the two-column ones — but the column a
    flow climbs is chosen by the shim bin packing
    (`AIRToAIESchedulingUtils.cpp:3825-3845`) and by the pathfinder, not by the plan. A herd
    whose fills were spread over columns would be rejected here and compile.
    """
    physical = plan.mapping.physical_herd
    if len(physical) != 2 or physical[1] < 2:
        return
    inbound = [channel for channel in plan.channels if l3_direction(channel, plan) == "in"]
    if not any(fans_down_the_column(channel) for channel in inbound):
        return
    flows = fill_flows(plan)
    if flows <= MSELS_PER_ARBITER:
        return
    raise _error(
        "DMA-CHANNELS",
        f"the herd is {physical[0]} column(s) of {physical[1]} rows, and its {flows} L3->L1 "
        f"fill flows ({', '.join(channel.name for channel in inbound)}) all climb through one "
        f"column's bottom core tile. One of them multicasts down the column, so its switchbox "
        f"port set only partially overlaps each of the others and every flow takes a master "
        f"select of its own on one arbiter: {flows} against the {MSELS_PER_ARBITER} of "
        f"mlir-aie's 'int numMselsPerArbiter = 4' (AIECreatePathFindFlows.cpp; aie.amsel's "
        f"msel is confined to 0..3 in AIEOps.td). aiecc fails with \"'aie.tile' op tile op "
        f"arbiter 0 has used up all its msels\" — measured, {MSELS_PER_ARBITER} such flows "
        f"compile on both generations and {MSELS_PER_ARBITER + 1} do not",
        f"shorten the column to at most {MSELS_PER_ARBITER - 1} rows, or stage one fewer "
        f"operand from L3: this target folds {_grid_clause(plan)} onto a "
        f"{physical[0]} x {physical[1]} herd (air.api's PHYSICAL_HERD, _trace.py:88-91), so a "
        f"smaller second grid extent is what shortens it",
        clause=_grid_clause(plan), herd_physical=list(physical), flows=flows,
        budget=MSELS_PER_ARBITER, channels=[channel.name for channel in inbound],
        multicast=[channel.name for channel in inbound if fans_down_the_column(channel)])


def warnings(plan: MappingPlan) -> tuple[str, ...]:
    """The P3 warnings — a budget exceeded only once packet-capable channels are counted.

    `self_check` returns `None` by contract (`06-interfaces.md` §7.2), so a §3.8 warning is
    recorded here rather than returned: the summary calls this and prints it, and the plan is
    returned unchanged (LLD §5, last paragraph).
    """
    out = []
    for row in dma_report(plan):
        if len(row.hard) > row.budget or len(row.all_) <= row.budget:
            continue
        packet = [name for name in row.all_ if name not in row.hard]
        physical = "2 S2MM" if row.kind == "get" else "2 MM2S"
        out.append(
            f"PE {list(row.coord)} names {len(row.all_)} {row.word} channels "
            f"({', '.join(row.all_)}) against {physical}, but {', '.join(packet)} "
            f"lower to packet flows and multiplex onto one shim DMA channel; the "
            f"circuit-switched count is {len(row.hard)}, within the budget of {row.budget} "
            f"— measured, not contractual "
            f"(design/03-lld-M4-mapping.md §3.8, risks R-19/R-21)")
    return tuple(out)


# --------------------------------------------------------------------------------------------
# The entry point (`06-interfaces.md` §7.2)
# --------------------------------------------------------------------------------------------


def self_check(plan: MappingPlan) -> None:
    """Assert put/get balance, channel-graph acyclicity, the plan invariants and the DMA budget.

    The checks of LLD §3.7-§3.8, in this order: P1' balance, P2b acyclicity, the structural
    invariants, the P3 DMA-channel budget and — §3.8's 2026-09-15 erratum — P3's broadcast-guard
    bound (**B-P37**) and its master-select budget (**B-P36**). Raises `MappingError` and
    nothing else (NFR-7);
    returns `None` on success, with any P3 warning available from `warnings(plan)`.

    `m4.plan` calls it on its own result before returning, so a `MappingPlan` that leaves M4 has
    always been checked; it is public so the corrupted-plan negatives can feed it hand-built
    literals (`design/04-test-plan.md` §2, M4).
    """
    balance(plan)
    acyclicity(plan)
    structural(plan)
    dma_channels(plan)
    broadcast_guard(plan)
    msels(plan)
