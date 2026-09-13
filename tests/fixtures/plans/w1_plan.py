"""W1 — GEMM, output-stationary. The `MappingPlan` of `03-lld-M4-mapping.md` §6.1.

Every node below is transcribed from §6.1's segment body (lines 0-9) and herd body (lines 0-10);
the names are §3.1's derivation table; the regions are §3.4; the summary is §3.9's format. The
`LegalMapping` it is built on is `tests/fixtures/mappings/w1_legal.legal(target)`.

Written by B at P0c: M4's stand-in until D2, then M5's input fixture — it is what makes
`test_emitter_makes_no_decisions` mean anything, and a D6 test asserts
`m4.plan(w1_legal.legal()) == w1_plan.plan()` (`03-lld-M7-tests.md` §3.8). **Person B owns it.**

Four readings §6.1 leaves open are recorded in `design/PROGRESS-B.md`, phase P0c:
`LoopPlan.depth` counts enclosing loops (0 at the top of its own body); `ChannelSite.order` is
the index in the enclosing body; an `Expr` over a loop IV names the `LoopPlan.axis` that binds
it; and `plan.buffers` is in allocation order (`06-interfaces.md` §5.6).

At `CONTRACT_VERSION = 4` §6.1 line 6's `<HERD>` is a real node: the `HerdPlan` at
`segment_body` index 2 (`06-interfaces.md` §5.6 invariant 8), which closed B-P7.

At `CONTRACT_VERSION = 5` two of §6.1's printed names are superseded by `06-interfaces.md`
§5.5's naming rule (the ruling that closed B-P18): the compute and zeroing nests are named by
the post-tiling axis they realise — `i1`, `j1`, `k1` — not positionally `m`, `n`, `t`, `m0`,
`n0`. And `C` is `declared`, because `stationary("C")` names its delivery (B-P17).
"""

from __future__ import annotations

from spatial.model import (BinOp, BufferPlan, ChannelPlan, ChannelSite, Const, Dtype, HerdPlan,
                           Load, LoopPlan, MappingPlan, MappingSummary, Region, StoreNode)

from tests.fixtures.mappings import L1_BUDGET, ZERO, const, lin
from tests.fixtures.mappings import w1_legal

M = N = K = 64
TM = TN = 32
TK = 16
PI = PJ = 2
F32 = 4

EMPTY = Region((), (), ())
"""The L1 end of a whole-buffer transfer (`03-lld-M4-mapping.md` §3.4, `(%alloc[] [] [])`)."""


def _site(channel: str, kind: str, order: int, scope: str, **rest) -> ChannelSite:
    """One site, with §3.1's `id` rule and W1's plan-wide `is_async=False`, `depends_on=()`."""
    return ChannelSite(id=f"{channel}.{kind}.{order}@{scope}", kind=kind, channel=channel,
                       scope=scope, guard=None, is_async=False, depends_on=(), order=order,
                       **rest)


# --------------------------------------------------------------------------------------------
# L3 interface — read-only params first, then written (§5.6 invariant 6). §3.3 TENSOR_PLAN.
# --------------------------------------------------------------------------------------------

TENSORS = tuple(
    BufferPlan(name=name, operand=name, level="L3", scope="tensor", shape=shape,
               dtype=Dtype.f32, bytes=shape[0] * shape[1] * F32, loop_depth=0,
               ping_pong_candidate=False)
    for name, shape in (("A", (M, K)), ("B", (K, N)), ("C", (M, N)))
)

# --------------------------------------------------------------------------------------------
# L1 buffers — §6.1's table, in allocation order
# --------------------------------------------------------------------------------------------

ACC = BufferPlan(name="acc", operand="C", level="L1", scope="herd.private", shape=(TM, TN),
                 dtype=Dtype.f32, bytes=TM * TN * F32, loop_depth=0, ping_pong_candidate=False)
A_TILE = BufferPlan(name="a", operand="A", level="L1", scope="herd.private", shape=(TM, TK),
                    dtype=Dtype.f32, bytes=TM * TK * F32, loop_depth=1,
                    ping_pong_candidate=True)
B_TILE = BufferPlan(name="b", operand="B", level="L1", scope="herd.private", shape=(TK, TN),
                    dtype=Dtype.f32, bytes=TK * TN * F32, loop_depth=1,
                    ping_pong_candidate=True)
BUFFERS = (ACC, A_TILE, B_TILE)

# --------------------------------------------------------------------------------------------
# Sites. §6.1's scopes and orders; §3.4's regions.
# --------------------------------------------------------------------------------------------

A_PUT = _site("A2L1", "put", 0, "segment", indices=(lin("pi_bundle"), ZERO), buffer="A",
              # A[pi*TM : (pi+1)*TM, kk : kk+TK] on A [64, 64] f32
              region=Region(offsets=(lin("pi_bundle", TM), lin("k0")),
                            sizes=(TM, TK), strides=(K, 1)))
B_PUT = _site("B2L1", "put", 0, "segment", indices=(ZERO, lin("pj_bundle")), buffer="B",
              region=Region(offsets=(lin("k0"), lin("pj_bundle", TN)),
                            sizes=(TK, TN), strides=(N, 1)))
A_GET = _site("A2L1", "get", 2, "herd", indices=(lin("tx"), lin("ty")), buffer="a", region=EMPTY)
B_GET = _site("B2L1", "get", 3, "herd", indices=(lin("tx"), lin("ty")), buffer="b", region=EMPTY)
C_PUT = _site("C2L3", "put", 3, "herd", indices=(lin("tx"), lin("ty")), buffer="acc",
              region=EMPTY)
C_GET = _site("C2L3", "get", 0, "segment", indices=(lin("i_drain"), lin("j_drain")), buffer="C",
              region=Region(offsets=(lin("i_drain", TM), lin("j_drain", TN)),
                            sizes=(TM, TN), strides=(N, 1)))

CHANNELS = (
    # A does not index j, so it multicasts along the PE row; B along the column (§3.2).
    ChannelPlan(name="A2L1", size=(PI, 1), broadcast_shape=(PI, PJ), channel_type=None,
                chain_direction=None, dtype=Dtype.f32, sites=(A_PUT, A_GET)),
    ChannelPlan(name="B2L1", size=(1, PJ), broadcast_shape=(PI, PJ), channel_type=None,
                chain_direction=None, dtype=Dtype.f32, sites=(B_PUT, B_GET)),
    ChannelPlan(name="C2L3", size=(PI, PJ), broadcast_shape=None, channel_type=None,
                chain_direction=None, dtype=Dtype.f32, sites=(C_PUT, C_GET)),
)

# --------------------------------------------------------------------------------------------
# Bodies
# --------------------------------------------------------------------------------------------

# 0-2: the bundle index must be a trace-time constant, so its loop is a Python loop;
#      the k loop reuses no L1 buffer here but stays air.sequential (LOOP_KIND's default).
FILL_A = LoopPlan(axis="pi_bundle", lo=ZERO, hi=const(PI), step=const(1), kind="unrolled",
                  depth=0,
                  body=(LoopPlan(axis="k0", lo=ZERO, hi=const(K), step=const(TK),
                                 kind="sequential", depth=1, body=(A_PUT,)),))
# 3-5
FILL_B = LoopPlan(axis="pj_bundle", lo=ZERO, hi=const(PJ), step=const(1), kind="unrolled",
                  depth=0,
                  body=(LoopPlan(axis="k0", lo=ZERO, hi=const(K), step=const(TK),
                                 kind="sequential", depth=1, body=(B_PUT,)),))
# 7-9: the drain, after the herd
DRAIN = LoopPlan(axis="i_drain", lo=ZERO, hi=const(PI), step=const(1), kind="unrolled", depth=0,
                 body=(LoopPlan(axis="j_drain", lo=ZERO, hi=const(PJ), step=const(1),
                                kind="unrolled", depth=1, body=(C_GET,)),))


def segment_body(herd: HerdPlan) -> tuple:
    """§6.1's segment body. Line 6 `<HERD>` **is** the `HerdPlan` node, at index 2.

    `06-interfaces.md` §5.6 invariant 8 at `CONTRACT_VERSION = 4`: `segment_body` holds exactly
    one `HerdPlan`, at top level, equal to `MappingPlan.herd`. It closes B-P7.
    """
    return (FILL_A, FILL_B, herd, DRAIN)

# The compute and zeroing nests are named by the post-tiling axis they realise — `i1`, `j1`,
# `k1` — and both nests realise `i1`/`j1` (`06-interfaces.md` §5.5 at CONTRACT_VERSION 5).
_ZERO_STORE = StoreNode(buffer_id="acc", subscripts=(lin("i1"), lin("j1")),
                        expr=Const(value=0.0, text="0.0", dtype=Dtype.f32))
_ACCUMULATE = StoreNode(
    buffer_id="acc", subscripts=(lin("i1"), lin("j1")),
    expr=BinOp(op="+",
               lhs=Load(buffer_id="acc", subscripts=(lin("i1"), lin("j1"))),
               rhs=BinOp(op="*",
                         lhs=Load(buffer_id="a", subscripts=(lin("i1"), lin("k1"))),
                         rhs=Load(buffer_id="b", subscripts=(lin("k1"), lin("j1"))))))
"""`Statement.expr` (§2.4 at v5) with each kernel `Load` rewritten into its L1 buffer: the
placed contribution `tx*TM` and the streamed `k0` cancel against the staged slab's origin."""

HERD_BODY = (
    ACC,                                                        # 0: air.alloc, depth 0
    LoopPlan(axis="i1", lo=ZERO, hi=const(TM), step=const(1), kind="sequential", depth=0,
             body=(LoopPlan(axis="j1", lo=ZERO, hi=const(TN), step=const(1),
                            kind="sequential", depth=1, body=(_ZERO_STORE,)),)),   # 1-3
    LoopPlan(axis="k0", lo=ZERO, hi=const(K), step=const(TK), kind="sequential", depth=0,
             body=(A_TILE,                                      # 5: DIRECT child of the K loop
                   B_TILE,                                      # 6: DIRECT child of the K loop
                   A_GET,                                       # 7
                   B_GET,                                       # 8
                   LoopPlan(axis="i1", lo=ZERO, hi=const(TM), step=const(1),
                            kind="sequential", depth=1,
                            body=(LoopPlan(axis="j1", lo=ZERO, hi=const(TN), step=const(1),
                                           kind="sequential", depth=2,
                                           body=(LoopPlan(axis="k1", lo=ZERO, hi=const(TK),
                                                          step=const(1), kind="sequential",
                                                          depth=3, body=(_ACCUMULATE,)),)),)),
                   )),                                          # 9
    C_PUT,                                                      # 10
)

DELIVERY = (("A", "MULTICAST", "py", False),
            ("B", "MULTICAST", "px", False),
            ("C", "STATIONARY", None, True))
"""`test_M1_trichotomy`'s expected tuple (`03-lld-M4-mapping.md` §7).

`C` is **declared**: `stationary("C")` names its delivery, and `declared` is a fact about the
schedule rather than about whether the derivation agreed (the ruling that closed **B-P17**)."""

RESIDENCY = (("A", "re-fetched per k0"),
             ("B", "re-fetched per k0"),
             ("C", "resident for the whole run"))
"""W1's only temporal tile axis is `k0` (`03-lld-M4-mapping.md` §3.9)."""

REDUCTION_SPLIT = ("span{e_k0, e_k1}", "{}")
"""The **tiled** split (§3.9 line 10): neither `k0` nor `k1` is placed, so all of `R` is
temporal. `LegalMapping.r_time`/`.r_space` stay untiled and are `span{e_k}` / `{}`."""

SUMMARY_CHANNELS = tuple((c.name, c.size, c.broadcast_shape) for c in CHANNELS)


def _summary(physical: tuple[int, ...], repeats: tuple[int, ...]) -> MappingSummary:
    """`03-lld-M4-mapping.md` §3.9's `SUMMARY`, line by line."""
    lines = [
        # lines 7-8: the delivery block — FR-M11's three literal strings
        "A: multicast along py (derived)",
        "B: multicast along px (derived)",
        "C: stationary (declared)",
        # lines 8a-8c: the residency block (RULING 9)
        "A: multicast along py, re-fetched per k0",
        "B: multicast along px, re-fetched per k0",
        "C: stationary (spatial), resident for the whole run",
        # line 9
        f"herd: logical {(PI, PJ)} physical {physical} repeats {repeats}",
        # line 10
        f"reduction (tiled axes): R_time = {REDUCTION_SPLIT[0]}, R_space = {REDUCTION_SPLIT[1]}",
    ]
    for buffer in BUFFERS:                                              # lines 11-12
        lines.append(f"  {buffer.name} {buffer.shape} {buffer.dtype.mlir} = {buffer.bytes} B"
                     + ("  x2 (ping-pong)" if buffer.ping_pong_candidate else ""))
    lines.append(f"L1: {w1_legal.legal().l1_bytes} of {L1_BUDGET} bytes")   # line 13
    for channel in CHANNELS:                                            # lines 14-16
        lines.append(f"  {channel.name} size={channel.size}"
                     + (f" broadcast_shape={channel.broadcast_shape}"
                        if channel.broadcast_shape is not None else "")
                     + (f" type={channel.channel_type}"
                        if channel.channel_type is not None else ""))
    return MappingSummary(
        lines=tuple(lines),
        residency=RESIDENCY,
        herd_logical=(PI, PJ),
        herd_physical=physical,
        repeats=repeats,
        reduction_split=REDUCTION_SPLIT,
        l1_bytes=w1_legal.legal().l1_bytes,
        l1_budget=L1_BUDGET,
        channels=SUMMARY_CHANNELS,
    )


def plan(target: str = "npu1") -> MappingPlan:
    """W1's `MappingPlan`, node by node from `03-lld-M4-mapping.md` §6.1."""
    mapping = w1_legal.legal(target)
    herd = HerdPlan(name="gemm_herd", grid=(PI, PJ), shape=mapping.physical_herd, at=None,
                    coords=("tx", "ty"))
    return MappingPlan(
        mapping=mapping,
        tensors=TENSORS,
        launch_name="gemm",                                 # kernel.name (§3.1)
        segment_name="gemm_seg",                            # f"{kernel.name}_seg" (§3.1)
        herd=herd,
        buffers=BUFFERS,
        channels=CHANNELS,
        segment_body=segment_body(herd),
        herd_body=HERD_BODY,
        delivery=DELIVERY,
        summary=_summary(mapping.physical_herd, mapping.repeats),
    )


__all__ = ["EMPTY", "TENSORS", "BUFFERS", "CHANNELS", "FILL_A", "FILL_B", "DRAIN",
           "segment_body", "HERD_BODY", "DELIVERY", "RESIDENCY", "REDUCTION_SPLIT", "plan"]
