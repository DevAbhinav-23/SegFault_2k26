"""The synthetic 2-PE halo plan the P1'/P2b/P3 negatives are built from. Owner: Person B.

**This is not W2.** W2's real protocol lands at P4 (`03-lld-M4-mapping.md` §3.6.1); what is here
is the smallest hand-written plan with the *shape* §3.6.1 and `02-hld.md` §7.2 specify — two
halo channels, one bundle index per physical link, the site order
`PUT(north) → PUT(south) → GET(north ghost) → GET(south ghost)`, and nothing else. It is built on
`w2_legal.legal("npu1", T=4, PI=PI)` because `MappingPlan.mapping` is required and a
`LegalMapping`-free stand-in is not constructible; its two strips charge exactly
`LegalMapping.l1_bytes` for every `PI`, so invariant 5 holds without a special case.

The corruptions each live in their own module and call `plan(...)` with one flag flipped:
`halo_guard_dropped`, `halo_reversed`, `dma_three_inbound`, `dma_packet_warn`.
"""

from __future__ import annotations

from spatial.model import (BufferPlan, ChannelPlan, ChannelSite, Dtype, Guard, HerdPlan, LoopPlan,
                           MappingPlan, MappingSummary, Region)

from tests.fixtures.mappings import L1_BUDGET, ZERO, const, lin, w2_legal

F32 = 4
EMPTY = Region((), (), ())
"""The L1 end of a whole-buffer transfer (`03-lld-M4-mapping.md` §3.4)."""


def site(channel: str, kind: str, order: int, scope: str, **rest) -> ChannelSite:
    """One site, with §3.1's `id` rule. The halo puts are async and its gets take no token."""
    rest.setdefault("guard", None)
    rest.setdefault("is_async", kind == "put")
    return ChannelSite(id=f"{channel}.{kind}.{order}@{scope}", kind=kind, channel=channel,
                       scope=scope, depends_on=(), order=order, **rest)


def summary(mapping, herd: HerdPlan, buffers: tuple[BufferPlan, ...],
            channels: tuple[ChannelPlan, ...]) -> MappingSummary:
    """A `MappingSummary` for a synthetic plan: the facts, rendered in §3.9's format."""
    lines = ["U: stationary (declared)", "U: stationary (spatial), resident for the whole run"]
    lines.append(f"herd: logical {herd.grid} physical {mapping.physical_herd} "
                 f"repeats {mapping.repeats}")
    lines.append("reduction (tiled axes): R_time = {}, R_space = {}")
    total = sum(b.bytes for b in buffers)
    lines.append(f"L1: {total} of {L1_BUDGET} bytes")
    return MappingSummary(lines=tuple(lines), residency=(("U", "resident for the whole run"),),
                          herd_logical=herd.grid, herd_physical=mapping.physical_herd,
                          repeats=mapping.repeats, reduction_split=("{}", "{}"), l1_bytes=total,
                          l1_budget=L1_BUDGET,
                          channels=tuple((c.name, c.size, c.broadcast_shape) for c in channels))


def tensor(mapping) -> BufferPlan:
    """`U`, the kernel's one L3 parameter, resolved through `KernelModel.bindings`."""
    shape = tuple(dict(mapping.kernel.bindings).get(extent, extent)
                  for extent in mapping.kernel.params[0].shape)
    return BufferPlan(name="U", operand="U", level="L3", scope="tensor", shape=shape,
                      dtype=Dtype.f32, bytes=shape[0] * shape[1] * shape[2] * F32, loop_depth=0,
                      ping_pong_candidate=False)


def plan(PI: int = 2, *, reversed_order: bool = False, north_guard: bool = True,
         staging: tuple[str, ...] = ()) -> MappingPlan:
    """The halo plan at `PI` PEs, optionally corrupted.

    `reversed_order` emits `[GET, GET, PUT, PUT]` — the cycle of §3.7.2's worked example.
    `north_guard=False` drops the guard from the north ghost get — the per-branch imbalance.
    `staging` names extra L3→L1 channels, each a segment put per PE and one herd get, which is
    what moves the P3 inbound count.
    """
    mapping = w2_legal.legal("npu1", T=4, PI=PI)
    HS = w2_legal.H // PI
    W = w2_legal.W
    rows, strip = HS + 2, (HS + 2) * W * F32
    herd = HerdPlan(name="jacobi_herd", grid=(PI,), shape=mapping.physical_herd, at=None,
                    coords=("tx",))
    buffers = tuple(
        BufferPlan(name=name, operand="U", level="L1", scope="herd.private", shape=(rows, W),
                   dtype=Dtype.f32, bytes=strip, loop_depth=0, ping_pong_candidate=False)
        for name in ("cur", "next"))

    def band(lo: int) -> Region:
        """One row of the strip — the halo payload (`03-lld-M4-mapping.md` §3.4's worked form)."""
        return Region(offsets=(const(lo), ZERO), sizes=(1, W), strides=(W, 1))

    stage_sites, stage_channels = [], []
    for position, name in enumerate(staging):       # herd body: the allocs, then the staging
        put = site(name, "put", 0, "segment", indices=(lin(f"{name}_stage"),), buffer="U",
                   region=Region(offsets=(ZERO, lin(f"{name}_stage", HS), ZERO),
                                 sizes=(1, rows, W),
                                 strides=((w2_legal.H + 2) * W, W, 1)))
        get = site(name, "get", len(buffers) + position, "herd", indices=(lin("tx"),),
                   buffer=buffers[position % len(buffers)].name, region=EMPTY)
        stage_sites.append((put, get))
        stage_channels.append(ChannelPlan(name=name, size=(PI,), broadcast_shape=None,
                                          channel_type=None, chain_direction=None,
                                          dtype=Dtype.f32, sites=(put, get)))

    order = len(buffers) + len(staging)             # then the four exchange sites, in order
    kinds = ["get", "get", "put", "put"] if reversed_order else ["put", "put", "get", "get"]
    where = {("ToNorth", "put"): (lin("tx", 1, -1), band(1), Guard("tx", ">", ZERO)),
             ("ToSouth", "put"): (lin("tx"), band(HS), Guard("tx", "<", const(PI - 1))),
             ("ToNorth", "get"): (lin("tx"), band(HS + 1),
                                  Guard("tx", "<", const(PI - 1)) if north_guard else None),
             ("ToSouth", "get"): (lin("tx", 1, -1), band(0), Guard("tx", ">", ZERO))}
    exchange = []
    for position, (name, kind) in enumerate(zip(["ToNorth", "ToSouth"] * 2, kinds)):
        index, region, guard = where[(name, kind)]
        exchange.append(site(name, kind, order + position, "herd", indices=(index,),
                             buffer="cur", region=region, guard=guard))

    channels = tuple(sorted(
        [ChannelPlan(name=name, size=(PI - 1,), broadcast_shape=None, channel_type=None,
                     chain_direction=None, dtype=Dtype.f32,
                     sites=tuple(s for s in exchange if s.channel == name))
         for name in ("ToNorth", "ToSouth")] + stage_channels, key=lambda c: c.name))

    segment_body = tuple(
        LoopPlan(axis=f"{name}_stage", lo=ZERO, hi=const(PI), step=const(1), kind="unrolled",
                 depth=0, body=(put,))
        for name, (put, _get) in zip(staging, stage_sites)) + (herd,)
    herd_body = buffers + tuple(get for _put, get in stage_sites) + tuple(exchange)
    return MappingPlan(mapping=mapping, tensors=(tensor(mapping),), launch_name="jacobi",
                       segment_name="jacobi_seg", herd=herd, buffers=buffers, channels=channels,
                       segment_body=segment_body, herd_body=herd_body,
                       delivery=(("U", "STATIONARY", None, True),),
                       summary=summary(mapping, herd, buffers, channels))


__all__ = ["EMPTY", "site", "summary", "tensor", "plan"]
