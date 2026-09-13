"""A synthetic 2-PE plan whose head and body gets sit on the two arms of one `BranchNode`.

**Not a corruption** — this plan is legal and `self_check` accepts it. It lives here because it is
what the two `BranchNode` rows of `03-lld-M4-mapping.md` §7 need and no real plan has a branch
until W3 lands at P5: `test_P3_dma_exclusive_branches` (two gets on mutually exclusive arms count
as **one** inbound channel, §3.8's `MAX_OVER_EXCLUSIVE_BRANCHES`) and
`test_M10_branches_not_joined` (rule 3 of §3.7.2 draws **no** edge between two arms).

The shape is W3's west chain in miniature, on `w2_legal.legal("npu1")` because
`MappingPlan.mapping` is required: PE 0 takes its edge value from L3 (`WestIn`), every other PE
from its neighbour (`West`), and both PEs drain a band to L3 (`SOut`).
"""

from __future__ import annotations

from spatial.model import (BranchNode, BufferPlan, ChannelPlan, Dtype, Guard, HerdPlan, LoopPlan,
                           MappingPlan, Region)

from tests.fixtures.corrupt import halo
from tests.fixtures.mappings import ZERO, const, lin, w2_legal

PI = 2
HS = w2_legal.H // PI
W = w2_legal.W
BAND = Region(offsets=(ZERO, lin("tx", HS), ZERO), sizes=(1, HS, W),
              strides=((w2_legal.H + 2) * W, W, 1))
"""The L3 slab PE `tx` drains — rows `tx·HS …` of plane 0 of `U`."""

WEST_IN_PUT = halo.site("WestIn", "put", 0, "segment", indices=(ZERO,), buffer="U", region=BAND)
WEST_IN_GET = halo.site("WestIn", "get", 0, "herd", indices=(ZERO,), buffer="cur",
                        region=halo.EMPTY)
WEST_GET = halo.site("West", "get", 0, "herd", indices=(lin("tx", 1, -1),), buffer="cur",
                     region=halo.EMPTY)
WEST_PUT = halo.site("West", "put", 3, "herd", indices=(lin("tx"),), buffer="cur",
                     region=halo.EMPTY, guard=Guard("tx", "<", const(PI - 1)))
SOUT_PUT = halo.site("SOut", "put", 4, "herd", indices=(lin("tx"),), buffer="cur",
                     region=halo.EMPTY)
SOUT_GET = halo.site("SOut", "get", 0, "segment", indices=(lin("tx_drain"),), buffer="U",
                     region=Region(offsets=(ZERO, lin("tx_drain", HS), ZERO), sizes=(1, HS, W),
                                   strides=((w2_legal.H + 2) * W, W, 1)))

HEAD_OR_BODY = BranchNode(predicate=Guard("tx", "==", ZERO), then=(WEST_IN_GET,),
                          otherwise=(WEST_GET,))
"""`with ops.branch(tx == 0): WestIn.get(...)` / `with h.otherwise(): West.get(...)` — W3's head."""


def plan() -> MappingPlan:
    """A legal 2-PE plan with one `BranchNode`, for the exclusive-branch and rule-3 tests."""
    mapping = w2_legal.legal("npu1", T=4, PI=PI)
    herd = HerdPlan(name="chain_herd", grid=(PI,), shape=mapping.physical_herd, at=None,
                    coords=("tx",))
    buffers = tuple(
        BufferPlan(name=name, operand="U", level="L1", scope="herd.private", shape=(HS + 2, W),
                   dtype=Dtype.f32, bytes=(HS + 2) * W * 4, loop_depth=0,
                   ping_pong_candidate=False)
        for name in ("cur", "next"))
    channels = (
        ChannelPlan(name="SOut", size=(PI,), broadcast_shape=None, channel_type=None,
                    chain_direction=None, dtype=Dtype.f32, sites=(SOUT_PUT, SOUT_GET)),
        ChannelPlan(name="West", size=(PI - 1,), broadcast_shape=None, channel_type=None,
                    chain_direction=None, dtype=Dtype.f32, sites=(WEST_PUT, WEST_GET)),
        ChannelPlan(name="WestIn", size=(1,), broadcast_shape=None, channel_type=None,
                    chain_direction=None, dtype=Dtype.f32, sites=(WEST_IN_PUT, WEST_IN_GET)),
    )
    drain = LoopPlan(axis="tx_drain", lo=ZERO, hi=const(PI), step=const(1), kind="unrolled",
                     depth=0, body=(SOUT_GET,))
    return MappingPlan(mapping=mapping, tensors=(halo.tensor(mapping),), launch_name="chain",
                       segment_name="chain_seg", herd=herd, buffers=buffers, channels=channels,
                       segment_body=(WEST_IN_PUT, herd, drain),
                       herd_body=buffers + (HEAD_OR_BODY, WEST_PUT, SOUT_PUT),
                       delivery=(("U", "STATIONARY", None, True),),
                       summary=halo.summary(mapping, herd, buffers, channels))


__all__ = ["HEAD_OR_BODY", "WEST_IN_GET", "WEST_GET", "plan"]
