"""W1 with the K loop's induction variable used as a bundle index — `BUNDLE-INDEX-IS-IV`.

Built with `dataclasses.replace` from `w1_plan.plan("npu1")`. `ChannelPutOp::verify` rejects a
bundle index that is an `scf.for` induction variable (`AIRDialect.cpp:3586-3593`); the plan keeps
the stricter form — no reference to a temporal IV at all — so `06-interfaces.md` §5.6 invariant 3
is sound by construction (`03-lld-M4-mapping.md` §3.5, finding N-6).
"""

from __future__ import annotations

from dataclasses import replace

from spatial.model import MappingPlan

from tests.fixtures.mappings import lin
from tests.fixtures.plans import w1_plan

GET = replace(w1_plan.A_GET, indices=(lin("k0"), lin("ty")))
"""`A2L1.get` indexed by `k0`, the `air.sequential` loop it sits inside."""


def plan() -> MappingPlan:
    """W1's plan with `A2L1.get` bundled on `k0` (`BUNDLE-INDEX-IS-IV`, naming the loop)."""
    base = w1_plan.plan("npu1")
    loop = base.herd_body[2]
    body = tuple(GET if node is w1_plan.A_GET else node for node in loop.body)
    channels = tuple(
        replace(c, sites=tuple(GET if s is w1_plan.A_GET else s for s in c.sites))
        if c.name == "A2L1" else c for c in base.channels)
    return replace(base, herd_body=base.herd_body[:2] + (replace(loop, body=body),)
                   + base.herd_body[3:], channels=channels)


__all__ = ["GET", "plan"]
