"""The synthetic halo emitted `[GET, GET, PUT, PUT]` — the P2b cycle of §3.7.2's worked example.

`halo.plan(PI=2, reversed_order=True)`. With the specified order
`PUT(north) → PUT(south) → GET(north ghost) → GET(south ghost)` every outgoing edge of a get is a
program-order edge to later compute, so no path leads back to a put. Reversed, one timestep
closes the cycle

    GET_n(p) → PUT_s(p) → GET_s(p+1) → PUT_n(p+1) → GET_n(p)

— two program-order edges and two channel edges, which is `details["cycle"]`. The plan is still
*balanced*, which is the point: P1' cannot see this and P2b is why it exists.
"""

from __future__ import annotations

from spatial.model import MappingPlan

from tests.fixtures.corrupt import halo


def plan() -> MappingPlan:
    """The 2-PE halo with its gets before its puts (`CHANNEL-CYCLE`, four sites)."""
    return halo.plan(PI=2, reversed_order=True)


__all__ = ["plan"]
