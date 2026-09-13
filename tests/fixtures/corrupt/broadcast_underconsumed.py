"""W1 with `A2L1`'s `ty = 1` get removed — the **fan-out** rule, decision D-2.

Built with `dataclasses.replace` from `w1_plan.plan("npu1")`. `A2L1` is `size=(2,1)` with
`broadcast_shape=(2,2)`, so a put at `[pi,0]` must be matched by a get at **each** of
`{[pi,0], [pi,1]}`. Guarding the herd-side get to `ty == 0` leaves the `ty = 1` column of the
grid unserved: the literal upstream P1 (which keys on `(name, indices)`) sees `4 puts, 4 gets`
and is happy; D-2's fan-out rule names the missing index `[0,1]`.

**Substitution, recorded per the brief.** The get is one `ChannelSite` covering all four PE
coordinates, so "the `ty=1` get removed" is spelled as a guard that makes it dead there.
"""

from __future__ import annotations

from dataclasses import replace

from spatial.model import Guard, MappingPlan

from tests.fixtures.mappings import ZERO
from tests.fixtures.plans import w1_plan

GET = replace(w1_plan.A_GET, guard=Guard(coord="ty", relation="==", value=ZERO))
"""`A2L1.get`, alive only on the `ty = 0` column of the herd."""


def plan() -> MappingPlan:
    """W1's plan with half the fan-out unconsumed (`BALANCE`, fan-out, index `[0,1]`)."""
    base = w1_plan.plan("npu1")
    loop = base.herd_body[2]
    body = tuple(GET if node is w1_plan.A_GET else node for node in loop.body)
    channels = tuple(
        replace(c, sites=tuple(GET if s is w1_plan.A_GET else s for s in c.sites))
        if c.name == "A2L1" else c for c in base.channels)
    return replace(base, herd_body=base.herd_body[:2] + (replace(loop, body=body),)
                   + base.herd_body[3:], channels=channels)


__all__ = ["GET", "plan"]
