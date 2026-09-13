"""W1 with a second `A2L1` put inside the segment K loop — the **per-iteration** balance rule.

Built with `dataclasses.replace` from `w1_plan.plan("npu1")`. The totals alone would not catch
this if the check counted whole bodies: the extra put is inside `air.sequential(0, 64, 16)`, so
it multiplies by the trip count and the key goes from `4 == 4` to `4 != 8`
(`03-lld-M4-mapping.md` §3.7.1, the first of the three things a naive count would miss).
"""

from __future__ import annotations

from dataclasses import replace

from spatial.model import MappingPlan

from tests.fixtures.plans import w1_plan

EXTRA = replace(w1_plan.A_PUT, id="A2L1.put.1@segment", order=1)
"""The duplicate put. A distinct `id`, because a site id names one site plan-wide (`I60`)."""


def plan() -> MappingPlan:
    """W1's plan with `A2L1` put twice per K trip (`BALANCE`, per-iteration, `4 != 8`)."""
    base = w1_plan.plan("npu1")
    fill = base.segment_body[0]
    doubled = replace(fill, body=(replace(fill.body[0], body=(w1_plan.A_PUT, EXTRA)),))
    channels = tuple(replace(c, sites=c.sites + (EXTRA,)) if c.name == "A2L1" else c
                     for c in base.channels)
    return replace(base, segment_body=(doubled,) + base.segment_body[1:], channels=channels)


__all__ = ["EXTRA", "plan"]
