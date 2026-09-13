"""W1 with part of the `C2L3` drain removed — the P1' balance check's simplest failure.

Built with `dataclasses.replace` from `w1_plan.plan("npu1")`: nothing else about the plan moves.

**Substitution, recorded per the brief.** `04-test-plan.md` §2 asks for *one* dropped `get`, but
W1's four `C2L3` gets are four trips of one `ChannelSite` under the unrolled `j_drain` nest, so
the nearest constructible corruption is to shorten that loop: `j_drain` runs once, column 1 of
the PE grid is never drained, and the keys `[0,1]` and `[1,1]` each show `1 put, 0 gets`.
"""

from __future__ import annotations

from dataclasses import replace

from spatial.model import MappingPlan

from tests.fixtures.mappings import const
from tests.fixtures.plans import w1_plan


def plan() -> MappingPlan:
    """W1's plan with the `j_drain` loop shortened to one trip (`BALANCE`, `1 != 0`)."""
    base = w1_plan.plan("npu1")
    outer = base.segment_body[3]
    shortened = replace(outer, body=(replace(outer.body[0], hi=const(1)),))
    return replace(base, segment_body=base.segment_body[:3] + (shortened,))


__all__ = ["plan"]
