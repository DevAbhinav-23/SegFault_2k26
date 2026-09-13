"""W1 with `a` allocated above the K loop yet still `ping_pong_candidate` — invariant 4.

Built with `dataclasses.replace` from `w1_plan.plan("npu1")`. `isPingPongCandidate`'s condition 2
is *the alloc is a direct child of the candidate loop* (`AIRDependencyScheduleOpt.cpp:1604`), so
this plan asserts a transform that cannot fire. Nothing the user wrote can cause it, which is why
it is an internal-consistency failure (**B-P23**) rather than a diagnostic with a `fix` that
blames a clause.

`BufferPlan.loop_depth` stays 1 — M0's `I41` requires it of any candidate — so what moves is only
the alloc's *position* in the herd body, which is exactly the condition the check reads.
"""

from __future__ import annotations

from dataclasses import replace

from spatial.model import MappingPlan

from tests.fixtures.plans import w1_plan


def plan() -> MappingPlan:
    """W1's plan with `a`'s alloc hoisted out of the K loop (internal error, invariant 4)."""
    base = w1_plan.plan("npu1")
    loop = base.herd_body[2]
    hoisted = replace(loop, body=tuple(n for n in loop.body if n is not w1_plan.A_TILE))
    return replace(base, herd_body=base.herd_body[:2] + (w1_plan.A_TILE, hoisted)
                   + base.herd_body[3:])


__all__ = ["plan"]
