"""W1 with the written `C` declared before the read-only `A` — invariant 6.

Built with `dataclasses.replace` from `w1_plan.plan("npu1")`. `air.api`'s `_check_interface`
raises a bare `RuntimeError`, *output tensors must be declared after all input tensors*
(`python/air/api/_compile.py:226-240`, measured REVIEW-round1 P-R4), so without this check the
violation would surface as an `EMIT-AIR-API` at trace time with an unhelpful message. W3's
`(q, r, S)` is the case that bites in practice; W1 is the case that is one `replace` away.
"""

from __future__ import annotations

from dataclasses import replace

from spatial.model import MappingPlan

from tests.fixtures.plans import w1_plan


def plan() -> MappingPlan:
    """W1's plan with `tensors == (C, A, B)` (internal error, invariant 6)."""
    base = w1_plan.plan("npu1")
    by_name = {tensor.name: tensor for tensor in base.tensors}
    return replace(base, tensors=(by_name["C"], by_name["A"], by_name["B"]))


__all__ = ["plan"]
