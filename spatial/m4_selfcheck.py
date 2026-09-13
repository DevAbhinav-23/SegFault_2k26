"""M4 — plan self-check (P1'/P2b/P3). Owner: Person B. LLD: design/03-lld-M4-mapping.md §3.8.

Entry point per design/06-interfaces.md §7.2.
"""

from __future__ import annotations

from spatial.model import MappingPlan


def self_check(plan: MappingPlan) -> None:
    """Assert put/get balance and channel-graph acyclicity. Raises MappingError.

    **Real implementation lands in P3** (design/03-lld-M4-mapping.md §3.7-§3.8: P1' balance,
    P2b acyclicity, the two structural checks and the per-core DMA-channel budget). Until then
    this is the stub of design/03-lld-M7-tests.md §3.8 — *a real implementation that is wrong
    yet*, at its final import path, with no `if STUB:` branch to ship: **it accepts every plan**.

    `m4.plan` calls it on its own result before returning, so the call site is already in place
    and finishing the function is the whole of the change.
    """
    return None
