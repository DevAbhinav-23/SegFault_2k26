"""M4 — plan self-check (P1'/P2b/P3). Owner: Person B. LLD: design/03-lld-M4-mapping.md §3.8.

Entry point per design/06-interfaces.md §7.2.
"""

from __future__ import annotations

_NOT_BUILT = "m4_selfcheck: not built yet; see design/03-lld-M4-mapping.md"


def self_check(plan: MappingPlan) -> None:  # noqa: F821
    """Assert put/get balance and channel-graph acyclicity. Raises MappingError."""
    raise NotImplementedError(_NOT_BUILT)
