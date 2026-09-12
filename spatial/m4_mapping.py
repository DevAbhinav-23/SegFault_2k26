"""M4 — mapping / protocol synthesis. Owner: Person B. LLD: design/03-lld-M4-mapping.md.

Entry point per design/06-interfaces.md §7.2.
"""

from __future__ import annotations

_NOT_BUILT = "m4_mapping: not built yet; see design/03-lld-M4-mapping.md"


def plan(mapping: LegalMapping) -> MappingPlan:  # noqa: F821
    """Derive the reuse trichotomy, buffers, channels, loops. Raises MappingError."""
    raise NotImplementedError(_NOT_BUILT)
