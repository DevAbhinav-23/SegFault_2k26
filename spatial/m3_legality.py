"""M3 — legality checker. Owner: Person A. LLD: design/03-lld-M3-checker.md.

Entry point per design/06-interfaces.md §7.2.
"""

from __future__ import annotations

_NOT_BUILT = "m3_legality: not built yet; see design/03-lld-M3-checker.md"


def check(kernel: KernelModel, schedule: ScheduleModel) -> LegalMapping:  # noqa: F821
    """Verify (sigma, pi) before any IR exists. Raises LegalityError."""
    raise NotImplementedError(_NOT_BUILT)
