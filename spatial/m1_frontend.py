"""M1 — frontend / kernel capture. Owner: Person A. LLD: design/03-lld-M1-frontend.md.

Entry point per design/06-interfaces.md §7.2.
"""

from __future__ import annotations

_NOT_BUILT = "m1_frontend: not built yet; see design/03-lld-M1-frontend.md"


def capture(fn: Callable) -> KernelModel:  # noqa: F821
    """Walk `fn`'s AST into a KernelModel. Raises GrammarError."""
    raise NotImplementedError(_NOT_BUILT)
