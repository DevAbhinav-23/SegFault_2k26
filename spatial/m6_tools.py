"""M6 — toolchain & runtime driver. Owner: Person C. LLD: design/03-lld-M6-toolchain.md.

Entry points per design/06-interfaces.md §7.2. `air` is imported lazily, inside a function, so
importing this module never requires the toolchain (FR-S20). Any stderr line matching `error:`
is a failure regardless of exit status (FR-T5, design/07-environment.md §4).
"""

from __future__ import annotations

_NOT_BUILT = "m6_tools: not built yet; see design/03-lld-M6-toolchain.md"


def artifact(mlir_path: str, target: Target, output_format: str) -> str:  # noqa: F821
    """Drive aircc; output_format is "none" | "pdi" | "xclbin". Raises ToolchainError."""
    raise NotImplementedError(_NOT_BUILT)


def run(artifact: str, inputs: Sequence[ndarray]) -> list[ndarray]:  # noqa: F821
    """Execute on a device. Raises ToolchainError."""
    raise NotImplementedError(_NOT_BUILT)


def diff(device: Sequence[ndarray], oracle: Sequence[ndarray], tol: float) -> DiffReport:  # noqa: F821
    """Compare device output against the oracle. Raises nothing."""
    raise NotImplementedError(_NOT_BUILT)


def trace(mlir_path: str, model_json: str) -> str:
    """Drive air-runner, a performance model only. Raises ToolchainError."""
    raise NotImplementedError(_NOT_BUILT)
