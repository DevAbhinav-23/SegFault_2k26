"""M6 — toolchain & runtime driver. Owner: Person C. LLD: design/03-lld-M6-toolchain.md.

Entry points per design/06-interfaces.md §7.2. `air` is imported lazily, inside a function, so
importing this module never requires the toolchain (FR-S20). Any stderr line matching `error:`
is a failure regardless of exit status (FR-T5, design/07-environment.md §4).
"""

from __future__ import annotations

from spatial.model import Diagnostic, ToolchainError

_NOT_BUILT = "m6_tools: not built yet; see design/03-lld-M6-toolchain.md"

PIN = {
    "mlir_air": "0.0.1.2026091204+ff95a9b",
    "mlir_aie": "1.4.3.dev55+g10767b5",
    "llvm_aie": "22.0.0.2026091201+386ca5c6",
}
"""The pinned wheel versions (FR-T6, design/03-lld-M6-toolchain.md §3.9)."""


def check_pin() -> None:
    """Raise `ToolchainError(TOOL-VERSION-PIN)` unless the installed wheels match `PIN`."""
    from importlib import metadata

    installed = {}
    for name in PIN:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    mismatched = {n: [PIN[n], installed[n]] for n in PIN if installed[n] != PIN[n]}
    if not mismatched:
        return
    raise ToolchainError(Diagnostic(
        code="TOOL-VERSION-PIN",
        stage="toolchain",
        clause="build()",
        reason="the installed toolchain is not the pinned one; "
               "every golden file is valid only for the pin",
        fix="pip install --no-index --find-links vendor/wheels 'mlir_air[aie]'",
        location=None,
        details={"expected": PIN, "installed": installed, "mismatched": mismatched},
    ))


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
