"""Five-minute pitch driver (M8 §5.1). Off-device only: CPython kernels + this slide.

Beats needing the schedule surface (A: M1/M2) or the checker (A: M3) print
their script line and a NOT-BUILT note instead of running — the demo degrades,
it never crashes. Surface beats activate when A's modules land; nothing here
changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kernels import w1_gemm, w2_jacobi, w3_sw  # noqa: E402


def beat(clock: str, title: str, fn) -> None:
    """Run one beat. Nothing a beat can raise is allowed to end the pitch."""
    print(f"[{clock}] {title}")
    try:
        fn()
    except NotImplementedError as exc:
        print(f"  NOT BUILT YET: {exc}")
    except Exception as exc:  # noqa: BLE001 — a stack trace on stage is worse than a line
        print(f"  BEAT FAILED — {type(exc).__name__}: {exc}")
    print()


def main() -> None:
    beat("0:00", "The claim — the kernel IS the spec (CPython, no toolchain)", w1_gemm.main)
    beat("0:40", "W1 summary (needs A surface)", lambda: w1_gemm.schedule_os("npu1"))
    beat("1:35", "The flip (needs A surface)", lambda: w1_gemm.schedule_ws("npu1"))
    beat("2:15", "Rejection before codegen (needs A checker)", _rejection)
    beat("3:40", "It lowers — aircc off-device (needs B text + aircc)", _lowers)
    beat("4:35", "Honest limits", _limits)
    _w2_w3_oracles()


def _rejection() -> None:
    """Beat 2:15. A rejection is this beat's **success**, so the diagnostic is the output.

    Until A's checker lands `bad_skew` raises `NotImplementedError` and `beat` prints the
    not-built note. Once it lands it raises `SpatialError`, which is the thing the pitch is
    there to show — printed here rather than left to `beat`'s failure branch, which would
    label the demo's headline moment "BEAT FAILED".
    """
    from kernels import rejections

    try:
        from spatial.model import SpatialError
    except ImportError:                       # pragma: no cover - M0 is always present
        SpatialError = ()                     # type: ignore[assignment]
    try:
        rejections.bad_skew("npu1")
    except SpatialError as exc:               # type: ignore[misc]
        print(f"  REJECTED BEFORE CODEGEN, as intended:\n{exc}")
        return
    print("  NOT REJECTED — the checker accepted a schedule it should have refused")


def _lowers() -> None:
    print("  aircc --device npu1 --output-format=none: see `pytest -m slow` (level S)")


def _limits() -> None:
    print((REPO / "demo" / "honest_limits.md").read_text(encoding="utf-8"))


def _w2_w3_oracles() -> None:
    w2_jacobi.main()
    w3_sw.main()


if __name__ == "__main__":
    main()
