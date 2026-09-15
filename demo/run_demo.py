"""Five-minute pitch driver (M8 §5.1). Off-device only: CPython kernels + this slide.

Every beat now runs for real — the surface (A: M1/M2), the checker (A: M3) and the emitter
(B: M4/M5) are all built. The NOT-BUILT branch in `beat` stays as the net: a beat that
raises must cost one printed line, never a stack trace on stage.

The only beat that still degrades is 3:40's `aircc` verdict, and it degrades on the
**environment**, not on the code: without `source scripts/airenv.sh` there is no `aircc` on
`PATH`, so it prints where to find that check instead of pretending to have run it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kernels import w1_gemm, w2_jacobi, w3_sw  # noqa: E402

_IR_OPS = ("air.herd", "air.channel.put", "air.channel.get", "air.channel")
"""Counted per target at beat 3:40. `air.channel` is counted last and net of the two
put/get spellings, because every `air.channel.put` line also contains `air.channel`."""


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
    beat("0:40", "W1 output-stationary — the mapping summary", _summary_os)
    beat("1:35", "The flip — weight-stationary, same kernel text", _summary_ws)
    beat("2:15", "Rejection before codegen", _rejection)
    beat("3:40", "It lowers — npu1 and npu2 from one schedule", _lowers)
    beat("4:35", "Honest limits", _limits)
    _w2_w3_oracles()


def _summary_os() -> None:
    """Beat 0:40. The rendered summary of W1 output-stationary on npu1."""
    print(w1_gemm.schedule_os("npu1").summary())


def _summary_ws() -> None:
    """Beat 1:35. The same, weight-stationary — and the line that says why it matters."""
    print(w1_gemm.schedule_ws("npu1").summary())
    print("  ^ same kernel text as 0:40, byte for byte: only the schedule clauses differ "
          "(grid(4), place(px=ax.k0), stationary(\"B\")). The kernel was never edited.")


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
    """Beat 3:40. Emit W1 for both AIE generations, count the ops, then compile if we can.

    `aircc` is the expensive half and is the half that needs `scripts/airenv.sh` on `PATH`.
    Without it the beat says where the check lives rather than claiming a verdict it did not
    get — `tool_available` is a `which`, not a run, so the branch costs nothing.
    """
    import tempfile

    from spatial import m6_tools as m6

    texts = {}
    for target in ("npu1", "npu2"):
        text = w1_gemm.schedule_os(target).mlir()
        texts[target] = text
        counts = []
        rest = text
        for op in _IR_OPS:
            n = rest.count(op)
            counts.append(f"{op} {n}")
            rest = rest.replace(op, "")
        print(f"  {target}: {len(text.splitlines())} lines, " + ", ".join(counts))

    if not m6.tool_available("aircc"):
        print("  aircc is not on PATH: `source scripts/airenv.sh .venv/bin/python`, then "
              "`.venv/bin/python -m pytest -m slow` runs the same check (level S)")
        return
    with tempfile.TemporaryDirectory(prefix="segfault-demo-") as work:
        for target, text in texts.items():
            path = Path(work) / f"w1.{target}.mlir"
            path.write_text(text, encoding="utf-8")
            m6.artifact(str(path), target, "none", workdir=work)
            print(f"  aircc --device {target} --output-format=none: OK "
                  f"(exit 0, no `error:` on stderr — FR-T5 reads stderr first)")


def _limits() -> None:
    print((REPO / "demo" / "honest_limits.md").read_text(encoding="utf-8"))


def _w2_w3_oracles() -> None:
    w2_jacobi.main()
    w3_sw.main()


if __name__ == "__main__":
    main()
