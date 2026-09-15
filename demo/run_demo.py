"""Five-minute pitch driver (M8 §5.1). Off-device only: CPython kernels + this slide.

Every beat now runs for real — the surface (A: M1/M2), the checker (A: M3) and the emitter
(B: M4/M5) are all built. The NOT-BUILT branch in `beat` stays as the net: a beat that
raises must cost one printed line, never a stack trace on stage.

Two beats degrade on the **environment** rather than on the code, and both say so in one line
instead of claiming a result they did not get: 3:40's `aircc` verdict needs
`source scripts/airenv.sh` for `aircc` to be on `PATH`, and 4:05's Tenstorrent run needs
`.venv-tt` and `vendor/tt/libttsim_wh.so` (`source scripts/tt_env.sh`). 4:05's *emission* half
always runs — `spatial.m5tt_emit` is standard library only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kernels import w1_gemm, w2_jacobi, w3_sw  # noqa: E402

_TTNN_LOG = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d[.,]\d+ \| ")
"""One `ttnn` bring-up log line on the 4:05 subprocess's stdout, e.g.
`2026-09-15 18:30:53.040 | info     |           Metal | Disabling multi-erisc mode …`."""

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
    beat("4:05", "Second backend — Tenstorrent Wormhole on ttsim, same plan", _tenstorrent)
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


def _tenstorrent() -> None:
    """Beat 4:05. The same W3 plan through the **second** emitter, then executed on ttsim.

    Two halves, and only the second one needs anything installed. The first emits the
    TT-Metalium program in this interpreter — `spatial.m5tt_emit` imports nothing but the
    standard library and `spatial.model`, so it runs in `.venv` — and prints what came out of
    the plan the 2:15 and 3:40 beats have already shown. The second spawns `.venv-tt`, the only
    interpreter that can import `ttnn`, and runs W3 for real against the CPython kernel.

    `demo/tt_w3_on_ttsim.py` is the script; the environment it needs is `scripts/tt_env.sh`'s
    three facts, passed here explicitly rather than by sourcing anything. When the venv or the
    simulator is missing the beat says which, and where the suite that proves it lives, instead
    of claiming a run it did not get.
    """
    import os
    import subprocess

    from kernels import w3_sw
    from spatial import m5tt_emit

    program = m5tt_emit.emit(w3_sw.schedule("npu1").plan())
    cores = len(program.runtime_args)
    tensors = ", ".join(f"{t.name}{list(t.shape)} {t.dtype.value}" for t in program.io_tensors)
    print(f"  the same MappingPlan, second emitter: {cores} Tensix cores "
          f"{program.core_range[0]}..{program.core_range[1]}, 1 data-movement kernel of "
          f"{len(program.source.splitlines())} lines of C++ on every core")
    print(f"  {len(program.semaphores)} semaphores per core "
          f"({', '.join(s.name for s in program.semaphores)}), "
          f"{len(program.cbs)} circular buffers, io tensors: {tensors}")

    python = REPO / ".venv-tt" / "bin" / "python"
    simulator = REPO / "vendor" / "tt" / "libttsim_wh.so"
    missing = [str(path) for path in (python, simulator) if not path.exists()]
    if missing:
        print(f"  not executed here: {', '.join(missing)} is missing — `source "
              f"scripts/tt_env.sh` builds the venv from vendor/tt/. The four workloads that "
              f"do execute, exactly and with negative controls, are "
              f"design/PROGRESS-TT.md §T5.3")
        return

    environment = {key: value for key, value in os.environ.items() if key != "TT_METAL_HOME"}
    environment.update(TT_METAL_SIMULATOR=str(simulator), TT_METAL_SLOW_DISPATCH_MODE="1",
                       PYTHONPATH=str(REPO))
    done = subprocess.run([str(python), str(REPO / "demo" / "tt_w3_on_ttsim.py")],
                          env=environment, capture_output=True, text=True, timeout=120,
                          check=False)
    # `ttnn` logs its device bring-up to **stdout**, one `<timestamp> | <level> | …` line per
    # step — thirty of them, none of them ours. They are dropped here and nowhere else: the
    # child's exit status is relayed below, and a failing run prints its stderr tail.
    for line in done.stdout.splitlines():
        if not _TTNN_LOG.match(line):
            print(f"  {line}")
    if done.returncode != 0:
        print(f"  the ttsim run exited {done.returncode}; its last words were:")
        for line in done.stderr.strip().splitlines()[-3:]:
            print(f"    {line}")


def _limits() -> None:
    print((REPO / "demo" / "honest_limits.md").read_text(encoding="utf-8"))


def _w2_w3_oracles() -> None:
    w2_jacobi.main()
    w3_sw.main()


if __name__ == "__main__":
    main()
