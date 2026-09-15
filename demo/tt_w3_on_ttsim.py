"""The pitch's second-backend beat, executed: W3 on Tenstorrent's functional simulator.

Run **under `.venv-tt`**, which is the only interpreter that can import `ttnn` (the wheel pins
`numpy<2` against the project's `numpy==2.5.3`). `demo/run_demo.py` spawns it as a subprocess
with the three environment facts `scripts/tt_env.sh` exports; by hand it is::

    source scripts/tt_env.sh
    .venv-tt/bin/python demo/tt_w3_on_ttsim.py

Nothing here is a second code path. It is the same `schedule()` the earlier beats print, the
same `MappingPlan` the AIR emitter consumes, `spatial.m5tt_emit` for the TT-Metalium program and
`spatial.m6tt_run` for the execution — and the oracle it compares against is
`kernels.w3_sw.sw` **run in CPython**, the kernel as it is written, which is the project's whole
claim in one line. The suite that proves it four times over, with negative controls, is
`pytest -m requires_ttsim tests/tt` (`design/PROGRESS-TT.md` §T5.3).

ttsim is a **functional** simulator of one Wormhole B0 part. No Tenstorrent hardware is touched,
nothing here is a timing claim, and Tenstorrent is not an AIR target.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np                                                       # noqa: E402

from kernels import w3_sw                                                # noqa: E402
from spatial import m5tt_emit, m6tt_run                                  # noqa: E402


def inputs() -> dict[str, np.ndarray]:
    """`kernels.w3_sw.main`'s own sequences, at the `i32` the plan declares."""
    return {"q": (np.arange(w3_sw.MQ) % 4).astype(np.int32),
            "r": (np.arange(w3_sw.NR) % 4).astype(np.int32),
            "S": np.zeros((w3_sw.MQ + 1, w3_sw.NR + 1), dtype=np.int32)}


def oracle(given: dict[str, np.ndarray]) -> np.ndarray:
    """The kernel itself, in CPython, on a fresh score matrix. Delete every `s.*` clause and
    this is what is left — FR-S18, and the reason the comparison below is exact."""
    scores = np.zeros_like(given["S"])
    w3_sw.sw(given["q"], given["r"], scores)
    return scores


def main() -> int:
    program = m5tt_emit.emit(w3_sw.schedule("npu1").plan())
    given = inputs()
    start = time.time()
    result = m6tt_run.run(program, given)
    wall = time.time() - start

    expected = oracle(given)
    cells = expected.size
    if np.array_equal(result["S"], expected):
        print(f"W3 on ttsim: EXACT ({cells} cells compared), {wall:.1f}s on "
              f"{program.grid[0]} Tensix cores")
        if not expected[1:, 1:].any():
            print("  but the score matrix is all zeros, so the comparison is vacuous")
            return 1
        return 0
    wrong = int(np.count_nonzero(result["S"] != expected))
    print(f"W3 on ttsim: MISMATCH — {wrong} of {cells} cells differ, max |Δ| "
          f"{int(np.abs(result['S'] - expected).max())}, {wall:.1f}s")
    return 1


if __name__ == "__main__":
    sys.exit(main())
