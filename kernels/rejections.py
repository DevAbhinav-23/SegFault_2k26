"""The three demo rejections (design/03-lld-M8-kernels-demo.md §3.5).

Each is a two-line edit of a working schedule, so the audience sees what changed. Each
**raises** `LegalityError` from its own `s.check()` — before any IR exists, which is the
entry's point; the `return s.check()` never returns. The docstrings carry the **rendered**
message (`str(exc)`), measured 2026-09-15, not a predicted shape; the same text is captured
byte for byte in `tests/golden/reject.<CODE>.txt`.
"""

import spatial as sp

from kernels import w1_gemm as w1
from kernels import w3_sw as w3

M192 = 192
TM192 = TN192 = 96
TK192 = 32
"""`bad_capacity`'s own shape (§3.5): 192^3 f32, 96x96 tiles, TK = 32."""


@sp.kernel
def gemm_192(A: sp.f32[M192, M192], B: sp.f32[M192, M192], C: sp.f32[M192, M192]):
    for i in range(M192):
        for j in range(M192):
            for k in range(M192):
                C[i, j] += A[i, k] * B[k, j]


def bad_skew(target):
    """L2-CAUSALITY. 'Time is just the row index' — the naive schedule, and the one that
    silently computes garbage in every other system. One line differs from
    `kernels.w3_sw.schedule`: `skew(time=(ax.i,))` instead of `skew(time=(ax.i, ax.j0))`.

    Rendered message, measured::

        L2-CAUSALITY: dependence (0, 1) on S is not causal: sigma.d = (0, -7), whose
        first nonzero entry is <= 0
          in clause: skew(...)
          because:   dependence=(0, 1), operand='S', representative=(0, 1, -7),
                     sigma_d=(0, -7)
          fix:       add an axis to skew(time=...) before the violating one, or drop the
                     offending tile
    """
    s = sp.schedule(w3.sw, target=target)
    ax = s.axes()
    s.grid(w3.PJ)
    s.tile(ax.j, w3.CW)
    s.place(px=ax.j0)
    s.skew(time=(ax.i,))                          # was  skew(time=(ax.i, ax.j0))
    s.forward("S", along=ax.j0, dir="W->E")
    s.reside(S="L1")
    return s.check()


def bad_stationary(target):
    """STATIONARITY. Place i and k, then claim C is stationary. Two lines differ from
    `kernels.w1_gemm.schedule_os`: the grid is `(PI, PK)` and `py` places `ax.k0`.

    Rendered message, measured::

        STATIONARITY: ker M_C = ((0, 0, 1),) is not contained in ker Spi = ((0, 1, 0),)
          in clause: stationary('C')
          because:   ker_M=((0, 0, 1),), ker_pi=((0, 1, 0),), operand='C',
                     placed=('i0', 'k0')
          fix:       place the axes that would kill the reuse direction
    """
    s = sp.schedule(w1.gemm, target=target)
    ax = s.axes()
    s.grid(w1.PI, w1.PK)
    s.tile(ax.i, w1.TM); s.tile(ax.j, w1.TN); s.tile(ax.k, w1.TK)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.k0)                   # was  place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    return s.check()


def bad_capacity(target):
    """L1-CAPACITY. Fits at 61 440 B — until `double_buffer` doubles A and B.

    `acc [96, 96] f32` = 36 864, `a [96, 32] f32` = 12 288, `b [32, 96] f32` = 12 288:
    61 440 undoubled, **86 016** once FR-L9 charges the ping-ponged figure for `a` and `b`.
    §3.5's worked table and the checker agree on 86 016 (measured 2026-09-15), so the
    docstring's old figure needed no correction.

    Rendered message, measured 2026-09-15 **after** FR-L9's per-buffer breakdown landed
    (`03-lld-M3-checker.md` §3.10 line 12); the `because:` line is one line in the rendered
    text and is wrapped here only to fit the margin::

        L1-CAPACITY: the per-core L1 working set is 86016 bytes, over the 65536-byte budget
          in clause: tile(...)/double_buffer(...)
          because:   budget=65536, doubled=('A', 'B'), per_buffer=(('A', (96, 32), 'f32',
                     24576), ('B', (32, 96), 'f32', 24576), ('C', (96, 96), 'f32', 36864)),
                     total=86016
          fix:       halve a tile factor, or drop a double_buffer(...) entry
    """
    s = sp.schedule(gemm_192, target=target)
    ax = s.axes()
    s.grid(2, 2)
    s.tile(ax.i, TM192); s.tile(ax.j, TN192); s.tile(ax.k, TK192)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")                     # <- this line is what tips it over
    return s.check()


REJECTIONS = {"L2-CAUSALITY": bad_skew, "STATIONARITY": bad_stationary,
              "L1-CAPACITY": bad_capacity}
"""Error code → the builder whose `check()` raises it. Read by the demo and by
`tests/integration/test_oracle.py::test_demo_rejections`."""
