"""W1-large — smoke/IR-facts shape: bf16 inputs, f32 output, 256^3.

Same triple loop as `w1_gemm`, same clause shape as its `schedule_os`; only the
shapes, the dtypes and the grid differ. No expected/oracle fixture (levels I and S
only), so this module has no golden: the live test asserts that it checks, plans and
emits — with the `bf16`→`f32` widening cast — and that `aircc`'s refusal is about the
shape rather than the dtype (`schedule_os`'s docstring has the two diagnostics).
"""

import spatial as sp

M, N, K = 256, 256, 256
TM, TN, TK = 64, 64, 64
PI, PJ = 4, 4
PARAMS_LARGE = {"M": M, "N": N, "K": K, "TM": TM, "TN": TN, "TK": TK,
                "PI": PI, "PJ": PJ}


@sp.kernel
def gemm_bf16(A: sp.bf16[M, K], B: sp.bf16[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]


def schedule_os(target):
    """Output-stationary, `bf16` in and `f32` out — **emittable since B-P32, 2026-09-15**.

    M1, M2, M3 and M4 accept it (`check()` gives `l1_bytes == 49152` — a/b doubled at 16 384
    each, acc 16 384 — with `physical_herd == (1, 4)`, `repeats == (4, 1)` on npu1 and
    `(2, 4)` / `(2, 1)` on npu2), and **M5 now emits it**: each `bf16` load is widened to the
    destination's `f32` with `air.api.ops.cast` before the arithmetic, so the accumulate is
    `f32` throughout. Measured 2026-09-15: 200 lines, two `arith.extf`, no `truncf`, on both
    generations. Until B-P32 this raised `NotImplementedError` because `air.api` refused the
    mixed store (*"dtype mismatch in elementwise assignment: destination is air.api.f32 but
    operand is air.api.bf16"*).

    **`aircc` still does not compile it, and the reason is the shape, not the dtype.** At
    this fixed 256³ / 4×4 / `TM = TN = TK = 64` shape (`03-lld-M8-kernels-demo.md` §4, VF §E.5)
    `aircc --output-format=none` exits 1 on both targets, measured 2026-09-15:

    * npu1 — `air-to-aie`: `'air.channel.get' op failed to get MM2S tile for L3 allocation`;
    * npu2 — `aiecc`: `'aie.tile' op allocated buffers exceeded available memory`, the lowered
      module asking 65 536 B of a 64 KB tile.

    The control that says it is not the cast is in
    `tests/integration/test_kernels_live.py::test_live_bf16_gemm_is_refused_by_aircc_for_its_shape`:
    the same clauses over an **`f32`** kernel at the same grid and the same 49 152 B of L1
    (`TK = 32`) emit **zero** casts and draw the **same two diagnostics**. `w1_large` therefore
    stays an inputs-only fixture at levels I and S, and this schedule is real up to `.mlir()`.
    """
    return _schedule_os_clauses(target)


def _schedule_os_clauses(target):
    """The clauses themselves. Not public: `schedule_os` is the documented entry point."""
    s = sp.schedule(gemm_bf16, target=target)
    ax = s.axes()
    s.grid(PI, PJ)
    s.tile(ax.i, TM); s.tile(ax.j, TN); s.tile(ax.k, TK)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")
    s.pipeline(ax.k0)
    return s


def main():
    s = schedule_os("npu1")
    print(s.check().l1_bytes)
    print(s.mlir().count("arith.extf"), "bf16 loads widened to the f32 accumulator")


if __name__ == "__main__":
    main()
