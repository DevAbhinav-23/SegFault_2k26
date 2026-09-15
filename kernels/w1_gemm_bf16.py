"""W1-large — smoke/IR-facts shape: bf16 inputs, f32 output, 256^3.

Same triple loop as `w1_gemm`, same clause shape as its `schedule_os`; only the
shapes, the dtypes and the grid differ. No expected/oracle fixture (levels I and S
only), so this module has no golden: the live tests assert that it checks, plans and
emits — with the `bf16`→`f32` widening cast — and that `aircc`'s refusal is about the
**grid** rather than the dtype. Two controls say so, both in
`tests/integration/test_kernels_live.py`: a cast-free f32 kernel at this grid draws
the *same* two diagnostics, and the same bf16 cast at `grid(1, 2)` compiles to exit 0
on both generations.
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
    `(2, 4)` / `(2, 1)` on npu2), and **M5 emits it**: each `bf16` load is widened to the
    destination's `f32` with `air.api.ops.cast` before the arithmetic, so the accumulate is
    `f32` throughout. Measured 2026-09-15: 200 lines, two `arith.extf`, no `truncf`, on both
    generations.

    **`aircc` does not compile it, and the reason is the grid, not the dtype.** At this fixed
    256³ / 4×4 / `TM = TN = TK = 64` shape (`03-lld-M8-kernels-demo.md` §4, VF §E.5) the
    logical 4×4 grid is larger than either physical herd, so M4 emits a **repeat loop**, and
    both refusals follow from it (measured 2026-09-15, `design/PROGRESS-B.md` B-P33/B-P34):

    * **npu1**, `repeats == (4, 1)` — `air-to-aie`: *"'air.channel.get' op failed to get MM2S
      tile for L3 allocation"* (`AIRToAIESchedulingUtils.cpp:3892-3894`). The herd-side `C2L3`
      put indexes its bundle with the repeat loop's induction variable, so the bundle position
      never resolves to a producer.
    * **npu2**, `repeats == (2, 1)` — `aiecc`: *"'aie.tile' op allocated buffers exceeded
      available memory"*. The repeat loop gives each core two accumulators:
      2 × 16 384 + 4 × 8 192 = 65 536 B of buffers plus the 2 048 B core stack, in a 65 536 B
      tile.

    Neither number is `l1_bytes`, which is 49 152 here **and** at the passing control — so this
    schedule's refusal is not something M3's L1 check can see. The two controls are in
    `tests/integration/test_kernels_live.py`:
    `test_live_bf16_gemm_is_refused_by_aircc_for_its_shape` runs a cast-free **f32** kernel at
    the same grid and the same 49 152 B and gets the *same two diagnostics*; and
    `test_live_bf16_gemm_compiles_when_the_grid_fits_the_herd` runs **this cast** at
    `grid(1, 2)` — one parameter moved, `repeats == (1, 1)` — and `aircc` exits 0 on npu1 and
    npu2. `w1_large` therefore stays an inputs-only fixture at levels I and S, and this
    schedule is real up to `.mlir()`.
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
