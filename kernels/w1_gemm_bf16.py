"""W1-large — smoke/IR-facts shape: bf16 inputs, f32 output, 256^3.

Same triple loop as `w1_gemm`, same clause shape as its `schedule_os`; only the
shapes, the dtypes and the grid differ. No expected/oracle fixture (levels I and S
only), so this module has no golden: the live test asserts only that it checks,
plans, emits and compiles.
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
    """**Not emittable on this toolchain.** M1, M2, M3 and M4 all accept this schedule —
    `check()` returns `l1_bytes == 49152` (a/b doubled at 16 384 each, acc 16 384),
    `physical_herd == (1, 4)` with `repeats == (4, 1)` on npu1, and `plan()` self-checks
    clean. **M5 rejects it at emission**, measured 2026-09-15:

        EmissionError EMIT-AIR-API: air.api rejected the StoreNode of 'gemm_bf16'
          because: air_api_message='dtype mismatch in elementwise assignment:
                   destination is air.api.f32 but operand is air.api.bf16'

    The accumulation `C[i, j] += A[i, k] * B[k, j]` reads `bf16` and stores into an `f32`
    accumulator, and `air.api` has no mixed-precision elementwise assignment — it is not a
    defect in M1/M2/M3/M4 and not one a schedule clause can express around. Making this
    real needs either a widening cast in the kernel subset (an M1 grammar change) or an
    `air.api` that accepts the mixed store; neither is in scope. `w1_large` therefore stays
    an inputs-only fixture at levels I and S, as `03-lld-M8-kernels-demo.md` §4 has it.
    """
    raise NotImplementedError(
        "W1-large is not emittable: M5 raises EMIT-AIR-API on the accumulate, "
        "air.api says 'dtype mismatch in elementwise assignment: destination is "
        "air.api.f32 but operand is air.api.bf16'. M1/M2/M3/M4 accept the schedule "
        "(l1_bytes 49152); see this function's docstring.")


def _schedule_os_clauses(target):
    """The schedule itself, kept because M1-M4 do accept it and the docstring above cites
    their figures. Not public: `schedule_os` is the entry point and it refuses."""
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
    print(_schedule_os_clauses("npu1").check().l1_bytes)
    try:
        schedule_os("npu1")
    except NotImplementedError as exc:
        print(exc)


if __name__ == "__main__":
    main()
