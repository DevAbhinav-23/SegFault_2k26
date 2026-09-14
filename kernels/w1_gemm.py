"""W1 — dense GEMM. The kernel is the specification: delete every s.* line
below and this file still runs in CPython and still computes C = A @ B."""

import spatial as sp

M, N, K = 64, 64, 64
TM, TN, TK = 32, 32, 16
PI, PJ = 2, 2
PK = K // TK  # 4 -- the flip's grid
PARAMS = {"M": M, "N": N, "K": K, "TM": TM, "TN": TN, "TK": TK,
          "PI": PI, "PJ": PJ, "PK": PK}


@sp.kernel
def gemm(A: sp.f32[M, K], B: sp.f32[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]


def schedule_os(target):
    """Output-stationary: C resident, stream A/B (design/03-lld-M2-schedule.md §6.1)."""
    s = sp.schedule(gemm, target=target)
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


def schedule_ws(target):
    """Weight-stationary flip: B resident, cascade reduction along k, no tile on j
    (design/03-lld-M2-schedule.md §6.2, RULING 9)."""
    s = sp.schedule(gemm, target=target)
    ax = s.axes()
    s.grid(PK)
    s.tile(ax.i, TM); s.tile(ax.k, TK)
    s.reduce(ax.k, op="+")
    s.place(px=ax.k0)
    s.stationary("B")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A")
    return s


def main():
    import numpy as np
    A = np.arange(M * K, dtype=np.float64).reshape(M, K) % 5
    B = np.arange(K * N, dtype=np.float64).reshape(K, N) % 5
    C = np.zeros((M, N))
    gemm(A, B, C)
    print(float(np.sum(C)))


if __name__ == "__main__":
    main()