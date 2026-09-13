"""W1 — dense GEMM. The kernel is the specification: delete every s.* line
below and this file still runs in CPython and still computes C = A @ B."""

M, N, K = 64, 64, 64
TM, TN, TK = 32, 32, 16
PI, PJ = 2, 2
PK = K // TK  # 4 — the flip's grid
PARAMS = {"M": M, "N": N, "K": K, "TM": TM, "TN": TN, "TK": TK,
          "PI": PI, "PJ": PJ, "PK": PK}


def gemm(A, B, C):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]


def schedule_os(target):
    raise NotImplementedError("surface: Person A M1/M2")


def schedule_ws(target):
    raise NotImplementedError("surface: Person A M1/M2")


def main():
    import numpy as np
    A = np.arange(M * K, dtype=np.float64).reshape(M, K) % 5
    B = np.arange(K * N, dtype=np.float64).reshape(K, N) % 5
    C = np.zeros((M, N))
    gemm(A, B, C)
    print(float(np.sum(C)))


if __name__ == "__main__":
    main()
