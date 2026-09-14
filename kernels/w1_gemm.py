import spatial as sp

M = N = K = 64

@sp.kernel
def gemm(A: sp.f32[M, K], B: sp.f32[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]