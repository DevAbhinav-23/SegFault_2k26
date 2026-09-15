"""W2 — 2-D 5-point Jacobi over T timesteps. The halo is never written by the
user: halo=1 is a width, and the ghost rows are derived from the access map."""

import spatial as sp

T = 4
H, W = 16, 16  # H, W are the INTERIOR extents; U carries the halo rows
PI = 2
HS = H // PI  # 8 interior rows per PE; PI*HS == H exactly
PARAMS = {"T": T, "H": H, "W": W, "PI": PI, "HS": HS}


@sp.kernel
def jacobi(U: sp.f32[T + 1, H + 2, W]):
    for t in range(0, T):
        for i in range(1, H + 1):
            for j in range(1, W - 1):
                U[t + 1, i, j] = 0.2 * (U[t, i, j]
                                        + U[t, i - 1, j] + U[t, i + 1, j]
                                        + U[t, i, j - 1] + U[t, i, j + 1])


def schedule(target):
    """Halo exchange: one PE row-block each, the {t, t+1} plane pair resident in L1
    (design/03-lld-M2-schedule.md §6.3)."""
    s = sp.schedule(jacobi, target=target)
    ax = s.axes()
    s.grid(PI)
    s.tile(ax.i, HS)
    s.place(px=ax.i0)
    s.sequential(ax.t)
    s.window("U", dims=(ax.i, ax.j), halo=1)
    s.exchange("U", along=ax.i0, halo=1)
    s.reside(U="L1")
    s.double_buffer("U")
    return s


def main():
    import numpy as np
    U = np.zeros((T + 1, H + 2, W))
    U[0, 1:H + 1, 1:W - 1] = 1.0
    jacobi(U)
    print(float(np.sum(U)))
    print(schedule("npu1").summary())


if __name__ == "__main__":
    main()
