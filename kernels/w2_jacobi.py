"""W2 — 2-D 5-point Jacobi over T timesteps. The halo is never written by the
user: halo=1 is a width, and the ghost rows are derived from the access map."""

T = 4
H, W = 16, 16  # H, W are the INTERIOR extents; U carries the halo rows
PI = 2
HS = H // PI  # 8 interior rows per PE; PI*HS == H exactly
PARAMS = {"T": T, "H": H, "W": W, "PI": PI, "HS": HS}


def jacobi(U):
    for t in range(T):
        for i in range(1, H + 1):
            for j in range(1, W - 1):
                U[t + 1, i, j] = 0.2 * (U[t, i, j]
                                        + U[t, i - 1, j] + U[t, i + 1, j]
                                        + U[t, i, j - 1] + U[t, i, j + 1])


def schedule(target):
    raise NotImplementedError("surface: Person A M1/M2")


def main():
    import numpy as np
    U = np.zeros((T + 1, H + 2, W))
    U[0, 1:H + 1, 1:W - 1] = 1.0
    jacobi(U)
    print(float(np.sum(U)))
    try:
        print(schedule("npu1"))
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    main()
