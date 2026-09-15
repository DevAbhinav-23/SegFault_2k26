"""W3 — Smith-Waterman local alignment, anti-diagonal wavefront. The skew is
the one thing the user must state, and the one thing we check before codegen."""

import spatial as sp

MQ, NR = 32, 32
PJ = 4
CW = NR // PJ  # 8 columns per PE
MATCH = 2  # substitution score on a match
MISMATCH = -1  # ... and on a mismatch
GAP = 1  # linear gap penalty
PARAMS = {"MQ": MQ, "NR": NR, "PJ": PJ, "CW": CW,
          "MATCH": MATCH, "MISMATCH": MISMATCH, "GAP": GAP}


@sp.kernel
def sw(q: sp.i32[MQ], r: sp.i32[NR], S: sp.i32[MQ + 1, NR + 1]):
    for i in range(1, MQ + 1):
        for j in range(1, NR + 1):
            sub = MATCH if q[i - 1] == r[j - 1] else MISMATCH
            S[i, j] = max(0, S[i - 1, j - 1] + sub,
                          S[i - 1, j] - GAP, S[i, j - 1] - GAP)


def schedule(target):
    """Wavefront: time is `i + j0`, S forwarded west to east along the placed axis
    (design/03-lld-M2-schedule.md §6.4). Dropping `ax.j0` from the skew is the demo's
    headline rejection — `kernels.rejections.bad_skew`."""
    s = sp.schedule(sw, target=target)
    ax = s.axes()
    s.grid(PJ)
    s.tile(ax.j, CW)
    s.place(px=ax.j0)
    s.skew(time=(ax.i, ax.j0))
    s.forward("S", along=ax.j0, dir="W->E")
    s.reside(S="L1")
    return s


def main():
    import numpy as np
    q = np.arange(MQ) % 4
    r = np.arange(NR) % 4
    S = np.zeros((MQ + 1, NR + 1), dtype=np.int64)
    sw(q, r, S)
    print(int(np.sum(S)))
    print(schedule("npu1").summary())


if __name__ == "__main__":
    main()
