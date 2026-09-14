"""W3 — Smith-Waterman local alignment, anti-diagonal wavefront. The skew is
the one thing the user must state, and the one thing we check before codegen."""

MQ, NR = 32, 32
PJ = 4
CW = NR // PJ  # 8 columns per PE
MATCH = 2  # substitution score on a match
MISMATCH = -1  # ... and on a mismatch
GAP = 1  # linear gap penalty
PARAMS = {"MQ": MQ, "NR": NR, "PJ": PJ, "CW": CW,
          "MATCH": MATCH, "MISMATCH": MISMATCH, "GAP": GAP}


def sw(q, r, S):
    for i in range(1, MQ + 1):
        for j in range(1, NR + 1):
            sub = MATCH if q[i - 1] == r[j - 1] else MISMATCH
            S[i, j] = max(0, S[i - 1, j - 1] + sub,
                          S[i - 1, j] - GAP, S[i, j - 1] - GAP)


def schedule(target):
    raise NotImplementedError("surface: Person A M1/M2")


def main():
    import numpy as np
    q = np.arange(MQ) % 4
    r = np.arange(NR) % 4
    S = np.zeros((MQ + 1, NR + 1), dtype=np.int64)
    sw(q, r, S)
    print(int(np.sum(S)))
    try:
        print(schedule("npu1"))
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    main()
