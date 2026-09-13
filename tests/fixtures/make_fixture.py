"""Fixture generator (design/03-lld-M8-kernels-demo.md §4.1). Fixed seeds, default_rng only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))

from kernels.w1_gemm import gemm  # noqa: E402
from kernels.w2_jacobi import jacobi  # noqa: E402
from kernels.w3_sw import MATCH, MISMATCH, GAP, sw  # noqa: E402


def _write(name, meta, arrays=None, only_meta=False):
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    if only_meta:
        for f in ("inputs.npz", "expected.npz", "oracle.npz"):
            p = d / f
            if p.exists():
                p.unlink()
        return
    np.savez(d / "inputs.npz", **arrays["inputs"])
    if "expected" in arrays:
        np.savez(d / "expected.npz", **arrays["expected"])
    if "oracle" in arrays:
        np.savez(d / "oracle.npz", **arrays["oracle"])


def _w1_arrays(seed=0, M=64, N=64, K=64):
    rng = np.random.default_rng(seed)
    A = rng.integers(-8, 8, (M, K)).astype(np.float32)
    B = rng.integers(-8, 8, (K, N)).astype(np.float32)
    C = np.zeros((M, N), np.float32)
    expected = {"C": (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)}
    Cc = C.copy()
    gemm(A, B, Cc)
    return {"inputs": {"A": A, "B": B, "C": C}, "expected": expected, "oracle": {"C": Cc}}


def _w2_arrays(seed=1, T=4, H=16, W=16):
    from kernels import w2_jacobi as w2mod
    rng = np.random.default_rng(seed)
    U = np.zeros((T + 1, H + 2, W), np.float32)
    U[0, 1:H + 1, 1:W - 1] = rng.integers(-8, 8, (H, W - 2)).astype(np.float32)
    # HONOUR B-P25: planes 1..T carry plane-0 boundary rows 0,H+1 and cols 0,W-1
    for t in range(1, T + 1):
        U[t, 0, :] = U[0, 0, :]
        U[t, H + 1, :] = U[0, H + 1, :]
        U[t, :, 0] = U[0, :, 0]
        U[t, :, W - 1] = U[0, :, W - 1]

    def jacobi_reference(Uin):
        # independent two-loop numpy, t-outermost + explicit two-plane copy
        H_, W_ = Uin.shape[1] - 2, Uin.shape[2]
        T_ = Uin.shape[0] - 1
        cur = Uin[0].copy()
        nxt = Uin[0].copy()
        out = Uin.copy()
        for t in range(T_):
            for i in range(1, H_ + 1):
                for j in range(1, W_ - 1):
                    nxt[i, j] = np.float32(0.2 * (cur[i, j] + cur[i - 1, j]
                                                  + cur[i + 1, j] + cur[i, j - 1]
                                                  + cur[i, j + 1]))
            cur, nxt = nxt, cur
            nxt[:] = cur  # explicit two-plane copy
            out[t + 1, 1:H_ + 1, 1:W_ - 1] = cur[1:H_ + 1, 1:W_ - 1]
        return out

    expected = {"U": jacobi_reference(U)}
    # oracle via plain kernel, with T patched for w2_odd
    old_T = w2mod.T
    w2mod.T = T
    try:
        Uc = U.copy()
        jacobi(Uc)
    finally:
        w2mod.T = old_T
    return {"inputs": {"U": U}, "expected": expected, "oracle": {"U": Uc}}


def _w3_arrays(seed=2, MQ=32, NR=32):
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 4, MQ).astype(np.int32)
    r = rng.integers(0, 4, NR).astype(np.int32)
    S = np.zeros((MQ + 1, NR + 1), np.int32)

    def sw_reference(q_, r_):
        # textbook two-loop DP with Python if STATEMENT
        Sout = np.zeros((len(q_) + 1, len(r_) + 1), np.int32)
        for i in range(1, len(q_) + 1):
            for j in range(1, len(r_) + 1):
                if q_[i - 1] == r_[j - 1]:
                    sub = MATCH
                else:
                    sub = MISMATCH
                Sout[i, j] = max(0, Sout[i - 1, j - 1] + sub,
                                 Sout[i - 1, j] - GAP, Sout[i, j - 1] - GAP)
        return Sout

    expected = {"S": sw_reference(q, r)}
    Sc = S.copy()
    sw(q, r, Sc)
    return {"inputs": {"q": q, "r": r, "S": S}, "expected": expected, "oracle": {"S": Sc}}


def main():
    w1meta = {"workload": "W1", "params": {"M": 64, "N": 64, "K": 64, "TM": 32,
              "TN": 32, "TK": 16, "PI": 2, "PJ": 2}, "dtype": "f32",
              "seed": 0, "tol": 0.0, "generator": "make_fixture.py"}
    _write("w1", w1meta, _w1_arrays())
    _write("w1_flip", {**w1meta, "params": {"M": 64, "N": 64, "K": 64, "TM": 32,
              "TK": 16, "PK": 4}}, _w1_arrays())
    large = {"workload": "W1", "params": {"M": 256, "N": 256, "K": 256, "TM": 64,
             "TN": 64, "TK": 64, "PI": 4, "PJ": 4}, "dtype": "bf16/f32",
             "seed": 0, "tol": None, "generator": "make_fixture.py"}
    rng = np.random.default_rng(0)
    A = rng.integers(-8, 8, (256, 256)).astype(np.float32)
    B = rng.integers(-8, 8, (256, 256)).astype(np.float32)
    C = np.zeros((256, 256), np.float32)
    d = ROOT / "w1_large"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(large, indent=2, sort_keys=True) + "\n")
    np.savez(d / "inputs.npz", A=A, B=B, C=C)
    for f in ("expected.npz", "oracle.npz"):
        p = d / f
        if p.exists():
            p.unlink()
    w2meta = {"workload": "W2", "params": {"T": 4, "H": 16, "W": 16, "PI": 2, "HS": 8},
              "dtype": "f32", "seed": 1, "tol": 1e-5, "generator": "make_fixture.py"}
    _write("w2", w2meta, _w2_arrays())
    _write("w2_odd", {**w2meta, "params": {**w2meta["params"], "T": 5}}, _w2_arrays(T=5))
    w3meta = {"workload": "W3", "params": {"MQ": 32, "NR": 32, "PJ": 4, "CW": 8,
              "MATCH": 2, "MISMATCH": -1, "GAP": 1}, "dtype": "i32",
              "seed": 2, "tol": 0.0, "generator": "make_fixture.py"}
    _write("w3", w3meta, _w3_arrays())
    _write("w1_l1_overflow", {"workload": "W1", "params": {"M": 192, "N": 192, "K": 192,
             "TM": 96, "TN": 96, "TK": 32, "PI": 2, "PJ": 2}, "dtype": "f32",
             "seed": None, "tol": None, "generator": "make_fixture.py",
             "expect": "reject", "code": "L1-CAPACITY"}, only_meta=True)
    _write("w2_zero_t", {"workload": "W2", "params": {"T": 0, "H": 16, "W": 16, "PI": 2, "HS": 8},
             "dtype": "f32", "seed": None, "tol": None, "generator": "make_fixture.py",
             "expect": "reject", "code": "SWAP-PARITY"}, only_meta=True)
    _write("w2_pi4", {"workload": "W2", "params": {"T": 4, "H": 16, "W": 16, "PI": 4, "HS": 4},
             "dtype": "f32", "seed": None, "tol": None, "generator": "make_fixture.py",
             "expect": "reject", "code": "DMA-CHANNELS"}, only_meta=True)


if __name__ == "__main__":
    main()
