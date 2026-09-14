"""W1-large — smoke/IR-facts shape: bf16 inputs, f32 output, 256^3.

Kernel body reused from w1_gemm (same triple loop); only the large-shape
params differ. No expected/oracle fixture (levels I and S only).
"""

from kernels.w1_gemm import gemm as gemm_bf16

M, N, K = 256, 256, 256
TM, TN, TK = 64, 64, 64
PI, PJ = 4, 4
PARAMS_LARGE = {"M": M, "N": N, "K": K, "TM": TM, "TN": TN, "TK": TK,
                "PI": PI, "PJ": PJ}
# bf16 inputs (A, B), f32 output (C) — stored as f32 arrays per §4.


def schedule_os(target):
    raise NotImplementedError("surface: Person A M1/M2")


def main():
    try:
        print(schedule_os("npu1"))
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    main()
