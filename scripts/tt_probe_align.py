"""Q-TT7 / ruling R-TT-A′: which (src offset, dst offset, size) triples does the NoC accept?

The measurement `design/08-tt-backend.md` §3.3 rests on. Run it, one case per process, with::

    source scripts/tt_env.sh
    for i in $(seq 0 22); do .venv-tt/bin/python scripts/tt_probe_align.py $i; done

**One case per process is not a style choice**: ttsim's `UndefinedBehavior` is fatal — it
prints, the host exits 1, and every later launch in that session is skipped.


A case that prints no `ERROR: UndefinedBehavior: noc_cmd_ctrl: ... alignment of src_addr=...
and dst_addr=... does not match` line was accepted. The verdicts are tabulated in
`design/PROGRESS-TT.md` §T2.4.

Case encoding in runtime args: [dram_addr, kind, src_off, dst_off, size]
kind 0 = L1 -> L1 (same core, scratch CB 1 -> CB 2)
kind 1 = L1 -> DRAM write
kind 2 = DRAM -> L1 read
"""
import sys
import time

import ttnn

SRC = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto t_args = TensorAccessorArgs<TA_t>();
    const auto t_ta = TensorAccessor(t_args, get_arg_val<uint32_t>(0));
    const uint32_t kind    = get_arg_val<uint32_t>(1);
    const uint32_t src_off = get_arg_val<uint32_t>(2);
    const uint32_t dst_off = get_arg_val<uint32_t>(3);
    const uint32_t size    = get_arg_val<uint32_t>(4);
    const uint32_t a_l1 = get_write_ptr(0);
    const uint32_t b_l1 = get_write_ptr(1);

    if (kind == 0u) {
        noc_async_write(a_l1 + src_off, get_noc_addr(my_x[0], my_y[0], b_l1 + dst_off), size);
        noc_async_write_barrier();
    } else if (kind == 1u) {
        noc_async_write(a_l1 + src_off, t_ta.get_noc_addr(0, dst_off), size);
        noc_async_write_barrier();
    } else {
        noc_async_read(t_ta.get_noc_addr(0, src_off), a_l1 + dst_off, size);
        noc_async_read_barrier();
    }
}
"""

CASES = [
    (0, 0, 0, 32, "L1->L1  src 0  dst 0   32B"),
    (0, 4, 4, 4, "L1->L1  src 4  dst 4    4B"),
    (0, 4, 0, 4, "L1->L1  src 4  dst 0    4B   (expected UB)"),
    (0, 16, 0, 16, "L1->L1  src 16 dst 0   16B"),
    (0, 16, 32, 16, "L1->L1  src 16 dst 32  16B"),
    (0, 8, 8, 8, "L1->L1  src 8  dst 8    8B"),
    (1, 0, 0, 32, "L1->DRAM src 0  dst 0   32B"),
    (1, 4, 4, 4, "L1->DRAM src 4  dst 4    4B"),
    (1, 4, 0, 4, "L1->DRAM src 4  dst 0    4B   (expected UB)"),
    (1, 4, 36, 32, "L1->DRAM src 4  dst 36  32B  (both == 4 mod 16/32)"),
    (1, 16, 0, 16, "L1->DRAM src 16 dst 0   16B"),
    (1, 16, 32, 16, "L1->DRAM src 16 dst 32  16B"),
    (1, 0, 16, 16, "L1->DRAM src 0  dst 16  16B"),
    (1, 0, 156, 4, "L1->DRAM src 0  dst 156  4B  (W3 EastOut, raw)"),
    (1, 28, 156, 4, "L1->DRAM src 28 dst 156  4B  (W3 EastOut, matched)"),
    (1, 0, 144, 16, "L1->DRAM src 0  dst 144 16B  (W3 EastOut RMW chunk)"),
    (2, 0, 0, 32, "DRAM->L1 src 0  dst 0   32B"),
    (2, 0, 0, 4, "DRAM->L1 src 0  dst 0    4B"),
    (2, 28, 28, 4, "DRAM->L1 src 28 dst 28   4B"),
    (2, 28, 0, 4, "DRAM->L1 src 28 dst 0    4B   (expected UB)"),
    (2, 32, 0, 32, "DRAM->L1 src 32 dst 0   32B"),
    (2, 16, 0, 16, "DRAM->L1 src 16 dst 0   16B"),
    (2, 128, 0, 32, "DRAM->L1 src 128 dst 0  32B  (W3 EastOut RMW read)"),
]


def main() -> int:
    device = ttnn.open_device(device_id=0)
    try:
        t = ttnn.Tensor([0] * 64, [1, 64], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device,
                        ttnn.DRAM_MEMORY_CONFIG)
        pad = ttnn.Tensor([0] * 8, [1, 8], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device,
                          ttnn.DRAM_MEMORY_CONFIG)
        print(f"t page bytes = {t.buffer_page_size()}", flush=True)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        cbs = [ttnn.CBDescriptor(
            total_size=256, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i,
                                                        data_format=ttnn.uint32,
                                                        page_size=256)])
               for i in range(2)]
        cta = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        for kind, s_off, d_off, size, label in [CASES[int(sys.argv[1])]]:
            rt = ttnn.RuntimeArgs()
            rt[0][0] = [t.buffer_address(), kind, s_off, d_off, size]
            kernel = ttnn.KernelDescriptor(
                kernel_source=SRC,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores, compile_time_args=cta, defines=[("TA_t", "0")],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.RISCV_0_default))
            print(f"CASE {label}", flush=True)
            sys.stdout.flush()
            try:
                ttnn.generic_op([pad, t], ttnn.ProgramDescriptor(kernels=[kernel],
                                                                 semaphores=[], cbs=cbs))
            except Exception as exc:                            # noqa: BLE001
                print(f"  RAISED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            time.sleep(0.05)
        print("DONE", flush=True)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
