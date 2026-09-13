"""Q-TT2 / C-TT3: which coordinate space does `get_noc_addr(noc_x, noc_y, addr)` accept?

    source scripts/tt_env.sh
    .venv-tt/bin/python scripts/tt_probe_coords.py

Verdicts in `design/PROGRESS-TT.md` §T2.3.


Chain of 4 cores. Core k writes its tag into core k+1's mailbox CB, barriers, then increments
core k+1's semaphore 0. Core k>0 waits sem0 >= 1 and reports what it received. No local zeroing
of the mailbox (that was round 1's race: core 0 writes before core 1 has zeroed).

Run A passes device.worker_core_from_logical_core(...) as (nx, ny) — the design's choice.
Run B passes the soc descriptor's *physical* functional_worker coordinates (x+1, 1) instead.
"""
import sys

import ttnn

SRC = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto dbg_args = TensorAccessorArgs<TA_dbg>();
    const auto dbg_ta = TensorAccessor(dbg_args, get_arg_val<uint32_t>(0));
    const int32_t tx    = (int32_t)get_arg_val<uint32_t>(1);
    const uint32_t nx   = get_arg_val<uint32_t>(2);
    const uint32_t ny   = get_arg_val<uint32_t>(3);
    const uint32_t mode = get_arg_val<uint32_t>(4);
    const uint32_t rep_l1 = get_write_ptr(0);
    const uint32_t box_l1 = get_write_ptr(1);
    const uint32_t out_l1 = get_write_ptr(2);
    volatile tt_l1_ptr uint32_t* rep  = (volatile tt_l1_ptr uint32_t*)rep_l1;
    volatile tt_l1_ptr uint32_t* box  = (volatile tt_l1_ptr uint32_t*)box_l1;
    volatile tt_l1_ptr uint32_t* outb = (volatile tt_l1_ptr uint32_t*)out_l1;

    rep[0] = rep_l1;
    rep[1] = box_l1;
    rep[2] = (uint32_t)get_semaphore(0);
    rep[3] = (uint32_t)get_semaphore(1);
    rep[4] = 0xffffffffu;
    rep[5] = 0u;
    rep[6] = 0u;
    rep[7] = out_l1;

    if (tx > 0) {
        if (mode == 1u) {
            noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*)get_semaphore(0), 1);
        } else {
            // bounded spin: the negative control must terminate even if nothing arrives
            volatile tt_l1_ptr uint32_t* sem =
                (volatile tt_l1_ptr uint32_t*)get_semaphore(0);
            for (int32_t s = 0; s < 20000; s += 1) { if (*sem >= 1u) { break; } }
        }
        rep[4] = box[0];
        rep[5] = *(volatile tt_l1_ptr uint32_t*)get_semaphore(0);
    }
    if (tx < 3) {
        outb[0] = 0xbeef0000u + (uint32_t)tx;
        noc_async_write(out_l1, get_noc_addr(nx, ny, box_l1), 4);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(nx, ny, get_semaphore(0)), 1);
        rep[6] = 0xbeef0000u + (uint32_t)tx;
    }

    noc_async_write(rep_l1, dbg_ta.get_noc_addr((uint32_t)tx, 0), 32);
    noc_async_write_barrier();
}
"""


def main() -> int:
    device = ttnn.open_device(device_id=0)
    try:
        worker = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)) for x in range(4)]
        worker = [(c.x, c.y) for c in worker]
        physical = [(x + 1, 1) for x in range(4)]     # soc_descriptor functional_workers row 1
        print(f"worker_core_from_logical_core: {worker}", flush=True)
        print(f"soc_descriptor physical row:   {physical}", flush=True)

        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
        cbs = [ttnn.CBDescriptor(
            total_size=size, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i,
                                                        data_format=ttnn.uint32,
                                                        page_size=size)])
               for i, size in enumerate((32, 16, 16))]

        wrong = [(c[0] + 1, c[1]) for c in worker]    # negative control: off by one core
        for label, coords in (("A worker/virtual", worker), ("B physical", physical),
                              ("C wrong (worker+1)", wrong)):
            dbg = ttnn.Tensor([0] * 32, [4, 8], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device,
                              ttnn.DRAM_MEMORY_CONFIG)
            pad = ttnn.Tensor([0] * 8, [1, 8], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device,
                              ttnn.DRAM_MEMORY_CONFIG)
            sems = [ttnn.SemaphoreDescriptor(id=i, core_type=ttnn.CoreType.WORKER,
                                             core_ranges=cores, initial_value=0)
                    for i in range(2)]
            rt = ttnn.RuntimeArgs()
            for x in range(4):
                nbr = coords[x + 1] if x < 3 else coords[x]
                rt[x][0] = [dbg.buffer_address(), x, nbr[0], nbr[1],
                            1 if label.startswith("A") else 0]
            kernel = ttnn.KernelDescriptor(
                kernel_source=SRC,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=list(ttnn.TensorAccessorArgs(dbg).get_compile_time_args()),
                defines=[("TA_dbg", "0")], runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.RISCV_0_default))
            print(f"== run {label} ==", flush=True)
            try:
                ttnn.generic_op([pad, dbg], ttnn.ProgramDescriptor(
                    kernels=[kernel], semaphores=sems, cbs=cbs))
                out = dbg.cpu().to_numpy().reshape(4, 8)
                for x in range(4):
                    print(f"  core {x}: " + " ".join(f"0x{v:08x}" for v in out[x]), flush=True)
                got = [int(out[x, 4]) for x in range(1, 4)]
                want = [0xbeef0000 + (x - 1) for x in range(1, 4)]
                print(f"  delivered: {got == want}  (got {[hex(v) for v in got]})", flush=True)
            except Exception as exc:                          # noqa: BLE001
                print(f"  RAISED {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
