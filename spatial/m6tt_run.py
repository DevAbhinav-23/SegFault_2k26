"""M6-TT — run a `TTProgram` on Tenstorrent's functional simulator. Owner: Person B.

State file: `design/PROGRESS-TT.md`. The counterpart of `spatial.m6_tools` for the TT backend:
`spatial.m5tt_emit` turns a `MappingPlan` into a `TTProgram` and this module executes it, on
`ttsim` or on real silicon — the difference is one environment variable, not one line of this
file.

`ttnn` is imported **inside** `run`, so the emitter, the checker and the mapper stay usable in
the project's own `.venv`, which cannot hold `ttnn` at all: the wheel pins `numpy<2` against the
project's `numpy==2.5.3`. The TT venv is `.venv-tt`, built by `scripts/tt_env.sh`.

This module decides nothing about the program either. Every descriptor field is read out of the
`TTProgram`: the core range, the kernel text, the circular-buffer table, the runtime-argument
ABI (`("addr", <tensor>)` / `("const", <int>)`) and the compile-time-define names. The one thing
it adds is what only a live device knows — each tensor's DRAM base address and its
`TensorAccessorArgs` — and it **checks** the emitter's page arithmetic against the device rather
than trusting it.

The `ttnn` host API used here is the gate report's, verbatim
(`/home/adi/Projects/Honours/tt-probe/t3_generic_op.py`, `t6_multicore.py`): `ttnn.open_device`,
`ttnn.Tensor`, `ttnn.TensorAccessorArgs(t).get_compile_time_args()`, `ttnn.CBDescriptor`,
`ttnn.CBFormatDescriptor`, `ttnn.KernelDescriptor` with
`SourceType.SOURCE_CODE`, `ttnn.DataMovementConfigDescriptor`, `ttnn.RuntimeArgs`,
`ttnn.ProgramDescriptor`, `ttnn.generic_op`, `Tensor.cpu().to_numpy()`.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from spatial.m5tt_emit import TTProgram

_ENV_HINT = "run `source scripts/tt_env.sh` first (design/PROGRESS-TT.md §2)"


class TTRunError(Exception):
    """Every failure out of the TT runner: a broken environment, or a program the device
    rejected."""


def _dtypes(ttnn: Any) -> dict[str, Any]:
    """`Dtype.value` → the `ttnn` data type, for the dtypes a scalar kernel can hold."""
    return {"f32": ttnn.float32, "i32": ttnn.int32}


def _check_environment() -> None:
    """The three facts `scripts/tt_env.sh` establishes. Checked, because each one fails far from
    here: a missing simulator aborts inside UMD, and a stale `TT_METAL_HOME` sends `tt_metal`
    looking for a source tree that the wheel does not ship."""
    simulator = os.environ.get("TT_METAL_SIMULATOR")
    if not simulator:
        raise TTRunError(f"TT_METAL_SIMULATOR is not set, so there is no device to open: "
                         f"{_ENV_HINT}")
    if not os.path.exists(simulator):
        raise TTRunError(f"TT_METAL_SIMULATOR={simulator!r} does not exist: {_ENV_HINT}")
    if os.environ.get("TT_METAL_HOME"):
        raise TTRunError(f"TT_METAL_HOME is set to "
                         f"{os.environ['TT_METAL_HOME']!r}; the ttnn wheel is self-rooted and a "
                         f"stale value breaks the kernel build: {_ENV_HINT}")
    if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") != "1":
        raise TTRunError(f"TT_METAL_SLOW_DISPATCH_MODE is not '1'; there is no fast-dispatch "
                         f"firmware for the simulator: {_ENV_HINT}")


def _upload(ttnn: Any, device: Any, spec: Any, array: np.ndarray) -> Any:
    """One L3 tensor as a row-major DRAM tensor, with the emitter's page arithmetic checked."""
    dtype = _dtypes(ttnn).get(spec.dtype.value)
    if dtype is None:
        raise TTRunError(f"tensor {spec.name!r} has dtype {spec.dtype.value!r}, which this "
                         f"runner does not upload (it uploads {sorted(_dtypes(ttnn))})")
    if tuple(array.shape) != tuple(spec.shape):
        raise TTRunError(f"tensor {spec.name!r} has shape {tuple(array.shape)} where the "
                         f"program declares {tuple(spec.shape)}")
    if array.dtype != np.dtype(spec.dtype.numpy):
        raise TTRunError(f"tensor {spec.name!r} has dtype {array.dtype} where the program "
                         f"declares {np.dtype(spec.dtype.numpy)}")
    tensor = ttnn.Tensor(np.ascontiguousarray(array).reshape(-1).tolist(), list(spec.shape),
                         dtype, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    if tensor.buffer_page_size() != spec.page_bytes:
        raise TTRunError(
            f"tensor {spec.name!r} is paged at {tensor.buffer_page_size()} B on the device "
            f"where the emitter addressed it at {spec.page_bytes} B a page; the kernel's "
            f"page_id arithmetic assumes one page is one row")
    return tensor


def _descriptor(ttnn: Any, program: TTProgram, uploaded: dict[str, Any]) -> Any:
    """The `ttnn.ProgramDescriptor`: one kernel over the core range, the CB table, the args."""
    (x0, y0), (x1, y1) = program.core_range
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])
    formats = _dtypes(ttnn)
    cbs = [ttnn.CBDescriptor(
        total_size=cb.bytes, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb.index,
                                                    data_format=formats[cb.dtype.value],
                                                    page_size=cb.page_bytes)])
        for cb in program.cbs]

    compile_time: list[int] = []
    defines: list[tuple[str, str]] = []
    for spec in program.io_tensors:
        defines.append((spec.cta_define, str(len(compile_time))))
        compile_time += list(
            ttnn.TensorAccessorArgs(uploaded[spec.name]).get_compile_time_args())

    runtime = ttnn.RuntimeArgs()
    for (x, y), args in program.runtime_args:
        runtime[x][y] = [uploaded[value].buffer_address() if kind == "addr" else int(value)
                         for kind, value in args]

    kernel = ttnn.KernelDescriptor(
        kernel_source=program.source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores, compile_time_args=compile_time, defines=defines,
        runtime_args=runtime,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default))
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)


def run(program: TTProgram, tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Execute `program` on the device `TT_METAL_SIMULATOR` names, returning every L3 tensor.

    `tensors` supplies one array per `program.io_tensors` entry, by name, with that entry's exact
    shape and dtype — including the output, whose contents are uploaded like any other (a plan
    that does not write every element of its output would otherwise read back whatever the
    allocator left there). The returned arrays are fresh copies read off the device; the inputs
    are not mutated.
    """
    _check_environment()
    import ttnn                                       # noqa: PLC0415 — lazy: see the docstring

    missing = [spec.name for spec in program.io_tensors if spec.name not in tensors]
    if missing:
        raise TTRunError(f"no array was given for {missing}, which {program.launch_name!r} "
                         f"declares as L3 tensors")

    device = ttnn.open_device(device_id=0)
    try:
        uploaded = {spec.name: _upload(ttnn, device, spec, tensors[spec.name])
                    for spec in program.io_tensors}
        order = [uploaded[spec.name] for spec in program.io_tensors]
        ttnn.generic_op(order, _descriptor(ttnn, program, uploaded))
        return {spec.name: uploaded[spec.name].cpu().to_numpy().reshape(spec.shape)
                for spec in program.io_tensors}
    finally:
        ttnn.close_device(device)


__all__ = ["TTRunError", "run"]
