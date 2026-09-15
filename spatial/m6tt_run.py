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
"""The one fix every environment failure below ends with: the script is the only place the three
variables are spelled, so a message that names it cannot go stale the way a copy would."""


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


def _padded_shape(spec: Any) -> tuple[int, ...]:
    """The shape actually uploaded: the tensor's, with its last dimension widened to the padded
    row `m5tt_emit` addressed (ruling R-TT-A′, `design/08-tt-backend.md` §3.3)."""
    return tuple(spec.shape[:-1]) + (spec.row_elems,)


def _upload(ttnn: Any, device: Any, spec: Any, array: np.ndarray) -> Any:
    """One L3 tensor as a row-major DRAM tensor under R-TT-A′: `pad_elems` elements of leading
    pad, rows widened to `page_bytes`, and the emitter's page arithmetic checked against the
    device. Padding here is the whole of the layout policy — the kernel reads the two numbers as
    literals, and nothing else in the program knows about it."""
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
    shape = _padded_shape(spec)
    padded = np.zeros(shape, dtype=array.dtype)
    padded[..., spec.pad_elems:spec.pad_elems + spec.shape[-1]] = array
    tensor = ttnn.Tensor(padded.reshape(-1).tolist(), list(shape),
                         dtype, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    if tensor.buffer_page_size() != spec.page_bytes:
        raise TTRunError(
            f"tensor {spec.name!r} is paged at {tensor.buffer_page_size()} B on the device "
            f"where the emitter addressed it at {spec.page_bytes} B a page; the kernel's "
            f"page_id arithmetic assumes one page is one row")
    return tensor


def _download(spec: Any, tensor: Any) -> np.ndarray:
    """The tensor back, with R-TT-A′'s padding stripped."""
    whole = tensor.cpu().to_numpy().reshape(_padded_shape(spec))
    return np.ascontiguousarray(whole[..., spec.pad_elems:spec.pad_elems + spec.shape[-1]])


def _descriptor(ttnn: Any, device: Any, program: TTProgram, uploaded: dict[str, Any]) -> Any:
    """The `ttnn.ProgramDescriptor`: one kernel over the core range, the CB table, the
    semaphores, the args."""
    (x0, y0), (x1, y1) = program.core_range
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])
    formats = _dtypes(ttnn)
    cbs = [ttnn.CBDescriptor(
        total_size=cb.bytes, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb.index,
                                                    data_format=formats[cb.dtype.value],
                                                    page_size=cb.page_bytes)])
        for cb in program.cbs]
    # Over the whole core range, so a given id names the same L1 address on every core — A-TT1,
    # which is what lets a producer address its consumer's semaphore as its own.
    semaphores = [ttnn.SemaphoreDescriptor(id=sem.id, core_type=ttnn.CoreType.WORKER,
                                           core_ranges=cores, initial_value=sem.initial_value)
                  for sem in program.semaphores]

    compile_time: list[int] = []
    defines: list[tuple[str, str]] = []
    for spec in program.io_tensors:
        defines.append((spec.cta_define, str(len(compile_time))))
        compile_time += list(
            ttnn.TensorAccessorArgs(uploaded[spec.name]).get_compile_time_args())

    def _noc(logical: tuple[int, int], axis: str) -> int:
        """The NoC coordinate of a logical worker core — the one thing about the grid that only
        a live device knows (C-TT3 / Q-TT2, `design/08-tt-backend.md` §3.4)."""
        core = device.worker_core_from_logical_core(ttnn.CoreCoord(logical[0], logical[1]))
        return int(core.x if axis == "x" else core.y)

    runtime = ttnn.RuntimeArgs()
    for (x, y), args in program.runtime_args:
        runtime[x][y] = [
            uploaded[value].buffer_address() if kind == "addr" else
            _noc(value, kind[-1]) if kind in ("noc_x", "noc_y") else int(value)
            for kind, value in args]

    kernel = ttnn.KernelDescriptor(
        kernel_source=program.source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores, compile_time_args=compile_time, defines=defines,
        runtime_args=runtime,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default))
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=semaphores, cbs=cbs)


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
        if len(order) == 1:
            # `ttnn.generic_op` refuses a single entry — `TT_FATAL @ generic_op_device_operation
            # .cpp:135: io_tensors.size() >= 2`, "must contain at least one input tensor and one
            # output tensor" — because it expects the pre-allocated output last. W2 declares one
            # L3 tensor, read and written in place, so the same handle stands in both roles. A
            # host-API shape, not a fact about the plan: the kernel is addressed through
            # `TensorAccessorArgs` built from `program.io_tensors`, which is unchanged.
            order = order + order
        ttnn.generic_op(order, _descriptor(ttnn, device, program, uploaded))
        return {spec.name: _download(spec, uploaded[spec.name])
                for spec in program.io_tensors}
    finally:
        ttnn.close_device(device)


__all__ = ["TTRunError", "run"]
