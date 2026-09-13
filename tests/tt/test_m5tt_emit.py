"""M5-TT unit tests: the `TTProgram` W1 emits, and the emitter's own discipline.

These run in the project's own `.venv` and join the default suite: `spatial.m5tt_emit` imports
nothing but the standard library and `spatial.model`, which is the point of keeping `ttnn`
confined to `spatial.m6tt_run`. The simulator run is `tests/tt/test_tt_w1.py`, marked
`requires_ttsim`.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from spatial import m4_mapping as m4
from spatial import m5tt_emit as m5tt
from tests.fixtures.mappings import w1_legal, w1flip_legal, w2_legal, w3_legal

SOURCE = inspect.getsource(m5tt)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def w1():
    """W1's `TTProgram`. The target string is an **AIR** target and means nothing to a Tensix
    core, so `emit` takes none; `test_target_does_not_reach_the_tt_program` holds it to that."""
    return m5tt.emit(m4.plan(w1_legal.legal("npu1")))


# -- the program W1 produces -------------------------------------------------

def test_W1_grid_is_the_logical_herd(w1):
    """`HerdPlan.grid` is `(2, 2)`, and the core range is its 4 cores — not the AIE physical
    herd `(1, 2)` with `repeats`, which is strip-mining and has no meaning on a Tensix grid."""
    assert w1.grid == (2, 2)
    assert w1.core_range == ((0, 0), (1, 1))
    assert [core for core, _ in w1.runtime_args] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_W1_runtime_args_are_addresses_then_coordinates(w1):
    """The ABI: one base address per L3 tensor in `plan.tensors` order, then the herd
    coordinates in `HerdPlan.coords` order. Only the emitter states it."""
    assert dict(w1.runtime_args)[(1, 0)] == (
        ("addr", "A"), ("addr", "B"), ("addr", "C"), ("const", 1), ("const", 0))


def test_W1_cbs_are_one_per_L1_buffer(w1):
    """`acc` 4096 B, `a` 2048 B, `b` 2048 B, one page each — `ping_pong_candidate` is ignored,
    so `a` and `b` are **not** doubled."""
    assert [(cb.index, cb.name, cb.bytes, cb.page_bytes) for cb in w1.cbs] == [
        (0, "acc", 4096, 4096), (1, "a", 2048, 2048), (2, "b", 2048, 2048)]


def test_W1_io_tensors_are_paged_by_the_row(w1):
    """One page is one row: `shape[-1] * dtype.sizeof`. `m6tt_run` checks the device agrees."""
    assert [(t.name, t.shape, t.page_bytes, t.cta_define) for t in w1.io_tensors] == [
        ("A", (64, 64), 256, "TA_A"), ("B", (64, 64), 256, "TA_B"),
        ("C", (64, 64), 256, "TA_C")]


def test_W1_kernel_has_the_k0_loop_with_four_trips(w1):
    """W1's reduction is tiled `TK = 16` over `K = 64`: four trips, from the plan's `LoopPlan`."""
    assert "for (int32_t k0 = 0; k0 < 64; k0 += 16) {" in w1.source


def test_W1_kernel_reads_the_A_and_B_slabs_row_by_row(w1):
    """The herd side does the DRAM transfer itself.

    `A2L1` is a broadcast channel of `size=(2, 1)` over `broadcast_shape=(2, 2)`, so the core at
    `(tx, ty)` is fed by source index `tx % 2`: its 32 rows start at row `(tx % 2) * 32` and its
    16 columns at `k0`, one 64 B run per row into `a`. `B2L1` is `size=(1, 2)`: 16 rows from
    `k0`, 32 columns from `(ty % 2) * 32`, 128 B a run into `b`.
    """
    assert ("noc_async_read(A_ta.get_noc_addr((uint32_t)((((tx % 2) * 32) + _r1_0)), "
            "(uint32_t)(k0 * 4)), a_l1 + (uint32_t)((_r1_0 * 16) * 4), 64);") in w1.source
    assert "for (int32_t _r1_0 = 0; _r1_0 < 32; _r1_0 += 1) {" in w1.source
    assert ("noc_async_read(B_ta.get_noc_addr((uint32_t)((k0 + _r2_0)), "
            "(uint32_t)(((ty % 2) * 32) * 4)), b_l1 + (uint32_t)((_r2_0 * 32) * 4), 128);"
            ) in w1.source
    assert "for (int32_t _r2_0 = 0; _r2_0 < 16; _r2_0 += 1) {" in w1.source
    assert w1.source.count("noc_async_read_barrier();") == 2


def test_W1_kernel_zeroes_the_accumulator_then_accumulates(w1):
    """Both nests come from the plan: the zeroing `LoopPlan` of `StoreNode(Const 0)`, and the
    `i1`/`j1`/`k1` nest carrying W1's one desugared `accumulate`."""
    assert "acc[((i1 * 32) + j1)] = ((float)(0.0));" in w1.source
    assert "for (int32_t k1 = 0; k1 < 16; k1 += 1) {" in w1.source
    assert ("acc[((i1 * 32) + j1)] = (acc[((i1 * 32) + j1)] + "
            "(a[((i1 * 16) + k1)] * b[((k1 * 32) + j1)]));") in w1.source


def test_W1_kernel_writes_the_C_tile_back(w1):
    """`C2L3` is unbroadcast, so the segment get's `(i_drain, j_drain)` inverts to `(tx, ty)`
    and the tile lands at `C[tx * 32 : +32, ty * 32 : +32]`, row by row."""
    assert ("noc_async_write(acc_l1 + (uint32_t)((_r3_0 * 32) * 4), "
            "C_ta.get_noc_addr((uint32_t)(((tx * 32) + _r3_0)), (uint32_t)((ty * 32) * 4)), "
            "128);") in w1.source
    assert w1.source.count("noc_async_write_barrier();") == 1


def test_W1_kernel_declares_one_l1_pointer_per_cb(w1):
    for cb in w1.cbs:
        assert f"const uint32_t {cb.name}_l1 = get_write_ptr({cb.index});" in w1.source
        assert (f"volatile tt_l1_ptr float* {cb.name} = "
                f"(volatile tt_l1_ptr float*){cb.name}_l1;") in w1.source


def test_W1_kernel_has_balanced_braces(w1):
    assert w1.source.count("{") == w1.source.count("}")
    assert w1.source.rstrip().endswith("}")


# -- what T1 does not do -----------------------------------------------------

@pytest.mark.parametrize("fixture", (w1flip_legal, w2_legal, w3_legal))
def test_out_of_scope_plans_raise_not_implemented(fixture):
    """A core-to-core channel (the flip's `CascadeK`, W2's `ToNorth`, W3's `West`) and the
    segment-side programs W2 and W3 need are T2/T3/T4. They must say so, and by raising a
    `NotImplementedError`, never by emitting something plausible."""
    with pytest.raises(m5tt.TTNotImplemented) as raised:
        m5tt.emit(m4.plan(fixture.legal("npu1")))
    assert isinstance(raised.value, NotImplementedError)
    assert isinstance(raised.value, m5tt.TTEmitError)


def test_cascade_channel_is_named_in_its_own_error():
    """The diagnostic has to name the construct, or it is not actionable."""
    with pytest.raises(m5tt.TTNotImplemented, match=r"CascadeK.*core-to-core.*T2/T3/T4"):
        m5tt.emit(m4.plan(w1flip_legal.legal("npu1")))


# -- D-14, and the module's own dependencies ---------------------------------

def _code_only(source: str) -> str:
    """`source` with its comments and every docstring removed — the *code*, which is what D-14
    is about. The module docstring has to be free to say which names the code may not use."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            del body[0]
    return ast.unparse(tree)


def test_emitter_makes_no_decisions():
    """D-14, as for M5: the TT emitter reads no field of the legality or schedule contracts and
    no kernel size. Everything it emits it read out of the `MappingPlan`."""
    code = _code_only(SOURCE)
    for forbidden in ("plan.mapping", "LegalMapping", "KernelModel", "ScheduleModel",
                      "physical_herd", "repeats", ".schedule", ".kernel", "if tx ==",
                      "npu1", "npu2"):
        assert forbidden not in code, f"M5-TT's code names {forbidden!r}"


def test_emitter_imports_nothing_but_stdlib_and_spatial():
    """It has to run in the project's own `.venv`, which cannot hold `ttnn`: the wheel pins
    `numpy<2` against the project's `numpy==2.5.3`."""
    allowed = {"spatial", "dataclasses", "math", "typing", "__future__"}
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert name.split(".")[0] in allowed, f"M5-TT imports {name!r}"


def test_runner_imports_ttnn_lazily():
    """`spatial.m6tt_run` is importable without `ttnn`, so the default suite can collect it."""
    source = (ROOT / "spatial" / "m6tt_run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
           for alias in node.names if _at_module_level(tree, node)}
    assert "ttnn" not in top, "m6tt_run imports ttnn at module level"
    assert any(isinstance(node, ast.Import) and any(a.name == "ttnn" for a in node.names)
               for node in ast.walk(tree)), "m6tt_run never imports ttnn at all"


def _at_module_level(tree: ast.Module, target: ast.AST) -> bool:
    return any(node is target for node in tree.body)


def test_emission_is_deterministic_across_hash_seeds():
    """No set or dict iteration order reaches the text: two interpreters with different
    `PYTHONHASHSEED` must emit byte-identical C++ and an identical program."""
    script = ("import sys;"
              "from spatial import m4_mapping as m4, m5tt_emit as tt;"
              "from tests.fixtures.mappings import w1_legal;"
              "p = tt.emit(m4.plan(w1_legal.legal('npu1')));"
              "sys.stdout.write(repr((p.source, p.cbs, p.io_tensors, p.runtime_args)))")
    outputs = [subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True,
                              capture_output=True, text=True,
                              env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}).stdout
               for seed in ("0", "12345")]
    assert outputs[0] == outputs[1]


def test_target_does_not_reach_the_tt_program():
    """`"npu1"` and `"npu2"` select an AIE generation for `air.api`'s `build()`. M4's plan is the
    same either way, so the TT program must be too — the claim `design/PROGRESS-TT.md` §4 makes
    about the plan being backend-neutral is exactly this."""
    assert (m5tt.emit(m4.plan(w1_legal.legal("npu1")))
            == m5tt.emit(m4.plan(w1_legal.legal("npu2"))))
    assert "target" not in inspect.signature(m5tt.emit).parameters
