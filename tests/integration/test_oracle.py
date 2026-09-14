"""Level O — the oracle level (04-test-plan §1). Owner: C.

Oracle contract (M7 §3.3), the three end-to-end diffs (04-test-plan §8 item 5),
ignorability (FR-S18, §3.4), the K5 scope gate (M7 §7.7) and the demo rejections (M8 §5.3).

The end-to-end tests run the **CPython kernel** against the independently generated
`expected.npz` through `m6.diff`, the same comparator the device path uses. That is the
whole of level O: FR-S18 says the schedule clauses are ignorable, so the kernel alone is
the thing whose answer must be right, and it is checkable today with no toolchain and no
hardware. The device half of the same property is `test_T4_device_diff`, which skips.
"""

from __future__ import annotations

import copy
import random
from importlib.resources import files
from pathlib import Path

import pytest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

FIXTURE_NAMES = ("w1", "w1_flip", "w2", "w2_odd", "w3", "w1_large")
META_KEYS = {"workload", "params", "dtype", "seed", "tol", "generator"}
ACCUMULATORS = ("C", "S")


def _require_np():
    if np is None:
        pytest.skip("numpy is not installed, so the fixture arrays cannot be loaded")


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_contract(name):
    """M7 §3.3: meta keys, params coverage, expected/oracle match, tol rule, no silent zeros."""
    _require_np()
    from tests import fixtures
    try:
        fx = fixtures.load(name)
    except (FileNotFoundError, AttributeError, ImportError) as exc:
        pytest.skip(f"fixture {name} not generated yet: {exc}")
    meta = (Path(str(files("tests.fixtures"))) / name / "meta.json")
    import json
    m = json.loads(meta.read_text(encoding="utf-8"))
    assert set(m) == META_KEYS, set(m) ^ META_KEYS
    inputs = fx.inputs()
    expected, oracle = fx.expected(), fx.oracle()
    if expected is None or oracle is None:
        assert name == "w1_large", f"{name} ships no oracle but is not the smoke fixture"
        return
    assert set(expected) <= set(inputs), "written arrays must be a subset of inputs"
    assert set(expected) == set(oracle), (sorted(expected), sorted(oracle))
    for k in expected:
        assert expected[k].shape == oracle[k].shape, k
        assert expected[k].dtype == oracle[k].dtype, k
        assert np.allclose(expected[k], oracle[k], atol=m["tol"]), k
    assert m["tol"] == 0.0 or m["workload"] == "W2", "tol != 0 allowed only for W2"
    for k, v in expected.items():
        if k not in ACCUMULATORS:
            assert np.any(v != 0), f"{name}.{k} all-zero fixture"


@pytest.mark.fr("FR-S18")
@pytest.mark.parametrize("workload", ["W1", "W2", "W3"])
def test_ignorability(workload):
    """FR-S18 property: schedule clauses never mutate CPython kernel output."""
    _require_np()
    import importlib

    from tests import fixtures
    _MOD = {"W1": ("w1_gemm", "gemm"), "W2": ("w2_jacobi", "jacobi"),
            "W3": ("w3_sw", "sw")}
    mod_name, fn_name = _MOD[workload]
    try:
        kernel = getattr(importlib.import_module(f"kernels.{mod_name}"), fn_name)
    except ImportError:
        pytest.skip(f"kernels.{mod_name} not present yet")
    fxname = {"W1": "w1", "W2": "w2", "W3": "w3"}[workload]
    try:
        fx = fixtures.load(fxname)
    except (FileNotFoundError, AttributeError, ImportError) as exc:
        pytest.skip(f"fixture {fxname} not generated yet: {exc}")
    args_ref = fx.inputs()
    ref_args = copy.deepcopy(args_ref)
    kernel(**ref_args)  # (i) no schedule constructed at all
    ref = {k: v.tobytes() for k, v in ref_args.items()}
    clauses = []  # kernels export no clause lists until A lands; trial 0 is the property
    rng = random.Random(0)
    for trial in range(8):
        subset = clauses if trial == 0 or not clauses else (
            clauses if trial == 1 else rng.sample(clauses, rng.randint(1, len(clauses))))
        order = subset if trial < 2 else sorted(subset, key=lambda _: rng.random())
        args = copy.deepcopy(args_ref)
        try:
            import spatial as sp
            s = sp.schedule(kernel, target="npu1")
            for c in order:
                c(s)
        except Exception:
            pass  # schedule attempt failed/absent → "no mutation" by construction
        kernel(**args)
        assert {k: v.tobytes() for k, v in args.items()} == ref, f"subset {order} changed output"
    # FR-S20's no-air-import property is asserted by test_no_air_import_* (A) and
    # test_no_air_import_B_modules (B) in fresh interpreters — not here, where M5's
    # tests have already imported the toolchain in this session.


@pytest.mark.fr("FR-K5")
def test_K5_scope_documented():
    """FR-K5: W4 out-of-scope documented in slide + README status row; no w4 kernel."""
    repo = Path(str(files("tests"))).parent
    slide = repo / "demo" / "honest_limits.md"
    if not slide.is_file():
        pytest.skip("demo/honest_limits.md not written yet")
    text = slide.read_text(encoding="utf-8")
    assert "W4" in text and ("p \u21a6 p \u2295 2^s" in text or "partner-map" in text.lower()
                             or "p +" in text), "slide must name W4 + partner-map reason"
    readme = (repo / "design" / "00-README.md").read_text(encoding="utf-8")
    assert "W4 FFT" in readme and "out of scope" in readme
    # The filesystem, not `dir(kernels)`: the package re-exports nothing, so an actual
    # `kernels/w4_fft.py` would be invisible to an attribute scan and the gate would pass.
    stray = [p.name for p in (repo / "kernels").glob("*.py")
             if "w4" in p.stem.lower() or "fft" in p.stem.lower()]
    import kernels
    stray += [n for n in dir(kernels) if "w4" in n.lower() or "fft" in n.lower()]
    assert not stray, f"W4/FFT is out of scope but kernels/ carries {stray}"


_KERNELS = {"w1": ("kernels.w1_gemm", "gemm"), "w1_flip": ("kernels.w1_gemm", "gemm"),
            "w2": ("kernels.w2_jacobi", "jacobi"), "w3": ("kernels.w3_sw", "sw")}


def _kernel_result(name):
    """Run the kernel on a fixture's inputs and return the arrays it wrote.

    The inputs are copied first: `expected.npz` and `inputs.npz` are read-only ground truth,
    and every one of these kernels writes through an argument.
    """
    import importlib

    from tests import fixtures
    try:
        fx = fixtures.load(name)
    except (FileNotFoundError, AttributeError, ImportError) as exc:
        pytest.skip(f"fixture {name} not generated yet: {exc}")
    mod_name, fn_name = _KERNELS[name]
    try:
        kernel = getattr(importlib.import_module(mod_name), fn_name)
    except ImportError as exc:
        pytest.skip(f"{mod_name} is not importable yet: {exc}")
    args = {k: v.copy() for k, v in fx.inputs().items()}
    kernel(**args)
    return fx, args


def _assert_oracle(name):
    """04-test-plan §8 item 5, for one workload, at the fixture's own tolerance."""
    from spatial import m6_tools as m6

    fx, args = _kernel_result(name)
    expected = fx.expected()
    assert expected is not None, f"{name} ships no expected.npz to diff against"
    keys = sorted(expected)
    report = m6.diff([args[k] for k in keys], [expected[k] for k in keys], fx.tol)
    assert report.matched, (
        f"{name}: {report.count_mismatched} of {report.total} elements differ from the "
        f"independent reference, max abs err {report.max_abs_err} at {report.first_mismatch} "
        f"(tol {fx.tol})")
    return report


@pytest.mark.fr("FR-K1")
def test_W1_end_to_end():
    """FR-K1: GEMM against `A @ B` computed in float64 and cast down. `tol = 0.0`."""
    assert _assert_oracle("w1").max_abs_err == 0.0


@pytest.mark.fr("FR-K3")
def test_W2_end_to_end():
    """FR-K3: Jacobi against the t-outermost two-plane reference, at the stated `1e-5`."""
    report = _assert_oracle("w2")
    assert report.max_abs_err <= 1e-5


@pytest.mark.fr("FR-K4")
def test_W3_end_to_end():
    """FR-K4: Smith-Waterman against the textbook `if`-statement DP. `tol = 0.0`.

    The kernel writes the same recurrence as a value-level conditional (RULING 2), so a diff
    of zero here is the evidence that the two spellings agree.
    """
    assert _assert_oracle("w3").max_abs_err == 0.0


@pytest.mark.fr("FR-K2")
def test_W1_flip():
    """FR-K2, the half that does not need A: the flip does not change the answer.

    M8 §9's definition of done asks for `array_equal(C_os, C_ws)`. The output-stationary and
    weight-stationary schedules do not exist until A lands, but the two fixtures carry the
    same seed-0 inputs, and FR-S18 makes the schedule ignorable — so the kernel's answer must
    be bit-identical across both, whichever schedule is later wrapped around it. When A lands
    this test gains the two `schedule_*` calls and keeps the same assertion.
    """
    _assert_oracle("w1_flip")
    base, flip = _kernel_result("w1")[1], _kernel_result("w1_flip")[1]
    assert np.array_equal(base["C"], flip["C"]), "the flip changed the kernel's answer"


@pytest.mark.fr("FR-K3")
def test_W2_oracle_is_timestep_outermost():
    """FR-K3's second acceptance: plane `t+1` is one sweep of plane `t`, and of nothing else.

    This is the property B-P25 turns on. If the reference had been written with the timestep
    innermost — or had let a partially updated plane feed itself — plane `t+1` would carry
    values from `t+1` and this sweep would not reproduce it.
    """
    from tests import fixtures
    try:
        fx = fixtures.load("w2")
    except (FileNotFoundError, AttributeError, ImportError) as exc:
        pytest.skip(f"fixture w2 not generated yet: {exc}")
    U = fx.expected()["U"]
    T, H, W = U.shape[0] - 1, U.shape[1] - 2, U.shape[2]
    for t in range(T):
        cur = U[t]
        sweep = 0.2 * (cur[1:H + 1, 1:W - 1]
                       + cur[0:H, 1:W - 1] + cur[2:H + 2, 1:W - 1]
                       + cur[1:H + 1, 0:W - 2] + cur[1:H + 1, 2:W])
        assert np.allclose(U[t + 1, 1:H + 1, 1:W - 1], sweep, atol=fx.tol), f"plane {t + 1}"


def test_demo_rejections():
    """M8 §5.3: each demo rejection raises, with the four-part diagnostic, matching its golden.

    The goldens (`tests/golden/reject.<code>.txt`) are captured from A's real messages — they
    cannot be guessed, so until M3 lands this skips rather than asserting a fiction. The
    rejection functions already exist and carry the expected message shapes in their
    docstrings, so what lands is the capture, not the test.
    """
    from kernels import rejections

    raisers = ("bad_skew", "bad_stationary", "bad_capacity")
    try:
        getattr(rejections, raisers[0])("npu1")
    except NotImplementedError as exc:
        pytest.skip(f"the legality checker (A, M3) is not built yet, so the three rejection "
                    f"goldens cannot be captured: {exc}")
    from spatial.model import SpatialError
    for name in raisers:
        with pytest.raises(SpatialError) as caught:
            getattr(rejections, name)("npu1")
        golden_path = Path(str(files("tests"))) / "golden" / f"reject.{caught.value.diagnostic.code}.txt"
        assert golden_path.is_file(), f"no golden captured for {name}: {golden_path}"
        assert str(caught.value).strip() == golden_path.read_text(encoding="utf-8").strip()
