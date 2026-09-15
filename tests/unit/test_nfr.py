"""The non-functional lint tests B owes for M4, M5 and M6's off-device half.

Spec: `01-requirements.md` §4 (NFR-2, NFR-5), FR-S20, `03-lld-M7-tests.md` §7.8. **Written by B
at P7; Person C owns NFR-1…NFR-7 as a set** — C's budget/lint/network/corpus tests
are appended below, and
`test_no_air_import_until_build` (the M2 test of that name, FR-S20's own acceptance) is A's.
What is here covers exactly the four modules B wrote.

`test_NFR5_docstrings` reads NFR-5 as *module-level public names*: a name with no leading
underscore, bound in the module that defines it, that can carry a runtime `__doc__` — a function
or a class. A module-level constant cannot: Python keeps no attribute docstrings at runtime, so
the `\"\"\"…\"\"\"` under `PIN` or `PIPELINES` is a source convention this test cannot see. That
half of NFR-5 would need an AST lint, which is C's `test_NFR4_no_bare_raise` shape and is
recorded in `design/PROGRESS-B.md` §P7 rather than written here.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.determinism import import_closure, in_fresh_process

B_MODULES = ("spatial.m4_mapping", "spatial.m4_selfcheck", "spatial.m5_emit",
             "spatial.m6_tools")
"""The four modules B wrote. `spatial.model` is M0 (frozen, A's to change) and is covered by
`tests/unit/test_m0_model.py`; `m1`/`m2`/`m3` are still stubs."""

ALLOWED_THIRD_PARTY = frozenset({"spatial", "numpy"})
"""NFR-2's runtime closure. `numpy` is the one third-party import allowed at module scope;
`mlir_air` (`air`), `mlir_aie` and `llvm-aie` are the pinned *toolchain*, reached only through
`m5_emit.emit`'s and `m6_tools`' lazy in-function imports (FR-S20)."""


@pytest.mark.parametrize("module_name", B_MODULES)
def test_NFR5_docstrings(module_name):
    """Every public module-level function and class has a non-empty docstring, and so does the
    module."""
    module = importlib.import_module(module_name)
    assert (module.__doc__ or "").strip(), f"{module_name} has no module docstring"

    public = {name: obj for name, obj in vars(module).items()
              if not name.startswith("_")
              and getattr(obj, "__module__", None) == module_name
              and (inspect.isfunction(obj) or inspect.isclass(obj))}
    assert public, f"{module_name} exports no public callable — the lint would be vacuous"
    undocumented = sorted(name for name, obj in public.items()
                          if not (obj.__doc__ or "").strip())
    assert not undocumented, (
        f"{module_name}: {len(undocumented)} of {len(public)} public name(s) have no "
        f"docstring: {undocumented}")


@pytest.mark.parametrize("module_name", B_MODULES)
def test_NFR2_deps(module_name):
    """Importing any of B's modules pulls in the standard library and nothing but `numpy`.

    Measured in a **fresh interpreter** per module (`in_fresh_process`), because the answer in
    this one is already spoiled: pytest, the fixtures and the other tests have imported plenty.
    The child imports only `tests.helpers.determinism`, which is stdlib-only.
    """
    extra = set(in_fresh_process(import_closure, module_name)) - ALLOWED_THIRD_PARTY
    assert not extra, (
        f"{module_name} imports {sorted(extra)} at module scope; NFR-2 allows the standard "
        f"library plus numpy, with the toolchain reached lazily inside a function")


@pytest.mark.parametrize("module_name", B_MODULES)
@pytest.mark.fr("FR-S20")
def test_no_air_import_B_modules(module_name):
    """FR-S20 for B's four modules: **no** `air` is imported until `build()` runs.

    The M2 test named `test_no_air_import_until_build` is Person A's and covers the schedule
    surface; this is the same property for the mapper, the self-check and the emitter. `m5_emit`
    and `m6_tools` do import `air` — inside `emit()` and inside the tool wrappers — and that is
    precisely what this asserts is *not* module scope.
    """
    closure = in_fresh_process(import_closure, module_name)
    assert "air" not in closure and "mlir_air" not in closure, (
        f"{module_name} imports the toolchain at module scope: {closure}")


def test_NFR3_budget():
    """NFR-3 machinery present **and armed**: the budgets, the hooks, and the gate's guard.

    The last assertions are the ones that matter. `_is_default_run` decides whether the 180 s
    total is checked at all, and an earlier version read `config.option.markexpr` — which
    `addopts` always fills in, so it answered False for every run and the gate never fired.
    Asserting the machinery exists is not the same as asserting it runs.
    """
    import tests.conftest as conftest
    assert conftest.NFR3_TOTAL_BUDGET == 180
    assert conftest.NFR3_PER_TEST_BUDGET == 3.0
    assert hasattr(conftest, "pytest_sessionfinish")
    assert hasattr(conftest, "pytest_runtest_makereport")

    def session(*args):
        return SimpleNamespace(config=SimpleNamespace(
            invocation_params=SimpleNamespace(args=args)))

    assert conftest._is_default_run(session())               # bare `pytest`
    assert conftest._is_default_run(session("-ra", "-q"))    # CI step 7's form
    assert not conftest._is_default_run(session("-m", "slow"))
    assert not conftest._is_default_run(session("-k", "golden"))
    assert not conftest._is_default_run(session("tests/unit"))


def test_NFR3_exempts_every_requires_marker():
    """Every `requires_*` marker is exempt from the §3.6 per-test budget.

    The budget measures our own code; a test gated on an external tool measures the tool. The
    rule was broken once: `requires_ttsim` was added to `_MARKERS` and not to `_EXEMPT_MARKS`,
    and two ttsim tests that take ~48 s each were force-failed by the 3 s budget on a suite
    that has no other way to run them.
    """
    import tests.conftest as conftest
    gated = sorted(name for name, _doc in conftest._MARKERS if name.startswith("requires_"))
    assert gated, "no requires_* marker declared — the lint would be vacuous"
    missing = [name for name in gated if name not in conftest._EXEMPT_MARKS]
    assert not missing, (
        f"{missing} are declared in conftest._MARKERS but missing from _EXEMPT_MARKS, so a "
        f"test carrying one would be failed on wall clock by pytest_runtest_makereport")


def test_NFR3_budget_fires(monkeypatch):
    """The gate **bites**: an over-budget default run leaves a non-zero exit status.

    Driven by calling `pytest_sessionfinish` with a fabricated session, because the real one
    is under budget — 17 s of 180 — and a test that only ever sees a passing run cannot tell a
    working gate from a disabled one. That is exactly how the `markexpr` bug survived.
    """
    import tests.conftest as conftest

    over = {"tests/fake.py::test_slow": conftest.NFR3_TOTAL_BUDGET + 1.0}
    monkeypatch.setattr(conftest, "DURATIONS", over)
    session = SimpleNamespace(exitstatus=0, config=SimpleNamespace(
        invocation_params=SimpleNamespace(args=())))
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1, "an over-budget default run must fail the session"

    under = {"tests/fake.py::test_quick": 1.0}
    monkeypatch.setattr(conftest, "DURATIONS", under)
    ok = SimpleNamespace(exitstatus=0, config=SimpleNamespace(
        invocation_params=SimpleNamespace(args=())))
    conftest.pytest_sessionfinish(ok, 0)
    assert ok.exitstatus == 0

    narrowed = SimpleNamespace(exitstatus=0, config=SimpleNamespace(
        invocation_params=SimpleNamespace(args=("-m", "slow"))))
    monkeypatch.setattr(conftest, "DURATIONS", over)
    conftest.pytest_sessionfinish(narrowed, 0)
    assert narrowed.exitstatus == 0, "a narrowed run is not measured against the whole budget"


_SKIP_REASON_MIN = 20
"""I-8: "one sentence a non-expert can read". Twenty characters is the floor that rejects
`skip("todo")` and `skip("no device")` without pretending to grade prose."""


def _skip_reason_length(node, name):
    """Characters the author typed in a `skip(...)` reason.

    `None` means "not judgeable here", and covers two different things: a call with no reason
    at all, which the caller reports, and a reason computed at run time — `str(exc)`, or an
    f-string carrying a diagnostic — which is taken on trust, because its length is not in the
    source. The lint exists to catch `skip("todo")`, not to second-guess a rendered
    diagnostic; `has_reason` is what separates the two.
    """
    args = [kw.value for kw in node.keywords if kw.arg == "reason"]
    if name == "skip":  # `skipif`'s first positional is the condition, never the reason
        args += [a for a in node.args if not isinstance(a, ast.Starred)]
    if not args:
        return False, None
    value = args[0]
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return True, len(value.value)
    return True, None


def test_skips_are_explained():
    """I-8 (M7 §4): every `skip(` under `tests/` states why, in a sentence, not a word.

    04-test-plan §8 item 10 makes the skip list part of the deliverable — it is read aloud at
    D7 and it is what the honest-limits slide says about the absent device. A skip with no
    reason, or a one-word one, is a hole in that list.
    """
    root = Path(__file__).resolve().parents[1]
    bad = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name not in ("skip", "skipif"):
                continue
            has_reason, length = _skip_reason_length(node, name)
            where = f"{path.relative_to(root.parent)}:{node.lineno}"
            if not has_reason:
                bad.append(f"{where}: `{name}(` passes no reason at all")
            elif length is not None and length < _SKIP_REASON_MIN:
                bad.append(f"{where}: reason is {length} characters, under "
                           f"{_SKIP_REASON_MIN} — say what is missing and what fixes it")
    assert not bad, "\n".join(bad)


ASSERT_EXEMPT: frozenset[str] = frozenset()
"""Files in `spatial/` whose `assert` statements NFR-4 tolerates — **empty**, 2026-09-15.

The exemption the rule was written expecting is M0's `__post_init__` machinery, which enforces
I01-I78 on a path NFR-4 does not govern (`spatial/model.py`'s own docstring: those are
programmer errors, not user diagnostics). It needs no entry: `_need` raises `TypeError` or
`ValueError` with the class, the field and the value, and M0 contains no `assert` at all. Keep
it empty. An entry here is a promise that no user input can reach that statement, and `-O`
deletes every one of them.
"""


def test_NFR4_no_bare_raise():
    """AST lint over `spatial/`: no bare `raise Exception/AssertionError/RuntimeError`, **and
    no `assert`** (2026-09-15).

    `assert` is the same defect as a bare `AssertionError` with a worse failure mode: it says
    nothing a user can act on, it names no `06-interfaces.md` §6.3 code, and `python -O` removes
    the check entirely, so a condition the code relied on silently stops being checked. A
    condition a user can reach is a `Diagnostic`; one only the compiler can reach is the
    `internal:` spelling of B-P23. Neither is an `assert`. `ASSERT_EXEMPT` above is the escape
    hatch and is empty.
    """
    root = Path(__file__).resolve().parents[2] / "spatial"
    if not root.is_dir():
        pytest.skip("spatial/ not present")
    bad = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                func = node.exc.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name in ("Exception", "AssertionError", "RuntimeError"):
                    bad.append(f"{path.name}:{node.lineno} bare raise {name}")
            elif isinstance(node, ast.Assert) and path.name not in ASSERT_EXEMPT:
                bad.append(f"{path.name}:{node.lineno} assert (NFR-4: `-O` deletes it; raise a "
                           f"Diagnostic, or the `internal:` spelling of B-P23)")
    assert not bad, f"bare raises or asserts on user paths: {bad}"


def test_NFR6_no_network():
    """AF_INET `connect` **and `connect_ex`** raise under the gate; AF_UNIX is left alone.

    `connect_ex` is the same syscall reporting failure as a return code instead of an
    exception. It was not overridden, so any caller that preferred it — several stdlib
    clients do — walked straight past the gate and reached the network.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError):
            s.connect(("127.0.0.1", 9))
        with pytest.raises(RuntimeError):
            s.connect_ex(("127.0.0.1", 9))
    finally:
        s.close()

    unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:  # XRT talks over AF_UNIX; the gate must not touch it
        with pytest.raises(OSError):  # refused by the kernel, not by us
            unix.connect("/nonexistent/spatial-dsl.sock")
    finally:
        unix.close()


def test_NFR7_all_errors_are_spatial():
    """Every failure in the negative corpus is a SpatialError."""
    neg = Path(__file__).resolve().parents[1] / "negative"
    mods = [p.stem for p in neg.glob("test_*.py")]
    if not mods:
        pytest.skip("tests/negative has no corpus modules yet")
    from spatial.model import SpatialError
    failures = 0
    for mod in mods:
        m = importlib.import_module(f"tests.negative.{mod}")
        for name in dir(m):
            if name.startswith("raises_") or name.startswith("corpus_"):
                try:
                    getattr(m, name)()
                except SpatialError:
                    failures += 1
                except Exception as exc:  # noqa: BLE001 — the assertion is the type
                    pytest.fail(f"{mod}.{name} raised {type(exc).__name__}, not SpatialError")
    assert failures > 0, "corpus ran but raised nothing"
