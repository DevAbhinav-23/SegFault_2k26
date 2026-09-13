"""The non-functional lint tests B owes for M4, M5 and M6's off-device half.

Spec: `01-requirements.md` §4 (NFR-2, NFR-5), FR-S20, `03-lld-M7-tests.md` §7.8. **Written by B
at P7; Person C owns NFR-1…NFR-7 as a set** — `test_NFR3_budget`, `test_NFR4_no_bare_raise`,
`test_NFR6_no_network` and `test_NFR7_all_errors_are_spatial` are C's and are not here, and
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

import importlib
import inspect

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
