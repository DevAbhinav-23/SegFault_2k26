"""The negative corpus's shared machinery. Owner: Person A. Spec: `04-test-plan.md` §5.

A corpus module (`tests/negative/test_*.py`) holds plain, argument-free functions, each of
which **raises**:

* `raises_<code>()` — the canonical case for one catalogue code, exactly one per code;
* `corpus_<code>__<label>()` — a further case for the same code, because the per-module
  negative tables of `03-lld-M1-frontend.md` §7, `03-lld-M2-schedule.md` §7 and
  `03-lld-M3-checker.md` §7 carry more rows than there are codes.

**The code is spelled in the name**: lower-cased, with `-` written `_`, and everything from a
double underscore on is a label for humans. That is the whole naming contract — `code_of`
inverts it, so no table can go stale.

Three consumers read these functions:

* `tests/unit/test_nfr.py::test_NFR7_all_errors_are_spatial` calls every one and asserts the
  failure is a `SpatialError` (NFR-7);
* each corpus module's own `test_*_codes` asserts every function raises exactly the code its
  name spells, as its stage's error class;
* `tests/negative/test_catalogue.py` asserts the union of the codes raised is exactly
  `spatial.model.CATALOGUE` (FR-D3) and that every diagnostic obeys §6.1's schema (FR-D1).

Every function is self-contained and deterministic: no network, no device, and no toolchain
binary — the `TOOL-*` cases patch `spatial.m6_tools` with `unittest.mock.patch` inside the
function, so nothing outside it is disturbed.
"""

from __future__ import annotations

from types import ModuleType
from typing import Callable

from spatial.model import CATALOGUE, SpatialError

PREFIXES = ("raises_", "corpus_")
"""The two function-name prefixes a corpus module exposes. `test_NFR7_all_errors_are_spatial`
hard-codes the same two."""

_CACHE: dict[str, tuple[tuple[str, SpatialError], ...]] = {}


def code_of(name: str) -> str:
    """`corpus_l1_capacity__demo` → `L1-CAPACITY`; `raises_balance` → `BALANCE`."""
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):].split("__")[0].upper().replace("_", "-")
    raise ValueError(f"{name!r} is not a corpus function name; prefixes are {PREFIXES}")


def cases(module: ModuleType) -> tuple[tuple[str, Callable[[], object]], ...]:
    """Every corpus function of `module`, by name, in a stable order."""
    return tuple(sorted((name, getattr(module, name)) for name in dir(module)
                        if name.startswith(PREFIXES)))


def collect(module: ModuleType) -> tuple[tuple[str, SpatialError], ...]:
    """Run every corpus function of `module` and return `(name, the error it raised)`.

    Cached per module: `test_D1_schema` and `test_D3_catalogue_complete` both want the whole
    corpus, and running it twice buys nothing but wall clock against NFR-3's per-test budget.
    A function that raises nothing, or raises something that is not a `SpatialError`, is an
    `AssertionError` here rather than a silent gap in the corpus.
    """
    if module.__name__ in _CACHE:
        return _CACHE[module.__name__]
    out = []
    for name, fn in cases(module):
        try:
            fn()
        except SpatialError as exc:
            out.append((name, exc))
            continue
        except Exception as exc:  # noqa: BLE001 — the assertion is the type
            raise AssertionError(
                f"{module.__name__}.{name} raised {type(exc).__name__}, not a SpatialError "
                f"(NFR-7): {exc}") from exc
        raise AssertionError(f"{module.__name__}.{name} raised nothing; a corpus function is "
                             f"a rejection, not a schedule that works")
    _CACHE[module.__name__] = tuple(out)
    return _CACHE[module.__name__]


def assert_codes(module: ModuleType, error_type: type) -> None:
    """Every corpus function of `module` raises the code its name spells, as `error_type`.

    Both halves matter. The code keeps a case honest — a rename cannot quietly re-point a
    function at a neighbouring code — and the exception class keeps the *stage* honest, since
    `06-interfaces.md` §6.2 gives each stage its own subclass and `CATALOGUE` its own stage.
    """
    found = cases(module)
    assert found, f"{module.__name__} exposes no corpus function"
    for name, exc in collect(module):
        code = code_of(name)
        assert code in CATALOGUE, f"{name} spells {code}, which is not a catalogue code"
        assert exc.diagnostic.code == code, (
            f"{module.__name__}.{name} raised {exc.diagnostic.code}, not the {code} its name "
            f"spells")
        assert type(exc) is error_type, (
            f"{module.__name__}.{name} raised {type(exc).__name__}; the stage's error class "
            f"is {error_type.__name__}")
