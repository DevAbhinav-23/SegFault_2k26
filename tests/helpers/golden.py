"""Golden files and the single `--update-goldens` switch. Spec: design/03-lld-M7-tests.md §3.1.

Registered as a pytest plugin by `tests/conftest.py`, so `pytest_terminal_summary` runs.

Written by B at P0b to unblock M0; C owns this file.
"""

from __future__ import annotations

import difflib
import json
from importlib.resources import files
from pathlib import Path

import pytest

GOLDEN_DIR = Path(str(files("tests"))) / "golden"
"""Where the golden files live (design/06-interfaces.md §8)."""

UPDATE_GOLDENS = False
"""Set once, by `tests/conftest.py`'s `pytest_configure`."""

UPDATED: list[tuple[str, str]] = []
"""`(name, diff summary)` for every golden this run rewrote."""


def canonical_json(obj: object) -> str:
    """The canonical JSON of design/06-interfaces.md §8: sorted keys, 2 spaces, trailing \\n."""
    return json.dumps(obj, sort_keys=True, indent=2) + "\n"


def _diff(old: str | None, new: str, name: str) -> str:
    if old is None:
        return f"{name}: new file, {len(new.splitlines())} lines"
    changed = sum(1 for line in difflib.unified_diff(old.splitlines(), new.splitlines())
                  if line[:1] in "+-" and line[:3] not in ("+++", "---"))
    return f"{name}: {changed} changed line(s)"


def assert_golden(name: str, actual: object, *, kind: str) -> None:
    """Compare `actual` against the golden `name`; `kind` is `"text"` or `"json"`."""
    if kind not in ("text", "json"):
        raise ValueError(f"kind must be 'text' or 'json', got {kind!r}")
    path = GOLDEN_DIR / name
    payload = actual if kind == "text" else canonical_json(actual)
    if not isinstance(payload, str):
        raise TypeError(f"a text golden needs str, got {type(payload).__name__}")
    if UPDATE_GOLDENS:
        old = path.read_text(encoding="utf-8") if path.exists() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        if old != payload:
            UPDATED.append((name, _diff(old, payload, name)))
        return
    if not path.exists():
        pytest.fail(f"golden {name} does not exist; run `pytest --update-goldens` and "
                    f"say in the commit message why it is new")
    expected = path.read_text(encoding="utf-8")
    if kind == "text":
        assert payload == expected, "\n".join(difflib.unified_diff(
            expected.splitlines(), payload.splitlines(),
            fromfile=f"golden/{name}", tofile="actual", lineterm=""))
    else:
        assert json.loads(payload) == json.loads(expected), f"golden/{name} differs"


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Report what `--update-goldens` rewrote."""
    if UPDATE_GOLDENS:
        terminalreporter.write_line(
            f"--update-goldens rewrote {len(UPDATED)} golden(s): "
            + "; ".join(summary for _, summary in UPDATED))
