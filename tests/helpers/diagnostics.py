"""The four-part negative-test assertion. Spec: design/03-lld-M7-tests.md §3.2.

`CATALOGUE` is parsed out of `design/06-interfaces.md` §6.3 at import, so the document is the
single source: a code added to the package but not the table fails the gate.

Written by B at P0b to unblock M0; C owns this file.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

_REPO = Path(str(files("tests"))).resolve().parent
_SPEC = _REPO / "design" / "06-interfaces.md"
_ROW = re.compile(r"^\|\s*`(?P<code>[A-Z0-9][A-Z0-9-]*)`\s*\|\s*(?P<stage>[a-z]+)\s*\|")
_EXPECTED_ROWS = 43


def _parse_catalogue() -> dict[str, str]:
    text = _SPEC.read_text(encoding="utf-8")
    start = text.index("### 6.3")
    section = text[start:].split("\n---", 1)[0]
    table = {}
    for line in section.splitlines():
        match = _ROW.match(line)
        if match:
            table[match["code"]] = match["stage"]
    if len(table) != _EXPECTED_ROWS:
        raise AssertionError(
            f"{_SPEC}: §6.3 parsed to {len(table)} codes, expected {_EXPECTED_ROWS}")
    return table


CATALOGUE: dict[str, str] = _parse_catalogue()
"""Every code of design/06-interfaces.md §6.3 mapped to its stage, parsed from the document."""

RAISED_CODES: set[str] = set()
"""Every code any test actually raised; `test_D3_catalogue_complete` reads it (FR-D3)."""


def _is_json_serialisable(value: object) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def assert_diagnostic(excinfo, *, code, clause=None, mentions=(), details_keys=()) -> None:
    """Assert the code and all four message parts of a raised `SpatialError` (M7 §3.2)."""
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == code, f"expected {code}, got {diagnostic.code}"
    assert diagnostic.code in CATALOGUE, f"{diagnostic.code} is not in 06-interfaces.md §6.3"
    assert diagnostic.stage == CATALOGUE[diagnostic.code]
    assert diagnostic.reason and not diagnostic.reason.endswith(".")
    assert diagnostic.fix
    if clause is not None:
        assert diagnostic.clause == clause
    if CATALOGUE[diagnostic.code] == "grammar":
        assert diagnostic.location is not None
    else:
        assert diagnostic.clause is not None
    haystack = str(diagnostic.reason) + str(diagnostic.fix) + str(dict(diagnostic.details))
    for token in mentions:
        assert str(token) in haystack, f"{token!r} is not mentioned in {haystack!r}"
    for key in details_keys:
        assert key in diagnostic.details, f"details has no {key!r}"
    assert _is_json_serialisable(dict(diagnostic.details))
    RAISED_CODES.add(diagnostic.code)
