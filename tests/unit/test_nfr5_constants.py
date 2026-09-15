"""NFR-5 for module **constants** — the half `test_NFR5_docstrings` cannot see. Closes B-P29.

`tests/unit/test_nfr.py::test_NFR5_docstrings` reads every public module-level name's runtime
`__doc__`, which settles functions and classes and can say nothing at all about a constant:
Python keeps no attribute docstring at runtime, so the `\"\"\"…\"\"\"` written under `PIN`,
`PIPELINES` or `CATALOGUE` is a **source** convention. B-P29 asked for the AST lint of the same
shape as `test_NFR4_no_bare_raise`, and this is it.

**The rule, exactly as the codebase writes it.** A module-level assignment whose target is an
UPPER_CASE name (a leading underscore allowed — a private constant is still a constant) must be
followed by a string-literal expression. Constants written as a **run** of consecutive
assignments share the one docstring that ends the run, because that is how the source already
groups them: `_READ_ALIGN` and `_WRITE_ALIGN` in `spatial/m5tt_emit.py` are one measured fact
about the NoC and carry one docstring between them, and splitting it would make the file worse,
not better. So a run is documented when a docstring follows its **last** member.

**What the exemption list is.** `KNOWN_UNDOCUMENTED` names the constants that were outside the
rule when it was written, 2026-09-15, one entry per constant with the owner who can close it.
The test asserts in both directions: no new offender may appear, and an entry that has been
documented must be **deleted** from the list — so the list shrinks and can never quietly become
the rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

SPATIAL = Path(__file__).resolve().parents[2] / "spatial"

KNOWN_UNDOCUMENTED: frozenset[str] = frozenset()
"""**Empty since 2026-09-15**, and the lint below keeps it that way.

It held the 16 constants that were outside the rule when this lint was written (B-P29) — 9 of
A's in `m1_frontend`, `m2_schedule` and `m3_legality`, 6 in the frozen `model.py`, 1 of C's in
`m6_tools` — each listed rather than silently excluded so the debt had an owner. All 16 were
documented in the closing pass; documenting a constant changes no field of any record, so
`model.py` stayed frozen in the sense `design/00-README.md` §4 means.

The two tests below are a ratchet in both directions: a new undocumented constant anywhere in
`spatial/` fails the first, and an entry here that has since been documented fails the second
until its line is deleted. With the set empty the second is vacuous, which is the point.
"""


def _constant_names(node: ast.stmt) -> list[str]:
    """The UPPER_CASE names this statement binds at module level, or `[]`."""
    if isinstance(node, ast.Assign):
        bound = [t.id for t in node.targets if isinstance(t, ast.Name)]
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        bound = [node.target.id]
    else:
        return []
    return [name for name in bound if name.lstrip("_") and name.lstrip("_").isupper()]


def _is_docstring(node: ast.stmt | None) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def _scan() -> tuple[set[str], set[str]]:
    """`(every constant, the undocumented ones)` over `spatial/*.py`, as `<file>:<NAME>`."""
    every: set[str] = set()
    undocumented: set[str] = set()
    for path in sorted(SPATIAL.glob("*.py")):
        body = ast.parse(path.read_text(encoding="utf-8")).body
        index = 0
        while index < len(body):
            if not _constant_names(body[index]):
                index += 1
                continue
            run = []
            while index < len(body) and _constant_names(body[index]):
                run += [f"{path.name}:{name}" for name in _constant_names(body[index])]
                index += 1
            every |= set(run)
            if not (index < len(body) and _is_docstring(body[index])):
                undocumented |= set(run)
    return every, undocumented


def test_NFR5_every_module_constant_carries_its_docstring():
    """Every module constant in `spatial/` carries its docstring — `KNOWN_UNDOCUMENTED` is
    empty, so this is now the whole rule with no exception."""
    every, undocumented = _scan()
    assert len(every) >= 80, (
        f"the walk found only {len(every)} module constants in {SPATIAL}; it is not linting "
        f"what it claims to lint")
    new = sorted(undocumented - KNOWN_UNDOCUMENTED)
    assert not new, (
        f"{len(new)} module constant(s) of {len(every)} carry no docstring, and NFR-5's "
        f"convention is the string literal under the assignment: {new}")


def test_NFR5_the_exemption_list_only_shrinks():
    """Every exemption is still a real offender — document one and delete its line.

    One test over the whole set rather than one per entry: the set is empty, and a
    `parametrize` over an empty set is a **skip**, which would read in the suite's tally as a
    gap rather than as the debt being paid off.
    """
    every, undocumented = _scan()
    gone = sorted(entry for entry in KNOWN_UNDOCUMENTED if entry not in every)
    assert not gone, f"{gone} no longer exist; delete them from KNOWN_UNDOCUMENTED"
    documented = sorted(entry for entry in KNOWN_UNDOCUMENTED if entry not in undocumented)
    assert not documented, (
        f"{documented} now carry their docstring — delete them from KNOWN_UNDOCUMENTED, which "
        f"is a debt list and must shrink")
