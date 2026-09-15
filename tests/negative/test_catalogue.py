"""Level N — FR-D1 and FR-D3, the two gates the whole corpus exists for. Owner: Person A.

`test_D3_catalogue_complete` closes the catalogue **both ways** (`04-test-plan.md` §5, §8 item
2): every one of the 43 codes of `06-interfaces.md` §6.3 is raised by at least one corpus
function, and no code the package can raise is missing from the table. The first direction runs
the corpus; the second is an AST scan of `spatial/*.py`, because a code that no test happens to
reach would otherwise slip in unnoticed.

`test_D1_schema` asserts the §6.1 schema over **every diagnostic the corpus raises** — one
parametrised assertion rather than one per code, which is what FR-D1's acceptance asks for.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from spatial.model import CATALOGUE
from tests.negative import (_corpus, test_clause, test_emission, test_grammar, test_legality,
                            test_mapping, test_toolchain)

MODULES = (test_grammar, test_clause, test_legality, test_mapping, test_emission,
           test_toolchain)
"""Every corpus module, in stage order. `tests/unit/test_nfr.py::test_NFR7_all_errors_are_spatial`
finds the same set by globbing `tests/negative/test_*.py`, so a module added there and left out
here fails `test_D3_every_corpus_module_is_listed` below."""

_HELPERS = ("_fail", "_raise", "_internal", "_error")
"""The diagnostic constructors of `spatial/`. Only the overloads whose **first parameter is
named `code`** carry a code there — `m4_mapping._fail(reason, fix, ...)` hard-codes
`PROTOCOL-UNSUPPORTED` inside and takes a reason first, and `m4_selfcheck._internal(plan,
invariant, reason)` takes a plan — so the scan reads each module's own `def` rather than
assuming one signature."""

SPATIAL = Path(__file__).resolve().parents[2] / "spatial"


def corpus() -> tuple[tuple[str, object], ...]:
    """`(qualified name, the error it raised)` for every corpus function of every module."""
    out = []
    for module in MODULES:
        for name, exc in _corpus.collect(module):
            out.append((f"{module.__name__.rsplit('.', 1)[-1]}.{name}", exc))
    return tuple(out)


def _code_literals() -> list[tuple[str, int, str]]:
    """Every error code spelled as a literal in `spatial/`: `(file, line, code)`."""
    out: list[tuple[str, int, str]] = []
    for path in sorted(SPATIAL.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        code_first = {node.name for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name in _HELPERS
                      and node.args.args and node.args.args[0].arg == "code"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if (name in code_first and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                out.append((path.name, node.lineno, node.args[0].value))
            if name == "Diagnostic":
                for keyword in node.keywords:
                    if keyword.arg == "code" and isinstance(keyword.value, ast.Constant):
                        out.append((path.name, node.lineno, keyword.value.value))
    return out


@pytest.mark.fr("FR-D3")
def test_D3_catalogue_complete():
    """Direction 1: the corpus raises every catalogue code. Direction 2: nothing else exists.

    The ruling of 2026-09-15 is that **every code must be reachable** — a catalogue entry
    nothing can raise is a defect, not a documentation row — so this fails rather than skips
    when one is missing.
    """
    raised = {exc.diagnostic.code for _, exc in corpus()}
    missing = sorted(set(CATALOGUE) - raised)
    assert not missing, (
        f"{len(missing)} of {len(CATALOGUE)} catalogue codes are raised by nothing: {missing}")
    assert raised == set(CATALOGUE), sorted(raised - set(CATALOGUE))

    literals = _code_literals()
    assert len(literals) > 40, "the AST scan found almost no call sites; it is not scanning"
    unlisted = sorted({(f, line, code) for f, line, code in literals if code not in CATALOGUE})
    assert not unlisted, (
        "these call sites raise a code design/06-interfaces.md §6.3 does not list: "
        + ", ".join(f"{f}:{line} {code}" for f, line, code in unlisted))
    # ...and every catalogue code is spelled somewhere in the package, not only in the table
    assert {code for _, _, code in literals} == set(CATALOGUE)


@pytest.mark.fr("FR-D3")
def test_D3_every_corpus_module_is_listed():
    """`MODULES` above and the glob `test_NFR7_all_errors_are_spatial` uses must agree."""
    on_disk = {path.stem for path in Path(__file__).parent.glob("test_*.py")}
    listed = {module.__name__.rsplit(".", 1)[-1] for module in MODULES}
    assert on_disk - listed == {"test_catalogue"}, sorted(on_disk - listed)


@pytest.mark.fr("FR-D1")
def test_D1_schema():
    """FR-D1: every rejection carries a populated `Diagnostic`, and renders as four parts.

    `clause` is asserted per stage rather than always: `06-interfaces.md` §6.1 requires it of a
    `clause` or `legality` diagnostic and requires a `location` of a `grammar` one (M0 invariant
    I71), and M1 has no clause to name because the user wrote a kernel, not a clause. What is
    asserted here is that the only diagnostics without a clause are the grammar ones.
    """
    cases = corpus()
    assert len(cases) >= len(CATALOGUE), f"only {len(cases)} corpus cases"
    for name, exc in cases:
        diagnostic = exc.diagnostic
        assert diagnostic.code in CATALOGUE, f"{name}: {diagnostic.code} is not in the table"
        assert diagnostic.stage == CATALOGUE[diagnostic.code], name
        assert diagnostic.reason.strip(), f"{name} has an empty reason"
        assert not diagnostic.reason.endswith("."), f"{name}'s reason ends in a period"
        assert diagnostic.fix.strip(), f"{name} has an empty fix"
        if diagnostic.stage == "grammar":
            assert diagnostic.clause is None, f"{name}: a grammar diagnostic names no clause"
            assert diagnostic.location is not None, f"{name} violates I71"
        else:
            assert diagnostic.clause and diagnostic.clause.strip(), f"{name} has no clause"
        if diagnostic.location is not None:
            filename, line = diagnostic.location
            assert isinstance(filename, str) and filename, name
            assert isinstance(line, int) and line >= 0, name
        json.dumps(dict(diagnostic.details))            # details is JSON-serialisable
        _assert_renders_in_four_parts(name, exc)


def _assert_renders_in_four_parts(name: str, exc) -> None:
    """`str(exc)` is `CODE: reason` / `in clause:` / `at:` / `because:` / `fix:`.

    The middle three are conditional and `render()` says so: a grammar diagnostic has no
    clause, a legality one no location, and a diagnostic with no numbers no `because:` line.
    The shape the goldens in `tests/golden/reject.*.txt` show is what is asserted — first line
    the code and the reason, last line the fix, and nothing outside the five known labels.
    """
    diagnostic = exc.diagnostic
    lines = str(exc).splitlines()
    assert lines[0] == f"{diagnostic.code}: {diagnostic.reason}", name
    assert lines[-1].startswith("  fix:       "), f"{name}: {lines[-1]!r}"
    assert diagnostic.fix in lines[-1], name
    labels = [line.split(":")[0].strip() for line in lines[1:-1]]
    assert labels == [label for label, present in
                      (("in clause", diagnostic.clause is not None),
                       ("at", diagnostic.location is not None),
                       ("because", bool(diagnostic.details))) if present], f"{name}: {labels}"


@pytest.mark.fr("FR-D1", "FR-D2")
def test_D2_an_accepted_schedule_produces_a_summary():
    """FR-D2 delegates to FR-M11: the corpus's mirror image is a schedule that is **accepted**.

    FR-D2's own acceptance is `test_M11_summary` (`01-requirements.md` §8), which is B's and
    runs on a plan fixture. What is added here is the surface half — the same `Schedule` object
    whose `.check()` raises for every case in this directory returns a rendered summary when
    the schedule is legal, so "rejection" and "summary" are two outcomes of one call path.
    """
    summary = test_legality._w1().summary()
    lines = summary.splitlines()
    assert len(lines) > 5 and all(line.strip() for line in lines[:3]), summary
    assert "stationary" in summary and "65536 bytes" in summary
