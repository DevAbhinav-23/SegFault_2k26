"""The traceability gate. Spec: design/03-lld-M7-tests.md §6.3, design/04-test-plan.md §7.

Written by B at P7. **Person C owns the full gate**: only groups **M**, **E** and **K1-K4** are
asserted here, because those are the only requirements anyone has built. Groups **S**, **L**,
**T** and **D** belong to M1/M2/M3 (Person A) and M6/M7 (Person C), and neither exists, so this
file *reports* their uncovered ids and skips instead of failing. Flipping the skip into an
assertion is the one edit C makes when A's and C's modules land — `_UNBUILT` is the list.

Two directions are asserted for the **whole** suite, B's groups and everyone else's alike:

* no marker may name an FR that does not exist (§6.3's first `ASSERT`), and
* `01-requirements.md` §8's declared count still parses.

The item index comes from `tests/conftest.py`'s `FR_MARKERS`, recorded at `tryfirst` collection
time so `slow` and `requires_*` items are in it even though the default `addopts` deselect them.
"""

from __future__ import annotations

import re
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

import pytest

from tests.conftest import FR_MARKERS
from tests.helpers import golden

REQUIREMENTS = Path(str(files("tests"))).parent / "design" / "01-requirements.md"
"""The one source of FR ids. §8 of that file declares the count this test re-derives."""

_FR_HEADING = re.compile(r"^\*\*FR-([A-Z]+\d+)", re.MULTILINE)
"""design/03-lld-M7-tests.md §6.3's regex, verbatim."""

DECLARED_COUNT = 70
"""`01-requirements.md` §8: S 20, L 14, M 12, E 10, T 6, D 3, K 5."""

_BUILT = ("M", "E")
"""Groups every one of whose FRs must have a test today. These are M4's and M5's — B's."""

_BUILT_EXTRA = ("FR-K1", "FR-K2", "FR-K3", "FR-K4")
"""The four workload FRs B built end to end. **FR-K5** (W4 FFT) is not one: its acceptance is a
line in the honest-limits slide, and `test_K5_scope_documented` (M7 §7.7) is Person C's."""

_UNBUILT = ("S", "L", "T", "D")
"""Groups owned by A (M1/M2/M3, the diagnostics corpus) and C (M6, M7). Reported, not asserted."""


def _group(fr: str) -> str:
    """`"FR-M12"` -> `"M"` — the letters between the dash and the number."""
    return re.fullmatch(r"FR-([A-Z]+)\d+", fr).group(1)


def declared_frs() -> tuple[str, ...]:
    """Every FR id declared in `design/01-requirements.md` §3, in document order."""
    return tuple(f"FR-{m}" for m in _FR_HEADING.findall(
        REQUIREMENTS.read_text(encoding="utf-8")))


def _covered() -> dict[str, list[str]]:
    """`FR id -> [nodeid, ...]` over every item collected this session."""
    index: dict[str, list[str]] = defaultdict(list)
    for nodeid, ids in FR_MARKERS:
        for fr in ids:
            index[fr].append(nodeid)
    return index


def _require_whole_suite(pytestconfig) -> None:
    """A partial collection cannot prove coverage, so the gate declines to try."""
    if pytestconfig.option.file_or_dir:
        pytest.skip("partial collection: the traceability gate needs a whole-suite `pytest` run")


def test_requirements_parse_to_the_declared_count():
    """`01-requirements.md` §8 says 70; the §6.3 regex must still find exactly those."""
    frs = declared_frs()
    assert len(frs) == len(set(frs)), "an FR id is declared twice"
    counts = {group: sum(1 for fr in frs if _group(fr) == group)
              for group in sorted({_group(fr) for fr in frs})}
    assert counts == {"S": 20, "L": 14, "M": 12, "E": 10, "T": 6, "D": 3, "K": 5}, counts
    assert len(frs) == DECLARED_COUNT, f"§8 declares {DECLARED_COUNT}, the file parses {len(frs)}"


def test_traceability_no_marker_invents_an_fr(pytestconfig):
    """Direction two of §6.3: `@pytest.mark.fr("FR-D9")` must not be believable.

    This one is asserted for the whole suite whoever wrote the test, because a marker naming a
    requirement that does not exist makes the coverage number a lie in the *other* direction.
    """
    _require_whole_suite(pytestconfig)
    declared = set(declared_frs())
    invented = sorted({(nodeid, fr) for nodeid, ids in FR_MARKERS
                       for fr in ids if fr not in declared})
    assert not invented, "\n".join(f"{nodeid} claims {fr}, which is not an FR"
                                   for nodeid, fr in invented)


def test_traceability_B_groups_covered(pytestconfig):
    """**B's gate.** Every FR of groups M and E, and FR-K1…K4, has at least one test.

    "At least one", not "exactly one": three tests legitimately accept two FRs each (M7 §6.3's
    note, which corrects `04-test-plan.md` §7).
    """
    _require_whole_suite(pytestconfig)
    covered = _covered()
    mine = [fr for fr in declared_frs()
            if _group(fr) in _BUILT or fr in _BUILT_EXTRA]
    missing = [fr for fr in mine if fr not in covered]
    assert not missing, (
        f"{len(missing)} of B's {len(mine)} FR(s) have no test: {missing}")


def test_traceability_full_gate_reports_the_rest(pytestconfig):
    """Groups S, L, T, D and FR-K5 — reported, then skipped. **Person C owns this one.**

    The reason is in the skip message, which is what `04-test-plan.md` §8 item 10 asks of every
    skip: this is not a passing gate, it is an unbuilt one.
    """
    _require_whole_suite(pytestconfig)
    covered = _covered()
    theirs = [fr for fr in declared_frs()
              if _group(fr) in _UNBUILT or (_group(fr) == "K" and fr not in _BUILT_EXTRA)]
    missing = [fr for fr in theirs if fr not in covered]
    if missing:
        pytest.skip(f"owned by A/C, not built yet: {len(missing)} of {len(theirs)} "
                    f"uncovered — {missing}")
    if golden.UPDATE_GOLDENS:  # the whole gate passes; C can delete the skip above
        pytest.fail("every FR is covered: turn this test into an assertion (M7 §6.3)")


def test_traceability_writes_the_inverse_index(pytestconfig):
    """§6.3's last line: under `--update-goldens`, render `tests/README.md` from the markers."""
    _require_whole_suite(pytestconfig)
    covered = _covered()
    lines = ["# FR → test inverse index", "",
             "Rendered by `test_traceability_writes_the_inverse_index` under",
             "`pytest --update-goldens` (design/03-lld-M7-tests.md §6.3). Every functional",
             "requirement of `design/01-requirements.md` §3 maps to the test ids that accept it;",
             "`(none)` is a requirement nobody has built yet — see that file's owner column in",
             "`design/04-test-plan.md` §7.", "",
             "| FR | Tests |", "|---|---|"]
    for fr in declared_frs():
        ids = covered.get(fr, ())
        lines.append(f"| {fr} | " + ("<br>".join(f"`{i}`" for i in ids) if ids else "*(none)*")
                     + " |")
    text = "\n".join(lines) + "\n"

    readme = Path(str(files("tests"))) / "README.md"
    if golden.UPDATE_GOLDENS:
        readme.write_text(text, encoding="utf-8")
        golden.UPDATED.append(("tests/README.md", "inverse index rendered"))

    # What is asserted is the **coverage** the document claims, not the test ids it lists: a
    # nodeid list goes stale every time anyone adds a test, and failing A's and C's runs over a
    # rendered document would make the gate a nuisance rather than a check. `--update-goldens`
    # refreshes the ids; this assertion catches a README that claims the wrong requirements.
    rows = dict(re.findall(r"^\| (FR-[A-Z]+\d+) \| (.*) \|$",
                           readme.read_text(encoding="utf-8"), re.MULTILINE))
    assert list(rows) == list(declared_frs()), (
        "tests/README.md lists the wrong FRs; regenerate it with `pytest --update-goldens`")
    wrong = {fr: rows[fr] for fr in rows if (rows[fr] == "*(none)*") != (fr not in covered)}
    assert not wrong, (f"tests/README.md disagrees with the suite about {sorted(wrong)}; "
                       f"regenerate it with `pytest --update-goldens`")
