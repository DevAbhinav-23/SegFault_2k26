"""The traceability gate. Spec: design/03-lld-M7-tests.md §6.3, design/04-test-plan.md §7.

Written by B at P7, extended by B at T5, **closed by A on 2026-09-15**. Both halves are now
assertions: `test_traceability_B_groups_covered` for groups M, E, TT and FR-K1…K4, and
`test_traceability_full_gate` for groups S, L, T, D and FR-K5, which reported and skipped
while M1/M2/M3 and the diagnostics corpus did not exist. `_UNBUILT` keeps its name and its
list — it is the second gate's denominator — but nothing in it is unbuilt any more.

**Two requirement documents, two denominators.** `01-requirements.md` §3 declares the **70**
FRs of the AIR path; `08-tt-backend.md` §6 declares the **14** of the Tenstorrent stretch group
(FR-TT1…FR-TT13 plus FR-TT7a), which are deliberately *not* part of that 70. Both are parsed
here, because a marker is only meaningful against the document that declares the id.

Two directions are asserted for the **whole** suite, B's groups and everyone else's alike:

* no marker may name an FR that does not exist (§6.3's first `ASSERT`), and
* `01-requirements.md` §8's declared count still parses.

The item index comes from `tests/conftest.py`'s `FR_MARKERS`, recorded at `tryfirst` collection
time so `slow` and `requires_*` items are in it even though the default `addopts` deselect them.
That is what lets the `requires_ttsim` tests of `tests/tt/` carry the FR-TT markers: they are
collected in the default run and only then deselected.
"""

from __future__ import annotations

import re
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

import pytest

from tests.conftest import FR_MARKERS
from tests.helpers import golden

_DESIGN = Path(str(files("tests"))).parent / "design"

REQUIREMENTS = _DESIGN / "01-requirements.md"
"""The source of the 70 AIR-path FR ids. §8 of that file declares the count this test
re-derives."""

TT_BACKEND = _DESIGN / "08-tt-backend.md"
"""The source of the FR-TT ids: §6 of the TT backend LLD, the stretch group B built at T1-T4."""

_FR_HEADING = re.compile(r"^(?:\| )?\*\*FR-([A-Z]+\d+[a-z]?)\b", re.MULTILINE)
"""design/03-lld-M7-tests.md §6.3's regex, widened twice and in no other way: `01-requirements.md`
opens a paragraph with the id (`**FR-S1 — …**`) while `08-tt-backend.md` §6 spells its group as a
table, one row per id (`| **FR-TT1** | …`), and FR-TT7a carries a letter suffix."""

DECLARED_COUNT = 70
"""`01-requirements.md` §8: S 20, L 14, M 12, E 10, T 6, D 3, K 5."""

TT_COUNT = 14
"""`08-tt-backend.md` §6: FR-TT1…FR-TT13 **plus** FR-TT7a, which §6 calls a structural
sub-requirement of FR-TT7 and which does not change the group's denominator of 13 ids + 1."""

_BUILT = ("M", "E", "TT")
"""Groups every one of whose FRs must have a test today. M4's, M5's and the TT emitter's — B's."""

_BUILT_EXTRA = ("FR-K1", "FR-K2", "FR-K3", "FR-K4")
"""The four workload FRs B built end to end. **FR-K5** (W4 FFT) is not one: its acceptance is a
line in the honest-limits slide, and `test_K5_scope_documented` (M7 §7.7) is Person C's."""

_UNBUILT = ("S", "L", "T", "D")
"""Groups owned by A (M1/M2/M3, the diagnostics corpus) and C (M6, M7) — the second gate's
denominator. The name is historical: they were unbuilt when B wrote this file, and as of
2026-09-15 they are asserted like everyone else's."""


def _group(fr: str) -> str:
    """`"FR-M12"` -> `"M"`, `"FR-TT7a"` -> `"TT"` — the letters between the dash and the number."""
    return re.fullmatch(r"FR-([A-Z]+)\d+[a-z]?", fr).group(1)


def _frs(path: Path) -> tuple[str, ...]:
    """Every FR id one requirement document declares, in document order."""
    return tuple(f"FR-{m}" for m in _FR_HEADING.findall(path.read_text(encoding="utf-8")))


def declared_frs() -> tuple[str, ...]:
    """Every FR id anyone may name: `01-requirements.md` §3 then `08-tt-backend.md` §6."""
    return _frs(REQUIREMENTS) + _frs(TT_BACKEND)


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
    frs = _frs(REQUIREMENTS)
    assert len(frs) == len(set(frs)), "an FR id is declared twice"
    counts = {group: sum(1 for fr in frs if _group(fr) == group)
              for group in sorted({_group(fr) for fr in frs})}
    assert counts == {"S": 20, "L": 14, "M": 12, "E": 10, "T": 6, "D": 3, "K": 5}, counts
    assert len(frs) == DECLARED_COUNT, f"§8 declares {DECLARED_COUNT}, the file parses {len(frs)}"


def test_the_tt_group_parses_and_is_not_counted_against_the_seventy():
    """`08-tt-backend.md` §6's own denominator, kept separate on purpose: the stretch group is
    **not** part of `01-requirements.md`'s 70 (§1 of that document says so), so a TT id must
    never appear in the count above and the two files must not both declare an id."""
    tt = _frs(TT_BACKEND)
    assert len(tt) == len(set(tt)) == TT_COUNT, tt
    assert all(_group(fr) == "TT" for fr in tt), tt
    assert not set(tt) & set(_frs(REQUIREMENTS))


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
    """**B's gate.** Every FR of groups M, E and TT, and FR-K1…K4, has at least one test.

    "At least one", not "exactly one": three tests legitimately accept two FRs each (M7 §6.3's
    note, which corrects `04-test-plan.md` §7).

    Group **TT** joined at T5, when all four workloads were green on ttsim. Its device tests are
    `requires_ttsim` and are deselected by the default `addopts`, but `conftest`'s `tryfirst`
    hook indexes them before that happens, so their markers count here.
    """
    _require_whole_suite(pytestconfig)
    covered = _covered()
    mine = [fr for fr in declared_frs()
            if _group(fr) in _BUILT or fr in _BUILT_EXTRA]
    missing = [fr for fr in mine if fr not in covered]
    assert not missing, (
        f"{len(missing)} of B's {len(mine)} FR(s) have no test: {missing}")


def test_traceability_full_gate(pytestconfig):
    """Groups S, L, T, D and FR-K5 — **asserted**, since 2026-09-15.

    This test used to report its groups and skip, because M1, M2, M3, the diagnostics corpus
    and M6 did not exist and a gate nobody can pass is noise rather than a check. The skip was
    the instruction for how to retire it — *"turn this test into an assertion once the
    uncovered list is empty"* — and the A-side correctness pass emptied it: the negative
    corpus of `tests/negative/` closed FR-D1/FR-D3 and, with it, the eighteen S- and L-group
    requirements that only a rejection can exercise.

    `_require_whole_suite` stays: a partial collection cannot prove coverage, and the gate
    declines to try rather than passing on an index of four items.
    """
    _require_whole_suite(pytestconfig)
    covered = _covered()
    theirs = [fr for fr in declared_frs()
              if _group(fr) in _UNBUILT or (_group(fr) == "K" and fr not in _BUILT_EXTRA)]
    missing = [fr for fr in theirs if fr not in covered]
    assert not missing, (
        f"{len(missing)} of {len(theirs)} FR(s) in groups {', '.join(_UNBUILT)} (plus FR-K5) "
        f"have no test: {missing}")


def test_traceability_writes_the_inverse_index(pytestconfig):
    """§6.3's last line: under `--update-goldens`, render `tests/README.md` from the markers."""
    _require_whole_suite(pytestconfig)
    covered = _covered()
    lines = ["# FR → test inverse index", "",
             "Rendered by `test_traceability_writes_the_inverse_index` under",
             "`pytest --update-goldens` (design/03-lld-M7-tests.md §6.3). Every functional",
             "requirement of `design/01-requirements.md` §3 — and, after it, every **FR-TT** of",
             "`design/08-tt-backend.md` §6, which is a separate stretch group and not part of",
             "that document's 70 — maps to the test ids that accept it;",
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
    rows = dict(re.findall(r"^\| (FR-[A-Z]+\d+[a-z]?) \| (.*) \|$",
                           readme.read_text(encoding="utf-8"), re.MULTILINE))
    assert list(rows) == list(declared_frs()), (
        "tests/README.md lists the wrong FRs; regenerate it with `pytest --update-goldens`")
    wrong = {fr: rows[fr] for fr in rows if (rows[fr] == "*(none)*") != (fr not in covered)}
    assert not wrong, (f"tests/README.md disagrees with the suite about {sorted(wrong)}; "
                       f"regenerate it with `pytest --update-goldens`")
