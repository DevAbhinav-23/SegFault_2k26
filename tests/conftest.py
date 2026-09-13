"""The suite's whole CLI surface. Owner: Person C. Spec: design/03-lld-M7-tests.md §2.2.

The `--update-goldens` guards of §3.1 are here; the fixture fixtures (§3.3), the time budget
(§3.6) and the network gate (§3.7) land in a later phase.

`FR_MARKERS` and its `pytest_collection_modifyitems` hook were **added by B at P7** so that
`tests/integration/test_traceability.py` (M7 §6.3) can see the *whole* suite: the default
`addopts` deselect `slow` and `requires_device` items inside pytest's own
`pytest_collection_modifyitems`, which runs **after** this `tryfirst` one, so the index is
complete before anything is filtered. **Person C still owns this file.**
"""

import os
import subprocess

import pytest

from tests.helpers import golden

_MARKERS = (
    ("slow", "minutes, not seconds; excluded by default"),
    ("requires_aircc", "needs aircc + aiecc + Peano"),
    ("requires_air_opt", "needs air-opt"),
    ("requires_device", "needs XRT and /dev/accel*"),
    ("requires_ttsim", "needs .venv-tt and the ttsim simulator: source scripts/tt_env.sh"),
    ("fr", "fr('FR-...', ...): the requirement(s) this test accepts"),
)


FR_MARKERS: list[tuple[str, tuple[str, ...]]] = []
"""`(nodeid, fr ids)` for every item collected this session, `slow` and `requires_*` included.

Read by `tests/integration/test_traceability.py`. Empty until a collection has happened, and
complete only for a whole-suite collection — the gate skips itself when pytest was given an
explicit path, because a partial collection cannot prove coverage.
"""


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Record every collected item's `fr(...)` ids **before** `-m` deselects anything."""
    FR_MARKERS.clear()
    for item in items:
        ids = tuple(fr for mark in item.iter_markers("fr") for fr in mark.args)
        FR_MARKERS.append((item.nodeid, ids))


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="rewrite every golden from the current run. Refused in CI and "
        "refused when the toolchain pin does not match.",
    )


def _git_dirty_under(path):
    done = subprocess.run(["git", "status", "--porcelain", "--", str(path)],
                          cwd=str(golden.GOLDEN_DIR.parent.parent), capture_output=True,
                          text=True, check=False)
    if done.returncode != 0:
        raise pytest.UsageError(
            f"--update-goldens cannot verify {path} is clean: git said {done.stderr.strip()!r}")
    return bool(done.stdout.strip())


def pytest_configure(config):
    for name, doc in _MARKERS:
        config.addinivalue_line("markers", f"{name}: {doc}")

    golden.UPDATE_GOLDENS = config.getoption("--update-goldens")
    if golden.UPDATE_GOLDENS:
        # Layer 1 - never from CI. A rewritten golden must be a human's commit.
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            raise pytest.UsageError(
                "--update-goldens is refused in CI. Regenerate locally, inspect "
                "the diff, and commit it with a reason (06-interfaces.md §8 rule 1).")
        # Layer 2 - never from an unpinned wheel. A golden is only valid for the pin.
        from spatial.m6_tools import check_pin
        from spatial.model import ToolchainError
        try:
            check_pin()
        except ToolchainError as exc:
            raise pytest.UsageError(
                f"--update-goldens is refused: {exc.diagnostic.reason}. "
                f"{exc.diagnostic.details['mismatched']}") from None
        # Layer 3 - never with a dirty tree, so the golden diff is the only diff.
        if _git_dirty_under("tests/golden"):
            raise pytest.UsageError(
                "tests/golden has uncommitted changes; commit or stash them first "
                "so the regeneration diff is readable.")

    if not config.pluginmanager.is_registered(golden):
        config.pluginmanager.register(golden, "spatial-golden")
