"""The suite's whole CLI surface. Owner: Person C. Spec: design/03-lld-M7-tests.md §2.2.

The `--update-goldens` guards of §3.1 are here; the fixture fixtures (§3.3), the time budget
(§3.6) and the network gate (§3.7) land in a later phase.
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
    ("fr", "fr('FR-...', ...): the requirement(s) this test accepts"),
)


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
