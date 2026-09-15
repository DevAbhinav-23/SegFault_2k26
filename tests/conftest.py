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
import glob
import json
import socket
import subprocess
import time

import pytest

from tests.helpers import golden

NFR3_TOTAL_BUDGET = 180.0
NFR3_PER_TEST_BUDGET = 3.0
DURATIONS: dict[str, float] = {}
_EXEMPT_MARKS = ("slow", "requires_air_opt", "requires_aircc", "requires_device",
                 "requires_ttsim")
"""Marks the §3.6 per-test budget does not apply to.

Every `requires_*` marker belongs here: the budget measures *our* code, and a test gated on an
external tool is measuring the tool. `test_NFR3_exempts_every_requires_marker` pins the rule so
a marker added to `_MARKERS` cannot silently start failing on wall clock.
"""

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


def _load_fixture(name):
    from tests import fixtures
    try:
        return fixtures.load(name)
    except (FileNotFoundError, AttributeError, ImportError) as exc:
        pytest.skip(f"fixture {name} not generated yet: {exc}")


@pytest.fixture(scope="session")
def w1(): return _load_fixture("w1")


@pytest.fixture(scope="session")
def w1_flip(): return _load_fixture("w1_flip")


@pytest.fixture(scope="session")
def w2(): return _load_fixture("w2")


@pytest.fixture(scope="session")
def w2_odd(): return _load_fixture("w2_odd")


@pytest.fixture(scope="session")
def w3(): return _load_fixture("w3")


@pytest.fixture(scope="session")
def w1_large(): return _load_fixture("w1_large")


@pytest.fixture(autouse=True, scope="session")
def _no_network():
    real = socket.socket

    class Blocked(real):
        def _refuse(self):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise RuntimeError("NFR-6: test suite must not reach the network")

        def connect(self, *a, **k):
            self._refuse()
            return super().connect(*a, **k)

        def connect_ex(self, *a, **k):
            # `connect_ex` is a second door into the same syscall: it reports errors as a
            # return code instead of raising, so a caller that used it walked past the gate.
            self._refuse()
            return super().connect_ex(*a, **k)

    socket.socket = Blocked
    yield
    socket.socket = real


def _tool_available(name):
    try:
        from spatial import m6_tools as m6
        return bool(m6.tool_available(name))
    except Exception:
        import shutil
        return shutil.which(name) is not None


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_device"):
        if not glob.glob("/dev/accel*"):
            pytest.skip("no device: /dev/accel* absent; needs XRT and an NPU")
    for mark in ("requires_aircc", "requires_air_opt"):
        if item.get_closest_marker(mark):
            tool = "aircc" if mark == "requires_aircc" else "air-opt"
            if not _tool_available(tool):
                pytest.skip(f"{tool} not installed; install the toolchain (07-environment.md)")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if call.when == "call":
        DURATIONS[item.nodeid] = call.duration
        if (call.duration > NFR3_PER_TEST_BUDGET
                and not any(item.get_closest_marker(m) for m in _EXEMPT_MARKS)):
            rep = report.get_result()
            rep.outcome = "failed"
            rep.longrepr = (f"{item.nodeid} took {call.duration:.1f}s. "
                            f"Mark it `slow` or shrink its fixture (NFR-3).")


_SELECTORS = ("-m", "-k", "--markexpr", "--keyword", "--deselect", "--ignore", "--lf",
              "--last-failed", "--stepwise", "--sw")


def _is_default_run(session):
    """True only for a whole-suite run that narrowed nothing.

    It reads the **invocation** args, not `config.option`: `addopts` in `pyproject.toml`
    always supplies `-m 'not slow and …'`, so `option.markexpr` is never empty and an
    `option`-based test would answer False for every run, silently disabling the 180 s gate
    below. An arg that is not a flag is a path, which also narrows the run.
    """
    return not any(arg.startswith(_SELECTORS) or not arg.startswith("-")
                   for arg in session.config.invocation_params.args)


def pytest_sessionfinish(session, exitstatus):
    try:
        out = golden.GOLDEN_DIR.parent / ".durations.json"
        out.write_text(json.dumps(DURATIONS, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    except OSError:
        pass
    if _is_default_run(session):
        total = sum(DURATIONS.values())
        if total > NFR3_TOTAL_BUDGET:
            slowest = sorted(DURATIONS.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print(f"NFR-3: default run took {total:.0f}s of a 180s budget. Ten slowest:")
            for nodeid, dur in slowest:
                print(f"  {dur:7.1f}s {nodeid}")
            session.exitstatus = 1
