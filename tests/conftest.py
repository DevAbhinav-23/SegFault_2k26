"""The suite's whole CLI surface. Owner: Person C. Spec: design/03-lld-M7-tests.md §2.2.

The golden helper's three `--update-goldens` guards (§3.1), the fixture fixtures (§3.3), the
time budget (§3.6) and the network gate (§3.7) land here in a later phase.
"""

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


def pytest_configure(config):
    for name, doc in _MARKERS:
        config.addinivalue_line("markers", f"{name}: {doc}")
