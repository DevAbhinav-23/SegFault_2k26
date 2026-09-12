"""Level I — `air-opt` inspection. Spec: design/04-test-plan.md §3.2.

Stub written by B at P0c to unblock; **Person C owns this file**.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest

from spatial import m6_tools as m6

MODULES = Path(str(files("tests"))) / "fixtures" / "modules"


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-T5")
def test_I_unpaired_channel(tmp_path):
    """The characterisation test FR-T5 exists for: a diagnostic on stderr **and exit 0**.

    This is the one test that runs a hand-written broken module rather than our emitter's
    output. It characterises the toolchain, not us: if upstream ever starts failing on this,
    the test breaks loudly and FR-T5 can be relaxed (`04-test-plan.md` §3.2).
    """
    if not m6.tool_available("air-opt"):
        pytest.skip("air-opt is not resolvable; install the toolchain (07-environment.md §2)")

    module = MODULES / "unpaired_channel.mlir"
    run = m6.invoke([m6.tool("air-opt"), module,
                     f"-pass-pipeline={m6.PIPELINES['pairs']}", "-o", tmp_path / "pairs.mlir"],
                    cwd=tmp_path)

    assert "found channel op not in pairs" in run.stderr
    assert run.returncode == 0, "the exit code is the half that lies (VF §B.6, §S11)"

    # ...and the wrapper fails it anyway, which is the whole of FR-T5.
    with pytest.raises(m6.ToolchainError) as excinfo:
        m6.verdict(run)
    assert excinfo.value.diagnostic.code == "TOOL-DIAGNOSTIC"
    assert excinfo.value.diagnostic.details["returncode"] == 0
