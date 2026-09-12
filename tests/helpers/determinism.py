"""The determinism gate (NFR-1). Spec: design/03-lld-M7-tests.md §3.5.

Written by B at P0b to unblock M0; C owns this file.
"""

from __future__ import annotations

import base64
import os
import pickle
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

_REPO = Path(str(files("tests"))).resolve().parent

_RUNNER = (
    "import base64, importlib, pickle, sys\n"
    "mod, qual, args = pickle.loads(base64.b64decode(sys.argv[1]))\n"
    "fn = importlib.import_module(mod)\n"
    "for part in qual.split('.'): fn = getattr(fn, part)\n"
    "sys.stdout.buffer.write(pickle.dumps(fn(*args)))\n"
)


def _next_seed() -> str:
    try:
        current = int(os.environ.get("PYTHONHASHSEED", "0"))
    except ValueError:  # "random"
        current = 0
    return str((current + 7) % 4294967296)


def in_fresh_process(fn, *args):
    """Run a module-level callable in a fresh interpreter under a different PYTHONHASHSEED."""
    payload = base64.b64encode(
        pickle.dumps((fn.__module__, fn.__qualname__, args))).decode("ascii")
    env = {**os.environ, "PYTHONHASHSEED": _next_seed(),
           "PYTHONPATH": os.pathsep.join(
               [str(_REPO), *filter(None, [os.environ.get("PYTHONPATH")])])}
    try:
        done = subprocess.run([sys.executable, "-c", _RUNNER, payload], env=env,
                              capture_output=True, check=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            f"{fn.__module__}.{fn.__qualname__} failed in a fresh process "
            f"(PYTHONHASHSEED={env['PYTHONHASHSEED']}):\n"
            f"{exc.stderr.decode('utf-8', 'replace')}") from None
    return pickle.loads(done.stdout)
