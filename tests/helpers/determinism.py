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


def import_closure(module_name: str) -> list[str]:
    """Top-level module names importing `module_name` adds, minus the standard library.

    Meant to be called through `in_fresh_process`, which is why it lives in this module and not
    in a test file: the child imports only *this* module, which is stdlib-only, so nothing
    pytest dragged in can mask a dependency. Added by B at P7 for `test_NFR2_deps`; C owns the
    file. Raises if the target is already imported, because then the answer would be `[]` for
    the wrong reason.
    """
    import importlib

    if module_name in sys.modules:
        raise RuntimeError(f"{module_name} is already imported; the closure would be empty")
    before = {name.split(".")[0] for name in sys.modules}
    importlib.import_module(module_name)
    added = {name.split(".")[0] for name in sys.modules} - before
    return sorted(name for name in added
                  if not name.startswith("_") and name not in sys.stdlib_module_names)


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
