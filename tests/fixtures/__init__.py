"""Fixtures in the frozen format of design/06-interfaces.md §9. Owner: Person C (M8)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import json
import numpy as np


@dataclass(frozen=True)
class Fixture:
    name: str
    workload: str
    params: dict
    dtype: str
    seed: int | None
    tol: float | None
    generator: str
    _inputs: dict[str, np.ndarray] | None = None
    _expected: dict[str, np.ndarray] | None = None
    _oracle: dict[str, np.ndarray] | None = None

    def inputs(self) -> dict[str, np.ndarray]:
        """`inputs.npz` contents (M7 §3.3)."""
        if self._inputs is None:  # not `assert`: `python -O` would drop it and return None
            raise FileNotFoundError(f"fixture {self.name} ships no inputs.npz")
        return self._inputs

    def expected(self) -> dict[str, np.ndarray] | None:
        """`expected.npz` contents, or None for execute-never fixtures (w1_large)."""
        return self._expected

    def oracle(self) -> dict[str, np.ndarray] | None:
        """`oracle.npz` contents, or None for execute-never fixtures (w1_large)."""
        return self._oracle


def load(name: str) -> Fixture:
    base = resources.files("tests.fixtures") / name
    meta = json.loads((base / "meta.json").read_text())
    inputs = dict(np.load(base / "inputs.npz")) if (base / "inputs.npz").is_file() else None
    expected = dict(np.load(base / "expected.npz")) if (base / "expected.npz").is_file() else None
    oracle = dict(np.load(base / "oracle.npz")) if (base / "oracle.npz").is_file() else None
    return Fixture(name=name, workload=meta["workload"], params=meta["params"],
                   dtype=meta["dtype"], seed=meta.get("seed"), tol=meta.get("tol"),
                   generator=meta["generator"], _inputs=inputs,
                   _expected=expected, _oracle=oracle)
