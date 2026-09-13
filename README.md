# SegFault_2k26 — a spatial DSL for NPUs, lowering to MLIR-AIR

Read [`hackathon/HANDOFF.md`](hackathon/HANDOFF.md) first. Ownership: [`design/00-README.md`](design/00-README.md) §2.

## Install

Python is pinned to **3.12**; the toolchain is pinned in [`design/07-environment.md`](design/07-environment.md) §1.

Branch **`role-b`** carries Person B's implementation (M0, M4, M5, M6's off-device half) and
`CONTRACT_VERSION = 5`; `main` may lag until it is fast-forwarded.

```bash
uv venv --python python3.12 --seed .venv
.venv/bin/python -m pip install --no-index --find-links vendor/wheels 'mlir_air[aie]' pytest
.venv/bin/python -m pip install -e . --no-deps
```

`vendor/wheels/` is git-ignored (~655 MB). Get the wheels from Person C or the shared drive, check
them with `sha256sum -c vendor/wheels/SHA256SUMS`, or download them online with the three-`-f`
recipe in `design/07-environment.md` §2. Offline install is the supported path (risk R-13: the
upstream release assets are pruned).

## Verify — with no environment variables set

```bash
.venv/bin/air-opt --version
.venv/bin/aircc --help
.venv/bin/python -c "import air.ir, air.passmanager; from air.dialects import air; print('ok')"
.venv/bin/python -c "from air import api as air; print('api ok')"
```

To actually drive `aircc` end to end, `aiecc` and Peano must be on `PATH`:

```bash
source scripts/airenv.fish        # bash/zsh: source scripts/airenv.sh
aircc --device npu1 --output-format=none design.mlir
```

Read `aircc`/`air-opt` **stderr**, not the exit code: `air-opt` prints `error:` and exits 0 (FR-T5).

## Tests

`.venv/bin/python -m pytest` — the default marks exclude `slow` and `requires_device`.

## Layout

`spatial/` the package (M0–M6) · `kernels/` W1/W2/W3 sources · `tests/` the suite and its
fixtures · `demo/` the pitch script · `design/` the frozen design set · `hackathon/` context and
the adversarial review trail · `scripts/` the `aircc` env · `vendor/` wheel cache and probe
reference · `spatial-dsl/`, `reading-group/`, `papers/`, `proposals/`, `notes/`, `docs/` prior
research.
