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

`.venv/bin/python -m pytest` — the default marks exclude `slow`, `requires_device` and
`requires_ttsim`.

## The Tenstorrent backend

A second emitter, `spatial/m5tt_emit.py`, turns the same backend-neutral `MappingPlan` into a
TT-Metalium program that `spatial/m6tt_run.py` executes on Tenstorrent's **functional** simulator
`ttsim` — no hardware, and nothing to do with MLIR-AIR. **All four variants — W1 GEMM, W3
Smith-Waterman, W1-flip cascade and W2 Jacobi — execute exactly on `ttsim` (gates T1–T4 green,
2026-09-13), each against numpy, each with a negative control, each equal to the plan
interpreter.** It needs its own venv, because the `ttnn`
wheel pins `numpy<2` against the project's `numpy==2.5.3`. Sourcing the script builds `.venv-tt`
from the checksummed cache in `vendor/tt/` on first use (git-ignored, like `vendor/wheels/`; get
the four artefacts from Person C or the shared drive and check them with `sha256sum -c
vendor/tt/SHA256SUMS`), then exports the simulator environment:

```bash
source scripts/tt_env.sh
.venv-tt/bin/python -m pytest -rA -q -m requires_ttsim tests/tt   # 25 PASSED, exit 0, ~286 s
```

Judge that run by its exit status and its `-rA` `PASSED` lines: ttsim's exit ends the process
without flushing Python's stdout, so pytest's final summary line never reaches a pipe or a file.
The emitter's own unit tests need none of that and run in the default suite. What the simulator
run establishes and what it does not — silicon, the Tensix compute engine, double buffering,
`f16`/`bf16`, and any timing claim, none of them touched — is
[`design/PROGRESS-TT.md`](design/PROGRESS-TT.md).

## Layout

`spatial/` the package (M0–M6, plus `m5tt_emit`/`m6tt_run` for Tenstorrent) · `kernels/` W1/W2/W3
sources · `tests/` the suite and its fixtures · `demo/` the pitch script · `design/` the frozen
design set · `hackathon/` context and the adversarial review trail · `scripts/` the `aircc` and
`ttsim` envs · `vendor/` wheel caches and probe reference · `spatial-dsl/`, `reading-group/`,
`papers/`, `proposals/`, `notes/`, `docs/` prior research.
