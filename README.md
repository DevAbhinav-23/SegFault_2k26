# SegFault_2k26 — a spatial DSL for NPUs, lowering to MLIR-AIR

Read [`hackathon/HANDOFF.md`](hackathon/HANDOFF.md) first. Ownership: [`design/00-README.md`](design/00-README.md) §2.

## Install

Python is pinned to **3.12**; the toolchain is pinned in [`design/07-environment.md`](design/07-environment.md) §1.

Everything is on `main`: the AIE path (M0–M6), the Tenstorrent stretch backend, the harness,
the fixtures and the demo, at `CONTRACT_VERSION = 6` (all three roles merged 2026-09-15).

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

## The surface, in eight lines

```python
import spatial as sp

M = N = K = 64

@sp.kernel                                       # the kernel runs in CPython unchanged
def gemm(A: sp.f32[M, K], B: sp.f32[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]

s = sp.schedule(gemm, target="npu1")
ax = s.axes()
s.grid(2, 2); s.tile(ax.i, 32); s.tile(ax.j, 32); s.tile(ax.k, 16)
s.reduce(ax.k, op="+"); s.place(px=ax.i0, py=ax.j0); s.stationary("C")
s.reside(A="L1", B="L1", C="L1"); s.double_buffer("A", "B")

print(s.summary())                               # deliveries, herd, L1 bytes, channels
print(s.mlir())                                  # the AIR module text
```

Delete every `s.*` line and `gemm(A, B, C)` still computes `A @ B` — that is FR-S18, and it is
tested. An illegal schedule is rejected by `s.check()` **before** any IR exists; the three demo
rejections are `kernels/rejections.py`. This example is `kernels/w1_gemm.py`'s `schedule_os`
verbatim; `kernels/w2_jacobi.py` and `kernels/w3_sw.py` are the halo and wavefront equivalents.

```bash
python demo/run_demo.py        # the six pitch beats, ~1.5 s, no device needed
```

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
without flushing Python's stdout, so no summary line prints after ttsim exits. Do not add a
second `-q` — `addopts` already carries one, and a doubled `-q` suppresses the count line in the
`.venv` suite as well. The emitter's own unit tests need none of that and run in the default suite. What the simulator
run establishes and what it does not — silicon, the Tensix compute engine, double buffering,
`f16`/`bf16`, and any timing claim, none of them touched — is
[`design/PROGRESS-TT.md`](design/PROGRESS-TT.md).

## Layout

`spatial/` the package (M0–M6, plus `m5tt_emit`/`m6tt_run` for Tenstorrent) · `kernels/` W1/W2/W3
sources · `tests/` the suite and its fixtures · `demo/` the pitch script · `design/` the frozen
design set · `hackathon/` context and the adversarial review trail · `scripts/` the `aircc` and
`ttsim` envs · `vendor/` wheel caches and probe reference · `spatial-dsl/`, `reading-group/`,
`papers/`, `proposals/`, `notes/`, `docs/` prior research.
