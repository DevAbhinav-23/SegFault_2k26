# PROGRESS — Person B

*State file for Person B's work. Phase **P0a — repository setup**, 2026-09-13. Scaffold, pinned
toolchain, stubs. No module logic in this phase.*

---

## Done

| # | Item | Where |
|---|---|---|
| 1 | venv on the pinned interpreter (`uv venv --python /home/adi/.local/bin/python3.12 --seed`) | `.venv/` (git-ignored) |
| 2 | Wheel cache: **16** wheels, **644 MB**, resolved against the three `-f` release pages | `vendor/wheels/` (git-ignored), `vendor/wheels/SHA256SUMS` (committed) |
| 3 | Offline install from the cache — proves the cache is sufficient (R-13 mitigation) | — |
| 4 | `scripts/airenv.fish`, `scripts/airenv.sh` — 07-environment §4's env block, 7 lines each, with a "no mlir_air here" guard | `scripts/` |
| 5 | Repo skeleton: `spatial/` (8 modules), `kernels/`, `tests/` (helpers, unit, negative, integration, golden, fixtures{plans,mappings,corrupt,stderr,modules}), `demo/`, `vendor/`, `scripts/` | 07-environment §6 + M7 §2.1 |
| 6 | Eight `spatial/*.py` stubs: docstring naming owner + LLD, the §7.2 entry points with exact parameter names, each body `raise NotImplementedError`. `model.py` = docstring + `CONTRACT_VERSION = 3`. **No `import air` anywhere in `spatial/`** (FR-S20) | `spatial/` |
| 7 | `tests/conftest.py` — `--update-goldens` option + the five marks, M7 §2.2. Nothing else yet | `tests/conftest.py` |
| 8 | `pyproject.toml` — `spatial-dsl`, `requires-python = "==3.12.*"`, `numpy==2.5.3` only, dev extra `pytest`, `[tool.pytest.ini_options]` per M7 §2.2 | — |
| 9 | `.gitignore` extended: `vendor/wheels/*.whl`, `vendor/probes/`, `air_project/`, `*.pyc`, `.pytest_cache/`, `*.egg-info/` | — |
| 10 | `design/07-environment.md` §2's D0 block filled with 16 filenames + sha256s, and the "no `v*.*.*` tag" finding | — |
| 11 | Root `README.md` replaced with the quickstart | — |
| 12 | Probe sources copied for reference (venvs and build dirs excluded, 2.7 MB of the 3.2 GB) | `vendor/probes/` (git-ignored) |

### The wheel cache

The pin (07-environment §1) is four wheels; the cache is **16**, because `mlir_air[aie]` pulls
eight transitive dependencies (`ml_dtypes`, `filelock`, `aiofiles`, `cloudpickle`, `rich`,
`pygments`, `markdown-it-py`, `mdurl`) and `pytest` four more (`pluggy`, `iniconfig`,
`packaging`, and `pygments` again). Filenames and sha256s: `design/07-environment.md` §2 and
`vendor/wheels/SHA256SUMS`. Download: 4.5–5.8 MB/s, 55 s + 54 s + 29 s + 2 s for the four big
ones, ~3 min for the lot.

**No `v*.*.*` release tag exists on `Xilinx/mlir-air`** (only `latest-air-wheels` and
`latest-air-wheels-no-rtti`), verified 2026-09-13, so 07-environment §2 action 1 is impossible
and the local cache is the **only** mitigation for R-13 (asset pruning). Distribute
`vendor/wheels/` out of band; `SHA256SUMS` is the check.

---

## Verified (command → result)

Every line below was run. The five verify commands ran under `env -i` — **no environment
variables at all**, which is stronger than 07-environment §2's "with no environment variables
set".

| Command | Result |
|---|---|
| `env -i .venv/bin/air-opt --version` | `AOMP-23.0-60 (http://github.com/ROCm/aomp):` / ` Source ID:23.0-60-8b7c0c42edfe088700d312231739eff0dabd913c` / `  LLVM version 24.0.0` / `  Optimized build with assertions.` — exit **0**. (§2 predicted "LLVM version 24.0.0"; the banner carries three more lines) |
| `env -i .venv/bin/aircc --help` | exit **0**, 255 lines, first line `OVERVIEW: AIR Compiler Driver` |
| `env -i .venv/bin/python -c "import air.ir, air.passmanager; from air.dialects import air; print('ok')"` | `ok` — exit 0 |
| `env -i .venv/bin/python -c "from air import api as air; print('api ok')"` | `api ok` — exit 0 |
| `env -i .venv/bin/python -c "…m.version(…)"` | `0.0.1.2026091204+ff95a9b 1.4.3.dev55+g10767b5 22.0.0.2026091201+386ca5c6 2.5.3` — **the pin, exactly** |
| `pip install --no-index --find-links vendor/wheels 'mlir_air[aie]==…' pytest` | **succeeded, 11.5 s**, 16 packages, no network. The cache is sufficient |
| `source scripts/airenv.sh` (bash) and `scripts/airenv.fish` (fish) | both put `…/mlir_air/bin/aircc` and `…/mlir_aie/bin/aiecc` on `PATH` and set `PEANO_INSTALL_DIR=…/site-packages/llvm-aie` |
| `pip install -e . --no-deps` → `env -i .venv/bin/python -c "import spatial"` | `spatial ok 3` (`CONTRACT_VERSION`) |
| `.venv/bin/python -m pytest` | **0 tests collected, 0.14 s wall** (budget: < 2 s). Exit code **5** = pytest's `NO_TESTS_COLLECTED` — see *Open* below |
| `.venv/bin/python -m pytest --markers` | all five marks registered: `slow`, `requires_aircc`, `requires_air_opt`, `requires_device`, `fr` |

### Toolchain baseline, end to end (FR-T5 discipline)

`vendor/probes/q/q7_a.py` — the upstream-API W1 shape (`M=N=K=64, TM=TN=32, TK=16, PI=PJ=2`,
herd `shape=(1,2)`), **not** our DSL. No edit was needed to run it from its new location.

```
python q7_a.py > w1_base.mlir                 exit 0, 127 lines, empty stderr, 0.15 s
source scripts/airenv.sh
timeout 600 aircc --device npu1 --output-format=none w1_base.mlir
```

* **exit 0**
* **`error:` lines in stderr: 0** (`grep -c 'error:'` — the exit code alone is never trusted, FR-T5)
* **wall 1.017 s** (user 1.517 s — it parallelises)
* all **22/22** aiecc stages ran; two core ELFs produced,
  `elfs_gemm_seg_core_0_{2,3}/elfs_gemm_seg_core_0_{2,3}.elf`, 7 392 B each; `air_project/`
  holds `aie.w1_base.mlir` (30 325 B) and `npu.w1_base.mlir` (16 838 B)

**The toolchain compiles a W1-shaped module off-device, on this machine, from the cache.** Gate
G1's technical precondition is met on machine 1 of 3.

---

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **B-P1** | `pytest` exits **5**, not 0, on an empty suite | M7 §9 step 1 asks for "collects zero tests and exits green in < 2 s". Zero collected is pytest's `NO_TESTS_COLLECTED` (exit 5) by design; `addopts` is set exactly as M7 §2.2 specifies and was not widened to mask it. It goes green the moment the first test lands. **No action proposed**; recorded so nobody re-diagnoses it. CI must not gate on `pytest` before D1 |
| **B-P2** | `tests/` is not a Python package | The brief's file list has no `tests/__init__.py`, but M7 §2.1 reaches fixtures with `importlib.resources.files("tests.fixtures")`. That resolves today only as an implicit namespace package from the repo root. **Person C decides at D1** whether to add `tests/__init__.py`; it is a one-line change either way |
| **B-P3** | `design/07-environment.md` §2 item 2 still reads "the four wheels" | The filled block immediately below it says 16 and lists them. Left as-is: the brief said change nothing else in that file. One-word edit whenever someone else touches §2 |
| **B-P4** | Toolchain installed on **1 of 3** machines | Machines 2 and 3 are Person C's D0 task (gate G1). `vendor/wheels/` must be moved by shared drive or USB — it is git-ignored and 644 MB |
| — | Nothing is blocked on another person. | |

Not verified: `aircc --output-format=pdi`, `air-opt` pass pipelines, `air-runner`, anything on a
device. None was in scope for this phase.

---

## Next phase

B's own critical path starts at M0/M4/M5. Before any of it:

1. **Sign `06-interfaces.md` v3** (`design/00-README.md` §4's signature block) — three signatures,
   still blank. M4 and M5 are written against it from D1.
2. **M0 `model` dataclasses** (Person A, D1) — `spatial/model.py` is a docstring and one constant
   until they land. B cannot type `m4.plan` without `LegalMapping` and `MappingPlan`.
3. **M5 skeleton at D1**, real at D2 (`design/03-lld-M5-emitter.md`); **M4 W1 at D2** — gate G2
   (W1 end to end) is D2.
4. `tests/fixtures/plans/w1_plan.py` — B's hand-written `MappingPlan` literal, M4's stand-in and
   later M5's input fixture (M7 §3.8). It is what makes `test_emitter_makes_no_decisions` mean
   anything.
5. Person C: machines 2 and 3, and the `.github/workflows/ci.yml` of M7 §3.9 — steps 2–4 restore
   from `vendor/wheels/SHA256SUMS`, so **CI never sees the `-f` release URLs**.
