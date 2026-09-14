# PROGRESS — Part C (Person C): M6 device path, M7 harness, M8 kernels/fixtures/demo

Branch `role-c`. Baseline: B's close-out (`role-b`, HANDOFF P7 — 611 passed).
Replaces `progress_partc.md`. Last verified 2026-09-13.

## 0. Results (measured, this machine, no NPU present)

| What | Command | Result |
|---|---|---|
| Default suite | `.venv/bin/python -m pytest` | **684 passed, 4 skipped, 37 deselected, 17.2 s** |
| Slow suite | `pytest -m slow` | **11 passed, 2 skipped, 704 deselected, 5.4 s** |
| Demo driver | `python demo/run_demo.py` | **exit 0**, 6 beats, 3 degrade to `NOT BUILT YET`; 0.18 / 0.22 / 0.17 s over three runs (M8 DoD 5 wants ≤ 300 s — but the three A-beats are stubs, so this timing is not the final one) |
| Tree cleanliness (I-5) | `git status --porcelain` before/after `pytest -m slow` | **no change** |
| Fixture oracles | `tests/fixtures/make_fixture.py` | max-abs-err **0.0** on all five; `.npz` byte-identical on re-run |
| Wheel cache | `sha256sum -c SHA256SUMS` in `vendor/wheels/` | **16/16 OK** |
| CI workflow | `.github/workflows/ci.yml` | parses; 3 jobs (default 10 steps, slow 7, device 3). **Never run on a runner** — the wheel cache it restores has not been primed |

The 4 default skips are all deliberate and each one *is* a deliverable:

- `test_traceability` — 29 of 44 FRs uncovered, owned by A; flips to assert when A lands.
- `test_T4_device_diff` — `/dev/accel*` absent; the skip is the R-09 no-device evidence.
- `test_NFR7_all_errors_are_spatial` — `tests/negative/` holds only `__init__.py` (A's corpus).
- `test_demo_rejections` — the three `reject.<code>.txt` goldens are captured from A's real
  messages; they cannot be guessed, so the test skips rather than asserting a fiction.

Slow skips: `test_smoke` probe sources (git-ignored, B's), `tests/tt/test_tt_w2.py` (needs
`TT_METAL_SIMULATOR`, B's path).

## 1. Environment (do this first on any new machine)

- No `uv` here; venv via `/home/abhinavv/.local/bin/python3.12 -m venv .venv`.
  System python is 3.14 — a prompt showing `via 🐍 v3.14.6` means the shell/editor is NOT on
  the venv. Always use `.venv/bin/python`, absolute when cwd isn't repo root
  (`cd vendor/wheels` breaks a relative `.venv/...`).
- Full 16-wheel cache in `vendor/wheels/` (git-ignored). `sha256sum -c SHA256SUMS` must run
  **inside** `vendor/wheels/` — SUMS lists bare filenames.
- `source scripts/airenv.sh` (bash) / `.fish` (fish) puts aiecc+Peano on PATH for slow tests,
  but it also prepends another venv's python. Invoke `.venv/bin/python` explicitly after.
- Zed has no "Select Interpreter": launch with
  `source .venv/bin/activate.fish && zed .` from repo root, or pin `pythonPath` in
  `.zed/settings.json`. Unresolved `numpy`/`air` squiggles = wrong interpreter, not code bugs.
- Upstream moved under us once: `origin/main` fast-forwarded `b220f24 → 09cb367` (Tenstorrent
  second emitter `spatial/m5tt_emit.py`, `m6tt_run.py`, `tests/tt/`, `design/08-tt-backend.md`).
  Separate path, doesn't touch M6-AIR work; suite grew 611 → 664 there, C adds the rest to 677.
  Our four original files merged with zero conflicts.

## 2. Done

### M6 device path — `spatial/m6_tools.py`, `spatial/model.py`

B built the off-device half (tool/invoke/verdict/artifact/ir_facts/check_pin — reviewed, kept;
3 documented deviations: `ERROR_LINE`'s extra `loc()` branch, `PIPELINES["aie"]` +
`air-place-herds` prefix, `cascade_channels` counting `aie.cascade_flow`). C added:

- `has_device()` — `glob("/dev/accel*")` (+`/dev/accel/accel*`), no subprocess, raises nothing.
  False here → device tests skip with a reason.
- `diff(device, oracle, tol)` — LLD §3.6 verbatim: float64 abs err per array; shape mismatch →
  `(False, inf, None, size, size)`; else `(matched, max_abs, first, mismatched, total)`.
- `run(artifact, inputs, target, kernel_name, workdir=None)` — LLD §3.5: `TOOL-NO-DEVICE` guard,
  `XRTBackend.load` invoker (**not** `CompiledKernel` — it zeroes W2's `U`), arity ≤ 5,
  `FileLock npu.lock`, reshape flat→shapes, `backend.unload()` in `finally`.
- `trace(mlir_path, model_json, function, workdir=None)` — LLD §3.8: `air-dependency` first,
  `air.launch` guard naming `Runner.cpp:547-551` (else segfault), return carries
  `NOT A CORRECTNESS ORACLE` (I-8).
- `DiffReport` frozen dataclass in `model.py` after `EmitResult` (contract §7.2 5-tuple +
  `_validate`, house `_need` style).
- Import hygiene: `numpy` top-level (NFR-2 allows stdlib+numpy at scope), `filelock`
  function-local, `air.*` lazy (FR-S20).

### M7 harness — `tests/conftest.py`, `tests/unit/test_nfr.py`, `tests/integration/test_oracle.py`

- Session fixtures `w1/w1_flip/w2/w2_odd/w3/w1_large` via `fixtures.load` (skip on missing).
- Autouse `_no_network` blocking `AF_INET`/`AF_INET6` `connect` only (AF_UNIX left for XRT).
- `pytest_runtest_setup` skips for `requires_device` / `requires_aircc` / `requires_air_opt`.
- `pytest_runtest_makereport` DURATIONS + >3.0 s fail for unmarked tests (exempts
  `slow`/`requires_*`); `pytest_sessionfinish` writes `tests/.durations.json` (git-ignored) and
  fails a default run over 180 s with the ten slowest.
- NFR tests: `test_NFR3_budget`, `test_NFR4_no_bare_raise` (AST lint over `spatial/`),
  `test_NFR6_no_network`, `test_NFR7_all_errors_are_spatial`.
- `test_oracle.py`: `test_fixture_contract` (M7 §3.3), `test_ignorability` (FR-S18, 8 trials ×
  3 workloads), `test_K5_scope_documented` (FR-K5).
- `tests/README.md` regenerated by whole-suite `pytest --update-goldens` (partial-path runs skip
  the gate by design).
- `test_skips_are_explained` (I-8, M7 §8 item 9): AST lint over `tests/**` — every `skip(`/
  `skipif(` passes a reason, and a *literal* reason is ≥ 20 characters. A reason computed at
  run time (`str(exc)`, an f-string over a diagnostic) is taken on trust, because its length is
  not in the source; the lint exists to catch `skip("todo")`, not to grade a rendered
  diagnostic.
- **Level O, the three end-to-end oracle diffs** (04-test-plan §8 item 5) —
  `test_W1_end_to_end`, `test_W2_end_to_end`, `test_W3_end_to_end`, plus `test_W1_flip`
  (FR-K2's `array_equal(C_os, C_ws)`, at kernel level) and
  `test_W2_oracle_is_timestep_outermost` (FR-K3's second acceptance). Each runs the CPython
  kernel on `inputs.npz` and diffs it against the independently generated `expected.npz`
  **through `m6.diff`**, the same comparator the device path uses. Mutation-checked: perturbing
  each of the three kernels fails its own test and nothing else.
- `test_T4_device_diff` carries `fr("FR-T4")`, which closes **C's last traceability gap**: the
  full gate now reports 28 uncovered, all of them A's `FR-S*`/`FR-L*`. It deliberately keeps
  gating on `has_device()` instead of the `requires_device` marker — that marker is deselected
  by the default `addopts`, and a deselected test prints nothing, but R-09 wants the device's
  absence in the default run's skip list.

### CI — `.github/workflows/ci.yml` (M7 §3.9)

Three jobs. `default` (push/PR): restore the wheel cache keyed on `SHA256SUMS` →
`sha256sum -c` → offline install → pin check → `PYTHONHASHSEED=0 pytest -ra` →
`PYTHONHASHSEED=7 pytest -ra -k "golden or plan or summary"` →
`git diff --exit-code -- tests/golden tests/README.md` → evidence upload on failure.
`slow` (nightly + dispatch): the same install, then `source scripts/airenv.sh` and
`pytest -m slow -ra`. `device` (self-hosted `npu`, `continue-on-error`): `pytest -m requires_device -ra`.

Two deliberate deviations from §3.9's outline, both recorded here rather than hidden:

1. **No "NETWORK OFF" step.** Dropping the runner's egress also kills its connection to
   GitHub, so the job would report nothing. NFR-6 is enforced in process by `conftest._no_network`
   and asserted by `test_NFR6_no_network`.
2. **No separate "fail if wall clock > 180 s" step.** `pytest_sessionfinish` already prints the
   ten slowest and sets a non-zero exit status, so the `pytest` step *is* §3.9's step 10.
   Duplicating the logic in YAML would give it a second, divergent definition.

### M8 kernels / fixtures / demo

- `kernels/{w1_gemm,w2_jacobi,w3_sw}.py`: import-safe plain Python — **no `import spatial` at
  scope** (A's surface is absent; `m1_frontend.py` is a 13-line stub). Kernel fns run in
  CPython; `schedule_*(target)` raise `NotImplementedError("surface: Person A")`; `main()`
  prints a checksum. `w1_gemm_bf16.py`: 256³ params reusing `gemm`.
- `rejections.py`: three raisers carrying the §5.3 message shapes in docstrings.
- `tests/fixtures/make_fixture.py` (`default_rng` only): W1 seed 0 / W2 seed 1 (honours B-P25 —
  planes 1..T carry plane-0 Dirichlet boundaries, reference is a t-outermost two-plane copy) /
  W2-odd T=5 / W3 seed 2 (reference uses an `if` **statement** vs the kernel's value-level
  conditional, so the diff tests something).
- Dirs: `w1`, `w1_flip` (full), `w1_large` (inputs-only), `w2`, `w2_odd`, `w3`, plus meta-only
  negatives `w1_l1_overflow`, `w2_zero_t`, `w2_pi4` (extra `expect`/`code` keys beyond M7's
  6-key set — deliberate).
- `tests/fixtures/__init__.py`: `Fixture` + `load()` with `inputs()/expected()/oracle()`
  **methods** (M7 §3.3 — properties broke the contract test).
- `demo/honest_limits.md` (§5.2 verbatim + no-device line) and `demo/run_demo.py` (6 beats,
  A-beats degrade to NOT-BUILT notes, never crashes).

## 3. To do

**C-owned, blocked on others — no C-side code change needed to activate:**

| Item | Unblocked by | What flips |
|---|---|---|
| Real schedules, demo beats 0:40–2:15, rejection beat | A: M1/M2/M3 surface + checker | `schedule_*` bodies; demo skip-branches |
| 30 legality codes, negative corpus, NFR7 live run | A: M3 | `tests/negative/` populated |
| Traceability skip → assert | A (29 of 44 FRs) | `_UNBUILT` list empties |
| `test_no_air_import_until_build` (K5-adjacent) | A | new test |
| Device run + `test_T4_device_diff` | hardware (D5 window: `/dev/accel*` + XRT) | skip → pass; `run()`/`diff()` ready |
| `test_demo_rejections`'s three `tests/golden/reject.*.txt` (M8 §9 step 5) | A: M3 messages | the test is written and skips; what lands is the capture |

**C-owned, actionable now — these are open gaps against the LLD definitions of done:**

- **Prime the CI wheel cache.** The workflow restores `vendor/wheels` with
  `fail-on-cache-miss: true` and never touches the release URLs (R-13), so until someone seeds
  that cache out of band (`07-environment.md` §2, Q-C3) the `default` job fails at step 3. This
  is the one piece of CI that cannot be written in the repo.
- **Q-C17: the PACT 2026 attribution** (M8 §5.2, due rehearsal #2) — **deferred by the user on
  2026-09-13**: "don't do anything regarding ownership yet". No file changed. The slide's
  current wording already makes no ownership claim — it names AIEHalide as a neighbour — so the
  unconfirmed state is the safe one, and the decision can wait for rehearsal #2 as the LLD
  intended. Unconfirmed at D7 ⇒ the claim is deleted, not hedged.
- **Commit `role-c`** — all of it is still uncommitted (`model.py`, `m6_tools.py`, 2 modified
  test files, all new M7/M8 files + fixtures). Do this before `main` moves again.
- **v6 contract sign-off** (§4 process, needs all three owners): `DiffReport` added to
  `model.py`; `run` takes `target`/`kernel_name`; `trace` takes `function`. The frozen §7.2
  shapes cannot supply these — same class as B-P27.

**Not C:** B-P26 (W1 `aie` pipeline — documented, golden keys dropped), B-P29 (NFR-5
module-const AST lint), B-O8 (`str(module)` stability).

## 4. Errors currently open

None failing. Everything unresolved is a *blocked* item in §3, not a bug: three deliberate
skips (§0), plus the v6 contract deviations awaiting A/B signature.

## 5. Errors faced and how they were resolved

| Error | Cause | Fix |
|---|---|---|
| 5 pin failures after a clean setup | `latest-*` URLs floated `llvm-aie ...1201 → ...1301`; `check_pin()` runs first in `artifact()`, so one mismatch cascades into T1/T3/T6 | Downgraded to the **pinned** wheel (still downloadable), **not** a pin change; cached all 16 wheels in `vendor/wheels/` |
| `test_NFR2_deps[spatial.m6_tools]` fail | `filelock` imported at module scope | Back to a function-local import |
| `test_fixture_contract` fail | `Fixture.inputs/expected/oracle` written as properties | Made them `()` methods, per LLD / M7 §3.3 |
| Contract assertion nonsense: `inputs ⊆ params-reads` | Compares arrays against sizes | Replaced with `expected ⊆ inputs` |
| `test_K5_scope_documented` fail | The TT merge changed K5's `00-README` row (bold dropped) | Substring assert instead of exact line |
| `test_ignorability` fail | Reference snapshot taken **pre**-kernel | Snapshot post-kernel |
| `assert "air" not in sys.modules` fail | Invalid once M5 imports `air` in-session | Dropped; FR-S20 is covered by fresh-interpreter tests |
| `kernels` bare name unresolvable in a test | Package has no re-exports | `importlib` + an explicit module map |
| `test_traceability` golden stale | README regenerated from a partial-path run | Whole-suite `pytest --update-goldens` |
| `sp.schedule` `AttributeError` in ignorability trials | A's surface absent | Catch broad `Exception` per LLD §3.4 |
| W2 device output `U` all zeros | `CompiledKernel` invoker zeroes it | `XRTBackend.load` invoker instead |
| `air-runner` segfault | Missing `air.launch` | Guard in `trace()` naming `Runner.cpp:547-551` |
| **NFR-3's 180 s gate never fired** (found 2026-09-13 auditing item 8, not by a failing test) | `_is_default_run` read `config.option.markexpr`, but `addopts` in `pyproject.toml` always supplies `-m 'not slow and …'`, so it was never empty and the guard answered False for **every** run | Read `config.invocation_params.args` instead — a default run is one that narrowed nothing. `test_NFR3_budget` now asserts the guard arms on `pytest` and on `pytest -ra`, and disarms on `-m`, `-k` and a path |
| **`diff()` reported a match when the device returned fewer buffers than the oracle** | `zip(device, oracle)` truncates to the shorter side, so the surviving pairs decided the verdict — a kernel returning 3 buffers instead of 4 passed | Length checked first; every oracle element charged as mismatched. `test_diff_denominator` covers it, plus the empty cases |
| **`connect_ex` walked past the NFR-6 network gate** | `_no_network` overrode `connect` only; `connect_ex` is the same syscall reporting failure as a return code, and several stdlib clients prefer it | Both routed through one `_refuse()`; `test_NFR6_no_network` asserts both, and that AF_UNIX (XRT) is untouched |
| **`run()` raised a bare `OSError` for a missing workdir, and truncated a short result list** | `FileLock` cannot create its lock file in a directory that does not exist (NFR-7 says every failure on a user path is a `SpatialError`); `zip(flat, shapes)` silently dropped extras | `mkdir(parents=True)` wrapped in a `TOOL-AIRCC-FAILED` diagnostic; result/shape lengths compared before the reshape |
| **`test_K5_scope_documented` could not see a W4 kernel** | It scanned `dir(kernels)`, but the package re-exports nothing, so an actual `kernels/w4_fft.py` would be invisible and the gate would pass | Globs `kernels/*.py` as well as the attribute scan |
| **`run_demo.py` would have crashed on its headline beat** | `beat()` caught `NotImplementedError` only. Once A lands, the 2:15 rejection raises `SpatialError` — the beat's *success* — and the pitch would have died at 2:15 on stage | `_rejection` catches `SpatialError` and prints the diagnostic as the beat's output; `beat` keeps a broad catch as the net |
| `_existing_fixture_dirs` was dead code; `Fixture.inputs()` guarded with a bare `assert` | Never called; `python -O` drops asserts and would have returned `None` | Deleted; raises `FileNotFoundError` |
| Six skips failed the new I-8 lint | Four TT skips pass `str(exc)`, one golden skip an f-string over a diagnostic — unjudgeable statically, not actually bad; one of C's own was 19 characters | Lint judges **literal** reasons only and trusts computed ones; `test_oracle.py`'s "numpy not installed" rewritten to say what is missing |
