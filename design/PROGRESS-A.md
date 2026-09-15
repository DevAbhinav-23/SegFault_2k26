# PROGRESS — Person A

## Status (2026-09-15, A-side correctness pass)

**The seven open items `hackathon/HANDOFF.md` §"Integration — state at 2026-09-15" lists under
*Person A* are closed, except item 4 (the `06-interfaces.md` signatures), which is the
architect's to sign and not a code change.**

* **The negative corpus exists.** `tests/negative/` holds six stage modules and
  `test_catalogue.py`; **all 43 catalogue codes are raised**, 30 of which were raised by
  nothing before this pass. `test_D1_schema` and `test_D3_catalogue_complete` are written and
  pass, and `test_NFR7_all_errors_are_spatial` no longer skips.
* **The 18 uncovered FRs are 0.** `tests/integration/test_traceability.py`'s full gate is an
  **assertion**, not a skip, per its own docstring's instruction.
* **Three defects in `spatial/` are fixed**: `GrammarError` built with `location=None`
  (M0 invariant I71 — an `exec`-compiled kernel raised `ValueError`), machine-dependent
  absolute paths in M1/M2 locations, and `L1-CAPACITY` with no per-buffer breakdown. Two more
  were found by the corpus and fixed with it: snippet-relative line numbers, and a bare
  `ValueError` out of `reduce(ax.k0, …)`.

Suite on this branch: **757 passed, 1 skipped, 37 deselected in 15.02 s** (was 723/3 at
`main` `d0e1534`). The one skip is the absent device. `pytest -m slow`: 11 passed, 2 skipped,
both skips environmental. Everything is off-device; no hardware has been touched.

---

## What changed, per module

### `spatial/m1_frontend.py` (A3, A4, and one erratum the corpus forced)

| # | Change | Why |
|---|---|---|
| 1 | `_fn_location(fn)` → `(co_filename, co_firstlineno)`, used by `capture`'s unreadable-source branch, its no-statement branch and `_fdef_of`; the two `GRAMMAR-NONUNIFORM-DEP` sites use the offending `Statement.line` | All five built a `stage="grammar"` diagnostic with `location=None`, which M0's **I71** rejects — so they raised `ValueError` out of `Diagnostic.__post_init__` instead of the `GrammarError` they meant. `GRAMMAR-NONUNIFORM-DEP` was **unreachable** as a result |
| 2 | `_relative_path(filename)`: `os.path.relpath` when `Path(file).resolve().is_relative_to(Path.cwd())`, else the path unchanged | Architect ruling. An absolute path is machine-dependent, so no grammar or clause golden could carry one. `m2_schedule` imports the same helper — **it is the one helper both modules share** |
| 3 | `ast.increment_lineno(tree, co_firstlineno - 1)` in `capture` | `ast.parse` numbers the snippet `inspect.getsource` returned, so every location and every `Statement.line` counted from the `def`, not from the file. FR-S4 (b) asks for the file **and line**. No golden moves: `test_kernels_live` already normalises `line` |

### `spatial/m2_schedule.py` (A4)

| # | Change | Why |
|---|---|---|
| 1 | `_caller_location` walks to the first frame **outside the package** instead of indexing `inspect.stack()[2]` | `_fail` sits between it and the clause method, and `Schedule.__init__` is reached through `schedule()`, so every `ClauseError` carried a line of `m2_schedule.py` itself. FR-S19 asks it to name what the user wrote |
| 2 | The path goes through M1's `_relative_path` | As above |

### `spatial/m3_legality.py` (A5, and one erratum the corpus forced)

| # | Change | Why |
|---|---|---|
| 1 | `L1-CAPACITY` gains `details["per_buffer"]` — `[operand, span, dtype, bytes]` in §3.10 line 7's order — and `details["doubled"]` | §3.10 line 12 and FR-L9's acceptance both ask for it; `total` alone does not say which tile to halve. Changes the rendered `because:` line, so `tests/golden/reject.L1-CAPACITY.txt` is regenerated and `kernels/rejections.py::bad_capacity`'s docstring re-quoted |
| 2 | `_unit` lifts a tile handle to its **root** axis | `R = ker Sf` is a `UCoord` space, so `reduce(ax.k0, op="+")` raised a bare `ValueError` out of `tuple.index` — an unhandled exception on a user path, which **NFR-7** forbids. `root` is the identity on an untiled name, so no accepted schedule changes |

### `tests/negative/` (A1, A2, A7) — new

`_corpus.py` fixes the naming contract and inverts it: a corpus function is
`raises_<code>()` (one per catalogue code) or `corpus_<code>__<label>()` (a further case for
the same code), with the code spelled in the name, lower-cased, `-` written `_`, and a label
after a double underscore. `code_of` is the inverse, so no table can go stale. `collect` runs a
module's functions once and caches the result, because three tests want the whole corpus and
NFR-3's per-test budget is 3 s.

| Module | Codes | Cases | Notes |
|---|---|---|---|
| `test_grammar.py` | 7 | 36 | Through `sp.kernel(...)`. Covers `03-lld-M1-frontend.md` §7's 28-row table, plus the no-loop, no-store and unreadable-source branches |
| `test_clause.py` | 8 | 32 | Through `sp.schedule(...)` and the clauses. Covers `03-lld-M2-schedule.md` §7's 28 rows; FR-S19's floor is 20 |
| `test_legality.py` | 15 | 18 | Through `.check()`. The three demo set pieces are built by `kernels/rejections.py`, so the corpus and the pitch reject the same schedules |
| `test_mapping.py` | 5 | 5 | Through `m4_selfcheck.self_check` on B's `tests/fixtures/corrupt/` plans |
| `test_emission.py` | 2 | 2 | Through `m5_emit.emit` on B's W1 plan fixture; `EMIT-VERIFY` patches `air.api`'s own `LaunchContext.build` |
| `test_toolchain.py` | 6 | 6 | Through `m6_tools`. **No toolchain binary, no device, no network**: `check_pin`, `PIN`, `shutil.which` and `has_device` are patched inside the function with `unittest.mock.patch`; the two verdict codes use a `ToolRun` literal |

**99 corpus functions over 43 codes**, against `04-test-plan.md` §5's floor of "43 codes, 43
tests minimum".

`test_catalogue.py` closes FR-D3 both ways — the corpus raises every catalogue code, and an
AST scan of `spatial/*.py` finds no code literal §6.3 does not list (reading each module's own
helper `def`, because `m4_mapping._fail(reason, fix, …)` takes a reason first and
`m4_selfcheck._internal(plan, invariant, reason)` takes a plan) — and asserts FR-D1's schema
plus the rendered four-part shape over every diagnostic the corpus raises.

### `tests/unit/test_a_smoke.py`, `tests/integration/test_traceability.py` (A6)

`test_kernel_call_is_oracle` (FR-S1), `test_schedule_is_pure_data` (FR-S5) and
`test_skew_sets_sigma` (FR-S17) are new; the other fifteen FRs are marked on the corpus tests
that exercise them. The full gate is now
`test_traceability_full_gate`, an assertion; `_require_whole_suite` is kept, because a partial
collection cannot prove coverage.

---

## The reachable/unreachable table

**Every one of the 43 codes is reachable, and the corpus raises every one.** The ruling of
2026-09-15 is that a catalogue entry nothing can raise is a defect, so `test_D3` fails rather
than skips. Three needed more than a schedule to reach, and each is recorded here because the
route is part of the claim:

| Code | Reached by | Route |
|---|---|---|
| `GRAMMAR-NONUNIFORM-DEP` | `raises_grammar_nonuniform_dep` | Public surface, **after** the I71 fix above. Before it, the diagnostic could not be constructed at all |
| `HERD-RANK` | `raises_herd_rank` | **Not** reachable from the clauses: `grid(2, 2, 2)` is `CLAUSE-GRID-RANK` at clause time. Reached through M3's own entry point, `m3_legality.check(kernel, schedule)`, on a programmatically built `ScheduleModel` — which is what FR-L11 asks for in as many words: *"even if the clause was constructed programmatically"* |
| `SWAP-PARITY` | `raises_swap_parity` | Needs the `w2_zero_t` shape (`T = 0`). After the D-4 override made an odd trip count accept-and-peel, `T = 0` is the **only** reachable condition — `03-lld-M3-checker.md` §7 says so |

The five `mapping`, two `emission` and six `toolchain` codes are reached through their owners'
module entry points, not the DSL surface, because nothing a user writes can produce them: they
are M4's self-check on a corrupted plan, `air.api`'s own refusals, and tool failures.

---

## Verified (command → result)

Every line was run from the worktree with `PYTHONPATH=<worktree>` and the main checkout's
`.venv` (`pip install -e .` there points at the main checkout, so the `PYTHONPATH` prefix is
what makes the package under test *this* tree — checked with
`python -c 'import spatial;print(spatial.__file__)'` before trusting any run).

| Command | Result |
|---|---|
| `pytest -rA -p no:cacheprovider` | **757 passed, 1 skipped, 37 deselected in 15.02 s**. The skip is `tests/unit/test_m6_tools.py::test_T4_device_diff` — "no NPU: /dev/accel* is absent" |
| — `test_NFR7_all_errors_are_spatial` | **PASSED** (it skipped at `main`: the corpus was empty) |
| — `test_D1_schema`, `test_D3_catalogue_complete` | **PASSED** (neither existed at `main`) |
| — `test_traceability_full_gate` | **PASSED** — an assertion now, not the skip that reported 18 uncovered FRs |
| `pytest --update-goldens` (whole suite) | rewrote `reject.L1-CAPACITY.txt` and `tests/README.md`; nothing else |
| `PYTHONHASHSEED=7 pytest -k "golden or plan or summary"` | **164 passed, 631 deselected in 3.53 s** |
| `pytest -m slow` with the toolchain on `PATH` | **11 passed, 2 skipped in 4.46 s**. Both skips are environmental: `tests/integration/test_smoke.py:43` wants the git-ignored probe sources, and `tests/tt/test_tt_w2.py:191` wants `ttnn` from `.venv-tt` |
| `git diff --exit-code main -- 'tests/golden/*.air.mlir' '*.plan.json' '*.summary.txt' '*.ir_facts.json'` | **clean** (exit 0). The only golden that moved is `reject.L1-CAPACITY.txt`, whose message this pass changed on purpose |

The L1 arithmetic the new breakdown prints, measured on `03-lld-M8-kernels-demo.md` §3.5's
fixture: `86 016 = C 36 864 + A 2×12 288 + B 2×12 288`, `20 480` over the 65 536-byte budget,
`doubled = ('A', 'B')`.

---

## Open, for A

1. **`06-interfaces.md` v3, v4, v5 and v6 have no signatures** (HANDOFF, Person A item 4).
   Unchanged by this pass: it is a signature block in `design/00-README.md` §4, not code.
2. **An unused kernel parameter is `STATIONARITY`.** The architect accepted the code on
   2026-09-15 as the closest in the frozen catalogue, and the message is written for a user;
   the right home is an M1 grammar rejection at capture, which needs a **new catalogue code**
   and therefore the next contract round. `corpus_stationarity__unused_parameter` is the case.
3. **`_check_l1_capacity`'s figure is a lower bound on M4's**, by design (§3.10's scope note):
   M3 charges operand tiles, M4 adds the protocol's own staging scalars. W3 is 232 against
   240. A design within a few bytes of 65 536 would pass M3 and fail M4. Unchanged, recorded.
4. **`test_NFR4_no_bare_raise` does not see a bare `assert`** — it walks `ast.Raise` nodes
   only. `m3_legality._build`'s `assert len(schedule.place) == len(grid)` was the one that
   mattered (a hand-built `ScheduleModel` reached it) and is now a `PLACE-EXTENT` diagnostic
   with `corpus_place_extent__rank_mismatch` behind it; widening the lint to `ast.Assert` is a
   change to C's test file and is left for whoever owns `tests/unit/test_nfr.py` next.

---

## Resume here

Read `hackathon/HANDOFF.md` §"Integration — state at 2026-09-15" first, then this file.

* The corpus's contract is `tests/negative/_corpus.py`'s docstring. **To add a code**: add a
  `raises_<code>()` to the module for its stage, and nothing else — `test_D3`, `test_D1` and
  `test_NFR7` pick it up by name. To add a case for a code that already has one, name it
  `corpus_<code>__<label>()`.
* **To add an FR mark**: put `@pytest.mark.fr("FR-…")` on the test that genuinely exercises it.
  Both traceability gates are assertions now, so a marker naming an FR that does not exist, or
  an FR with no test, fails the suite rather than skipping it.
* **Regenerating a golden** needs a whole-suite `pytest --update-goldens` and a clean
  `tests/golden` (`tests/conftest.py`'s three guards: never in CI, never off the pin, never
  with a dirty tree). `reject.*.txt` is the only golden family this pass may touch; the four
  `*.air.mlir` / `*.plan.json` / `*.summary.txt` / `*.ir_facts.json` families are B's and must
  diff clean against `main`.
* **The next message change** to any of the three demo rejections must update
  `kernels/rejections.py`'s docstring in the same commit: it quotes the rendered text, and it
  is the only place the pitch reads it from.
