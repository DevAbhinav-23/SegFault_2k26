# Hackathon working directory — Spatial DSL for NPUs

This directory holds the SegFault 2026 (IICT / compilertech.org) entry: a **spatial DSL for NPUs
that lowers to MLIR-AIR**, together with its design-decision trail. Start with
[`HANDOFF.md`](HANDOFF.md), then [`01-paradigm-comparison.md`](01-paradigm-comparison.md) (rev r3),
which picks a **pragma surface over a declared-intent SPMD per-PE core** emitted through
mlir-air's `air.api`, and records the rival option (ship the legality checker alone). The
`REDTEAM-round*.md` / `RESPONSE-round*.md` pairs are the adversarial review that produced r3.
Scope is fixed: NPUs only, DSL + lowering only, 3–4 dense kernels. See `/CLAUDE.md` at the repo
root for the binding scope rules.

## Folder map (paths relative to the repo root)

| Path | Contents |
| --- | --- |
| `hackathon/` | This entry: handoff, paradigm comparison r3, red-team and response rounds. |
| `spatial-dsl/` | Design docs 01–05, README, REFERENCES.md (the long-form DSL design). |
| `proposals/` | `spatial_dsl_project_proposal.md` — the research proposal this is cut from. |
| `reading-group/` | Syllabus, abstraction ledger, spatial-accelerator landscape tutorial. |
| `notes/` | `mlir-air-notes.tex` / `.pdf` — the MLIR-AIR source-reading notes. |
| `docs/` | `notes.html` — ARIES / paper notes referenced by `spatial-dsl/README.md`. |
| `papers/` | `maestro.md` only (our own notes). Third-party PDFs are **not** here. |

## Provenance

Context copied from the amd-npus research repo on 2026-09-12; third-party PDFs and the private
`PACT/56.txt` reviews file were intentionally not copied.

`PACT/56.txt` (peer reviews of the group's own PACT 2026 paper) is **intentionally not copied** and
lives only in the amd-npus repo at `/home/adi/Projects/Honours/amd-npus/PACT/56.txt`; links to it in
these documents are left in place, and Appendix A of `01-paradigm-comparison.md` already quotes the
lines it relies on. Links to `papers/allo.pdf` and `papers/loopnest2silicon.pdf` likewise dangle
here by design — both are listed in `spatial-dsl/REFERENCES.md`.

## Two inputs the team must decide

1. **Team size N.** Several conclusions in `HANDOFF.md` turn on it — most sharply, if **N = 1 and
   experiment E1 fails**, the recommendation flips to the rival (checker-only) plan.
2. **Confirm Sept 19–20 2026 is the evaluation date.** Taken from the venue site on 2026-09-12 and
   not re-confirmed since; the whole schedule hangs off it.

## Next action

Run experiments **E1–E5** from §9 of `01-paradigm-comparison.md` (`## 9. Open experiments
(day-1 list)`; the same table is summarised in `HANDOFF.md`). **E1 and E2 gate everything else.**
