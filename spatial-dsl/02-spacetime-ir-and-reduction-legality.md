# Space-time-map IR & reduction legality — worked out

*2026-06-19. Develops the IR core promised in [`01-design-overview.md`](01-design-overview.md) §3 and §5.
Notation is light polyhedral; see [`REFERENCES.md`](REFERENCES.md) for the lineage.*

The goal: an IR where **time** and **space** are explicit affine maps over the iteration
domain, the three canonical dataflows (output-/weight-/row-stationary) are *derived* not
guessed, and **reduction legality** — when may a reduction be spread across PEs vs. kept
local — is a checkable condition, not a user footgun. We then ground every abstract piece
in concrete AIE hardware (the facts in `../docs/notes.html`).

---

## 1. Objects

A perfectly-nested affine kernel gives:

- **Iteration domain** `D ⊆ Z^n` — the integer points of the loop nest (polyhedral).
- For each array operand `a`, an **access map** `F_a : D → Z^{d_a}`, affine, with linear
  part `M_a`. Example (GEMM, `n=3`, indices `(i,j,k)`): `F_A(i,j,k) = (i,k)`,
  `F_B = (k,j)`, `F_C = (i,j)`.
- **Dependences** `δ : producer → consumer` between iterations (RAW/WAR/WAW), each with a
  direction; for uniform recurrences the dependence is a constant vector `d`.

We choose a **space-time map**, two affine functions on iterations:

```
  σ : D → Z^t     (schedule / TIME)   linear part Sσ      — logical issue time
  π : D → Z^s     (allocation / SPACE) linear part Sπ     — PE coordinate (s = 2 for AIE)
```

`T = (σ ; π)` is the whole transform. **This pair is the IR.** Everything a pragma sets
(`place`, `pipeline`, `stationary`, `stream`) is sugar that constrains `σ` and `π`; the
backend reads `(σ, π)` and emits placement, routing, and accumulator code.

---

## 2. Map legality (when is `(σ, π)` a valid mapping at all)

Two conditions, both classical (Quinton 1984; Feautrier 1992):

**(L1) No structural conflict — `(σ, π)` injective on `D`.**
Two distinct iterations may not occupy the *same PE at the same time*:
```
  ∀ x ≠ y ∈ D :  ¬( σ(x) = σ(y)  ∧  π(x) = π(y) )
```
Equivalently `ker Sσ ∩ ker Sπ = {0}`. (A PE does one thing per cycle.)

**(L2) Causality — `σ` respects dependences.**
For every dependence `x → y` (value produced at `x`, used at `y`):
```
  σ(y) ⪰ σ(x) + 1      (lexicographically, strict)
```
For a uniform dependence vector `d`: `Sσ · d ⪰ 1`.

If (L1)+(L2) hold, the map *runs*. Performance and routing come next.

---

## 3. Stationarity is derived, not primitive

This settles **01-design-overview.md open problem #3**.

Operand `a` is **reused** along directions where its access map is constant — the
**reuse space** `Lₐ = ker Mₐ`. Moving one step along `r ∈ Lₐ` needs the *same* value of
`a`. Two cases under `π`:

```
  π(r) = 0  (r ∈ ker Sπ)  ⟹  reuse stays on the SAME PE   ⟹  a is STATIONARY along r
  π(r) ≠ 0                ⟹  the value is needed at OTHER PEs along r
```

> **Refined in [doc 04](04-spatial-triton-dsl.md) §3.** The `π(r) ≠ 0` case splits in two:
> if `a` is *constant* along the spatial `r` (`r ∈ L_a`, as here) the value is **replicated
> → MULTICAST** along that row/column; only if `a` *varies* along `r` is it a true systolic
> **stream**. Multicast vs systolic-forward is then a performance lever, not a fixed choice.

So:

> **Operand `a` is stationary  ⟺  Lₐ = ker Mₐ ⊆ ker Sπ.**

`stationary(a)` in the pragma surface is therefore a *constraint on `π`* — the compiler
either checks it or **solves for a `π`** whose kernel contains `ker Mₐ`. Naming it is
ergonomic sugar (humans think "weight-stationary", not "`ker M_B ⊆ ker Sπ`"), and it
cannot be inconsistent with the map because it *is* a clause of the map's definition.

### 3.1 The three GEMM dataflows = three choices of which index to drop from `π`

A 2-D AIE array gives `π` rank ≤ 2, so `π` maps **two** of `{i,j,k}` to space and leaves
**one** in `ker Sπ`. That excluded index decides the dataflow:

| `π` maps | `ker Sπ` ⊇ | Stationary operand (its reuse axis is killed) | Name |
|---|---|---|---|
| `(i, j)` | `e_k` | **C** — `F_C=(i,j)` reused along `k` | **output-stationary** |
| `(k, j)` | `e_i` | **B** — `F_B=(k,j)` reused along `i` | **weight-stationary** |
| `(i, k)` | `e_j` | **A** — `F_A=(i,k)` reused along `j` | **row/input-stationary** |

This is exactly the "output-stationary-by-omission" we criticized in ARIES (§4.3 of the
paper: `NPU[i%4, 2+j%2]` omits `k`) — but now it is **named, derived, and checkable**
instead of an emergent side effect of an index you forgot to write. Same mechanism,
honest surface.

---

## 4. Reductions: represent them explicitly

Do **not** carry a reduction as a `C[i,j] += …` loop. Carry it as a first-class operator
(the Alpha/AlphaZ decision — Le Verge 1992, Yuki 2012):

```
  C[c] = REDUCE( ⊕ , f , E )        where
    ⊕            associative + commutative accumulation operator   (TAGGED in the IR)
    f : D → C-space   the projection (which indices are summed away)
    E : D → value     the summand expression
```

GEMM: `C[i,j] = REDUCE(+, (i,j,k)↦(i,j), A[i,k]*B[k,j])`. The **reduction space**

```
  R = ker(linear part of f)
```

is the set of directions along which contributions land in the *same* accumulator. For
GEMM `R = span{e_k}`. Representing `f` explicitly makes `R` a first-class object the
legality check can read off — instead of recovering it with dependence analysis.

> **Why the operator must be tagged associative+commutative:** without that tag the
> accumulation chain `…→(i,j,k)→(i,j,k+1)→…` is a *rigid* sequential dependence, and the
> only legal mapping is temporal/output-stationary. The A/C tag is precisely what
> **unlocks reassociation**, hence the freedom to reduce in space. This is the formal
> reason `reduce(k)` is first-class in the pragma surface (01-design-overview.md §2): it is not
> decoration — it changes the legal solution space.

---

## 5. Reduction legality theorem (the core result)

Given `C[c]=REDUCE(⊕,f,E)` with reduction space `R = ker Sf`, `⊕` assoc+comm, and a
space-time map `(σ,π)` satisfying (L1),(L2) for the dependences of `E`. Split `R` by how
`π` treats it:

```
  R_time  = R ∩ ker Sπ        directions kept on one PE   (temporal accumulation)
  R_space = R / R_time        directions that cross PEs    (spatial accumulation)
```

**(R1) Temporal part — output-stationary, free.**
For `r ∈ R_time`: all its contributions are on one PE; `σ` orders them (by (L1),
`Sσ·r ≠ 0`, so they are sequential). Realize as a **single local accumulator register**,
read out at `t_out(c) = max{ σ(x) : f(x)=c }`. Correctness hazard = *do not evacuate `C[c]`
before `t_out(c)`* — now a scheduled fact, not a user worry. (This is the
"don't-write-C-until-the-K-sweep-finishes" hazard from the Figure-4 deep dive, discharged.)

**(R2) Spatial part — cascade/tree, legal *because* `⊕` is A/C.**
For `R_space ≠ ∅`, contributions to `C[c]` are computed on different PEs and must be
combined by a **reduction network**. Legality:

1. `⊕` assoc+comm ⟹ partials may be combined in *any* order (the reassociation that the
   tag bought us);
2. the network needs its own schedule `σ'` on the "combine tree" with each combine placed
   *after* both its inputs: `σ'(combine) ⪰ max(σ-of-left, σ-of-right) + 1`;
3. each combine edge PE→PE must be **routable** (see §6) within its time slack.

**(R3) Hybrid — tile `R`.**
General case `R = R_time ⊕ R_space`: accumulate locally along `R_time`, then cascade the
per-PE partials along `R_space`. For GEMM this is exactly the multi-level `k`-tiling
(`k0/k1` temporal, `k2` spatial) we saw hand-written in the paper's Figure 4 — here it is
*derived* from the `(σ,π)` choice on the single reduction axis, with the partial-sum
network synthesized, not hand-placed.

**Legality summary (checkable):** a reduction mapping is legal iff
`(L1)∧(L2)` hold for `E`'s deps, **and** (`R_space = ∅`) **or** (`⊕` is tagged A/C **and**
every `R_space` combine edge is routable under (R2.2)). Reject otherwise — surfacing
"un-mappable schedule" rather than silently miscompiling.

---

## 6. AIE realization — grounding in the hardware (from `../docs/notes.html`)

The abstract `(σ,π)` + reduction network lower onto concrete AIE mechanisms. This is where
our design earns its keep over a generic systolic synthesizer — it targets the real
memory/forwarding model:

| Abstract construct | AIE mechanism (notes: AIE arch / VCK5000 tabs) |
|---|---|
| Stream of operand `a` to **adjacent** PE (`‖π(r)‖=1`) | **shared-L1 neighbor access** (a tile reads N/S/W neighbor's L1) |
| Stream to **non-adjacent** PE (`‖π(r)‖>1`) | **AXI4-Stream** through stream-switch hops |
| **Spatial reduction** (R2) along a line of PEs | **cascade stream** — the dedicated unilateral datapath for *accumulator chaining / partial-sum forwarding*. This is a hardware-perfect match for a linear cascade reduction. |
| **Temporal reduction** (R1) | local **accumulator register** (512-bit AIE2 / 2048-bit AIE2P) |
| Operand residence `reside(a:L2)` + broadcast | **memory tile (L2, 512 KB/col)** staging + multicast down a column |
| Boundary feed `io_budget` | **shim DMA / PLIO** bandwidth at the array edge |

Two AIE-specific legality riders the generic theory does not give you:

- **Cascade is unidirectional and linear.** R2's combine tree must degrade to a *chain*
  along a physical row/column when realized on the cascade stream. So for AIE, prefer
  `R_space` of rank 1 mapped along a contiguous PE line; a rank-2 spatial reduction needs
  either two cascade passes or an AXIS-based tree (costlier).
- **L1 is a hard 64 KB scratchpad, no cache** (notes: "exceeding L1 is a *compile*
  failure"). So `reside`/`double_buffer` choices feed a capacity constraint the mapper
  must satisfy *before* `(σ,π)` is accepted — unlike CPU/GPU where it's a perf knob.

---

## 7. Pragma surface → IR lowering

How the 01-design-overview.md strawman pragmas become `(σ, π, REDUCE)`:

```
  parallel(i,j) reduce(k)      ⟹  declare REDUCE with f killing k; R = span{e_k}; tag ⊕=+ A/C
  place(i->px, j->py)          ⟹  Sπ rows = [e_i; e_j]   (so ker Sπ ⊇ e_k)
  stationary(C)                ⟹  constraint ker M_C ⊆ ker Sπ  (consistency check on the above)
  pipeline(k) ii(auto)         ⟹  σ has k innermost-fastest; II = free coefficient solved by search
  stream(A:forward(px), B:forward(py))  ⟹  routing pattern for the streamed operands (A along rows, B along cols)
  reside(A:L2,B:L2,C:L1) double_buffer(A,B)  ⟹  memory-level + capacity constraints feeding §6
```

The compiler then: (1) builds `D, F_a, f` from the nest; (2) accepts/solves `σ,π` under
(L1),(L2) and the `stationary` constraint; (3) classifies `R` into `R_time/R_space`;
(4) checks reduction legality §5; (5) synthesizes the §6 AIE mechanisms; (6) emits code.
Strip the pragmas → plain sequential nest = the **reference oracle** (01-design-overview.md §1).

---

## 8. What is solved vs. open

**Solved / borrowed (cite, don't reinvent):**
- `(σ,π)` map legality (L1),(L2): Quinton 1984; Feautrier 1992; Darte-Robert-Vivien 2000.
- Stationarity = `ker` containment: systolic-synthesis folklore, formalized in
  Darte-Robert-Vivien.
- Reduction A/C-reassociation legality + temporal/spatial choice: Redon-Feautrier 1994;
  Gupta-Rajopadhye 2006; explicit-reduce IR: Alpha / AlphaZ.

**Open (our actual contribution surface):**
1. **AIE-specific routing & deadlock legality** for the R2 combine network on
   cascade-vs-AXIS-vs-shared-L1, with bounded FIFOs (Kahn-style proof). The generic theory
   assumes routability; AIE's finite stream channels and unidirectional cascade do not.
2. **Pragma surface design + solver**: turning the clauses of §7 into an LP/ILP over
   `σ,π` coefficients that also honors the §6 L1-capacity hard constraint.
3. **Virtual→physical folding** under a fixed array size (4×5 / 8×8): Moldovan-Fortes
   partitioning, but re-derived for AIE columns + memory-tile structure.
4. **II / register-pressure interaction**: `R_time` length bounds live accumulators per
   PE vs. the pipeline II `σ` search — a joint, not separable, optimization.

---

## 9. End-to-end worked example: output-stationary GEMM

```
Algorithm:   C[i,j] = REDUCE(+, (i,j,k)↦(i,j), A[i,k]*B[k,j]),   D = [0,M)×[0,N)×[0,K)
Pragmas:     parallel(i,j) reduce(k); place(i->px,j->py); stationary(C); pipeline(k)
Lower:
  Sπ = [[1,0,0],[0,1,0]]      ⟹ ker Sπ = span{e_k}
  R  = ker Sf = span{e_k}     ⟹ R ⊆ ker Sπ  ⟹ R_time = R, R_space = ∅
  σ  = (i, j, k)  (k fastest) ⟹ (L2) ok: Sσ·e_k = 1 > 0 for the accumulation order
  Check (L1): ker Sσ ∩ ker Sπ = {0} ✓     stationary(C): ker M_C = span{e_k} ⊆ ker Sπ ✓
Reduction legality: R_space = ∅  ⟹ trivially legal (R1 only), no network needed.
Realize (§6):
  PE(px,py) holds C[i,j] in a local accumulator (output-stationary).
  A[i,k] streams along rows (reused over j = e_j; π(e_j)=e_py ≠ 0 ⟹ streams) → AXIS / shared-L1.
  B[k,j] streams along cols (reused over i)                                  → AXIS / shared-L1.
  Evacuate C[i,j] at t_out = σ(i,j,K-1); not before. (Figure-4 hazard discharged.)
```

Flip one pragma — `place(i->px, k->py)` instead — and the *same algorithm* re-lowers to
`R_space = span{e_k}`, a **cascade reduction** down a PE column (R2), with **A** now
stationary. No kernel edit. That is the property ARIES's tile model could not give
cleanly, obtained here from the `(σ,π)` IR.
