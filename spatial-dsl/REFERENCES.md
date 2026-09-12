# Reading list — space-time-map IR & reduction legality

Curated for the `spatial-dsl` design (see [`01-design-overview.md`](01-design-overview.md) and
[`02-spacetime-ir-and-reduction-legality.md`](02-spacetime-ir-and-reduction-legality.md)).
Priority-ordered; the **Critical Path** is the minimum to understand the IR + reduction
legality theory we develop.

## Critical path (read these 5 first)
1. **Karp, Miller, Winograd**, "The Organization of Computations for Uniform Recurrence
   Equations," *JACM* 14(3), 1967. — origin of scheduling over uniform recurrences.
2. **Quinton**, "Automatic Synthesis of Systolic Arrays from Uniform Recurrent
   Equations," *ISCA* 1984. — the canonical space-time (σ/π) synthesis.
3. **Feautrier**, "Some Efficient Solutions to the Affine Scheduling Problem, Part I &
   II," *IJPP* 21, 1992. — the computable affine-scheduling formalism (the σ map).
4. **Gupta & Rajopadhye**, "Simplifying Reductions," *POPL* 2006, pp. 30–41. — reductions
   first-class in an equational/polyhedral IR; legality + complexity reduction.
   <https://dl.acm.org/doi/10.1145/1111320.1111041>
5. **Yuki, Gupta, Kim, Pathan, Rajopadhye**, "AlphaZ: A System for Design Space
   Exploration in the Polyhedral Model," *LCPC* 2012. — the closest existing realization:
   polyhedral equational IR with reductions + space-time first-class.
   <https://link.springer.com/chapter/10.1007/978-3-642-37658-0_2>

## A. Space-time mapping foundations (systolic synthesis)
- **Lamport**, "The Parallel Execution of DO Loops," *CACM* 17(2), 1974. — hyperplane
  method; origin of affine *time* scheduling.
- **Kung & Leiserson**, "Systolic Arrays (for VLSI)," 1978 (Mead & Conway, *Intro to VLSI
  Systems*, 1980).
- **H.T. Kung**, "Why Systolic Architectures?," *IEEE Computer* 15(1), 1982.
- **Moldovan**, "On the Design of Algorithms for VLSI Systolic Arrays," *Proc. IEEE*
  71(1), 1983.
- **Moldovan & Fortes**, "Partitioning and Mapping Algorithms into Fixed Size Systolic
  Arrays," *IEEE TC* C-35(1), 1986. — partitioning onto a fixed array = the
  virtual→physical folding problem.
- **Rao & Kailath**, "Regular Iterative Algorithms and their Implementation on Processor
  Arrays," *Proc. IEEE* 76(3), 1988.
- **Quinton & Van Dongen**, "The Mapping of Linear Recurrence Equations on Regular
  Arrays," *J. VLSI Signal Processing* 1(2), 1989.
- **S.Y. Kung**, *VLSI Array Processors*, Prentice Hall, 1988. (book)
- **Darte, Robert, Vivien**, *Scheduling and Automatic Parallelization*, Birkhäuser, 2000.
  (book — **best single secondary source** unifying dependence + space-time)

## B. Polyhedral scheduling & allocation (space-time, made computable)
- **Feautrier**, "Dataflow Analysis of Array and Scalar References," *IJPP* 20(1), 1991.
- **Lim & Lam**, "Maximizing Parallelism and Minimizing Synchronization with Affine
  Partitions," *POPL* 1997 (jrnl: *Parallel Computing* 24, 1998). — affine partitioning =
  the π (space) map.
- **Bondhugula et al.**, "A Practical Automatic Polyhedral Parallelizer and Locality
  Optimizer" (Pluto), *PLDI* 2008.
- **Verdoolaege**, "isl: An Integer Set Library for the Polyhedral Model," *ICMS* 2010.

## C. Reductions — detection, scheduling, legality, simplification
- **Redon & Feautrier**, "Detection of Recurrences in Sequential Programs with Loops,"
  *PARLE* 1993. — finding reductions.
- **Redon & Feautrier**, "Scheduling Reductions," *ICS* 1994. — scheduling them under
  affine maps; directly the legality question.
- **Fisher & Ghuloum**, "Parallelizing Complex Scans and Reductions," *PLDI* 1994.
- **Le Verge**, "Reduction Operators in Alpha," *CONPAR/VAPP* 1992.
- **Yuki & Rajopadhye et al.**, "Simplifying Dependent Reductions in the Polyhedral
  Model," arXiv:2007.11203 / *IMPACT* 2021.
- "Maximal Simplification of Polyhedral Reductions," *PACMPL/POPL* 2025 (Rajopadhye
  group). <https://dl.acm.org/doi/abs/10.1145/3704839>

## D. Equational / reduction-first IRs (closest prior art to *our IR*)
- **Le Verge, Mauras, Quinton**, "The ALPHA Language and its Use for the Design of
  Systolic Arrays," *J. VLSI Signal Processing* 3, 1991. — Alpha: equational language for
  systolic synthesis with explicit space-time.
- **AlphaZ** (LCPC 2012, above) and **MMAlpha** — the tool realizations.

## E. Modern spatial DSLs/compilers (design context)
- **Srivastava, Rong, Barua et al.**, "T2S-Tensor: Productively Generating
  High-Performance Spatial Hardware for Dense Tensor Computations," *FCCM* 2019. — **read
  early.** Decouples functional spec from spatial mapping; implements MTTKRP/TTM/TTMc (the
  same kernels as ARIES) six years earlier. <https://nitish2112.github.io/publication/t2s-tensor/>
- **Lai, Rong, Zheng et al.**, "SuSy: A Programming Model for Productive Construction of
  High-Performance Systolic Arrays on FPGAs," *ICCAD* 2020. — Halide-based systolic
  construction with space-time transforms. <https://ieeexplore.ieee.org/document/9256583>
- **Ragan-Kelley et al.**, "Halide," *PLDI* 2013. — algorithm/schedule separation origin.
- **Koeplinger et al.**, "Spatial: A Language and Compiler for Application Accelerators,"
  *PLDI* 2018. — first-class `SRAM`/`FIFO`/`Stream`/`FSM`; escape-hatch source.
- **Lai et al.**, "HeteroCL," *FPGA* 2019; **Chen et al.**, "Allo," *PLDI* 2024 (in this
  repo). — decoupled MLIR substrate.
- **Lattner et al.**, "MLIR," *CGO* 2021.

## Honest framing
Clusters **C + D** (Simplifying Reductions, AlphaZ, Alpha) already *are* "a polyhedral
equational IR with first-class reductions and space-time choice." Our contribution is
**not** green-field: it is re-surfacing that lineage under a **pragma frontend**,
retargeted to **AIE's concrete memory/forwarding model** (cascade stream, shared-L1
neighbor access, AXIS, memory tiles). The *theory* of reduction legality is largely
solved; the *pragma surface + AIE-specific routing/deadlock legality* is the open part.
