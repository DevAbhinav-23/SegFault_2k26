# Project Proposal: Spatial DSL — An Architecture-Visible Programming Abstraction for Spatial Accelerators

## 1. Project Summary

Modern programmable spatial accelerators such as AMD NPUs/AIE arrays and Tenstorrent processors expose large arrays of compute tiles connected through on-chip communication networks and equipped with distributed local memories. These architectures offer substantial compute capability and energy efficiency, but programming them efficiently requires reasoning about far more than the computation itself. The programmer or compiler must make decisions about spatial decomposition, placement, communication, buffering, synchronization, and pipelining.

Domain-specific languages (DSLs), on the other hand, are intentionally designed to hide architectural detail and let programmers express computation naturally for a particular domain. Halide, for example, provides a powerful abstraction for image-processing pipelines. However, directly compiling such DSLs to a spatial accelerator typically requires substantial architecture-specific compiler logic.

Our recent work demonstrated an end-to-end **Halide-to-AMD-NPU compiler**, published at **PACT 2026**. That experience motivates the next research question:

> **Can the architectural knowledge required to efficiently map domain-specific programs onto spatial accelerators be captured in a common, architecture-visible but architecture-independent programming abstraction?**

We propose to investigate a new abstraction, tentatively called **Spatial DSL** and its compiler representation, **Spatial IR**.

Spatial DSL is intended to occupy the middle ground between high-level domain DSLs and low-level accelerator APIs. It exposes the important architectural choices—computation placement, local storage, communication, replication, and pipelining—while hiding device-specific mechanisms such as DMA descriptors, NoC packet configuration, circular-buffer APIs, synchronization primitives, and runtime boilerplate.

The project will use:

- **Halide** as the first high-level frontend,
- **AMD NPU** as an existing, validated backend,
- **Tenstorrent** as the first new architecture used to test portability of the abstraction.

The long-term objective is a reusable compiler substrate for mapping multiple domain DSLs onto multiple programmable spatial accelerators.

---

## 2. Motivation

### 2.1 The programming gap in spatial accelerators

Traditional compiler abstractions work well for CPUs and GPUs because common execution models—threads, vector instructions, caches, and shared memory—have relatively mature compiler representations.

Programmable spatial accelerators present a different execution model.

A typical spatial architecture consists of:

- a two-dimensional array of compute tiles,
- private or tile-local memories,
- an on-chip communication network,
- explicit data movement,
- bounded communication buffers,
- independent or partially independent compute and data-movement engines,
- concurrent execution across tiles.

Efficient execution therefore requires the compiler to answer questions such as:

- Which computation should execute on each tile?
- How should tensors be partitioned?
- Which data should remain local?
- Which values should be streamed?
- Which data should be multicast?
- How large should communication buffers be?
- Which stages should execute concurrently?
- How should computation and data movement overlap?
- Can the resulting communication schedule deadlock?
- How should a logical mapping be realized using the mechanisms of a particular architecture?

Existing high-level DSLs generally do not expose these decisions directly, while low-level accelerator programming models expose too much implementation detail.

This creates a substantial abstraction gap.

---

## 3. Prior Work and Starting Point

A key starting point for this project is our existing **Halide-to-AMD-NPU compiler**, published at **PACT 2026**.

That compiler establishes that a high-level functional pipeline expressed in Halide can be transformed into an execution strategy for a programmable spatial accelerator.

Conceptually, the compiler performs transformations of the form:

```text
Halide computation
        |
        v
Spatial decomposition
Placement
Communication planning
Buffering
Pipelining
        |
        v
AMD-NPU-specific realization
```

The central observation motivating this proposal is that the middle portion of this flow is not inherently AMD-specific.

For example, concepts such as

- divide a computation across spatial tiles,
- place producer and consumer stages near each other,
- retain intermediate values on-chip,
- stream values between stages,
- multicast shared inputs,
- double-buffer communication,
- overlap data movement and computation,

are general spatial-execution concepts.

However, their implementations differ significantly across architectures.

This project proposes to make these **spatial decisions first-class compiler abstractions** rather than embedding them implicitly inside a device-specific backend.

---

## 4. Central Research Hypothesis

The project is based on the following hypothesis:

> **A small set of architecture-visible spatial abstractions can express efficient mappings across multiple programmable spatial accelerators while insulating domain DSLs from device-specific programming mechanisms.**

We distinguish three levels of abstraction.

### Level 1: Domain computation

The programmer describes **what** should be computed.

Examples include:

```text
Halide:
blur -> sharpen -> resize
```

or

```text
ML:
MatMul -> ReLU -> MatMul
```

or

```text
Genomics:
seed -> filter -> alignment
```

### Level 2: Spatial intent

The program or compiler describes **how the computation should inhabit the spatial architecture**.

Examples:

```text
partition tensor into tiles
map computation across a 4x4 region
multicast A across rows
stream intermediate values
double-buffer input
pipeline load / compute / store
```

### Level 3: Machine mechanism

The backend determines **how that intent is implemented on a specific machine**.

For example:

```text
AMD NPU:
AIE tiles
ObjectFIFOs
DMA
shim tiles
locks / synchronization
```

versus

```text
Tenstorrent:
Tensix cores
circular buffers
NoC transfers
data-movement kernels
semaphores
```

The proposed Spatial DSL/IR primarily occupies **Level 2**.

---

## 5. Design Principle

The guiding design principle is:

> **Expose architectural intent; hide architectural mechanism.**

Spatial DSL should make important performance decisions visible to the compiler programmer or expert user, without requiring them to manage hardware-specific details.

A second way of stating this principle is:

> **Architectural visibility without architectural micromanagement.**

The abstraction must therefore avoid both extremes.

At one extreme:

```text
C = A @ B
```

provides too little architectural control.

At the other extreme:

```text
configure DMA
allocate channel
program packet route
configure semaphore
launch data-movement kernel
configure circular buffer
...
```

provides too much low-level mechanism.

Spatial DSL seeks the useful middle ground.

---

## 6. Proposed Spatial Abstractions

We initially propose six core concepts.

### 6.1 Compute

A logical unit of computation.

Examples:

```text
compute blur_x
compute matmul
compute softmax
compute alignment_tile
```

Compute operations may eventually encapsulate vector kernels, tensor kernels, scalar kernels, or domain-specific kernels.

---

### 6.2 Partitioning

How a computation or data domain is decomposed.

Example:

```text
partition C into tiles [TM, TN]
```

or

```text
partition image along y into strips
```

Partitioning describes logical decomposition independently of physical device coordinates.

---

### 6.3 Placement

Where computation executes.

Example:

```text
place matmul_tiles on cores [0:3, 0:3]
```

or

```text
map pipeline_stage_1 to row 2
```

Placement can initially be explicit and later be inferred or optimized automatically.

---

### 6.4 Memory

Where intermediate and persistent data reside.

Example:

```text
buffer A local
buffer B local
buffer intermediate streaming
```

Potential memory classes could include:

- external,
- local,
- shared,
- streaming,
- replicated.

The abstraction should express intent rather than architecture-specific memory banks.

---

### 6.5 Communication

How values move between computations.

Potential communication primitives include:

```text
send
receive
broadcast
multicast
stream
gather
scatter
```

Example:

```text
multicast A across rows
multicast B across columns
```

Communication should be explicit enough for analysis and optimization while remaining independent of the underlying NoC APIs.

---

### 6.6 Concurrency and Pipelining

Spatial architectures rely heavily on overlapping computation and communication.

An abstract construct could resemble:

```text
pipeline {
    load
    compute
    store
}
```

or

```text
pipeline stage0 -> stage1 -> stage2
```

The compiler backend determines the required buffering and synchronization mechanisms.

---

## 7. Illustrative Example

Consider tiled matrix multiplication:

```text
C = A x B
```

A high-level spatial program might describe:

```text
spatial matmul {

    partition C into [TM, TN]

    map C[i,j] to core[i,j]

    distribute A by rows
    distribute B by columns

    buffer A double
    buffer B double

    pipeline {
        load
        compute
        store
    }
}
```

The same logical mapping could then be implemented differently on different architectures.

### AMD realization

```text
Spatial multicast
    ->
ObjectFIFO / DMA distribution

Spatial local buffer
    ->
AIE tile memory

Spatial placement
    ->
AIE tile coordinates
```

### Tenstorrent realization

```text
Spatial multicast
    ->
NoC multicast / data-movement kernels

Spatial local buffer
    ->
L1 / circular buffers

Spatial placement
    ->
Tensix core coordinates
```

The Spatial program expresses the algorithmic mapping without exposing either machine's low-level APIs.

---

## 8. Proposed Compiler Architecture

The envisioned compiler stack is:

```text
                   Halide
                      |
              Future Domain DSLs
                      |
                      v
             +------------------+
             |    Spatial IR    |
             +------------------+
             | partitioning     |
             | placement        |
             | memory           |
             | communication    |
             | replication      |
             | pipelining       |
             +------------------+
                    / \
                   /   \
                  v     v
          AMD-NPU        Tenstorrent
          Backend        Backend
             |               |
        MLIR-AIE /       TT-MLIR /
        IRON / AIE       TT-Metalium
```

Spatial IR serves as the central compiler contract.

Spatial DSL may provide a textual or embedded user-facing representation of the same concepts.

This separation permits two usage modes.

### Mode A: Compiler-generated Spatial IR

```text
Halide
   |
   v
Spatial IR
   |
   v
Target backend
```

Here the frontend compiler automatically derives a spatial execution strategy.

### Mode B: Expert-authored Spatial DSL

```text
Spatial DSL
   |
   v
Spatial IR
   |
   v
Target backend
```

An architecture-aware programmer can directly describe a spatial mapping without writing low-level target code.

This makes Spatial IR the meeting point between **automatic compilation** and **expert-guided spatial programming**.

---

## 9. Research Questions

### RQ1. What is the minimal useful spatial abstraction?

Can computation, partitioning, placement, memory, communication, and concurrency capture the important mapping decisions required for modern programmable spatial accelerators?

A central goal is to identify what belongs in a common abstraction and what must remain architecture-specific.

---

### RQ2. Can spatial mappings be portable across architectures?

We distinguish ordinary functional portability from **mapping portability**.

Functional portability asks:

> Can the same computation execute on two architectures?

Mapping portability asks:

> Can the same high-level decomposition, locality strategy, communication pattern, and concurrency structure be preserved across two different spatial architectures?

For example:

```text
partition image into strips
pipeline neighboring stages
retain intermediate values on-chip
double-buffer communication
```

should ideally remain valid regardless of whether the target is AMD NPU or Tenstorrent.

---

### RQ3. What architecture-specific information must leak through the abstraction?

Perfect architecture independence is unlikely to be possible or desirable.

The project will investigate:

- Which spatial concepts generalize cleanly?
- Which require parameterization?
- Which require target-specific annotations?
- When are architecture-specific escape hatches unavoidable?

This boundary is itself an important research result.

---

### RQ4. Can Spatial IR support automatic optimization?

Once mapping decisions are explicit, the compiler can explore alternatives across:

```text
tiling
placement
communication
buffering
fusion
replication
pipelining
```

The optimization problem can be expressed conceptually as:

\[
\min
\left(
\alpha T_{\text{compute}}
+
\beta T_{\text{communication}}
+
\gamma T_{\text{external-memory}}
\right)
\]

subject to constraints such as:

\[
M_{\text{local}} \leq M_{\text{available}}
\]

\[
BW_{\text{NoC}} \leq BW_{\text{available}}
\]

and finite buffer, compute, routing, and synchronization resources.

---

### RQ5. Can the abstraction enable correctness analysis?

Making communication and buffering explicit opens the possibility of compiler analyses for:

- bounded-buffer execution,
- producer-consumer compatibility,
- synchronization correctness,
- circular wait,
- deadlock freedom.

This is particularly important because spatial architectures frequently employ bounded channels and backpressure.

A Spatial IR could enable construction of a channel dependency graph and static detection of potentially unsafe execution schedules.

---

## 10. Project Phases

### Phase 1 — Extract Spatial IR from the existing AMD compiler

The first step is not to design Spatial IR from scratch.

Instead, we will analyze the existing Halide-to-AMD-NPU compiler and identify the architecture-independent decisions already being made.

Candidate concepts include:

- Func-to-tile mapping,
- pipeline fusion,
- data tiling,
- communication edges,
- local buffer creation,
- streaming,
- multicast,
- compute/data-movement overlap.

The existing backend will then be refactored conceptually as:

```text
Halide
   |
   v
Spatial IR
   |
   v
AMD backend
```

Success criterion:

> The refactored compiler should retain the functionality and performance characteristics of the existing Halide-to-AMD compiler while making spatial decisions explicit in an intermediate representation.

---

### Phase 2 — Define a minimal Spatial DSL/IR

We will design a small initial IR supporting only the abstractions required by a carefully selected set of workloads.

The first version should deliberately remain small.

Candidate operations:

```text
spatial.compute
spatial.partition
spatial.place

spatial.buffer
spatial.stream

spatial.send
spatial.receive
spatial.multicast

spatial.pipeline
spatial.replicate
```

The design should avoid premature inclusion of target-specific mechanisms.

---

### Phase 3 — Build a Tenstorrent backend

The first major portability test will be lowering Spatial IR to a Tenstorrent-like execution model.

The backend will map spatial concepts onto:

- Tensix cores,
- local L1 memory,
- circular buffers,
- NoC communication,
- compute kernels,
- data-movement kernels.

Initial applications should be deliberately simple:

1. tiled matrix multiplication,
2. element-wise producer-consumer pipeline,
3. image-processing pipeline,
4. small fused ML graph.

The goal is initially correctness and expressiveness, followed by performance.

---

### Phase 4 — Cross-architecture comparison

The same Spatial IR program will be lowered to both:

```text
AMD NPU
```

and

```text
Tenstorrent
```

We will study:

- how much IR is common,
- how many target-specific annotations are required,
- whether the same logical placement strategy remains effective,
- whether the same communication strategy is realizable,
- differences in buffering and synchronization,
- performance loss relative to target-specific implementations.

This phase directly evaluates the hypothesis of mapping portability.

---

### Phase 5 — Halide → Spatial IR → Tenstorrent

Once the Spatial IR and Tenstorrent backend stabilize, the existing Halide frontend can target Spatial IR.

The complete flow becomes:

```text
Halide
   |
   v
Spatial IR
   |
   v
Tenstorrent
```

This is intentionally different from implementing a direct Halide-to-Tenstorrent compiler.

The research question becomes:

> Can the same Halide frontend and spatial mapping abstraction support fundamentally different spatial machines?

---

### Phase 6 — Automatic spatial scheduling

The next step is to distinguish:

```text
computation
```

from

```text
spatial schedule
```

A frontend could produce an unscheduled or partially scheduled Spatial IR, after which the compiler searches over:

```text
tile sizes
number of cores
placement
fusion
replication
buffer sizes
communication strategies
pipeline depth
```

Possible techniques include:

- analytical cost models,
- rule-based scheduling,
- design-space exploration,
- autotuning,
- learned cost models.

---

## 11. Workloads

The project should initially avoid attempting full neural networks or arbitrary Halide pipelines.

A staged benchmark suite is preferable.

### Stage 1: Primitive kernels

- MatMul
- elementwise operations
- reduction
- stencil
- convolution

### Stage 2: Producer-consumer pipelines

- blur pipeline
- Sobel pipeline
- convolution + activation
- MatMul + activation
- chained stencil computations

### Stage 3: Fused ML operators

- Linear + GELU
- MatMul + bias + ReLU
- QKᵀ + softmax
- small attention block

### Stage 4: Domain DSL programs

- image-processing pipelines from Halide,
- potentially genomics kernels,
- potentially stencil or scientific-computing DSLs.

Using domains beyond ML will help establish that Spatial DSL is an architecture abstraction rather than another ML-specific language.

---

## 12. Evaluation

### 12.1 Expressiveness

Can representative spatial mappings be described without target-specific constructs?

Metrics may include:

- number of target-independent operations,
- number of target-specific annotations,
- number of unsupported mappings,
- amount of backend-specific escape code.

---

### 12.2 Programmability

Compare Spatial DSL against low-level implementations.

Potential metrics:

- lines of code,
- number of explicit synchronization operations,
- number of explicit transfers,
- number of hardware-specific concepts exposed to the programmer.

---

### 12.3 Mapping Portability

For the same logical program, measure how much of the following can remain unchanged:

- decomposition,
- placement structure,
- communication strategy,
- memory strategy,
- pipeline structure.

A useful metric could quantify the fraction of the spatial program that remains target independent.

---

### 12.4 Performance

Compare against:

- existing Halide-to-AMD compiler output,
- native AMD implementations,
- native Tenstorrent implementations where available.

Metrics:

- execution time,
- throughput,
- compute utilization,
- external-memory traffic,
- NoC traffic,
- local-memory usage.

The objective is not necessarily to outperform vendor compilers initially.

The important question is:

> What performance cost, if any, is introduced by the architecture-independent spatial abstraction?

---

### 12.5 Compiler optimization quality

For automatically generated mappings, compare:

```text
baseline mapping
vs.
optimized spatial mapping
```

Study the impact of:

- communication-aware placement,
- fusion,
- multicast,
- double buffering,
- pipeline overlap,
- local-memory optimization.

---

## 13. Potential Novel Contributions

A mature version of the project could make the following contributions.

### Contribution 1: Spatial execution as a first-class compiler abstraction

An IR explicitly representing:

```text
computation
+
spatial decomposition
+
placement
+
memory
+
communication
+
concurrency
```

without committing to a specific accelerator.

---

### Contribution 2: Architecture-visible portability

A programming abstraction that differs from conventional hardware abstraction by deliberately preserving architectural structure.

The goal is not to hide the machine completely, but to expose the aspects that matter for spatial performance.

---

### Contribution 3: Mapping portability

A definition and experimental study of portability of **execution mappings**, not merely portability of computations.

---

### Contribution 4: Multiple backends

Demonstration on at least two significantly different programmable spatial architectures:

- AMD NPU/AIE,
- Tenstorrent.

---

### Contribution 5: Domain DSL integration

Demonstration that a domain DSL such as Halide can generate Spatial IR automatically.

---

### Contribution 6: Communication-aware compiler analysis

Potential static analysis and optimization of:

- buffering,
- routing,
- backpressure,
- deadlock,
- synchronization.

---

## 14. Important Research Risks

### 14.1 Spatial IR could become too close to an existing architecture IR

A major risk is that the proposed abstraction becomes only a renamed version of an existing target-specific IR.

The project must therefore continually test:

> Can this operation be interpreted naturally on both AMD and Tenstorrent?

If not, it may belong in a lower-level backend IR.

---

### 14.2 The abstraction may be too high level

If Spatial IR merely contains operations such as:

```text
matmul
conv
relu
```

then it does not expose enough of the architecture to justify a new abstraction.

The IR must make mapping decisions explicit.

---

### 14.3 The abstraction may be too low level

If the IR contains:

```text
DMA descriptor
hardware channel number
physical semaphore
packet routing field
```

then architecture portability has already been lost.

---

### 14.4 AMD and Tenstorrent may differ in important ways

This is not necessarily a weakness.

Understanding precisely which concepts generalize and which do not is one of the core scientific questions of the project.

---

## 15. Recommended Student Work Packages

The project can support multiple students working in parallel.

### Student A — Spatial IR design

Responsibilities:

- analyze existing Halide-to-AMD compiler,
- identify implicit spatial decisions,
- design initial Spatial IR,
- implement verifier and textual representation.

---

### Student B — AMD backend refactoring

Responsibilities:

- lower Spatial IR to the existing AMD backend,
- validate correctness,
- compare performance against the current compiler.

---

### Student C — Tenstorrent architecture and backend

Responsibilities:

- study Tensix execution and memory model,
- implement mappings for compute, buffers, and communication,
- lower simple Spatial IR programs to Tenstorrent.

---

### Student D — Spatial optimization

Responsibilities:

- placement,
- communication-aware mapping,
- buffer sizing,
- multicast,
- scheduling and pipelining.

---

### Student E — Analysis and correctness

Responsibilities:

- channel dependency graph,
- resource validation,
- buffer bounds,
- deadlock analysis,
- synchronization correctness.

---

## 16. Suggested Initial Milestones

### Milestone 1

Represent a tiled MatMul in Spatial IR and visualize:

- tile decomposition,
- core placement,
- communication,
- local buffers.

### Milestone 2

Lower the MatMul Spatial IR to the existing AMD backend.

### Milestone 3

Lower the same Spatial IR to a simplified Tenstorrent backend.

### Milestone 4

Implement producer-consumer streaming:

```text
producer -> FIFO -> consumer
```

on both architectures.

### Milestone 5

Implement multicast and double buffering.

### Milestone 6

Compile a small Halide pipeline through:

```text
Halide
  ->
Spatial IR
  ->
AMD / Tenstorrent
```

### Milestone 7

Implement one automatic optimization:

```text
communication-aware placement
```

or

```text
fusion/locality optimization.
```

---

## 17. Longer-Term Research Agenda

The long-term vision is broader than Halide or ML.

```text
                 Domain DSLs

       Halide      ML       Genomics
          \         |          /
           \        |         /
            +---------------+
            |  Spatial IR   |
            +---------------+
             /      |       \
            /       |        \
       AMD NPU   Tenstorrent  Future
                             Spatial
                           Accelerator
```

Different domain DSLs should describe computations in the natural vocabulary of their domains.

Spatial IR should describe how those computations **inhabit space**.

This gives a useful separation:

> **Domain DSLs describe what computation means in a domain. Spatial DSL describes how computation is organized across space.**

The ultimate goal is to determine whether spatial execution deserves a common compiler abstraction in much the same way that loops, vectors, threads, and memory hierarchies became first-class abstractions in conventional compiler systems.

---

## 18. Possible Paper Positioning

A future paper could be positioned around the following thesis:

> Existing domain DSLs abstract hardware away, while low-level spatial accelerator APIs expose architecture-specific mechanisms. We introduce a middle-level spatial abstraction that makes decomposition, placement, communication, memory, and concurrency explicit while remaining independent of the concrete mechanisms used by individual spatial accelerators.

Potential title directions include:

- **Spatial: An Architecture-Visible Programming Abstraction for Spatial Accelerators**
- **Spatial IR: A Portable Intermediate Representation for Programmable Spatial Architectures**
- **From Domain DSLs to Spatial Machines: An Intermediate Abstraction for Mapping Portability**
- **Programming Spatial Accelerators Through Architectural Intent**
- **Mapping Portability for Programmable Spatial Accelerators**
- **Spatial IR: Separating Spatial Intent from Accelerator Mechanism**

---

## 19. Immediate Next Step

The recommended first research activity is:

> **Take three representative mappings from the existing PACT 2026 Halide-to-AMD-NPU compiler and rewrite them manually using a hypothetical Spatial IR.**

For each mapping, classify every compiler decision into one of three categories:

```text
Domain-specific
Spatial / architecture-independent
AMD-specific
```

Repeat the same exercise while asking how the spatial decisions would map to Tenstorrent.

This will provide an evidence-driven basis for designing the first version of Spatial IR and will avoid prematurely inventing abstractions that are either unnecessary or overly tied to one architecture.
