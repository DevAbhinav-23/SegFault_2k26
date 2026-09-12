"""M0 — the shared contract. Owner: all three (frozen). Spec: design/06-interfaces.md §1-§6.

Every dataclass named in §2-§5 (`Param`, `Axis`, `AccessMap`, `Statement`, `Dependence`,
`ReductionSpec`, `KernelModel`, `AxisRef`, `ScheduleModel`, `LegalMapping`, `BufferPlan`,
`ChannelSite`, `ChannelPlan`, `HerdPlan`, `LoopPlan`, `MappingPlan`, `MappingSummary`,
`EmitResult`) and the `Diagnostic` / error classes of §6 land here in the next phase.
They are frozen, tuple-valued and structurally hashable.
"""

CONTRACT_VERSION = 3
