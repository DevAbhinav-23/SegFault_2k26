// Provenance: copied from vendor/probes/e3_i.mlir (design-phase probe E3(i), 2026-09-12):
// one air.channel.put with no matching get. air-opt's air-dependency-canonicalize prints
// "error: 'air.channel.put' op found channel op not in pairs" and exits 0 -- the
// characterisation test of FR-T5 (04-test-plan.md S3.2, 03-lld-M6-toolchain.md S6.1).
// vendor/probes/e3_i_unmatched_put.mlir is NOT this module: it has an undeclared SSA
// name and exits 1 with a parse error (see design/PROGRESS-B.md, phase P0c).
module {
  air.channel @C [1, 1]
  func.func @k(%arg0: memref<32xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    air.segment @seg {
      %c0b = arith.constant 0 : index
      %alloc = memref.alloc() : memref<32xi32, 1>
      air.channel.put @C[%c0b, %c0b] (%alloc[] [] []) : (memref<32xi32, 1>)
      air.segment_terminator
    }
    return
  }
}
