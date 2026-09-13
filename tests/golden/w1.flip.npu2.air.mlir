#map = affine_map<()[s0] -> (s0)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
module {
  air.channel @A2L1 [4]
  air.channel @B2L1 [4]
  air.channel @C2L3 [1]
  air.channel @CascadeK [3] {channel_type = "npu_cascade"}
  func.func @gemm(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1_0) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> {
      air.segment @gemm_seg  args(%arg10=%arg7, %arg11=%arg8, %arg12=%arg9) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> {
        %c0 = arith.constant 0 : index
        air.channel.put  @B2L1[%c0] (%arg11[0, 0] [16, 64] [64, 1]) : (memref<64x64xf32>)
        %c1_1 = arith.constant 1 : index
        air.channel.put  @B2L1[%c1_1] (%arg11[16, 0] [16, 64] [64, 1]) : (memref<64x64xf32>)
        %c2 = arith.constant 2 : index
        air.channel.put  @B2L1[%c2] (%arg11[32, 0] [16, 64] [64, 1]) : (memref<64x64xf32>)
        %c3 = arith.constant 3 : index
        air.channel.put  @B2L1[%c3] (%arg11[48, 0] [16, 64] [64, 1]) : (memref<64x64xf32>)
        %c0_2 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        scf.for %arg13 = %c0_2 to %c64 step %c32 {
          %0 = affine.apply #map()[%arg13]
          %c0_16 = arith.constant 0 : index
          air.channel.put  @A2L1[%c0_16] (%arg10[%0, 0] [32, 16] [64, 1]) : (memref<64x64xf32>)
        }
        %c0_3 = arith.constant 0 : index
        %c64_4 = arith.constant 64 : index
        %c32_5 = arith.constant 32 : index
        scf.for %arg13 = %c0_3 to %c64_4 step %c32_5 {
          %0 = affine.apply #map()[%arg13]
          %c1_16 = arith.constant 1 : index
          air.channel.put  @A2L1[%c1_16] (%arg10[%0, 16] [32, 16] [64, 1]) : (memref<64x64xf32>)
        }
        %c0_6 = arith.constant 0 : index
        %c64_7 = arith.constant 64 : index
        %c32_8 = arith.constant 32 : index
        scf.for %arg13 = %c0_6 to %c64_7 step %c32_8 {
          %0 = affine.apply #map()[%arg13]
          %c2_16 = arith.constant 2 : index
          air.channel.put  @A2L1[%c2_16] (%arg10[%0, 32] [32, 16] [64, 1]) : (memref<64x64xf32>)
        }
        %c0_9 = arith.constant 0 : index
        %c64_10 = arith.constant 64 : index
        %c32_11 = arith.constant 32 : index
        scf.for %arg13 = %c0_9 to %c64_10 step %c32_11 {
          %0 = affine.apply #map()[%arg13]
          %c3_16 = arith.constant 3 : index
          air.channel.put  @A2L1[%c3_16] (%arg10[%0, 48] [32, 16] [64, 1]) : (memref<64x64xf32>)
        }
        %c4 = arith.constant 4 : index
        %c1_12 = arith.constant 1 : index
        air.herd @gemm_herd  tile (%arg13, %arg14) in (%arg15=%c4, %arg16=%c1_12) {
          %alloc = memref.alloc() : memref<16x64xf32, 2 : i32>
          %0 = affine.apply #map()[%arg13]
          air.channel.get  @B2L1[%0] (%alloc[] [] []) : (memref<16x64xf32, 2 : i32>)
          %alloc_16 = memref.alloc() : memref<32x64xf32, 2 : i32>
          %alloc_17 = memref.alloc() : memref<32x64xf32, 2 : i32>
          %c0_18 = arith.constant 0 : index
          %c64_19 = arith.constant 64 : index
          %c32_20 = arith.constant 32 : index
          scf.for %arg17 = %c0_18 to %c64_19 step %c32_20 {
            %alloc_21 = memref.alloc() : memref<32x16xf32, 2 : i32>
            %1 = affine.apply #map()[%arg13]
            air.channel.get  @A2L1[%1] (%alloc_21[] [] []) : (memref<32x16xf32, 2 : i32>)
            %c0_22 = arith.constant 0 : index
            %c32_23 = arith.constant 32 : index
            %c1_24 = arith.constant 1 : index
            scf.for %arg18 = %c0_22 to %c32_23 step %c1_24 {
              %c0_29 = arith.constant 0 : index
              %c64_30 = arith.constant 64 : index
              %c1_31 = arith.constant 1 : index
              scf.for %arg19 = %c0_29 to %c64_30 step %c1_31 {
                %cst = arith.constant 0.000000e+00 : f32
                memref.store %cst, %alloc_16[%arg18, %arg19] : memref<32x64xf32, 2 : i32>
              }
            }
            %c0_25 = arith.constant 0 : index
            %c32_26 = arith.constant 32 : index
            %c1_27 = arith.constant 1 : index
            scf.for %arg18 = %c0_25 to %c32_26 step %c1_27 {
              %c0_29 = arith.constant 0 : index
              %c64_30 = arith.constant 64 : index
              %c1_31 = arith.constant 1 : index
              scf.for %arg19 = %c0_29 to %c64_30 step %c1_31 {
                %c0_32 = arith.constant 0 : index
                %c16 = arith.constant 16 : index
                %c1_33 = arith.constant 1 : index
                scf.for %arg20 = %c0_32 to %c16 step %c1_33 {
                  %4 = memref.load %alloc_16[%arg18, %arg19] : memref<32x64xf32, 2 : i32>
                  %5 = memref.load %alloc_21[%arg18, %arg20] : memref<32x16xf32, 2 : i32>
                  %6 = memref.load %alloc[%arg20, %arg19] : memref<16x64xf32, 2 : i32>
                  %7 = arith.mulf %5, %6 : f32
                  %8 = arith.addf %4, %7 : f32
                  memref.store %8, %alloc_16[%arg18, %arg19] : memref<32x64xf32, 2 : i32>
                }
              }
            }
            memref.dealloc %alloc_21 : memref<32x16xf32, 2 : i32>
            %2 = affine.apply #map()[%arg13]
            %c0_28 = arith.constant 0 : index
            %3 = arith.cmpi eq, %2, %c0_28 : index
            scf.if %3 {
              %4 = affine.apply #map()[%arg13]
              air.channel.put  @CascadeK[%4] (%alloc_16[] [] []) : (memref<32x64xf32, 2 : i32>)
            } else {
              %4 = affine.apply #map1()[%arg13]
              air.channel.get  @CascadeK[%4] (%alloc_17[] [] []) : (memref<32x64xf32, 2 : i32>)
              %c0_29 = arith.constant 0 : index
              %c32_30 = arith.constant 32 : index
              %c1_31 = arith.constant 1 : index
              scf.for %arg18 = %c0_29 to %c32_30 step %c1_31 {
                %c0_33 = arith.constant 0 : index
                %c64_34 = arith.constant 64 : index
                %c1_35 = arith.constant 1 : index
                scf.for %arg19 = %c0_33 to %c64_34 step %c1_35 {
                  %7 = memref.load %alloc_16[%arg18, %arg19] : memref<32x64xf32, 2 : i32>
                  %8 = memref.load %alloc_17[%arg18, %arg19] : memref<32x64xf32, 2 : i32>
                  %9 = arith.addf %7, %8 : f32
                  memref.store %9, %alloc_16[%arg18, %arg19] : memref<32x64xf32, 2 : i32>
                }
              }
              %5 = affine.apply #map()[%arg13]
              %c3_32 = arith.constant 3 : index
              %6 = arith.cmpi eq, %5, %c3_32 : index
              scf.if %6 {
                %c0_33 = arith.constant 0 : index
                air.channel.put  @C2L3[%c0_33] (%alloc_16[] [] []) : (memref<32x64xf32, 2 : i32>)
              } else {
                %7 = affine.apply #map()[%arg13]
                air.channel.put  @CascadeK[%7] (%alloc_16[] [] []) : (memref<32x64xf32, 2 : i32>)
              }
            }
          }
          memref.dealloc %alloc_17 : memref<32x64xf32, 2 : i32>
          memref.dealloc %alloc_16 : memref<32x64xf32, 2 : i32>
          memref.dealloc %alloc : memref<16x64xf32, 2 : i32>
        }
        %c0_13 = arith.constant 0 : index
        %c64_14 = arith.constant 64 : index
        %c32_15 = arith.constant 32 : index
        scf.for %arg13 = %c0_13 to %c64_14 step %c32_15 {
          %0 = affine.apply #map()[%arg13]
          %c0_16 = arith.constant 0 : index
          air.channel.get  @C2L3[%c0_16] (%arg12[%0, 0] [32, 64] [64, 1]) : (memref<64x64xf32>)
        }
      }
    }
    return
  }
}
