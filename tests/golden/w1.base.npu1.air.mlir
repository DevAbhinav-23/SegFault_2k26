#map = affine_map<()[s0] -> (s0)>
#map1 = affine_map<()[s0, s1] -> (s0 * 2 + s1)>
module {
  air.channel @A2L1 [2, 1] {broadcast_shape = [2 : index, 2 : index]}
  air.channel @B2L1 [1, 2] {broadcast_shape = [2 : index, 2 : index]}
  air.channel @C2L3 [2, 2]
  func.func @gemm(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1_0) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> {
      air.segment @gemm_seg  args(%arg10=%arg7, %arg11=%arg8, %arg12=%arg9) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c16 = arith.constant 16 : index
        scf.for %arg13 = %c0 to %c64 step %c16 {
          %0 = affine.apply #map()[%arg13]
          %c0_19 = arith.constant 0 : index
          %c0_20 = arith.constant 0 : index
          air.channel.put  @A2L1[%c0_19, %c0_20] (%arg10[0, %0] [32, 16] [64, 1]) : (memref<64x64xf32>)
        }
        %c0_1 = arith.constant 0 : index
        %c64_2 = arith.constant 64 : index
        %c16_3 = arith.constant 16 : index
        scf.for %arg13 = %c0_1 to %c64_2 step %c16_3 {
          %0 = affine.apply #map()[%arg13]
          %c1_19 = arith.constant 1 : index
          %c0_20 = arith.constant 0 : index
          air.channel.put  @A2L1[%c1_19, %c0_20] (%arg10[32, %0] [32, 16] [64, 1]) : (memref<64x64xf32>)
        }
        %c0_4 = arith.constant 0 : index
        %c64_5 = arith.constant 64 : index
        %c16_6 = arith.constant 16 : index
        scf.for %arg13 = %c0_4 to %c64_5 step %c16_6 {
          %0 = affine.apply #map()[%arg13]
          %c0_19 = arith.constant 0 : index
          %c0_20 = arith.constant 0 : index
          air.channel.put  @B2L1[%c0_19, %c0_20] (%arg11[%0, 0] [16, 32] [64, 1]) : (memref<64x64xf32>)
        }
        %c0_7 = arith.constant 0 : index
        %c64_8 = arith.constant 64 : index
        %c16_9 = arith.constant 16 : index
        scf.for %arg13 = %c0_7 to %c64_8 step %c16_9 {
          %0 = affine.apply #map()[%arg13]
          %c0_19 = arith.constant 0 : index
          %c1_20 = arith.constant 1 : index
          air.channel.put  @B2L1[%c0_19, %c1_20] (%arg11[%0, 32] [16, 32] [64, 1]) : (memref<64x64xf32>)
        }
        %c1_10 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        air.herd @gemm_herd  tile (%arg13, %arg14) in (%arg15=%c1_10, %arg16=%c2) {
          %c0_19 = arith.constant 0 : index
          %c2_20 = arith.constant 2 : index
          %c1_21 = arith.constant 1 : index
          scf.for %arg17 = %c0_19 to %c2_20 step %c1_21 {
            %alloc = memref.alloc() : memref<32x32xf32, 2 : i32>
            %c0_22 = arith.constant 0 : index
            %c32 = arith.constant 32 : index
            %c1_23 = arith.constant 1 : index
            scf.for %arg18 = %c0_22 to %c32 step %c1_23 {
              %c0_27 = arith.constant 0 : index
              %c32_28 = arith.constant 32 : index
              %c1_29 = arith.constant 1 : index
              scf.for %arg19 = %c0_27 to %c32_28 step %c1_29 {
                %cst = arith.constant 0.000000e+00 : f32
                memref.store %cst, %alloc[%arg18, %arg19] : memref<32x32xf32, 2 : i32>
              }
            }
            %c0_24 = arith.constant 0 : index
            %c64_25 = arith.constant 64 : index
            %c16_26 = arith.constant 16 : index
            scf.for %arg18 = %c0_24 to %c64_25 step %c16_26 {
              %alloc_27 = memref.alloc() : memref<32x16xf32, 2 : i32>
              %alloc_28 = memref.alloc() : memref<16x32xf32, 2 : i32>
              %2 = affine.apply #map1()[%arg13, %arg17]
              %3 = affine.apply #map()[%arg14]
              air.channel.get  @A2L1[%2, %3] (%alloc_27[] [] []) : (memref<32x16xf32, 2 : i32>)
              %4 = affine.apply #map1()[%arg13, %arg17]
              %5 = affine.apply #map()[%arg14]
              air.channel.get  @B2L1[%4, %5] (%alloc_28[] [] []) : (memref<16x32xf32, 2 : i32>)
              %c0_29 = arith.constant 0 : index
              %c32_30 = arith.constant 32 : index
              %c1_31 = arith.constant 1 : index
              scf.for %arg19 = %c0_29 to %c32_30 step %c1_31 {
                %c0_32 = arith.constant 0 : index
                %c32_33 = arith.constant 32 : index
                %c1_34 = arith.constant 1 : index
                scf.for %arg20 = %c0_32 to %c32_33 step %c1_34 {
                  %c0_35 = arith.constant 0 : index
                  %c16_36 = arith.constant 16 : index
                  %c1_37 = arith.constant 1 : index
                  scf.for %arg21 = %c0_35 to %c16_36 step %c1_37 {
                    %6 = memref.load %alloc[%arg19, %arg20] : memref<32x32xf32, 2 : i32>
                    %7 = memref.load %alloc_27[%arg19, %arg21] : memref<32x16xf32, 2 : i32>
                    %8 = memref.load %alloc_28[%arg21, %arg20] : memref<16x32xf32, 2 : i32>
                    %9 = arith.mulf %7, %8 : f32
                    %10 = arith.addf %6, %9 : f32
                    memref.store %10, %alloc[%arg19, %arg20] : memref<32x32xf32, 2 : i32>
                  }
                }
              }
              memref.dealloc %alloc_28 : memref<16x32xf32, 2 : i32>
              memref.dealloc %alloc_27 : memref<32x16xf32, 2 : i32>
            }
            %0 = affine.apply #map1()[%arg13, %arg17]
            %1 = affine.apply #map()[%arg14]
            air.channel.put  @C2L3[%0, %1] (%alloc[] [] []) : (memref<32x32xf32, 2 : i32>)
            memref.dealloc %alloc : memref<32x32xf32, 2 : i32>
          }
        }
        %c0_11 = arith.constant 0 : index
        %c0_12 = arith.constant 0 : index
        air.channel.get  @C2L3[%c0_11, %c0_12] (%arg12[0, 0] [32, 32] [64, 1]) : (memref<64x64xf32>)
        %c0_13 = arith.constant 0 : index
        %c1_14 = arith.constant 1 : index
        air.channel.get  @C2L3[%c0_13, %c1_14] (%arg12[0, 32] [32, 32] [64, 1]) : (memref<64x64xf32>)
        %c1_15 = arith.constant 1 : index
        %c0_16 = arith.constant 0 : index
        air.channel.get  @C2L3[%c1_15, %c0_16] (%arg12[32, 0] [32, 32] [64, 1]) : (memref<64x64xf32>)
        %c1_17 = arith.constant 1 : index
        %c1_18 = arith.constant 1 : index
        air.channel.get  @C2L3[%c1_17, %c1_18] (%arg12[32, 32] [32, 32] [64, 1]) : (memref<64x64xf32>)
      }
    }
    return
  }
}
