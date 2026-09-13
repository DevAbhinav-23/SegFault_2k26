#map = affine_map<()[s0] -> (s0)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<()[s0] -> (s0 + 1)>
module {
  air.channel @EastOut [1]
  air.channel @QIn [1] {broadcast_shape = [4 : index]}
  air.channel @RIn [4]
  air.channel @SOut [4]
  air.channel @West [3]
  air.channel @WestIn [1]
  func.func @sw(%arg0: memref<32xi32>, %arg1: memref<32xi32>, %arg2: memref<33x33xi32>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1_0) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<32xi32>, memref<32xi32>, memref<33x33xi32> {
      air.segment @sw_seg  args(%arg10=%arg7, %arg11=%arg8, %arg12=%arg9) : memref<32xi32>, memref<32xi32>, memref<33x33xi32> {
        %c0 = arith.constant 0 : index
        air.channel.put  @QIn[%c0] (%arg10[0] [32] [1]) : (memref<32xi32>)
        %c0_1 = arith.constant 0 : index
        air.channel.put  @RIn[%c0_1] (%arg11[0] [8] [1]) : (memref<32xi32>)
        %c1_2 = arith.constant 1 : index
        air.channel.put  @RIn[%c1_2] (%arg11[8] [8] [1]) : (memref<32xi32>)
        %c2 = arith.constant 2 : index
        air.channel.put  @RIn[%c2] (%arg11[16] [8] [1]) : (memref<32xi32>)
        %c3 = arith.constant 3 : index
        air.channel.put  @RIn[%c3] (%arg11[24] [8] [1]) : (memref<32xi32>)
        %c1_3 = arith.constant 1 : index
        %c33 = arith.constant 33 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg13 = %c1_3 to %c33 step %c1_4 {
          %0 = affine.apply #map()[%arg13]
          %c0_21 = arith.constant 0 : index
          air.channel.put  @WestIn[%c0_21] (%arg12[%0, 0] [1, 1] [33, 1]) : (memref<33x33xi32>)
        }
        %c4 = arith.constant 4 : index
        %c1_5 = arith.constant 1 : index
        air.herd @sw_herd  tile (%arg13, %arg14) in (%arg15=%c4, %arg16=%c1_5) {
          %alloc = memref.alloc() : memref<32xi32, 2 : i32>
          %alloc_21 = memref.alloc() : memref<8xi32, 2 : i32>
          %alloc_22 = memref.alloc() : memref<9xi32, 2 : i32>
          %alloc_23 = memref.alloc() : memref<9xi32, 2 : i32>
          %alloc_24 = memref.alloc() : memref<1xi32, 2 : i32>
          %alloc_25 = memref.alloc() : memref<1xi32, 2 : i32>
          %0 = affine.apply #map()[%arg13]
          air.channel.get  @QIn[%0] (%alloc[] [] []) : (memref<32xi32, 2 : i32>)
          %1 = affine.apply #map()[%arg13]
          air.channel.get  @RIn[%1] (%alloc_21[] [] []) : (memref<8xi32, 2 : i32>)
          %c0_26 = arith.constant 0 : index
          %c9 = arith.constant 9 : index
          %c1_27 = arith.constant 1 : index
          scf.for %arg17 = %c0_26 to %c9 step %c1_27 {
            %c0_i32 = arith.constant 0 : i32
            memref.store %c0_i32, %alloc_22[%arg17] : memref<9xi32, 2 : i32>
          }
          %c1_28 = arith.constant 1 : index
          %c33_29 = arith.constant 33 : index
          %c2_30 = arith.constant 2 : index
          scf.for %arg17 = %c1_28 to %c33_29 step %c2_30 {
            %2 = affine.apply #map()[%arg13]
            %c0_31 = arith.constant 0 : index
            %3 = arith.cmpi eq, %2, %c0_31 : index
            scf.if %3 {
              %c0_48 = arith.constant 0 : index
              air.channel.get  @WestIn[%c0_48] (%alloc_24[] [] []) : (memref<1xi32, 2 : i32>)
            } else {
              %16 = affine.apply #map1()[%arg13]
              air.channel.get  @West[%16] (%alloc_24[] [] []) : (memref<1xi32, 2 : i32>)
            }
            %c0_32 = arith.constant 0 : index
            %4 = memref.load %alloc_24[%c0_32] : memref<1xi32, 2 : i32>
            %c0_33 = arith.constant 0 : index
            memref.store %4, %alloc_23[%c0_33] : memref<9xi32, 2 : i32>
            %c0_34 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1_35 = arith.constant 1 : index
            scf.for %arg18 = %c0_34 to %c8 step %c1_35 {
              %c0_i32 = arith.constant 0 : i32
              %16 = memref.load %alloc_22[%arg18] : memref<9xi32, 2 : i32>
              %17 = affine.apply #map1()[%arg17]
              %18 = memref.load %alloc[%17] : memref<32xi32, 2 : i32>
              %19 = memref.load %alloc_21[%arg18] : memref<8xi32, 2 : i32>
              %20 = arith.cmpi eq, %18, %19 : i32
              %c2_i32 = arith.constant 2 : i32
              %c-1_i32 = arith.constant -1 : i32
              %21 = arith.select %20, %c2_i32, %c-1_i32 : i32
              %22 = arith.addi %16, %21 : i32
              %23 = affine.apply #map2()[%arg18]
              %24 = memref.load %alloc_22[%23] : memref<9xi32, 2 : i32>
              %c1_i32 = arith.constant 1 : i32
              %25 = arith.subi %24, %c1_i32 : i32
              %26 = memref.load %alloc_23[%arg18] : memref<9xi32, 2 : i32>
              %c1_i32_48 = arith.constant 1 : i32
              %27 = arith.subi %26, %c1_i32_48 : i32
              %28 = arith.maxsi %25, %27 : i32
              %29 = arith.maxsi %22, %28 : i32
              %30 = arith.maxsi %c0_i32, %29 : i32
              %31 = affine.apply #map2()[%arg18]
              memref.store %30, %alloc_23[%31] : memref<9xi32, 2 : i32>
            }
            %c8_36 = arith.constant 8 : index
            %5 = memref.load %alloc_23[%c8_36] : memref<9xi32, 2 : i32>
            %c0_37 = arith.constant 0 : index
            memref.store %5, %alloc_25[%c0_37] : memref<1xi32, 2 : i32>
            %6 = affine.apply #map()[%arg13]
            %c3_38 = arith.constant 3 : index
            %7 = arith.cmpi eq, %6, %c3_38 : index
            scf.if %7 {
              %c0_48 = arith.constant 0 : index
              air.channel.put  @EastOut[%c0_48] (%alloc_25[] [] []) : (memref<1xi32, 2 : i32>)
            } else {
              %16 = affine.apply #map()[%arg13]
              air.channel.put  @West[%16] (%alloc_25[] [] []) : (memref<1xi32, 2 : i32>)
            }
            %8 = affine.apply #map()[%arg13]
            air.channel.put  @SOut[%8] (%alloc_23[1] [8] [1]) : (memref<9xi32, 2 : i32>)
            %9 = affine.apply #map()[%arg13]
            %c0_39 = arith.constant 0 : index
            %10 = arith.cmpi eq, %9, %c0_39 : index
            scf.if %10 {
              %c0_48 = arith.constant 0 : index
              air.channel.get  @WestIn[%c0_48] (%alloc_24[] [] []) : (memref<1xi32, 2 : i32>)
            } else {
              %16 = affine.apply #map1()[%arg13]
              air.channel.get  @West[%16] (%alloc_24[] [] []) : (memref<1xi32, 2 : i32>)
            }
            %c0_40 = arith.constant 0 : index
            %11 = memref.load %alloc_24[%c0_40] : memref<1xi32, 2 : i32>
            %c0_41 = arith.constant 0 : index
            memref.store %11, %alloc_22[%c0_41] : memref<9xi32, 2 : i32>
            %c0_42 = arith.constant 0 : index
            %c8_43 = arith.constant 8 : index
            %c1_44 = arith.constant 1 : index
            scf.for %arg18 = %c0_42 to %c8_43 step %c1_44 {
              %c0_i32 = arith.constant 0 : i32
              %16 = memref.load %alloc_23[%arg18] : memref<9xi32, 2 : i32>
              %17 = memref.load %alloc[%arg17] : memref<32xi32, 2 : i32>
              %18 = memref.load %alloc_21[%arg18] : memref<8xi32, 2 : i32>
              %19 = arith.cmpi eq, %17, %18 : i32
              %c2_i32 = arith.constant 2 : i32
              %c-1_i32 = arith.constant -1 : i32
              %20 = arith.select %19, %c2_i32, %c-1_i32 : i32
              %21 = arith.addi %16, %20 : i32
              %22 = affine.apply #map2()[%arg18]
              %23 = memref.load %alloc_23[%22] : memref<9xi32, 2 : i32>
              %c1_i32 = arith.constant 1 : i32
              %24 = arith.subi %23, %c1_i32 : i32
              %25 = memref.load %alloc_22[%arg18] : memref<9xi32, 2 : i32>
              %c1_i32_48 = arith.constant 1 : i32
              %26 = arith.subi %25, %c1_i32_48 : i32
              %27 = arith.maxsi %24, %26 : i32
              %28 = arith.maxsi %21, %27 : i32
              %29 = arith.maxsi %c0_i32, %28 : i32
              %30 = affine.apply #map2()[%arg18]
              memref.store %29, %alloc_22[%30] : memref<9xi32, 2 : i32>
            }
            %c8_45 = arith.constant 8 : index
            %12 = memref.load %alloc_22[%c8_45] : memref<9xi32, 2 : i32>
            %c0_46 = arith.constant 0 : index
            memref.store %12, %alloc_25[%c0_46] : memref<1xi32, 2 : i32>
            %13 = affine.apply #map()[%arg13]
            %c3_47 = arith.constant 3 : index
            %14 = arith.cmpi eq, %13, %c3_47 : index
            scf.if %14 {
              %c0_48 = arith.constant 0 : index
              air.channel.put  @EastOut[%c0_48] (%alloc_25[] [] []) : (memref<1xi32, 2 : i32>)
            } else {
              %16 = affine.apply #map()[%arg13]
              air.channel.put  @West[%16] (%alloc_25[] [] []) : (memref<1xi32, 2 : i32>)
            }
            %15 = affine.apply #map()[%arg13]
            air.channel.put  @SOut[%15] (%alloc_22[1] [8] [1]) : (memref<9xi32, 2 : i32>)
          }
          memref.dealloc %alloc_25 : memref<1xi32, 2 : i32>
          memref.dealloc %alloc_24 : memref<1xi32, 2 : i32>
          memref.dealloc %alloc_23 : memref<9xi32, 2 : i32>
          memref.dealloc %alloc_22 : memref<9xi32, 2 : i32>
          memref.dealloc %alloc_21 : memref<8xi32, 2 : i32>
          memref.dealloc %alloc : memref<32xi32, 2 : i32>
        }
        %c1_6 = arith.constant 1 : index
        %c33_7 = arith.constant 33 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg13 = %c1_6 to %c33_7 step %c1_8 {
          %0 = affine.apply #map()[%arg13]
          %c0_21 = arith.constant 0 : index
          air.channel.get  @EastOut[%c0_21] (%arg12[%0, 32] [1, 1] [33, 1]) : (memref<33x33xi32>)
        }
        %c1_9 = arith.constant 1 : index
        %c33_10 = arith.constant 33 : index
        %c1_11 = arith.constant 1 : index
        scf.for %arg13 = %c1_9 to %c33_10 step %c1_11 {
          %0 = affine.apply #map()[%arg13]
          %c0_21 = arith.constant 0 : index
          air.channel.get  @SOut[%c0_21] (%arg12[%0, 1] [1, 8] [33, 1]) : (memref<33x33xi32>)
        }
        %c1_12 = arith.constant 1 : index
        %c33_13 = arith.constant 33 : index
        %c1_14 = arith.constant 1 : index
        scf.for %arg13 = %c1_12 to %c33_13 step %c1_14 {
          %0 = affine.apply #map()[%arg13]
          %c1_21 = arith.constant 1 : index
          air.channel.get  @SOut[%c1_21] (%arg12[%0, 9] [1, 8] [33, 1]) : (memref<33x33xi32>)
        }
        %c1_15 = arith.constant 1 : index
        %c33_16 = arith.constant 33 : index
        %c1_17 = arith.constant 1 : index
        scf.for %arg13 = %c1_15 to %c33_16 step %c1_17 {
          %0 = affine.apply #map()[%arg13]
          %c2_21 = arith.constant 2 : index
          air.channel.get  @SOut[%c2_21] (%arg12[%0, 17] [1, 8] [33, 1]) : (memref<33x33xi32>)
        }
        %c1_18 = arith.constant 1 : index
        %c33_19 = arith.constant 33 : index
        %c1_20 = arith.constant 1 : index
        scf.for %arg13 = %c1_18 to %c33_19 step %c1_20 {
          %0 = affine.apply #map()[%arg13]
          %c3_21 = arith.constant 3 : index
          air.channel.get  @SOut[%c3_21] (%arg12[%0, 25] [1, 8] [33, 1]) : (memref<33x33xi32>)
        }
      }
    }
    return
  }
}
