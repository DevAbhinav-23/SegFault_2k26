#map = affine_map<()[s0] -> (s0)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<()[s0] -> (s0 + 1)>
#map3 = affine_map<()[s0] -> (s0 + 2)>
module {
  air.channel @ToNorth [1]
  air.channel @ToSouth [1]
  air.channel @UIn [2]
  air.channel @UOut [2]
  func.func @jacobi(%arg0: memref<5x18x16xf32>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1_0) args(%arg5=%arg0) : memref<5x18x16xf32> {
      air.segment @jacobi_seg  args(%arg6=%arg5) : memref<5x18x16xf32> {
        %c0 = arith.constant 0 : index
        air.channel.put  @UIn[%c0] (%arg6[0, 0, 0] [1, 10, 16] [288, 16, 1]) : (memref<5x18x16xf32>)
        %c1_1 = arith.constant 1 : index
        air.channel.put  @UIn[%c1_1] (%arg6[0, 8, 0] [1, 10, 16] [288, 16, 1]) : (memref<5x18x16xf32>)
        %c2 = arith.constant 2 : index
        %c1_2 = arith.constant 1 : index
        air.herd @jacobi_herd  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c1_2) {
          %alloc = memref.alloc() : memref<10x16xf32, 2 : i32>
          %alloc_8 = memref.alloc() : memref<10x16xf32, 2 : i32>
          %0 = affine.apply #map()[%arg7]
          air.channel.get  @UIn[%0] (%alloc[] [] []) : (memref<10x16xf32, 2 : i32>)
          %c0_9 = arith.constant 0 : index
          %c10 = arith.constant 10 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg11 = %c0_9 to %c10 step %c1_10 {
            %c0_14 = arith.constant 0 : index
            %c16 = arith.constant 16 : index
            %c1_15 = arith.constant 1 : index
            scf.for %arg12 = %c0_14 to %c16 step %c1_15 {
              %1 = memref.load %alloc[%arg11, %arg12] : memref<10x16xf32, 2 : i32>
              memref.store %1, %alloc_8[%arg11, %arg12] : memref<10x16xf32, 2 : i32>
            }
          }
          %c0_11 = arith.constant 0 : index
          %c4_12 = arith.constant 4 : index
          %c2_13 = arith.constant 2 : index
          scf.for %arg11 = %c0_11 to %c4_12 step %c2_13 {
            %1 = affine.apply #map()[%arg7]
            %c0_14 = arith.constant 0 : index
            %2 = arith.cmpi sgt, %1, %c0_14 : index
            scf.if %2 {
              %19 = affine.apply #map1()[%arg7]
              air.channel.put  @ToNorth[%19] (%alloc[1, 0] [1, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
            } else {
            }
            %3 = affine.apply #map()[%arg7]
            %c1_15 = arith.constant 1 : index
            %4 = arith.cmpi slt, %3, %c1_15 : index
            scf.if %4 {
              %19 = affine.apply #map()[%arg7]
              air.channel.put  @ToSouth[%19] (%alloc[8, 0] [1, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
            } else {
            }
            %5 = affine.apply #map()[%arg7]
            %c1_16 = arith.constant 1 : index
            %6 = arith.cmpi slt, %5, %c1_16 : index
            scf.if %6 {
              %19 = affine.apply #map()[%arg7]
              air.channel.get  @ToNorth[%19] (%alloc[9, 0] [1, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
            } else {
            }
            %7 = affine.apply #map()[%arg7]
            %c0_17 = arith.constant 0 : index
            %8 = arith.cmpi sgt, %7, %c0_17 : index
            scf.if %8 {
              %19 = affine.apply #map1()[%arg7]
              air.channel.get  @ToSouth[%19] (%alloc[0, 0] [1, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
            } else {
            }
            %c0_18 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1_19 = arith.constant 1 : index
            scf.for %arg12 = %c0_18 to %c8 step %c1_19 {
              %c1_27 = arith.constant 1 : index
              %c15 = arith.constant 15 : index
              %c1_28 = arith.constant 1 : index
              scf.for %arg13 = %c1_27 to %c15 step %c1_28 {
                %cst = arith.constant 2.000000e-01 : f32
                %19 = affine.apply #map2()[%arg12]
                %20 = memref.load %alloc[%19, %arg13] : memref<10x16xf32, 2 : i32>
                %21 = memref.load %alloc[%arg12, %arg13] : memref<10x16xf32, 2 : i32>
                %22 = arith.addf %20, %21 : f32
                %23 = affine.apply #map3()[%arg12]
                %24 = memref.load %alloc[%23, %arg13] : memref<10x16xf32, 2 : i32>
                %25 = arith.addf %22, %24 : f32
                %26 = affine.apply #map2()[%arg12]
                %27 = affine.apply #map1()[%arg13]
                %28 = memref.load %alloc[%26, %27] : memref<10x16xf32, 2 : i32>
                %29 = arith.addf %25, %28 : f32
                %30 = affine.apply #map2()[%arg12]
                %31 = affine.apply #map2()[%arg13]
                %32 = memref.load %alloc[%30, %31] : memref<10x16xf32, 2 : i32>
                %33 = arith.addf %29, %32 : f32
                %34 = arith.mulf %cst, %33 : f32
                %35 = affine.apply #map2()[%arg12]
                memref.store %34, %alloc_8[%35, %arg13] : memref<10x16xf32, 2 : i32>
              }
            }
            %9 = affine.apply #map()[%arg7]
            air.channel.put  @UOut[%9] (%alloc_8[1, 0] [8, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
            %10 = affine.apply #map()[%arg7]
            %c0_20 = arith.constant 0 : index
            %11 = arith.cmpi sgt, %10, %c0_20 : index
            scf.if %11 {
              %19 = affine.apply #map1()[%arg7]
              air.channel.put  @ToNorth[%19] (%alloc_8[1, 0] [1, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
            } else {
            }
            %12 = affine.apply #map()[%arg7]
            %c1_21 = arith.constant 1 : index
            %13 = arith.cmpi slt, %12, %c1_21 : index
            scf.if %13 {
              %19 = affine.apply #map()[%arg7]
              air.channel.put  @ToSouth[%19] (%alloc_8[8, 0] [1, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
            } else {
            }
            %14 = affine.apply #map()[%arg7]
            %c1_22 = arith.constant 1 : index
            %15 = arith.cmpi slt, %14, %c1_22 : index
            scf.if %15 {
              %19 = affine.apply #map()[%arg7]
              air.channel.get  @ToNorth[%19] (%alloc_8[9, 0] [1, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
            } else {
            }
            %16 = affine.apply #map()[%arg7]
            %c0_23 = arith.constant 0 : index
            %17 = arith.cmpi sgt, %16, %c0_23 : index
            scf.if %17 {
              %19 = affine.apply #map1()[%arg7]
              air.channel.get  @ToSouth[%19] (%alloc_8[0, 0] [1, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
            } else {
            }
            %c0_24 = arith.constant 0 : index
            %c8_25 = arith.constant 8 : index
            %c1_26 = arith.constant 1 : index
            scf.for %arg12 = %c0_24 to %c8_25 step %c1_26 {
              %c1_27 = arith.constant 1 : index
              %c15 = arith.constant 15 : index
              %c1_28 = arith.constant 1 : index
              scf.for %arg13 = %c1_27 to %c15 step %c1_28 {
                %cst = arith.constant 2.000000e-01 : f32
                %19 = affine.apply #map2()[%arg12]
                %20 = memref.load %alloc_8[%19, %arg13] : memref<10x16xf32, 2 : i32>
                %21 = memref.load %alloc_8[%arg12, %arg13] : memref<10x16xf32, 2 : i32>
                %22 = arith.addf %20, %21 : f32
                %23 = affine.apply #map3()[%arg12]
                %24 = memref.load %alloc_8[%23, %arg13] : memref<10x16xf32, 2 : i32>
                %25 = arith.addf %22, %24 : f32
                %26 = affine.apply #map2()[%arg12]
                %27 = affine.apply #map1()[%arg13]
                %28 = memref.load %alloc_8[%26, %27] : memref<10x16xf32, 2 : i32>
                %29 = arith.addf %25, %28 : f32
                %30 = affine.apply #map2()[%arg12]
                %31 = affine.apply #map2()[%arg13]
                %32 = memref.load %alloc_8[%30, %31] : memref<10x16xf32, 2 : i32>
                %33 = arith.addf %29, %32 : f32
                %34 = arith.mulf %cst, %33 : f32
                %35 = affine.apply #map2()[%arg12]
                memref.store %34, %alloc[%35, %arg13] : memref<10x16xf32, 2 : i32>
              }
            }
            %18 = affine.apply #map()[%arg7]
            air.channel.put  @UOut[%18] (%alloc[1, 0] [8, 16] [16, 1]) : (memref<10x16xf32, 2 : i32>)
          }
          memref.dealloc %alloc_8 : memref<10x16xf32, 2 : i32>
          memref.dealloc %alloc : memref<10x16xf32, 2 : i32>
        }
        %c0_3 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg7 = %c0_3 to %c4 step %c1_4 {
          %0 = affine.apply #map2()[%arg7]
          %c0_8 = arith.constant 0 : index
          air.channel.get  @UOut[%c0_8] (%arg6[%0, 1, 0] [1, 8, 16] [288, 16, 1]) : (memref<5x18x16xf32>)
        }
        %c0_5 = arith.constant 0 : index
        %c4_6 = arith.constant 4 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg7 = %c0_5 to %c4_6 step %c1_7 {
          %0 = affine.apply #map2()[%arg7]
          %c1_8 = arith.constant 1 : index
          air.channel.get  @UOut[%c1_8] (%arg6[%0, 9, 0] [1, 8, 16] [288, 16, 1]) : (memref<5x18x16xf32>)
        }
      }
    }
    return
  }
}
