// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access="use-tile-matmul=true" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
module {
  func.func private @no_loops(%arg0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) attributes {d2m.thread = #d2m.thread<compute>} {
    %c0 = arith.constant 0 : index
    // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
    %0 = affine.load %arg0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK: %[[ARG2_VAL:.*]] = affine.load %arg2
    %2 = affine.load %arg2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    // Check that the third operand (accumulator) is stored to dst memory space
    // CHECK: affine.store %[[ARG2_VAL]], %[[DST]]
    // Check that the accumulator is loaded back from dst memory space for the matmul
    // CHECK: %[[DST_VAL:.*]] = affine.load %[[DST]]
    // CHECK: %[[MATMUL_RESULT:.*]] = "d2m.tile_matmul"
    %3 = "d2m.tile_matmul"(%0, %1, %2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // Check that matmul result is stored back to dst memory space
    // CHECK: affine.store %[[MATMUL_RESULT]], %[[DST]]
    // Check that result is loaded from dst memory space
    // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]]
    // Check that final result is stored back to original #l1 memory space
    // CHECK: affine.store %[[FINAL_VAL]], %arg2
    affine.store %3, %arg2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  func.func private @binary(%arg0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) attributes {d2m.thread = #d2m.thread<compute>} {
    %c0 = arith.constant 0 : index
    // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
    %0 = affine.load %arg0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    // Check that the operands are stored to dst memory space
    // CHECK: %[[ARG0_VAL:.*]] = affine.load %arg0
    // CHECK: affine.store %[[ARG0_VAL]], %[[DST]]
    // CHECK: %[[DST0_VAL:.*]] = affine.load %[[DST]]
    // CHECK: %[[ARG1_VAL:.*]] = affine.load %arg1
    // CHECK: affine.store %[[ARG1_VAL]], %[[DST]]
    // CHECK: %[[DST1_VAL:.*]] = affine.load %[[DST]]
    // CHECK: %[[MAXIMUM_RESULT:.*]] = "d2m.tile_maximum"
    %3 = "d2m.tile_maximum"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // Check that maximum result is stored back to dst memory space
    // CHECK: affine.store %[[MAXIMUM_RESULT]], %[[DST]]
    // Check that result is loaded from dst memory space
    // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]]
    // Check that final result is stored back to original #l1 memory space
    // CHECK: affine.store %[[FINAL_VAL]], %arg2
    affine.store %3, %arg2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  func.func private @generic_matmul(
    %in0: memref<1x1x3x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #ttcore.memory_space<l1>>,
    %in1: memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.memory_space<l1>>,
    %out0: memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.memory_space<l1>>
    ) {
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x3x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #ttcore.memory_space<l1>>, memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.memory_space<l1>>)
        outs(%out0 : memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.memory_space<l1>>)  {
    ^compute0(%cb0: memref<3x3x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %cb1: memref<3x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %cb2: memref<3x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>):
      // Check that constants and destination buffer are created
      // CHECK: %[[C0:.*]] = arith.constant 0 : index
      // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>

      // Check for iteration index and conditional initialization
      // CHECK: %[[ITER2:.*]] = d2m.iter_index(2) : index
      // CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[ITER2]], %[[C0]] : index
      // CHECK: scf.if %[[CMP]] {

      // Check conditional initialization loop structure (2D loop for initialization)
      // CHECK: affine.for %[[INIT_I:.*]] = 0 to 3 {
      // CHECK-NEXT: affine.for %[[INIT_J:.*]] = 0 to 2 {

      // Check initialization: load from l1, store to dst
      // CHECK: %[[INIT_VAL:.*]] = affine.load %cb2[%[[INIT_I]], %[[INIT_J]]] : memref<3x2x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: affine.store %[[INIT_VAL]], %[[DST]][0, %[[INIT_I]], %[[INIT_J]]] : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>

      // Check main computation loop structure (3D loop nest)
      // CHECK: affine.for %[[I:.*]] = 0 to 3 {
      // CHECK-NEXT: affine.for %[[J:.*]] = 0 to 2 {
      // CHECK-NEXT: affine.for %[[K:.*]] = 0 to 3 {

      // Check loads in computation loop: first two from l1, accumulator from dst
      // CHECK: %[[A_VAL:.*]] = affine.load %cb0[%[[I]], %[[K]]] : memref<3x3x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: %[[B_VAL:.*]] = affine.load %cb1[%[[K]], %[[J]]] : memref<3x2x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: %[[C_VAL:.*]] = affine.load %[[DST]][0, %[[I]], %[[J]]] : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>

      // Check matmul operation uses values from correct memory spaces
      // CHECK: %[[MATMUL_RESULT:.*]] = "d2m.tile_matmul"(%[[A_VAL]], %[[B_VAL]], %[[C_VAL]]) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

      // Check result is stored back to dst
      // CHECK: affine.store %[[MATMUL_RESULT]], %[[DST]][0, %[[I]], %[[J]]] : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>
      %c0 = arith.constant 0 : index
      %c3_10 = arith.constant 3 : index
      %c3_11 = arith.constant 3 : index
      %c0_12 = arith.constant 0 : index
      %c2_13 = arith.constant 2 : index
      %c2_14 = arith.constant 2 : index
      %c0_15 = arith.constant 0 : index
      %c3_16 = arith.constant 3 : index
      %c3_17 = arith.constant 3 : index
      scf.for %arg2 = %c0 to %c3_10 step %c3_11 {
        scf.for %arg3 = %c0_12 to %c2_13 step %c2_14 {
          scf.for %arg4 = %c0_15 to %c3_16 step %c3_17 {
            %subview = memref.subview %cb0[%arg2, %arg4] [3, 3] [1, 1] : memref<3x3x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<3x3x!ttcore.tile<32x32, f32>, strided<[3, 1], offset: ?>, #ttcore.memory_space<l1>>
            %subview_18 = memref.subview %cb1[%arg4, %arg3] [3, 2] [1, 1] : memref<3x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #ttcore.memory_space<l1>>
            %subview_19 = memref.subview %cb2[%arg2, %arg3] [3, 2] [1, 1] : memref<3x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #ttcore.memory_space<l1>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview, %subview_18 : memref<3x3x!ttcore.tile<32x32, f32>, strided<[3, 1], offset: ?>, #ttcore.memory_space<l1>>, memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #ttcore.memory_space<l1>>) outs(%subview_19 : memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #ttcore.memory_space<l1>>) {
            ^bb0(%in: !ttcore.tile<32x32, f32>, %in_20: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
              %0 = "d2m.tile_matmul"(%in, %in_20, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %0 : !ttcore.tile<32x32, f32>
            }
          }
        }
      }

      // Check final writeback loop structure (2D loop)
      // CHECK: affine.for %[[WB_I:.*]] = 0 to 3 {
      // CHECK-NEXT: affine.for %[[WB_J:.*]] = 0 to 2 {

      // Check writeback: load from dst, store to l1
      // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]][0, %[[WB_I]], %[[WB_J]]] : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>
      // CHECK: affine.store %[[FINAL_VAL]], %cb2[%[[WB_I]], %[[WB_J]]] : memref<3x2x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }
}
