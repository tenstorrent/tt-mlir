// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access="use-tile-matmul=false max-dst-physical-size-tiles=32" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
module {
  func.func @no_loops(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                      %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                      %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>):
      %c0 = arith.constant 0 : index
      %subview = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_1 = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_2 = memref.subview %cb2[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
        // CHECK: %[[GUARD_IDX:.*]] = d2m.iter_index
        // CHECK: %[[NOT_FIRST:.*]] = arith.cmpi ne, %[[GUARD_IDX]]
        // CHECK: scf.if %[[NOT_FIRST]]
        // CHECK: %[[ARG2_VAL:.*]] = affine.load %[[ARG2:.*]]
        // Check that the third operand (accumulator) is stored to dst memory space
        // CHECK: affine.store %[[ARG2_VAL]], %[[DST]]
        // CHECK: "d2m.tile_matmul_block"
        %0 = "d2m.tile_matmul"(%arg0, %arg1, %arg2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        // Check that result is loaded from dst memory space
        // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]]
        // Check that final result is stored back to original #l1 memory space
        // CHECK: affine.store %[[FINAL_VAL]], %[[ARG2]]
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    return
  }

  func.func @generic_matmul(
    %in0: memref<1x1x3x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #l1_>,
    %in1: memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>,
    %out0: memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
    ) {
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x3x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #l1_>, memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>)
        outs(%out0 : memref<1x1x3x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<3x3x!ttcore.tile<32x32, f32>, #l1_>>, %arg1_cb: !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>>, %arg2_cb: !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %cb0 = d2m.pop %arg0_cb : !d2m.cb<memref<3x3x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x3x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.pop %arg1_cb : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
      // Check that constants and destination buffer are created and blocks are created as casts from the CBs
      // CHECK: %[[C0:.*]] = arith.constant 0 : index
      // CHECK: %[[blockA:.*]] = memref.cast {{%.*}} : memref<3x3x!ttcore.tile<32x32, f32>, #l1> to memref<3x3x!ttcore.tile<32x32, f32>, strided<[3, 1], offset: ?>, #l1>
      // CHECK: %[[blockB:.*]] = memref.cast {{%.*}} : memref<3x2x!ttcore.tile<32x32, f32>, #l1> to memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #l1>
      // CHECK: %[[blockOut:.*]] = memref.cast {{%.*}} : memref<3x2x!ttcore.tile<32x32, f32>, #l1> to memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #l1>
      // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>

      // Check for iteration index and conditional initialization
      // CHECK: %[[ITER2:.*]] = d2m.iter_index(2) : index
      // CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[ITER2]], %[[C0]] : index
      // CHECK: scf.if %[[CMP]] {

      // Check conditional initialization loop structure (2D loop for initialization)
      // CHECK: affine.for %[[INIT_I:.*]] = 0 to 3 {
      // CHECK-NEXT: affine.for %[[INIT_J:.*]] = 0 to 2 {

      // Check initialization: load from l1, store to dst
      // CHECK: %[[INIT_VAL:.*]] = affine.load {{%.*}}[%[[INIT_I]], %[[INIT_J]]] : memref<3x2x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: affine.store %[[INIT_VAL]], %[[DST]][0, %[[INIT_I]], %[[INIT_J]]] : memref<1x3x2x!ttcore.tile<32x32, f32>, #dst>

      // Check matmul operation uses values from correct memory spaces
      // CHECK: "d2m.tile_matmul_block"(%[[blockA]], %[[blockB]], %[[blockOut]]) : (memref<3x3x!ttcore.tile<32x32, f32>, strided<[3, 1], offset: ?>, #l1>, memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #l1>, memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #l1>) -> ()

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
            %subview = memref.subview %cb0[%arg2, %arg4] [3, 3] [1, 1] : memref<3x3x!ttcore.tile<32x32, f32>, #l1_> to memref<3x3x!ttcore.tile<32x32, f32>, strided<[3, 1], offset: ?>, #l1_>
            %subview_18 = memref.subview %cb1[%arg4, %arg3] [3, 2] [1, 1] : memref<3x2x!ttcore.tile<32x32, f32>, #l1_> to memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #l1_>
            %subview_19 = memref.subview %cb2[%arg2, %arg3] [3, 2] [1, 1] : memref<3x2x!ttcore.tile<32x32, f32>, #l1_> to memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #l1_>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview, %subview_18 : memref<3x3x!ttcore.tile<32x32, f32>, strided<[3, 1], offset: ?>, #l1_>, memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #l1_>) outs(%subview_19 : memref<3x2x!ttcore.tile<32x32, f32>, strided<[2, 1], offset: ?>, #l1_>) {
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
      // CHECK: affine.store %[[FINAL_VAL]], {{%.*}}[%[[WB_I]], %[[WB_J]]] : memref<3x2x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }
}
