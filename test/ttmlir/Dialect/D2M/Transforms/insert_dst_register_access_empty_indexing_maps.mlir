// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access --canonicalize -o %t %s --split-input-file
// RUN: FileCheck %s --input-file=%t
//
// Test that InsertDstRegisterAccess correctly handles d2m.generic operations
// in explicit datamovement form (empty block_factors, indexing_maps, and iterator_types).
// The pass should still convert linalg.generic to affine loops and insert DST register access.

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @empty_indexing_maps_test
  func.func @empty_indexing_maps_test(
    %in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
    %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
    %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
  ) {
    // d2m.generic with empty indexing_maps
    d2m.generic {
      block_factors = [],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [],
      iterator_types = [],
      threads = [#d2m.thread<compute>]
    }
    ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
    outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %subview = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_1 = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_2 = memref.subview %cb2[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>

      // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<{{.*}}x!ttcore.tile<32x32, f32>, #dst>
      // CHECK: affine.for
      linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>)
      outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        // Verify DST register access is inserted
        // CHECK: %[[ARG0_VAL:.*]] = affine.load %{{.*}}
        // CHECK: affine.store %[[ARG0_VAL]], %[[DST]]
        // CHECK: %[[ARG1_VAL:.*]] = affine.load %{{.*}}
        // CHECK: affine.store %[[ARG1_VAL]], %[[DST]]
        // CHECK: %[[DST0_VAL:.*]] = affine.load %[[DST]]
        // CHECK: %[[DST1_VAL:.*]] = affine.load %[[DST]]
        // CHECK: %[[ADD_RESULT:.*]] = "d2m.tile_add"(%[[DST0_VAL]], %[[DST1_VAL]])
        %0 = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

        // CHECK: affine.store %[[ADD_RESULT]], %[[DST]]
        // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]]
        // CHECK: affine.store %[[FINAL_VAL]], %{{.*}}
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    return
  }
}

// -----

// Test that exercises the rewriteTileMatmulAsTileMatmulBlock path when
// useTileMatmul is false, indexing_maps is non-empty, and linalg.generic
// contains d2m.tile_matmul. This covers the else branch in
// D2MInsertDstRegisterAccessRewriter::matchAndRewrite that calls
// rewriteTileMatmulAsTileMatmulBlock when indexing_maps is non-empty.

// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access="use-tile-matmul=false" --canonicalize -o %t %s --split-input-file
// RUN: FileCheck %s --check-prefix=CHECK-MATMUL --input-file=%t

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-MATMUL-LABEL: func.func @non_empty_indexing_maps_with_tile_matmul
  func.func @non_empty_indexing_maps_with_tile_matmul(
    %in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
    %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
    %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
  ) {
    // d2m.generic with non-empty indexing_maps
    d2m.generic {
      block_factors = [1, 1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>],
      threads = [#d2m.thread<compute>]
    }
    ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
    outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %subview = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_1 = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_2 = memref.subview %cb2[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>

      // CHECK-MATMUL: %[[DST:.*]] = d2m.acquire_dst() : memref<{{.*}}x!ttcore.tile<32x32, f32>, #dst>
      // CHECK-MATMUL: "d2m.tile_matmul_block"
      linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]
      }
      ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>)
      outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_matmul"(%arg0, %arg1, %arg2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    return
  }
}
