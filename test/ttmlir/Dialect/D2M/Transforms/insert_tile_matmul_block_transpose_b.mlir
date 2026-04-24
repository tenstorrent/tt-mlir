// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-access --d2m-insert-tile-matmul-block --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that the `transpose_b` attribute on `d2m.tile_matmul` is forwarded to
// the lowered `d2m.tile_matmul_block` op by the D2MInsertTileMatmulBlock pass.
// The block-level RHS shape here is (N=2, K=3) instead of the usual (K, N)
// because B is semantically transposed.

#l1_ = #ttcore.memory_space<l1>
module {
  // CHECK-LABEL: func.func @generic_matmul_transpose_b
  func.func @generic_matmul_transpose_b(
    %in0: memref<1x1x3x3x!ttcore.tile<32x32, f16>, #ttcore.shard<6144x2048, 1>, #l1_>,
    %in1: memref<1x1x2x3x!ttcore.tile<32x32, f16>, #ttcore.shard<6144x2048, 1>, #l1_>,
    %out0: memref<1x1x3x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>
    ) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %in1 : memref<1x1x3x3x!ttcore.tile<32x32, f16>, #ttcore.shard<6144x2048, 1>, #l1_>, memref<1x1x2x3x!ttcore.tile<32x32, f16>, #ttcore.shard<6144x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x3x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>)  {
    ^unified0:
      %arg0_cb = d2m.get_cb(0) : !d2m.cb<memref<3x3x!ttcore.tile<32x32, f16>, #l1_>>
      %arg1_cb = d2m.get_cb(1) : !d2m.cb<memref<2x3x!ttcore.tile<32x32, f16>, #l1_>>
      %arg2_cb = d2m.get_cb(2) : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f16>, #l1_>>
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<3x3x!ttcore.tile<32x32, f16>, #l1_>> -> memref<3x3x!ttcore.tile<32x32, f16>, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<2x3x!ttcore.tile<32x32, f16>, #l1_>> -> memref<2x3x!ttcore.tile<32x32, f16>, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<3x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<3x2x!ttcore.tile<32x32, f16>, #l1_>

      // The compute loop below is replaced by a single `d2m.tile_matmul_block`
      // that carries the same `transpose_b = true` flag as the inner
      // `d2m.tile_matmul`.
      // CHECK: "d2m.tile_matmul_block"(%[[A:.*]], %[[B:.*]], %[[OUT:.*]]) {{<\{transpose_b = true\}>}}
      // CHECK-NOT: "d2m.tile_matmul"
      // CHECK-NOT: transpose_b = false

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
            %subview = memref.subview %cb0[%arg2, %arg4] [3, 3] [1, 1] : memref<3x3x!ttcore.tile<32x32, f16>, #l1_> to memref<3x3x!ttcore.tile<32x32, f16>, strided<[3, 1], offset: ?>, #l1_>
            // RHS subview mirrors the transposed (N, K) block layout.
            %subview_18 = memref.subview %cb1[%arg3, %arg4] [2, 3] [1, 1] : memref<2x3x!ttcore.tile<32x32, f16>, #l1_> to memref<2x3x!ttcore.tile<32x32, f16>, strided<[3, 1], offset: ?>, #l1_>
            %subview_19 = memref.subview %cb2[%arg2, %arg3] [3, 2] [1, 1] : memref<3x2x!ttcore.tile<32x32, f16>, #l1_> to memref<3x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview, %subview_18 : memref<3x3x!ttcore.tile<32x32, f16>, strided<[3, 1], offset: ?>, #l1_>, memref<2x3x!ttcore.tile<32x32, f16>, strided<[3, 1], offset: ?>, #l1_>) outs(%subview_19 : memref<3x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>) {
            ^bb0(%in: !ttcore.tile<32x32, f16>, %in_20: !ttcore.tile<32x32, f16>, %out: !ttcore.tile<32x32, f16>):
              %0 = "d2m.tile_matmul"(%in, %in_20, %out) <{transpose_b = true}> : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
              linalg.yield %0 : !ttcore.tile<32x32, f16>
            }
          }
        }
      }
    }
    return
  }
}
