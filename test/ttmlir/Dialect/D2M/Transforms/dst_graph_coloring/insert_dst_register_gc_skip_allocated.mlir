// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc %s | FileCheck %s
//
// Verifies that d2m-insert-dst-register-gc skips d2m.generic regions that
// This ensures the pass is idempotent and doesn't double-allocate.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @skip_already_allocated
  // CHECK: d2m.generic
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK-NOT: d2m.acquire_dst
  // CHECK: d2m.tile_add
  func.func @skip_already_allocated(
    %in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
    %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
    %out: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>
  ) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 :
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
      // This region already has DST allocation - pass should skip it
      %dst = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>
      %c0 = arith.constant 0 : index

      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

      // Load from L1, copy to DST
      %v0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      affine.store %v0, %dst[0, %c0, %c0] : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>
      %dst_v0 = affine.load %dst[0, %c0, %c0] : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>

      %v1 = affine.load %mem1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      affine.store %v1, %dst[1, %c0, %c0] : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>
      %dst_v1 = affine.load %dst[1, %c0, %c0] : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>

      // Compute and write back
      %add_result = "d2m.tile_add"(%dst_v0, %dst_v1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
      affine.store %add_result, %dst[0, %c0, %c0] : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>
      %final = affine.load %dst[0, %c0, %c0] : memref<2x1x1x!ttcore.tile<32x32, f16>, #dst>
      affine.store %final, %mem_out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

    }
    return
  }
}
