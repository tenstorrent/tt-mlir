// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>

// CHECK-LABEL: func.func @test_paged_update_cache_bfp8
// CHECK-SAME: %arg0: tensor<1x32x64x128x!ttcore.tile<32x32, bfp_bf8>,
// CHECK-SAME: %arg1: tensor<1x32x1x128xbf16,
module attributes {} {
  func.func @test_paged_update_cache_bfp8(
      %arg0: tensor<1x32x64x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.kv_cache},
      %arg1: tensor<1x32x1x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
      %arg2: tensor<32xi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xi32, #dram>, <interleaved>>>
  ) {
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.paged_update_cache"(%arg0, %arg1, %arg2)
    "ttnn.paged_update_cache"(%arg0, %arg1, %arg2) <{share_cache = false}> : (
        tensor<1x32x64x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
        tensor<1x32x1x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
        tensor<32xi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xi32, #dram>, <interleaved>>>
    ) -> ()
    return
  }
}
