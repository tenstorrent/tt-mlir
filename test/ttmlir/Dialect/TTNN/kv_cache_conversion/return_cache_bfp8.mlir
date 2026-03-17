// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>

// CHECK-LABEL: func.func @test_return_cache_bfp8
// CHECK-SAME: %arg0: tensor<1x32x64x128x!ttcore.tile<32x32, bfp_bf8>,
// CHECK-SAME: %arg1: tensor<1x32x1x128xbf16,
// CHECK-SAME: -> tensor<1x32x64x128x!ttcore.tile<32x32, bfp_bf8>,
// CHECK-NOT: -> tensor<1x32x64x128xbf16
module attributes {} {
  func.func @test_return_cache_bfp8(
      %arg0: tensor<1x32x64x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.kv_cache},
      %arg1: tensor<1x32x1x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  ) -> tensor<1x32x64x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {
    "ttnn.fill_cache"(%arg0, %arg1) <{batch_offset = 0 : i32}> : (
        tensor<1x32x64x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
        tensor<1x32x1x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
    ) -> ()
    // CHECK: return %arg0 : tensor<1x32x64x128x!ttcore.tile<32x32, bfp_bf8>,
    return %arg0 : tensor<1x32x64x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
