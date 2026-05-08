// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>

// CHECK-LABEL: func.func @test_paged_fill_cache_bfp8
// CHECK-SAME: %arg0: tensor<1x32x64x128x!ttcore.tile<32x32, bfp_bf8>,
// CHECK-SAME: %arg1: tensor<1x32x1x128xbf16,
module attributes {} {
  func.func @test_paged_fill_cache_bfp8(
      %arg0: tensor<1x32x64x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.kv_cache},
      %arg1: tensor<1x32x1x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
      %arg2: tensor<1x2xi32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2xi32, #dram>, <interleaved>>>
  ) attributes {tt.function_type = "forward_device"} {
    // CHECK: %[[TYPECAST:.*]] = "ttnn.typecast"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    // CHECK-SAME: -> tensor<1x32x1x128x!ttcore.tile<32x32, bfp_bf8>,
    // CHECK: "ttnn.paged_fill_cache"(%arg0, %[[TYPECAST]], %arg2)
    // CHECK-NOT: "ttnn.typecast"(%arg0)
    "ttnn.paged_fill_cache"(%arg0, %arg1, %arg2) : (
        tensor<1x32x64x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
        tensor<1x32x1x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
        tensor<1x2xi32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2xi32, #dram>, <interleaved>>>
    ) -> ()
    return
  }
}
