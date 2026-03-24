// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>

// A private const-eval function is created to hold the kv_cache typecast.
// CHECK: func.func private @test_fill_cache_bfp8_kv_cache_const_eval_0(
// CHECK-SAME: %arg0: tensor<1x32x64x128xbf16,
// CHECK-SAME: {ttcore.kv_cache}
// CHECK: "ttnn.typecast"(%arg0)
// CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>

// The forward function signature is unchanged (bf16 boundary).
// CHECK-LABEL: func.func @test_fill_cache_bfp8
// CHECK-SAME: %arg0: tensor<1x32x64x128xbf16,
// CHECK-SAME: %arg1: tensor<1x32x1x128xbf16,
module attributes {} {
  func.func @test_fill_cache_bfp8(
      %arg0: tensor<1x32x64x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.kv_cache},
      %arg1: tensor<1x32x1x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  ) attributes {tt.function_type = "forward_device"} {
    // kv_cache is loaded from the const-eval cache (bfp_bf8) at block start.
    // CHECK: %[[CACHED:.*]] = ttcore.load_cached(@test_fill_cache_bfp8_kv_cache_const_eval_0
    // Input is typecast to match the cache dtype.
    // CHECK: %[[INPUT_CAST:.*]] = "ttnn.typecast"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    // fill_cache uses the cached bfp_bf8 tensor and the cast input.
    // CHECK: "ttnn.fill_cache"(%[[CACHED]], %[[INPUT_CAST]])
    "ttnn.fill_cache"(%arg0, %arg1) <{batch_offset = 0 : i32}> : (
        tensor<1x32x64x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
        tensor<1x32x1x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
    ) -> ()
    return
  }
}
