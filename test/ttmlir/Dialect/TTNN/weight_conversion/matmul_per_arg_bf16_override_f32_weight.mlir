// RUN: ttmlir-opt --ttnn-weight-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

// Test bf16 per-op override when the weight is f32 and the global default is
// bfp_bf8. The per-op override must win: only a single f32->bf16 typecast
// should be inserted, NOT a chained f32->bf16->bfp_bf8. This guards against
// a greedy-rewriter re-match after the attribute is consumed.

#dram = #ttnn.buffer_type<dram>

module attributes {} {
  // CHECK-LABEL: func.func @test_bf16_override_f32_weight
  func.func @test_bf16_override_f32_weight(
    %arg0: tensor<64x720xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x23x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>>,
    %arg1: tensor<720x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<23x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}
  ) -> tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>> {

    // Per-op says bf16. Should insert exactly one f32->bf16 typecast.
    // CHECK: %[[TYPECAST:.*]] = "ttnn.typecast"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: -> tensor<720x128xbf16,

    // Must NOT see a second typecast to bfp_bf8 from the global default.
    // CHECK-NOT: supportedDataTypes<bfp_bf8>

    // CHECK: "ttnn.matmul"(%arg0, %[[TYPECAST]])
    %0 = "ttnn.matmul"(%arg0, %arg1) {ttcore.weight_dtype = "bf16"} : (tensor<64x720xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x23x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>>, tensor<720x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<23x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>>) -> tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>>

    return %0 : tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>>
  }
}
