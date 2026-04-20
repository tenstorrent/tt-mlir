// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

// Layouts for non-tile-aligned input width W=34 (padded to 64 = 2 tiles).
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_weight = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Layouts for tile-aligned input width W=64 (should NOT trigger decomposition).
#ttnn_layout_aligned = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_weight_aligned = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module @test_layer_norm_post_all_gather_decomposition attributes {} {

  // Test: Non-tile-aligned input (W=34) with no weight or bias.
  // Pattern must fire and replace the op with primitive ops.
  func.func public @test_decompose_no_weight_no_bias(
      %arg0: tensor<1x1x32x34xbf16, #ttnn_layout>,
      %arg1: tensor<1x1x32x64xbf16, #ttnn_layout>) -> tensor<1x1x32x34xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_decompose_no_weight_no_bias
    // Slice sum(x²) and sum(x) from stats
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // winv constant and E[x²], E[x]
    // CHECK: "ttnn.full"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // var = E[x²] - E[x]*E[x]
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.subtract"
    // rstd = rsqrt(var + eps)
    // CHECK: "ttnn.full"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.rsqrt"
    // normalized = (input - E[x]) * rstd
    // CHECK: "ttnn.subtract"
    // CHECK: "ttnn.multiply"
    // No weight or bias ops expected
    // CHECK-NOT: "ttnn.add"{{.*}}_bias
    // Fully decomposed — no layer_norm_post_all_gather remains
    // CHECK-NOT: "ttnn.layer_norm_post_all_gather"
    %0 = "ttnn.layer_norm_post_all_gather"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<1x1x32x34xbf16, #ttnn_layout>, tensor<1x1x32x64xbf16, #ttnn_layout>) -> tensor<1x1x32x34xbf16, #ttnn_layout>
    return %0 : tensor<1x1x32x34xbf16, #ttnn_layout>
  }

  // Test: Non-tile-aligned (W=34) with weight and bias — all primitive ops emitted.
  func.func public @test_decompose_with_weight_and_bias(
      %arg0: tensor<1x1x32x34xbf16, #ttnn_layout>,
      %arg1: tensor<1x1x32x64xbf16, #ttnn_layout>,
      %arg2: tensor<34xbf16, #ttnn_layout_weight>,
      %arg3: tensor<34xbf16, #ttnn_layout_weight>) -> tensor<1x1x32x34xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_decompose_with_weight_and_bias
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.full"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.subtract"
    // CHECK: "ttnn.full"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.rsqrt"
    // CHECK: "ttnn.subtract"
    // CHECK: "ttnn.multiply"
    // Weight applied
    // CHECK: "ttnn.multiply"
    // Bias applied
    // CHECK: "ttnn.add"
    // CHECK-NOT: "ttnn.layer_norm_post_all_gather"
    %0 = "ttnn.layer_norm_post_all_gather"(%arg0, %arg1, %arg2, %arg3) <{dtype = #ttcore.supportedDataTypes<bf16>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<1x1x32x34xbf16, #ttnn_layout>, tensor<1x1x32x64xbf16, #ttnn_layout>, tensor<34xbf16, #ttnn_layout_weight>, tensor<34xbf16, #ttnn_layout_weight>) -> tensor<1x1x32x34xbf16, #ttnn_layout>
    return %0 : tensor<1x1x32x34xbf16, #ttnn_layout>
  }

  // Test: Tile-aligned input (W=64) — pattern must NOT fire.
  func.func public @test_no_decompose_tile_aligned(
      %arg0: tensor<1x1x32x64xbf16, #ttnn_layout_aligned>,
      %arg1: tensor<1x1x32x64xbf16, #ttnn_layout_aligned>) -> tensor<1x1x32x64xbf16, #ttnn_layout_aligned> {
    // CHECK-LABEL: func.func public @test_no_decompose_tile_aligned
    // Kernel is correct when W is tile-aligned — op must survive.
    // CHECK: "ttnn.layer_norm_post_all_gather"
    // CHECK-NOT: "ttnn.slice_static"
    // CHECK-NOT: "ttnn.rsqrt"
    %0 = "ttnn.layer_norm_post_all_gather"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<1x1x32x64xbf16, #ttnn_layout_aligned>, tensor<1x1x32x64xbf16, #ttnn_layout_aligned>) -> tensor<1x1x32x64xbf16, #ttnn_layout_aligned>
    return %0 : tensor<1x1x32x64xbf16, #ttnn_layout_aligned>
  }

}
