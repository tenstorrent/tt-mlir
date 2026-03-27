// RUN: ttmlir-opt %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // Test basic layer_norm_pre_all_gather with input only.
  func.func @forward(%arg0: tensor<1x1x32x128xbf16, #ttnn_layout>) -> tensor<1x1x32x64xbf16, #ttnn_layout_out> {
    // CHECK: "ttnn.layer_norm_pre_all_gather"
    %0 = "ttnn.layer_norm_pre_all_gather"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x1x32x128xbf16, #ttnn_layout>) -> tensor<1x1x32x64xbf16, #ttnn_layout_out>
    return %0 : tensor<1x1x32x64xbf16, #ttnn_layout_out>
  }

  // Test layer_norm_pre_all_gather with residual_input.
  func.func @forward_with_residual(%arg0: tensor<1x1x32x128xbf16, #ttnn_layout>, %arg1: tensor<1x1x32x128xbf16, #ttnn_layout>) -> tensor<1x1x32x64xbf16, #ttnn_layout_out> {
    // CHECK: "ttnn.layer_norm_pre_all_gather"
    %0 = "ttnn.layer_norm_pre_all_gather"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x128xbf16, #ttnn_layout>, tensor<1x1x32x128xbf16, #ttnn_layout>) -> tensor<1x1x32x64xbf16, #ttnn_layout_out>
    return %0 : tensor<1x1x32x64xbf16, #ttnn_layout_out>
  }

  // Test layer_norm_pre_all_gather with dtype attribute.
  func.func @forward_with_dtype(%arg0: tensor<1x1x32x128xbf16, #ttnn_layout>) -> tensor<1x1x32x64xbf16, #ttnn_layout_out> {
    // CHECK: "ttnn.layer_norm_pre_all_gather"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    %0 = "ttnn.layer_norm_pre_all_gather"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x1x32x128xbf16, #ttnn_layout>) -> tensor<1x1x32x64xbf16, #ttnn_layout_out>
    return %0 : tensor<1x1x32x64xbf16, #ttnn_layout_out>
  }
}
