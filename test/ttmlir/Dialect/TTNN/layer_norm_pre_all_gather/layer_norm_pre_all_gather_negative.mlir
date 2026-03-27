// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_small = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16 + d1, d2, d3), <1x1>, memref<16x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // CHECK: error: 'ttnn.layer_norm_pre_all_gather' op residual_input shape must match input shape
  func.func @forward_residual_shape_mismatch(%arg0: tensor<1x1x32x128xbf16, #ttnn_layout>, %arg1: tensor<1x1x16x128xbf16, #ttnn_layout_small>) -> tensor<1x1x32x64xbf16, #ttnn_layout_out> {
    %0 = "ttnn.layer_norm_pre_all_gather"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x32x128xbf16, #ttnn_layout>, tensor<1x1x16x128xbf16, #ttnn_layout_small>) -> tensor<1x1x32x64xbf16, #ttnn_layout_out>
    return %0 : tensor<1x1x32x64xbf16, #ttnn_layout_out>
  }
}

// -----

#dram2 = #ttnn.buffer_type<dram>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram2>, <interleaved>>
#ttnn_layout_bad_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram2>, <interleaved>>

module {
  // CHECK: error: 'ttnn.layer_norm_pre_all_gather' op output last dimension must be 64 (2 * TILE_WIDTH) for layernorm partial statistics, got 32
  func.func @forward_bad_output_shape(%arg0: tensor<1x1x32x128xbf16, #ttnn_layout2>) -> tensor<1x1x32x32xbf16, #ttnn_layout_bad_out> {
    %0 = "ttnn.layer_norm_pre_all_gather"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x1x32x128xbf16, #ttnn_layout2>) -> tensor<1x1x32x32xbf16, #ttnn_layout_bad_out>
    return %0 : tensor<1x1x32x32xbf16, #ttnn_layout_bad_out>
  }
}
