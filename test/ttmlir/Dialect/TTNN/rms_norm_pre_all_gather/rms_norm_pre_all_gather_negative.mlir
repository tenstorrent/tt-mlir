// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_small = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16 + d1, d2, d3), <1x1>, memref<16x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>


// residual tensor must have same shape as input
module attributes {} {
  func.func @rms_norm_pre_all_gather_residual_shape_mismatch(%input: tensor<1x1x128x64xbf16, #ttnn_layout>, %residual: tensor<1x1x128x16xbf16, #ttnn_layout_small>) -> tensor<1x1x128x32xbf16, #ttnn_layout_out> {
    %0 = "ttnn.rms_norm_pre_all_gather"(%input, %residual) <{dtype = #ttcore.supportedDataTypes<bf16>, use_2d_core_grid = false}> : (tensor<1x1x128x64xbf16, #ttnn_layout>, tensor<1x1x128x16xbf16, #ttnn_layout_small>) -> tensor<1x1x128x32xbf16, #ttnn_layout_out>
    return %0 : tensor<1x1x128x32xbf16, #ttnn_layout_out>
  }
}
// CHECK: error: 'ttnn.rms_norm_pre_all_gather' op residual tensor shape must match the input tensor shape
