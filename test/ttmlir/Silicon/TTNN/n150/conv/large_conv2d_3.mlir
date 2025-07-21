// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// This convolution will not pass if the input is in tile layout. The TTNNWorkaround which sets convolution inputs to row-major should allow this to execute.
module {
  func.func @main(%arg0: tensor<1x128x180x320xbf16>, %arg1: tensor<128x128x3x3xbf16>) -> tensor<1x128x90x160xbf16> {
    %0 = ttir.empty() : tensor<1x128x90x160xbf16>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x128x180x320xbf16>, tensor<128x128x3x3xbf16>, tensor<1x128x90x160xbf16>) -> tensor<1x128x90x160xbf16>
    return %1 : tensor<1x128x90x160xbf16>
  }
}
