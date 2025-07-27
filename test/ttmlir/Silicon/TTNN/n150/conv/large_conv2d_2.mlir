
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// This convolution will not pass if the input is in tile layout. The TTNNWorkaround which sets convolution inputs to row-major should allow this to execute.
module {
  func.func @main(%arg0: tensor<1x3x518x518xbf16>, %arg1: tensor<1280x3x14x14xbf16>, %arg2: tensor<1280xbf16>) -> tensor<1x1280x37x37xbf16> {
    %0 = ttir.empty() : tensor<1x1280x37x37xbf16>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 14, 14>}> : (tensor<1x3x518x518xbf16>, tensor<1280x3x14x14xbf16>, tensor<1x1280x37x37xbf16>) -> tensor<1x1280x37x37xbf16>
    %2 = ttir.empty() : tensor<1280x1x1xbf16>
    %3 = "ttir.reshape"(%arg2, %2) <{shape = [1280 : i32, 1 : i32, 1 : i32]}> : (tensor<1280xbf16>, tensor<1280x1x1xbf16>) -> tensor<1280x1x1xbf16>
    %4 = ttir.empty() : tensor<1x1280x37x37xbf16>
    %5 = "ttir.broadcast"(%1, %4) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x37x37xbf16>, tensor<1x1280x37x37xbf16>) -> tensor<1x1280x37x37xbf16>
    %6 = ttir.empty() : tensor<1x1280x1x1xbf16>
    %7 = "ttir.reshape"(%3, %6) <{shape = [1 : i32, 1280 : i32, 1 : i32, 1 : i32]}> : (tensor<1280x1x1xbf16>, tensor<1x1280x1x1xbf16>) -> tensor<1x1280x1x1xbf16>
    %8 = ttir.empty() : tensor<1x1280x37x37xbf16>
    %9 = "ttir.broadcast"(%7, %8) <{broadcast_dimensions = array<i64: 1, 1, 37, 37>}> : (tensor<1x1280x1x1xbf16>, tensor<1x1280x37x37xbf16>) -> tensor<1x1280x37x37xbf16>
    %10 = ttir.empty() : tensor<1x1280x37x37xbf16>
    %11 = "ttir.add"(%5, %9, %10) : (tensor<1x1280x37x37xbf16>, tensor<1x1280x37x37xbf16>, tensor<1x1280x37x37xbf16>) -> tensor<1x1280x37x37xbf16>
    return %11 : tensor<1x1280x37x37xbf16>
  }
}
