// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @main(%arg0: tensor<1x256x512x512xbf16>, %arg1: tensor<128x256x3x3xbf16>, %arg2: tensor<128xbf16>) -> tensor<1x128x512x512xbf16> {
    %0 = ttir.empty() : tensor<1x128x512x512xbf16>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x512x512xbf16>, tensor<128x256x3x3xbf16>, tensor<1x128x512x512xbf16>) -> tensor<1x128x512x512xbf16>
    %2 = ttir.empty() : tensor<128x1x1xbf16>
    %3 = "ttir.reshape"(%arg2, %2) <{shape = [128 : i32, 1 : i32, 1 : i32]}> : (tensor<128xbf16>, tensor<128x1x1xbf16>) -> tensor<128x1x1xbf16>
    %4 = ttir.empty() : tensor<1x128x512x512xbf16>
    %5 = "ttir.broadcast"(%1, %4) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x128x512x512xbf16>, tensor<1x128x512x512xbf16>) -> tensor<1x128x512x512xbf16>
    %6 = ttir.empty() : tensor<1x128x1x1xbf16>
    %7 = "ttir.reshape"(%3, %6) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xbf16>, tensor<1x128x1x1xbf16>) -> tensor<1x128x1x1xbf16>
    %8 = ttir.empty() : tensor<1x128x512x512xbf16>
    %9 = "ttir.broadcast"(%7, %8) <{broadcast_dimensions = array<i64: 1, 1, 512, 512>}> : (tensor<1x128x1x1xbf16>, tensor<1x128x512x512xbf16>) -> tensor<1x128x512x512xbf16>
    %10 = ttir.empty() : tensor<1x128x512x512xbf16>
    %11 = "ttir.add"(%5, %9, %10) : (tensor<1x128x512x512xbf16>, tensor<1x128x512x512xbf16>, tensor<1x128x512x512xbf16>) -> tensor<1x128x512x512xbf16>
    return %11 : tensor<1x128x512x512xbf16>
  }
}
