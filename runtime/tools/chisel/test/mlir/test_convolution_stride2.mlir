module {
  func.func @convolution_stride2(%arg0: tensor<1x3x16x16xf32>, %arg1: tensor<8x3x3x3xf32>) -> tensor<1x8x8x8xf32> {
    %0 = ttir.empty() : tensor<1x8x8x8xf32>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x3x16x16xf32>, tensor<8x3x3x3xf32>, tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
    return %1 : tensor<1x8x8x8xf32>
  }
}
