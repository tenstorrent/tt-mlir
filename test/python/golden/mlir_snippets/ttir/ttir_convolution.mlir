module {
  func.func @model(%arg0: tensor<12x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<12x64x112x112xf32> {
    %1 = "ttir.convolution"(%arg0, %arg1) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 3, 3, 3, 3>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<12x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<12x64x112x112xf32>
    return %1 : tensor<12x64x112x112xf32>
  }
}
