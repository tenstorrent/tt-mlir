module {
  func.func @main_conv_reverse(%arg_input: tensor<1x64x1x6400xf32>, %arg_weight: tensor<1x8x32x64xf32>, %arg_bias: tensor<32xf32>) -> tensor<1x32x1x25600xf32> {
    %999 = ttir.empty() : tensor<1x8x32x64xf32>
    %1000 = "ttir.reverse"(%arg_weight, %999) <{dimensions = array<i64: 0, 1>}> : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x64xf32>
    %1001 = ttir.empty() : tensor<1x32x1x25600xf32>
    %1002 = ttir.empty() : tensor<1x32x1x1xf32>
    %1003 = "ttir.reshape"(%arg_bias, %1002) <{shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<32xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1004 = "ttir.convolution"(%arg_input, %1000, %1003, %1001) <{
        batch_group_count = 1 : i64,
        convolution_layout = #ttir<
            convolution_layout 
            input_batch = 0, // N[0] -> 1
            input_feature = 1, // C[1] -> 64
            input_spatial_dimensions = 2x3, // 2x3 HxW (1x6400)
            kernel_output_feature = 2,
            kernel_input_feature = 3,
            kernel_spatial_dimensions = 0x1,
            output_batch = 0,
            output_feature = 1,
            output_spatial_dimensions = 2x3>,
        feature_group_count = 1 : i64,
        input_dilation = array<i64: 1, 4>,
        padding = array<i64: 0, 0, 5, 5>,
        weight_dilation = array<i64: 1, 1>,
        window_reversal = array<i1: false, false>,
        window_strides = array<i64: 1, 1>}> : (tensor<1x64x1x6400xf32>, tensor<1x8x32x64xf32>, tensor<1x32x1x1xf32>, tensor<1x32x1x25600xf32>) -> tensor<1x32x1x25600xf32>
    return %1004 : tensor<1x32x1x25600xf32>
  }
}