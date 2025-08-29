// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
module @jit_convolution_bad_spatial_dimensions {
  func.func public @test_illegal_convolution(%arg0: tensor<1x3x100x100xbf16>, %arg1: tensor<7x3x3x3xbf16>) -> tensor<1x7x100x100xbf16> {
    %0 = ttir.empty() : tensor<1x7x100x100xbf16>
    // CHECK: error: 'ttir.convolution' op Convolution input, output, and kernel must have the same number of spatial dimensions
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<convolution_layout
        input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2
      >,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      padding = array<i64: 1, 1, 1 ,1>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1>
    }> : (tensor<1x3x100x100xbf16>, tensor<7x3x3x3xbf16>, tensor<1x7x100x100xbf16>) -> tensor<1x7x100x100xbf16>
    return %1 : tensor<1x7x100x100xbf16>
  }
}

// -----
module @jit_convolution_bad_stride_dimensions {
  func.func public @test_illegal_convolution(%arg0: tensor<1x3x100x100xbf16>, %arg1: tensor<7x3x3x3xbf16>) -> tensor<1x7x100x100xbf16> {
    %0 = ttir.empty() : tensor<1x7x100x100xbf16>
    // CHECK: error: 'ttir.convolution' op Window strides must have the same number of elements as the spatial dimensions of the input tensor
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<convolution_layout
        input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2x3
      >,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      padding = array<i64: 1, 1, 1 ,1>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1, 1>
    }> : (tensor<1x3x100x100xbf16>, tensor<7x3x3x3xbf16>, tensor<1x7x100x100xbf16>) -> tensor<1x7x100x100xbf16>
    return %1 : tensor<1x7x100x100xbf16>
  }
}

// -----
module @jit_convolution_bad_input_tensor {
  func.func public @test_illegal_convolution(%arg0: tensor<1x3x100x100x100xbf16>, %arg1: tensor<7x3x3x3xbf16>) -> tensor<1x7x100x100xbf16> {
    %0 = ttir.empty() : tensor<1x7x100x100xbf16>
    // CHECK: error: 'ttir.convolution' op Input tensor must have the same number of spatial dimensions as specified in the ConvolutionLayout
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<convolution_layout
        input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2x3
      >,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      padding = array<i64: 1, 1, 1 ,1>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1>
    }> : (tensor<1x3x100x100x100xbf16>, tensor<7x3x3x3xbf16>, tensor<1x7x100x100xbf16>) -> tensor<1x7x100x100xbf16>
    return %1 : tensor<1x7x100x100xbf16>
  }
}

// -----
module @jit_convolution_bad_weight_tensor {
  func.func public @test_illegal_convolution(%arg0: tensor<1x3x100x100xbf16>, %arg1: tensor<20x7x3x3x3xbf16>) -> tensor<1x7x100x100xbf16> {
    %0 = ttir.empty() : tensor<1x7x100x100xbf16>
    // CHECK: error: 'ttir.convolution' op Weight tensor must have the same number of spatial dimensions as specified in the ConvolutionLayout
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<convolution_layout
        input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2x3
      >,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      padding = array<i64: 1, 1, 1 ,1>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1>
    }> : (tensor<1x3x100x100xbf16>, tensor<20x7x3x3x3xbf16>, tensor<1x7x100x100xbf16>) -> tensor<1x7x100x100xbf16>
    return %1 : tensor<1x7x100x100xbf16>
  }
}

// -----
module @jit_convolution_bad_bias_tensor {
  func.func public @test_illegal_convolution(%arg0: tensor<1x3x100x100xbf16>, %arg1: tensor<7x3x3x3xbf16>, %arg2: tensor<1x1x7xbf16>) -> tensor<1x7x100x100xbf16> {
    %0 = ttir.empty() : tensor<1x7x100x100xbf16>
    // CHECK: error: 'ttir.convolution' op Bias tensor must have the same rank as the output tensor
    %1 = "ttir.convolution"(%arg0, %arg1, %arg2, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<convolution_layout
        input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2x3
      >,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      padding = array<i64: 1, 1, 1 ,1>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1>
    }> : (tensor<1x3x100x100xbf16>, tensor<7x3x3x3xbf16>, tensor<1x1x7xbf16>, tensor<1x7x100x100xbf16>) -> tensor<1x7x100x100xbf16>
    return %1 : tensor<1x7x100x100xbf16>
  }
}

// -----
module @jit_convolution_bad_bias_dimensions {
  func.func public @test_illegal_convolution(%arg0: tensor<1x3x100x100xbf16>, %arg1: tensor<7x3x3x3xbf16>, %arg2: tensor<2x7x2x3xbf16>) -> tensor<1x7x100x100xbf16> {
    %0 = ttir.empty() : tensor<1x7x100x100xbf16>
    // CHECK: error: 'ttir.convolution' op Bias tensor must have size 1 in all dimensions except the output feature dimension
    %1 = "ttir.convolution"(%arg0, %arg1, %arg2, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<convolution_layout
        input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2x3
      >,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      padding = array<i64: 1, 1, 1 ,1>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1>
    }> : (tensor<1x3x100x100xbf16>, tensor<7x3x3x3xbf16>, tensor<2x7x2x3xbf16>, tensor<1x7x100x100xbf16>) -> tensor<1x7x100x100xbf16>
    return %1 : tensor<1x7x100x100xbf16>
  }
}
