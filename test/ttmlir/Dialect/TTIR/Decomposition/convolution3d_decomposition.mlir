// RUN: ttmlir-opt --ttir-to-ttir-decomposition --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module @test_convolution3d {

    // Test 3D convolution with different layout (NCDHW -> NDHWC conversion)
    func.func @test_conv3d_layout_conversion() -> tensor<1x16x6x26x26xbf16> {
    %0 = ttir.empty() : tensor<1x4x8x28x28xbf16>
    %1 = ttir.empty() : tensor<16x4x3x3x3xbf16>
    // CHECK: "ttir.permute"
    // CHECK: "ttir.conv3d"
    // CHECK: "ttir.permute"
    %2 = "ttir.convolution"(%0, %1) <{
        batch_group_count = 1 : i64,
        convolution_layout = #ttir<
            convolution_layout input_batch = 0,
            input_feature = 1,
            input_spatial_dimensions = 2x3x4,
            kernel_output_feature = 0,
            kernel_input_feature = 1,
            kernel_spatial_dimensions = 2x3x4,
            output_batch = 0,
            output_feature = 1,
            output_spatial_dimensions = 2x3x4>,
        feature_group_count = 1 : i64,
        input_dilation = array<i64: 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 0>,
        weight_dilation = array<i64: 1, 1, 1>,
        window_reversal = array<i1: false, false, false>,
        window_strides = array<i64: 1, 1, 1>
        }> : (tensor<1x4x8x28x28xbf16>, tensor<16x4x3x3x3xbf16>) -> tensor<1x16x6x26x26xbf16>
    return %2 : tensor<1x16x6x26x26xbf16>
  }

    // Test 3D convolution with stride and padding
    func.func @test_conv3d_stride_padding() -> tensor<1x32x4x14x14xbf16> {
    %0 = ttir.empty() : tensor<1x16x8x28x28xbf16>
    %1 = ttir.empty() : tensor<32x16x3x3x3xbf16>
    // CHECK: "ttir.permute"
    // CHECK: "ttir.conv3d"
    // CHECK: "ttir.permute"
    %2 = "ttir.convolution"(%0, %1) <{
        batch_group_count = 1 : i64,
        convolution_layout = #ttir<
            convolution_layout input_batch = 0,
            input_feature = 1,
            input_spatial_dimensions = 2x3x4,
            kernel_output_feature = 0,
            kernel_input_feature = 1,
            kernel_spatial_dimensions = 2x3x4,
            output_batch = 0,
            output_feature = 1,
            output_spatial_dimensions = 2x3x4>,
        feature_group_count = 1 : i64,
        input_dilation = array<i64: 1, 1, 1>,
        padding = array<i64: 1, 1, 1, 1, 1, 1>,
        weight_dilation = array<i64: 1, 1, 1>,
        window_reversal = array<i1: false, false, false>,
        window_strides = array<i64: 2, 2, 2>
        }> : (tensor<1x16x8x28x28xbf16>, tensor<32x16x3x3x3xbf16>) -> tensor<1x32x4x14x14xbf16>
    return %2 : tensor<1x32x4x14x14xbf16>
  }

    // Test 3D convolution with bias
    func.func @test_conv3d_with_bias(%bias: tensor<1x16x1x1x1xbf16>) -> tensor<1x16x6x26x26xbf16> {
    %0 = ttir.empty() : tensor<1x4x8x28x28xbf16>
    %1 = ttir.empty() : tensor<16x4x3x3x3xbf16>
    // CHECK: "ttir.permute"
    // CHECK: "ttir.conv3d"
    // CHECK: "ttir.permute"
    %2 = "ttir.convolution"(%0, %1, %bias) <{
        batch_group_count = 1 : i64,
        convolution_layout = #ttir<
            convolution_layout input_batch = 0,
            input_feature = 1,
            input_spatial_dimensions = 2x3x4,
            kernel_output_feature = 0,
            kernel_input_feature = 1,
            kernel_spatial_dimensions = 2x3x4,
            output_batch = 0,
            output_feature = 1,
            output_spatial_dimensions = 2x3x4>,
        feature_group_count = 1 : i64,
        input_dilation = array<i64: 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 0>,
        weight_dilation = array<i64: 1, 1, 1>,
        window_reversal = array<i1: false, false, false>,
        window_strides = array<i64: 1, 1, 1>
        }> : (tensor<1x4x8x28x28xbf16>, tensor<16x4x3x3x3xbf16>, tensor<1x16x1x1x1xbf16>) -> tensor<1x16x6x26x26xbf16>
    return %2 : tensor<1x16x6x26x26xbf16>
  }
}
