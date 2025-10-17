// RUN: ttmlir-opt --ttir-to-ttir-decomposition --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module @test_convolution {
    func.func @test_conv_output_order_not_nchw() -> tensor<3x3x32x64xbf16> {
    %0 = ttir.empty() : tensor<1x32x26x26xbf16>
    %1 = ttir.empty() : tensor<1x64x24x24xbf16>
    %2 = ttir.empty() : tensor<3x3x32x64xbf16>
    // CHECK: "ttir.permute"
    // CHECK-SAME: <{permutation = array<i64: 1, 2, 3, 0>}>
    // CHECK: "ttir.permute"
    // CHECK-SAME: <{permutation = array<i64: 1, 0, 2, 3>}>
    %3 = "ttir.convolution"(%0, %1, %2) <{
        batch_group_count = 1 : i64,
        convolution_layout = #ttir<
            convolution_layout input_batch = 1,
            input_feature = 0,
            input_spatial_dimensions = 2x3,
            kernel_output_feature = 1,
            kernel_input_feature = 0,
            kernel_spatial_dimensions = 2x3,
            output_batch = 2,
            output_feature = 3,
            output_spatial_dimensions = 0x1>,
        feature_group_count = 1 : i64,
        input_dilation = array<i64: 1, 1>,
        padding = array<i64: 0, 0, 0, 0>,
        weight_dilation = array<i64: 1, 1>,
        window_reversal = array<i1: false, false>,
        window_strides = array<i64: 1, 1>
        }> : (tensor<1x32x26x26xbf16>, tensor<1x64x24x24xbf16>, tensor<3x3x32x64xbf16>) -> tensor<3x3x32x64xbf16>
    // CHECK: "ttir.permute"
    // CHECK-SAME: <{permutation = array<i64: 1, 2, 0, 3>}>
    return %3 : tensor<3x3x32x64xbf16>
  }


  func.func @test_conv_sliced() -> tensor<1x768x768xbf16>{
        %0 = ttir.empty() : tensor<1x9x3072xbf16>
        %1 = ttir.empty() : tensor<1x9x768xbf16>
        %2 = ttir.empty() : tensor<1x768x768xbf16>
        //CHECK-COUNT-8: "ttir.slice_static"
        //CHECK-COUNT-4: "ttir.conv2d"
        //CHECK: "ttir.concat"
        %3 = "ttir.convolution"(%0, %1, %2) <{
            batch_group_count = 4 : i64,
            convolution_layout = #ttir<
                convolution_layout input_batch = 2,
                input_feature = 0,
                input_spatial_dimensions = 1,
                kernel_output_feature = 2,
                kernel_input_feature = 0,
                kernel_spatial_dimensions = 1,
                output_batch = 1,
                output_feature = 2,
                output_spatial_dimensions = 0>,
            feature_group_count = 1 : i64,
            input_dilation = array<i64: 1>,
            padding = array<i64: 0, 0>,
            weight_dilation = array<i64: 1>,
            window_reversal = array<i1: false>,
            window_strides = array<i64: 1>
            }> : (tensor<1x9x3072xbf16>, tensor<1x9x768xbf16>, tensor<1x768x768xbf16>) -> tensor<1x768x768xbf16>
        return %3 : tensor<1x768x768xbf16>
    }


func.func @test_conv2d_sliced() -> tensor<1x16x32x32xbf16> {
    %0 = ttir.empty() : tensor<2x4x32x32xbf16>
    %1 = ttir.empty() : tensor<16x4x3x3xbf16>
    %2 = ttir.empty() : tensor<1x16x32x32xbf16>
    //CHECK-COUNT-4: "ttir.slice_static"
    //CHECK-COUNT-2: "ttir.conv2d"
    //CHECK: "ttir.concat"
    %3 = "ttir.convolution"(%0, %1, %2) <{
        batch_group_count = 2 : i64,
        convolution_layout = #ttir<
            convolution_layout input_batch = 0,
            input_feature = 1,
            input_spatial_dimensions = 2x3,
            kernel_output_feature = 0,
            kernel_input_feature = 1,
            kernel_spatial_dimensions = 2x3,
            output_batch = 0,
            output_feature = 1,
            output_spatial_dimensions = 2x3>,
        feature_group_count = 1 : i64,
        input_dilation = array<i64: 1, 1>,
        padding = array<i64: 1, 1, 1, 1>,
        weight_dilation = array<i64: 1, 1>,
        window_reversal = array<i1: false, false>,
        window_strides = array<i64: 1, 1>
        }> : (tensor<2x4x32x32xbf16>, tensor<16x4x3x3xbf16>, tensor<1x16x32x32xbf16>) -> tensor<1x16x32x32xbf16>
    return %3 : tensor<1x16x32x32xbf16>
  }
}
