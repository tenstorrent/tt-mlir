// RUN: ttmlir-opt --ttir-to-ttir-decomposition --canonicalize %s | FileCheck %s

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
}
