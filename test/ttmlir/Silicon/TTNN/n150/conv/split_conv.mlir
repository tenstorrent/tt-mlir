// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
    func.func @test_conv_sliced_batch_group_count() -> tensor<1x768x768xbf16>{
        %0 = ttir.empty() : tensor<1x9x3072xbf16>
        %1 = ttir.empty() : tensor<1x9x768xbf16>
        %2 = ttir.empty() : tensor<1x768x768xbf16>
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


    func.func @test_conv2d_sliced_batch_group_count() -> tensor<1x16x32x32xbf16> {
        %0 = ttir.empty() : tensor<2x4x32x32xbf16>
        %1 = ttir.empty() : tensor<16x4x3x3xbf16>
        %2 = ttir.empty() : tensor<1x16x32x32xbf16>
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
