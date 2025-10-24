// RUN: ttmlir-opt --ttir-fusing="ttnn-enable-conv2d-with-multiply-pattern=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
func.func @test_batch_norm_decomposition(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg4: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg5: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x64x112x112xbf16> {
    %0 = ttir.empty() : tensor<1x64x112x112xbf16>
    // CHECK: "ttir.add"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.convolution"
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 3, 3, 3, 3>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %2 = ttir.empty() : tensor<1x64x112x112xbf16>
    %3 = "ttir.batch_norm_inference"(%1, %arg2, %arg3, %arg4, %arg5, %2) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x64x112x112xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    return %3 : tensor<1x64x112x112xbf16>
}
}
