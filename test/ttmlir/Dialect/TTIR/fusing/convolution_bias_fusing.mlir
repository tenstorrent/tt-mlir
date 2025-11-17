// RUN: ttmlir-opt -ttir-fusing -ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
func.func @test_conv_transpose_bias_fusing(%arg0: tensor<1x1024x14x14xbf16>, %arg1: tensor<1024x512x2x2xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<2x2x512x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x512x28x28xbf16> {
    // CHECK: func.func @test_conv_transpose_bias_fusing
    // CHECK: "ttir.conv_transpose2d"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    // CHECK-NOT: "ttir.add"
    %0 = ttir.empty() : tensor<1x512x28x28xbf16>
    %1 = "ttir.convolution"(%arg0, %arg3, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 2, kernel_input_feature = 3, kernel_spatial_dimensions = 0x1, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 2, 2>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<2x2x512x1024xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %2 = ttir.empty() : tensor<1x512x1x1xbf16>
    %3 = "ttir.reshape"(%arg2, %2) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512xbf16>, tensor<1x512x1x1xbf16>) -> tensor<1x512x1x1xbf16>
    %4 = ttir.empty() : tensor<1x512x28x28xbf16>
    %5 = "ttir.broadcast"(%3, %4) <{broadcast_dimensions = array<i64: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    %6 = ttir.empty() : tensor<1x512x28x28xbf16>
    %7 = "ttir.add"(%1, %5, %6) : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    return %7 : tensor<1x512x28x28xbf16>
}
}
// -----
module {
func.func @test_conv_bias_fusing(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2:  tensor<1x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x64x112x112xbf16> {
    // CHECK: func.func @test_conv_bias_fusing
    // CHECK: "ttir.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    // CHECK-NOT: "ttir.add"
    %0 = ttir.empty() : tensor<1x64x112x112xbf16>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 3, 3, 3, 3>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %2 = ttir.empty() : tensor<1x64x112x112xbf16>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<1x64x112x112xbf16>, tensor<1x64x1x1xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    return %3 : tensor<1x64x112x112xbf16>
}
}
// -----
module {
func.func @test_conv_bias_fusing_with_broadcast(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2:  tensor<1x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x64x112x112xbf16> {
    // CHECK: func.func @test_conv_bias_fusing_with_broadcast
    // CHECK: "ttir.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    // CHECK-NOT: "ttir.add"
    %0 = ttir.empty() : tensor<1x64x112x112xbf16>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 3, 3, 3, 3>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %2 = ttir.empty() : tensor<1x64x112x112xbf16>
    %3 = "ttir.broadcast"(%arg2, %2) <{broadcast_dimensions = array<i64: 1, 1, 112, 112>}> : (tensor<1x64x1x1xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %4 = "ttir.add"(%1, %3, %2) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    return %4 : tensor<1x64x112x112xbf16>
}
}
// -----
module {
func.func @test_conv_double_add_bias_fusing(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<1x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x64x112x112xbf16> {
    // CHECK: func.func @test_conv_double_add_bias_fusing
    // CHECK: "ttir.add"
    // CHECK: "ttir.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    // CHECK-NOT: "ttir.add"
    %0 = ttir.empty() : tensor<1x64x112x112xbf16>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 3, 3, 3, 3>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %2 = ttir.empty() : tensor<1x64x1x1xbf16>
    %3 = "ttir.add"(%arg2, %arg3, %2) : (tensor<1x64x1x1xbf16>, tensor<1x64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %4 = ttir.empty() : tensor<1x64x112x112xbf16>
    %5 = "ttir.add"(%1, %3, %4) : (tensor<1x64x112x112xbf16>, tensor<1x64x1x1xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    return %5 : tensor<1x64x112x112xbf16>
}
}
