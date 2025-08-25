// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true enable-fusing-conv2d-with-multiply-pattern=true" -o %t %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t

module {
func.func @resnet50_bottleneck_2048x1024(%arg0: tensor<1x1024x14x14xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg2: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg4: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg5: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg6: tensor<2048x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg223: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg224: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg225: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg226: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg227: tensor<2048x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg228: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg229: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg230: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg231: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg232: tensor<512x512x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg233: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg234: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg235: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg236: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg237: tensor<512x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x2048x7x7xbf16> {
    // CHECK: func.func @resnet50_bottleneck_2048x1024
    %2 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x512x7x7xbf16>}> : () -> tensor<1x512x7x7xbf16>
    %3 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x512x14x14xbf16>}> : () -> tensor<1x512x14x14xbf16>
    %296 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x1024x14x14xbf16>}> : () -> tensor<1x1024x14x14xbf16>
    %297 = ttir.empty() : tensor<1x1024x14x14xbf16>
    // CHECK: "ttnn.relu"
    %298 = "ttir.maximum"(%arg0, %296, %297) : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %299 = ttir.empty() : tensor<1x512x14x14xbf16>
    // CHECK: "ttnn.conv2d"
    %300 = "ttir.convolution"(%298, %arg237, %299) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x1024x14x14xbf16>, tensor<512x1024x1x1xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %301 = ttir.empty() : tensor<1x512x14x14xbf16>
    // CHECK-NOT: "ttnn.batch_norm"
    // CHECK-NOT: "ttnn.multiply"
    // CHECK-NOT: "ttnn.add"
    %302 = "ttir.batch_norm"(%300, %arg236, %arg235, %arg234, %arg233, %301) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x512x14x14xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %303 = ttir.empty() : tensor<1x512x14x14xbf16>
    %304 = "ttir.maximum"(%302, %3, %303) : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    %305 = ttir.empty() : tensor<1x512x7x7xbf16>
    // CHECK: "ttnn.conv2d"
    %306 = "ttir.convolution"(%304, %arg232, %305) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x512x14x14xbf16>, tensor<512x512x3x3xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %307 = ttir.empty() : tensor<1x512x7x7xbf16>
    %308 = "ttir.batch_norm"(%306, %arg231, %arg230, %arg229, %arg228, %307) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x512x7x7xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %309 = ttir.empty() : tensor<1x512x7x7xbf16>
    %310 = "ttir.maximum"(%308, %2, %309) : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %311 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %312 = "ttir.convolution"(%310, %arg227, %311) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %313 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %314 = "ttir.batch_norm"(%312, %arg226, %arg225, %arg224, %arg223, %313) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x2048x7x7xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %315 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %316 = "ttir.convolution"(%298, %arg6, %315) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<1x1024x14x14xbf16>, tensor<2048x1024x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %317 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %318 = "ttir.batch_norm"(%316, %arg5, %arg4, %arg3, %arg2, %317) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x2048x7x7xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %319 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %320 = "ttir.add"(%314, %318, %319) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    return %320 : tensor<1x2048x7x7xbf16>
    }
}
