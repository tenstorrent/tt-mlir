// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true enable-fusing-conv2d-with-multiply-pattern=true" -o %t %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t

module {
func.func @resnet50_last_module(%arg0: tensor<1x2048x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<512x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg4: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg5: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg6: tensor<512x512x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg7: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg8: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg9: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg10: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg11: tensor<2048x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg12: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg13: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg14: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg15: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg16: tensor<2048x1000xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg17: tensor<1000xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x1000xbf16> {
    %0 = "ttir.constant"() <{value = dense<2.04081633E-5> : tensor<1x2048xbf16>}> : () -> tensor<1x2048xbf16>
    %1 = "ttir.constant"() <{value = dense<0.0> : tensor<1x2048x7x7xbf16>}> : () -> tensor<1x2048x7x7xbf16>
    %2 = "ttir.constant"() <{value = dense<0.0> : tensor<1x512x7x7xbf16>}> : () -> tensor<1x512x7x7xbf16>
    // CHECK: "ttnn.relu"
    %3 = "ttir.maximum"(%arg0, %1) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    // CHECK: "ttnn.conv2d"
    %4 = "ttir.convolution"(%3, %arg1) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<1x512x7x7xbf16>
    // CHECK-NOT: "ttnn.batch_norm_inference"
    // CHECK-NOT: "ttnn.multiply"
    // CHECK-NOT: "ttnn.add"
    %5 = "ttir.batch_norm_inference"(%4, %arg2, %arg3, %arg4, %arg5) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x512x7x7xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>) -> tensor<1x512x7x7xbf16>
    %6 = "ttir.maximum"(%5, %2) : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    // CHECK: "ttnn.conv2d"
    %7 = "ttir.convolution"(%6, %arg6) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x7x7xbf16>
    %8 = "ttir.batch_norm_inference"(%7, %arg7, %arg8, %arg9, %arg10) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x512x7x7xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>) -> tensor<1x512x7x7xbf16>
    %9 = "ttir.maximum"(%8, %2) : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %10 = "ttir.convolution"(%9, %arg11) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %11 = "ttir.batch_norm_inference"(%10, %arg12, %arg13, %arg14, %arg15) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x2048x7x7xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>) -> tensor<1x2048x7x7xbf16>
    %12 = "ttir.add"(%11, %3) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %13 = "ttir.maximum"(%12, %1) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %14 = "ttir.sum"(%13) <{dim_arg = [2 : i32, 3 : i32], keep_dim = false}> : (tensor<1x2048x7x7xbf16>) -> tensor<1x2048xbf16>
    %15 = "ttir.multiply"(%14, %0) : (tensor<1x2048xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
    %16 = "ttir.dot_general"(%15, %arg16) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x2048xbf16>, tensor<2048x1000xbf16>) -> tensor<1x1000xbf16>
    %17 = "ttir.reshape"(%arg17) <{shape = [1 : i32, 1000 : i32]}> : (tensor<1000xbf16>) -> tensor<1x1000xbf16>
    %18 = "ttir.add"(%16, %17) : (tensor<1x1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    return %18 : tensor<1x1000xbf16>
  }
}
