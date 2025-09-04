// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true enable-fusing-conv2d-with-multiply-pattern=true" -o %t %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t

module {
func.func @resnet50_last_module(%arg0: tensor<1x2048x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<512x2048x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg4: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg5: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg6: tensor<512x512x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg7: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg8: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg9: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg10: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg11: tensor<2048x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg12: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg13: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg14: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg15: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg16: tensor<2048x1000xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg17: tensor<1000xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x1000xbf16> {
    %0 = "ttir.constant"() <{value = dense<2.04081633E-5> : tensor<1x2048xbf16>}> : () -> tensor<1x2048xbf16>
    %1 = "ttir.constant"() <{value = dense<0.0> : tensor<1x2048x7x7xbf16>}> : () -> tensor<1x2048x7x7xbf16>
    %2 = "ttir.constant"() <{value = dense<0.0> : tensor<1x512x7x7xbf16>}> : () -> tensor<1x512x7x7xbf16>
    %3 = ttir.empty() : tensor<1x2048x7x7xbf16>
    // CHECK: "ttnn.relu"
    %4 = "ttir.maximum"(%arg0, %1, %3) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %5 = ttir.empty() : tensor<1x512x7x7xbf16>
    // CHECK: "ttnn.conv2d"
    %6 = "ttir.convolution"(%4, %arg1, %5) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %7 = ttir.empty() : tensor<1x512x7x7xbf16>
    // CHECK-NOT: "ttnn.batch_norm"
    // CHECK-NOT: "ttnn.multiply"
    // CHECK-NOT: "ttnn.add"
    %8 = "ttir.batch_norm"(%6, %arg2, %arg3, %arg4, %arg5, %7) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x512x7x7xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %9 = ttir.empty() : tensor<1x512x7x7xbf16>
    %10 = "ttir.maximum"(%8, %2, %9) : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %11 = ttir.empty() : tensor<1x512x7x7xbf16>
    // CHECK: "ttnn.conv2d"
    %12 = "ttir.convolution"(%10, %arg6, %11) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<512x512x3x3xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %13 = ttir.empty() : tensor<1x512x7x7xbf16>
    %14 = "ttir.batch_norm"(%12, %arg7, %arg8, %arg9, %arg10, %13) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x512x7x7xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %15 = ttir.empty() : tensor<1x512x7x7xbf16>
    %16 = "ttir.maximum"(%14, %2, %15) : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %17 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %18 = "ttir.convolution"(%16, %arg11, %17) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %19 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %20 = "ttir.batch_norm"(%18, %arg12, %arg13, %arg14, %arg15, %19) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<1x2048x7x7xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %21 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %22 = "ttir.add"(%20, %4, %21) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %23 = ttir.empty() : tensor<1x2048x7x7xbf16>
    %24 = "ttir.maximum"(%22, %1, %23) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    %25 = ttir.empty() : tensor<1x2048xbf16>
    %26 = "ttir.sum"(%24, %25) <{dim_arg = [2 : i32, 3 : i32], keep_dim = false}> : (tensor<1x2048x7x7xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
    %27 = ttir.empty() : tensor<1x2048xbf16>
    %28 = "ttir.multiply"(%26, %0, %27) : (tensor<1x2048xbf16>, tensor<1x2048xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
    %29 = "ttir.dot_general"(%28, %arg16) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x2048xbf16>, tensor<2048x1000xbf16>) -> tensor<1x1000xbf16>
    %30 = ttir.empty() : tensor<1x1000xbf16>
    %31 = "ttir.reshape"(%arg17, %30) <{shape = [1 : i32, 1000 : i32]}> : (tensor<1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    %32 = ttir.empty() : tensor<1x1000xbf16>
    %33 = "ttir.add"(%29, %31, %32) : (tensor<1x1000xbf16>, tensor<1x1000xbf16>, tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
    return %33 : tensor<1x1000xbf16>
  }
}
