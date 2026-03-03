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
    %4 = "ttir.conv2d"(%3, %arg1) <{stride = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, height_dim = 2 : i64, width_dim = 3 : i64, channel_dim = 1 : i64}> : (tensor<1x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<1x512x7x7xbf16>
    // CHECK-NOT: "ttnn.batch_norm_inference"
    // CHECK-NOT: "ttnn.multiply"
    // CHECK-NOT: "ttnn.add"
    %5 = "ttir.batch_norm_inference"(%4, %arg2, %arg3, %arg4, %arg5) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x512x7x7xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>) -> tensor<1x512x7x7xbf16>
    %6 = "ttir.maximum"(%5, %2) : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    // CHECK: "ttnn.conv2d"
    %7 = "ttir.conv2d"(%6, %arg6) <{stride = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, height_dim = 2 : i64, width_dim = 3 : i64, channel_dim = 1 : i64}> : (tensor<1x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x7x7xbf16>
    %8 = "ttir.batch_norm_inference"(%7, %arg7, %arg8, %arg9, %arg10) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x512x7x7xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>) -> tensor<1x512x7x7xbf16>
    %9 = "ttir.maximum"(%8, %2) : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %10 = "ttir.conv2d"(%9, %arg11) <{stride = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, height_dim = 2 : i64, width_dim = 3 : i64, channel_dim = 1 : i64}> : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
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
