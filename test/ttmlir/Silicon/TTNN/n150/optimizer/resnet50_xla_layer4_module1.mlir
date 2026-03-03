// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true enable-fusing-conv2d-with-multiply-pattern=true" -o %t %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t

module {
func.func @resnet50_bottleneck_2048x1024(%arg0: tensor<1x1024x14x14xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg2: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg4: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg5: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg6: tensor<2048x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg223: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg224: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg225: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg226: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg227: tensor<2048x512x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg228: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg229: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg230: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg231: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg232: tensor<512x512x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg233: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg234: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg235: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg236: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg237: tensor<512x1024x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x2048x7x7xbf16> {
    // CHECK: func.func @resnet50_bottleneck_2048x1024
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x512x7x7xbf16>}> : () -> tensor<1x512x7x7xbf16>
    %1 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x512x14x14xbf16>}> : () -> tensor<1x512x14x14xbf16>
    %2 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x1024x14x14xbf16>}> : () -> tensor<1x1024x14x14xbf16>
    // CHECK: "ttnn.relu"
    %3 = "ttir.maximum"(%arg0, %2) : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    // CHECK: "ttnn.conv2d"
    %4 = "ttir.conv2d"(%3, %arg237) <{stride = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, height_dim = 2 : i64, width_dim = 3 : i64, channel_dim = 1 : i64}> : (tensor<1x1024x14x14xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<1x512x14x14xbf16>
    // CHECK-NOT: "ttnn.batch_norm_inference"
    // CHECK-NOT: "ttnn.multiply"
    // CHECK-NOT: "ttnn.add"
    %5 = "ttir.batch_norm_inference"(%4, %arg236, %arg235, %arg234, %arg233) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x512x14x14xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>) -> tensor<1x512x14x14xbf16>
    %6 = "ttir.maximum"(%5, %1) : (tensor<1x512x14x14xbf16>, tensor<1x512x14x14xbf16>) -> tensor<1x512x14x14xbf16>
    // CHECK: "ttnn.conv2d"
    %7 = "ttir.conv2d"(%6, %arg232) <{stride = array<i32: 2, 2>, padding = array<i32: 1, 1, 1, 1>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, height_dim = 2 : i64, width_dim = 3 : i64, channel_dim = 1 : i64}> : (tensor<1x512x14x14xbf16>, tensor<512x512x3x3xbf16>) -> tensor<1x512x7x7xbf16>
    %8 = "ttir.batch_norm_inference"(%7, %arg231, %arg230, %arg229, %arg228) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x512x7x7xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>, tensor<512xbf16>) -> tensor<1x512x7x7xbf16>
    %9 = "ttir.maximum"(%8, %0) : (tensor<1x512x7x7xbf16>, tensor<1x512x7x7xbf16>) -> tensor<1x512x7x7xbf16>
    %10 = "ttir.conv2d"(%9, %arg227) <{stride = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, height_dim = 2 : i64, width_dim = 3 : i64, channel_dim = 1 : i64}> : (tensor<1x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %11 = "ttir.batch_norm_inference"(%10, %arg226, %arg225, %arg224, %arg223) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x2048x7x7xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>) -> tensor<1x2048x7x7xbf16>
    %12 = "ttir.conv2d"(%3, %arg6) <{stride = array<i32: 2, 2>, padding = array<i32: 0, 0, 0, 0>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, height_dim = 2 : i64, width_dim = 3 : i64, channel_dim = 1 : i64}> : (tensor<1x1024x14x14xbf16>, tensor<2048x1024x1x1xbf16>) -> tensor<1x2048x7x7xbf16>
    %13 = "ttir.batch_norm_inference"(%12, %arg5, %arg4, %arg3, %arg2) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x2048x7x7xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>) -> tensor<1x2048x7x7xbf16>
    %14 = "ttir.add"(%11, %13) : (tensor<1x2048x7x7xbf16>, tensor<1x2048x7x7xbf16>) -> tensor<1x2048x7x7xbf16>
    return %14 : tensor<1x2048x7x7xbf16>
    }
}
