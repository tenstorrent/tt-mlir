// RUN: ttmlir-opt --ttir-fusing="ttnn-enable-conv2d-with-multiply-pattern=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
func.func @test_batch_norm_decomposition(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg4: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg5: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x64x112x112xbf16> {
    // CHECK: "ttir.add"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.conv2d"
    %1 = "ttir.conv2d"(%arg0, %arg1) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, height_dim = 2 : i64, width_dim = 3 : i64, channel_dim = 1 : i64}> : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>) -> tensor<1x64x112x112xbf16>
    %3 = "ttir.batch_norm_inference"(%1, %arg2, %arg3, %arg4, %arg5) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x64x112x112xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>) -> tensor<1x64x112x112xbf16>
    return %3 : tensor<1x64x112x112xbf16>
}
}
