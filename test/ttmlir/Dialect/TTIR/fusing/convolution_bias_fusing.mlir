// RUN: ttmlir-opt -ttir-fusing -ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
func.func @test_conv_transpose_bias_fusing(%arg0: tensor<1x1024x14x14xbf16>, %arg1: tensor<1024x512x2x2xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<2x2x512x1024xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x512x28x28xbf16> {
    // CHECK: func.func @test_conv_transpose_bias_fusing
    // CHECK: "ttir.conv_transpose2d"(%{{.*}}, %{{.*}}, %{{.*}}
    // CHECK-NOT: "ttir.add"

    %1 = "ttir.conv_transpose2d"(%arg0, %arg1) <{stride = array<i32: 2, 2>, padding = array<i32: 0, 0, 0, 0>, output_padding = array<i32: 0, 0>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x1024x14x14xbf16>, tensor<1024x512x2x2xbf16>) -> tensor<1x512x28x28xbf16>
    %3 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 512 : i32, 1 : i32, 1 : i32]}> : (tensor<512xbf16>) -> tensor<1x512x1x1xbf16>
    %5 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 28, 28>}> : (tensor<1x512x1x1xbf16>) -> tensor<1x512x28x28xbf16>
    %7 = "ttir.add"(%1, %5) : (tensor<1x512x28x28xbf16>, tensor<1x512x28x28xbf16>) -> tensor<1x512x28x28xbf16>
    return %7 : tensor<1x512x28x28xbf16>
}
}
// -----
module {
func.func @test_conv_bias_fusing(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2:  tensor<1x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x64x112x112xbf16> {
    // CHECK: func.func @test_conv_bias_fusing
    // CHECK: "ttir.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}
    // CHECK-NOT: "ttir.add"
    %1 = "ttir.conv2d"(%arg0, %arg1) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>) -> tensor<1x64x112x112xbf16>
    %3 = "ttir.add"(%1, %arg2) : (tensor<1x64x112x112xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x112x112xbf16>
    return %3 : tensor<1x64x112x112xbf16>
}
}
// -----
module {
func.func @test_conv_bias_fusing_with_broadcast(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2:  tensor<1x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x64x112x112xbf16> {
    // CHECK: func.func @test_conv_bias_fusing_with_broadcast
    // CHECK: "ttir.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}
    // CHECK-NOT: "ttir.add"
    %1 = "ttir.conv2d"(%arg0, %arg1) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>) -> tensor<1x64x112x112xbf16>
    %3 = "ttir.broadcast"(%arg2) <{broadcast_dimensions = array<i64: 1, 1, 112, 112>}> : (tensor<1x64x1x1xbf16>) -> tensor<1x64x112x112xbf16>
    %4 = "ttir.add"(%1, %3) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    return %4 : tensor<1x64x112x112xbf16>
}
}
// -----
module {
func.func @test_conv_double_add_bias_fusing(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<64x3x7x7xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<1x64x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x64x112x112xbf16> {
    // CHECK: func.func @test_conv_double_add_bias_fusing
    // CHECK: "ttir.add"
    // CHECK: "ttir.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}
    // CHECK-NOT: "ttir.add"
    %1 = "ttir.conv2d"(%arg0, %arg1) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xbf16>, tensor<64x3x7x7xbf16>) -> tensor<1x64x112x112xbf16>
    %3 = "ttir.add"(%arg2, %arg3) : (tensor<1x64x1x1xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x1x1xbf16>
    %5 = "ttir.add"(%1, %3) : (tensor<1x64x112x112xbf16>, tensor<1x64x1x1xbf16>) -> tensor<1x64x112x112xbf16>
    return %5 : tensor<1x64x112x112xbf16>
}
}
// -----
module {
func.func @test_conv3d_bias_fusing(%arg0: tensor<1x8x28x28x32xbf16>, %arg1: tensor<32x32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x1x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x6x26x26x32xbf16> {
    // CHECK: func.func @test_conv3d_bias_fusing
    // CHECK: "ttir.conv3d"(%{{.*}}, %{{.*}}, %{{.*}}
    // CHECK-NOT: "ttir.add"
    %1 = "ttir.conv3d"(%arg0, %arg1)
            <{
              stride = array<i32: 1, 1, 1>,
              padding = array<i32: 0, 0, 0>,
              groups = 1 : i32,
              padding_mode = "zeros"
            }> : (tensor<1x8x28x28x32xbf16>, tensor<32x32x3x3x3xbf16>) -> tensor<1x6x26x26x32xbf16>
    %2 = "ttir.add"(%1, %arg2) : (tensor<1x6x26x26x32xbf16>, tensor<1x1x1x1x32xbf16>) -> tensor<1x6x26x26x32xbf16>
    return %2 : tensor<1x6x26x26x32xbf16>
}
}
