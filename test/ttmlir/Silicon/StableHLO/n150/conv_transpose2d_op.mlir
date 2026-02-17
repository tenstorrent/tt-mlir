// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s -o %t.mlir --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%"
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s
// UNSUPPORTED: true

module @test_conv_transpose {
  func.func @test_conv_transpose2d(%arg0: tensor<1x256x32x32xf32>, %arg1: tensor<256x128x2x2xf32>, %arg2: tensor<128xf32>) -> tensor<1x128x64x64xf32> {
    // CHECK-LABEL: @test_conv_transpose2d
    %0 = stablehlo.transpose %arg1, dims = [2, 3, 1, 0] : (tensor<256x128x2x2xf32>) -> tensor<2x2x128x256xf32>
    %1 = stablehlo.reverse %0, dims = [0, 1] : tensor<2x2x128x256xf32>
    // CHECK: %{{[0-9]+}} = "ttnn.permute"(%arg0)
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: tensor<1x256x32x32xf32,
    // CHECK-SAME: -> tensor<1x32x32x256xf32,
    // CHECK: %[[CONV_T:[0-9]+]] = "ttnn.conv_transpose2d"
    // CHECK-SAME: batch_size = 1 : i32,
    // CHECK-SAME: dilation = array<i32: 1, 1>,
    // CHECK-SAME: groups = 1 : i32,
    // CHECK-SAME: in_channels = 256 : i32,
    // CHECK-SAME: input_height = 32 : i32, input_width = 32 : i32,
    // CHECK-SAME: kernel_size = array<i32: 2, 2>,
    // CHECK-SAME: out_channels = 128 : i32,
    // CHECK-SAME: output_padding = array<i32: 0, 0>,
    // CHECK-SAME: padding = array<i32: 0, 0>,
    // CHECK-SAME: stride = array<i32: 2, 2>}>
    // CHECK-SAME: tensor<1x1x1024x256xf32,
    // CHECK-SAME: tensor<256x128x2x2xf32,
    // CHECK-SAME: -> tensor<1x1x4096x128xf32,
    %2 = stablehlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xf32>, tensor<2x2x128x256xf32>) -> tensor<1x128x64x64xf32>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"(%[[CONV_T]])
    // CHECK-SAME: shape = [1 : i32, 64 : i32, 64 : i32, 128 : i32]
    // CHECK-SAME: tensor<1x1x4096x128xf32,
    // CHECK-SAME: -> tensor<1x64x64x128xf32,
    // CHECK: %{{[0-9]+}} = "ttnn.permute"(%[[RESHAPE]])
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: tensor<1x64x64x128xf32,
    // CHECK-SAME: -> tensor<1x128x64x64xf32,
    %3 = stablehlo.reshape %arg2 : (tensor<128xf32>) -> tensor<128x1x1xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %5 = stablehlo.broadcast_in_dim %3, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %6 = stablehlo.add %4, %5 : tensor<1x128x64x64xf32>
    return %6 : tensor<1x128x64x64xf32>
  }

  func.func @test_conv_transpose2d_feature_group_count(%arg0: tensor<1x7x7x1088xf32>, %arg1: tensor<3x3x1088x64xf32>) -> tensor<1x14x14x1088xf32> {
    // CHECK-LABEL: @test_conv_transpose2d_feature_group_count
    // CHECK: %[[PERM1:[0-9]+]] = "ttnn.permute"(%arg1)
    // CHECK-SAME: permutation = array<i64: 3, 2, 0, 1>
    // CHECK-SAME: tensor<3x3x1088x64xf32,
    // CHECK-SAME: -> tensor<64x1088x3x3xf32,
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttnn.reshape"(%[[PERM1]])
    // CHECK-SAME: shape = [64 : i32, 17 : i32, 64 : i32, 3 : i32, 3 : i32]
    // CHECK-SAME: tensor<64x1088x3x3xf32,
    // CHECK-SAME: -> tensor<64x17x64x3x3xf32,
    // CHECK: %[[PERM2:[0-9]+]] = "ttnn.permute"(%[[RESHAPE1]])
    // CHECK-SAME: permutation = array<i64: 1, 0, 2, 3, 4>
    // CHECK-SAME: tensor<64x17x64x3x3xf32,
    // CHECK-SAME: -> tensor<17x64x64x3x3xf32,
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttnn.reshape"(%[[PERM2]])
    // CHECK-SAME: shape = [1088 : i32, 64 : i32, 3 : i32, 3 : i32]
    // CHECK-SAME: tensor<17x64x64x3x3xf32,
    // CHECK-SAME: -> tensor<1088x64x3x3xf32,
    // CHECK: %{{[0-9]+}} = "ttnn.reshape"(%arg0)
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 49 : i32, 1088 : i32]
    // CHECK-SAME: tensor<1x7x7x1088xf32,
    // CHECK-SAME: -> tensor<1x1x49x1088xf32,
    // CHECK: %[[CONV_T:[0-9]+]] = "ttnn.conv_transpose2d"
    // CHECK-SAME: batch_size = 1 : i32,
    // CHECK-SAME: dilation = array<i32: 1, 1>,
    // CHECK-SAME: groups = 17 : i32,
    // CHECK-SAME: in_channels = 1088 : i32,
    // CHECK-SAME: input_height = 7 : i32, input_width = 7 : i32,
    // CHECK-SAME: kernel_size = array<i32: 3, 3>,
    // CHECK-SAME: out_channels = 1088 : i32,
    // CHECK-SAME: output_padding = array<i32: 1, 1>,
    // CHECK-SAME: padding = array<i32: 1, 1>,
    // CHECK-SAME: stride = array<i32: 2, 2>}>
    // CHECK-SAME: tensor<1x1x49x1088xf32,
    // CHECK-SAME: tensor<1088x64x3x3xf32,
    // CHECK-SAME: -> tensor<1x1x196x1088xf32,
    // CHECK: %{{[0-9]+}} = "ttnn.reshape"(%[[CONV_T]])
    // CHECK-SAME: shape = [1 : i32, 14 : i32, 14 : i32, 1088 : i32]
    // CHECK-SAME: tensor<1x1x196x1088xf32,
    // CHECK-SAME: -> tensor<1x14x14x1088xf32,
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 17 : i64} : (tensor<1x7x7x1088xf32>, tensor<3x3x1088x64xf32>) -> tensor<1x14x14x1088xf32>
    return %0 : tensor<1x14x14x1088xf32>
}
}
