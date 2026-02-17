module {
  func.func @conv_transpose2d(%arg0: tensor<1x8x8x16xf32>, %arg1: tensor<16x32x3x3xf32>) -> tensor<1x10x10x32xf32> {
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1) <{stride = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, output_padding = array<i32: 0, 0>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, height_dim = 1 : i64, width_dim = 2 : i64, channel_dim = 3 : i64}> : (tensor<1x8x8x16xf32>, tensor<16x32x3x3xf32>) -> tensor<1x10x10x32xf32>
    return %1 : tensor<1x10x10x32xf32>
  }
}
