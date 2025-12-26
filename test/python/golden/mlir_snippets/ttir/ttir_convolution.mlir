module {
  func.func @model(%arg0: tensor<12x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<12x64x112x112xf32> {
    %1 = "ttir.conv2d"(%arg0, %arg1) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32, batch_dim = 0 : i64, height_dim = 2 : i64, width_dim = 3 : i64, channel_dim = 1 : i64}> : (tensor<12x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<12x64x112x112xf32>
    return %1 : tensor<12x64x112x112xf32>
  }
}
