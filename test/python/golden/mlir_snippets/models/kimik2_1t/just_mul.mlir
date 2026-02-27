module {
  func.func @binary_op_fn(%arg0: tensor<64x4608xf32>, %arg1: tensor<64x4608xf32>) -> tensor<64x4608xf32> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<64x4608xf32>, tensor<64x4608xf32>) -> tensor<64x4608xf32>
    return %0 : tensor<64x4608xf32>
  }
}
