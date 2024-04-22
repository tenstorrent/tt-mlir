module attributes {torch.debug_module_name = "_lambda"} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}

