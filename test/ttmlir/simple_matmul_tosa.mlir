module attributes {torch.debug_module_name = "_lambda"} {
  func.func @forward(%arg0: tensor<1x128x64xf32>, %arg1: tensor<1x64x128xf32>) -> tensor<1x128x128xf32> {
    %0 = tosa.matmul %arg0, %arg1 : (tensor<1x128x64xf32>, tensor<1x64x128xf32>) -> tensor<1x128x128xf32>
    return %0 : tensor<1x128x128xf32>
  }
}

