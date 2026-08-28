module {
  func.func @round(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %1 = "ttir.round"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
