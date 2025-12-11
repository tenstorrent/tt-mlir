module {
  func.func @cumsum(%arg0: tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32> {
    %0 = "ttir.cumsum"(%arg0) <{dim = 1 : i64}> : (tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32>
    return %0 : tensor<4x4x128x128xf32>
  }
}
