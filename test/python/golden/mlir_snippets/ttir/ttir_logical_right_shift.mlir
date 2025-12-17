module {
  func.func @logical_right_shift(%arg0: tensor<128x128xi32>, %arg1: tensor<128x128xi32>) -> tensor<128x128xi32> {
    %0 = "ttir.logical_right_shift"(%arg0, %arg1) : (tensor<128x128xi32>, tensor<128x128xi32>) -> tensor<128x128xi32>
    return %0 : tensor<128x128xi32>
  }
}
