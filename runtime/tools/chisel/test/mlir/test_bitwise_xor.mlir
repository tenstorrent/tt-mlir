module {
  func.func @bitwise_xor(%arg0: tensor<128x128xi32>, %arg1: tensor<128x128xi32>) -> tensor<128x128xi32> {
    %0 = ttir.empty() : tensor<128x128xi32>
    %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) : (tensor<128x128xi32>, tensor<128x128xi32>, tensor<128x128xi32>) -> tensor<128x128xi32>
    return %1 : tensor<128x128xi32>
  }
}
