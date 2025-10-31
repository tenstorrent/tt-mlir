module {
  func.func @bitwise_not(%arg0: tensor<128x128xi32>) -> tensor<128x128xi32> {
    %0 = ttir.empty() : tensor<128x128xi32>
    %1 = "ttir.bitwise_not"(%arg0, %0) : (tensor<128x128xi32>, tensor<128x128xi32>) -> tensor<128x128xi32>
    return %1 : tensor<128x128xi32>
  }
}
