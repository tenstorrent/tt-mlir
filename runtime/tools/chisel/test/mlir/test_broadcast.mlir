module {
  func.func @broadcast(%arg0: tensor<128xf32>) -> tensor<128x256xf32> {
    %0 = ttir.empty() : tensor<128x256xf32>
    %1 = "ttir.broadcast"(%arg0, %0) : (tensor<128xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
    return %1 : tensor<128x256xf32>
  }
}
