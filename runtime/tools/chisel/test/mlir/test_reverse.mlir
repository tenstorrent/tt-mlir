module {
  func.func @reverse(%arg0: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %0 = ttir.empty() : tensor<128x256xf32>
    %1 = "ttir.reverse"(%arg0, %0) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
    return %1 : tensor<128x256xf32>
  }
}
