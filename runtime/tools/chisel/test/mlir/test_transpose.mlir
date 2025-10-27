module {
  func.func @transpose(%arg0: tensor<128x256xf32>) -> tensor<256x128xf32> {
    %0 = ttir.empty() : tensor<256x128xf32>
    %1 = "ttir.transpose"(%arg0, %0) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    return %1 : tensor<256x128xf32>
  }
}
