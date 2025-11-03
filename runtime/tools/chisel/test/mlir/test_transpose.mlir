module {
  func.func @transpose(%arg0: tensor<128x256xf32>) -> tensor<256x128xf32> {
    %0 = ttir.empty() : tensor<256x128xf32>
    %1 = "ttir.transpose"(%arg0, %0) {dim0 = 0 : si32, dim1 = 1 : si32} : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    return %1 : tensor<256x128xf32>
  }
}
