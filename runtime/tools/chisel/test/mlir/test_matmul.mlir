module {
  func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x128xf32>) -> tensor<128x128xf32> {
    %0 = ttir.empty() : tensor<128x128xf32>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<128x256xf32>, tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}
