module {
  func.func @repeat(%arg0: tensor<128xf32>) -> tensor<256xf32> {
    %0 = ttir.empty() : tensor<256xf32>
    %1 = "ttir.repeat"(%arg0, %0) : (tensor<128xf32>, tensor<256xf32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
}
