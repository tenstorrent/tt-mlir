module {
  func.func @neg(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = ttir.empty() : tensor<128x128xf32>
    %1 = "ttir.neg"(%arg0, %0) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}
