module {
  func.func @unsqueeze(%arg0: tensor<128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = ttir.empty() : tensor<1x128x128xf32>
    %1 = "ttir.unsqueeze"(%arg0, %0) : (tensor<128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    return %1 : tensor<1x128x128xf32>
  }
}
