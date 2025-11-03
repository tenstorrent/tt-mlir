module {
  func.func @is_finite(%arg0: tensor<128x128xf32>) -> tensor<128x128xi1> {
    %0 = ttir.empty() : tensor<128x128xi1>
    %1 = "ttir.is_finite"(%arg0, %0) : (tensor<128x128xf32>, tensor<128x128xi1>) -> tensor<128x128xi1>
    return %1 : tensor<128x128xi1>
  }
}
