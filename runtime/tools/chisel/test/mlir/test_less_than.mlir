module {
  func.func @less_than(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xi1> {
    %0 = ttir.empty() : tensor<128x128xi1>
    %1 = "ttir.less_than"(%arg0, %arg1, %0) : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xi1>) -> tensor<128x128xi1>
    return %1 : tensor<128x128xi1>
  }
}
