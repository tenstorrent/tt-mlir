module {
  func.func @reshape(%arg0: tensor<128x128xf32>) -> tensor<16384xf32> {
    %0 = ttir.empty() : tensor<16384xf32>
    %1 = "ttir.reshape"(%arg0, %0) {shape = [16384 : i32]} : (tensor<128x128xf32>, tensor<16384xf32>) -> tensor<16384xf32>
    return %1 : tensor<16384xf32>
  }
}
