module {
  func.func @clamp_scalar(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "ttir.clamp_scalar"(%arg0) <{min = 0.000000e+00 : f32, max = 6.000000e+00 : f32}> : (tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
