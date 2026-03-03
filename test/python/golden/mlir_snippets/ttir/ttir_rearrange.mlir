module {
  func.func @rearrange(%arg0: tensor<3x32x32xf32>) -> tensor<96x32xf32> {
    %1 = "ttir.rearrange"(%arg0) <{pattern = "z y x -> (y z) x"}> : (tensor<3x32x32xf32>) -> tensor<96x32xf32>
    return %1 : tensor<96x32xf32>
  }
}
