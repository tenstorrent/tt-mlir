module {
  func.func @repeat(%arg0: tensor<1x32x32xf32>) -> tensor<32x32x32xf32> {
    %1 = "ttir.repeat"(%arg0) {repeat_dimensions = array<i64: 32, 1, 1>} : (tensor<1x32x32xf32>) -> tensor<32x32x32xf32>
    return %1 : tensor<32x32x32xf32>
  }
}
