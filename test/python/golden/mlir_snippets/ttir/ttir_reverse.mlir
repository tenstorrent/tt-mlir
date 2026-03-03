module {
  func.func @model(%arg0: tensor<32x784x1x64xf32>) -> tensor<32x784x1x64xf32> {
    %1 = "ttir.reverse"(%arg0) <{dimensions = array<i64: 1>}> : (tensor<32x784x1x64xf32>) -> tensor<32x784x1x64xf32>
    return %1 : tensor<32x784x1x64xf32>
  }
}
