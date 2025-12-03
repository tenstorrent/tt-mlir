module {
  func.func @model(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
