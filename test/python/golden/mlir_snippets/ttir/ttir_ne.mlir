module attributes {} {
  func.func @not_equal(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xf32> {
    %1 = "ttir.ne"(%arg0, %arg1) : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xf32>
    return %1 : tensor<13x31xf32>
  }
}
