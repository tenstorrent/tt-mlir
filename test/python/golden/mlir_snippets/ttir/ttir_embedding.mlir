module {
  func.func @embedding_module(%arg0: tensor<2xi32>, %arg1: tensor<4x3xf32>) -> tensor<2x3xf32> {
    %0 = "ttir.embedding"(%arg0, %arg1) : (tensor<2xi32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
