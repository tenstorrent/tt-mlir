module {
  func.func @model(%arg0: tensor<1x16x1x1xf32>, %arg1: tensor<1x4x1x1xf32>, %arg2: tensor<1x4x1x1xf32>) -> tensor<1x16x1x1xf32> {
    %0 = ttir.empty() : tensor<1x16x1x1xf32>
    %1 = "ttir.scatter_in_dim"(%arg0, %arg1, %arg2, %0) <{dim = 1 : i32}> : (tensor<1x16x1x1xf32>, tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    return %1 : tensor<1x16x1x1xf32>
  }
}
