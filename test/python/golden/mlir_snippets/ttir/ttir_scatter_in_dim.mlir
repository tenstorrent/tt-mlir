module {
  func.func @model(%arg0: tensor<1x16x1x1xf32>, %arg1: tensor<1x4x1x1xi32>, %arg2: tensor<1x4x1x1xf32>) -> tensor<1x16x1x1xf32> {
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2) <{dim = 1 : i32, scatter_reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x16x1x1xf32>, tensor<1x4x1x1xi32>, tensor<1x4x1x1xf32>) -> tensor<1x16x1x1xf32>
    return %1 : tensor<1x16x1x1xf32>
  }
}
