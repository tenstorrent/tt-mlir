module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func @matmul_constrained_inputs(%arg0: tensor<2048x2048xf32>, %arg1: tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
    %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    return %0 : tensor<2048x2048xf32>
  }
}
