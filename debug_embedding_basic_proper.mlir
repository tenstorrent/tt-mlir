func.func @test_embedding_basic(%indices: tensor<4xi32>, %weights: tensor<100x128xf32>) -> tensor<4x128xf32> {
  %0 = ttir.empty() : tensor<4x128xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<4xi32>, tensor<100x128xf32>, tensor<4x128xf32>) -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}
