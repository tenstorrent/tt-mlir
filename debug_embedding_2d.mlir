func.func @test_embedding_2d(%indices: tensor<2x4xi32>, %weights: tensor<100x128xf32>) -> tensor<2x4x128xf32> {
  %0 = ttir.empty() : tensor<2x4x128xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<2x4xi32>, tensor<100x128xf32>, tensor<2x4x128xf32>) -> tensor<2x4x128xf32>
  return %result : tensor<2x4x128xf32>
}
