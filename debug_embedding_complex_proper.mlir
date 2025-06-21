func.func @test_embedding_complex(%indices: tensor<8xi32>, %weights: tensor<1000x256xf32>) -> tensor<8x256xf32> {
  %0 = ttir.empty() : tensor<8x256xf32>
  %result = "ttir.embedding"(%indices, %weights, %0) : (tensor<8xi32>, tensor<1000x256xf32>, tensor<8x256xf32>) -> tensor<8x256xf32>
  return %result : tensor<8x256xf32>
}
