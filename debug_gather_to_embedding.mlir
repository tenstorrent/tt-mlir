func.func @test_gather_to_embedding(%input: tensor<100x128xf32>, %indices: tensor<4x1xi32>) -> tensor<4x128xf32> {
  %0 = tensor.empty() : tensor<4x128xf32>
  %result = ttir.gather %input, %indices, %0 : tensor<100x128xf32>, tensor<4x1xi32>, tensor<4x128xf32> -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}
