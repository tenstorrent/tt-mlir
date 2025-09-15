module attributes {} {
  func.func @test_add(%arg0: tensor<5x2x2x2xf32>, %arg1: tensor<5x2x2x2xf32>) -> tensor<5x2x2x2xf32> {
    %0 = ttir.empty() : tensor<5x2x2x2xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<5x2x2x2xf32>, tensor<5x2x2x2xf32>, tensor<5x2x2x2xf32>) -> tensor<5x2x2x2xf32>
    return %1 : tensor<5x2x2x2xf32>
  }
}
