module attributes {} {
  func.func @test_simple(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = ttir.empty() : tensor<2xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}
