module {
  func.func public @test_clamp_tensor(%arg0: tensor<32x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %1 = "ttir.clamp_tensor"(%arg0, %arg1, %arg2) : (tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }
}
