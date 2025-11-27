module {
  func.func @model(%arg0: tensor<32x32xf32>) -> tensor<32x34xf32> {
    %0 = ttir.empty() : tensor<32x34xf32>
    %1 = "ttir.pad"(%arg0, %0) <{padding = array<i32: 0, 0, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<32x32xf32>, tensor<32x34xf32>) -> tensor<32x34xf32>
    return %1 : tensor<32x34xf32>
  }
}
