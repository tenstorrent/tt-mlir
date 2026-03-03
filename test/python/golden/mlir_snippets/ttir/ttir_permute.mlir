module {
  func.func @permute(%arg0: tensor<1x4x32x64xf32>) -> tensor<4x32x64x1xf32> {
    %1 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 3, 0>}> : (tensor<1x4x32x64xf32>) -> tensor<4x32x64x1xf32>
    return %1 : tensor<4x32x64x1xf32>
  }
}
