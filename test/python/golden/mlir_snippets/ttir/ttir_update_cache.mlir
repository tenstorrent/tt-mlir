// Note: Cache dim 2 (sequence length) must be >= 256 because the test
// infrastructure generates random indices in [0, 256) via torch.randint.
// Constraints: input dim 0 must be 1, and dims 1/3 must match cache dims 1/3.
module {
  func.func @update_cache(%arg0: tensor<1x16x256x64xf32>, %arg1: tensor<1x16x1x64xf32>, %arg2: tensor<1xi32>) -> tensor<1x16x256x64xf32> {
    %1 = "ttir.update_cache"(%arg0, %arg1, %arg2) {batch_offset = 0 : i32} : (tensor<1x16x256x64xf32>, tensor<1x16x1x64xf32>, tensor<1xi32>) -> tensor<1x16x256x64xf32>
    return %1 : tensor<1x16x256x64xf32>
  }
}
