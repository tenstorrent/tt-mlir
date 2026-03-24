module {
  func.func @update_cache(%arg0: tensor<2x16x1x64xf32>, %arg1: tensor<2x1x1x64xf32>, %arg2: tensor<1xi32>) -> tensor<2x16x1x64xf32> {
    %1 = "ttir.update_cache"(%arg0, %arg1, %arg2) {batch_offset = 0 : i32} : (tensor<2x16x1x64xf32>, tensor<2x1x1x64xf32>, tensor<1xi32>) -> tensor<2x16x1x64xf32>
    return %1 : tensor<2x16x1x64xf32>
  }
}
