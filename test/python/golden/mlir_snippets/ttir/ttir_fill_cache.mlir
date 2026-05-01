module {
  func.func @fill_cache(%arg0: tensor<1x32x256x64xf32>, %arg1: tensor<1x32x16x64xf32>) -> tensor<1x32x256x64xf32> {
    %0 = "ttir.fill_cache"(%arg0, %arg1) {batch_offset = 0 : i32} : (tensor<1x32x256x64xf32>, tensor<1x32x16x64xf32>) -> tensor<1x32x256x64xf32>
    return %0 : tensor<1x32x256x64xf32>
  }
}
