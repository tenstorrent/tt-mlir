module {
  func.func @test_fill_cache_parse_split(%arg0: tensor<2x16x64xf32>, %arg1: tensor<2x16x64xf32>) -> tensor<2x16x64xf32> {
    %1 = "ttir.fill_cache"(%arg0, %arg1) {batch_offset = 0 : i32} : (tensor<2x16x64xf32>, tensor<2x16x64xf32>) -> tensor<2x16x64xf32>
    return %1 : tensor<2x16x64xf32>
  }
}