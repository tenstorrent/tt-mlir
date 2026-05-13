module {
  func.func @paged_fill_cache(%arg0: tensor<10x12x32x64xbf16>, %arg1: tensor<1x12x128x64xbf16>, %arg2: tensor<1x4xsi32>, %arg3: tensor<1xsi32>) -> tensor<10x12x32x64xbf16> {
    %0 = "ttir.paged_fill_cache"(%arg0, %arg1, %arg2, %arg3) : (tensor<10x12x32x64xbf16>, tensor<1x12x128x64xbf16>, tensor<1x4xsi32>, tensor<1xsi32>) -> tensor<10x12x32x64xbf16>
    return %0 : tensor<10x12x32x64xbf16>
  }
}
