module {
  func.func @concatenate_heads(%arg0: tensor<1x8x32x64xf32>) -> tensor<1x32x512xf32> {
    %0 = "ttir.concatenate_heads"(%arg0) : (tensor<1x8x32x64xf32>) -> tensor<1x32x512xf32>
    return %0 : tensor<1x32x512xf32>
  }
}
