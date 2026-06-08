module {
  func.func @flash_mla_prefill(%arg0: tensor<1x16x32x128xbf16>, %arg1: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    %0 = "ttir.flash_mla_prefill"(%arg0, %arg1) <{head_dim_v = 64 : ui32, is_causal = true, operandSegmentSizes = array<i32: 1, 1, 0, 0>, scale = 0.0883883461 : f32}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}
