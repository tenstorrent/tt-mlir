module {
  func.func @scaled_dot_product_attention(%arg0: tensor<1x12x128x64xbf16>, %arg1: tensor<1x12x128x64xbf16>, %arg2: tensor<1x12x128x64xbf16>) -> tensor<1x12x128x64xbf16> {
    %0 = "ttir.scaled_dot_product_attention"(%arg0, %arg1, %arg2) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x12x128x64xbf16>, tensor<1x12x128x64xbf16>, tensor<1x12x128x64xbf16>) -> tensor<1x12x128x64xbf16>
    return %0 : tensor<1x12x128x64xbf16>
  }
}
