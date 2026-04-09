module {
  func.func @paged_scaled_dot_product_attention_decode(%arg0: tensor<1x1x12x64xbf16>, %arg1: tensor<10x12x32x64xbf16>, %arg2: tensor<10x12x32x64xbf16>, %arg3: tensor<1x4xsi32>, %arg4: tensor<1x1x12x64xbf16>, %arg5: tensor<1xsi32>) -> tensor<1x1x12x64xbf16> {
    %0 = "ttir.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 0>}> : (tensor<1x1x12x64xbf16>, tensor<10x12x32x64xbf16>, tensor<10x12x32x64xbf16>, tensor<1x4xsi32>, tensor<1x1x12x64xbf16>, tensor<1xsi32>) -> tensor<1x1x12x64xbf16>
    return %0 : tensor<1x1x12x64xbf16>
  }
}
