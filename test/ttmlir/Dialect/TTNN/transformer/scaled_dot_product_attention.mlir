// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @decode_sk_not_multiple_of_32(%query: tensor<1x8x1x64xbf16>, %key: tensor<1x8x16x64xbf16>, %value: tensor<1x8x16x64xbf16>) -> tensor<1x8x1x64xbf16> {
    // CHECK-LABEL: @decode_sk_not_multiple_of_32
    // CHECK: "ttnn.scaled_dot_product_attention"
    // CHECK-NOT: scaled_dot_product_attention_decode
    %1 = "ttir.scaled_dot_product_attention"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, is_causal = false, scale = 1.0 : f32}> : (tensor<1x8x1x64xbf16>, tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x1x64xbf16>
    return %1 : tensor<1x8x1x64xbf16>
  }
}
