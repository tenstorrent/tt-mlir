// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-layout --ttnn-decompose-ops-on-validation-failure="force-decompose=true" %s | FileCheck %s

module {
  // Test 1: SDPADecode MHA — full cascade to component ops via SDPA
  // Q: [1, B, H, D] -> permute -> SDPA decomposed -> permute back
  func.func @sdpa_decode_mha(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>,
    %mask: tensor<32x1x1x128xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_mha
    // No SDPA ops should remain — fully decomposed.
    // CHECK-NOT: ttnn.scaled_dot_product_attention_decode
    // CHECK-NOT: ttnn.scaled_dot_product_attention
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 1, 2, 0, 3>
    // CHECK: "ttnn.transpose"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.full"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.softmax"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 2, 0, 1, 3>
    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>, tensor<32x1x1x128xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }

  // Test 2: SDPADecode causal — is_causal becomes false in decomposed form
  func.func @sdpa_decode_causal(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_causal
    // CHECK-NOT: ttnn.scaled_dot_product_attention_decode
    // CHECK-NOT: ttnn.scaled_dot_product_attention
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.transpose"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.softmax"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.permute"
    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
      is_causal = true,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }
}
