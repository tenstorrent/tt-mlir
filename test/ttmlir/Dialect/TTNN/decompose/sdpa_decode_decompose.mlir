// RUN: ttmlir-opt --ttnn-decompose-ops-on-validation-failure="force-decompose=true" %s | FileCheck %s

module {
  // Test 1: Basic SDPADecode MHA decomposition -> SDPA with permutes
  // Q: [1, B, H, D] -> permute to [B, H, 1, D] -> SDPA -> permute back
  func.func @sdpa_decode_mha(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>,
    %mask: tensor<32x1x1x128xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_mha
    // CHECK: %[[PERMUTED_Q:.*]] = "ttnn.permute"(%arg0)
    // CHECK-SAME: permutation = array<i64: 1, 2, 0, 3>
    // CHECK: %[[SDPA:.*]] = "ttnn.scaled_dot_product_attention"(%[[PERMUTED_Q]], %arg1, %arg2, %arg3)
    // CHECK-SAME: is_causal = false
    // CHECK-SAME: scale = 1.250000e-01
    // CHECK: %[[RESULT:.*]] = "ttnn.permute"(%[[SDPA]])
    // CHECK-SAME: permutation = array<i64: 2, 0, 1, 3>
    // CHECK: return %[[RESULT]]
    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>, tensor<32x1x1x128xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }

  // Test 2: SDPADecode without mask (causal)
  func.func @sdpa_decode_causal(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_causal
    // CHECK: %[[PERMUTED_Q:.*]] = "ttnn.permute"(%arg0)
    // CHECK-SAME: permutation = array<i64: 1, 2, 0, 3>
    // Decode causal is decomposed with is_causal=false since Sq=1 makes causal a no-op
    // CHECK: %[[SDPA:.*]] = "ttnn.scaled_dot_product_attention"(%[[PERMUTED_Q]], %arg1, %arg2)
    // CHECK-SAME: is_causal = false
    // CHECK: %[[RESULT:.*]] = "ttnn.permute"(%[[SDPA]])
    // CHECK-SAME: permutation = array<i64: 2, 0, 1, 3>
    // CHECK: return %[[RESULT]]
    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
      is_causal = true,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }

  // Test 3: SDPADecode with attention_sink
  func.func @sdpa_decode_with_attention_sink(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>,
    %mask: tensor<32x1x1x128xbf16>,
    %sink: tensor<1x32x1x1xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_with_attention_sink
    // CHECK: %[[PERMUTED_Q:.*]] = "ttnn.permute"(%arg0)
    // CHECK: "ttnn.scaled_dot_product_attention"(%[[PERMUTED_Q]], %arg1, %arg2, %arg3, %arg4)
    // CHECK: "ttnn.permute"
    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %mask, %sink) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>, tensor<32x1x1x128xbf16>,
         tensor<1x32x1x1xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }
}
