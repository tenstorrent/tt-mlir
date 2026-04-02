// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-layout --ttnn-decompose-ops-on-validation-failure="force-decompose=true" %s | FileCheck %s

module {
  // Test 1: Basic MHA SDPA with mask and explicit scale
  func.func @sdpa_mha_with_mask(
    %query: tensor<1x8x64x64xbf16>,
    %key: tensor<1x8x64x64xbf16>,
    %value: tensor<1x8x64x64xbf16>,
    %mask: tensor<1x1x64x64xbf16>
  ) -> tensor<1x8x64x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_mha_with_mask
    // CHECK: "ttnn.transpose"
    // CHECK-SAME: dim0 = -2 : si32
    // CHECK-SAME: dim1 = -1 : si32
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 1.250000e-01
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.softmax"
    // CHECK-SAME: dimension = -1
    // CHECK: "ttnn.matmul"
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
         tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>)
      -> tensor<1x8x64x64xbf16>
    return %result : tensor<1x8x64x64xbf16>
  }

  // Test 2: GQA — 32 query heads, 8 KV heads (4:1 ratio)
  func.func @sdpa_gqa(
    %query: tensor<1x32x64x64xbf16>,
    %key: tensor<1x8x64x64xbf16>,
    %value: tensor<1x8x64x64xbf16>,
    %mask: tensor<1x1x64x64xbf16>
  ) -> tensor<1x32x64x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_gqa
    // CHECK: "ttnn.repeat_interleave"
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: repeats = 4 : ui32
    // CHECK: "ttnn.repeat_interleave"
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: repeats = 4 : ui32
    // CHECK: "ttnn.transpose"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.full"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.softmax"
    // CHECK: "ttnn.matmul"
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x64x64xbf16>, tensor<1x8x64x64xbf16>,
         tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>)
      -> tensor<1x32x64x64xbf16>
    return %result : tensor<1x32x64x64xbf16>
  }

  // Test 3: SDPA with attention_sink
  func.func @sdpa_with_attention_sink(
    %query: tensor<1x8x64x64xbf16>,
    %key: tensor<1x8x64x64xbf16>,
    %value: tensor<1x8x64x64xbf16>,
    %mask: tensor<1x1x64x64xbf16>,
    %sink: tensor<1x8x64x1xbf16>
  ) -> tensor<1x8x64x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_with_attention_sink
    // CHECK: "ttnn.transpose"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.concat"
    // CHECK: "ttnn.softmax"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.matmul"
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask, %sink) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
         tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>,
         tensor<1x8x64x1xbf16>)
      -> tensor<1x8x64x64xbf16>
    return %result : tensor<1x8x64x64xbf16>
  }
}
