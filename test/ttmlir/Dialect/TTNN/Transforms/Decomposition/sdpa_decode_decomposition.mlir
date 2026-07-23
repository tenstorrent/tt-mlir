// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-layout --ttnn-decomposition %s | FileCheck %s

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

  // Test 3: SDPADecode with a sliding window (no cur_pos) — decomposed, with the
  // window baked into a static mask. With no cur_pos the kernel anchors the
  // window at the last kv position, keeping keys [Sk-W, Sk-1]; the prefill op we
  // lower to can't place that window (it anchors at query row 0), so the
  // decomposition synthesizes the mask via arange + compare + where and adds it
  // to the scores before softmax.
  func.func @sdpa_decode_sliding_window(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_sliding_window
    // CHECK-NOT: "ttnn.scaled_dot_product_attention_decode"
    // window-mask synthesis: arange over Sk, threshold compare, select 0/-inf.
    // CHECK: "ttnn.arange"
    // CHECK: "ttnn.gt"
    // CHECK: "ttnn.where"
    // mask added to scores, then softmax over the kv axis.
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.softmax"
    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
      is_causal = true,
      scale = 0.125 : f32,
      sliding_window_size = 64 : ui32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }

  // Test 4: SDPADecode with a sliding window AND a per-batch cur_pos_tensor
  // (is_causal => the kernel reads cur_pos). Decomposed with a runtime window
  // mask anchored at cur_pos[b]: keep keys (cur_pos-W, cur_pos], synthesized as
  // cur_pos (cast) compared against arange for both the causal cutoff and the
  // window lower bound.
  func.func @sdpa_decode_sliding_window_cur_pos(
    %query: tensor<1x4x32x64xbf16>,
    %key: tensor<4x32x128x64xbf16>,
    %value: tensor<4x32x128x64xbf16>,
    %cur_pos: tensor<4xi32>
  ) -> tensor<1x4x32x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_sliding_window_cur_pos
    // CHECK-NOT: "ttnn.scaled_dot_product_attention_decode"
    // window-mask synthesis with cur_pos cast as the per-batch anchor
    // (the typecast distinguishes the runtime form from the static Sk-1 form).
    // CHECK: "ttnn.arange"
    // CHECK: "ttnn.typecast"
    // CHECK: "ttnn.gt"
    // CHECK: "ttnn.where"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.softmax"
    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %cur_pos) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
      is_causal = true,
      scale = 0.125 : f32,
      sliding_window_size = 64 : ui32
    }> : (tensor<1x4x32x64xbf16>, tensor<4x32x128x64xbf16>,
         tensor<4x32x128x64xbf16>, tensor<4xi32>)
      -> tensor<1x4x32x64xbf16>
    return %result : tensor<1x4x32x64xbf16>
  }
}
