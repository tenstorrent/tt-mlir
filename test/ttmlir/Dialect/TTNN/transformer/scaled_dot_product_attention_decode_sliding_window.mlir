// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-ttnn-decomposition-pass=false" %s | FileCheck %s

// A decode-shaped (Sq=1) ttir.scaled_dot_product_attention carrying a
// sliding_window_size lowers to ttnn.scaled_dot_product_attention_decode with
// the window preserved on the native op (the tt-metal decode kernel applies it,
// anchored at the decode position). Decomposition is disabled here so the
// native op is observed; when decomposition runs, the window is instead baked
// into an explicit mask (see sdpa_decode_decomposition.mlir).
module attributes {} {
  func.func @sdpa_decode_sliding_window(
      %q: tensor<1x8x1x64xf32>,
      %k: tensor<1x8x32x64xf32>,
      %v: tensor<1x8x32x64xf32>) -> tensor<1x8x1x64xf32> {
    // CHECK-LABEL: @sdpa_decode_sliding_window
    // CHECK: "ttnn.scaled_dot_product_attention_decode"
    // CHECK-SAME: sliding_window_size = 4 : ui32
    // CHECK-NOT: "ttnn.softmax"
    %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, scale = 1.250000e-01 : f32, sliding_window_size = 4 : ui32}> : (tensor<1x8x1x64xf32>, tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x1x64xf32>
    return %0 : tensor<1x8x1x64xf32>
  }
}
