// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-ttnn-decomposition-pass=false enable-const-eval=false" %s | FileCheck %s

// A decode-shaped (Sq=1) ttir.scaled_dot_product_attention carrying an
// attention sink must lower to ttnn.scaled_dot_product_attention_decode WITHOUT
// dropping the sink. Regression: lowerToDecodeOp previously hardcoded the
// decode op's attention_sink operand to null.
module attributes {} {
  func.func @sdpa_decode_keeps_attention_sink(
      %q: tensor<1x8x1x64xf32>,
      %k: tensor<1x8x32x64xf32>,
      %v: tensor<1x8x32x64xf32>,
      %sink: tensor<1x8x1x1xf32>) -> tensor<1x8x1x64xf32> {
    // CHECK-LABEL: @sdpa_decode_keeps_attention_sink
    // The sink is a faithful post-scale logit; the tt-metal kernel applies scale
    // to it, so the lowering pre-divides by 1/scale (= 8.0 for scale 0.125) so
    // the kernel reproduces the original value (tt-metal issue 40470).
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 8.000000e+00
    // CHECK: "ttnn.multiply"
    // CHECK: ttnn.scaled_dot_product_attention_decode
    // The trailing attention_sink segment must be present (1), not dropped (0).
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1>
    %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v, %sink) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>, scale = 1.250000e-01 : f32}> : (tensor<1x8x1x64xf32>, tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>, tensor<1x8x1x1xf32>) -> tensor<1x8x1x64xf32>
    return %0 : tensor<1x8x1x64xf32>
  }
}
