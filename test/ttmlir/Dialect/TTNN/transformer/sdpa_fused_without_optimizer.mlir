// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s | FileCheck %s

// Regression for tt-xla#6081: a runnable scaled_dot_product_attention must stay
// fused through TTIR->TTNN even when the optimizer is OFF. The decomposition pass
// validates against the op model (op-constraints enabled independent of the
// optimizer), so an SDPA tt-metal can run is kept instead of being decomposed to
// softmax + matmul. Before the fix, the no-optimizer pipeline unconditionally
// decomposed every SDPA.
module {
  func.func @sdpa_kept(%query: tensor<8x12x32x32xbf16>, %key: tensor<8x3x32x32xbf16>, %value: tensor<8x3x32x32xbf16>) -> tensor<8x12x32x32xbf16> {
    // CHECK: "ttnn.scaled_dot_product_attention"
    // CHECK-NOT: "ttnn.softmax"
    %0 = "ttir.scaled_dot_product_attention"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, is_causal = true, scale = 1.0 : f32}> : (tensor<8x12x32x32xbf16>, tensor<8x3x32x32xbf16>, tensor<8x3x32x32xbf16>) -> tensor<8x12x32x32xbf16>
    return %0 : tensor<8x12x32x32xbf16>
  }
}
