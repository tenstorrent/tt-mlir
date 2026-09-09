// RUN: ttmlir-opt -ttir-to-ttir-decomposition -ttir-implicit-broadcast-fold -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// ===----------------------------------------------------------------------===
// Test SharedLHSMatmulFusion bias handling for LinearOp
// ===----------------------------------------------------------------------===

// All three LinearOps have no bias — should fuse into a single LinearOp
// with concatenated weights and no bias.
module {
  func.func @shared_lhs_linear_no_bias(
      %input: tensor<32x512xbf16>,
      %w0: tensor<512x384xbf16>,
      %w1: tensor<512x384xbf16>,
      %w2: tensor<512x384xbf16>) -> (tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<32x384xbf16>) {
    // CHECK-LABEL: func.func @shared_lhs_linear_no_bias
    // Weights should be concatenated.
    // CHECK: "ttir.concat"
    // CHECK-SAME: (tensor<512x384xbf16>, tensor<512x384xbf16>, tensor<512x384xbf16>) -> tensor<512x1152xbf16>
    // Fused linear with no bias (only two operands before attributes).
    // CHECK: "ttir.linear"(%arg0, %{{[0-9]+}})
    // CHECK-SAME: -> tensor<32x1152xbf16>
    // Results sliced back out.
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    %0 = "ttir.linear"(%input, %w0) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
    %1 = "ttir.linear"(%input, %w1) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
    %2 = "ttir.linear"(%input, %w2) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
    return %0, %1, %2 : tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<32x384xbf16>
  }
}

// All three LinearOps have bias — should fuse into a single LinearOp
// with concatenated weights and concatenated bias.
module {
  func.func @shared_lhs_linear_all_bias(
      %input: tensor<32x512xbf16>,
      %w0: tensor<512x384xbf16>, %b0: tensor<384xbf16>,
      %w1: tensor<512x384xbf16>, %b1: tensor<384xbf16>,
      %w2: tensor<512x384xbf16>, %b2: tensor<384xbf16>) -> (tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<32x384xbf16>) {
    // CHECK-LABEL: func.func @shared_lhs_linear_all_bias
    // Weights should be concatenated.
    // CHECK: "ttir.concat"
    // CHECK-SAME: (tensor<512x384xbf16>, tensor<512x384xbf16>, tensor<512x384xbf16>) -> tensor<512x1152xbf16>
    // Biases should be concatenated.
    // CHECK: "ttir.concat"
    // CHECK-SAME: (tensor<384xbf16>, tensor<384xbf16>, tensor<384xbf16>) -> tensor<1152xbf16>
    // Fused linear with concatenated bias.
    // CHECK: "ttir.linear"(%arg0, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: -> tensor<32x1152xbf16>
    // Results sliced back out.
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    %0 = "ttir.linear"(%input, %w0, %b0) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>, tensor<384xbf16>) -> tensor<32x384xbf16>
    %1 = "ttir.linear"(%input, %w1, %b1) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>, tensor<384xbf16>) -> tensor<32x384xbf16>
    %2 = "ttir.linear"(%input, %w2, %b2) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>, tensor<384xbf16>) -> tensor<32x384xbf16>
    return %0, %1, %2 : tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<32x384xbf16>
  }
}

// Mixed bias: two LinearOps have bias, one does not — should NOT fuse.
module {
  func.func @shared_lhs_linear_mixed_bias(
      %input: tensor<32x512xbf16>,
      %w0: tensor<512x384xbf16>, %b0: tensor<384xbf16>,
      %w1: tensor<512x384xbf16>,
      %w2: tensor<512x384xbf16>, %b2: tensor<384xbf16>) -> (tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<32x384xbf16>) {
    // CHECK-LABEL: func.func @shared_lhs_linear_mixed_bias
    // Should NOT fuse — three separate linear ops remain.
    // CHECK: "ttir.linear"
    // CHECK: "ttir.linear"
    // CHECK: "ttir.linear"
    // CHECK-NOT: "ttir.concat"
    // CHECK-NOT: "ttir.slice_static"
    %0 = "ttir.linear"(%input, %w0, %b0) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>, tensor<384xbf16>) -> tensor<32x384xbf16>
    %1 = "ttir.linear"(%input, %w1) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
    %2 = "ttir.linear"(%input, %w2, %b2) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x384xbf16>, tensor<384xbf16>) -> tensor<32x384xbf16>
    return %0, %1, %2 : tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<32x384xbf16>
  }
}

// Oversized fusion: the combined output width exceeds kMaxFusedOutputDim
// (262144), so the three LinearOps must NOT be fused. Fusing them would build
// one enormous concatenated weight/bias; because the fused bias is rank-1 it
// forces ttnn.concat onto the row-major (stick) path, which overflows per-core
// L1. This guards against pathological many-way shared-LHS fusions such as
// CogVideoX's per-layer AdaLN modulation linears.
module {
  func.func @shared_lhs_linear_oversized(
      %input: tensor<2x512xf32>,
      %w0: tensor<512x131072xf32>, %b0: tensor<131072xf32>,
      %w1: tensor<512x131072xf32>, %b1: tensor<131072xf32>,
      %w2: tensor<512x131072xf32>, %b2: tensor<131072xf32>) -> (tensor<2x131072xf32>, tensor<2x131072xf32>, tensor<2x131072xf32>) {
    // CHECK-LABEL: func.func @shared_lhs_linear_oversized
    // Should NOT fuse — three separate linear ops remain.
    // CHECK: "ttir.linear"
    // CHECK: "ttir.linear"
    // CHECK: "ttir.linear"
    // CHECK-NOT: "ttir.concat"
    // CHECK-NOT: "ttir.slice_static"
    %0 = "ttir.linear"(%input, %w0, %b0) <{transpose_a = false, transpose_b = false}> : (tensor<2x512xf32>, tensor<512x131072xf32>, tensor<131072xf32>) -> tensor<2x131072xf32>
    %1 = "ttir.linear"(%input, %w1, %b1) <{transpose_a = false, transpose_b = false}> : (tensor<2x512xf32>, tensor<512x131072xf32>, tensor<131072xf32>) -> tensor<2x131072xf32>
    %2 = "ttir.linear"(%input, %w2, %b2) <{transpose_a = false, transpose_b = false}> : (tensor<2x512xf32>, tensor<512x131072xf32>, tensor<131072xf32>) -> tensor<2x131072xf32>
    return %0, %1, %2 : tensor<2x131072xf32>, tensor<2x131072xf32>, tensor<2x131072xf32>
  }
}
