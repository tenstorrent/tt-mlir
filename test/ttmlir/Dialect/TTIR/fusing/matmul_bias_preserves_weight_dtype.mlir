// RUN: ttmlir-opt --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that the matmul + add -> linear fusion preserves the per-tensor
// weight dtype override. Without this, weights followed by a residual add
// (e.g. o_proj, down_proj) would fall back to the global weight dtype.

module {
  // CHECK-LABEL: func.func @matmul_bias_preserves_weight_dtype
  func.func @matmul_bias_preserves_weight_dtype(
    %arg0: tensor<32x512xbf16>,
    %w: tensor<512x256xbf16>,
    %bias: tensor<32x256xbf16>) -> tensor<32x256xbf16> {
    // CHECK: "ttir.linear"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf4"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.matmul"(%arg0, %w) <{transpose_a = false, transpose_b = false}> {ttcore.weight_dtype = "bfp_bf4"} : (tensor<32x512xbf16>, tensor<512x256xbf16>) -> tensor<32x256xbf16>
    %1 = "ttir.add"(%0, %bias) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
    return %1 : tensor<32x256xbf16>
  }
}
