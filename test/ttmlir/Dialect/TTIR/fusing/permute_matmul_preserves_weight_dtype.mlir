// RUN: ttmlir-opt --ttir-fusing="enable-permute-matmul-fusion=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that fusing a weight permute into the matmul transpose attribute
// preserves the per-tensor weight dtype override.

module {
  // CHECK-LABEL: func.func @permute_matmul_preserves_weight_dtype
  func.func @permute_matmul_preserves_weight_dtype(
    %arg0: tensor<32x512xbf16>,
    %w: tensor<256x512xbf16>) -> tensor<32x256xbf16> {
    %0 = "ttir.permute"(%w) <{permutation = array<i64: 1, 0>}> : (tensor<256x512xbf16>) -> tensor<512x256xbf16>
    // CHECK: "ttir.matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf4"
    // CHECK-NOT: "ttir.permute"
    %1 = "ttir.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> {ttcore.weight_dtype = "bfp_bf4"} : (tensor<32x512xbf16>, tensor<512x256xbf16>) -> tensor<32x256xbf16>
    return %1 : tensor<32x256xbf16>
  }
}
