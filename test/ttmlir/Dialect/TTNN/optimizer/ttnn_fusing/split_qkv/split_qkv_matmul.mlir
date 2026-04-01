// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true" %s | FileCheck %s

// Test SplitQueryKeyValueAndSplitHeadsFusing for matmul (no bias) and linear (with bias).
// Three matmuls/linears sharing the same LHS, each followed by slice -> reshape -> permute,
// should fuse into ttnn.split_query_key_value_and_split_heads.

module {
  // CHECK-LABEL: func.func @split_qkv_matmul_mha
  // CHECK: "ttnn.split_query_key_value_and_split_heads"
  func.func @split_qkv_matmul_mha(
      %input: tensor<1x32x512xbf16>,
      %wq: tensor<512x512xbf16>,
      %wk: tensor<512x512xbf16>,
      %wv: tensor<512x512xbf16>) -> (tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>) {

    %0 = "ttir.reshape"(%input) <{shape = [32 : i32, 512 : i32]}> : (tensor<1x32x512xbf16>) -> tensor<32x512xbf16>

    // Q projection
    %q0 = "ttir.matmul"(%0, %wq) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %q1 = "ttir.reshape"(%q0) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %q2 = "ttir.permute"(%q1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    // K projection
    %k0 = "ttir.matmul"(%0, %wk) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %k1 = "ttir.reshape"(%k0) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %k2 = "ttir.permute"(%k1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    // V projection
    %v0 = "ttir.matmul"(%0, %wv) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %v1 = "ttir.reshape"(%v0) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %v2 = "ttir.permute"(%v1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    return %q2, %k2, %v2 : tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>
  }

  // CHECK-LABEL: func.func @split_qkv_linear_mha
  // CHECK: "ttnn.split_query_key_value_and_split_heads"
  func.func @split_qkv_linear_mha(
      %input: tensor<1x32x512xbf16>,
      %wq: tensor<512x512xbf16>, %bq: tensor<512xbf16>,
      %wk: tensor<512x512xbf16>, %bk: tensor<512xbf16>,
      %wv: tensor<512x512xbf16>, %bv: tensor<512xbf16>) -> (tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>) {

    %0 = "ttir.reshape"(%input) <{shape = [32 : i32, 512 : i32]}> : (tensor<1x32x512xbf16>) -> tensor<32x512xbf16>

    // Q projection with bias
    %q0 = "ttir.matmul"(%0, %wq) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %q0r = "ttir.reshape"(%q0) <{shape = [1 : i32, 32 : i32, 512 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x512xbf16>
    %q0b = "ttir.add"(%q0r, %bq) : (tensor<1x32x512xbf16>, tensor<512xbf16>) -> tensor<1x32x512xbf16>
    %q1 = "ttir.reshape"(%q0b) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<1x32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %q2 = "ttir.permute"(%q1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    // K projection with bias
    %k0 = "ttir.matmul"(%0, %wk) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %k0r = "ttir.reshape"(%k0) <{shape = [1 : i32, 32 : i32, 512 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x512xbf16>
    %k0b = "ttir.add"(%k0r, %bk) : (tensor<1x32x512xbf16>, tensor<512xbf16>) -> tensor<1x32x512xbf16>
    %k1 = "ttir.reshape"(%k0b) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<1x32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %k2 = "ttir.permute"(%k1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    // V projection with bias
    %v0 = "ttir.matmul"(%0, %wv) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %v0r = "ttir.reshape"(%v0) <{shape = [1 : i32, 32 : i32, 512 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x512xbf16>
    %v0b = "ttir.add"(%v0r, %bv) : (tensor<1x32x512xbf16>, tensor<512xbf16>) -> tensor<1x32x512xbf16>
    %v1 = "ttir.reshape"(%v0b) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<1x32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %v2 = "ttir.permute"(%v1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    return %q2, %k2, %v2 : tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>
  }

  // CHECK-LABEL: func.func @split_qkv_matmul_gqa
  // CHECK: "ttnn.split_query_key_value_and_split_heads"
  func.func @split_qkv_matmul_gqa(
      %input: tensor<1x32x512xbf16>,
      %wq: tensor<512x512xbf16>,
      %wk: tensor<128x512xbf16>,
      %wv: tensor<128x512xbf16>) -> (tensor<1x8x32x64xbf16>, tensor<1x2x32x64xbf16>, tensor<1x2x32x64xbf16>) {

    %0 = "ttir.reshape"(%input) <{shape = [32 : i32, 512 : i32]}> : (tensor<1x32x512xbf16>) -> tensor<32x512xbf16>

    // Q projection: 8 heads
    %q0 = "ttir.matmul"(%0, %wq) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %q1 = "ttir.reshape"(%q0) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %q2 = "ttir.permute"(%q1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    // K projection: 2 heads
    %k0 = "ttir.matmul"(%0, %wk) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<128x512xbf16>) -> tensor<32x128xbf16>
    %k1 = "ttir.reshape"(%k0) <{shape = [1 : i32, 32 : i32, 2 : i32, 64 : i32]}> : (tensor<32x128xbf16>) -> tensor<1x32x2x64xbf16>
    %k2 = "ttir.permute"(%k1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x2x64xbf16>) -> tensor<1x2x32x64xbf16>

    // V projection: 2 heads
    %v0 = "ttir.matmul"(%0, %wv) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<128x512xbf16>) -> tensor<32x128xbf16>
    %v1 = "ttir.reshape"(%v0) <{shape = [1 : i32, 32 : i32, 2 : i32, 64 : i32]}> : (tensor<32x128xbf16>) -> tensor<1x32x2x64xbf16>
    %v2 = "ttir.permute"(%v1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x2x64xbf16>) -> tensor<1x2x32x64xbf16>

    return %q2, %k2, %v2 : tensor<1x8x32x64xbf16>, tensor<1x2x32x64xbf16>, tensor<1x2x32x64xbf16>
  }
}
