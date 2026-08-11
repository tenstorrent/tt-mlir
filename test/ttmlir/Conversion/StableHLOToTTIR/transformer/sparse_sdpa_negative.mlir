// REQUIRES: stablehlo
// RUN: not ttmlir-opt --split-input-file --stablehlo-to-ttir-pipeline %s 2>&1 | FileCheck %s

// sparse_sdpa expects exactly 3 operands (q, kv, indices).
module {
  func.func public @sparse_sdpa_bad_operand_count(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: failed to legalize operation 'stablehlo.custom_call'
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%q, %kv) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "32"}} : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// v_dim is required.
module {
  func.func public @sparse_sdpa_missing_v_dim(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: failed to legalize operation 'stablehlo.custom_call'
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%q, %kv, %idx) {api_version = 0 : i32, mhlo.frontend_attributes = {k_chunk_size = "32"}} : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// v_dim must be a positive integer; a non-integer value is rejected (match
// failure -> legalization fails).
module {
  func.func public @sparse_sdpa_bad_v_dim(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: failed to legalize operation 'stablehlo.custom_call'
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%q, %kv, %idx) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "notanumber"}} : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}

// -----

// k_chunk_size must be a positive integer.
module {
  func.func public @sparse_sdpa_bad_k_chunk_size(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: error: failed to legalize operation 'stablehlo.custom_call'
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%q, %kv, %idx) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "32", k_chunk_size = "0"}} : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}
