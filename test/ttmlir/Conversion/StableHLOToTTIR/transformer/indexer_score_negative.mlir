// REQUIRES: stablehlo
// RUN: not ttmlir-opt --split-input-file --stablehlo-to-ttir-pipeline %s 2>&1 | FileCheck %s

// indexer_score expects exactly 3 operands (query, key, weights).
module {
  func.func public @indexer_score_bad_operand_count(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: failed to legalize operation 'stablehlo.custom_call'
    %0 = stablehlo.custom_call @tt.indexer_score(%q, %k) {api_version = 0 : i32} : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}

// -----

// chunk_start_idx must be a non-negative integer; a non-integer value is
// rejected (match failure -> legalization fails).
module {
  func.func public @indexer_score_bad_chunk_start_idx(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: failed to legalize operation 'stablehlo.custom_call'
    %0 = stablehlo.custom_call @tt.indexer_score(%q, %k, %w) {api_version = 0 : i32, mhlo.frontend_attributes = {chunk_start_idx = "notanumber"}} : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}
