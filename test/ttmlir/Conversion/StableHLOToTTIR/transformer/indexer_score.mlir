// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @indexer_score attributes {} {
  // Default chunk_start_idx (no frontend attributes).
  func.func public @indexer_score(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK-LABEL: @indexer_score
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: chunk_start_idx = 0 : ui32
    // CHECK-SAME: composite_name = "indexer_score"
    // CHECK-SAME: decomposition = @indexer_score_decomp
    %0 = stablehlo.custom_call @tt.indexer_score(%q, %k, %w) {api_version = 0 : i32} : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }

  // Explicit chunk_start_idx parsed from mhlo.frontend_attributes.
  func.func public @indexer_score_chunked(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x64x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16> {
    // CHECK-LABEL: @indexer_score_chunked
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: chunk_start_idx = 32 : ui32
    // CHECK-SAME: composite_name = "indexer_score"
    %0 = stablehlo.custom_call @tt.indexer_score(%q, %k, %w) {api_version = 0 : i32, mhlo.frontend_attributes = {chunk_start_idx = "32"}} : (tensor<1x8x32x128xbf16>, tensor<1x1x64x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }

  // The synthesized decomposition holds the full primitive lowering: QK^T (with
  // head-folding reshape), relu, per-head gate multiply, head-sum reduction, and
  // the causal additive -inf mask.
  // CHECK: func.func private @indexer_score_decomp
  // CHECK: "ttir.matmul"
  // CHECK: "ttir.relu"
  // CHECK: "ttir.sum"
  // CHECK: "ttir.where"
}
