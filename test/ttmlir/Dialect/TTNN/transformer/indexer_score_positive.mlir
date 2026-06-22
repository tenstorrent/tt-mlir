// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole composite-resolution=force-promote" %s | FileCheck %s

// Resolves a ttcore.composite "indexer_score" through the TTNN backend pipeline
// (TTNNResolveComposites) on a Blackhole target and verifies it is promoted to
// the typed ttnn.indexer_score op carrying chunk_start_idx. The synthesized
// decomposition function is the fallback body and is deleted once the typed
// promotion succeeds.

module {
  func.func @indexer_score(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK-LABEL: @indexer_score
    // CHECK: "ttnn.indexer_score"
    // CHECK-SAME: chunk_start_idx = 0 : ui32
    // CHECK-NOT: "ttcore.composite"
    %0 = "ttcore.composite"(%q, %k, %w) <{composite_name = "indexer_score", decomposition = @decomp, composite_attributes = {chunk_start_idx = 0 : ui32}}> : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
  func.func private @decomp(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x32x128xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }

  // Explicit chunk_start_idx is preserved on the promoted op.
  func.func @indexer_score_chunked(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x64x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16> {
    // CHECK-LABEL: @indexer_score_chunked
    // CHECK: "ttnn.indexer_score"
    // CHECK-SAME: chunk_start_idx = 32 : ui32
    %0 = "ttcore.composite"(%q, %k, %w) <{composite_name = "indexer_score", decomposition = @decomp_chunked, composite_attributes = {chunk_start_idx = 32 : ui32}}> : (tensor<1x8x32x128xbf16>, tensor<1x1x64x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
  func.func private @decomp_chunked(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x64x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttir.slice_static"(%k) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x64x128xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
}
