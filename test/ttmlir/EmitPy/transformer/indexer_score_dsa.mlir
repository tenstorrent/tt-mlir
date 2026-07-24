// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="mock-system-desc-arch=blackhole composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// The ttcore.composite "indexer_score_dsa" is promoted to ttnn.indexer_score_dsa by
// TTNNResolveComposites and then emitted to Python as
// ttnn.experimental.indexer_score_dsa.

func.func @indexer_score_dsa(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
  // CHECK-LABEL: def indexer_score_dsa
  // CHECK: ttnn.experimental.indexer_score_dsa({{[a-z_0-9]+}}, {{[a-z_0-9]+}}, {{[a-z_0-9]+}}, chunk_start_idx=0)
  %0 = "ttcore.composite"(%q, %k, %w) <{composite_name = "indexer_score_dsa", decomposition = @decomp, composite_attributes = {chunk_start_idx = 0 : ui32}}> : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
  return %0 : tensor<1x1x32x32xbf16>
}
func.func private @decomp(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
  %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x32x128xbf16>) -> tensor<1x1x32x32xbf16>
  return %0 : tensor<1x1x32x32xbf16>
}
