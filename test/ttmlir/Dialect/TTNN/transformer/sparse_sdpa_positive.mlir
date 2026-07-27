// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole composite-resolution=force-promote" %s | FileCheck %s

// Resolves a ttcore.composite "sparse_sdpa" through the TTNN backend pipeline
// (TTNNResolveComposites) on a Blackhole target with batch 1 and verifies it is
// promoted to the typed ttnn.sparse_sdpa op carrying v_dim / scale /
// k_chunk_size. The synthesized decomposition function is the fallback body and
// is deleted once the typed promotion succeeds.

module {
  func.func @sparse_sdpa(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK-LABEL: @sparse_sdpa
    // CHECK: "ttnn.sparse_sdpa"
    // CHECK-SAME: k_chunk_size = 32 : ui32
    // CHECK-SAME: v_dim = 32 : ui32
    // CHECK-NOT: "ttcore.composite"
    %0 = "ttcore.composite"(%q, %kv, %idx) <{composite_name = "sparse_sdpa", decomposition = @decomp, composite_attributes = {v_dim = 32 : ui32, k_chunk_size = 32 : ui32}}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
  func.func private @decomp(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }

  // An explicit scale is preserved on the promoted op.
  func.func @sparse_sdpa_scaled(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK-LABEL: @sparse_sdpa_scaled
    // CHECK: "ttnn.sparse_sdpa"
    // CHECK-SAME: scale = 1.250000e-01 : f32
    %0 = "ttcore.composite"(%q, %kv, %idx) <{composite_name = "sparse_sdpa", decomposition = @decomp_scaled, composite_attributes = {v_dim = 32 : ui32, k_chunk_size = 32 : ui32, scale = 1.250000e-01 : f32}}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
  func.func private @decomp_scaled(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}
