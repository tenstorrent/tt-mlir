// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole composite-resolution=force-promote" %s | FileCheck %s --implicit-check-not="ttnn.sparse_sdpa"

// Even on Blackhole, ttnn::transformer::sparse_sdpa is a single-batch kernel, so
// the promotion guard vetoes a batch > 1 and TTNNResolveComposites inlines the
// decomposition body instead -- again under force-promote, which would otherwise
// promote unconditionally.

module {
  func.func @sparse_sdpa_batched(%q: tensor<4x32x32x64xbf16>, %kv: tensor<4x1x64x64xbf16>, %idx: tensor<4x1x32x32xui32>) -> tensor<4x32x32x32xbf16> {
    // CHECK-LABEL: @sparse_sdpa_batched
    // CHECK-NOT: "ttcore.composite"
    // The decomposition body is spliced in and lowered to TTNN primitives.
    // CHECK: "ttnn.slice_static"
    %0 = "ttcore.composite"(%q, %kv, %idx) <{composite_name = "sparse_sdpa", decomposition = @decomp, composite_attributes = {v_dim = 32 : ui32, k_chunk_size = 32 : ui32}}> : (tensor<4x32x32x64xbf16>, tensor<4x1x64x64xbf16>, tensor<4x1x32x32xui32>) -> tensor<4x32x32x32xbf16>
    return %0 : tensor<4x32x32x32xbf16>
  }
  func.func private @decomp(%q: tensor<4x32x32x64xbf16>, %kv: tensor<4x1x64x64xbf16>, %idx: tensor<4x1x32x32xui32>) -> tensor<4x32x32x32xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [4 : i32, 32 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<4x32x32x64xbf16>) -> tensor<4x32x32x32xbf16>
    return %0 : tensor<4x32x32x32xbf16>
  }
}
