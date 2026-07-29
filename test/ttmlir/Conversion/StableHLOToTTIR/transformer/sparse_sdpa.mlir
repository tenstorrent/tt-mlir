// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @sparse_sdpa attributes {} {
  // Default scale and k_chunk_size (only v_dim supplied).
  func.func public @sparse_sdpa(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK-LABEL: @sparse_sdpa
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: k_chunk_size = 32 : ui32
    // CHECK-SAME: v_dim = 32 : ui32
    // CHECK-SAME: composite_name = "sparse_sdpa"
    // CHECK-SAME: decomposition = @sparse_sdpa_decomp
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%q, %kv, %idx) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "32", k_chunk_size = "32"}} : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }

  // Explicit scale parsed from mhlo.frontend_attributes.
  func.func public @sparse_sdpa_scaled(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK-LABEL: @sparse_sdpa_scaled
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: k_chunk_size = 32 : ui32
    // CHECK-SAME: scale = 1.250000e-01 : f32
    // CHECK-SAME: v_dim = 32 : ui32
    // CHECK-SAME: composite_name = "sparse_sdpa"
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%q, %kv, %idx) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "32", k_chunk_size = "32", scale = "0.125"}} : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }

  // k_chunk_size defaults to the tt-metal default (128) when absent, so TOPK
  // must be a multiple of 128 here.
  func.func public @sparse_sdpa_default_chunk(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x256x64xbf16>, %idx: tensor<1x1x32x128xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK-LABEL: @sparse_sdpa_default_chunk
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: k_chunk_size = 128 : ui32
    // CHECK-SAME: v_dim = 32 : ui32
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%q, %kv, %idx) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "32"}} : (tensor<1x32x32x64xbf16>, tensor<1x1x256x64xbf16>, tensor<1x1x32x128xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }

  // The synthesized decomposition holds the full primitive lowering: QK^T (with
  // head-folding reshape), the scale multiply, the dense sparsity mask built
  // from `indices`, softmax, and the probs @ V matmul against the leading v_dim
  // columns of the latent cache.
  // CHECK: func.func private @sparse_sdpa_decomp
  // CHECK-DAG: "ttir.permute"
  // CHECK-DAG: "ttir.matmul"
  // CHECK-DAG: "ttir.softmax"
  // CHECK-DAG: "ttir.slice_static"
  // The mask index arithmetic runs in f32, not the bf16 element type, so that
  // key positions past bf16's exact-integer range (256) are not conflated. The
  // typecast of `indices` and the in-range predicate are f32; only the additive
  // 0/-inf mask is bf16.
  //
  // The membership test is a scatter-accumulate into a [B, S, T] hit-count
  // buffer, NOT a one-hot [B, S, TOPK, T] compare-and-reduce: the latter needs
  // O(S * TOPK * T) memory (1.07e9 elements at S = T = TOPK = 1024) and does not
  // fit. Reduction must be `sum` so that a masked slot redirected onto key 0
  // (contributing 0.0) cannot clear a genuine hit on key 0 in the same row.
  // CHECK-DAG: "ttir.lt"{{.*}}(tensor<{{.*}}xf32>, tensor<{{.*}}xf32>) -> tensor<{{.*}}xf32>
  // CHECK-DAG: "ttir.ge"{{.*}}(tensor<{{.*}}xf32>, tensor<{{.*}}xf32>) -> tensor<{{.*}}xf32>
  // CHECK-DAG: "ttir.scatter"{{.*}}scatter_reduce_type = #ttcore.reduce_type<sum>
  // CHECK-DAG: "ttir.where"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) {{.*}}: (tensor<{{.*}}xf32>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
  // No one-hot slot tensor may reappear.
  // CHECK-NOT: "ttir.eq"
}
