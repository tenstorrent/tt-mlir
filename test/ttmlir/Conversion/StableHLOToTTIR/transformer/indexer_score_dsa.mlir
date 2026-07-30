// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @indexer_score_dsa attributes {} {
  // Default chunk_start_idx (no frontend attributes).
  func.func public @indexer_score_dsa(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK-LABEL: @indexer_score_dsa
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: chunk_start_idx = 0 : ui32
    // CHECK-SAME: composite_name = "indexer_score_dsa"
    // CHECK-SAME: decomposition = @indexer_score_dsa_decomp
    // cluster_axis is only carried when the caller names an axis, so an
    // unannotated call must produce a composite without it.
    // CHECK-NOT: cluster_axis
    %0 = stablehlo.custom_call @tt.indexer_score_dsa(%q, %k, %w) {api_version = 0 : i32} : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }

  // Explicit chunk_start_idx parsed from mhlo.frontend_attributes.
  func.func public @indexer_score_dsa_chunked(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x64x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16> {
    // CHECK-LABEL: @indexer_score_dsa_chunked
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: chunk_start_idx = 32 : ui32
    // CHECK-SAME: composite_name = "indexer_score_dsa"
    %0 = stablehlo.custom_call @tt.indexer_score_dsa(%q, %k, %w) {api_version = 0 : i32, mhlo.frontend_attributes = {chunk_start_idx = "32"}} : (tensor<1x8x32x128xbf16>, tensor<1x1x64x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }

  // cluster_axis names the mesh axis carrying the query sequence shard; it is
  // parsed from mhlo.frontend_attributes alongside chunk_start_idx and lands on
  // the composite as a ui32.
  func.func public @indexer_score_dsa_cluster_axis(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x64x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16> {
    // CHECK-LABEL: @indexer_score_dsa_cluster_axis
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: chunk_start_idx = 0 : ui32
    // CHECK-SAME: cluster_axis = 1 : ui32
    // CHECK-SAME: composite_name = "indexer_score_dsa"
    %0 = stablehlo.custom_call @tt.indexer_score_dsa(%q, %k, %w) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1"}} : (tensor<1x8x32x128xbf16>, tensor<1x1x64x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }

  // Axis 0 must stay distinguishable from "unset": both attributes together.
  func.func public @indexer_score_dsa_cluster_axis_zero(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x64x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16> {
    // CHECK-LABEL: @indexer_score_dsa_cluster_axis_zero
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: chunk_start_idx = 32 : ui32
    // CHECK-SAME: cluster_axis = 0 : ui32
    %0 = stablehlo.custom_call @tt.indexer_score_dsa(%q, %k, %w) {api_version = 0 : i32, mhlo.frontend_attributes = {chunk_start_idx = "32", cluster_axis = "0"}} : (tensor<1x8x32x128xbf16>, tensor<1x1x64x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }

  // The synthesized decomposition holds the full primitive lowering: QK^T (with
  // head-folding reshape), relu, per-head gate multiply, head-sum reduction, and
  // the causal additive -inf mask.
  // CHECK: func.func private @indexer_score_dsa_decomp
  // CHECK: "ttir.matmul"
  // CHECK: "ttir.relu"
  // CHECK: "ttir.sum"
  // The causal-mask index arithmetic runs in i32, not the bf16 element type, so
  // that key/query positions past bf16's exact-integer range (256) are not
  // conflated. The arange, chunk_start_idx constant, threshold add and the
  // comparison are all i32; only the additive 0/-inf mask is bf16.
  // CHECK: "ttir.arange"{{.*}} -> tensor<{{.*}}xi32>
  // CHECK: "ttir.arange"{{.*}} -> tensor<{{.*}}xi32>
  // CHECK: "ttir.ge"{{.*}}(tensor<{{.*}}xi32>, tensor<{{.*}}xi32>) -> tensor<{{.*}}xi32>
  // CHECK: "ttir.where"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) {{.*}}: (tensor<{{.*}}xi32>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
}
