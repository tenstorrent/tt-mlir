// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Query-sequence-sharded indexer_score_dsa: cluster_axis + num_devices are set
// (num_devices > 1), so the composite's synthesized decomposition -- the
// fallback TTNNResolveComposites inlines whenever the promoted
// ttnn.indexer_score_dsa kernel isn't used -- must recover each device's true
// causal-window offset itself, since it is a single MLIR region shared across
// every device with no per-device specialization of its own (unlike the
// promoted kernel, which is instantiated once per mesh coordinate). It does
// this via ttir.mesh_partition over a global row-index arange rather than a
// device-local one; see StableHLOToTTIRPatterns.cpp's
// buildIndexerScoreDsaDecompositionBody and the bug this fixes
// (docs/dsa_blackhole_tt-mlir_changes.md, RegisterCustomShardingRule.cpp's
// getIndexerScoreDsaShardingRule comment).

module @indexer_score_dsa_sharded attributes {} {
  // Local query seq = 32 (one of num_devices=4 shards of a global 128-row
  // query sequence sharded over cluster_axis=1); key stays at the full,
  // unsharded 128.
  func.func public @indexer_score_dsa_sharded(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x128x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK-LABEL: @indexer_score_dsa_sharded
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: chunk_start_idx = 0 : ui32
    // CHECK-SAME: cluster_axis = 1 : ui32
    // CHECK-SAME: composite_name = "indexer_score_dsa"
    // CHECK-SAME: decomposition = @indexer_score_dsa_decomp
    %0 = stablehlo.custom_call @tt.indexer_score_dsa(%q, %k, %w) {api_version = 0 : i32, mhlo.frontend_attributes = {chunk_start_idx = "0", cluster_axis = "1", num_devices = "4"}} : (tensor<1x8x32x128xbf16>, tensor<1x1x128x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x128xbf16>
    return %0 : tensor<1x1x32x128xbf16>
  }

  // The synthesized decomposition still holds the full primitive lowering
  // (QK^T, relu, gate multiply, head-sum, causal mask) -- only the row-index
  // construction differs from the unsharded case.
  // CHECK: func.func private @indexer_score_dsa_decomp
  // CHECK: "ttir.matmul"
  // CHECK: "ttir.relu"
  // CHECK: "ttir.sum"
  //
  // Row index: a GLOBAL arange over all num_devices=4 shards worth of query
  // rows (4 * 32 = 128), narrowed to a single [1,1,128,1] column rather than
  // the full [B,Hi,128,T] tensor (kept small; broadcast happens after the
  // partition, not before it).
  // CHECK: "ttir.arange"{{.*}} -> tensor<1x1x128x1xi32>
  // Partitioned down to this device's own [1,1,32,1] window via
  // ttir.mesh_partition on dim 2, naming cluster_axis=1 -- the same axis and
  // convention (a device's coordinate along that axis IS its rank) the
  // promoted kernel's get_linearized_index_from_physical_coord uses, so the
  // decomposition and the kernel agree on which row range a given device
  // owns.
  // CHECK: "ttir.mesh_partition"
  // CHECK-SAME: cluster_axis = 1 : ui32
  // CHECK-SAME: dim = 2 : si32
  // Broadcast the per-device row column back out to the score's shape.
  // CHECK: "ttir.broadcast"
  // Key index: still a plain local arange -- key is replicated, never sharded.
  // CHECK: "ttir.arange"{{.*}} -> tensor<{{.*}}xi32>
  // CHECK: "ttir.ge"{{.*}}(tensor<{{.*}}xi32>, tensor<{{.*}}xi32>) -> tensor<{{.*}}xi32>
  // CHECK: "ttir.where"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) {{.*}}: (tensor<{{.*}}xi32>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
}
