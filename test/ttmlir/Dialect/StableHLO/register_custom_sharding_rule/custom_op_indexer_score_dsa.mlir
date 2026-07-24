// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// tt.indexer_score_dsa layout:
//   query [B, Hi, Sq, D], key [B, 1, T, D], weights [B, Hi, Sq, 1]
//   -> score [B, 1, Sq, T]   (heads summed away).

// Tensor parallelism over query heads. query/weights are sharded on the head
// dim (Hi); the key's single shared head stays replicated. Because the head
// dim is summed away, sharding it makes each device produce a partial score,
// so the partitioner inserts an all_reduce(sum) to combine them.
// CHECK-LABEL: module @IndexerScoreDsa_Sharding_Head
module @IndexerScoreDsa_Sharding_Head attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x8x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %arg1: tensor<1x1x32x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %arg2: tensor<1x8x32x1xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "weights"}) -> tensor<1x1x32x32xbf16> {
    // The custom call runs on the local head-shard shapes; the key stays
    // replicated and the head dim is not all-gathered before the op.
    // CHECK-NOT: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.indexer_score_dsa
    // CHECK-SAME: tensor<1x4x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x4x32x1xbf16>
    // CHECK-SAME: -> tensor<1x1x32x32xbf16>
    // CHECK: stablehlo.all_reduce
    // CHECK: stablehlo.add
    %0 = stablehlo.custom_call @tt.indexer_score_dsa(%arg0, %arg1, %arg2) {api_version = 0 : i32} : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}

// -----

// Data parallelism over batch. Every tensor (including the result) shards on
// the batch dim; the op runs entirely on local shards with no collective.
// CHECK-LABEL: module @IndexerScoreDsa_Sharding_Batch
module @IndexerScoreDsa_Sharding_Batch attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x8x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %arg1: tensor<2x1x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %arg2: tensor<2x8x32x1xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "weights"}) -> tensor<2x1x32x32xbf16> {
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tt.indexer_score_dsa
    // CHECK-SAME: tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x1xbf16>
    // CHECK-SAME: -> tensor<1x1x32x32xbf16>
    %0 = stablehlo.custom_call @tt.indexer_score_dsa(%arg0, %arg1, %arg2) {api_version = 0 : i32} : (tensor<2x8x32x128xbf16>, tensor<2x1x32x128xbf16>, tensor<2x8x32x1xbf16>) -> tensor<2x1x32x32xbf16>
    return %0 : tensor<2x1x32x32xbf16>
  }
}
