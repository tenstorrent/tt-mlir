// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file -verify-diagnostics=only-expected --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// tt.sparse_sdpa layout:
//   query [B, H, S, K_DIM], kv [B, 1, T, K_DIM], indices [B, 1, S, TOPK]
//   -> out [B, H, S, v_dim]   (v_dim = leading columns of kv).

// Head (tensor) parallel: the query heads split across the mesh axis on
// query/out; the shared latent cache and the indices stay replicated. No
// collective is required.
// CHECK-LABEL: module @SparseSdpa_Sharding_Head
module @SparseSdpa_Sharding_Head attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x64x128x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %kv: tensor<1x1x512x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "kv"}, %indices: tensor<1x1x128x128xui32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "indices"}) -> tensor<1x64x128x512xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tt.sparse_sdpa
    // CHECK-SAME: tensor<1x32x128x576xbf16>, tensor<1x1x512x576xbf16>, tensor<1x1x128x128xui32>
    // CHECK-SAME: -> tensor<1x32x128x512xbf16>
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%query, %kv, %indices) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "512", k_chunk_size = "128"}} : (tensor<1x64x128x576xbf16>, tensor<1x1x512x576xbf16>, tensor<1x1x128x128xui32>) -> tensor<1x64x128x512xbf16>
    return %0 : tensor<1x64x128x512xbf16>
  }
}

// -----

// Query-sequence (context) parallel: query/indices/out shard on the query seq
// dim while the latent cache stays replicated. Masking is carried entirely by
// `indices` (absolute key positions), so each shard is self-contained and no
// collective is required.
// CHECK-LABEL: module @SparseSdpa_Sharding_QuerySeq
module @SparseSdpa_Sharding_QuerySeq attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x64x128x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %kv: tensor<1x1x512x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "kv"}, %indices: tensor<1x1x128x128xui32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "indices"}) -> tensor<1x64x128x512xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tt.sparse_sdpa
    // CHECK-SAME: tensor<1x64x64x576xbf16>, tensor<1x1x512x576xbf16>, tensor<1x1x64x128xui32>
    // CHECK-SAME: -> tensor<1x64x64x512xbf16>
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%query, %kv, %indices) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "512", k_chunk_size = "128"}} : (tensor<1x64x128x576xbf16>, tensor<1x1x512x576xbf16>, tensor<1x1x128x128xui32>) -> tensor<1x64x128x512xbf16>
    return %0 : tensor<1x64x128x512xbf16>
  }
}

// -----

// Key-sequence sharded kv. The key seq factor is kNeedReplication (indices
// address the full cache), so Shardy must all_gather kv on dim 2 (256 -> 512)
// before the custom_call.
// CHECK-LABEL: module @SparseSdpa_Sharding_KeySeq_NeedsAllGather
module @SparseSdpa_Sharding_KeySeq_NeedsAllGather attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x64x128x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %kv: tensor<1x1x512x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "kv"}, %indices: tensor<1x1x128x128xui32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "indices"}) -> tensor<1x64x128x512xbf16> {
    // CHECK: stablehlo.all_gather
    // CHECK-SAME: all_gather_dim = 2
    // CHECK: stablehlo.custom_call @tt.sparse_sdpa
    // CHECK-SAME: tensor<1x64x128x576xbf16>, tensor<1x1x512x576xbf16>, tensor<1x1x128x128xui32>
    // CHECK-SAME: -> tensor<1x64x128x512xbf16>
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%query, %kv, %indices) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "512", k_chunk_size = "128"}} : (tensor<1x64x128x576xbf16>, tensor<1x1x512x576xbf16>, tensor<1x1x128x128xui32>) -> tensor<1x64x128x512xbf16>
    return %0 : tensor<1x64x128x512xbf16>
  }
}

// -----

// Batch (data) parallel: every tensor (including the result) shards on the
// batch dim; the op runs entirely on local shards with no collective. Batch > 1
// falls back to the primitive decomposition on device, but the sharding rule is
// independent of that choice.
// CHECK-LABEL: module @SparseSdpa_Sharding_Batch
module @SparseSdpa_Sharding_Batch attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<2x64x128x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %kv: tensor<2x1x512x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "kv"}, %indices: tensor<2x1x128x128xui32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "indices"}) -> tensor<2x64x128x512xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tt.sparse_sdpa
    // CHECK-SAME: tensor<1x64x128x576xbf16>, tensor<1x1x512x576xbf16>, tensor<1x1x128x128xui32>
    // CHECK-SAME: -> tensor<1x64x128x512xbf16>
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%query, %kv, %indices) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "512", k_chunk_size = "128"}} : (tensor<2x64x128x576xbf16>, tensor<2x1x512x576xbf16>, tensor<2x1x128x128xui32>) -> tensor<2x64x128x512xbf16>
    return %0 : tensor<2x64x128x512xbf16>
  }
}
