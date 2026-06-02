// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file -verify-diagnostics=only-expected --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Tests for MLA Prefill with Q, K, V tensors using the following parameters (from DSv3.1)
//   Hq = Hkv   = 128   (num_attention_heads; K/V materialized per head)
//   dh_qk      = 192   (qk_nope_head_dim 128 + qk_rope_head_dim 64)
//   head_dim_v = 128   (v_head_dim)
//   S          = 2048  (a prefill chunk)
// Value is an explicit per-head tensor (the W_UV projection), so has_value is
// True. Because every tensor shares the head dim, head/tensor-parallel
// sharding splits Hq across the mesh on Q, K, V, and the output alike.

// Head (tensor) parallel: the 128 heads split across the mesh axis (-> 64 per
// shard) on Q/K/V/Out. No collective is required.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_HeadParallel
module @FlashMlaPrefill_Sharding_HeadParallel attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %value: tensor<1x128x2048x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<1x128x2048x128xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x64x2048x192xbf16>, tensor<1x64x2048x192xbf16>, tensor<1x64x2048x128xbf16>
    // CHECK-SAME: -> tensor<1x64x2048x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "True", has_value = "True", has_attention_mask = "False"}} : (tensor<1x128x2048x192xbf16>, tensor<1x128x2048x192xbf16>, tensor<1x128x2048x128xbf16>) -> tensor<1x128x2048x128xbf16>
    return %0 : tensor<1x128x2048x128xbf16>
  }
}

// -----

// Batch (data) parallel: batch splits across the mesh axis; heads, sequence,
// and head dims replicate. No collective is required.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_BatchParallel
module @FlashMlaPrefill_Sharding_BatchParallel attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<2x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<2x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %value: tensor<2x128x2048x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<2x128x2048x128xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x128x2048x192xbf16>, tensor<1x128x2048x192xbf16>, tensor<1x128x2048x128xbf16>
    // CHECK-SAME: -> tensor<1x128x2048x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "True", has_value = "True", has_attention_mask = "False"}} : (tensor<2x128x2048x192xbf16>, tensor<2x128x2048x192xbf16>, tensor<2x128x2048x128xbf16>) -> tensor<2x128x2048x128xbf16>
    return %0 : tensor<2x128x2048x128xbf16>
  }
}

// -----

// Batch (data) parallel with a broadcast mask ([1, 1, S, S]). The mask sits
// out of the Batch factor (kNullDim), so it stays full on every shard.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_BatchParallel_BroadcastMask
module @FlashMlaPrefill_Sharding_BatchParallel_BroadcastMask attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<2x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<2x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %value: tensor<2x128x2048x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}, %mask: tensor<1x1x2048x2048xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mask"}) -> tensor<2x128x2048x128xbf16> {
    // Batch-sharded Q/K/V; broadcast mask stays at [1, 1, 2048, 2048].
    // CHECK-NOT: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x128x2048x192xbf16>, tensor<1x128x2048x192xbf16>, tensor<1x128x2048x128xbf16>, tensor<1x1x2048x2048xbf16>
    // CHECK-SAME: -> tensor<1x128x2048x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value, %mask) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "False", has_value = "True", has_attention_mask = "True"}} : (tensor<2x128x2048x192xbf16>, tensor<2x128x2048x192xbf16>, tensor<2x128x2048x128xbf16>, tensor<1x1x2048x2048xbf16>) -> tensor<2x128x2048x128xbf16>
    return %0 : tensor<2x128x2048x128xbf16>
  }
}

// -----

// Batch (data) parallel with a per-batch mask ([B, 1, S, S]). The mask's batch
// dim is mapped to the Batch factor (maskBatch=0), so it shards with Q/K/V.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_BatchParallel_PerBatchMask
module @FlashMlaPrefill_Sharding_BatchParallel_PerBatchMask attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<2x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<2x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %value: tensor<2x128x2048x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}, %mask: tensor<2x1x2048x2048xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mask"}) -> tensor<2x128x2048x128xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x128x2048x192xbf16>, tensor<1x128x2048x192xbf16>, tensor<1x128x2048x128xbf16>, tensor<1x1x2048x2048xbf16>
    // CHECK-SAME: -> tensor<1x128x2048x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value, %mask) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "False", has_value = "True", has_attention_mask = "True"}} : (tensor<2x128x2048x192xbf16>, tensor<2x128x2048x192xbf16>, tensor<2x128x2048x128xbf16>, tensor<2x1x2048x2048xbf16>) -> tensor<2x128x2048x128xbf16>
    return %0 : tensor<2x128x2048x128xbf16>
  }
}

// -----

// Sequence (context) parallel Q/K/V. Seq is kNeedReplication in the rule, but
// since every tensor shares the head dim, Shardy converts the seq sharding
// into head sharding via all_to_all (rather than gathering), runs the head-
// sharded attention, then all_to_alls the output back to seq sharding.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_SeqParallel_RedistributedToHead
module @FlashMlaPrefill_Sharding_SeqParallel_RedistributedToHead attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %value: tensor<1x128x2048x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<1x128x2048x128xbf16> {
    // CHECK: stablehlo.all_to_all
    // CHECK: stablehlo.all_to_all
    // CHECK: stablehlo.all_to_all
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x64x2048x192xbf16>, tensor<1x64x2048x192xbf16>, tensor<1x64x2048x128xbf16>
    // CHECK-SAME: -> tensor<1x64x2048x128xbf16>
    // CHECK: stablehlo.all_to_all
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "True", has_value = "True", has_attention_mask = "False"}} : (tensor<1x128x2048x192xbf16>, tensor<1x128x2048x192xbf16>, tensor<1x128x2048x128xbf16>) -> tensor<1x128x2048x128xbf16>
    return %0 : tensor<1x128x2048x128xbf16>
  }
}

// -----

// dh_qk-sharded Q/K. dh_qk is kNeedReplication (Q/K-only factor), so Shardy
// must insert an all_gather on dim 3 (96 -> 192) before the custom_call.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_DhQk_NeedsAllGather
module @FlashMlaPrefill_Sharding_DhQk_NeedsAllGather attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %value: tensor<1x128x2048x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<1x128x2048x128xbf16> {
    // CHECK: stablehlo.all_gather
    // CHECK-SAME: all_gather_dim = 3
    // CHECK: stablehlo.all_gather
    // CHECK-SAME: all_gather_dim = 3
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x128x2048x192xbf16>, tensor<1x128x2048x192xbf16>, tensor<1x128x2048x128xbf16>
    // CHECK-SAME: -> tensor<1x128x2048x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "True", has_value = "True", has_attention_mask = "False"}} : (tensor<1x128x2048x192xbf16>, tensor<1x128x2048x192xbf16>, tensor<1x128x2048x128xbf16>) -> tensor<1x128x2048x128xbf16>
    return %0 : tensor<1x128x2048x128xbf16>
  }
}

// -----

// head_dim_v-sharded value. head_dim_v is kNeedReplication (V/Out-only factor),
// so Shardy must insert an all_gather on V's dim 3 (64 -> 128).
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_HeadDimV_NeedsAllGather
module @FlashMlaPrefill_Sharding_HeadDimV_NeedsAllGather attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x128x2048x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %value: tensor<1x128x2048x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<1x128x2048x128xbf16> {
    // CHECK: stablehlo.all_gather
    // CHECK-SAME: all_gather_dim = 3
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x128x2048x192xbf16>, tensor<1x128x2048x192xbf16>, tensor<1x128x2048x128xbf16>
    // CHECK-SAME: -> tensor<1x128x2048x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "True", has_value = "True", has_attention_mask = "False"}} : (tensor<1x128x2048x192xbf16>, tensor<1x128x2048x192xbf16>, tensor<1x128x2048x128xbf16>) -> tensor<1x128x2048x128xbf16>
    return %0 : tensor<1x128x2048x128xbf16>
  }
}


// -----

// Compressed latent K/V: a single shared KV head (kvHeads == 1) is broadcast
// across every query head. The head factor maps only Q/Out (kvHeadDim is
// kNullDim for K/V), so sharding the 128 query heads (-> 64 per shard) leaves
// the latent K head replicated and requires no collective.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_LatentKV_HeadParallel
module @FlashMlaPrefill_Sharding_LatentKV_HeadParallel attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x128x2048x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x1x2048x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}) -> tensor<1x128x2048x512xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_to_all
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x64x2048x576xbf16>, tensor<1x1x2048x576xbf16>
    // CHECK-SAME: -> tensor<1x64x2048x512xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "512", is_causal = "True", has_value = "False", has_attention_mask = "False"}} : (tensor<1x128x2048x576xbf16>, tensor<1x1x2048x576xbf16>) -> tensor<1x128x2048x512xbf16>
    return %0 : tensor<1x128x2048x512xbf16>
  }
}
