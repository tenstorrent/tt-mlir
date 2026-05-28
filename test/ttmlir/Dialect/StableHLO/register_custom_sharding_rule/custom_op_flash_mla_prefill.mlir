// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Head sharding (no value, no mask, causal). qHeads=4, kvHeads=2 — both
// divisible by the mesh axis size, so Shardy can shard the head factor.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_Head
module @FlashMlaPrefill_Sharding_Head attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x4x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x2x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}) -> tensor<1x4x32x128xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x2x32x192xbf16>, tensor<1x1x32x192xbf16>
    // CHECK-SAME: -> tensor<1x2x32x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "True", has_value = "False", has_attention_mask = "False"}} : (tensor<1x4x32x192xbf16>, tensor<1x2x32x192xbf16>) -> tensor<1x4x32x128xbf16>
    return %0 : tensor<1x4x32x128xbf16>
  }
}

// -----

// Head sharding with a value tensor. V should shard the same as K.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_Head_WithValue
module @FlashMlaPrefill_Sharding_Head_WithValue attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x4x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x2x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %value: tensor<1x2x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<1x4x32x128xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x2x32x192xbf16>, tensor<1x1x32x192xbf16>, tensor<1x1x32x128xbf16>
    // CHECK-SAME: -> tensor<1x2x32x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "True", has_value = "True", has_attention_mask = "False"}} : (tensor<1x4x32x192xbf16>, tensor<1x2x32x192xbf16>, tensor<1x2x32x128xbf16>) -> tensor<1x4x32x128xbf16>
    return %0 : tensor<1x4x32x128xbf16>
  }
}

// -----

// Batch sharding with a broadcast mask ([1, 1, S, S]). Mask sits out of the
// Batch factor (kNullDim), so it should remain full on every shard.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_Batch_BroadcastMask
module @FlashMlaPrefill_Sharding_Batch_BroadcastMask attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<2x4x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<2x2x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %mask: tensor<1x1x32x32xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mask"}) -> tensor<2x4x32x128xbf16> {
    // Batch-sharded Q/K; broadcast mask stays at [1, 1, 32, 32].
    // CHECK-NOT: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x4x32x192xbf16>, tensor<1x2x32x192xbf16>, tensor<1x1x32x32xbf16>
    // CHECK-SAME: -> tensor<1x4x32x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %mask) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "False", has_value = "False", has_attention_mask = "True"}} : (tensor<2x4x32x192xbf16>, tensor<2x2x32x192xbf16>, tensor<1x1x32x32xbf16>) -> tensor<2x4x32x128xbf16>
    return %0 : tensor<2x4x32x128xbf16>
  }
}

// -----

// Batch sharding with a per-batch mask ([B, 1, S, S]). The mask's batch dim
// is mapped to the Batch factor (maskBatch=0), so it shards alongside Q/K.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_Batch_PerBatchMask
module @FlashMlaPrefill_Sharding_Batch_PerBatchMask attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<2x4x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<2x2x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %mask: tensor<2x1x32x32xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mask"}) -> tensor<2x4x32x128xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x4x32x192xbf16>, tensor<1x2x32x192xbf16>, tensor<1x1x32x32xbf16>
    // CHECK-SAME: -> tensor<1x4x32x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %mask) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "False", has_value = "False", has_attention_mask = "True"}} : (tensor<2x4x32x192xbf16>, tensor<2x2x32x192xbf16>, tensor<2x1x32x32xbf16>) -> tensor<2x4x32x128xbf16>
    return %0 : tensor<2x4x32x128xbf16>
  }
}

// -----

// Sequence-length sharded Q/K. Seq is kNeedReplication in the rule, so it
// cannot stay sharded on the custom_call. Shardy converts the seq-sharded
// inputs into head-sharded inputs via an `all_to_all`, then runs the
// custom_call on the head-sharded layout.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_Seq_RedistributedToHead
module @FlashMlaPrefill_Sharding_Seq_RedistributedToHead attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x4x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x2x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}) -> tensor<1x4x32x128xbf16> {
    // CHECK: stablehlo.all_to_all
    // CHECK: stablehlo.all_to_all
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x2x32x192xbf16>, tensor<1x1x32x192xbf16>
    // CHECK-SAME: -> tensor<1x2x32x128xbf16>
    // CHECK: stablehlo.all_to_all
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "True", has_value = "False", has_attention_mask = "False"}} : (tensor<1x4x32x192xbf16>, tensor<1x2x32x192xbf16>) -> tensor<1x4x32x128xbf16>
    return %0 : tensor<1x4x32x128xbf16>
  }
}

// -----

// dh_qk-sharded Q/K. dh_qk is kNeedReplication (Q/K-only factor), so Shardy
// must insert an all_gather on dim 3 before the custom_call.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_DhQk_NeedsAllGather
module @FlashMlaPrefill_Sharding_DhQk_NeedsAllGather attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x4x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x2x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}) -> tensor<1x4x32x128xbf16> {
    // CHECK: stablehlo.all_gather
    // CHECK-SAME: all_gather_dim = 3
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x4x32x192xbf16>, tensor<1x2x32x192xbf16>
    // CHECK-SAME: -> tensor<1x4x32x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "True", has_value = "False", has_attention_mask = "False"}} : (tensor<1x4x32x192xbf16>, tensor<1x2x32x192xbf16>) -> tensor<1x4x32x128xbf16>
    return %0 : tensor<1x4x32x128xbf16>
  }
}

// -----

// head_dim_v-sharded value tensor. head_dim_v is kNeedReplication (V/Out-only
// factor), so Shardy must insert an all_gather on V's dim 3.
// CHECK-LABEL: module @FlashMlaPrefill_Sharding_HeadDimV_NeedsAllGather
module @FlashMlaPrefill_Sharding_HeadDimV_NeedsAllGather attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x4x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x2x32x192xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %value: tensor<1x2x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<1x4x32x128xbf16> {
    // CHECK: stablehlo.all_gather
    // CHECK-SAME: all_gather_dim = 3
    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
    // CHECK-SAME: tensor<1x4x32x192xbf16>, tensor<1x2x32x192xbf16>, tensor<1x2x32x128xbf16>
    // CHECK-SAME: -> tensor<1x4x32x128xbf16>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "128", is_causal = "True", has_value = "True", has_attention_mask = "False"}} : (tensor<1x4x32x192xbf16>, tensor<1x2x32x192xbf16>, tensor<1x2x32x128xbf16>) -> tensor<1x4x32x128xbf16>
    return %0 : tensor<1x4x32x128xbf16>
  }
}
