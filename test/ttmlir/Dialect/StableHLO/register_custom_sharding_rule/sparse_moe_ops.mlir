// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// =====================================================================
// all_to_all_dispatch: blocks H sharding → inserts all_gather before dispatch
// =====================================================================
// Input is sharded on H (dim 3) via _axis_0. All dispatch factors use
// kNeedReplication+isBlocked, so Shardy must insert all_gather to replicate H.
module @Dispatch_BlocksH attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x32x1x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"}, %arg1: tensor<1x32x1x4xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "indices"}, %arg2: tensor<1x1x8x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mapping"}) -> (tensor<1x2x32x128xbf16>, tensor<1x2x32x4xbf16>) {
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.all_to_all_dispatch
    %0:2 = stablehlo.custom_call @tt.all_to_all_dispatch(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2"}} : (tensor<1x32x1x128xbf16>, tensor<1x32x1x4xbf16>, tensor<1x1x8x2xbf16>) -> (tensor<1x2x32x128xbf16>, tensor<1x2x32x4xbf16>)
    return %0#0, %0#1 : tensor<1x2x32x128xbf16>, tensor<1x2x32x4xbf16>
  }
}

// -----

// =====================================================================
// all_to_all_combine: blocks H sharding → inserts all_gather before combine
// =====================================================================
// Expert output is sharded on H (dim 3) via _axis_0. Combine's H factor uses
// kNeedReplication+isBlocked, so all_gather is needed to replicate H.
module @Combine_BlocksH attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<4x2x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "expert_output"}, %arg1: tensor<1x2x32x4xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "metadata"}, %arg2: tensor<1x1x8x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mapping"}) -> tensor<4x1x32x128xbf16> {
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.all_to_all_combine
    %0 = stablehlo.custom_call @tt.all_to_all_combine(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts_per_tok = "4"}} : (tensor<4x2x32x128xbf16>, tensor<1x2x32x4xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x1x32x128xbf16>
    return %0 : tensor<4x1x32x128xbf16>
  }
}

// -----

// =====================================================================
// all_to_all_combine: E factor is kPassThrough → EP sharding propagates
// =====================================================================
// Expert output is sharded on E (dim 0) via _axis_0. Combine's E factor uses
// kPassThrough, so expert parallelism propagates without CCL insertion.
module @Combine_EPPassthrough attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<4x2x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "expert_output"}, %arg1: tensor<1x2x32x4xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "metadata"}, %arg2: tensor<1x1x8x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mapping"}) -> tensor<4x1x32x128xbf16> {
    // E is kPassThrough: no all_gather needed for EP dimension.
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tt.all_to_all_combine
    %0 = stablehlo.custom_call @tt.all_to_all_combine(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts_per_tok = "4"}} : (tensor<4x2x32x128xbf16>, tensor<1x2x32x4xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x1x32x128xbf16>
    return %0 : tensor<4x1x32x128xbf16>
  }
}

// -----

// =====================================================================
// moe_expert_token_remap: E input is blocked → all_gather, E output passthrough
// =====================================================================
// Topk input is sharded on E (dim 3) via _axis_0. The input E factor uses
// kNeedReplication+blocked (requires all_gather), but the output E factor uses
// kPassThrough so compound EP sharding propagates to the outputs.
module @Remap_EPPassthrough attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x1x32x8xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "topk"}, %arg1: tensor<1x1x8x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mapping"}, %arg2: tensor<2x1x32x4xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "metadata"}) -> (tensor<1x1x32x8xbf16>, tensor<1x1x1x8xbf16>) {
    // Input E is kNeedReplication → all_gather to replicate E on inputs.
    // Output E is kPassThrough → EP sharding propagates to outputs.
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.moe_expert_token_remap
    %0:2 = stablehlo.custom_call @tt.moe_expert_token_remap(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {num_experts_per_tok = "4", reduction_factor = "32"}} : (tensor<2x1x32x8xbf16>, tensor<1x1x8x2xbf16>, tensor<2x1x32x4xbf16>) -> (tensor<1x1x32x8xbf16>, tensor<1x1x1x8xbf16>)
    return %0#0, %0#1 : tensor<1x1x32x8xbf16>, tensor<1x1x1x8xbf16>
  }
}

// -----

// =====================================================================
// moe_expert_token_remap: D factor is blocked → all_gather for D-sharded input
// =====================================================================
// Topk and metadata are sharded on D (dim 0) via _axis_0. D factor uses
// kNeedReplication+blocked, so all_gather is needed.
module @Remap_BlocksD attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x1x32x8xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "topk"}, %arg1: tensor<1x1x8x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mapping"}, %arg2: tensor<2x1x32x4xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "metadata"}) -> (tensor<1x1x32x8xbf16>, tensor<1x1x1x8xbf16>) {
    // D is kNeedReplication+blocked → all_gather for D dimension
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.moe_expert_token_remap
    %0:2 = stablehlo.custom_call @tt.moe_expert_token_remap(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {num_experts_per_tok = "4", reduction_factor = "32"}} : (tensor<2x1x32x8xbf16>, tensor<1x1x8x2xbf16>, tensor<2x1x32x4xbf16>) -> (tensor<1x1x32x8xbf16>, tensor<1x1x1x8xbf16>)
    return %0#0, %0#1 : tensor<1x1x32x8xbf16>, tensor<1x1x1x8xbf16>
  }
}

// -----

// =====================================================================
// all_to_all_dispatch: rank-3 input path [B,S,H] + [B*S,K]
// =====================================================================
// Input is rank-3 [B,S,H] and sharded on H (dim 2). Dispatch rule should still
// block H sharding and force all_gather before the custom call.
module @Dispatch_Rank3Input attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"}, %arg1: tensor<32x4xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "indices"}, %arg2: tensor<1x1x8x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mapping"}) -> (tensor<1x2x32x128xbf16>, tensor<1x2x32x4xbf16>) {
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.all_to_all_dispatch
    %0:2 = stablehlo.custom_call @tt.all_to_all_dispatch(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2"}} : (tensor<1x32x128xbf16>, tensor<32x4xbf16>, tensor<1x1x8x2xbf16>) -> (tensor<1x2x32x128xbf16>, tensor<1x2x32x4xbf16>)
    return %0#0, %0#1 : tensor<1x2x32x128xbf16>, tensor<1x2x32x4xbf16>
  }
}

// -----

// =====================================================================
// all_to_all_combine: [BD,S,E,H] input layout (non-canonical)
// =====================================================================
// Input is [BD,S,E,H] instead of canonical [E,BD,S,H]. Combine rule should
// still block H sharding and insert all_gather before custom call.
module @Combine_BdsehInput attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x32x4x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "expert_output"}, %arg1: tensor<1x2x32x4xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "metadata"}, %arg2: tensor<1x1x8x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mapping"}) -> tensor<4x1x32x128xbf16> {
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.all_to_all_combine
    %0 = stablehlo.custom_call @tt.all_to_all_combine(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts_per_tok = "4"}} : (tensor<2x32x4x128xbf16>, tensor<1x2x32x4xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x1x32x128xbf16>
    return %0 : tensor<4x1x32x128xbf16>
  }
}

// -----

// =====================================================================
// moe_expert_token_remap: rank-2 topk input [B*S, E]
// =====================================================================
// topk input is rank-2 and sharded on E (dim 1). Input E factor is blocked,
// so all_gather must still be inserted before the custom call.
module @Remap_Topk2DInput attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<32x8xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "topk"}, %arg1: tensor<1x1x8x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mapping"}, %arg2: tensor<1x2x32x4xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "metadata"}) -> (tensor<1x2x32x8xbf16>, tensor<1x1x2x8xbf16>) {
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.moe_expert_token_remap
    %0:2 = stablehlo.custom_call @tt.moe_expert_token_remap(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {num_devices = "2", reduction_size = "32"}} : (tensor<32x8xbf16>, tensor<1x1x8x2xbf16>, tensor<1x2x32x4xbf16>) -> (tensor<1x2x32x8xbf16>, tensor<1x1x2x8xbf16>)
    return %0#0, %0#1 : tensor<1x2x32x8xbf16>, tensor<1x1x2x8xbf16>
  }
}
