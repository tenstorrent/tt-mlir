// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// =====================================================================
// topk_router_gpt: batch factor is passthrough, so no collectives are needed
// =====================================================================
module @TopkRouter_BatchPassthrough attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<4x1x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "hidden"}, %arg1: tensor<8x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "weight"}, %arg2: tensor<8xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "bias"}) -> (tensor<4x1x2xi64>, tensor<4x1x2xbf16>) {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tt.topk_router_gpt
    %0:2 = stablehlo.custom_call @tt.topk_router_gpt(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {k = "2"}} : (tensor<4x1x128xbf16>, tensor<8x128xbf16>, tensor<8xbf16>) -> (tensor<4x1x2xi64>, tensor<4x1x2xbf16>)
    return %0#0, %0#1 : tensor<4x1x2xi64>, tensor<4x1x2xbf16>
  }
}

// -----

// =====================================================================
// all_to_all_dispatch_metadata: H is blocked, so all_gather is inserted
// =====================================================================
module @DispatchMetadata_BlocksH attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x1x1x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "hidden"}, %arg1: tensor<2x1x1x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "indices"}, %arg2: tensor<2x1x1x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "scores"}, %arg3: tensor<1x1x8x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mapping"}) -> (tensor<1x4x1x128xbf16>, tensor<1x4x1x2xi64>, tensor<1x4x1x2xbf16>) {
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.all_to_all_dispatch_metadata
    %0:3 = stablehlo.custom_call @tt.all_to_all_dispatch_metadata(%arg0, %arg1, %arg2, %arg3) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2"}} : (tensor<2x1x1x128xbf16>, tensor<2x1x1x2xi64>, tensor<2x1x1x2xbf16>, tensor<1x1x8x2xi64>) -> (tensor<1x4x1x128xbf16>, tensor<1x4x1x2xi64>, tensor<1x4x1x2xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x4x1x128xbf16>, tensor<1x4x1x2xi64>, tensor<1x4x1x2xbf16>
  }
}

// -----

// =====================================================================
// moe_gpt: hidden input H is blocked, so all_gather is inserted
// =====================================================================
module @MoeGpt_BlocksInputH attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x4x1x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,1,1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "hidden"}, %arg1: tensor<1x4x1x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "indices"}, %arg2: tensor<1x4x1x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "scores"}, %arg3: tensor<1x1x8x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mapping"}, %arg4: tensor<8x128x256xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "gate_up"}, %arg5: tensor<8x256xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "gate_up_bias"}, %arg6: tensor<8x128x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "down"}, %arg7: tensor<8x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "down_bias"}) -> tensor<8x1x4x128xbf16> {
    // CHECK: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.moe_gpt
    %0:5 = stablehlo.custom_call @tt.moe_gpt(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts_per_tok = "2"}} : (tensor<1x4x1x128xbf16>, tensor<1x4x1x2xi64>, tensor<1x4x1x2xbf16>, tensor<1x1x8x2xi64>, tensor<8x128x256xbf16>, tensor<8x256xbf16>, tensor<8x128x128xbf16>, tensor<8x128xbf16>) -> (tensor<1x1x8x2xi64>, tensor<1x4x1x2xi64>, tensor<1x4x1x2xbf16>, tensor<1x4x1x2xbf16>, tensor<8x1x4x128xbf16>)
    return %0#4 : tensor<8x1x4x128xbf16>
  }
}

// -----

// =====================================================================
// selective_reduce_combine: E factor is passthrough, so EP sharding propagates
// =====================================================================
module @SelectiveReduceCombine_EPPassthrough attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<8x1x4x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "expert_output"}, %arg1: tensor<1x4x1x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "metadata_indices"}, %arg2: tensor<1x4x1x2xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "metadata_scores"}, %arg3: tensor<1x1x8x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "combine_metadata"}) -> tensor<2x1x2x128xbf16> {
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_reduce
    // CHECK: stablehlo.custom_call @tt.selective_reduce_combine
    %0 = stablehlo.custom_call @tt.selective_reduce_combine(%arg0, %arg1, %arg2, %arg3) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts_per_tok = "2", output_shard_dim = "2"}} : (tensor<8x1x4x128xbf16>, tensor<1x4x1x2xi64>, tensor<1x4x1x2xbf16>, tensor<1x1x8x2xi64>) -> tensor<2x1x2x128xbf16>
    return %0 : tensor<2x1x2x128xbf16>
  }
}
