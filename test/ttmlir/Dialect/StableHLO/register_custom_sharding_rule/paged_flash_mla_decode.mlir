// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Query-head sharding for paged flash MLA decode, MLA-from-latent variant
// (value absent). The compressed latent KV cache (nkv = 1) is shared across all
// query heads, so it must stay replicated while the query head dimension is
// sharded. The custom sharding rule maps only query dim 2 <-> output dim 2, so
// the 16 query heads shard to 8 per device while key/page_table/cur_pos stay
// replicated.
module @PagedFlashMlaDecodeQHeadShardingLatent attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x2x16x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %key: tensor<8x1x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %page_table: tensor<2x4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %cur_pos: tensor<2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}) -> tensor<1x2x16x64xbf16> {
    // CHECK: stablehlo.custom_call @tt.paged_flash_mla_decode
    // CHECK-SAME: tensor<1x2x8x64xbf16>, tensor<8x1x32x64xbf16>, tensor<2x4xi64>, tensor<2xi64>
    // CHECK-SAME: -> tensor<1x2x8x64xbf16>
    %0 = stablehlo.custom_call @tt.paged_flash_mla_decode(%query, %key, %page_table, %cur_pos) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", has_attention_mask = "False", has_attention_sink = "False", has_cur_pos_tensor = "True", has_value = "False", is_causal = "True"}} : (tensor<1x2x16x64xbf16>, tensor<8x1x32x64xbf16>, tensor<2x4xi64>, tensor<2xi64>) -> tensor<1x2x16x64xbf16>
    return %0 : tensor<1x2x16x64xbf16>
  }
}

// -----

// Same query-head sharding, but with a provided (non-latent) value cache. The
// value cache, like the key cache, is not head-sharded by the rule and stays
// replicated; only the query/output head dimension is sharded.
module @PagedFlashMlaDecodeQHeadShardingValue attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x2x16x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %key: tensor<8x1x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %value: tensor<8x1x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %page_table: tensor<2x4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}, %cur_pos: tensor<2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_4"}) -> tensor<1x2x16x64xbf16> {
    // CHECK: stablehlo.custom_call @tt.paged_flash_mla_decode
    // CHECK-SAME: tensor<1x2x8x64xbf16>, tensor<8x1x32x64xbf16>, tensor<8x1x32x64xbf16>, tensor<2x4xi64>, tensor<2xi64>
    // CHECK-SAME: -> tensor<1x2x8x64xbf16>
    %0 = stablehlo.custom_call @tt.paged_flash_mla_decode(%query, %key, %value, %page_table, %cur_pos) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", has_attention_mask = "False", has_attention_sink = "False", has_cur_pos_tensor = "True", has_value = "True", is_causal = "True"}} : (tensor<1x2x16x64xbf16>, tensor<8x1x32x64xbf16>, tensor<8x1x32x64xbf16>, tensor<2x4xi64>, tensor<2xi64>) -> tensor<1x2x16x64xbf16>
    return %0 : tensor<1x2x16x64xbf16>
  }
}

// -----

// Batch (num_users) sharding for paged flash MLA decode, MLA-from-latent
// variant (value absent). The query num_users dim is sharded, so each device
// handles a slice of users. The user-indexed operands (page_table, cur_pos)
// are sharded on their num_users dim in lockstep, while the shared latent KV
// cache stays replicated (its blocks are addressed by the per-device page
// table into the full, replicated block pool). 4 users shard to 2 per device.
module @PagedFlashMlaDecodeBatchSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x4x16x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %key: tensor<8x1x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %page_table: tensor<4x4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %cur_pos: tensor<4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}) -> tensor<1x4x16x64xbf16> {
    // CHECK: stablehlo.custom_call @tt.paged_flash_mla_decode
    // CHECK-SAME: tensor<1x2x16x64xbf16>, tensor<8x1x32x64xbf16>, tensor<2x4xi64>, tensor<2xi64>
    // CHECK-SAME: -> tensor<1x2x16x64xbf16>
    %0 = stablehlo.custom_call @tt.paged_flash_mla_decode(%query, %key, %page_table, %cur_pos) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", has_attention_mask = "False", has_attention_sink = "False", has_cur_pos_tensor = "True", has_value = "False", is_causal = "True"}} : (tensor<1x4x16x64xbf16>, tensor<8x1x32x64xbf16>, tensor<4x4xi64>, tensor<4xi64>) -> tensor<1x4x16x64xbf16>
    return %0 : tensor<1x4x16x64xbf16>
  }
}

// -----

// Combined head + batch sharding on a 2D mesh: the query num_users dim is
// sharded over _axis_0 and the query head (nqh) dim over _axis_1. page_table
// and cur_pos follow the num_users sharding, the head dim shards only the
// query/output, and the KV cache stays fully replicated. 4 users -> 2 and
// 16 heads -> 8 per device.
module @PagedFlashMlaDecodeBatchAndHeadSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2, \22_axis_1\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x4x16x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {\22_axis_1\22}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %key: tensor<8x1x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %value: tensor<8x1x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %page_table: tensor<4x4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}, %cur_pos: tensor<4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}]>"}, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_4"}) -> tensor<1x4x16x64xbf16> {
    // CHECK: stablehlo.custom_call @tt.paged_flash_mla_decode
    // CHECK-SAME: tensor<1x2x8x64xbf16>, tensor<8x1x32x64xbf16>, tensor<8x1x32x64xbf16>, tensor<2x4xi64>, tensor<2xi64>
    // CHECK-SAME: -> tensor<1x2x8x64xbf16>
    %0 = stablehlo.custom_call @tt.paged_flash_mla_decode(%query, %key, %value, %page_table, %cur_pos) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", has_attention_mask = "False", has_attention_sink = "False", has_cur_pos_tensor = "True", has_value = "True", is_causal = "True"}} : (tensor<1x4x16x64xbf16>, tensor<8x1x32x64xbf16>, tensor<8x1x32x64xbf16>, tensor<4x4xi64>, tensor<4xi64>) -> tensor<1x4x16x64xbf16>
    return %0 : tensor<1x4x16x64xbf16>
  }
}
