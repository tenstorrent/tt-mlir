// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module @PagedSDPADecodeQKVHeadSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %arg1: tensor<2x4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg2: tensor<8x12x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %arg3: tensor<8x12x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}, %arg4: tensor<1x2x12x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_4"}) -> tensor<1x2x12x16xbf16> {
    %0 = stablehlo.reshape %arg1 : (tensor<2x4xi64>) -> tensor<1x2x4xi64>
    %1 = stablehlo.reshape %0 : (tensor<1x2x4xi64>) -> tensor<2x4xi64>
    %2 = stablehlo.reshape %arg0 : (tensor<2xi64>) -> tensor<1x1x2xi64>
    %3 = stablehlo.reshape %2 : (tensor<1x1x2xi64>) -> tensor<2xi64>
    // CHECK: stablehlo.custom_call @tt.paged_scaled_dot_product_attention_decode
    // CHECK-SAME: tensor<1x2x6x16xbf16>, tensor<8x6x32x16xbf16>, tensor<8x6x32x16xbf16>, tensor<2x4xi64>, tensor<2xi64>
    // CHECK-SAME: -> tensor<1x2x6x16xbf16>
    %4 = stablehlo.custom_call @tt.paged_scaled_dot_product_attention_decode(%arg4, %arg3, %arg2, %1, %3) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_mask = "False", has_attention_sink = "False", has_cur_pos_tensor = "True", is_causal = "True", scale = "1.0"}} : (tensor<1x2x12x16xbf16>, tensor<8x12x32x16xbf16>, tensor<8x12x32x16xbf16>, tensor<2x4xi64>, tensor<2xi64>) -> tensor<1x2x12x16xbf16>
    return %4 : tensor<1x2x12x16xbf16>
  }
}

// -----

module @PagedSDPADecodeQUserSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %arg1: tensor<2x4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg2: tensor<8x12x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %arg3: tensor<8x12x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}, %arg4: tensor<1x2x12x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_4"}) -> tensor<1x2x12x16xbf16> {
    %0 = stablehlo.reshape %arg1 : (tensor<2x4xi64>) -> tensor<1x2x4xi64>
    %1 = stablehlo.reshape %0 : (tensor<1x2x4xi64>) -> tensor<2x4xi64>
    %2 = stablehlo.reshape %arg0 : (tensor<2xi64>) -> tensor<1x1x2xi64>
    %3 = stablehlo.reshape %2 : (tensor<1x1x2xi64>) -> tensor<2xi64>
    // CHECK: stablehlo.all_to_all
    // CHECK-SAME: (tensor<1x1x12x16xbf16>)
    // CHECK-SAME: -> tensor<1x2x6x16xbf16>
    // CHECK: stablehlo.custom_call @tt.paged_scaled_dot_product_attention_decode
    // CHECK-SAME: tensor<1x2x6x16xbf16>, tensor<8x6x32x16xbf16>, tensor<8x6x32x16xbf16>, tensor<2x4xi64>, tensor<2xi64>
    // CHECK-SAME: -> tensor<1x2x6x16xbf16>
    %4 = stablehlo.custom_call @tt.paged_scaled_dot_product_attention_decode(%arg4, %arg3, %arg2, %1, %3) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_mask = "False", has_attention_sink = "False", has_cur_pos_tensor = "True", is_causal = "True", scale = "1.0"}} : (tensor<1x2x12x16xbf16>, tensor<8x12x32x16xbf16>, tensor<8x12x32x16xbf16>, tensor<2x4xi64>, tensor<2xi64>) -> tensor<1x2x12x16xbf16>
    return %4 : tensor<1x2x12x16xbf16>
  }
}

// -----

module @PagedUpdateCacheHeadSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %arg1: tensor<2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg2: tensor<1x2x12x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %arg3: tensor<2x4x12x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}) -> tensor<4x12x32x16xbf16> {
    %0 = stablehlo.slice %arg3 [1:2, 0:4, 0:12, 0:32, 0:16] : (tensor<2x4x12x32x16xbf16>) -> tensor<1x4x12x32x16xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x4x12x32x16xbf16>) -> tensor<4x12x32x16xbf16>
    %2 = stablehlo.reshape %arg1 : (tensor<2xi64>) -> tensor<1x1x2xi64>
    %3 = stablehlo.reshape %2 : (tensor<1x1x2xi64>) -> tensor<2xi64>
    %4 = stablehlo.reshape %arg0 : (tensor<2x2xi64>) -> tensor<1x2x2xi64>
    %5 = stablehlo.reshape %4 : (tensor<1x2x2xi64>) -> tensor<2x2xi64>
    // CHECK: stablehlo.custom_call @tt.paged_update_cache
    // CHECK-SAME: tensor<4x6x32x16xbf16>, tensor<1x2x6x16xbf16>, tensor<2xi64>, tensor<2x2xi64>
    // CHECK-SAME: -> tensor<4x6x32x16xbf16>
    %6 = stablehlo.custom_call @tt.paged_update_cache(%1, %arg2, %3, %5) {api_version = 0 : i32, mhlo.frontend_attributes = {share_cache = "False"}} : (tensor<4x12x32x16xbf16>, tensor<1x2x12x16xbf16>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<4x12x32x16xbf16>
    return %6 : tensor<4x12x32x16xbf16>
  }
}

// -----

module @PagedUpdateCacheFillUserSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %arg1: tensor<2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg2: tensor<1x2x12x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %arg3: tensor<2x4x12x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}) -> tensor<4x12x32x16xbf16> {
    %0 = stablehlo.slice %arg3 [1:2, 0:4, 0:12, 0:32, 0:16] : (tensor<2x4x12x32x16xbf16>) -> tensor<1x4x12x32x16xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x4x12x32x16xbf16>) -> tensor<4x12x32x16xbf16>
    %2 = stablehlo.reshape %arg1 : (tensor<2xi64>) -> tensor<1x1x2xi64>
    %3 = stablehlo.reshape %2 : (tensor<1x1x2xi64>) -> tensor<2xi64>
    %4 = stablehlo.reshape %arg0 : (tensor<2x2xi64>) -> tensor<1x2x2xi64>
    %5 = stablehlo.reshape %4 : (tensor<1x2x2xi64>) -> tensor<2x2xi64>
    // CHECK: stablehlo.all_to_all
    // CHECK: stablehlo.custom_call @tt.paged_update_cache
    // CHECK-SAME: tensor<4x6x32x16xbf16>, tensor<1x2x6x16xbf16>, tensor<2xi64>, tensor<2x2xi64>
    // CHECK-SAME: -> tensor<4x6x32x16xbf16>
    %6 = stablehlo.custom_call @tt.paged_update_cache(%1, %arg2, %3, %5) {api_version = 0 : i32, mhlo.frontend_attributes = {share_cache = "False"}} : (tensor<4x12x32x16xbf16>, tensor<1x2x12x16xbf16>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<4x12x32x16xbf16>
    return %6 : tensor<4x12x32x16xbf16>
  }
}

// -----

module @PagedFillCacheHeadSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<2x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %arg2: tensor<1x16x64x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg3: tensor<4x16x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}) -> tensor<4x16x32x16xbf16> {
    %0 = stablehlo.reshape %arg1 : (tensor<2x2xi64>) -> tensor<1x2x2xi64>
    %1 = stablehlo.reshape %0 : (tensor<1x2x2xi64>) -> tensor<2x2xi64>
    // CHECK: stablehlo.custom_call @tt.paged_fill_cache
    // CHECK-SAME: tensor<4x8x32x16xbf16>, tensor<1x8x64x16xbf16>, tensor<2x2xi64>, tensor<1xi32>
    // CHECK-SAME: -> tensor<4x8x32x16xbf16>
    %2 = stablehlo.custom_call @tt.paged_fill_cache(%arg3, %arg2, %1, %arg0) {api_version = 0 : i32} : (tensor<4x16x32x16xbf16>, tensor<1x16x64x16xbf16>, tensor<2x2xi64>, tensor<1xi32>) -> tensor<4x16x32x16xbf16>
    return %2 : tensor<4x16x32x16xbf16>
  }
}

// -----

module @PagedFillCacheSeqlenSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<2x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %arg2: tensor<1x16x64x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg3: tensor<4x16x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}) -> tensor<4x16x32x16xbf16> {
    %0 = stablehlo.reshape %arg1 : (tensor<2x2xi64>) -> tensor<1x2x2xi64>
    %1 = stablehlo.reshape %0 : (tensor<1x2x2xi64>) -> tensor<2x2xi64>
    // CHECK: stablehlo.all_to_all
    // CHECK: stablehlo.custom_call @tt.paged_fill_cache
    // CHECK-SAME: tensor<4x8x32x16xbf16>, tensor<1x8x64x16xbf16>, tensor<2x2xi64>, tensor<1xi32>
    // CHECK-SAME: -> tensor<4x8x32x16xbf16>
    %2 = stablehlo.custom_call @tt.paged_fill_cache(%arg3, %arg2, %1, %arg0) {api_version = 0 : i32} : (tensor<4x16x32x16xbf16>, tensor<1x16x64x16xbf16>, tensor<2x2xi64>, tensor<1xi32>) -> tensor<4x16x32x16xbf16>
    return %2 : tensor<4x16x32x16xbf16>
  }
}

// -----

// When an attention_mask is present (has_attention_mask = "True"), the optional
// cur_pos_tensor shifts from operand index 4 to index 5. The users factor must
// still attach to cur_pos (and page_table), NOT to the attention_mask: the mask
// stays replicated (full tensor<2x12x1x32xbf16>) while the head dim shards
// 12 -> 6. This guards the operand-index logic in getPagedAttentionShardingRule.
module @PagedSDPADecodeWithMaskQUserSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %arg1: tensor<2x4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg2: tensor<8x12x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %arg3: tensor<8x12x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}, %arg4: tensor<1x2x12x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_4"}, %arg5: tensor<2x12x1x32xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_5"}) -> tensor<1x2x12x16xbf16> {
    %0 = stablehlo.reshape %arg1 : (tensor<2x4xi64>) -> tensor<1x2x4xi64>
    %1 = stablehlo.reshape %0 : (tensor<1x2x4xi64>) -> tensor<2x4xi64>
    %2 = stablehlo.reshape %arg0 : (tensor<2xi64>) -> tensor<1x1x2xi64>
    %3 = stablehlo.reshape %2 : (tensor<1x1x2xi64>) -> tensor<2xi64>
    // CHECK: stablehlo.custom_call @tt.paged_scaled_dot_product_attention_decode
    // CHECK-SAME: tensor<1x2x6x16xbf16>, tensor<8x6x32x16xbf16>, tensor<8x6x32x16xbf16>, tensor<2x4xi64>, tensor<2x12x1x32xbf16>, tensor<2xi64>
    // CHECK-SAME: -> tensor<1x2x6x16xbf16>
    %4 = stablehlo.custom_call @tt.paged_scaled_dot_product_attention_decode(%arg4, %arg3, %arg2, %1, %arg5, %3) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_mask = "True", has_attention_sink = "False", has_cur_pos_tensor = "True", is_causal = "False", scale = "1.0"}} : (tensor<1x2x12x16xbf16>, tensor<8x12x32x16xbf16>, tensor<8x12x32x16xbf16>, tensor<2x4xi64>, tensor<2x12x1x32xbf16>, tensor<2xi64>) -> tensor<1x2x12x16xbf16>
    return %4 : tensor<1x2x12x16xbf16>
  }
}

// -----

module @ChunkedSDPAQKVHeadSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<8x4x64x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %arg1: tensor<32x4x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %arg2: tensor<32x4x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}, %arg3: tensor<8x4xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "page_table"}, %arg4: tensor<1xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "chunk_start_idx"}) -> tensor<8x4x64x16xbf16> {
    // The head factor carries dim 1 of query, key and value to the result.
    // CHECK: stablehlo.custom_call @tt.chunked_scaled_dot_product_attention
    // CHECK-SAME: tensor<8x2x64x16xbf16>, tensor<32x2x32x16xbf16>, tensor<32x2x32x16xbf16>, tensor<8x4xi32>, tensor<1xi32>
    // CHECK-SAME: -> tensor<8x2x64x16xbf16>
    %0 = stablehlo.custom_call @tt.chunked_scaled_dot_product_attention(%arg0, %arg1, %arg2, %arg3, %arg4) {api_version = 0 : i32, mhlo.frontend_attributes = {scale = "0.25"}} : (tensor<8x4x64x16xbf16>, tensor<32x4x32x16xbf16>, tensor<32x4x32x16xbf16>, tensor<8x4xi32>, tensor<1xi32>) -> tensor<8x4x64x16xbf16>
    return %0 : tensor<8x4x64x16xbf16>
  }
}

// -----

module @ChunkedSDPAQUserSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<8x4x64x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %arg1: tensor<32x2x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %arg2: tensor<32x2x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}, %arg3: tensor<8x4xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "page_table"}, %arg4: tensor<1xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "chunk_start_idx"}) -> tensor<8x4x64x16xbf16> {
    // The users factor carries the query's dim-0 sharding to the result, so both
    // stay at 4 users per shard. K/V dim 0 is the block pool and stays replicated.
    // The sharding matches on both sides, so no resharding collective is needed
    // anywhere in the body. CHECK-LABEL and the trailing sdy.return bound the
    // negative checks to this module.
    // CHECK-LABEL: module @ChunkedSDPAQUserSharding
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_to_all
    // CHECK: stablehlo.custom_call @tt.chunked_scaled_dot_product_attention
    // CHECK-SAME: tensor<4x4x64x16xbf16>, tensor<32x2x32x16xbf16>, tensor<32x2x32x16xbf16>, tensor<4x4xi32>, tensor<1xi32>
    // CHECK-SAME: -> tensor<4x4x64x16xbf16>
    // CHECK-NOT: stablehlo.all_gather
    // CHECK-NOT: stablehlo.all_to_all
    // CHECK: sdy.return
    %0 = stablehlo.custom_call @tt.chunked_scaled_dot_product_attention(%arg0, %arg1, %arg2, %arg3, %arg4) {api_version = 0 : i32, mhlo.frontend_attributes = {scale = "0.25"}} : (tensor<8x4x64x16xbf16>, tensor<32x2x32x16xbf16>, tensor<32x2x32x16xbf16>, tensor<8x4xi32>, tensor<1xi32>) -> tensor<8x4x64x16xbf16>
    return %0 : tensor<8x4x64x16xbf16>
  }
}
