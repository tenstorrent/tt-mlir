// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module @PagedSDPADecodeHeadSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %arg1: tensor<4x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg2: tensor<4x8x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %arg3: tensor<4x8x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}, %arg4: tensor<1x4x8x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_4"}) -> tensor<1x4x8x16xbf16> {
    %0 = stablehlo.reshape %arg1 : (tensor<4x2xi64>) -> tensor<1x4x2xi64>
    %1 = stablehlo.reshape %0 : (tensor<1x4x2xi64>) -> tensor<4x2xi64>
    %2 = stablehlo.reshape %arg0 : (tensor<4xi64>) -> tensor<1x1x4xi64>
    %3 = stablehlo.reshape %2 : (tensor<1x1x4xi64>) -> tensor<4xi64>
    // CHECK: stablehlo.custom_call @tt.paged_scaled_dot_product_attention_decode
    // CHECK-SAME: tensor<1x4x4x16xbf16>, tensor<4x4x32x16xbf16>, tensor<4x4x32x16xbf16>
    // CHECK-SAME: -> tensor<1x4x4x16xbf16>
    %4 = stablehlo.custom_call @tt.paged_scaled_dot_product_attention_decode(%arg4, %arg3, %arg2, %1, %3) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_mask = "False", has_attention_sink = "False", has_cur_pos_tensor = "True", is_causal = "True", scale = "1.0"}} : (tensor<1x4x8x16xbf16>, tensor<4x8x32x16xbf16>, tensor<4x8x32x16xbf16>, tensor<4x2xi64>, tensor<4xi64>) -> tensor<1x4x8x16xbf16>
    return %4 : tensor<1x4x8x16xbf16>
  }
}

// -----

module @PagedSDPADecodeNonHeadShardingOnV attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<4xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}, %arg1: tensor<4x2xi64> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg2: tensor<4x8x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %arg3: tensor<4x8x32x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_3"}, %arg4: tensor<1x4x8x16xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_4"}) -> tensor<1x4x8x16xbf16> {
    %0 = stablehlo.reshape %arg1 : (tensor<4x2xi64>) -> tensor<1x4x2xi64>
    %1 = stablehlo.reshape %0 : (tensor<1x4x2xi64>) -> tensor<4x2xi64>
    %2 = stablehlo.reshape %arg0 : (tensor<4xi64>) -> tensor<1x1x4xi64>
    %3 = stablehlo.reshape %2 : (tensor<1x1x4xi64>) -> tensor<4xi64>
    // CHECK: stablehlo.all_to_all
    // CHECK-SAME: tensor<4x8x16x16xbf16>
    // CHECK-SAME: -> tensor<4x4x32x16xbf16>
    // CHECK: stablehlo.custom_call @tt.paged_scaled_dot_product_attention_decode
    // CHECK-SAME: tensor<1x4x4x16xbf16>, tensor<4x4x32x16xbf16>, tensor<4x4x32x16xbf16>
    // CHECK-SAME: -> tensor<1x4x4x16xbf16>
    %4 = stablehlo.custom_call @tt.paged_scaled_dot_product_attention_decode(%arg4, %arg3, %arg2, %1, %3) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_mask = "False", has_attention_sink = "False", has_cur_pos_tensor = "True", is_causal = "True", scale = "1.0"}} : (tensor<1x4x8x16xbf16>, tensor<4x8x32x16xbf16>, tensor<4x8x32x16xbf16>, tensor<4x2xi64>, tensor<4xi64>) -> tensor<1x4x8x16xbf16>
    return %4 : tensor<1x4x8x16xbf16>
  }
}
