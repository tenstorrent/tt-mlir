// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module @SDPA_Sharding_Head attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x12x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg1: tensor<1x12x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %arg2: tensor<1x12x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}) -> tensor<1x12x32x128xbf16> {
    // CHECK: stablehlo.custom_call @tt.scaled_dot_product_attention
    // CHECK-SAME: tensor<1x6x32x128xbf16>, tensor<1x6x32x128xbf16>, tensor<1x6x32x128xbf16>
    // CHECK-SAME: -> tensor<1x6x32x128xbf16>
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%arg2, %arg1, %arg0) {api_version = 0 : i32, mhlo.frontend_attributes = {is_causal = "True"}} : (tensor<1x12x32x128xbf16>, tensor<1x12x32x128xbf16>, tensor<1x12x32x128xbf16>) -> tensor<1x12x32x128xbf16>
    return %0 : tensor<1x12x32x128xbf16>
  }
}

// -----

// Test: Multi-Query Attention (MQA) with different Q/K/V head counts
module @SDPA_MQA_Sharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x32x64x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %arg1: tensor<1x4x64x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %arg2: tensor<1x4x64x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<1x32x64x128xbf16> {
    // CHECK: stablehlo.custom_call @tt.scaled_dot_product_attention
    // CHECK-SAME: tensor<1x16x64x128xbf16>, tensor<1x2x64x128xbf16>, tensor<1x2x64x128xbf16>
    // CHECK-SAME: -> tensor<1x16x64x128xbf16>
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {is_causal = "True", scale = "0.088388"}} : (tensor<1x32x64x128xbf16>, tensor<1x4x64x128xbf16>, tensor<1x4x64x128xbf16>) -> tensor<1x32x64x128xbf16>
    return %0 : tensor<1x32x64x128xbf16>
  }
}

// -----

// Test: SDPA with Attention Mask (4 operands)
module @SDPA_AttentionMask attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=4]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x16x128x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,4,1,1]<=[4]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %arg1: tensor<2x16x128x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,4,1,1]<=[4]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %arg2: tensor<2x16x128x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,4,1,1]<=[4]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}, %arg3: tensor<2x1x128x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "attention_mask"}) -> tensor<2x16x128x64xbf16> {
    // CHECK: stablehlo.custom_call @tt.scaled_dot_product_attention
    // CHECK-SAME: tensor<2x4x128x64xbf16>, tensor<2x4x128x64xbf16>, tensor<2x4x128x64xbf16>, tensor<2x1x128x128xbf16>
    // CHECK-SAME: -> tensor<2x4x128x64xbf16>
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%arg0, %arg1, %arg2, %arg3) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_mask = "True", is_causal = "False"}} : (tensor<2x16x128x64xbf16>, tensor<2x16x128x64xbf16>, tensor<2x16x128x64xbf16>, tensor<2x1x128x128xbf16>) -> tensor<2x16x128x64xbf16>
    return %0 : tensor<2x16x128x64xbf16>
  }
}

// -----

// Test: MQA with Attention Sink
module @SDPA_MQA_AttentionSink attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x64x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %arg1: tensor<1x8x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %arg2: tensor<1x8x32x64xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}, %arg3: tensor<64x1xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "attention_sink"}) -> tensor<1x64x32x64xbf16> {
    // CHECK: stablehlo.custom_call @tt.scaled_dot_product_attention
    // CHECK-SAME: tensor<1x32x32x64xbf16>, tensor<1x4x32x64xbf16>, tensor<1x4x32x64xbf16>, tensor<32x1xbf16>
    // CHECK-SAME: -> tensor<1x32x32x64xbf16>
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%arg0, %arg1, %arg2, %arg3) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_mask = "False", has_attention_sink = "True", is_causal = "True"}} : (tensor<1x64x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<64x1xbf16>) -> tensor<1x64x32x64xbf16>
    return %0 : tensor<1x64x32x64xbf16>
  }
}

// -----

// Test: Higher sharding factor (8x mesh)
module @SDPA_HighSharding attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=8]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x64x256x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,8,1,1]<=[8]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %arg1: tensor<1x8x256x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,8,1,1]<=[8]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %arg2: tensor<1x8x256x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,8,1,1]<=[8]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<1x64x256x128xbf16> {
    // CHECK: stablehlo.custom_call @tt.scaled_dot_product_attention
    // CHECK-SAME: tensor<1x8x256x128xbf16>, tensor<1x1x256x128xbf16>, tensor<1x1x256x128xbf16>
    // CHECK-SAME: -> tensor<1x8x256x128xbf16>
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {is_causal = "True", scale = "0.088388"}} : (tensor<1x64x256x128xbf16>, tensor<1x8x256x128xbf16>, tensor<1x8x256x128xbf16>) -> tensor<1x64x256x128xbf16>
    return %0 : tensor<1x64x256x128xbf16>
  }
}
