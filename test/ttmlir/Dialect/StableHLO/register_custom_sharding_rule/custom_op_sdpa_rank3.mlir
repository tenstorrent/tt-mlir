// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: module @SDPA_Rank3_Sharding_Batch
module @SDPA_Rank3_Sharding_Batch attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %arg1: tensor<2x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %arg2: tensor<2x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<2x32x128xbf16> {
    // Batch is kPassThrough, so it shards without any collective.
    // CHECK-NOT: stablehlo.all_gather
    // CHECK: stablehlo.custom_call @tt.scaled_dot_product_attention
    // CHECK-SAME: tensor<1x32x128xbf16>, tensor<1x32x128xbf16>, tensor<1x32x128xbf16>
    // CHECK-SAME: -> tensor<1x32x128xbf16>
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {is_causal = "False"}} : (tensor<2x32x128xbf16>, tensor<2x32x128xbf16>, tensor<2x32x128xbf16>) -> tensor<2x32x128xbf16>
    return %0 : tensor<2x32x128xbf16>
  }
}

// -----

// CHECK-LABEL: module @SDPA_Rank3_Sharding_Sequence
module @SDPA_Rank3_Sharding_Sequence attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %arg1: tensor<1x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}, %arg2: tensor<1x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[1,2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "value"}) -> tensor<1x32x128xbf16> {
    // Sequence must be gathered back to full length before the custom call.
    // CHECK: stablehlo.custom_call @tt.scaled_dot_product_attention
    // CHECK-SAME: tensor<1x32x128xbf16>, tensor<1x32x128xbf16>, tensor<1x32x128xbf16>
    // CHECK-SAME: -> tensor<1x32x128xbf16>
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%arg0, %arg1, %arg2) {api_version = 0 : i32, mhlo.frontend_attributes = {is_causal = "False"}} : (tensor<1x32x128xbf16>, tensor<1x32x128xbf16>, tensor<1x32x128xbf16>) -> tensor<1x32x128xbf16>
    return %0 : tensor<1x32x128xbf16>
  }
}
