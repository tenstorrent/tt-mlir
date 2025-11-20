// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// UNSUPPORTED: true

module @SDPA_Sharding_Head attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x12x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_1"}, %arg1: tensor<1x12x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_2"}, %arg2: tensor<1x12x32x128xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}) -> tensor<1x12x32x128xbf16> {
    // CHECK: stablehlo.custom_call @tt.scaled_dot_product_attention
    // CHECK-SAME: tensor<1x6x32x128xbf16>, tensor<1x6x32x128xbf16>, tensor<1x6x32x128xbf16>
    // CHECK-SAME: -> tensor<1x6x32x128xbf16>
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%arg2, %arg1, %arg0) {api_version = 0 : i32, mhlo.frontend_attributes = {is_causal = "True"}} : (tensor<1x12x32x128xbf16>, tensor<1x12x32x128xbf16>, tensor<1x12x32x128xbf16>) -> tensor<1x12x32x128xbf16>
    return %0 : tensor<1x12x32x128xbf16>
  }
}
