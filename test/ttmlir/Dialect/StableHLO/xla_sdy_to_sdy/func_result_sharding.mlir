// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: module @ReplicateShardedData
module @ReplicateShardedData attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  // CHECK: sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  // CHECK: sdy.all_gather [{}, {"_axis_0"}] %{{.*}} out_sharding=<@mesh, [{}, {}]>
  func.func @main(%arg0: tensor<4096x4096xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>}) -> (tensor<4096x4096xf32> {mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"}} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }
}

// -----

// CHECK-LABEL: module @ReshardToDifferentDim
module @ReshardToDifferentDim attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  // CHECK: sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  // CHECK: sdy.all_to_all [{"_axis_0"}: 1->0] %{{.*}} out_sharding=<@mesh, [{"_axis_0"}, {}]>
  func.func @main(%arg0: tensor<4096x4096xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>}) -> (tensor<4096x4096xf32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
    %0 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22_axis_0\22}, {}]>]>"}} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }
}
