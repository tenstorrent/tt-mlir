// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: module @ReplicateShardedData.6
module @ReplicateShardedData.6 attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  // CHECK: sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  // CHECK: func.func @main(%arg0: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>})
  // CHECK-SAME: -> (tensor<4096x4096xf32> {mhlo.sharding = "{replicated}", sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>, ttcore.shard_status = #ttcore.shard_status<presharded>})
  // CHECK: return %arg0 : tensor<4096x4096xf32>
  func.func @main(%arg0: tensor<4096x4096xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>} loc("p0.1")) -> (tensor<4096x4096xf32> {mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"}} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32> loc(#loc1)
    return %0 : tensor<4096x4096xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("jit(f)/jit(main)/sharding_constraint")
