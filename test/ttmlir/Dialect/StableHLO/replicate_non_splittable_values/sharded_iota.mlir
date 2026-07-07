// REQUIRES: stablehlo
// RUN: ttmlir-opt --replicate-non-splittable-values %s | FileCheck %s

// A stablehlo.iota over its iota_dimension (dim 0) sharded on _axis_0 (2
// devices) is non-splat and non-periodic: pass must change its sharding to
// replicated so each shard later gets the correct per-shard offset.
module @ShardedIota attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>}) -> tensor<64x64xf32> {
    // Iota over the sharded axis: sharding must become fully replicated.
    // CHECK: stablehlo.iota
    // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>
    %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}]>]>} : tensor<64xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : (tensor<64xf32>) -> tensor<64x64xf32>
    %2 = stablehlo.add %arg0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : tensor<64x64xf32>
    return %2 : tensor<64x64xf32>
  }
}
