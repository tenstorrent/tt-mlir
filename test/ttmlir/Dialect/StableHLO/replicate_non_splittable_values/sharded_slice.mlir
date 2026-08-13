// REQUIRES: stablehlo
// RUN: ttmlir-opt --replicate-non-splittable-values %s | FileCheck %s

// A slice narrowing a dimension sharded on _axis_0 (4 devices): dividing [0:4]
// by 4 gives [0:1], so shard 1 reads global 12 where it must hold global 1.
// Sharding must become replicated so Shardy gathers before the slice.
module @ShardedSlice attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=4]>
  func.func @main(%arg0: tensor<2x1x48xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"_axis_0"}]>}) -> tensor<2x1x4xf32> {
    // CHECK: stablehlo.slice
    // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>
    %0 = stablehlo.slice %arg0 [0:2, 0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"_axis_0", ?}]>]>} : (tensor<2x1x48xf32>) -> tensor<2x1x4xf32>
    return %0 : tensor<2x1x4xf32>
  }
}
