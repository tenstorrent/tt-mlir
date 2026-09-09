// REQUIRES: stablehlo
// RUN: ttmlir-opt --replicate-non-splittable-values %s | FileCheck %s

// A strided slice [::2] over a sharded dimension: with start < stride every
// shard extracts the same pattern from its own data, which localizes correctly
// by adjusting only the limit, so sharding is unchanged.
module @ShardedSliceStrided attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=4]>
  func.func @main(%arg0: tensor<2x1x48xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"_axis_0"}]>}) -> tensor<2x1x24xf32> {
    // CHECK: stablehlo.slice
    // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"_axis_0", ?}]>]>
    %0 = stablehlo.slice %arg0 [0:2, 0:1, 0:48:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"_axis_0", ?}]>]>} : (tensor<2x1x48xf32>) -> tensor<2x1x24xf32>
    return %0 : tensor<2x1x24xf32>
  }
}
