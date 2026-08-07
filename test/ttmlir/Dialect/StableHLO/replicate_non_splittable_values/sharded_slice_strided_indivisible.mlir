// REQUIRES: stablehlo
// RUN: ttmlir-opt --replicate-non-splittable-values %s | FileCheck %s

// A strided slice whose stride does not divide the per-shard width (48/4 = 12,
// stride 8): shard d's first match is at local (start - d*12) mod 8, not start,
// so keeping start unchanged is invalid and the sharding must become replicated.
module @ShardedSliceStridedIndivisible attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=4]>
  func.func @main(%arg0: tensor<2x1x48xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"_axis_0"}]>}) -> tensor<2x1x6xf32> {
    // CHECK: stablehlo.slice
    // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>
    %0 = stablehlo.slice %arg0 [0:2, 0:1, 0:48:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"_axis_0", ?}]>]>} : (tensor<2x1x48xf32>) -> tensor<2x1x6xf32>
    return %0 : tensor<2x1x6xf32>
  }
}
