// REQUIRES: stablehlo
// RUN: ttmlir-opt --replicate-non-splittable-values %s | FileCheck %s

// A strided slice that stops before the end of a sharded dim: the shards no
// longer tile the result (the slice ends inside shard 0), so dividing the limit
// is invalid and the sharding must become replicated.
module @ShardedSliceStridedPartial attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=4]>
  func.func @main(%arg0: tensor<2x1x48xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"_axis_0"}]>}) -> tensor<2x1x12xf32> {
    // CHECK: stablehlo.slice
    // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>
    %0 = stablehlo.slice %arg0 [0:2, 0:1, 0:24:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"_axis_0", ?}]>]>} : (tensor<2x1x48xf32>) -> tensor<2x1x12xf32>
    return %0 : tensor<2x1x12xf32>
  }
}
