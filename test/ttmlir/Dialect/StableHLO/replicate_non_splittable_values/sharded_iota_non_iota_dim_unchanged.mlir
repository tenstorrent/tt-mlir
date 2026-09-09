// REQUIRES: stablehlo
// RUN: ttmlir-opt --replicate-non-splittable-values %s | FileCheck %s

// A 2D stablehlo.iota with iota_dimension = 1 sharded only on dim 0 (_axis_0).
// The values are constant (broadcast) along dim 0, so the sharded axis is
// periodic and localizes correctly by shrinking the shape: the pass must leave
// the iota's sharding unchanged.
module @ShardedIotaNonIotaDim attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>}) -> tensor<64x64xf32> {
    // Iota over an unsharded axis: sharding must stay as-is (dim 0 sharded).
    // CHECK: stablehlo.iota
    // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>
    %0 = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : tensor<64x64xf32>
    %1 = stablehlo.add %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }
}
