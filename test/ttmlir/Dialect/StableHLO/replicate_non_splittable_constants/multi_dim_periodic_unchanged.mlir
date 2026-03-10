// REQUIRES: stablehlo
// RUN: ttmlir-opt --replicate-non-splittable-constants %s | FileCheck %s

// 3D periodic constant sharded on dim 0 and dim 2 (dim 1 replicated).
// The 2x3x2 base block tiles exactly, so the pass must leave sharding unchanged.
module @MultiDimPeriodicUnchanged attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<4x3x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {"_axis_1"}]>}) -> tensor<4x3x8xf32> {
    // CHECK: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}, {"_axis_1", ?}]>]>
    // CHECK-SAME: tensor<4x3x8xf32>
    %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}, {"_axis_1", ?}]>]>} dense<[[[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0], [5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0]], [[7.0, 8.0, 7.0, 8.0, 7.0, 8.0, 7.0, 8.0], [9.0, 10.0, 9.0, 10.0, 9.0, 10.0, 9.0, 10.0], [11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0]], [[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0], [5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0]], [[7.0, 8.0, 7.0, 8.0, 7.0, 8.0, 7.0, 8.0], [9.0, 10.0, 9.0, 10.0, 9.0, 10.0, 9.0, 10.0], [11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0]]]> : tensor<4x3x8xf32>
    %0 = stablehlo.add %arg0, %cst {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}, {"_axis_1", ?}]>]>} : tensor<4x3x8xf32>
    return %0 : tensor<4x3x8xf32>
  }
}
