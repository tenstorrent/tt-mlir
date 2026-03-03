// REQUIRES: stablehlo
// RUN: ttmlir-opt --replicate-non-splittable-constants %s | FileCheck %s

// 3D non-periodic constant sharded on dim 0 and dim 2 (dim 1 replicated).
// Periodic on dim 2 but NOT on dim 0, so the pass must replicate the constant.
module @MultiDimNonPeriodicReplicated attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<4x3x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {"_axis_1"}]>}) -> tensor<4x3x8xf32> {
    // CHECK: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>
    // CHECK-SAME: tensor<4x3x8xf32>
    %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}, {"_axis_1", ?}]>]>} dense<[[[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0], [5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0]], [[7.0, 8.0, 7.0, 8.0, 7.0, 8.0, 7.0, 8.0], [9.0, 10.0, 9.0, 10.0, 9.0, 10.0, 9.0, 10.0], [11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0]], [[13.0, 14.0, 13.0, 14.0, 13.0, 14.0, 13.0, 14.0], [15.0, 16.0, 15.0, 16.0, 15.0, 16.0, 15.0, 16.0], [17.0, 18.0, 17.0, 18.0, 17.0, 18.0, 17.0, 18.0]], [[19.0, 20.0, 19.0, 20.0, 19.0, 20.0, 19.0, 20.0], [21.0, 22.0, 21.0, 22.0, 21.0, 22.0, 21.0, 22.0], [23.0, 24.0, 23.0, 24.0, 23.0, 24.0, 23.0, 24.0]]]> : tensor<4x3x8xf32>
    %0 = stablehlo.add %arg0, %cst {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}, {"_axis_1", ?}]>]>} : tensor<4x3x8xf32>
    return %0 : tensor<4x3x8xf32>
  }
}
