// REQUIRES: stablehlo
// RUN: ttmlir-opt --replicate-non-splittable-constants %s | FileCheck %s

// Periodic constant: the 8-element pattern [0,1,2,3,4,5,6,7] repeats 8 times
// across 64 elements. Sharded on _axis_0 (2 devices), each shard of 32 elements
// is identical. The pass must leave its sharding unchanged.
module @PeriodicConstantUnchanged attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x64xf32>>}) -> (tensor<f32> {ttcore.shard_status = #ttcore.shard_status<unsharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<f32>>}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    // Periodic: [0,1,2,3,4,5,6,7] repeated 8 times. Each shard gets the same slice.
    // CHECK: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}]>]>
    // CHECK-SAME: tensor<64xf32>
    %cst_0 = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}]>]>} dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<64xf32>
    %0 = stablehlo.broadcast_in_dim %cst_0, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : (tensor<64xf32>) -> tensor<64x64xf32>
    %1 = stablehlo.add %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : tensor<64x64xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<64x64xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
}
