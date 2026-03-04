// REQUIRES: stablehlo
// RUN: ttmlir-opt --replicate-non-splittable-constants %s | FileCheck %s

// Iota-like constant [0.0, 1.0, ..., 63.0] sharded on _axis_0 (2 devices)
// is non-splat and non-periodic: pass must change its sharding to replicated.
module @SyncTensorsGraph.18 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x64xf32>>}) -> (tensor<f32> {ttcore.shard_status = #ttcore.shard_status<unsharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<f32>>}) {
    // Scalar splat with no sharding annotation: unchanged.
    // CHECK: stablehlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NOT: sdy.sharding
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    // Iota constant: sharding must become fully replicated.
    // CHECK: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>
    // CHECK-SAME: dense<[0.000000e+00, 1.000000e+00,
    %cst_0 = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}]>]>} dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01, 4.000000e+01, 4.100000e+01, 4.200000e+01, 4.300000e+01, 4.400000e+01, 4.500000e+01, 4.600000e+01, 4.700000e+01, 4.800000e+01, 4.900000e+01, 5.000000e+01, 5.100000e+01, 5.200000e+01, 5.300000e+01, 5.400000e+01, 5.500000e+01, 5.600000e+01, 5.700000e+01, 5.800000e+01, 5.900000e+01, 6.000000e+01, 6.100000e+01, 6.200000e+01, 6.300000e+01]> : tensor<64xf32>
    %0 = stablehlo.broadcast_in_dim %cst_0, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : (tensor<64xf32>) -> tensor<64x64xf32>
    %1 = stablehlo.add %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0", ?}, {?}]>]>} : tensor<64x64xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<64x64xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
}
