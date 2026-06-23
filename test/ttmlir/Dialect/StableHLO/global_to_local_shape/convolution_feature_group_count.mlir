// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --split-input-file --update-global-to-local-shapes -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module @jit_depthwise_conv attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["_axis_0"=2]>
  func.func @main(%arg0: tensor<1x64x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x8x8xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg1: tensor<64x1x3x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x1x3x3xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<1x64x6x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}, {}, {}]>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x6x6xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}, {"_axis_0"}, {}, {}]>, <@mesh, [{"_axis_0"}, {}, {}, {}]>] out_shardings=[<@mesh, [{}, {"_axis_0"}, {}, {}]>] manual_axes={} (%arg2: tensor<1x64x8x8xf32>, %arg3: tensor<64x1x3x3xf32>) {
      // The feature_group_count is rescaled from the global value (64) to the
      // local value (32), and the operand/result shapes are localized.
      // CHECK: stablehlo.convolution
      // CHECK-SAME: feature_group_count = 32 : i64
      // CHECK-SAME: (tensor<1x32x8x8xf32>, tensor<32x1x3x3xf32>) -> tensor<1x32x6x6xf32>
      %1 = stablehlo.convolution(%arg2, %arg3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 64 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"_axis_0"}, {}, {}]>]>} : (tensor<1x64x8x8xf32>, tensor<64x1x3x3xf32>) -> tensor<1x64x6x6xf32>
      sdy.return %1 : tensor<1x64x6x6xf32>
    } : (tensor<1x64x8x8xf32>, tensor<64x1x3x3xf32>) -> tensor<1x64x6x6xf32>
    return %0 : tensor<1x64x6x6xf32>
  }
}
