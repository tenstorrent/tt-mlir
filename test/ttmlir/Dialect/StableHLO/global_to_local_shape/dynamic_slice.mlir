// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --split-input-file --update-global-to-local-shapes -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module @DynamicSliceLocalShape attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["b"=2]>
  func.func @main(%arg0: tensor<4x64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x32x64xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<1x64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}, {}]>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x64xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"b"}, {}]>] out_shardings=[<@mesh, [{}, {"b"}, {}]>] manual_axes={} (%arg1: tensor<4x64x64xf32>) {
      %c0 = stablehlo.constant dense<0> : tensor<i32>
      // CHECK: stablehlo.dynamic_slice
      // CHECK-SAME: sizes = [1, 32, 64]
      // CHECK-SAME: (tensor<4x32x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x32x64xf32>
      %1 = stablehlo.dynamic_slice %arg1, %c0, %c0, %c0, sizes = [1, 64, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}, {}]>]>} : (tensor<4x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x64x64xf32>
      sdy.return %1 : tensor<1x64x64xf32>
    } : (tensor<4x64x64xf32>) -> tensor<1x64x64xf32>
    return %0 : tensor<1x64x64xf32>
  }
}
