// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --split-input-file --update-global-to-local-shapes -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Test that out_sharding from all_gather is used to determine local shapes
// for downstream ops (e.g. slice indices).
module @SyncTensorsGraph.6 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<32x2880x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {"_axis_1"}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<16x720x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<i64> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i64>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<2880x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1", ?}, {?}]>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<720x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"_axis_0"}, {"_axis_1"}, {}]>] out_shardings=[<@mesh, []>, <@mesh, [{"_axis_1", ?}, {?}]>] manual_axes={} (%arg1: tensor<32x2880x2880xbf16>) {
      // CHECK: stablehlo.slice
      // CHECK-SAME: [0:1, 0:720, 0:2880]
      %c = stablehlo.constant dense<0> : tensor<i64>
      %1 = sdy.all_gather [{"_axis_0"}, {}, {}] %arg1 out_sharding=<@mesh, [{}, {"_axis_1"}, {}]> : tensor<32x2880x2880xbf16>
      %2 = stablehlo.slice %1 [0:1, 0:2880, 0:2880] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"_axis_1", ?}, {?}]>]>} : (tensor<32x2880x2880xbf16>) -> tensor<1x2880x2880xbf16>
      %3 = stablehlo.reshape %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_1", ?}, {?}]>]>} : (tensor<1x2880x2880xbf16>) -> tensor<2880x2880xbf16>
      sdy.return %c, %3 : tensor<i64>, tensor<2880x2880xbf16>
    } : (tensor<32x2880x2880xbf16>) -> (tensor<i64>, tensor<2880x2880xbf16>)
    return %0#0, %0#1 : tensor<i64>, tensor<2880x2880xbf16>
  }
}
