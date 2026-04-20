// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --update-global-to-local-shapes -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<1xf32> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{}]>] manual_axes={"_axis_0"} (%arg1: tensor<1xf32>) {
      // CHECK: stablehlo.all_reduce
      %used_result = sdy.all_reduce {"_axis_0"} %arg1 out_sharding=<@mesh, [{}]> : tensor<1xf32>
      // CHECK-NOT: sdy.all_reduce
      // CHECK-NOT: stablehlo.all_reduce
      %unused_result = sdy.all_reduce {"_axis_0"} %arg1 out_sharding=<@mesh, [{}]> : tensor<1xf32>
      sdy.return %used_result : tensor<1xf32>
    } : (tensor<1xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
