// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @SyncTensorsGraph.94 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["x"=1, "y"=1]>
  func.func @main(%arg0: tensor<768xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"}, %arg1: tensor<768xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_1"}, %arg2: tensor<1x1024x768xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_2"}) -> (tensor<1x1024x768xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.reshape %arg1 : (tensor<768xbf16>) -> tensor<1x1x768xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1x1x768xbf16>) -> tensor<768xbf16>
    %2 = stablehlo.reshape %arg0 : (tensor<768xbf16>) -> tensor<1x1x768xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1x1x768xbf16>) -> tensor<768xbf16>
    // CHECK: "ttir.layer_norm"(%{{.*}}, %{{.*}}, %{{.*}})
    %4 = stablehlo.composite "tenstorrent.layer_norm" %arg2, %1, %3 {composite_attributes = {epsilon = 9.99999974E-6 : f32, normalized_shape = dense<768> : tensor<1xi64>}, decomposition = @tenstorrent.layer_norm.impl} : (tensor<1x1024x768xbf16>, tensor<768xbf16>, tensor<768xbf16>) -> tensor<1x1024x768xbf16>
    return %4 : tensor<1x1024x768xbf16>
  }
  func.func private @tenstorrent.layer_norm.impl(%arg0: tensor<1x1024x768xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<768xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg2: tensor<768xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<1x1024x768xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %cst = stablehlo.constant dense<1.001360e-05> : tensor<1x1024x1xbf16>
    %cst_0 = stablehlo.constant dense<1.304630e-03> : tensor<1x1024xbf16>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.reduce(%arg0 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x768xbf16>, tensor<bf16>) -> tensor<1x1024xbf16>
    %1 = stablehlo.multiply %0, %cst_0 : tensor<1x1024xbf16>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x1024xbf16>) -> tensor<1x1024x768xbf16>
    %3 = stablehlo.subtract %arg0, %2 : tensor<1x1024x768xbf16>
    %4 = stablehlo.multiply %3, %3 : tensor<1x1024x768xbf16>
    %5 = stablehlo.reduce(%4 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x768xbf16>, tensor<bf16>) -> tensor<1x1024xbf16>
    %6 = stablehlo.multiply %5, %cst_0 : tensor<1x1024xbf16>
    %7 = stablehlo.reshape %6 : (tensor<1x1024xbf16>) -> tensor<1x1024x1xbf16>
    %8 = stablehlo.add %7, %cst : tensor<1x1024x1xbf16>
    %9 = stablehlo.rsqrt %8 : tensor<1x1024x1xbf16>
    %10 = stablehlo.reshape %9 : (tensor<1x1024x1xbf16>) -> tensor<1x1024xbf16>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1024xbf16>) -> tensor<1x1024x768xbf16>
    %12 = stablehlo.multiply %3, %11 : tensor<1x1024x768xbf16>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xbf16>) -> tensor<1x1024x768xbf16>
    %14 = stablehlo.multiply %12, %13 : tensor<1x1024x768xbf16>
    %15 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xbf16>) -> tensor<1x1024x768xbf16>
    %16 = stablehlo.add %14, %15 : tensor<1x1024x768xbf16>
    return %16 : tensor<1x1024x768xbf16>
  }
}
