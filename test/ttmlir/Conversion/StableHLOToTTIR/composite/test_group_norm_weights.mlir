// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @GroupNormWeightsModule attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["x"=1, "y"=1]>
  func.func @main(%arg0: tensor<480xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"}, %arg1: tensor<480xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_1"}, %arg2: tensor<1x1x64x480xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_2"}) -> (tensor<1x1x64x480xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    // CHECK: "ttir.group_norm"(%{{.*}}, %{{.*}}, %{{.*}})
    %0 = stablehlo.composite "tenstorrent.group_norm" %arg2, %arg0, %arg1 {composite_attributes = {epsilon = 1.000000e-05 : f32, num_groups = 8 : i64}, decomposition = @tenstorrent.group_norm.impl} : (tensor<1x1x64x480xbf16>, tensor<480xbf16>, tensor<480xbf16>) -> tensor<1x1x64x480xbf16>
    return %0 : tensor<1x1x64x480xbf16>
  }
  func.func private @tenstorrent.group_norm.impl(%arg0: tensor<1x1x64x480xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<480xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg2: tensor<480xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<1x1x64x480xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    // Placeholder decomposition body (not relevant for composite matching)
    return %arg0 : tensor<1x1x64x480xbf16>
  }
}
