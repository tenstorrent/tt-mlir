// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --split-input-file --reoutline-composite -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// The flattened group_norm body has already been localized by
// UpdateGlobalToLocalShapesPass (channel dim 512 -> 128 on a 4-way shard), but the
// stashed composite attributes still hold the global num_groups=32. Re-outlining
// must rescale it to 8 so the group size (512/32 = 16) is preserved on the local
// 128-channel shard. Without the rescale each device would normalize 4 channels
// per group instead of 16 -- and nothing downstream would flag it, since
// 128 % 32 == 0 satisfies the ttir.group_norm verifier.
module @jit_reoutline_group_norm_sharded attributes {mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=4]>
  func.func @main(%arg0: tensor<1x512x1x60x90xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<1x512x1x60x90xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"_axis_0"}, {}, {}, {}]>] out_shardings=[<@mesh, [{?}, {"_axis_0", ?}, {?}, {?}, {?}]>] manual_axes={"_axis_0"} (%arg1: tensor<1x128x1x60x90xbf16>) {
      // CHECK: stablehlo.composite "tenstorrent.group_norm"
      // CHECK-SAME: num_groups = 8 : i64
      // CHECK-SAME: (tensor<1x128x1x60x90xbf16>) -> tensor<1x128x1x60x90xbf16>
      %cst = stablehlo.constant {reoutline.comp_attrs = {channel_dim = 1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 32 : i64}, reoutline.group = "composite_tenstorrent.group_norm.impl", reoutline.group_norm_global_channels = 512 : i64, reoutline.orig_name = "tenstorrent.group_norm", reoutline.seed} dense<1.000000e+00> : tensor<bf16>
      %1 = stablehlo.broadcast_in_dim %cst, dims = [] {reoutline.group = "composite_tenstorrent.group_norm.impl"} : (tensor<bf16>) -> tensor<1x128x1x60x90xbf16>
      %2 = stablehlo.multiply %arg1, %1 {reoutline.group = "composite_tenstorrent.group_norm.impl"} : tensor<1x128x1x60x90xbf16>
      sdy.return %2 : tensor<1x128x1x60x90xbf16>
    } : (tensor<1x512x1x60x90xbf16>) -> tensor<1x512x1x60x90xbf16>
    return %0 : tensor<1x512x1x60x90xbf16>
  }
}

// -----

// Channel dim left replicated (only the batch dim is sharded): the stashed global
// channel count matches the local one, so num_groups must be left alone.
module @jit_reoutline_group_norm_replicated attributes {mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=4]>
  func.func @main(%arg0: tensor<4x512x1x60x90xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<4x512x1x60x90xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"_axis_0"}, {}, {}, {}, {}]>] out_shardings=[<@mesh, [{"_axis_0", ?}, {?}, {?}, {?}, {?}]>] manual_axes={"_axis_0"} (%arg1: tensor<1x512x1x60x90xbf16>) {
      // CHECK: stablehlo.composite "tenstorrent.group_norm"
      // CHECK-SAME: num_groups = 32 : i64
      // CHECK-SAME: (tensor<1x512x1x60x90xbf16>) -> tensor<1x512x1x60x90xbf16>
      %cst = stablehlo.constant {reoutline.comp_attrs = {channel_dim = 1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 32 : i64}, reoutline.group = "composite_tenstorrent.group_norm.impl", reoutline.group_norm_global_channels = 512 : i64, reoutline.orig_name = "tenstorrent.group_norm", reoutline.seed} dense<1.000000e+00> : tensor<bf16>
      %1 = stablehlo.broadcast_in_dim %cst, dims = [] {reoutline.group = "composite_tenstorrent.group_norm.impl"} : (tensor<bf16>) -> tensor<1x512x1x60x90xbf16>
      %2 = stablehlo.multiply %arg1, %1 {reoutline.group = "composite_tenstorrent.group_norm.impl"} : tensor<1x512x1x60x90xbf16>
      sdy.return %2 : tensor<1x512x1x60x90xbf16>
    } : (tensor<4x512x1x60x90xbf16>) -> tensor<4x512x1x60x90xbf16>
    return %0 : tensor<4x512x1x60x90xbf16>
  }
}
