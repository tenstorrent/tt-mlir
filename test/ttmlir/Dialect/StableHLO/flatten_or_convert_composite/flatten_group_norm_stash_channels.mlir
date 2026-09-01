// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --flatten-or-convert-composites -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// tenstorrent.group_norm has no custom sharding rule, so it is flattened and its
// composite attributes are stashed on the seed op for later re-outlining. Those
// attributes hold the *global* num_groups, so the global channel count is stashed
// alongside them: it is how ReoutlineCompositePass recovers the group size once
// shapes have been localized.
module @jit_flatten_group_norm attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["_axis_0"=4]>
  func.func @main(%arg0: tensor<1x512x1x60x90xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}, {}, {}, {}]>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> tensor<1x512x1x60x90xbf16> {
    // CHECK-NOT: stablehlo.composite
    // CHECK: stablehlo.multiply
    // CHECK-SAME: reoutline.comp_attrs = {channel_dim = 1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 32 : i64}
    // CHECK-SAME: reoutline.group_norm_global_channels = 512 : i64
    // CHECK-SAME: reoutline.orig_name = "tenstorrent.group_norm"
    %0 = stablehlo.composite "tenstorrent.group_norm" %arg0 {composite_attributes = {channel_dim = 1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 32 : i64}, decomposition = @tenstorrent.group_norm.impl} : (tensor<1x512x1x60x90xbf16>) -> tensor<1x512x1x60x90xbf16>
    return %0 : tensor<1x512x1x60x90xbf16>
  }
  func.func private @tenstorrent.group_norm.impl(%arg0: tensor<1x512x1x60x90xbf16>) -> tensor<1x512x1x60x90xbf16> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x512x1x60x90xbf16>
    return %0 : tensor<1x512x1x60x90xbf16>
  }
}
