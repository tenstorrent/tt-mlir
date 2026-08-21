// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --reoutline-composite -o %t %s
// RUN: FileCheck %s --input-file=%t

// `num_groups` on a group_norm composite counts groups over the global channel
// dimension. When Shardy shards that dimension the flattened body is rewritten
// to local shapes, so re-outlining has to rescale the attribute it restores --
// StableHLOLegalizeCompositePass builds ttir.group_norm from the attribute, not
// from the decomposition body.
//
// Here 512 channels / 32 groups sharded 4 ways gives 128 channels per device,
// which must be 8 groups of 16, not 32 groups of 4.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.composite "tenstorrent.group_norm"
// CHECK-SAME: num_groups = 8 : i64
module @ReoutlineGroupNormShardedChannels attributes {} {
  func.func @main(%arg0: tensor<1x128x1x40x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<1x128x1x40x64xf32> {
    %0 = stablehlo.reshape %arg0 {reoutline.arg_operand_indices = array<i64: 0>, reoutline.comp_attrs = {channel_dim = 1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 32 : i64}, reoutline.global_channels = 512 : i64, reoutline.group = "composite_tenstorrent.group_norm.impl", reoutline.orig_name = "tenstorrent.group_norm", reoutline.seed} : (tensor<1x128x1x40x64xf32>) -> tensor<1x8x40960xf32>
    %1 = stablehlo.reshape %0 {reoutline.group = "composite_tenstorrent.group_norm.impl"} : (tensor<1x8x40960xf32>) -> tensor<1x128x1x40x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.group_norm.impl"} : (tensor<128xf32>) -> tensor<1x128x1x40x64xf32>
    %3 = stablehlo.multiply %1, %2 {reoutline.group = "composite_tenstorrent.group_norm.impl"} : tensor<1x128x1x40x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {reoutline.arg_operand_indices = array<i64: 2>, reoutline.group = "composite_tenstorrent.group_norm.impl"} : (tensor<128xf32>) -> tensor<1x128x1x40x64xf32>
    %5 = stablehlo.add %3, %4 {reoutline.group = "composite_tenstorrent.group_norm.impl"} : tensor<1x128x1x40x64xf32>
    return %5 : tensor<1x128x1x40x64xf32>
  }
}

// -----

// When the channel dimension is left replicated the local shape matches the
// recorded global one, so the attributes must be restored untouched.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.composite "tenstorrent.group_norm"
// CHECK-SAME: num_groups = 32 : i64
module @ReoutlineGroupNormReplicatedChannels attributes {} {
  func.func @main(%arg0: tensor<1x512x1x40x64xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<1x512x1x40x64xf32> {
    %0 = stablehlo.reshape %arg0 {reoutline.arg_operand_indices = array<i64: 0>, reoutline.comp_attrs = {channel_dim = 1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 32 : i64}, reoutline.global_channels = 512 : i64, reoutline.group = "composite_tenstorrent.group_norm.impl", reoutline.orig_name = "tenstorrent.group_norm", reoutline.seed} : (tensor<1x512x1x40x64xf32>) -> tensor<1x32x40960xf32>
    %1 = stablehlo.reshape %0 {reoutline.group = "composite_tenstorrent.group_norm.impl"} : (tensor<1x32x40960xf32>) -> tensor<1x512x1x40x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.group_norm.impl"} : (tensor<512xf32>) -> tensor<1x512x1x40x64xf32>
    %3 = stablehlo.multiply %1, %2 {reoutline.group = "composite_tenstorrent.group_norm.impl"} : tensor<1x512x1x40x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {reoutline.arg_operand_indices = array<i64: 2>, reoutline.group = "composite_tenstorrent.group_norm.impl"} : (tensor<512xf32>) -> tensor<1x512x1x40x64xf32>
    %5 = stablehlo.add %3, %4 {reoutline.group = "composite_tenstorrent.group_norm.impl"} : tensor<1x512x1x40x64xf32>
    return %5 : tensor<1x512x1x40x64xf32>
  }
}
