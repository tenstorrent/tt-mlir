// REQUIRES: stablehlo
// RUN: ttmlir-opt --split-input-file --verify-diagnostics --reoutline-composite %s

// num_groups=2 over 32 global channels means a group size of 16, but a 4-way shard
// of the channel dim leaves only 8 local channels -- half a group. There is no
// valid local group count, so re-outlining must fail loudly rather than let each
// device normalize over the wrong channels. Note the ttir.group_norm verifier
// would not catch this either: 8 % 2 == 0.
module @jit_reoutline_group_norm_too_fine attributes {mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=4]>
  func.func @main(%arg0: tensor<1x32x1x60x90xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<1x32x1x60x90xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"_axis_0"}, {}, {}, {}]>] out_shardings=[<@mesh, [{?}, {"_axis_0", ?}, {?}, {?}, {?}]>] manual_axes={"_axis_0"} (%arg1: tensor<1x8x1x60x90xbf16>) {
      // The diagnostic is reported on the seed op that carries the stashed attrs.
      // expected-error @+1 {{cannot rescale num_groups: local channel dim (8) does not hold a whole number of groups of size 16}}
      %cst = stablehlo.constant {reoutline.comp_attrs = {channel_dim = 1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 2 : i64}, reoutline.group = "composite_tenstorrent.group_norm.impl", reoutline.group_norm_global_channels = 32 : i64, reoutline.orig_name = "tenstorrent.group_norm", reoutline.seed} dense<1.000000e+00> : tensor<bf16>
      %1 = stablehlo.broadcast_in_dim %cst, dims = [] {reoutline.group = "composite_tenstorrent.group_norm.impl"} : (tensor<bf16>) -> tensor<1x8x1x60x90xbf16>
      %2 = stablehlo.multiply %arg1, %1 {reoutline.group = "composite_tenstorrent.group_norm.impl"} : tensor<1x8x1x60x90xbf16>
      sdy.return %2 : tensor<1x8x1x60x90xbf16>
    } : (tensor<1x32x1x60x90xbf16>) -> tensor<1x32x1x60x90xbf16>
    return %0 : tensor<1x32x1x60x90xbf16>
  }
}
