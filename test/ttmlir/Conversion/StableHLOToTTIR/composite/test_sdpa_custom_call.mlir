// REQUIRES: stablehlo
// RUN: ttmlir-opt --flatten-or-convert-composites -o %t %s
// RUN: FileCheck %s --input-file=%t

// SDPA is registered in kCompositesWithCustomSharding, so in a sharded module
// FlattenOrConvertComposites keeps the composite as a custom_call (so Shardy can
// propagate the head sharding) instead of flattening it to its f32 decomposition.
module @jit_sdpa attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  // CHECK: stablehlo.custom_call @tenstorrent.scaled_dot_product_attention
  // CHECK-SAME: tt.composite_attributes
  // CHECK-SAME: tt.has_custom_sharding
  // CHECK-NOT: stablehlo.composite
  func.func @main(
    %q: tensor<1x2x32x16xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}, {}, {}]>, ttcore.shard_status = #ttcore.shard_status<presharded>},
    %k: tensor<1x2x32x16xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}, {}, {}]>, ttcore.shard_status = #ttcore.shard_status<presharded>},
    %v: tensor<1x2x32x16xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}, {}, {}]>, ttcore.shard_status = #ttcore.shard_status<presharded>},
    %mask: tensor<1x1x32x32xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>, ttcore.shard_status = #ttcore.shard_status<presharded>}
  ) -> (tensor<1x2x32x16xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.composite "tenstorrent.scaled_dot_product_attention" %q, %k, %v, %mask {
        composite_attributes = {is_causal = false},
        decomposition = @sdpa_impl
    } : (tensor<1x2x32x16xbf16>, tensor<1x2x32x16xbf16>, tensor<1x2x32x16xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x2x32x16xbf16>
    return %0 : tensor<1x2x32x16xbf16>
  }
  func.func private @sdpa_impl(
      %arg0: tensor<1x2x32x16xbf16>, %arg1: tensor<1x2x32x16xbf16>,
      %arg2: tensor<1x2x32x16xbf16>, %arg3: tensor<1x1x32x32xbf16>) -> tensor<1x2x32x16xbf16> {
    return %arg0 : tensor<1x2x32x16xbf16>
  }
}
