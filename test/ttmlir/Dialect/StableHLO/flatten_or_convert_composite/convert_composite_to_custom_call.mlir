// REQUIRES: stablehlo
// RUN: ttmlir-opt --flatten-or-convert-composites -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_rms_norm attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  // CHECK: stablehlo.custom_call @tenstorrent.rms_norm(%{{.*}}, %{{.*}})
  // CHECK-SAME: tt.composite_attributes = {epsilon = {{.*}} : f32, normalized_shape = dense<128> : tensor<1xi64>}
  // CHECK-SAME: tt.has_custom_sharding
  // CHECK-SAME: (tensor<1x1x32x128xbf16>, tensor<128xbf16>) -> tensor<1x1x32x128xbf16>
  // CHECK-NOT: stablehlo.composite
  // CHECK-NOT: @tenstorrent.rms_norm.impl
  func.func @main(
    %arg0: tensor<1x1x32x128xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"_axis_0"}]>, ttcore.shard_status = #ttcore.shard_status<presharded>},
    %arg1: tensor<128xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}]>, ttcore.shard_status = #ttcore.shard_status<presharded>}
  ) -> (tensor<1x1x32x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.composite "tenstorrent.rms_norm" %arg0, %arg1 {
      composite_attributes = {epsilon = 9.99999974E-6 : f32, normalized_shape = dense<128> : tensor<1xi64>},
      decomposition = @tenstorrent.rms_norm.impl
    } : (tensor<1x1x32x128xbf16>, tensor<128xbf16>) -> tensor<1x1x32x128xbf16>
    return %0 : tensor<1x1x32x128xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl(%arg0: tensor<1x1x32x128xbf16>, %arg1: tensor<128xbf16>) -> tensor<1x1x32x128xbf16> {
    return %arg0 : tensor<1x1x32x128xbf16>
  }
}
