// REQUIRES: stablehlo
// RUN: ttmlir-opt --annotate-local-shapes %s | FileCheck %s

// Test that batch-parallel presharded arguments where the argument type is
// already the per-device (local) shape are annotated correctly. In this
// scenario the sharding axis size (8) does not divide evenly into the
// declared tensor dimension (4) because the dimension already reflects the
// local shard size, not the global shape.

// CHECK-LABEL: func.func public @main
// CHECK-SAME: ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x32xf32>>
// CHECK-SAME: ttcore.shard_status = #ttcore.shard_status<presharded>
module {
  sdy.mesh @mesh = <["_axis_0_aux"=1, "_axis_0"=8]>
  func.func public @main(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> tensor<4x32xf32> {
    return %arg0 : tensor<4x32xf32>
  }
}
