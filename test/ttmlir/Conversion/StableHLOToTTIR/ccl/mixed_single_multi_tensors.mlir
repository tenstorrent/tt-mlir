// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @mixed_single_multi_tensors {
  sdy.mesh @mesh = <["x"=1, "y"=2]>
  func.func public @main(%arg0: tensor<1x1x512x128xf32>, %arg1: tensor<1x1x128xf32>, %arg2: tensor<1x1x32x512xf32>) -> tensor<1x1x32x128xf32> {
    %0 = sdy.manual_computation(%arg2, %arg0, %arg1) in_shardings=[<@mesh, [{}, {}, {}, {"y"}]>, <@mesh, [{}, {}, {"y"}, {}]>, <@mesh, [{}, {}, {"y"}]>] out_shardings=[<@mesh, [{}, {}, {}, {"y"}]>] manual_axes={"x", "y"} (%arg3: tensor<1x1x32x256xf32>, %arg4: tensor<1x1x256x128xf32>, %arg5: tensor<1x1x64xf32>) {
      %2 = stablehlo.dot_general %arg3, %arg4, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x1x32x256xf32>, tensor<1x1x256x128xf32>) -> tensor<1x1x32x128xf32>
      %3 = "stablehlo.reduce_scatter"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        %7 = stablehlo.add %arg6, %arg7 : tensor<f32>
        stablehlo.return %7 : tensor<f32>
      }) : (tensor<1x1x32x128xf32>) -> tensor<1x1x32x64xf32>
      %4 = stablehlo.broadcast_in_dim %arg5, dims = [0, 1, 3] : (tensor<1x1x64xf32>) -> tensor<1x1x1x64xf32>
      %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x1x32x64xf32>
      %6 = stablehlo.add %3, %5 : tensor<1x1x32x64xf32>
      sdy.return %6 : tensor<1x1x32x64xf32>
    } : (tensor<1x1x32x512xf32>, tensor<1x1x512x128xf32>, tensor<1x1x128xf32>) -> tensor<1x1x32x128xf32>
    %1 = call @relu(%0) : (tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32>
    return %1 : tensor<1x1x32x128xf32>
  }
  func.func private @relu(%arg0: tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x32x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x1x32x128xf32>
    return %1 : tensor<1x1x32x128xf32>
  }
}

// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 3>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 2>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 2>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 1, 2>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 3>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.reshape"
// CHECK: "ttir.broadcast"
// CHECK: "ttir.maximum"
