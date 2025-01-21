// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
// UNSUPPORTED: true

module @jit_fwd attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=4, "model"=2]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"batch", "model"} (%arg3: tensor<2048x392xf32>, %arg4: tensor<392x2048xf32>, %arg5: tensor<2048xf32>) {
      %1 = stablehlo.dot_general %arg3, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2048x392xf32>, tensor<392x2048xf32>) -> tensor<2048x2048xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        %6 = stablehlo.add %arg6, %arg7 : tensor<f32>
        stablehlo.return %6 : tensor<f32>
      }) : (tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
      %3 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<2048x2048xf32>
      %5 = stablehlo.add %2, %4 : tensor<2048x2048xf32>
      sdy.return %5 : tensor<2048x2048xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>) -> tensor<8192x2048xf32>
    return %0 : tensor<8192x2048xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["model"=4, "batch"=2]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"model", "batch"} (%arg3: tensor<4096x196xf32>, %arg4: tensor<196x2048xf32>, %arg5: tensor<2048xf32>) {
      %1 = stablehlo.dot_general %arg3, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4096x196xf32>, tensor<196x2048xf32>) -> tensor<4096x2048xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        %6 = stablehlo.add %arg6, %arg7 : tensor<f32>
        stablehlo.return %6 : tensor<f32>
      }) : (tensor<4096x2048xf32>) -> tensor<4096x2048xf32>
      %3 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<4096x2048xf32>
      %5 = stablehlo.add %2, %4 : tensor<4096x2048xf32>
      sdy.return %5 : tensor<4096x2048xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>) -> tensor<8192x2048xf32>
    return %0 : tensor<8192x2048xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["model"=2, "batch"=4]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"model", "batch"} (%arg3: tensor<2048x392xf32>, %arg4: tensor<392x2048xf32>, %arg5: tensor<2048xf32>) {
      %1 = stablehlo.dot_general %arg3, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2048x392xf32>, tensor<392x2048xf32>) -> tensor<2048x2048xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        %6 = stablehlo.add %arg6, %arg7 : tensor<f32>
        stablehlo.return %6 : tensor<f32>
      }) : (tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
      %3 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<2048x2048xf32>
      %5 = stablehlo.add %2, %4 : tensor<2048x2048xf32>
      sdy.return %5 : tensor<2048x2048xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>) -> tensor<8192x2048xf32>
    return %0 : tensor<8192x2048xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=2, "model"=4]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"batch", "model"} (%arg3: tensor<4096x196xf32>, %arg4: tensor<196x2048xf32>, %arg5: tensor<2048xf32>) {
      %1 = stablehlo.dot_general %arg3, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4096x196xf32>, tensor<196x2048xf32>) -> tensor<4096x2048xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        %6 = stablehlo.add %arg6, %arg7 : tensor<f32>
        stablehlo.return %6 : tensor<f32>
      }) : (tensor<4096x2048xf32>) -> tensor<4096x2048xf32>
      %3 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<4096x2048xf32>
      %5 = stablehlo.add %2, %4 : tensor<4096x2048xf32>
      sdy.return %5 : tensor<4096x2048xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>) -> tensor<8192x2048xf32>
    return %0 : tensor<8192x2048xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=8, "model"=4]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"batch", "model"} (%arg3: tensor<1024x196xf32>, %arg4: tensor<196x2048xf32>, %arg5: tensor<2048xf32>) {
      %1 = stablehlo.dot_general %arg3, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x196xf32>, tensor<196x2048xf32>) -> tensor<1024x2048xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        %6 = stablehlo.add %arg6, %arg7 : tensor<f32>
        stablehlo.return %6 : tensor<f32>
      }) : (tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
      %3 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<1024x2048xf32>
      %5 = stablehlo.add %2, %4 : tensor<1024x2048xf32>
      sdy.return %5 : tensor<1024x2048xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>) -> tensor<8192x2048xf32>
    return %0 : tensor<8192x2048xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["model"=8, "batch"=4]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"model", "batch"} (%arg3: tensor<2048x98xf32>, %arg4: tensor<98x2048xf32>, %arg5: tensor<2048xf32>) {
      %1 = stablehlo.dot_general %arg3, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2048x98xf32>, tensor<98x2048xf32>) -> tensor<2048x2048xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]> : tensor<4x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        %6 = stablehlo.add %arg6, %arg7 : tensor<f32>
        stablehlo.return %6 : tensor<f32>
      }) : (tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
      %3 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<2048x2048xf32>
      %5 = stablehlo.add %2, %4 : tensor<2048x2048xf32>
      sdy.return %5 : tensor<2048x2048xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>) -> tensor<8192x2048xf32>
    return %0 : tensor<8192x2048xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["model"=4, "batch"=8]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"model", "batch"} (%arg3: tensor<1024x196xf32>, %arg4: tensor<196x2048xf32>, %arg5: tensor<2048xf32>) {
      %1 = stablehlo.dot_general %arg3, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x196xf32>, tensor<196x2048xf32>) -> tensor<1024x2048xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], [4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        %6 = stablehlo.add %arg6, %arg7 : tensor<f32>
        stablehlo.return %6 : tensor<f32>
      }) : (tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
      %3 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<1024x2048xf32>
      %5 = stablehlo.add %2, %4 : tensor<1024x2048xf32>
      sdy.return %5 : tensor<1024x2048xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>) -> tensor<8192x2048xf32>
    return %0 : tensor<8192x2048xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=4, "model"=8]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"batch", "model"} (%arg3: tensor<2048x98xf32>, %arg4: tensor<98x2048xf32>, %arg5: tensor<2048xf32>) {
      %1 = stablehlo.dot_general %arg3, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2048x98xf32>, tensor<98x2048xf32>) -> tensor<2048x2048xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        %6 = stablehlo.add %arg6, %arg7 : tensor<f32>
        stablehlo.return %6 : tensor<f32>
      }) : (tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
      %3 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<2048x2048xf32>
      %5 = stablehlo.add %2, %4 : tensor<2048x2048xf32>
      sdy.return %5 : tensor<2048x2048xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>) -> tensor<8192x2048xf32>
    return %0 : tensor<8192x2048xf32>
  }
}
