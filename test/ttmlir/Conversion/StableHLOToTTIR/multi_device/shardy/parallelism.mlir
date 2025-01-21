// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
// UNSUPPORTED: true

module @jit_matmul_basic attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=4, "y"=2]>
  func.func public @main(%arg0: tensor<8x16xf32>, %arg1: tensor<16x4xf32>) -> (tensor<8x4xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x", "y"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x4xf32>) {
      %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x8xf32>, tensor<8x4xf32>) -> tensor<2x4xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
      ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
        %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
        stablehlo.return %3 : tensor<f32>
      }) : (tensor<2x4xf32>) -> tensor<2x4xf32>
      sdy.return %2 : tensor<2x4xf32>
    } : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=8]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<8192x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4) in_shardings=[<@mesh, [{"batch"}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"batch"} (%arg5: tensor<1024x784xf32>, %arg6: tensor<784x2048xf32>, %arg7: tensor<2048xf32>, %arg8: tensor<2048x1024xf32>, %arg9: tensor<1024xf32>) {
      %1 = stablehlo.dot_general %arg5, %arg6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x784xf32>, tensor<784x2048xf32>) -> tensor<1024x2048xf32>
      %2 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<1024x2048xf32>
      %4 = stablehlo.add %1, %3 : tensor<1024x2048xf32>
      %5 = stablehlo.dot_general %4, %arg8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x2048xf32>, tensor<2048x1024xf32>) -> tensor<1024x1024xf32>
      %6 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<1024x1024xf32>
      %8 = stablehlo.add %5, %7 : tensor<1024x1024xf32>
      sdy.return %8 : tensor<1024x1024xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>, tensor<2048x1024xf32>, tensor<1024xf32>) -> tensor<8192x1024xf32>
    return %0 : tensor<8192x1024xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=4, "model"=2]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<8192x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{"model"}]>] out_shardings=[<@mesh, [{"batch"}, {"model"}]>] manual_axes={"batch", "model"} (%arg5: tensor<2048x392xf32>, %arg6: tensor<392x2048xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<1024x1024xf32>, %arg9: tensor<512xf32>) {
      %1 = stablehlo.dot_general %arg5, %arg6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2048x392xf32>, tensor<392x2048xf32>) -> tensor<2048x2048xf32>
      %2 = "stablehlo.reduce_scatter"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):
        %11 = stablehlo.add %arg10, %arg11 : tensor<f32>
        stablehlo.return %11 : tensor<f32>
      }) : (tensor<2048x2048xf32>) -> tensor<2048x1024xf32>
      %3 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<2048x1024xf32>
      %5 = stablehlo.add %2, %4 : tensor<2048x1024xf32>
      %6 = stablehlo.dot_general %5, %arg8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2048x1024xf32>, tensor<1024x1024xf32>) -> tensor<2048x1024xf32>
      %7 = "stablehlo.reduce_scatter"(%6) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):
        %11 = stablehlo.add %arg10, %arg11 : tensor<f32>
        stablehlo.return %11 : tensor<f32>
      }) : (tensor<2048x1024xf32>) -> tensor<2048x512xf32>
      %8 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<2048x512xf32>
      %10 = stablehlo.add %7, %9 : tensor<2048x512xf32>
      sdy.return %10 : tensor<2048x512xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>, tensor<2048x1024xf32>, tensor<1024xf32>) -> tensor<8192x1024xf32>
    return %0 : tensor<8192x1024xf32>
  }
}

module @jit_matmul_basic attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=8, "y"=4]>
  func.func public @main(%arg0: tensor<8x16xf32>, %arg1: tensor<16x4xf32>) -> (tensor<8x4xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x", "y"} (%arg2: tensor<1x4xf32>, %arg3: tensor<4x4xf32>) {
      %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
        %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
        stablehlo.return %3 : tensor<f32>
      }) : (tensor<1x4xf32>) -> tensor<1x4xf32>
      sdy.return %2 : tensor<1x4xf32>
    } : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=8, "model"=4]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<8192x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{"model"}]>, <@mesh, [{"model"}, {}]>, <@mesh, [{"model"}]>] out_shardings=[<@mesh, [{"batch"}, {"model"}]>] manual_axes={"batch", "model"} (%arg5: tensor<1024x196xf32>, %arg6: tensor<196x2048xf32>, %arg7: tensor<512xf32>, %arg8: tensor<512x1024xf32>, %arg9: tensor<256xf32>) {
      %1 = stablehlo.dot_general %arg5, %arg6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x196xf32>, tensor<196x2048xf32>) -> tensor<1024x2048xf32>
      %2 = "stablehlo.reduce_scatter"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):
        %11 = stablehlo.add %arg10, %arg11 : tensor<f32>
        stablehlo.return %11 : tensor<f32>
      }) : (tensor<1024x2048xf32>) -> tensor<1024x512xf32>
      %3 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<1024x512xf32>
      %5 = stablehlo.add %2, %4 : tensor<1024x512xf32>
      %6 = stablehlo.dot_general %5, %arg8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x512xf32>, tensor<512x1024xf32>) -> tensor<1024x1024xf32>
      %7 = "stablehlo.reduce_scatter"(%6) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg10: tensor<f32>, %arg11: tensor<f32>):
        %11 = stablehlo.add %arg10, %arg11 : tensor<f32>
        stablehlo.return %11 : tensor<f32>
      }) : (tensor<1024x1024xf32>) -> tensor<1024x256xf32>
      %8 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<1024x256xf32>
      %10 = stablehlo.add %7, %9 : tensor<1024x256xf32>
      sdy.return %10 : tensor<1024x256xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>, tensor<2048x1024xf32>, tensor<1024xf32>) -> tensor<8192x1024xf32>
    return %0 : tensor<8192x1024xf32>
  }
}

module @jit_fwd attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=32]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<8192x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4) in_shardings=[<@mesh, [{"batch"}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}]>] manual_axes={"batch"} (%arg5: tensor<256x784xf32>, %arg6: tensor<784x2048xf32>, %arg7: tensor<2048xf32>, %arg8: tensor<2048x1024xf32>, %arg9: tensor<1024xf32>) {
      %1 = stablehlo.dot_general %arg5, %arg6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x784xf32>, tensor<784x2048xf32>) -> tensor<256x2048xf32>
      %2 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<256x2048xf32>
      %4 = stablehlo.add %1, %3 : tensor<256x2048xf32>
      %5 = stablehlo.dot_general %4, %arg8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf32>, tensor<2048x1024xf32>) -> tensor<256x1024xf32>
      %6 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<256x1024xf32>
      %8 = stablehlo.add %5, %7 : tensor<256x1024xf32>
      sdy.return %8 : tensor<256x1024xf32>
    } : (tensor<8192x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>, tensor<2048x1024xf32>, tensor<1024xf32>) -> tensor<8192x1024xf32>
    return %0 : tensor<8192x1024xf32>
  }
}
