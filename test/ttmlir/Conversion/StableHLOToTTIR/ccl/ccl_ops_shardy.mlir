// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-to-ttir-pipeline %s | FileCheck %s

// jax/pjrt sharding target 1x2 for t3k - Shardy all_reduce
module @jit_matmul_shardy0 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=1, "y"=2]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"y", "x"} (%arg2: tensor<8192x392xf32>, %arg3: tensor<392x16384xf32>) {
      %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8192x392xf32>, tensor<392x16384xf32>) -> tensor<8192x16384xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
      ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
        %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
        stablehlo.return %3 : tensor<f32>
      }) : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
      sdy.return %2 : tensor<8192x16384xf32>
    } : (tensor<8192x784xf32>, tensor<784x16384xf32>) -> tensor<8192x16384xf32>
    return %0 : tensor<8192x16384xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 2>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 2, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: = "ttir.all_reduce"
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<replicate>

// -----

// jax/pjrt sharding target 2x4 for t3k - Shardy all_reduce
module @jit_matmul_shardy1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x", "y"} (%arg2: tensor<4096x196xf32>, %arg3: tensor<196x16384xf32>) {
      %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4096x196xf32>, tensor<196x16384xf32>) -> tensor<4096x16384xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
        %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
        stablehlo.return %3 : tensor<f32>
      }) : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
      sdy.return %2 : tensor<4096x16384xf32>
    } : (tensor<8192x784xf32>, tensor<784x16384xf32>) -> tensor<8192x16384xf32>
    return %0 : tensor<8192x16384xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 2, 4>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 4, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: = "ttir.all_reduce"
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 2, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>

// -----

// jax/pjrt sharding target 1x8 for t3k - Shardy all_reduce
module @jit_matmul_shardy2 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=1, "y"=8]>
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"y", "x"} (%arg2: tensor<8192x98xf32>, %arg3: tensor<98x16384xf32>) {
      %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8192x98xf32>, tensor<98x16384xf32>) -> tensor<8192x16384xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
        %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
        stablehlo.return %3 : tensor<f32>
      }) : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
      sdy.return %2 : tensor<8192x16384xf32>
    } : (tensor<8192x784xf32>, tensor<784x16384xf32>) -> tensor<8192x16384xf32>
    return %0 : tensor<8192x16384xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: = "ttir.all_reduce"
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<replicate>

// -----

// jax/pjrt sharding target 2x4 for t3k - Shardy negative, sharding [None, "x", None, "y"]
module @jit_neg_shardy0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"x"}, {}, {"y"}]>] out_shardings=[<@mesh, [{}, {"x"}, {}, {"y"}]>] manual_axes={"y", "x"} (%arg1: tensor<1x512x128x256xf32>) {
      %1 = stablehlo.negate %arg1 : tensor<1x512x128x256xf32>
      sdy.return %1 : tensor<1x512x128x256xf32>
    } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    return %0 : tensor<1x1024x128x1024xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 1, 3>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 4>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 1, 3>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 4>
// CHECK-SAME: shard_type = #tt.shard_type<devices>

// -----

// jax/pjrt sharding target 2x4 for t3k - Shardy negative, sharding [None, "x", None, None]
module @jit_neg_shardy1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"x"}, {}, {}]>] out_shardings=[<@mesh, [{}, {"x"}, {}, {}]>] manual_axes={"x", "y"} (%arg1: tensor<1x512x128x1024xf32>) {
      %1 = stablehlo.negate %arg1 : tensor<1x512x128x1024xf32>
      sdy.return %1 : tensor<1x512x128x1024xf32>
    } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    return %0 : tensor<1x1024x128x1024xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 1, -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 1, -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>

// -----

// jax/pjrt sharding target 2x4 for t3k - Shardy negative, sharding [None, None, None, "y"]
module @jit_neg_shardy2 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}, {}, {"y"}]>] out_shardings=[<@mesh, [{}, {}, {}, {"y"}]>] manual_axes={"y", "x"} (%arg1: tensor<1x1024x128x256xf32>) {
      %1 = stablehlo.negate %arg1 : tensor<1x1024x128x256xf32>
      sdy.return %1 : tensor<1x1024x128x256xf32>
    } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    return %0 : tensor<1x1024x128x1024xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 3>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 4>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 3>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 4>
// CHECK-SAME: shard_type = #tt.shard_type<devices>

// -----

// jax/pjrt sharding target 2x4 for t3k - Shardy negative, sharding [None, "y", None, "x"]
module @jit_neg_shardy3 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"y"}, {}, {"x"}]>] out_shardings=[<@mesh, [{}, {"y"}, {}, {"x"}]>] manual_axes={"x", "y"} (%arg1: tensor<1x256x128x512xf32>) {
      %1 = stablehlo.negate %arg1 : tensor<1x256x128x512xf32>
      sdy.return %1 : tensor<1x256x128x512xf32>
    } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    return %0 : tensor<1x1024x128x1024xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 3, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 2>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 3, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 2>
// CHECK-SAME: shard_type = #tt.shard_type<devices>

// -----

// jax/pjrt sharding target 2x4 for t3k - Shardy negative, sharding [None, "y", None, None]
module @jit_neg_shardy4 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"y"}, {}, {}]>] out_shardings=[<@mesh, [{}, {"y"}, {}, {}]>] manual_axes={"x", "y"} (%arg1: tensor<1x256x128x1024xf32>) {
      %1 = stablehlo.negate %arg1 : tensor<1x256x128x1024xf32>
      sdy.return %1 : tensor<1x256x128x1024xf32>
    } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    return %0 : tensor<1x1024x128x1024xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>

// -----

// jax/pjrt sharding target 2x4 for t3k - Shardy negative, sharding [None, None, None, "x"]
module @jit_neg_shardy5 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}, {}, {"x"}]>] out_shardings=[<@mesh, [{}, {}, {}, {"x"}]>] manual_axes={"x", "y"} (%arg1: tensor<1x1024x128x512xf32>) {
      %1 = stablehlo.negate %arg1 : tensor<1x1024x128x512xf32>
      sdy.return %1 : tensor<1x1024x128x512xf32>
    } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    return %0 : tensor<1x1024x128x1024xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 3, -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 3, -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
// CHECK-SAME: shard_type = #tt.shard_type<devices>

// -----

// jax/pjrt sharding target 1x8 for t3k - Shardy negative, sharding [None, None, None, "y"]
module @jit_neg_shardy6 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=1, "y"=8]>
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}, {}, {"y"}]>] out_shardings=[<@mesh, [{}, {}, {}, {"y"}]>] manual_axes={"y", "x"} (%arg1: tensor<1x1024x128x128xf32>) {
      %1 = stablehlo.negate %arg1 : tensor<1x1024x128x128xf32>
      sdy.return %1 : tensor<1x1024x128x128xf32>
    } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    return %0 : tensor<1x1024x128x1024xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 3>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 3>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
// CHECK-SAME: shard_type = #tt.shard_type<devices>

// -----

// jax/pjrt sharding target 1x8 for t3k - Shardy negative, sharding [None, "y", None, None]
module @jit_neg_shardy7 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=1, "y"=8]>
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"y"}, {}, {}]>] out_shardings=[<@mesh, [{}, {"y"}, {}, {}]>] manual_axes={"y", "x"} (%arg1: tensor<1x128x128x1024xf32>) {
      %1 = stablehlo.negate %arg1 : tensor<1x128x128x1024xf32>
      sdy.return %1 : tensor<1x128x128x1024xf32>
    } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    return %0 : tensor<1x1024x128x1024xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8, 1, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8, 1, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>

// -----

// jax/pjrt sharding target 2x4 for t3k - Shardy all_reduce with automatic input sharding
module @jit_matmul_shardy_automatic attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func public @main(%arg0: tensor<8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<784x16384xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x", "y"} (%arg2: tensor<4096x196xf32>, %arg3: tensor<196x16384xf32>) {
      %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4096x196xf32>, tensor<196x16384xf32>) -> tensor<4096x16384xf32>
      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
        %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
        stablehlo.return %3 : tensor<f32>
      }) : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
      sdy.return %2 : tensor<4096x16384xf32>
    } : (tensor<8192x784xf32>, tensor<784x16384xf32>) -> tensor<8192x16384xf32>
    return %0 : tensor<8192x16384xf32>
  }
}
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 2, 4>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 4, 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: = "ttir.all_reduce"
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 2, 1>
// CHECK-SAME: shard_type = #tt.shard_type<devices>
