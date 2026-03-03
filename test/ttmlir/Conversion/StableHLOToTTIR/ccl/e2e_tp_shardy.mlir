// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_loss_tp attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=1, "y"=8]>
  func.func public @main(%arg0: tensor<784x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg1: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg2: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg3: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg4: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg5: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg6: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg7: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg8: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg9: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg10: tensor<128x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg11: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg12: tensor<32x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg13: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg12, %arg0, %arg1) in_shardings=[<@mesh, [{}, {"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>] out_shardings=[<@mesh, [{}, {"y"}]>] manual_axes={"x", "y"} (%arg14: tensor<32x98xf32>, %arg15: tensor<98x128xf32>, %arg16: tensor<16xf32>) {
      %16 = stablehlo.dot_general %arg14, %arg15, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x98xf32>, tensor<98x128xf32>) -> tensor<32x128xf32>
      %17 = "stablehlo.reduce_scatter"(%16) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg17: tensor<f32>, %arg18: tensor<f32>):
        %21 = stablehlo.add %arg17, %arg18 : tensor<f32>
        stablehlo.return %21 : tensor<f32>
      }) : (tensor<32x128xf32>) -> tensor<32x16xf32>
      %18 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
      %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
      %20 = stablehlo.add %17, %19 : tensor<32x16xf32>
      sdy.return %20 : tensor<32x16xf32>
    } : (tensor<32x784xf32>, tensor<784x128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
    %1 = call @relu(%0) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %2 = sdy.manual_computation(%1, %arg2, %arg3) in_shardings=[<@mesh, [{}, {"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>] out_shardings=[<@mesh, [{}, {"y"}]>] manual_axes={"x", "y"} (%arg14: tensor<32x16xf32>, %arg15: tensor<16x128xf32>, %arg16: tensor<16xf32>) {
      %16 = stablehlo.dot_general %arg14, %arg15, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<16x128xf32>) -> tensor<32x128xf32>
      %17 = "stablehlo.reduce_scatter"(%16) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg17: tensor<f32>, %arg18: tensor<f32>):
        %21 = stablehlo.add %arg17, %arg18 : tensor<f32>
        stablehlo.return %21 : tensor<f32>
      }) : (tensor<32x128xf32>) -> tensor<32x16xf32>
      %18 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
      %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
      %20 = stablehlo.add %17, %19 : tensor<32x16xf32>
      sdy.return %20 : tensor<32x16xf32>
    } : (tensor<32x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
    %3 = call @relu(%2) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %4 = sdy.manual_computation(%3, %arg4, %arg5) in_shardings=[<@mesh, [{}, {"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>] out_shardings=[<@mesh, [{}, {"y"}]>] manual_axes={"x", "y"} (%arg14: tensor<32x16xf32>, %arg15: tensor<16x128xf32>, %arg16: tensor<16xf32>) {
      %16 = stablehlo.dot_general %arg14, %arg15, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<16x128xf32>) -> tensor<32x128xf32>
      %17 = "stablehlo.reduce_scatter"(%16) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg17: tensor<f32>, %arg18: tensor<f32>):
        %21 = stablehlo.add %arg17, %arg18 : tensor<f32>
        stablehlo.return %21 : tensor<f32>
      }) : (tensor<32x128xf32>) -> tensor<32x16xf32>
      %18 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
      %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
      %20 = stablehlo.add %17, %19 : tensor<32x16xf32>
      sdy.return %20 : tensor<32x16xf32>
    } : (tensor<32x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
    %5 = call @relu(%4) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %6 = sdy.manual_computation(%5, %arg6, %arg7) in_shardings=[<@mesh, [{}, {"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>] out_shardings=[<@mesh, [{}, {"y"}]>] manual_axes={"x", "y"} (%arg14: tensor<32x16xf32>, %arg15: tensor<16x128xf32>, %arg16: tensor<16xf32>) {
      %16 = stablehlo.dot_general %arg14, %arg15, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<16x128xf32>) -> tensor<32x128xf32>
      %17 = "stablehlo.reduce_scatter"(%16) <{channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg17: tensor<f32>, %arg18: tensor<f32>):
        %21 = stablehlo.add %arg17, %arg18 : tensor<f32>
        stablehlo.return %21 : tensor<f32>
      }) : (tensor<32x128xf32>) -> tensor<32x16xf32>
      %18 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
      %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
      %20 = stablehlo.add %17, %19 : tensor<32x16xf32>
      sdy.return %20 : tensor<32x16xf32>
    } : (tensor<32x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
    %7 = call @relu(%6) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %8 = sdy.manual_computation(%7, %arg8, %arg9) in_shardings=[<@mesh, [{}, {"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>] out_shardings=[<@mesh, [{}, {"y"}]>] manual_axes={"x", "y"} (%arg14: tensor<32x16xf32>, %arg15: tensor<16x128xf32>, %arg16: tensor<16xf32>) {
      %16 = stablehlo.dot_general %arg14, %arg15, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<16x128xf32>) -> tensor<32x128xf32>
      %17 = "stablehlo.reduce_scatter"(%16) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg17: tensor<f32>, %arg18: tensor<f32>):
        %21 = stablehlo.add %arg17, %arg18 : tensor<f32>
        stablehlo.return %21 : tensor<f32>
      }) : (tensor<32x128xf32>) -> tensor<32x16xf32>
      %18 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
      %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
      %20 = stablehlo.add %17, %19 : tensor<32x16xf32>
      sdy.return %20 : tensor<32x16xf32>
    } : (tensor<32x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
    %9 = call @relu(%8) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %10 = sdy.manual_computation(%9, %arg10, %arg11) in_shardings=[<@mesh, [{}, {"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>] out_shardings=[<@mesh, [{}, {"y"}]>] manual_axes={"x", "y"} (%arg14: tensor<32x16xf32>, %arg15: tensor<16x8xf32>, %arg16: tensor<1xf32>) {
      %16 = stablehlo.dot_general %arg14, %arg15, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<16x8xf32>) -> tensor<32x8xf32>
      %17 = "stablehlo.reduce_scatter"(%16) <{channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg17: tensor<f32>, %arg18: tensor<f32>):
        %21 = stablehlo.add %arg17, %arg18 : tensor<f32>
        stablehlo.return %21 : tensor<f32>
      }) : (tensor<32x8xf32>) -> tensor<32x1xf32>
      %18 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
      %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x1xf32>
      %20 = stablehlo.add %17, %19 : tensor<32x1xf32>
      sdy.return %20 : tensor<32x1xf32>
    } : (tensor<32x128xf32>, tensor<128x8xf32>, tensor<8xf32>) -> tensor<32x8xf32>
    %11 = stablehlo.subtract %10, %arg13 : tensor<32x8xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<32x8xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<32x8xf32>, tensor<f32>) -> tensor<32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%13 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %cst_1 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %15 = stablehlo.divide %14, %cst_1 : tensor<f32>
    return %15 : tensor<f32>
  }
  func.func private @relu(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<32x128xf32>
    return %1 : tensor<32x128xf32>
  }
}

// CHECK-LABEL @main
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
