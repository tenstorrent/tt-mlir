// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_loss_dp attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=1, "y"=8]>
  func.func public @main(%arg0: tensor<784x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg2: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg3: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg4: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg5: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg6: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg7: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg8: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg9: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg10: tensor<128x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg11: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg12: tensor<32x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg13: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, []>] manual_axes={"y", "x"} (%arg14: tensor<784x128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128x128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128x128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128x128xf32>, %arg21: tensor<128xf32>, %arg22: tensor<128x128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128x8xf32>, %arg25: tensor<8xf32>, %arg26: tensor<4x784xf32>, %arg27: tensor<4x8xf32>) {
      %1 = stablehlo.dot_general %arg26, %arg14, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x784xf32>, tensor<784x128xf32>) -> tensor<4x128xf32>
      %2 = stablehlo.broadcast_in_dim %arg15, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
      %4 = stablehlo.add %1, %3 : tensor<4x128xf32>
      %5 = func.call @relu(%4) : (tensor<4x128xf32>) -> tensor<4x128xf32>
      %6 = stablehlo.dot_general %5, %arg16, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
      %7 = stablehlo.broadcast_in_dim %arg17, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
      %9 = stablehlo.add %6, %8 : tensor<4x128xf32>
      %10 = func.call @relu_0(%9) : (tensor<4x128xf32>) -> tensor<4x128xf32>
      %11 = stablehlo.dot_general %10, %arg18, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
      %12 = stablehlo.broadcast_in_dim %arg19, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
      %14 = stablehlo.add %11, %13 : tensor<4x128xf32>
      %15 = func.call @relu_1(%14) : (tensor<4x128xf32>) -> tensor<4x128xf32>
      %16 = stablehlo.dot_general %15, %arg20, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
      %17 = stablehlo.broadcast_in_dim %arg21, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
      %19 = stablehlo.add %16, %18 : tensor<4x128xf32>
      %20 = func.call @relu_2(%19) : (tensor<4x128xf32>) -> tensor<4x128xf32>
      %21 = stablehlo.dot_general %20, %arg22, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
      %22 = stablehlo.broadcast_in_dim %arg23, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
      %24 = stablehlo.add %21, %23 : tensor<4x128xf32>
      %25 = func.call @relu_3(%24) : (tensor<4x128xf32>) -> tensor<4x128xf32>
      %26 = stablehlo.dot_general %25, %arg24, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x8xf32>) -> tensor<4x8xf32>
      %27 = stablehlo.broadcast_in_dim %arg25, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
      %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<4x8xf32>
      %29 = stablehlo.add %26, %28 : tensor<4x8xf32>
      %30 = stablehlo.subtract %29, %arg27 : tensor<4x8xf32>
      %31 = stablehlo.multiply %30, %30 : tensor<4x8xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %32 = stablehlo.reduce(%31 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %33 = stablehlo.reduce(%32 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
      %cst_1 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %34 = stablehlo.divide %33, %cst_1 : tensor<f32>
      %35 = "stablehlo.all_reduce"(%34) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg28: tensor<f32>, %arg29: tensor<f32>):
        %37 = stablehlo.add %arg28, %arg29 : tensor<f32>
        stablehlo.return %37 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_2 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
      %36 = stablehlo.divide %35, %cst_2 : tensor<f32>
      sdy.return %36 : tensor<f32>
    } : (tensor<784x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x8xf32>, tensor<8xf32>, tensor<32x784xf32>, tensor<32x8xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @relu(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
  func.func private @relu_0(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
  func.func private @relu_1(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
  func.func private @relu_2(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
  func.func private @relu_3(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
}

// CHECK-LABEL @main
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: return
// CHECK-SAME: : tensor<f32>
