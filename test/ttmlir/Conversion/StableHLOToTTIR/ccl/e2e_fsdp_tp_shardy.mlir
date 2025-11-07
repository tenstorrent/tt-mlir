// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_loss_fsdp_tp attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func public @main(%arg0: tensor<784x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128x128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128x128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x8xf32>, %arg11: tensor<8xf32>, %arg12: tensor<32x784xf32>, %arg13: tensor<32x8xf32>) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13) in_shardings=[<@mesh, [{"y", "x"}, {}]>, <@mesh, [{"y", "x"}]>, <@mesh, [{"y", "x"}, {}]>, <@mesh, [{"y", "x"}]>, <@mesh, [{"y", "x"}, {}]>, <@mesh, [{"y", "x"}]>, <@mesh, [{"y", "x"}, {}]>, <@mesh, [{"y", "x"}]>, <@mesh, [{"y", "x"}, {}]>, <@mesh, [{"y", "x"}]>, <@mesh, [{"y", "x"}, {}]>, <@mesh, [{"y", "x"}]>, <@mesh, [{"x"}, {"y"}]>, <@mesh, [{"x"}, {"y"}]>] out_shardings=[<@mesh, []>] manual_axes={"x", "y"} (%arg14: tensor<98x128xf32>, %arg15: tensor<16xf32>, %arg16: tensor<16x128xf32>, %arg17: tensor<16xf32>, %arg18: tensor<16x128xf32>, %arg19: tensor<16xf32>, %arg20: tensor<16x128xf32>, %arg21: tensor<16xf32>, %arg22: tensor<16x128xf32>, %arg23: tensor<16xf32>, %arg24: tensor<16x8xf32>, %arg25: tensor<1xf32>, %arg26: tensor<16x196xf32>, %arg27: tensor<16x2xf32>) {
      %1 = "stablehlo.all_gather"(%arg14) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<98x128xf32>) -> tensor<196x128xf32>
      %2 = "stablehlo.all_gather"(%arg15) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<32xf32>
      %3 = stablehlo.dot_general %arg26, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x196xf32>, tensor<196x128xf32>) -> tensor<16x128xf32>
      %4 = "stablehlo.reduce_scatter"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg28: tensor<f32>, %arg29: tensor<f32>):
        %56 = stablehlo.add %arg28, %arg29 : tensor<f32>
        stablehlo.return %56 : tensor<f32>
      }) : (tensor<16x128xf32>) -> tensor<16x32xf32>
      %5 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
      %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
      %7 = stablehlo.add %4, %6 : tensor<16x32xf32>
      %8 = func.call @relu(%7) : (tensor<16x32xf32>) -> tensor<16x32xf32>
      %9 = "stablehlo.all_gather"(%arg16) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<32x128xf32>
      %10 = "stablehlo.all_gather"(%arg17) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<32xf32>
      %11 = stablehlo.dot_general %8, %9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x128xf32>) -> tensor<16x128xf32>
      %12 = "stablehlo.reduce_scatter"(%11) <{channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg28: tensor<f32>, %arg29: tensor<f32>):
        %56 = stablehlo.add %arg28, %arg29 : tensor<f32>
        stablehlo.return %56 : tensor<f32>
      }) : (tensor<16x128xf32>) -> tensor<16x32xf32>
      %13 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
      %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
      %15 = stablehlo.add %12, %14 : tensor<16x32xf32>
      %16 = func.call @relu_0(%15) : (tensor<16x32xf32>) -> tensor<16x32xf32>
      %17 = "stablehlo.all_gather"(%arg18) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<32x128xf32>
      %18 = "stablehlo.all_gather"(%arg19) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 8, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<32xf32>
      %19 = stablehlo.dot_general %16, %17, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x128xf32>) -> tensor<16x128xf32>
      %20 = "stablehlo.reduce_scatter"(%19) <{channel_handle = #stablehlo.channel_handle<handle = 9, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg28: tensor<f32>, %arg29: tensor<f32>):
        %56 = stablehlo.add %arg28, %arg29 : tensor<f32>
        stablehlo.return %56 : tensor<f32>
      }) : (tensor<16x128xf32>) -> tensor<16x32xf32>
      %21 = stablehlo.broadcast_in_dim %18, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
      %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
      %23 = stablehlo.add %20, %22 : tensor<16x32xf32>
      %24 = func.call @relu_1(%23) : (tensor<16x32xf32>) -> tensor<16x32xf32>
      %25 = "stablehlo.all_gather"(%arg20) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 10, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<32x128xf32>
      %26 = "stablehlo.all_gather"(%arg21) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 11, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<32xf32>
      %27 = stablehlo.dot_general %24, %25, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x128xf32>) -> tensor<16x128xf32>
      %28 = "stablehlo.reduce_scatter"(%27) <{channel_handle = #stablehlo.channel_handle<handle = 12, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg28: tensor<f32>, %arg29: tensor<f32>):
        %56 = stablehlo.add %arg28, %arg29 : tensor<f32>
        stablehlo.return %56 : tensor<f32>
      }) : (tensor<16x128xf32>) -> tensor<16x32xf32>
      %29 = stablehlo.broadcast_in_dim %26, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
      %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
      %31 = stablehlo.add %28, %30 : tensor<16x32xf32>
      %32 = func.call @relu_2(%31) : (tensor<16x32xf32>) -> tensor<16x32xf32>
      %33 = "stablehlo.all_gather"(%arg22) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 13, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<32x128xf32>
      %34 = "stablehlo.all_gather"(%arg23) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 14, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<32xf32>
      %35 = stablehlo.dot_general %32, %33, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x128xf32>) -> tensor<16x128xf32>
      %36 = "stablehlo.reduce_scatter"(%35) <{channel_handle = #stablehlo.channel_handle<handle = 15, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg28: tensor<f32>, %arg29: tensor<f32>):
        %56 = stablehlo.add %arg28, %arg29 : tensor<f32>
        stablehlo.return %56 : tensor<f32>
      }) : (tensor<16x128xf32>) -> tensor<16x32xf32>
      %37 = stablehlo.broadcast_in_dim %34, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
      %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
      %39 = stablehlo.add %36, %38 : tensor<16x32xf32>
      %40 = func.call @relu_3(%39) : (tensor<16x32xf32>) -> tensor<16x32xf32>
      %41 = "stablehlo.all_gather"(%arg24) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 16, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16x8xf32>) -> tensor<32x8xf32>
      %42 = "stablehlo.all_gather"(%arg25) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 17, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<1xf32>) -> tensor<2xf32>
      %43 = stablehlo.dot_general %40, %41, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
      %44 = "stablehlo.reduce_scatter"(%43) <{channel_handle = #stablehlo.channel_handle<handle = 18, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
      ^bb0(%arg28: tensor<f32>, %arg29: tensor<f32>):
        %56 = stablehlo.add %arg28, %arg29 : tensor<f32>
        stablehlo.return %56 : tensor<f32>
      }) : (tensor<16x8xf32>) -> tensor<16x2xf32>
      %45 = stablehlo.broadcast_in_dim %42, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
      %46 = stablehlo.broadcast_in_dim %45, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<16x2xf32>
      %47 = stablehlo.add %44, %46 : tensor<16x2xf32>
      %48 = stablehlo.subtract %47, %arg27 : tensor<16x2xf32>
      %49 = stablehlo.multiply %48, %48 : tensor<16x2xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %50 = stablehlo.reduce(%49 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
      %51 = "stablehlo.all_reduce"(%50) <{channel_handle = #stablehlo.channel_handle<handle = 19, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg28: tensor<f32>, %arg29: tensor<f32>):
        %56 = stablehlo.add %arg28, %arg29 : tensor<f32>
        stablehlo.return %56 : tensor<f32>
      }) : (tensor<16xf32>) -> tensor<16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %52 = stablehlo.reduce(%51 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
      %cst_1 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
      %53 = stablehlo.divide %52, %cst_1 : tensor<f32>
      %54 = "stablehlo.all_reduce"(%53) <{channel_handle = #stablehlo.channel_handle<handle = 20, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
      ^bb0(%arg28: tensor<f32>, %arg29: tensor<f32>):
        %56 = stablehlo.add %arg28, %arg29 : tensor<f32>
        stablehlo.return %56 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %55 = stablehlo.divide %54, %cst_2 : tensor<f32>
      sdy.return %55 : tensor<f32>
    } : (tensor<784x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x8xf32>, tensor<8xf32>, tensor<32x784xf32>, tensor<32x8xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @relu(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x32xf32>
    return %1 : tensor<16x32xf32>
  }
  func.func private @relu_0(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x32xf32>
    return %1 : tensor<16x32xf32>
  }
  func.func private @relu_1(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x32xf32>
    return %1 : tensor<16x32xf32>
  }
  func.func private @relu_2(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x32xf32>
    return %1 : tensor<16x32xf32>
  }
  func.func private @relu_3(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x32xf32>
    return %1 : tensor<16x32xf32>
  }
}

// CHECK-LABEL @main
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 2, 4>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: 0, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 2, 4>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: return
// CHECK-SAME: : tensor<f32>
