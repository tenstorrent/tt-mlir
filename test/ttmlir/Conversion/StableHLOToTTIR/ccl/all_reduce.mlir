// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-to-ttir-pipeline %s | FileCheck %s

module {
  func.func @all_reduce_variadic(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %cst_0 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.add %arg0, %cst_0 : tensor<f32>
    %1 = stablehlo.add %arg1, %cst_1 : tensor<f32>
    %2:2 = "stablehlo.all_reduce"(%0, %1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    return %2#0, %2#1 : tensor<f32>, tensor<f32>
  }
}

// CHECK: "ttir.all_reduce"
// CHECK-SAME: channel_handle = 1
// CHECK-SAME: dim = 0
// CHECK-SAME: reduce_type = #tt.reduce_type<sum>
// CHECK: "ttir.all_reduce"
// CHECK-SAME: channel_handle = 1
// CHECK-SAME: dim = 0
// CHECK-SAME: reduce_type = #tt.reduce_type<sum>
