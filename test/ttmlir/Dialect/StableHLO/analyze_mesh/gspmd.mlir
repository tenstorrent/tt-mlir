// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: module {
module {
  // CHECK: sdy.mesh @mesh = <["x"=1, "y"=8]>
  func.func @llmbox_mesh(%arg0: tensor<8x1024xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1]0,1,2,3,4,5,6,7}"} : (tensor<8x1024xf32>) -> tensor<8x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8x1024xf32>) -> tensor<8x128xf32>
    %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<8x128xf32>) -> tensor<8x128xf32>
    return %2 : tensor<8x128xf32>
  }
}

// -----

// CHECK-LABEL: module {
module {
  // CHECK: sdy.mesh @mesh = <["x"=8, "y"=4]>
  func.func @tg_mesh(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x64xf32>
    %3 = stablehlo.custom_call @Sharding(%1) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x64xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x2048x64xf32>) -> tensor<1x1x8192x512xf32>
    return %4 : tensor<1x1x8192x512xf32>
  }
}
