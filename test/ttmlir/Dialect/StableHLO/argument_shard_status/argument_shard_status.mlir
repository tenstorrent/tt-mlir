// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: module {
module {
  func.func public @main(%arg0: tensor<1024x1024xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}) -> (tensor<1024x1024xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} : (tensor<1024x1024xf32>) -> tensor<128x1024xf32>
    %2 = stablehlo.custom_call @Sharding(%1) {mhlo.sharding = "{manual}"} : (tensor<128x1024xf32>) -> tensor<128x1024xf32>
    %3 = stablehlo.custom_call @SPMDShardToFullShape(%2) {mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x1024xf32>) -> tensor<1024x1024xf32>
    return %3 : tensor<1024x1024xf32>
  }
}

// CHECK: ttcore.shard_status = #ttcore.shard_status<presharded>
// CHECK: ttcore.shard_status = #ttcore.shard_status<presharded>

// -----
// CHECK-LABEL: module {
module {
  func.func @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x64xf32>
    %3 = stablehlo.custom_call @Sharding(%1) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x64xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x2048x64xf32>) -> tensor<1x1x8192x512xf32>
    return %4 : tensor<1x1x8192x512xf32>
  }
}

// CHECK: ttcore.shard_status = #ttcore.shard_status<unsharded>
// CHECK: ttcore.shard_status = #ttcore.shard_status<unsharded>

// -----
// CHECK-LABEL: module {
module {
  sdy.mesh @mesh = <["x"=1, "batch"=8]>
  func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> (tensor<2048x1024xf32>) {
    %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
    return %0 : tensor<2048x1024xf32>
  }
}

// CHECK: ttcore.shard_status = #ttcore.shard_status<presharded>
// CHECK: ttcore.shard_status = #ttcore.shard_status<unsharded>

// -----
// CHECK-LABEL: module {
module {
  sdy.mesh @mesh = <["x"=1, "batch"=8]>
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, []>] out_shardings=[<@mesh, []>] manual_axes={"x", "batch"} (%arg1: tensor<f32>) {
      sdy.return %arg1 : tensor<f32>
    } : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

// CHECK: ttcore.shard_status = #ttcore.shard_status<unsharded>
// CHECK: ttcore.shard_status = #ttcore.shard_status<unsharded>
