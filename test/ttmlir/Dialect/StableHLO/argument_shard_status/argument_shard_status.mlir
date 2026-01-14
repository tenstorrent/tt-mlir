// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// -----
// CHECK-LABEL: module {
module {
  sdy.mesh @mesh = <["x"=1, "batch"=8]>
  func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> (tensor<2048x1024xf32>) {
    %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
    return %0 : tensor<2048x1024xf32>
  }
}

// CHECK: ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<128x2x32x32xf32>>
// CHECK: ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<2048x1024xf32>>

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

// CHECK: ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<f32>>
// CHECK: ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<f32>>
