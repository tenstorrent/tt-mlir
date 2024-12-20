// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_reduce_add attributes {} {
  func.func public @test_reduce_add_4to0dim(%arg0: tensor<128x10x32x4xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: tensor.empty
    // CHECK: "ttir.sum"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xf32>,
    // CHECK-SAME: -> tensor<1xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<128x10x32x4xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_add_3to2dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<128x4xf32> {
    // CHECK: tensor.empty
    // CHECK: "ttir.sum"
    // CHECK-SAME: dim_arg = [1 : i32],
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xf32>,
    // CHECK-SAME: -> tensor<128x4xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<128x4xf32>
    return %0 : tensor<128x4xf32>
  }

  func.func public @test_reduce_add_3to1dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: tensor.empty
    // CHECK: "ttir.sum"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32],
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xf32>,
    // CHECK-SAME: -> tensor<128xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_add_3to0dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: tensor.empty
    // CHECK: "ttir.sum"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32],
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xf32>,
    // CHECK-SAME: -> tensor<1xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_add_2to1dim(%arg0: tensor<128x10xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: tensor.empty
    // CHECK: "ttir.sum"
    // CHECK-SAME: dim_arg = [1 : i32],
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10xf32>,
    // CHECK-SAME: -> tensor<128xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_add_2to0dim(%arg0: tensor<128x10xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: tensor.empty
    // CHECK: "ttir.sum"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32],
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10xf32>,
    // CHECK-SAME: -> tensor<1xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_add_1to0dim(%arg0: tensor<128xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: tensor.empty
    // CHECK: "ttir.sum"
    // CHECK-SAME: dim_arg = [0 : i32],
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128xf32>,
    // CHECK-SAME: -> tensor<1xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
