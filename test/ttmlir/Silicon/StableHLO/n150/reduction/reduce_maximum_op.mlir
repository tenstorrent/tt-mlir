// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module @jit_reduce_maximum attributes {} {
  func.func public @test_reduce_maximum_4to0dim(%arg0: tensor<128x64x32x96xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64x32x96xf32,
    // CHECK-SAME: -> tensor<f32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [0, 1, 2, 3] : (tensor<128x64x32x96xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_maximum_4to1dim(%arg0: tensor<128x64x32x96xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK-LABEL: @test_reduce_maximum_4to1dim(
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64x32x96xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [1, 2, 3] : (tensor<128x64x32x96xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_maximum_3to2dim(%arg0: tensor<128x64x96xf32>, %cst_0: tensor<f32>) -> tensor<128x96xf32> {
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64x96xf32,
    // CHECK-SAME: -> tensor<128x96xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<128x64x96xf32>, tensor<f32>) -> tensor<128x96xf32>
    return %0 : tensor<128x96xf32>
  }

  func.func public @test_reduce_maximum_3to1dim(%arg0: tensor<128x64x96xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64x96xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [1, 2] : (tensor<128x64x96xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_maximum_3to0dim(%arg0: tensor<128x64x96xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64x96xf32,
    // CHECK-SAME: -> tensor<f32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [0, 1, 2] : (tensor<128x64x96xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_maximum_2to1dim(%arg0: tensor<128x64xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<128x64xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_maximum_2to0dim(%arg0: tensor<128x64xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64xf32,
    // CHECK-SAME: -> tensor<f32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<128x64xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_maximum_1to0dim(%arg0: tensor<128xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [0 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128xf32,
    // CHECK-SAME: -> tensor<f32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
