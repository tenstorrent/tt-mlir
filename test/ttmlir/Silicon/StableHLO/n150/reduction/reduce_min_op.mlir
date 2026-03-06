// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s -o %t.mlir --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%"
// RUN: FileCheck --input-file=%t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module @jit_reduce_minimum attributes {} {
  func.func public @test_reduce_minimum_4to0dim(%arg0: tensor<128x64x32x96xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64x32x96xf32,
    // CHECK-SAME: -> tensor<f32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0, 1, 2, 3] : (tensor<128x64x32x96xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_minimum_4to1dim(%arg0: tensor<128x64x32x96xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK-LABEL: @test_reduce_minimum_4to1dim(
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64x32x96xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1, 2, 3] : (tensor<128x64x32x96xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_minimum_3to2dim(%arg0: tensor<128x64x96xf32>, %cst_0: tensor<f32>) -> tensor<128x96xf32> {
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64x96xf32,
    // CHECK-SAME: -> tensor<128x96xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1] : (tensor<128x64x96xf32>, tensor<f32>) -> tensor<128x96xf32>
    return %0 : tensor<128x96xf32>
  }

  func.func public @test_reduce_minimum_3to1dim(%arg0: tensor<128x64x96xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64x96xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1, 2] : (tensor<128x64x96xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_minimum_3to0dim(%arg0: tensor<128x64x96xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64x96xf32,
    // CHECK-SAME: -> tensor<f32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0, 1, 2] : (tensor<128x64x96xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_minimum_2to1dim(%arg0: tensor<128x64xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1] : (tensor<128x64xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_minimum_2to0dim(%arg0: tensor<128x64xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x64xf32,
    // CHECK-SAME: -> tensor<f32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0, 1] : (tensor<128x64xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_minimum_1to0dim(%arg0: tensor<128xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [0 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128xf32,
    // CHECK-SAME: -> tensor<f32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
