// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_reduce_prod attributes {} {
  func.func public @test_reduce_prod_4to3dim(%arg0: tensor<128x10x32x4xf32>, %cst_0: tensor<f32>) -> tensor<128x32x4xf32> {
    // CHECK-LABEL: func.func public @test_reduce_prod_4to3dim
    // CHECK: tensor.empty
    // CHECK: "ttir.prod"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xf32>
    // CHECK-SAME: -> tensor<128x32x4xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [1] : (tensor<128x10x32x4xf32>, tensor<f32>) -> tensor<128x32x4xf32>
    return %0 : tensor<128x32x4xf32>
  }

  func.func public @test_reduce_prod_4to0dim(%arg0: tensor<128x10x32x4xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_4to0dim
    // CHECK: tensor.empty
    // CHECK: "ttir.prod"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xbf16>
    // CHECK-SAME: -> tensor<1xbf16>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0, 1, 2, 3] : (tensor<128x10x32x4xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_prod_3to2dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<128x4xf32> {
    // CHECK: tensor.empty
    // CHECK: "ttir.prod"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xf32>
    // CHECK-SAME: -> tensor<128x4xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [1] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<128x4xf32>
    return %0 : tensor<128x4xf32>
  }

  func.func public @test_reduce_prod_3to0dim(%arg0: tensor<128x10x4xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK: tensor.empty
    // CHECK: "ttir.prod"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xbf16>
    // CHECK-SAME: -> tensor<1xbf16>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0, 1, 2] : (tensor<128x10x4xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_prod_2to1dim(%arg0: tensor<128x10xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: tensor.empty
    // CHECK: "ttir.prod"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10xf32>
    // CHECK-SAME: -> tensor<128xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_prod_2to0dim(%arg0: tensor<128x10xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK: tensor.empty
    // CHECK: "ttir.prod"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10xbf16>
    // CHECK-SAME: -> tensor<1xbf16>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0, 1] : (tensor<128x10xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_prod_1to0dim(%arg0: tensor<128xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK: tensor.empty
    // CHECK: "ttir.prod"
    // CHECK-SAME: dim_arg = [0 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128xbf16>
    // CHECK-SAME: -> tensor<1xbf16>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0] : (tensor<128xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }
}
