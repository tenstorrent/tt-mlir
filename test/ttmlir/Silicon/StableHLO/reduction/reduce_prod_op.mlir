// REQUIRES: stablehlo, num-chips-1 || num-chips-2
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline \
// RUN:     --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_reduce_prod attributes {} {
  func.func public @test_reduce_prod_4to3dim(%arg0: tensor<128x10x32x4xf32>, %cst_0: tensor<f32>) -> tensor<128x32x4xf32> {
    // CHECK-LABEL: func.func public @test_reduce_prod_4to3dim
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = false
    // CHECK-SAME: dim_arg = 1
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10x32x4xf32,
    // CHECK-SAME: -> tensor<128x32x4xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [1] : (tensor<128x10x32x4xf32>, tensor<f32>) -> tensor<128x32x4xf32>
    return %0 : tensor<128x32x4xf32>
  }

  func.func public @test_reduce_prod_3to2dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<128x10xf32> {
    // CHECK-LABEL: func.func public @test_reduce_prod_3to2dim
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = false
    // CHECK-SAME: dim_arg = 2
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10x4xf32,
    // CHECK-SAME: -> tensor<128x10xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [2] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<128x10xf32>
    return %0 : tensor<128x10xf32>
  }

  func.func public @test_reduce_prod_2to1dim(%arg0: tensor<128x10xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK-LABEL: func.func public @test_reduce_prod_2to1dim
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = false
    // CHECK-SAME: dim_arg = 1
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
}
