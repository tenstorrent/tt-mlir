// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_reduce_prod attributes {} {
  func.func public @test_reduce_prod_4to0dim_bfloat16(%arg0: tensor<128x10x32x4xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_4to0dim_bfloat16
    // CHECK-NOT: dim_arg
    // CHECK: %[[PROD:[0-9]+]] = "ttnn.prod"
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xbf16,
    // CHECK-SAME: -> tensor<bf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0, 1, 2, 3] : (tensor<128x10x32x4xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_prod_3to0dim(%arg0: tensor<128x10x4xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_3to0dim
    // CHECK-NOT: dim_arg
    // CHECK: %[[PROD:[0-9]+]] = "ttnn.prod"
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xbf16,
    // CHECK-SAME: -> tensor<bf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0, 1, 2] : (tensor<128x10x4xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_prod_2to0dim(%arg0: tensor<128x10xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_2to0dim
    // CHECK-NOT: dim_arg
    // CHECK: %[[PROD:[0-9]+]] = "ttnn.prod"
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10xbf16,
    // CHECK-SAME: -> tensor<bf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0, 1] : (tensor<128x10xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_prod_4to0dim_float32(%arg0: tensor<128x10x32x4xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK-LABEL: func.func public @test_reduce_prod_4to0dim_float32
    // CHECK-NOT: dim_arg
    // CHECK: %[[PROD:[0-9]+]] = "ttnn.prod"
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xbf16,
    // CHECK-SAME: -> tensor<bf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0, 1, 2, 3] : (tensor<128x10x32x4xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
