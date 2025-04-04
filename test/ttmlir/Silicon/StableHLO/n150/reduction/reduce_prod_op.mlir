// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline \
// RUN:     --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_reduce_prod attributes {} {
  func.func public @test_reduce_prod_4to3dim(%arg0: tensor<128x10x32x4xbf16>, %cst_0: tensor<f32>) -> tensor<128x32x4xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_4to3dim
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = false
    // CHECK-SAME: dim_arg = 1
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10x32x4xbf16,
    // CHECK-SAME: -> tensor<128x32x4xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [1] : (tensor<128x10x32x4xbf16>, tensor<f32>) -> tensor<128x32x4xbf16>
    return %0 : tensor<128x32x4xbf16>
  }

  func.func public @test_reduce_prod_3to2dim(%arg0: tensor<128x10x4xbf16>, %cst_0: tensor<f32>) -> tensor<128x10xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_3to2dim
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = false
    // CHECK-SAME: dim_arg = 2
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10x4xbf16,
    // CHECK-SAME: -> tensor<128x10xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [2] : (tensor<128x10x4xbf16>, tensor<f32>) -> tensor<128x10xbf16>
    return %0 : tensor<128x10xbf16>
  }

  func.func public @test_reduce_prod_2to1dim(%arg0: tensor<128x10xbf16>, %cst_0: tensor<f32>) -> tensor<128xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_2to1dim
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = false
    // CHECK-SAME: dim_arg = 1
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: (tensor<128x10xbf16,
    // CHECK-SAME: -> tensor<128xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [1] : (tensor<128x10xbf16>, tensor<f32>) -> tensor<128xbf16>
    return %0 : tensor<128xbf16>
  }
}
