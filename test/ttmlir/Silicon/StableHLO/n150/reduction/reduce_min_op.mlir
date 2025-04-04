// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s \
// RUN:     --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module @jit_reduce_minimum attributes {} {
  func.func public @test_reduce_minimum_4to0dim(%arg0: tensor<128x10x32x4xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK: "ttnn.min"
    // CEHCK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10x32x4xbf16,
    // CHECK-SAME: -> tensor<1x1x1x1xbf16,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32]
    // CHECK-SAME: tensor<1x1x1x1xbf16,
    // CHECK-SAME: -> tensor<1xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0, 1, 2, 3] : (tensor<128x10x32x4xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_minimum_4to1dim(%arg0: tensor<128x10x32x4xbf16>, %cst_0: tensor<bf16>) -> tensor<128xbf16> {
    // CHECK-LABEL: @test_reduce_minimum_4to1dim(
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xbf16,
    // CHECK-SAME: -> tensor<128xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1, 2, 3] : (tensor<128x10x32x4xbf16>, tensor<bf16>) -> tensor<128xbf16>
    return %0 : tensor<128xbf16>
  }

  func.func public @test_reduce_minimum_3to2dim(%arg0: tensor<128x10x4xbf16>, %cst_0: tensor<bf16>) -> tensor<128x4xbf16> {
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xbf16,
    // CHECK-SAME: -> tensor<128x4xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1] : (tensor<128x10x4xbf16>, tensor<bf16>) -> tensor<128x4xbf16>
    return %0 : tensor<128x4xbf16>
  }

  func.func public @test_reduce_minimum_3to1dim(%arg0: tensor<128x10x4xbf16>, %cst_0: tensor<bf16>) -> tensor<128xbf16> {
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xbf16,
    // CHECK-SAME: -> tensor<128xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1, 2] : (tensor<128x10x4xbf16>, tensor<bf16>) -> tensor<128xbf16>
    return %0 : tensor<128xbf16>
  }

  func.func public @test_reduce_minimum_3to0dim(%arg0: tensor<128x10x4xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK: "ttnn.min"
    // CEHCK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10x4xbf16,
    // CHECK-SAME: -> tensor<1x1x1xbf16,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32]
    // CHECK-SAME: tensor<1x1x1xbf16,
    // CHECK-SAME: -> tensor<1xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0, 1, 2] : (tensor<128x10x4xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_minimum_2to1dim(%arg0: tensor<128x10xbf16>, %cst_0: tensor<bf16>) -> tensor<128xbf16> {
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10xbf16,
    // CHECK-SAME: -> tensor<128xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1] : (tensor<128x10xbf16>, tensor<bf16>) -> tensor<128xbf16>
    return %0 : tensor<128xbf16>
  }

  func.func public @test_reduce_minimum_2to0dim(%arg0: tensor<128x10xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK: "ttnn.min"
    // CEHCK-SAME: dim_arg = [0 : i32, 1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10xbf16,
    // CHECK-SAME: -> tensor<1x1xbf16,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32]
    // CHECK-SAME: tensor<1x1xbf16,
    // CHECK-SAME: -> tensor<1xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0, 1] : (tensor<128x10xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_minimum_1to0dim(%arg0: tensor<128xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK: "ttnn.min"
    // CEHCK-SAME: dim_arg = [0 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128xbf16,
    // CHECK-SAME: -> tensor<1xbf16,
    // CHECK-NOT: "ttnn.reshape"
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0] : (tensor<128xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }
}
