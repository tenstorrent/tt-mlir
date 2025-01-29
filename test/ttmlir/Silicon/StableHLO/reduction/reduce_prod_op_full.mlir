// REQUIRES: stablehlo, num-chips-1 || num-chips-2
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline \
// RUN:     --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s
// UNSUPPORTED: true
// These tests are failing due to two reasons.
// 1. Input tensor rank is less than 4.
// tt-mlir issue: https://github.com/tenstorrent/tt-mlir/issues/1859
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/16909
// 2. Inconsistent output tensor shape compared to other tt-metal reduction ops
// (e.g. ttnn.sum, ttnn.max, etc.)
// tt-mlir issue: https://github.com/tenstorrent/tt-mlir/issues/1890
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/16915
// These issues can be fixed by updating workarounds in tt-mlir.
// TODO(mmanzoor): Enable these tests either issues are fixed in tt-metal or
// workarounds are added in tt-mlir.

module @jit_reduce_prod attributes {} {
  func.func public @test_reduce_prod_4to0dim(%arg0: tensor<128x10x32x4xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_4to0dim
    // CHECK-NOT: dim_arg
    // CHECK: %[[PROD:[0-9]+]] = "ttnn.prod"
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: (tensor<128x10x32x4xbf16,
    // CHECK-SAME: -> tensor<1x1x1x1xbf16,
    // CHECK: "ttnn.reshape"(%[[PROD]])
    // CHECK-SAME: shape = [1 : i32]
    // CHECK-SAME: tensor<1x1x1x1xbf16,
    // CHECK-SAME: -> tensor<1xbf16
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0, 1, 2, 3] : (tensor<128x10x32x4xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_prod_3to0dim(%arg0: tensor<128x10x4xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_3to0dim
    // CHECK-NOT: dim_arg
    // CHECK: %[[PROD:[0-9]+]] = "ttnn.prod"
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: (tensor<128x10x4xbf16,
    // CHECK-SAME: -> tensor<1x1x1xbf16,
    // CHECK: "ttnn.reshape"(%[[PROD]])
    // CHECK-SAME: shape = [1 : i32]
    // CHECK-SAME: tensor<1x1x1xbf16,
    // CHECK-SAME: -> tensor<1xbf16
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0, 1, 2] : (tensor<128x10x4xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_prod_2to0dim(%arg0: tensor<128x10xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_2to0dim
    // CHECK-NOT: dim_arg
    // CHECK: %[[PROD:[0-9]+]] = "ttnn.prod"
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: (tensor<128x10xbf16,
    // CHECK-SAME: -> tensor<1x1xbf16,
    // CHECK: "ttnn.reshape"(%[[PROD]])
    // CHECK-SAME: shape = [1 : i32]
    // CHECK-SAME: tensor<1x1xbf16,
    // CHECK-SAME: -> tensor<1xbf16
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0, 1] : (tensor<128x10xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_reduce_prod_1to0dim(%arg0: tensor<128xbf16>, %cst_0: tensor<bf16>) -> tensor<bf16> {
    // CHECK-LABEL: func.func public @test_reduce_prod_1to0dim
    // CHECK-NOT: dim_arg
    // CHECK-NOT: ttnn.reshape
    // CHECK: %[[PROD:[0-9]+]] = "ttnn.prod"
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: (tensor<128xbf16,
    // CHECK-SAME: -> tensor<1xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.multiply across dimensions = [0] : (tensor<128xbf16>, tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }
}
