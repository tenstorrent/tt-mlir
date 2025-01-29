// REQUIRES: stablehlo, num-chips-1 || num-chips-2
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true
// These tests are currently failing until a fix for this issue is uplifted
// with new version of Metal: https://github.com/tenstorrent/tt-metal/issues/16104
// TODO(mrakita): Enable and edit these tests after the Metal issue is fixed.
// Tracked by: https://github.com/tenstorrent/tt-mlir/issues/1640

module @jit_reduce_add attributes {} {
  func.func public @test_reduce_add_4to0dim(%arg0: tensor<128x10x32x4xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.sum"
    // CHECK-NOT: dim_arg
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10x32x4xf32,
    // CHECK-SAME: -> tensor<1x1x1x1xf32,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32]
    // CHECK-SAME: tensor<1x1x1x1xf32,
    // CHECK-SAME: -> tensor<1xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<128x10x32x4xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_add_3to2dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<128x4xf32> {
    // CHECK: "ttnn.sum"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10x4xf32,
    // CHECK-SAME: -> tensor<128x1x4xf32,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [128 : i32, 4 : i32]
    // CHECK-SAME: tensor<128x1x4xf32,
    // CHECK-SAME: -> tensor<128x4xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<128x4xf32>
    return %0 : tensor<128x4xf32>
  }

  func.func public @test_reduce_add_3to1dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: "ttnn.sum"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10x4xf32,
    // CHECK-SAME: -> tensor<128x1x1xf32,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [128 : i32]
    // CHECK-SAME: tensor<128x1x1xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_add_3to0dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.sum"
    // CHECK-NOT: dim_arg
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10x4xf32,
    // CHECK-SAME: -> tensor<1x1x1xf32,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32]
    // CHECK-SAME: tensor<1x1x1xf32,
    // CHECK-SAME: -> tensor<1xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_add_2to1dim(%arg0: tensor<128x10xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: "ttnn.sum"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10xf32,
    // CHECK-SAME: -> tensor<128x1xf32,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [128 : i32]
    // CHECK-SAME: tensor<128x1xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_add_2to0dim(%arg0: tensor<128x10xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.sum"
    // CHECK-NOT: dim_arg
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10xf32,
    // CHECK-SAME: -> tensor<1x1xf32,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32]
    // CHECK-SAME: tensor<1x1xf32,
    // CHECK-SAME: -> tensor<1xf32,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_add_1to0dim(%arg0: tensor<128xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: "ttnn.sum"
    // CHECK-NOT: dim_arg
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128xf32,
    // CHECK-SAME: -> tensor<1xf32,
    // CHECK-NOT: "ttnn.reshape"
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
