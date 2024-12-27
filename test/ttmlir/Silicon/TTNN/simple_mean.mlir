// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true
// These tests are currently failing until a fix for this issue is uplifted
// with new version of Metal: https://github.com/tenstorrent/tt-metal/issues/16104
// TODO(mrakita): Enable and edit these tests after the Metal issue is fixed.
// Tracked by: https://github.com/tenstorrent/tt-mlir/issues/1640

module {
  func.func public @reduce_not_keep_dim(%arg0: tensor<128x10xf32>) -> tensor<128xf32> {
    %0 = tensor.empty() : tensor<128xf32>
    // CHECK: "ttnn.mean"
    // CHECK-SAME: dim = [1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10xf32,
    // CHECK-SAME: -> tensor<128x1xf32,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [128 : i32]
    // CHECK-SAME: tensor<128x1xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %1 = "ttir.mean"(%arg0, %0) <{dim = [1 : i32], keep_dim = false}> : (tensor<128x10xf32>, tensor<128xf32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }

  func.func public @reduce_keep_dim(%arg0: tensor<128x10xf32>) -> tensor<128x1xf32> {
    %0 = tensor.empty() : tensor<128x1xf32>
    // CHECK: "ttnn.mean"
    // CHECK-SAME: dim = [1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10xf32,
    // CHECK-SAME: -> tensor<128x1xf32,
    // CHECK-NOT: "ttnn.reshape"
    %1 = "ttir.mean"(%arg0, %0) <{dim = [1 : i32], keep_dim = true}> : (tensor<128x10xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    return %1 : tensor<128x1xf32>
  }

  func.func public @mean_into_reshape_dim_array(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x2048x1x1xf32> {
    // CHECK: "ttnn.mean"
    // CHECK-SAME: {dim = [-2 : i32], keep_dim = true}
    // CHECK: "ttnn.reshape"
    %1 = tensor.empty() : tensor<1x1x1x2048xf32>
    %2 = "ttir.mean"(%arg0, %1) <{dim = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<1x1x1x2048xf32>
    %3 = tensor.empty() : tensor<1x2048x1x1xf32>
    %4 = "ttir.reshape"(%2, %3) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x2048xf32>, tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
    return %4 : tensor<1x2048x1x1xf32>
  }

  func.func public @mean_into_reshape_dim_scalar(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x2048x1x1xf32> {
    // CHECK: "ttnn.mean"
    // CHECK-SAME: {dim = -2 : si32, keep_dim = true}
    // CHECK: "ttnn.reshape"
    %1 = tensor.empty() : tensor<1x1x1x2048xf32>
    %2 = "ttir.mean"(%arg0, %1) <{dim = -2 : si32, keep_dim = true}> : (tensor<1x1x49x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<1x1x1x2048xf32>
    %3 = tensor.empty() : tensor<1x2048x1x1xf32>
    %4 = "ttir.reshape"(%2, %3) <{shape = [1 : i32, 2048 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x2048xf32>, tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
    return %4 : tensor<1x2048x1x1xf32>
  }
}
