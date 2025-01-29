// REQUIRES: num-chips-1 || num-chips-2
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
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10xf32,
    // CHECK-SAME: -> tensor<128x1xf32,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [128 : i32]
    // CHECK-SAME: tensor<128x1xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %1 = "ttir.max"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x10xf32>, tensor<128xf32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }

  func.func public @reduce_keep_dim(%arg0: tensor<128x10xf32>) -> tensor<128x1xf32> {
    %0 = tensor.empty() : tensor<128x1xf32>
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10xf32,
    // CHECK-SAME: -> tensor<128x1xf32,
    // CHECK-NOT: "ttnn.reshape"
    %1 = "ttir.max"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<128x10xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    return %1 : tensor<128x1xf32>
  }
}
