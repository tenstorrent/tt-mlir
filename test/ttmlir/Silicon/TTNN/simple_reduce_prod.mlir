// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func public @reduce_prod_not_keep_dim(%arg0: tensor<128x10x32x4xf32>) -> tensor<128x32x4xf32> {
    // CHECK-LABEL: func.func public @reduce_prod_not_keep_dim
    %0 = tensor.empty() : tensor<128x32x4xf32>
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = false
    // CHECK-SAME: dim_arg = 1
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xf32,
    // CHECK-SAME: -> tensor<128x32x4xf32,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [1: i32], keep_dim = false}> : (tensor<128x10x32x4xf32>, tensor<128x32x4xf32>) -> tensor<128x32x4xf32>
    return %1 : tensor<128x32x4xf32>
  }

  func.func public @reduce_prod_keep_dim(%arg0: tensor<128x10x32x4xf32>) -> tensor<128x1x32x4xf32> {
    // CHECK-LABEL: func.func public @reduce_prod_keep_dim
    %0 = tensor.empty() : tensor<128x1x32x4xf32>
    // CHECK: "ttnn.prod"
    // CHECK: all_dimensions = false
    // CHECK-SAME: dim_arg = 1
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10x32x4xf32,
    // CHECK-SAME: -> tensor<128x1x32x4xf32,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<128x10x32x4xf32>, tensor<128x1x32x4xf32>) -> tensor<128x1x32x4xf32>
    return %1 : tensor<128x1x32x4xf32>
  }
}
