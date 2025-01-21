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
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10x32x4xf32,
    // CHECK-SAME: -> tensor<128x1x32x4xf32,
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [128 : i32, 32 : i32, 4 : i32]
    // CHECK-SAME: tensor<128x1x32x4xf32,
    // CHECK-SAME: -> tensor<128x32x4xf32,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [1: i32], keep_dim = false}> : (tensor<128x10x32x4xf32>, tensor<128x32x4xf32>) -> tensor<128x32x4xf32>
    return %1 : tensor<128x32x4xf32>
  }

  func.func public @reduce_prod_keep_dim(%arg0: tensor<128x10x32x4xf32>) -> tensor<128x1x32x4xf32> {
    // CHECK-LABEL: func.func public @reduce_prod_keep_dim
    %0 = tensor.empty() : tensor<128x1x32x4xf32>
    // CHECK-NOT: "ttnn.reshape"
    // CHECK: "ttnn.prod"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10x32x4xf32,
    // CHECK-SAME: -> tensor<128x1x32x4xf32,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<128x10x32x4xf32>, tensor<128x1x32x4xf32>) -> tensor<128x1x32x4xf32>
    return %1 : tensor<128x1x32x4xf32>
  }

  func.func public @reduce_prod_keep_dim_0(%arg0: tensor<128x10x32x4xbf16>) -> tensor<128x10x32x4xbf16> {
    // CHECK-LABEL: func.func public @reduce_prod_keep_dim
    %0 = tensor.empty() : tensor<128x10x32x4xbf16>
    // CHECK-NOT: "ttnn.reshape"
    // CHECK: "ttnn.prod"
    // CaHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x10x32x4xbf16,
    // CHEaCK-SAME: -> tensor<128x1x32x4xbf16,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [0: i32, 1 : i32, 2: i32, 3 : i32], keep_dim = true}> : (tensor<128x10x32x4xbf16>, tensor<128x10x32x4xbf16>) -> tensor<128x10x32x4xbf16>
    return %1 : tensor<128x10x32x4xbf16>
  }
}
