// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func public @reduce_min_not_keep_dim(%arg0: tensor<128x10x32x4xbf16>) -> tensor<128x32x4xbf16> {
    // CHECK-LABEL: func.func public @reduce_min_not_keep_dim
    %0 = ttir.empty() : tensor<128x32x4xbf16>
    // CHECK: "ttnn.min"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xbf16,
    // CHECK-SAME: -> tensor<128x32x4xbf16,
    %1 = "ttir.min"(%arg0, %0) <{dim_arg = [1: i32], keep_dim = false}> : (tensor<128x10x32x4xbf16>, tensor<128x32x4xbf16>) -> tensor<128x32x4xbf16>
    return %1 : tensor<128x32x4xbf16>
  }
}
