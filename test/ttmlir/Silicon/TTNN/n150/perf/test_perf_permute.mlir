// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @permute(%arg0: tensor<1x4x32x64xf32>) -> tensor<4x64x32x1xf32> {
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 1, 3, 2, 0>
    // CHECK-SAME: tensor<1x4x32x64xf32
    // CHECK-SAME: tensor<4x64x32x1xf32
    %1 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 3, 2, 0>}> : (tensor<1x4x32x64xf32>) -> tensor<4x64x32x1xf32>
    return %1 : tensor<4x64x32x1xf32>
  }
}
