// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module {
  func.func @permute(%arg0: tensor<1x4x32x64xbf16>) -> tensor<4x32x64x1xbf16> {
    %0 = ttir.empty() : tensor<4x32x64x1xbf16>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 1, 2, 3, 0>
    // CHECK-SAME: tensor<1x4x32x64xbf16
    // CHECK-SAME: tensor<4x32x64x1xbf16
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 1, 2, 3, 0>}> : (tensor<1x4x32x64xbf16>, tensor<4x32x64x1xbf16>) -> tensor<4x32x64x1xbf16>
    return %1 : tensor<4x32x64x1xbf16>
  }
}
