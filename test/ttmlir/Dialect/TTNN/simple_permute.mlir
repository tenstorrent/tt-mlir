// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @forward(%arg0: tensor<1x4x32x64xf32>) -> tensor<4x64x32x1xf32> {
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 1, 3, 2, 0>
    %1 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 3, 2, 0>}> : (tensor<1x4x32x64xf32>) -> tensor<4x64x32x1xf32>
    return %1 : tensor<4x64x32x1xf32>
  }
}
