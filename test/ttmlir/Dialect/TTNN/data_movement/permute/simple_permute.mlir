// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @permute(%arg0: tensor<1x4x32x64xf32>) -> tensor<4x64x32x1xf32> {
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 1, 3, 2, 0>
    // CHECK-SAME: tensor<1x4x32x64xf32
    // CHECK-SAME: tensor<4x64x32x1xf32
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 3, 2, 0>}> : (tensor<1x4x32x64xf32>) -> tensor<4x64x32x1xf32>
    return %0 : tensor<4x64x32x1xf32>
  }
}
