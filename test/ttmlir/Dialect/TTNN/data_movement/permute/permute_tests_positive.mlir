// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @permute_general(%arg0: tensor<8x32x64x128xf32>) -> tensor<64x8x128x32xf32> {
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 2, 0, 3, 1>
    // CHECK-SAME: tensor<8x32x64x128xf32
    // CHECK-SAME: tensor<64x8x128x32xf32
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 2, 0, 3, 1>}> : (tensor<8x32x64x128xf32>) -> tensor<64x8x128x32xf32>
    return %0 : tensor<64x8x128x32xf32>
  }
}
