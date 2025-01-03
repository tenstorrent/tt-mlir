// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @permute_identity(%arg0: tensor<8x32x64x128xf32>) -> tensor<8x32x64x128xf32> {
    %0 = tensor.empty() : tensor<8x32x64x128xf32>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 1, 2, 3>
    // CHECK-SAME: tensor<8x32x64x128xf32
    // CHECK-SAME: tensor<8x32x64x128xf32
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<8x32x64x128xf32>, tensor<8x32x64x128xf32>) -> tensor<8x32x64x128xf32>
    return %1 : tensor<8x32x64x128xf32>
  }

  func.func @permute_general(%arg0: tensor<8x32x64x128xf32>) -> tensor<64x8x128x32xf32> {
    %0 = tensor.empty() : tensor<64x8x128x32xf32>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 2, 0, 3, 1>
    // CHECK-SAME: tensor<8x32x64x128xf32
    // CHECK-SAME: tensor<64x8x128x32xf32
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 2, 0, 3, 1>}> : (tensor<8x32x64x128xf32>, tensor<64x8x128x32xf32>) -> tensor<64x8x128x32xf32>
    return %1 : tensor<64x8x128x32xf32>
  }

  func.func @permute_1d(%arg0: tensor<32xf32>) -> tensor<32xf32> {
    %0 = tensor.empty() : tensor<32xf32>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0>
    // CHECK-SAME: tensor<32xf32
    // CHECK-SAME: tensor<32xf32
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0>}> : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    return %1 : tensor<32xf32>
  }
}
