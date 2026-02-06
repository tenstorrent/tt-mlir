// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t


module {
  func.func @identity_permute(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    // CHECK-NOT: ttnn.permute
    // CHECK: return %arg0 : tensor<2x3x4xf32>
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 0, 1, 2>}> :
        (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }

  func.func @consecutive_permute(%arg0: tensor<2x3x4xf32>) -> tensor<4x3x2xf32> {
    // CHECK: ttnn.permute
    // CHECK-SAME: permutation = array<i64: 2, 1, 0>
    // CHECK-NOT: ttnn.permute
    // CHECK-NEXT: return
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 2, 0, 1>}> :
        (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
    %1 = "ttnn.permute"(%0) <{permutation = array<i64: 0, 2, 1>}> :
        (tensor<4x2x3xf32>) -> tensor<4x3x2xf32>
    return %1 : tensor<4x3x2xf32>
  }

  func.func @consecutive_permute_4d(%arg0: tensor<5x2x3x4xf32>) -> tensor<4x3x2x5xf32> {
    // CHECK: ttnn.permute
    // CHECK-SAME: permutation = array<i64: 3, 2, 1, 0>
    // CHECK-NOT: ttnn.permute
    // CHECK-NEXT: return
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 2, 0, 1, 3>}> :
        (tensor<5x2x3x4xf32>) -> tensor<3x5x2x4xf32>
    %1 = "ttnn.permute"(%0) <{permutation = array<i64: 3, 0, 2, 1>}> :
        (tensor<3x5x2x4xf32>) -> tensor<4x3x2x5xf32>
    return %1 : tensor<4x3x2x5xf32>
  }

  func.func @no_fold(%arg0: tensor<2x3x4xf32>) -> (tensor<4x3x2xf32>, tensor<4x2x3xf32>) {
    // CHECK: ttnn.permute
    // CHECK-SAME: permutation = array<i64: 2, 1, 0>
    // CHECK: ttnn.permute
    // CHECK-SAME: permutation = array<i64: 0, 2, 1>

    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 2, 1, 0>}> :
      (tensor<2x3x4xf32>) -> tensor<4x3x2xf32>
    %1 = "ttnn.permute"(%0) <{permutation = array<i64: 0, 2, 1>}> :
      (tensor<4x3x2xf32>) -> tensor<4x2x3xf32>

    return %0, %1 : tensor<4x3x2xf32>, tensor<4x2x3xf32>
  }

}
