// RUN: ttmlir-opt --canonicalize %s 2>&1 | FileCheck %s

// Verify that two permutations are merged into single permutation.
module {
  func.func @forward(%arg0: tensor<1x4x32x64xf32>) -> tensor<32x64x1x4xf32> {
    %0 = tensor.empty() : tensor<4x32x64x1xf32>
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 2, 3, 0, 1>
    // CHECK-NOT: "ttir.permute"
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 1, 2, 3, 0>}> : (tensor<1x4x32x64xf32>, tensor<4x32x64x1xf32>) -> tensor<4x32x64x1xf32>
    %2 = tensor.empty() : tensor<32x64x1x4xf32>
    %3 = "ttir.permute"(%1, %2) <{permutation = array<i64: 1, 2, 3, 0>}> : (tensor<4x32x64x1xf32>, tensor<32x64x1x4xf32>) -> tensor<32x64x1x4xf32>
    return %3 : tensor<32x64x1x4xf32>
  }
}

// Verify that permute op is removed in case second permutation is inverse of first permutation.
module {
  func.func @forward(%arg0: tensor<1x4x32x64xf32>) -> tensor<1x4x32x64xf32> {
    %0 = tensor.empty() : tensor<4x32x64x1xf32>
    // CHECK-NOT: "ttir.permute"
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 1, 2, 3, 0>}> : (tensor<1x4x32x64xf32>, tensor<4x32x64x1xf32>) -> tensor<4x32x64x1xf32>
    %2 = tensor.empty() : tensor<1x4x32x64xf32>
    %3 = "ttir.permute"(%1, %2) <{permutation = array<i64: 3, 0, 1, 2>}> : (tensor<4x32x64x1xf32>, tensor<1x4x32x64xf32>) -> tensor<1x4x32x64xf32>
    return %3 : tensor<1x4x32x64xf32>
  }
}

// Verify that permute op cannot be removed because it has more then 1 user.
module {
  func.func @forward(%arg0: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 1, 2, 0>
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 2, 0, 1>
    // CHECK: "ttir.add"
    %0 = tensor.empty() : tensor<2x2x2xf32>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 1, 2, 0>}> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    %2 = tensor.empty() : tensor<2x2x2xf32>
    %3 = "ttir.permute"(%1, %2) <{permutation = array<i64: 1, 2, 0>}> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    %4 = tensor.empty() : tensor<2x2x2xf32>
    %5 = "ttir.add"(%1, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    return %5 : tensor<2x2x2xf32>
  }
}

// Verify that we are removing identity permutation.
module {
  func.func @forward(%arg0: tensor<1x4x32x64xf32>) -> tensor<1x4x32x64xf32> {
    %0 = tensor.empty() : tensor<1x4x32x64xf32>
    // CHECK-NOT: "ttir.permute"
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x4x32x64xf32>, tensor<1x4x32x64xf32>) -> tensor<1x4x32x64xf32>
    return %1 : tensor<1x4x32x64xf32>
  }
}
