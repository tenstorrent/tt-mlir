// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,4" %s | FileCheck %s
// Unit tests for ttnn all_gather op

// -----

// Verify lowering of ttir all_gather to ttnn ops
module attributes {} {
  // CHECK-LABEL: all_gather_positive
  func.func @all_gather_positive(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x128xbf16> {
    %0 = ttir.empty() : tensor<1x1x32x128xbf16>
    %1 = "ttir.all_gather"(%arg0, %0) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x32x32xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 3, 1, 2, 0>
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 3, 1, 2, 0>
    return %1 : tensor<1x1x32x128xbf16>
  }
}

// -----

// Verify op folding for single mesh device communication
module attributes {} {
  // CHECK-LABEL: all_gather_positive_folding
  func.func @all_gather_positive_folding(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16> {
    %0 = ttir.empty() : tensor<1x1x32x128xbf16>
    %1 = "ttir.all_gather"(%arg0, %0) <{all_gather_dim = 3 : si32, cluster_axis = 0 : ui32}> : (tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    // CHECK-NOT: "ttnn.all_gather"
    return %1 : tensor<1x1x32x128xbf16>
  }
}

// -----

// Verify that the we reshape the input of the all_gather op, on 2D tensors
module attributes {} {
  // CHECK-LABEL: all_gather_reshape_2D
  func.func @all_gather_reshape_2D(%arg0: tensor<32x128xbf16>) -> tensor<128x128xbf16> {
    %0 = ttir.empty() : tensor<128x128xbf16>
    %1 = "ttir.all_gather"(%arg0, %0) <{all_gather_dim = 0 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 32 : i32, 128 : i32]
    // CHECK-SAME: tensor<32x128xbf16
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: {shape = [128 : i32, 128 : i32]}
    // CHECK-SAME: tensor<4x32x128xbf16
    return %1 : tensor<128x128xbf16>
  }
}

// -----

// Verify that the we reshape the input of the all_gather op, on 1D tensors
module attributes {} {
  // CHECK-LABEL: all_gather_reshape_1D
  func.func @all_gather_reshape_1D(%arg0: tensor<32xbf16>) -> tensor<128xbf16> {
    %0 = ttir.empty() : tensor<128xbf16>
    %1 = "ttir.all_gather"(%arg0, %0) <{all_gather_dim = 0 : si32, cluster_axis = 1 : ui32}> : (tensor<32xbf16>, tensor<128xbf16>) -> tensor<128xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 32 : i32, 1 : i32]
    // CHECK-SAME: tensor<32xbf16
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: {shape = [128 : i32]}
    // CHECK-SAME: tensor<4x32x1xbf16
    return %1 : tensor<128xbf16>
  }
}
