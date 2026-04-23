// RUN: ttmlir-opt %s | FileCheck %s

// Verify basic gather operation along dim 0.
module {
  func.func @gather_2d_dim0(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    // CHECK: "ttir.gather"
    // CHECK-SAME: dim = 0 : i32
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 0 : i32}> : (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Verify gather operation along dim 1.
module {
  func.func @gather_2d_dim1(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xui32>) -> tensor<3x2xbf16> {
    // CHECK: "ttir.gather"
    // CHECK-SAME: dim = 1 : i32
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 1 : i32}> : (tensor<3x5xbf16>, tensor<3x2xui32>) -> tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}

// -----

// Verify gather with negative dimension.
module {
  func.func @gather_negative_dim(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xui32>) -> tensor<4x3xf32> {
    // CHECK: "ttir.gather"
    // CHECK-SAME: dim = -1 : i32
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = -1 : i32}> : (tensor<4x6xf32>, tensor<4x3xui32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// -----

// Verify gather with 3D tensors.
module {
  func.func @gather_3d(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xui32>) -> tensor<2x4x3xf32> {
    // CHECK: "ttir.gather"
    // CHECK-SAME: dim = 2 : i32
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 2 : i32}> : (tensor<2x4x6xf32>, tensor<2x4x3xui32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}
