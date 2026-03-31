// RUN: ttmlir-opt %s | FileCheck %s

// Verify basic gather_dim operation along dim 0.
module {
  func.func @gather_dim_2d_dim0(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = 0 : i32
    // CHECK-SAME: sparse_grad = false
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 0 : i32, sparse_grad = false}> : (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Verify gather_dim operation along dim 1.
module {
  func.func @gather_dim_2d_dim1(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xui64>) -> tensor<3x2xbf16> {
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = 1 : i32
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 1 : i32, sparse_grad = false}> : (tensor<3x5xbf16>, tensor<3x2xui64>) -> tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}

// -----

// Verify gather_dim with negative dimension.
module {
  func.func @gather_dim_negative_dim(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xui32>) -> tensor<4x3xf32> {
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = -1 : i32
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = -1 : i32, sparse_grad = false}> : (tensor<4x6xf32>, tensor<4x3xui32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// -----

// Verify gather_dim with 3D tensors.
module {
  func.func @gather_dim_3d(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xui32>) -> tensor<2x4x3xf32> {
    // CHECK: "ttir.gather_dim"
    // CHECK-SAME: dim = 2 : i32
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 2 : i32, sparse_grad = false}> : (tensor<2x4x6xf32>, tensor<2x4x3xui32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}
