// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// Test: ttir.gather_dim along dim 0 with ui32 index lowers to ttnn.gather.
// CHECK-LABEL: func.func @gather_dim0
// CHECK: "ttnn.gather"(%arg0, %arg1)
// CHECK-SAME: dim = 0 : i32
// CHECK-SAME: sparse_grad = false
module attributes {} {
  func.func @gather_dim0(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 0 : i32, sparse_grad = false}> : (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Test: ttir.gather_dim along dim 1 with ui32 index lowers to ttnn.gather.
// CHECK-LABEL: func.func @gather_dim1
// CHECK: "ttnn.gather"(%arg0, %arg1)
// CHECK-SAME: dim = 1 : i32
// CHECK-SAME: sparse_grad = false
module attributes {} {
  func.func @gather_dim1(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xui32>) -> tensor<3x2xbf16> {
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 1 : i32, sparse_grad = false}> : (tensor<3x5xbf16>, tensor<3x2xui32>) -> tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}

// -----

// Test: ttir.gather_dim with negative dim lowers to ttnn.gather preserving the sign.
// CHECK-LABEL: func.func @gather_negative_dim
// CHECK: "ttnn.gather"(%arg0, %arg1)
// CHECK-SAME: dim = -1 : i32
// CHECK-SAME: sparse_grad = false
module attributes {} {
  func.func @gather_negative_dim(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xui32>) -> tensor<4x3xf32> {
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = -1 : i32, sparse_grad = false}> : (tensor<4x6xf32>, tensor<4x3xui32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// -----

// Test: ttir.gather_dim on 3D tensors along dim 2 lowers to ttnn.gather.
// CHECK-LABEL: func.func @gather_3d
// CHECK: "ttnn.gather"(%arg0, %arg1)
// CHECK-SAME: dim = 2 : i32
// CHECK-SAME: sparse_grad = false
module attributes {} {
  func.func @gather_3d(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xui32>) -> tensor<2x4x3xf32> {
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 2 : i32, sparse_grad = false}> : (tensor<2x4x6xf32>, tensor<2x4x3xui32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}

// -----

// Test: ttir.gather_dim with ui16 index lowers to ttnn.gather without any cast.
// CHECK-LABEL: func.func @gather_ui16_index
// CHECK-NOT: "ttnn.typecast"
// CHECK: "ttnn.gather"(%arg0, %arg1)
// CHECK-SAME: dim = 0 : i32
// CHECK-SAME: sparse_grad = false
module attributes {} {
  func.func @gather_ui16_index(%arg0: tensor<5x3xbf16>, %arg1: tensor<2x3xui16>) -> tensor<2x3xbf16> {
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 0 : i32, sparse_grad = false}> : (tensor<5x3xbf16>, tensor<2x3xui16>) -> tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
}

// -----

// Test: ttir.gather_dim with sparse_grad=true lowers to ttnn.gather with sparse_grad=true.
// CHECK-LABEL: func.func @gather_sparse_grad
// CHECK: "ttnn.gather"(%arg0, %arg1)
// CHECK-SAME: dim = 0 : i32
// CHECK-SAME: sparse_grad = true
module attributes {} {
  func.func @gather_sparse_grad(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    %0 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 0 : i32, sparse_grad = true}> : (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
