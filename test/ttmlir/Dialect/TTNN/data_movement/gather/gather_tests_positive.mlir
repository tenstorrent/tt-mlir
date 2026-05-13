// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify gather lowers to ttnn.gather along dim 0.
module attributes {} {
  func.func @gather0(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @gather0
    // CHECK: "ttnn.gather"
    // CHECK-SAME: dim = 0 : i32
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 0 : i32}> : (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Verify gather lowers to ttnn.gather along dim 1.
  func.func @gather1(%arg0: tensor<3x5xbf16>, %arg1: tensor<3x2xui32>) -> tensor<3x2xbf16> {
    // CHECK-LABEL: func.func @gather1
    // CHECK: "ttnn.gather"
    // CHECK-SAME: dim = 1 : i32
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 1 : i32}> : (tensor<3x5xbf16>, tensor<3x2xui32>) -> tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }

  // Verify gather with negative dimension.
  func.func @gather_negative_dim(%arg0: tensor<4x6xf32>, %arg1: tensor<4x3xui32>) -> tensor<4x3xf32> {
    // CHECK-LABEL: func.func @gather_negative_dim
    // CHECK: "ttnn.gather"
    // CHECK-SAME: dim = -1 : i32
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = -1 : i32}> : (tensor<4x6xf32>, tensor<4x3xui32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }

  // Verify gather with 3D tensors.
  func.func @gather_3d(%arg0: tensor<2x4x6xf32>, %arg1: tensor<2x4x3xui32>) -> tensor<2x4x3xf32> {
    // CHECK-LABEL: func.func @gather_3d
    // CHECK: "ttnn.gather"
    // CHECK-SAME: dim = 2 : i32
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 2 : i32}> : (tensor<2x4x6xf32>, tensor<2x4x3xui32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}
