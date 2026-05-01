// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test assumes the difference in values of %arg2 and %arg1 will match output shape.
module attributes {} {
  func.func @dynamic_slice(%arg0: tensor<4x32x32xbf16>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<2x16x16xbf16> {
    // CHECK: = "ttnn.slice_dynamic"
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2) <{step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xbf16>, tensor<3xi32>, tensor<3xi32>) -> tensor<2x16x16xbf16>
    return %1 : tensor<2x16x16xbf16>
  }

  // Unsigned integer narrower than 32 bits: input and output must be promoted
  // to UInt32 (line 339 in TTNNWorkaroundsPass.cpp).
  func.func @dynamic_slice_ui8(%arg0: tensor<4x32xui8>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2x16xui8> {
    // CHECK: = "ttnn.slice_dynamic"
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2) <{step = [1: i32, 1: i32]}> : (tensor<4x32xui8>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x16xui8>
    return %1 : tensor<2x16xui8>
  }
}
