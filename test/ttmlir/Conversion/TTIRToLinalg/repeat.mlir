// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module {
  // Test basic repeat along single dimension
  func.func @repeat_dim0(%arg0: tensor<2x3xf32>) -> tensor<4x3xf32> {
    // CHECK-LABEL: func.func @repeat_dim0
    // CHECK: tosa.const_shape
    // CHECK: tosa.tile
    // CHECK-NOT: ttir.repeat
    %0 = "ttir.repeat"(%arg0) <{repeat_dimensions = array<i64: 2, 1>}> : (tensor<2x3xf32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }

  // Test repeat along last dimension
  func.func @repeat_dim1(%arg0: tensor<2x3xf32>) -> tensor<2x9xf32> {
    // CHECK-LABEL: func.func @repeat_dim1
    // CHECK: tosa.const_shape
    // CHECK: tosa.tile
    // CHECK-NOT: ttir.repeat
    %0 = "ttir.repeat"(%arg0) <{repeat_dimensions = array<i64: 1, 3>}> : (tensor<2x3xf32>) -> tensor<2x9xf32>
    return %0 : tensor<2x9xf32>
  }

  // Test repeat along multiple dimensions
  func.func @repeat_multi_dim(%arg0: tensor<2x3xf32>) -> tensor<4x6xf32> {
    // CHECK-LABEL: func.func @repeat_multi_dim
    // CHECK: tosa.const_shape
    // CHECK: tosa.tile
    // CHECK-NOT: ttir.repeat
    %0 = "ttir.repeat"(%arg0) <{repeat_dimensions = array<i64: 2, 2>}> : (tensor<2x3xf32>) -> tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }

  // Test repeat with 3D tensor
  func.func @repeat_3d(%arg0: tensor<2x3x4xf32>) -> tensor<2x6x8xf32> {
    // CHECK-LABEL: func.func @repeat_3d
    // CHECK: tosa.const_shape
    // CHECK: tosa.tile
    // CHECK-NOT: ttir.repeat
    %0 = "ttir.repeat"(%arg0) <{repeat_dimensions = array<i64: 1, 2, 2>}> : (tensor<2x3x4xf32>) -> tensor<2x6x8xf32>
    return %0 : tensor<2x6x8xf32>
  }

  // Test repeat with bf16 element type
  func.func @repeat_bf16(%arg0: tensor<2x3xbf16>) -> tensor<4x3xbf16> {
    // CHECK-LABEL: func.func @repeat_bf16
    // CHECK: tosa.const_shape
    // CHECK: tosa.tile
    // CHECK-NOT: ttir.repeat
    %0 = "ttir.repeat"(%arg0) <{repeat_dimensions = array<i64: 2, 1>}> : (tensor<2x3xbf16>) -> tensor<4x3xbf16>
    return %0 : tensor<4x3xbf16>
  }
}
