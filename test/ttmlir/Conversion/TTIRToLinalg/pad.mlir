// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Test basic 2D padding
  func.func @pad_2d(%arg0: tensor<2x3xf32>) -> tensor<4x6xf32> {
    // CHECK: arith.constant 0.000000e+00
    // CHECK: tensor.pad
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 1, 1, 2, 1>, value = 0.000000e+00 : f32}> : (tensor<2x3xf32>) -> tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }

  // Test 4D padding (common for images)
  func.func @pad_4d(%arg0: tensor<1x1x5x5xf32>) -> tensor<2x9x15x20xf32> {
    // CHECK: arith.constant 0.000000e+00
    // CHECK: tensor.pad
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 1, 2, 6, 4, 6, 7, 8>, value = 0.000000e+00 : f32}> : (tensor<1x1x5x5xf32>) -> tensor<2x9x15x20xf32>
    return %0 : tensor<2x9x15x20xf32>
  }

  // Test padding with non-zero value
  func.func @pad_nonzero_value(%arg0: tensor<3x3xf32>) -> tensor<5x5xf32> {
    // CHECK: arith.constant 1.000000e+00
    // CHECK: tensor.pad
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 1, 1, 1, 1>, value = 1.000000e+00 : f32}> : (tensor<3x3xf32>) -> tensor<5x5xf32>
    return %0 : tensor<5x5xf32>
  }

  // Test padding with bfloat16
  func.func @pad_bf16(%arg0: tensor<4x4xbf16>) -> tensor<6x8xbf16> {
    // CHECK: arith.constant 0.000000e+00
    // CHECK: tensor.pad
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 1, 1, 2, 2>, value = 0.000000e+00 : f32}> : (tensor<4x4xbf16>) -> tensor<6x8xbf16>
    return %0 : tensor<6x8xbf16>
  }
}
