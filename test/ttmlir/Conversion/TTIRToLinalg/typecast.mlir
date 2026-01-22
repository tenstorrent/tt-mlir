// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module {
  // Test f32 to bf16 cast
  func.func @typecast_f32_to_bf16(%arg0: tensor<2x3xf32>) -> tensor<2x3xbf16> {
    // CHECK-LABEL: func.func @typecast_f32_to_bf16
    // CHECK: tosa.cast
    // CHECK-NOT: ttir.typecast
    %0 = "ttir.typecast"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }

  // Test bf16 to f32 cast
  func.func @typecast_bf16_to_f32(%arg0: tensor<2x3xbf16>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @typecast_bf16_to_f32
    // CHECK: tosa.cast
    // CHECK-NOT: ttir.typecast
    %0 = "ttir.typecast"(%arg0) : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Test i32 to f32 cast
  func.func @typecast_i32_to_f32(%arg0: tensor<2x3xi32>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @typecast_i32_to_f32
    // CHECK: tosa.cast
    // CHECK-NOT: ttir.typecast
    %0 = "ttir.typecast"(%arg0) : (tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Test f32 to i32 cast
  func.func @typecast_f32_to_i32(%arg0: tensor<2x3xf32>) -> tensor<2x3xi32> {
    // CHECK-LABEL: func.func @typecast_f32_to_i32
    // CHECK: tosa.cast
    // CHECK-NOT: ttir.typecast
    %0 = "ttir.typecast"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  // Test with 3D tensor
  func.func @typecast_3d(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xbf16> {
    // CHECK-LABEL: func.func @typecast_3d
    // CHECK: tosa.cast
    // CHECK-NOT: ttir.typecast
    %0 = "ttir.typecast"(%arg0) : (tensor<2x3x4xf32>) -> tensor<2x3x4xbf16>
    return %0 : tensor<2x3x4xbf16>
  }

  // Test with 1D tensor
  func.func @typecast_1d(%arg0: tensor<10xf32>) -> tensor<10xbf16> {
    // CHECK-LABEL: func.func @typecast_1d
    // CHECK: tosa.cast
    // CHECK-NOT: ttir.typecast
    %0 = "ttir.typecast"(%arg0) : (tensor<10xf32>) -> tensor<10xbf16>
    return %0 : tensor<10xbf16>
  }
}
