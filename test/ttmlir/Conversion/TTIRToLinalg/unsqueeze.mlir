// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module {
  // Test basic unsqueeze at dimension 0
  func.func @unsqueeze_dim0(%arg0: tensor<3x4xf32>) -> tensor<1x3x4xf32> {
    // CHECK-LABEL: func.func @unsqueeze_dim0
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    // CHECK: linalg.copy
    // CHECK-NOT: ttir.unsqueeze
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 0 : si32}> : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    return %0 : tensor<1x3x4xf32>
  }

  // Test unsqueeze at dimension 1
  func.func @unsqueeze_dim1(%arg0: tensor<3x4xf32>) -> tensor<3x1x4xf32> {
    // CHECK-LABEL: func.func @unsqueeze_dim1
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    // CHECK: linalg.copy
    // CHECK-NOT: ttir.unsqueeze
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 1 : si32}> : (tensor<3x4xf32>) -> tensor<3x1x4xf32>
    return %0 : tensor<3x1x4xf32>
  }

  // Test unsqueeze at last dimension
  func.func @unsqueeze_dim_last(%arg0: tensor<3x4xf32>) -> tensor<3x4x1xf32> {
    // CHECK-LABEL: func.func @unsqueeze_dim_last
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    // CHECK: linalg.copy
    // CHECK-NOT: ttir.unsqueeze
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 2 : si32}> : (tensor<3x4xf32>) -> tensor<3x4x1xf32>
    return %0 : tensor<3x4x1xf32>
  }

  // Test unsqueeze with negative dimension
  func.func @unsqueeze_negative_dim(%arg0: tensor<3x4xf32>) -> tensor<3x4x1xf32> {
    // CHECK-LABEL: func.func @unsqueeze_negative_dim
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    // CHECK: linalg.copy
    // CHECK-NOT: ttir.unsqueeze
    %0 = "ttir.unsqueeze"(%arg0) <{dim = -1 : si32}> : (tensor<3x4xf32>) -> tensor<3x4x1xf32>
    return %0 : tensor<3x4x1xf32>
  }

  // Test unsqueeze with bf16 element type
  func.func @unsqueeze_bf16(%arg0: tensor<2x3xbf16>) -> tensor<1x2x3xbf16> {
    // CHECK-LABEL: func.func @unsqueeze_bf16
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    // CHECK: linalg.copy
    // CHECK-NOT: ttir.unsqueeze
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 0 : si32}> : (tensor<2x3xbf16>) -> tensor<1x2x3xbf16>
    return %0 : tensor<1x2x3xbf16>
  }

  // Test unsqueeze on 1D tensor
  func.func @unsqueeze_1d(%arg0: tensor<5xf32>) -> tensor<1x5xf32> {
    // CHECK-LABEL: func.func @unsqueeze_1d
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    // CHECK: linalg.copy
    // CHECK-NOT: ttir.unsqueeze
    %0 = "ttir.unsqueeze"(%arg0) <{dim = 0 : si32}> : (tensor<5xf32>) -> tensor<1x5xf32>
    return %0 : tensor<1x5xf32>
  }
}
