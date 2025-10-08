// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @squeeze_test(%arg0: tensor<1x1x2048x1xbf16>) -> tensor<1x2048x1xbf16> {
    %0 = ttir.empty() : tensor<1x2048x1xbf16>
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    %1 = "ttir.squeeze"(%arg0, %0) <{dim = 0 : si32}> : (tensor<1x1x2048x1xbf16>, tensor<1x2048x1xbf16>) -> tensor<1x2048x1xbf16>
    return %1 : tensor<1x2048x1xbf16>
  }
  func.func @squeeze_test_neg(%arg0: tensor<1x1x2048x1xbf16>) -> tensor<1x1x2048xbf16> {
    %0 = ttir.empty() : tensor<1x1x2048xbf16>
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    %1 = "ttir.squeeze"(%arg0, %0) <{dim = -1 : si32}> : (tensor<1x1x2048x1xbf16>, tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    return %1 : tensor<1x1x2048xbf16>
  }
}
