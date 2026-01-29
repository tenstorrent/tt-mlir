// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

// Softmax is decomposed into elementary TOSA ops:
// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

module {
  // CHECK-LABEL: func.func @softmax_simple
  func.func @softmax_simple(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    // CHECK: tosa.reduce_max
    // CHECK: tosa.sub
    // CHECK: tosa.exp
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reciprocal
    // CHECK: tosa.mul
    %0 = "ttir.softmax"(%arg0) <{dimension = 0 : si32}> : (tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %0 : tensor<512x1024xbf16>
  }

  // CHECK-LABEL: func.func @softmax
  func.func @softmax(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    // Check for positive dimension attribute (dim=1)
    // CHECK: tosa.reduce_max %arg0 {axis = 1 : i32}
    // CHECK: tosa.sub
    // CHECK: tosa.exp
    // CHECK: tosa.reduce_sum {{.*}} {axis = 1 : i32}
    // CHECK: tosa.reciprocal
    // CHECK: tosa.mul
    %0 = "ttir.softmax"(%arg0) <{dimension = 1 : si32}> : (tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    // Check for negative dimension attribute (dim=-1 becomes dim=1)
    // CHECK: tosa.reduce_max {{.*}} {axis = 1 : i32}
    // CHECK: tosa.sub
    // CHECK: tosa.exp
    // CHECK: tosa.reduce_sum {{.*}} {axis = 1 : i32}
    // CHECK: tosa.reciprocal
    // CHECK: tosa.mul
    %1 = "ttir.softmax"(%0) <{dimension = -1 : si32}> : (tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %1 : tensor<512x1024xbf16>
  }
}
