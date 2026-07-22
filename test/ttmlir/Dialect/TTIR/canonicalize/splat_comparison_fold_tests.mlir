// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // eq of two equal integer splats -> all-true (ones), despite i64 -> i1.
  func.func @eq_equal_int() -> tensor<4x4xi1> {
    // CHECK-LABEL: @eq_equal_int
    // CHECK: "ttir.ones"
    // CHECK-NOT: "ttir.eq"
    %0 = "ttir.constant"() <{value = dense<7> : tensor<4x4xi64>}> : () -> tensor<4x4xi64>
    %1 = "ttir.constant"() <{value = dense<7> : tensor<4x4xi64>}> : () -> tensor<4x4xi64>
    %2 = "ttir.eq"(%0, %1) : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
    return %2 : tensor<4x4xi1>
  }

  // eq of two different integer splats -> all-false (zeros).
  func.func @eq_different_int() -> tensor<4x4xi1> {
    // CHECK-LABEL: @eq_different_int
    // CHECK: "ttir.zeros"
    // CHECK-NOT: "ttir.eq"
    %0 = "ttir.constant"() <{value = dense<7> : tensor<4x4xi64>}> : () -> tensor<4x4xi64>
    %1 = "ttir.constant"() <{value = dense<9> : tensor<4x4xi64>}> : () -> tensor<4x4xi64>
    %2 = "ttir.eq"(%0, %1) : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
    return %2 : tensor<4x4xi1>
  }

  // ne of two different float splats -> all-true (ones), despite f32 -> i1.
  func.func @ne_different_float() -> tensor<3xi1> {
    // CHECK-LABEL: @ne_different_float
    // CHECK: "ttir.ones"
    // CHECK-NOT: "ttir.ne"
    %0 = "ttir.constant"() <{value = dense<1.5> : tensor<3xf32>}> : () -> tensor<3xf32>
    %1 = "ttir.constant"() <{value = dense<2.5> : tensor<3xf32>}> : () -> tensor<3xf32>
    %2 = "ttir.ne"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    return %2 : tensor<3xi1>
  }

  // ne of two equal float splats -> all-false (zeros).
  func.func @ne_equal_float() -> tensor<3xi1> {
    // CHECK-LABEL: @ne_equal_float
    // CHECK: "ttir.zeros"
    // CHECK-NOT: "ttir.ne"
    %0 = "ttir.constant"() <{value = dense<2.5> : tensor<3xf32>}> : () -> tensor<3xf32>
    %1 = "ttir.constant"() <{value = dense<2.5> : tensor<3xf32>}> : () -> tensor<3xf32>
    %2 = "ttir.ne"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    return %2 : tensor<3xi1>
  }

  // Comparison of a value with itself folds even for non-constant operands.
  func.func @eq_self(%arg0: tensor<16x1x1xi64>) -> tensor<16x1x1xi1> {
    // CHECK-LABEL: @eq_self
    // CHECK: "ttir.ones"
    // CHECK-NOT: "ttir.eq"
    %0 = "ttir.eq"(%arg0, %arg0) : (tensor<16x1x1xi64>, tensor<16x1x1xi64>) -> tensor<16x1x1xi1>
    return %0 : tensor<16x1x1xi1>
  }

  func.func @ne_self(%arg0: tensor<16x1x1xi64>) -> tensor<16x1x1xi1> {
    // CHECK-LABEL: @ne_self
    // CHECK: "ttir.zeros"
    // CHECK-NOT: "ttir.ne"
    %0 = "ttir.ne"(%arg0, %arg0) : (tensor<16x1x1xi64>, tensor<16x1x1xi64>) -> tensor<16x1x1xi1>
    return %0 : tensor<16x1x1xi1>
  }

  // Distinct non-constant operands are not folded.
  func.func @ne_non_constant(%arg0: tensor<3xf32>) -> tensor<3xi1> {
    // CHECK-LABEL: @ne_non_constant
    // CHECK: "ttir.ne"
    %0 = "ttir.constant"() <{value = dense<2.5> : tensor<3xf32>}> : () -> tensor<3xf32>
    %1 = "ttir.ne"(%arg0, %0) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    return %1 : tensor<3xi1>
  }

  // Distinct non-splat constant operands are not folded (avoids large
  // comparisons).
  func.func @eq_non_splat() -> tensor<3xi1> {
    // CHECK-LABEL: @eq_non_splat
    // CHECK: "ttir.eq"
    %0 = "ttir.constant"() <{value = dense<[1, 2, 3]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "ttir.constant"() <{value = dense<[4, 5, 6]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "ttir.eq"(%0, %1) : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    return %2 : tensor<3xi1>
  }
}
