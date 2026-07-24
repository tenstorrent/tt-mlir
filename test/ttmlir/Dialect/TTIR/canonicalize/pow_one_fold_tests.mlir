// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @pow_one_f32
  func.func @pow_one_f32(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.constant"() {value = dense<1.000000e+00> : tensor<32x32xf32>} : () -> tensor<32x32xf32>
    // CHECK-NOT: "ttir.pow"
    // CHECK: return %arg0
    %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }

  // CHECK-LABEL: func.func @pow_one_int
  func.func @pow_one_int(%arg0: tensor<4xsi32>) -> tensor<4xsi32> {
    %0 = "ttir.constant"() {value = dense<1> : tensor<4xsi32>} : () -> tensor<4xsi32>
    // CHECK-NOT: "ttir.pow"
    // CHECK: return %arg0
    %1 = "ttir.pow"(%arg0, %0) : (tensor<4xsi32>, tensor<4xsi32>) -> tensor<4xsi32>
    return %1 : tensor<4xsi32>
  }

  // CHECK-LABEL: func.func @no_fold_pow_two
  func.func @no_fold_pow_two(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.constant"() {value = dense<2.000000e+00> : tensor<32x32xf32>} : () -> tensor<32x32xf32>
    // CHECK: "ttir.pow"
    %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }

  // CHECK-LABEL: func.func @no_fold_pow_one_lhs_broadcast
  func.func @no_fold_pow_one_lhs_broadcast(%arg0: tensor<1x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.constant"() {value = dense<1.000000e+00> : tensor<32x32xf32>} : () -> tensor<32x32xf32>
    // CHECK: "ttir.pow"
    %1 = "ttir.pow"(%arg0, %0) : (tensor<1x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
