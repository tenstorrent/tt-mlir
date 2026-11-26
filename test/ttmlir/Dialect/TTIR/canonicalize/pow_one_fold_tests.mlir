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

  // CHECK-LABEL: func.func @pow_one_bf16
  func.func @pow_one_bf16(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %0 = "ttir.constant"() {value = dense<1.000000e+00> : tensor<32x32xbf16>} : () -> tensor<32x32xbf16>
    // CHECK-NOT: "ttir.pow"
    // CHECK: return %arg0
    %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func @pow_one_int
  func.func @pow_one_int(%arg0: tensor<4xsi32>) -> tensor<4xsi32> {
    %0 = "ttir.constant"() {value = dense<1> : tensor<4xsi32>} : () -> tensor<4xsi32>
    // CHECK-NOT: "ttir.pow"
    // CHECK: return %arg0
    %1 = "ttir.pow"(%arg0, %0) : (tensor<4xsi32>, tensor<4xsi32>) -> tensor<4xsi32>
    return %1 : tensor<4xsi32>
  }

  // CHECK-LABEL: func.func @pow_one_uint
  func.func @pow_one_uint(%arg0: tensor<4xui8>) -> tensor<4xui8> {
    %0 = "ttir.constant"() {value = dense<1> : tensor<4xui8>} : () -> tensor<4xui8>
    // CHECK-NOT: "ttir.pow"
    // CHECK: return %arg0
    %1 = "ttir.pow"(%arg0, %0) : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
    return %1 : tensor<4xui8>
  }

  // Exponent element type may differ from the base element type.
  // CHECK-LABEL: func.func @pow_one_int_exponent_float_base
  func.func @pow_one_int_exponent_float_base(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "ttir.constant"() {value = dense<1> : tensor<4xsi32>} : () -> tensor<4xsi32>
    // CHECK-NOT: "ttir.pow"
    // CHECK: return %arg0
    %1 = "ttir.pow"(%arg0, %0) : (tensor<4xf32>, tensor<4xsi32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  // A constant whose elements are all equal canonicalizes to `ttir.full`, so
  // that is the form the fold sees in practice.
  // CHECK-LABEL: func.func @pow_one_full
  func.func @pow_one_full(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.full"() <{shape = array<i32: 32, 32>, fill_value = 1.000000e+00 : f32}> : () -> tensor<32x32xf32>
    // CHECK-NOT: "ttir.pow"
    // CHECK: return %arg0
    %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }

  // CHECK-LABEL: func.func @pow_one_ones
  func.func @pow_one_ones(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.ones"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xf32>
    // CHECK-NOT: "ttir.pow"
    // CHECK: return %arg0
    %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }

  // Exponent is all ones and broadcasts, base shape matches the result.
  // CHECK-LABEL: func.func @pow_one_broadcast_exponent
  func.func @pow_one_broadcast_exponent(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.constant"() {value = dense<1.000000e+00> : tensor<1x32xf32>} : () -> tensor<1x32xf32>
    // CHECK-NOT: "ttir.pow"
    // CHECK: return %arg0
    %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xf32>, tensor<1x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }

  // CHECK-LABEL: func.func @no_fold_pow_two
  func.func @no_fold_pow_two(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.constant"() {value = dense<2.000000e+00> : tensor<32x32xf32>} : () -> tensor<32x32xf32>
    // CHECK: "ttir.pow"
    %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }

  // An exponent that only partly consists of ones must not fold.
  // CHECK-LABEL: func.func @no_fold_pow_mixed_exponent
  func.func @no_fold_pow_mixed_exponent(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "ttir.constant"() {value = dense<[1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    // CHECK: "ttir.pow"
    %1 = "ttir.pow"(%arg0, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @no_fold_pow_one_lhs_broadcast
  func.func @no_fold_pow_one_lhs_broadcast(%arg0: tensor<1x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.constant"() {value = dense<1.000000e+00> : tensor<32x32xf32>} : () -> tensor<32x32xf32>
    // CHECK: "ttir.pow"
    %1 = "ttir.pow"(%arg0, %0) : (tensor<1x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
