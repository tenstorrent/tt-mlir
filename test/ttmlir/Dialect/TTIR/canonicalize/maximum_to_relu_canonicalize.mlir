// RUN: ttmlir-opt --canonicalize %s | FileCheck %s

module attributes {} {
  // Test case to verify the canonicalization of max(input, 0.0) into relu(input).
  // CHECK-LABEL: maximum_canonicalize_to_relu_float
  func.func @maximum_canonicalize_to_relu_float(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // Create constant zero tensor
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    %1 = ttir.empty() : tensor<64x128xf32>

    // This maximum operation should be canonicalized into a relu
    // CHECK: "ttir.relu"
    // CHECK-NOT: ttir.maximum
    %2 = "ttir.maximum"(%arg0, %0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %2 : tensor<64x128xf32>
  }

  // Test case to verify the canonicalization of max(0.0, input) into relu(input).
  // CHECK-LABEL: maximum_canonicalize_to_relu_float_reversed
  func.func @maximum_canonicalize_to_relu_float_reversed(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // Create constant zero tensor
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    %1 = ttir.empty() : tensor<64x128xf32>

    // This maximum operation should be canonicalized into a relu (with operands reversed)
    // CHECK: "ttir.relu"
    // CHECK-NOT: ttir.maximum
    %2 = "ttir.maximum"(%0, %arg0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %2 : tensor<64x128xf32>
  }

  // Test case to verify the canonicalization of max(input, 0) into relu(input) for integer types.
  // CHECK-LABEL: maximum_canonicalize_to_relu_int
  func.func @maximum_canonicalize_to_relu_int(%arg0: tensor<64x128xi32>) -> tensor<64x128xi32> {
    // Create constant zero tensor
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    %1 = ttir.empty() : tensor<64x128xi32>

    // This maximum operation should be canonicalized into a relu
    // CHECK: "ttir.relu"
    // CHECK-NOT: ttir.maximum
    %2 = "ttir.maximum"(%arg0, %0, %1) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    return %2 : tensor<64x128xi32>
  }

  // Test case to verify that max(input, non_zero) is NOT canonicalized.
  // CHECK-LABEL: maximum_no_canonicalize_non_zero
  func.func @maximum_no_canonicalize_non_zero(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // Create constant non-zero tensor
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    %1 = ttir.empty() : tensor<64x128xf32>

    // This maximum operation should NOT be canonicalized since constant is not zero
    // CHECK: "ttir.maximum"
    // CHECK-NOT: ttir.relu
    %2 = "ttir.maximum"(%arg0, %0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %2 : tensor<64x128xf32>
  }

  // Test case to verify that max(input1, input2) is NOT canonicalized when both are variables.
  // CHECK-LABEL: maximum_no_canonicalize_two_variables
  func.func @maximum_no_canonicalize_two_variables(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>

    // This maximum operation should NOT be canonicalized since neither operand is constant zero
    // CHECK: "ttir.maximum"
    // CHECK-NOT: ttir.relu
    %1 = "ttir.maximum"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
