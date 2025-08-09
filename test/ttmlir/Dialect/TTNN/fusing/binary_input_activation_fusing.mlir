// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline --ttnn-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Test fusion of LHS input activation with binary operation
  // CHECK-LABEL: func.func @add_with_lhs_relu_fusion
  func.func @add_with_lhs_relu_fusion(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.add"(%1, %arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK-NOT: "ttnn.relu"
    // CHECK: "ttnn.add"(%arg0, %arg1)
    // CHECK-SAME: lhs_activations = [#ttnn.unary_with_param<op_type = relu>]

    return %3 : tensor<64x128xf32>
  }

  // Test fusion of RHS input activation with binary operation
  // CHECK-LABEL: func.func @multiply_with_rhs_sigmoid_fusion
  func.func @multiply_with_rhs_sigmoid_fusion(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.sigmoid"(%arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK-NOT: "ttnn.sigmoid"
    // CHECK: "ttnn.multiply"(%arg0, %arg1)
    // CHECK-SAME: rhs_activations = [#ttnn.unary_with_param<op_type = sigmoid>]

    return %3 : tensor<64x128xf32>
  }

  // Test fusion of both LHS and RHS input activations with binary operation
  // CHECK-LABEL: func.func @subtract_with_both_activations_fusion
  func.func @subtract_with_both_activations_fusion(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.tanh"(%arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %4 = ttir.empty() : tensor<64x128xf32>
    %5 = "ttir.subtract"(%1, %3, %4) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK-NOT: "ttnn.relu"
    // CHECK-NOT: "ttnn.tanh"
    // CHECK: "ttnn.subtract"(%arg0, %arg1)
    // CHECK-SAME: lhs_activations = [#ttnn.unary_with_param<op_type = relu>]
    // CHECK-SAME: rhs_activations = [#ttnn.unary_with_param<op_type = tanh>]

    return %5 : tensor<64x128xf32>
  }

  // Test that fusion doesn't happen when unary operation has multiple uses
  // CHECK-LABEL: func.func @no_fusion_multiple_uses
  func.func @no_fusion_multiple_uses(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.add"(%1, %arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %4 = ttir.empty() : tensor<64x128xf32>
    // Second use of %1 - should prevent fusion
    %5 = "ttir.multiply"(%1, %arg1, %4) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %6 = ttir.empty() : tensor<64x128xf32>
    %7 = "ttir.add"(%3, %5, %6) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK: "ttnn.relu"
    // CHECK: "ttnn.add"
    // CHECK-NOT: lhs_activations
    // CHECK: "ttnn.multiply"
    // CHECK-NOT: lhs_activations

    return %7 : tensor<64x128xf32>
  }

  // Test fusion with unary operation with parameters (leaky_relu)
  // CHECK-LABEL: func.func @add_with_leaky_relu_fusion
  func.func @add_with_leaky_relu_fusion(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.leaky_relu"(%arg0, %0) {parameter = 0.1 : f32} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.add"(%1, %arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK-NOT: "ttnn.leaky_relu"
    // CHECK: "ttnn.add"(%arg0, %arg1)
    // CHECK-SAME: lhs_activations = [#ttnn.unary_with_param<op_type = leaky_relu, params = [1.000000e-01 : f32]>]

    return %3 : tensor<64x128xf32>
  }

  // Test that Unknown unary operation type is not fused
  // CHECK-LABEL: func.func @no_fusion_unknown_unary
  func.func @no_fusion_unknown_unary(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    // cbrt maps to Unknown type
    %1 = "ttir.cbrt"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.add"(%1, %arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK: "ttnn.cbrt"
    // CHECK: "ttnn.add"
    // CHECK-NOT: lhs_activations

    return %3 : tensor<64x128xf32>
  }

  // Test fusion of multiple LHS activations with binary operation
  // CHECK-LABEL: func.func @add_with_multiple_lhs_activations
  func.func @add_with_multiple_lhs_activations(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.sigmoid"(%1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %4 = ttir.empty() : tensor<64x128xf32>
    %5 = "ttir.add"(%3, %arg1, %4) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK-NOT: "ttnn.relu"
    // CHECK-NOT: "ttnn.sigmoid"
    // CHECK: "ttnn.add"(%arg0, %arg1)
    // CHECK-SAME: lhs_activations = [#ttnn.unary_with_param<op_type = relu>, #ttnn.unary_with_param<op_type = sigmoid>]

    return %5 : tensor<64x128xf32>
  }

  // Test fusion of multiple RHS activations with binary operation
  // CHECK-LABEL: func.func @multiply_with_multiple_rhs_activations
  func.func @multiply_with_multiple_rhs_activations(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.tanh"(%arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.leaky_relu"(%1, %2) {parameter = 0.2 : f32} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %4 = ttir.empty() : tensor<64x128xf32>
    %5 = "ttir.multiply"(%arg0, %3, %4) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK-NOT: "ttnn.tanh"
    // CHECK-NOT: "ttnn.leaky_relu"
    // CHECK: "ttnn.multiply"(%arg0, %arg1)
    // CHECK-SAME: rhs_activations = [#ttnn.unary_with_param<op_type = tanh>, #ttnn.unary_with_param<op_type = leaky_relu, params = [2.000000e-01 : f32]>]

    return %5 : tensor<64x128xf32>
  }

  // Test fusion of multiple activations on both LHS and RHS
  // CHECK-LABEL: func.func @subtract_with_multiple_activations_both_sides
  func.func @subtract_with_multiple_activations_both_sides(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // LHS chain: arg0 -> relu -> sigmoid
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.sigmoid"(%1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // RHS chain: arg1 -> tanh -> gelu
    %4 = ttir.empty() : tensor<64x128xf32>
    %5 = "ttir.tanh"(%arg1, %4) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %6 = ttir.empty() : tensor<64x128xf32>
    %7 = "ttir.gelu"(%5, %6) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %8 = ttir.empty() : tensor<64x128xf32>
    %9 = "ttir.subtract"(%3, %7, %8) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK-NOT: "ttnn.relu"
    // CHECK-NOT: "ttnn.sigmoid"
    // CHECK-NOT: "ttnn.tanh"
    // CHECK-NOT: "ttnn.gelu"
    // CHECK: "ttnn.subtract"(%arg0, %arg1)
    // CHECK-SAME: lhs_activations = [#ttnn.unary_with_param<op_type = relu>, #ttnn.unary_with_param<op_type = sigmoid>]
    // CHECK-SAME: rhs_activations = [#ttnn.unary_with_param<op_type = tanh>, #ttnn.unary_with_param<op_type = gelu>]

    return %9 : tensor<64x128xf32>
  }
}
