// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-fusing-pass=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Test basic input and output activation fusion
  // CHECK-LABEL: func.func @basic_activation_fusion
  func.func @basic_activation_fusion(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // Input activations: LHS relu, RHS sigmoid
    %0 = "ttir.relu"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    %1 = "ttir.sigmoid"(%arg1) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = "ttir.add"(%0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // Output activation: tanh
    %3 = "ttir.tanh"(%2) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK-NOT: "ttnn.relu"
    // CHECK-NOT: "ttnn.sigmoid"
    // CHECK-NOT: "ttnn.tanh"
    // CHECK: "ttnn.add"(%arg0, %arg1)
    // CHECK-SAME: lhs_activations = [#ttnn.unary_with_param<op_type = relu>]
    // CHECK-SAME: post_activations = [#ttnn.unary_with_param<op_type = tanh>]
    // CHECK-SAME: rhs_activations = [#ttnn.unary_with_param<op_type = sigmoid>]

    return %3 : tensor<64x128xf32>
  }

  // Test parametric operations and multiple activations
  // CHECK-LABEL: func.func @parametric_and_multiple_activations
  func.func @parametric_and_multiple_activations(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // Multiple input activations: LHS relu->sigmoid, RHS leaky_relu
    %0 = "ttir.relu"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    %1 = "ttir.sigmoid"(%0) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = "ttir.leaky_relu"(%arg1) {parameter = 0.1 : f32} : (tensor<64x128xf32>) -> tensor<64x128xf32>

    %3 = "ttir.multiply"(%1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // Multiple output activations: gelu->leaky_relu
    %4 = "ttir.gelu"(%3) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    %5 = "ttir.leaky_relu"(%4) {parameter = 0.2 : f32} : (tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK-NOT: "ttnn.relu"
    // CHECK-NOT: "ttnn.sigmoid"
    // CHECK-NOT: "ttnn.leaky_relu"
    // CHECK-NOT: "ttnn.gelu"
    // CHECK: "ttnn.multiply"(%arg0, %arg1)
    // CHECK-SAME: lhs_activations = [#ttnn.unary_with_param<op_type = sigmoid>, #ttnn.unary_with_param<op_type = relu>]
    // CHECK-SAME: post_activations = [#ttnn.unary_with_param<op_type = gelu>, #ttnn.unary_with_param<op_type = leaky_relu, params = [2.000000e-01 : f32]>]
    // CHECK-SAME: rhs_activations = [#ttnn.unary_with_param<op_type = leaky_relu, params = [1.000000e-01 : f32]>]

    return %5 : tensor<64x128xf32>
  }

  // Test that fusion doesn't happen when operations have multiple uses
  // CHECK-LABEL: func.func @no_fusion_multiple_uses
  func.func @no_fusion_multiple_uses(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.relu"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    %1 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = "ttir.sigmoid"(%1) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    %3 = "ttir.multiply"(%0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %4 = "ttir.subtract"(%2, %3) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK: "ttnn.add"(%arg0, %arg1)
    // CHECK-SAME: post_activations = []
    // CHECK: "ttnn.multiply"
    // CHECK-SAME: lhs_activations = [#ttnn.unary_with_param<op_type = relu>]
    // CHECK: "ttnn.subtract"
    // CHECK-SAME: lhs_activations = [#ttnn.unary_with_param<op_type = sigmoid>]

    return %4 : tensor<64x128xf32>
  }

  // Test that Unknown unary operation types are not fused
  // CHECK-LABEL: func.func @no_fusion_unknown_operations
  func.func @no_fusion_unknown_operations(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // cbrt maps to Unknown type according to TTNNOps.cpp line 73
    %0 = "ttir.cbrt"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    %1 = "ttir.add"(%0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = "ttir.cbrt"(%1) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK: "ttnn.cbrt"
    // CHECK: "ttnn.add"
    // CHECK-SAME: lhs_activations = []
    // CHECK-SAME: post_activations = []
    // CHECK: "ttnn.cbrt"

    return %2 : tensor<64x128xf32>
  }
}
