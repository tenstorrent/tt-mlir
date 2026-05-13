// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @pow_tensor(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.pow"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: "ttnn.pow_tensor"
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: -> tensor<64x128xf32
    return %0 : tensor<64x128xf32>
  }

  func.func @power_scalar_float(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<2.0> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    // CHECK: "ttnn.pow_scalar"
    // CHECK: <{rhs = 2.000000e+00 : f32}>
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %1 = "ttir.pow"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }

  func.func @power_scalar_integer(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<2> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    // CHECK: "ttnn.pow_scalar"
    // CHECK: <{rhs = 2 : i32}>
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %1 = "ttir.pow"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xi32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }

  // Negative float exponent must fall back to pow_tensor, because
  // ttnn.pow_scalar's verifier rejects negative exponents.
  func.func @power_scalar_negative_float(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<-2.0> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    // CHECK-LABEL: func.func @power_scalar_negative_float
    // CHECK-NOT: "ttnn.pow_scalar"
    // CHECK: "ttnn.pow_tensor"
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: -> tensor<64x128xf32
    %1 = "ttir.pow"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }

  // Negative integer exponent must also fall back to pow_tensor.
  func.func @power_scalar_negative_integer(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<-3> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    // CHECK-LABEL: func.func @power_scalar_negative_integer
    // CHECK-NOT: "ttnn.pow_scalar"
    // CHECK: "ttnn.pow_tensor"
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: -> tensor<64x128xf32
    %1 = "ttir.pow"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xi32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
