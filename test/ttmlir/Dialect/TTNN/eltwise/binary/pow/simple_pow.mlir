// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @pow_tensor(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.pow"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: "ttnn.pow_tensor"
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: -> tensor<64x128xf32
    return %1 : tensor<64x128xf32>
  }

  func.func @power_scalar(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<2.0> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    %1 = ttir.empty() : tensor<64x128xf32>
    // CHECK: "ttnn.pow_scalar"
    // CHECK: <{exponent = 2.000000e+00 : f32}>
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %2 = "ttir.pow"(%arg0, %0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %2 : tensor<64x128xf32>
  }
}
