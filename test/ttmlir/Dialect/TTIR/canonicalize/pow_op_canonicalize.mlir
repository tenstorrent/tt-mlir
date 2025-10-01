// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func public @test_pow_constant(%arg0: tensor<4x32xf32>) -> tensor<4x32xf32> {
    // CHECK-LABEL: @test_pow_constant
    %0 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<4x32xf32>}> : () -> tensor<4x32xf32>
    %1 = ttir.empty() : tensor<4x32xf32>
    // CHECK: %{{[0-9]+}} = "ttir.pow_scalar"(%arg0, %{{[0-9]+}})
    // CHECK-SAME: <{exponent = 2.000000e+00 : f32}>
    // CHECK-SAME: (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    %2 = "ttir.pow_tensor"(%arg0, %0, %1) : (tensor<4x32xf32>, tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    return %2 : tensor<4x32xf32>
  }

  func.func public @test_pow_tensor(%arg0: tensor<4x32xf32>, %arg1: tensor<4x32xf32>) -> tensor<4x32xf32> {
    // CHECK-LABEL: @test_pow_tensor
    %0 = ttir.empty() : tensor<4x32xf32>
    // CHECK: %{{[0-9]+}} = "ttir.pow_tensor"(%arg0, %arg1, %0)
    // CHECK-SAME: (tensor<4x32xf32>, tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    %1 = "ttir.pow_tensor"(%arg0, %arg1, %0) : (tensor<4x32xf32>, tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    return %1 : tensor<4x32xf32>
  }
}
