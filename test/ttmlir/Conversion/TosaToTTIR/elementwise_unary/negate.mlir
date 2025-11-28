// RUN: ttmlir-opt --convert-tosa-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.const"() { values = dense<0.0> : tensor<1xf32> } : () -> tensor<1xf32>
    %1 = tosa.negate %arg0, %0, %0 : (tensor<13x21x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<13x21x3xf32>
    // CHECK: %[[CST:[0-9]+]] = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    // CHECK: %[[VAL:[0-9]+]] = "ttir.neg"(%arg{{[0-9]+}}) : ([[TENSOR_SIZE:tensor<13x21x3xf32>]]) -> [[TENSOR_SIZE]]
    // CHECK: return %[[VAL]] : [[TENSOR_SIZE]]
    return %1 : tensor<13x21x3xf32>
  }
}
