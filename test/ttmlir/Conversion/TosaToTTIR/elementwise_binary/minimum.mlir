// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_minimum(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = tensor.empty() : [[TENSOR_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+xf[0-9]+>]]
    // CHECK: %[[VAL:[0-9]+]] = "ttir.minimum"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, %[[OP_OUT]]){{.+}} : ([[TENSOR_SIZE]], [[TENSOR_SIZE]], [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    // CHECK: return %[[VAL]] : [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xf32>
  }
}
