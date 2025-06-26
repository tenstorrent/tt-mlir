// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_bitwise_xor(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
    // CHECK: [[VAL0:%[0-9]+]] = ttir.empty() : [[TENSOR_SIZE:tensor<13x21x3xi32>]]
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.bitwise_xor"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, [[VAL0]]){{.+}}([[TENSOR_SIZE]], [[TENSOR_SIZE]], [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xi32>
    // CHECK: return [[VAL1]] : [[TENSOR_SIZE]]
  }
}
