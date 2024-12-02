// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_cos(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = tosa.cos %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[TENSOR_SIZE:tensor<13x21x3xf32>]]
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.cos"(%arg{{[0-9]+}}, [[VAL0]]){{.+}}: ([[TENSOR_SIZE]], [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xf32>
    // CHECK: return [[VAL1]] : [[TENSOR_SIZE]]
  }
}
