// RUN: ttmlir-opt --convert-tosa-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_greater(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
    %0 = tosa.greater %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.gt"(%arg{{[0-9]+}}, %arg{{[0-9]+}}) : ([[TENSOR_SIZE:tensor<13x21x3xf32>]], [[TENSOR_SIZE]]) -> [[RESULT_SIZE:tensor<13x21x3xi1>]]
    return %0 : tensor<13x21x3xi1>
    // CHECK: return [[VAL1]] : [[RESULT_SIZE]]
  }
}
