// RUN: ttmlir-opt --convert-tosa-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = tosa.ceil %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.ceil"(%arg{{[0-9]+}}) : ([[TENSOR_SIZE:tensor<13x21x3xf32>]]) -> [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xf32>
    // CHECK: return [[VAL1]] : [[TENSOR_SIZE]]
  }
}
