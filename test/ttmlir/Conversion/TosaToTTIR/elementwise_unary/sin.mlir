// RUN: ttmlir-opt --convert-tosa-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_sin(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = tosa.sin %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    // CHECK: %[[VAL:[0-9]+]] = "ttir.sin"(%arg{{[0-9]+}}) : ([[TENSOR_SIZE:tensor<13x21x3xf32>]]) -> [[TENSOR_SIZE]]
    // CHECK: return %[[VAL]] : [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xf32>
  }
}
