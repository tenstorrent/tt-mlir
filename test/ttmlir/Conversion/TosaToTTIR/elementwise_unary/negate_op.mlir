// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = tosa.negate %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = tensor.empty() : tensor<13x21x3xf32>
    // CHECK: %{{[0-9]+}} = "ttir.neg"(%arg{{[0-9]+}}, %[[OP_OUT]]){{.+}}-> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
}
