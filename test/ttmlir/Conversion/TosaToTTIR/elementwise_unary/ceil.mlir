// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = tosa.ceil %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    // CHECK: %[[CEIL_OUT:[0-9]+]] = tensor.empty() : tensor<13x21x3xf32>
    // CHECK: %{{[0-9]+}} = "ttir.ceil"(%arg{{[0-9]+}}, %[[CEIL_OUT]]){{.+}}-> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
}
