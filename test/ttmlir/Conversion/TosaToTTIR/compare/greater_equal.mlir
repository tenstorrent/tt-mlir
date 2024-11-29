// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_greater_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
    // CHECK: %[[GE_OUT:[0-9]+]] = tensor.empty() : tensor<13x21x3xi1
    // CHECK: %{{[0-9]+}} = "ttir.ge"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, %[[GE_OUT]]){{.+}} -> tensor<13x21x3xi1>
    return %0 : tensor<13x21x3xi1>
  }
}
