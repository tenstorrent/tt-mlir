// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_logical_xor(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<13x21x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
    // CHECK: %[[LOGICAL_XOR_OUT:[0-9]+]] = tensor.empty() : tensor<13x21x3xi1>
    // CHECK: %{{[0-9]+}} = "ttir.logical_xor"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, %[[LOGICAL_XOR_OUT]]){{.+}} -> tensor<13x21x3xi1>
    return %0 : tensor<13x21x3xi1>
  }
}
