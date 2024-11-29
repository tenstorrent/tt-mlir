// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_add(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = tosa.add %arg0, %arg1 {shift = 0 : i8} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    // CHECK: %[[ADD_OUT:[0-9]+]] = tensor.empty() : tensor<13x21x3xf32>
    // CHECK: %{{[0-9]+}} = "ttir.add"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, %[[ADD_OUT]]){{.+}} -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
}
