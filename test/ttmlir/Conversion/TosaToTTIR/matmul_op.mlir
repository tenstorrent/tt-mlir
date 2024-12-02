// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_matmul(%arg0: tensor<13x21x16xf32>, %arg1: tensor<13x16x31xf32>) -> tensor<13x21x31xf32> {
    %0 = tosa.matmul %arg0, %arg1 : (tensor<13x21x16xf32>, tensor<13x16x31xf32>) -> tensor<13x21x31xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = tensor.empty() : tensor<13x21x31xf32>
    // CHECK: %{{[0-9]+}} = "ttir.matmul"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, %[[OP_OUT]]){{.+}} (tensor<13x21x16xf32>, tensor<13x16x31xf32>, tensor<13x21x31xf32>) -> tensor<13x21x31xf32>
    return %0 : tensor<13x21x31xf32>
  }
}
