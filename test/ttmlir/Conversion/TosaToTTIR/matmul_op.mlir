// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_matmul(%arg0: tensor<13x21x16xf32>, %arg1: tensor<13x16x31xf32>) -> tensor<13x21x31xf32> {
    %0 = tosa.matmul %arg0, %arg1 : (tensor<13x21x16xf32>, tensor<13x16x31xf32>) -> tensor<13x21x31xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.matmul"[[C:.*]]
    return %0 : tensor<13x21x31xf32>
  }
}
