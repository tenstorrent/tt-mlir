// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_select(%arg0: tensor<32x128xi1>, %arg1: tensor<32x128xf32>, %arg2: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<32x128xi1>, tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = tensor.empty() : tensor<32x128xf32>
    // CHECK: %{{[0-9]+}} = "ttir.where"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, %arg{{[0-9]+}}, %[[OP_OUT]]){{.+}} -> tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }
}
