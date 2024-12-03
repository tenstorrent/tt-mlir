// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_max(%arg0: tensor<13x21x3xf32>) -> tensor<13x1x3xf32> {
    // CHECK: func.func {{.+}} [[IN_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+xf32>]]{{.*}} ->
    %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<13x21x3xf32>) -> tensor<13x1x3xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = tensor.empty() : [[OUT_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+xf32>]]
    // CHECK: %[[VAL:[0-9]+]] = "ttir.max"(%arg{{[0-9]+}}, %[[OP_OUT]]){{.+}} ([[IN_SIZE]], [[OUT_SIZE]]) -> [[OUT_SIZE]]
    return %0 : tensor<13x1x3xf32>
  }
}
