// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_reshape(%arg0: tensor<63xf32>) -> tensor<1x3x3x7xf32> {
    // CHECK: func.func {{.+}} [[IN_SIZE:tensor<[0-9]+xf32>]]
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 3, 3, 7>} : (tensor<63xf32>) -> tensor<1x3x3x7xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = tensor.empty() : [[OUT_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xf32>]]
    // CHECK: %[[VAL:[0-9]+]] = "ttir.reshape"(%arg{{[0-9]+}}, %[[OP_OUT]]){{.+}} ([[IN_SIZE]], [[OUT_SIZE]]) -> [[OUT_SIZE]]
    // CHECK: return %[[VAL]] : [[OUT_SIZE]]
    return %0 : tensor<1x3x3x7xf32>
  }
}
