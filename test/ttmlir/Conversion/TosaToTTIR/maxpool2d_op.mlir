// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_maxpool(%arg0: tensor<32x800x600x6xf32>) -> tensor<32x400x300x6xf32> {
    // CHECK: func.func {{.+}} [[IN_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xf32>]]{{.*}} ->
    %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<32x800x600x6xf32>) -> tensor<32x400x300x6xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = tensor.empty() : [[OUT_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xf32>]]
    // CHECK: %[[VAL:[0-9]+]] = "ttir.max_pool2d"(%arg{{[0-9]+}}, %[[OP_OUT]])
    // CHECK-SAME: channel_last = true
    // CHECK-SAME: ([[IN_SIZE]], [[OUT_SIZE]]) -> [[OUT_SIZE]]
    // CHECK: return %[[VAL]] : [[OUT_SIZE]]
    return %1 : tensor<32x400x300x6xf32>
  }
}
