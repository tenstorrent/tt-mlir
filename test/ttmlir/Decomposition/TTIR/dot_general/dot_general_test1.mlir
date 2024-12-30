// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module @jit_dot_general {
  func.func public @test_dot_general(%arg0: tensor<127x1xf32>, %arg1: tensor<127x2xf32>) -> tensor<1x2xf32> {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batchdims_a = array<i64>, batchdims_b = array<i64>, contractdims_a = array<i64: 0>, contractdims_b = array<i64: 0>}> : (tensor<127x1xf32>, tensor<127x2xf32>) -> tensor<1x2xf32>
    // CHECK: "ttir.permute"
    // CHECK: <{permutation = array<i64: 1, 0>}> : (tensor<127x1xf32>, tensor<1x127xf32>) -> tensor<1x127xf32>
    // CHECK: "ttir.matmul"
    // CHECK: (tensor<1x127xf32>, tensor<127x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
}
