// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module @jit_dot_general {
  func.func public @test_dot_general(%arg0: tensor<4x10x1xf32>, %arg1: tensor<4x10x2xf32>) -> tensor<1x2xf32> {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batchdims_a = array<i64>, batchdims_b = array<i64>, contractdims_a = array<i64: 0, 1>, contractdims_b = array<i64: 0, 1>}> : (tensor<4x10x1xf32>, tensor<4x10x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
}